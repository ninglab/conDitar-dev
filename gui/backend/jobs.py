from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import smtplib
import subprocess
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path


TERMINAL_STATES = {"completed", "failed", "canceled"}
SLURM_GPU_TARGET = "slurm_gpu"
LEGACY_SLURM_GPU_TARGET = "osc_gpu"
SLURM_GPU_TARGETS = {SLURM_GPU_TARGET, LEGACY_SLURM_GPU_TARGET}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SLURM_PENDING_STATES = {"CONFIGURING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED"}
SLURM_RUNNING_STATES = {"COMPLETING", "RUNNING", "STAGE_OUT"}
SLURM_SUCCESS_STATES = {"COMPLETED"}
SLURM_FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(name: str, fallback: str) -> str:
    cleaned = "".join(char for char in name if char.isalnum() or char in "._-")
    return cleaned or fallback


def is_slurm_gpu_target(target: str | None) -> bool:
    return target in SLURM_GPU_TARGETS


@dataclass
class JobPaths:
    root: Path
    inputs: Path
    outputs: Path
    logs: Path
    metadata: Path
    stdout: Path
    stderr: Path


class LocalJobManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.job_root = project_root / "job_data" / "jobs"
        self.docker_image = os.environ.get("CONDITAR_DOCKER_IMAGE", "localhost/conditar-dev:container-dev")
        self.source_mount = os.environ.get("CONDITAR_SOURCE_MOUNT", "").strip()
        self.container_runtime_kind, self.container_runtime = self._resolve_container_runtime()
        self.default_tmp = Path(os.environ.get("CONDITAR_TMP", "/tmp/conditar-gui"))
        self.sbatch_bin = os.environ.get("SBATCH_BIN") or shutil.which("sbatch")
        self.squeue_bin = os.environ.get("SQUEUE_BIN") or shutil.which("squeue")
        self.sacct_bin = os.environ.get("SACCT_BIN") or shutil.which("sacct")
        self.slurm_defaults = {
            "account": os.environ.get("CONDITAR_SLURM_ACCOUNT", ""),
            "partition": os.environ.get("CONDITAR_SLURM_PARTITION", ""),
            "time": os.environ.get("CONDITAR_SLURM_TIME", "04:00:00"),
            "mem": os.environ.get("CONDITAR_SLURM_MEM", "32G"),
            "cpus": os.environ.get("CONDITAR_SLURM_CPUS", "4"),
            "gpus": os.environ.get("CONDITAR_SLURM_GPUS", "1"),
        }
        self.docker_tar = os.environ.get("CONDITAR_DOCKER_TAR", "")
        self._queue: queue.Queue[str] = queue.Queue()
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._recover_incomplete_jobs()
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    def submit(self, payload: dict, defer_slurm_submit: bool = False) -> dict:
        payload = self._validated_payload(payload)
        target = payload.get("target", "local_cpu")
        if target == LEGACY_SLURM_GPU_TARGET:
            target = SLURM_GPU_TARGET
        if target not in {"local_cpu", SLURM_GPU_TARGET}:
            raise ValueError("Only local CPU and Slurm GPU jobs are supported.")
        pdb = payload.get("pdb") or {}
        if not pdb.get("text"):
            raise ValueError("A PDB input is required.")

        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        paths = self._paths(job_id)
        paths.inputs.mkdir(parents=True)
        paths.outputs.mkdir(parents=True)
        paths.logs.mkdir(parents=True)

        pdb_name = safe_name(pdb.get("name", "input.pdb"), "input.pdb")
        pdb_path = paths.inputs / pdb_name
        pdb_path.write_text(pdb["text"])

        sdf_path = None
        sdf = payload.get("sdf")
        if sdf and sdf.get("text"):
            sdf_name = safe_name(sdf.get("name", "reference.sdf"), "reference.sdf")
            sdf_path = paths.inputs / sdf_name
            sdf_path.write_text(sdf["text"])

        parameters = payload.get("parameters") or {}
        parameters["device"] = "cuda:0" if is_slurm_gpu_target(target) else "cpu"
        postprocess = self._postprocess_options(payload.get("postprocess") or {})
        command = self._build_command(paths, pdb_path, sdf_path, parameters, target, postprocess)
        slurm_options = self._slurm_options(payload.get("slurm") or {}) if is_slurm_gpu_target(target) else None
        if is_slurm_gpu_target(target) and not slurm_options["account"]:
            raise ValueError("Slurm GPU jobs require a Slurm account number. Enter it in Run setup.")
        job = {
            "id": job_id,
            "target": target,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "email": payload.get("email") or None,
            "mode": payload.get("mode") or "pocket",
            "example_id": payload.get("example_id") or None,
            "input_name": payload.get("input_name") or pdb_name,
            "inputs": {
                "pdb": str(pdb_path.relative_to(paths.root)),
                "sdf": str(sdf_path.relative_to(paths.root)) if sdf_path else None,
            },
            "outputs": {
                "directory": str(paths.outputs.relative_to(paths.root)),
            },
            "parameters": parameters,
            "postprocess": postprocess,
            "slurm": slurm_options,
            "container": {
                "backend": "slurm_podman" if is_slurm_gpu_target(target) else self.container_runtime_kind,
                "runtime": os.environ.get("PODMAN_BIN", "podman") if is_slurm_gpu_target(target) else self.container_runtime,
                "docker_image": self.docker_image if is_slurm_gpu_target(target) or self.container_runtime_kind in {"docker", "podman"} else None,
                "source_mount": self.source_mount or None,
            },
            "command": command,
            "exit_code": None,
            "error_message": None,
        }
        self._write_job(paths, job)
        if is_slurm_gpu_target(target) and not defer_slurm_submit:
            job = self._submit_slurm_job(job, paths, pdb_path, sdf_path)
        elif not is_slurm_gpu_target(target):
            self._queue.put(job_id)
        return job

    def submit_batch(self, payload: dict) -> dict:
        jobs_payload = payload.get("jobs")
        if not isinstance(jobs_payload, list) or not jobs_payload:
            raise ValueError("Batch submission requires a non-empty jobs list.")
        if len(jobs_payload) > 100:
            raise ValueError("Batch submission is limited to 100 jobs at a time.")
        targets = {str(item.get("target") or "local_cpu") for item in jobs_payload}
        if len(targets) > 1:
            raise ValueError("All folders in a batch must use the same run target.")

        submitted = []
        errors = []
        for index, job_payload in enumerate(jobs_payload, start=1):
            try:
                submitted.append(self.submit(job_payload, defer_slurm_submit=bool(jobs_payload and is_slurm_gpu_target(job_payload.get("target")))))
            except Exception as error:
                errors.append({"index": index, "input_name": job_payload.get("input_name"), "error": str(error)})
        if submitted and all(is_slurm_gpu_target(job.get("target")) for job in submitted):
            self._submit_slurm_array(submitted)
        if not submitted and errors:
            raise ValueError("; ".join(item["error"] for item in errors[:3]))
        return {"jobs": submitted, "errors": errors}

    def _submit_slurm_array(self, jobs: list[dict]) -> None:
        """Submit a Slurm GPU batch as one array (one task per input folder)."""
        first = jobs[0]
        slurm = first["slurm"]
        scripts = []
        for job in jobs:
            paths = self._paths(job["id"])
            pdb_path = paths.root / job["inputs"]["pdb"]
            sdf_path = paths.root / job["inputs"]["sdf"] if job["inputs"].get("sdf") else None
            script = paths.root / "run.slurm"
            script.write_text(self._slurm_script(job, paths, pdb_path, sdf_path, slurm))
            scripts.append(script)
        master = self.job_root / f"batch-{first['id']}" / "run_array.slurm"
        master.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name=conditar-batch-{first['id'][-8:]}", f"#SBATCH --array=0-{len(jobs)-1}", f"#SBATCH --cpus-per-task={slurm['cpus']}", f"#SBATCH --mem={slurm['mem']}", f"#SBATCH --time={slurm['time']}", f"#SBATCH --gpus={slurm['gpus']}"]
        if slurm["account"]: lines.append(f"#SBATCH --account={slurm['account']}")
        if slurm["partition"]: lines.append(f"#SBATCH --partition={slurm['partition']}")
        lines += ["set -e", "case \"${SLURM_ARRAY_TASK_ID}\" in"]
        lines += [f"  {i}) bash {shlex.quote(str(script))} ;;" for i, script in enumerate(scripts)]
        lines += ["esac", ""]
        master.write_text("\n".join(lines))
        result = subprocess.run([self.sbatch_bin, str(master)], cwd=str(self.project_root), text=True, capture_output=True, check=False)
        if result.returncode != 0:
            for job in jobs:
                paths = self._paths(job["id"]); job["status"] = "failed"; job["exit_code"] = result.returncode; job["finished_at"] = utc_now(); job["error_message"] = f"Slurm batch-array submission failed: {result.stderr.strip()} See {paths.root / 'run.slurm'}."; self._write_job(paths, job)
            return
        array_id = self._parse_sbatch_job_id(result.stdout)
        if not array_id:
            for job in jobs:
                paths = self._paths(job["id"])
                job["status"] = "failed"
                job["exit_code"] = 1
                job["finished_at"] = utc_now()
                job["error_message"] = (
                    "Slurm batch-array submission returned no job ID. See the batch run_array.slurm "
                    f"and logs under {paths.logs}."
                )
                self._write_job(paths, job)
            return
        for i, job in enumerate(jobs):
            job["slurm"]["job_id"] = f"{array_id}_{i}" if array_id else None
            job["slurm"]["array_job_id"] = array_id
            self._write_job(self._paths(job["id"]), job)

    def list_jobs(self) -> list[dict]:
        jobs = [self._refresh_job(self._read_job(path.parent.name)) for path in self.job_root.glob("*/job.json")]
        return sorted((job for job in jobs if job), key=lambda item: item["created_at"], reverse=True)

    def get_job(self, job_id: str) -> dict | None:
        return self._refresh_job(self._read_job(job_id))

    def _read_job(self, job_id: str) -> dict | None:
        metadata = self._paths(job_id).metadata
        if not metadata.exists():
            return None
        return json.loads(metadata.read_text())

    def logs(self, job_id: str) -> dict:
        paths = self._paths(job_id)
        job = self._read_job(job_id) or {}
        extra_logs = self._extra_log_text(paths, job)
        return {
            "stdout": paths.stdout.read_text(errors="replace") if paths.stdout.exists() else "",
            "stderr": paths.stderr.read_text(errors="replace") if paths.stderr.exists() else "",
            "extra": extra_logs,
        }

    def results(self, job_id: str) -> dict:
        paths = self._paths(job_id)
        job = self._refresh_job(self._read_job(job_id)) or {}
        inputs = {}
        for key in ("pdb", "sdf"):
            relative = (job.get("inputs") or {}).get(key)
            if not relative:
                continue
            path = paths.root / relative
            if path.exists():
                inputs[key] = {
                    "name": path.name,
                    "relative_path": str(path.relative_to(paths.root)),
                    "text": path.read_text(errors="replace"),
                }
        files = []
        artifacts = []
        if paths.outputs.exists():
            for path in sorted(paths.outputs.rglob("*.sdf")):
                files.append({
                    "name": path.name,
                    "relative_path": str(path.relative_to(paths.root)),
                    "text": path.read_text(errors="replace"),
                })
            for path in sorted(paths.outputs.rglob("*")):
                if not path.is_file() or path.suffix.lower() == ".sdf":
                    continue
                artifacts.append({
                    "name": path.name,
                    "relative_path": str(path.relative_to(paths.root)),
                    "size": path.stat().st_size,
                })
        return {
            "job_id": job_id,
            "job": job,
            "inputs": inputs,
            "files": files,
            "artifacts": artifacts,
            "logs": self.logs(job_id),
            "summary": {
                "sdf_count": len(files),
                "artifact_count": len(artifacts),
                "output_directory": str(paths.outputs),
            },
        }

    def export_job(self, job_id: str) -> dict:
        paths = self._paths(job_id)
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Unknown job.")
        if job.get("status") != "completed":
            raise ValueError("Only completed jobs can be exported.")
        archive = paths.outputs / f"{job_id}_study.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(paths.root.rglob("*")):
                if not path.is_file() or path == archive:
                    continue
                bundle.write(path, path.relative_to(paths.root))
        return {"path": str(archive), "relative_path": str(archive.relative_to(paths.root)), "size": archive.stat().st_size}

    def archive_job(self, job_id: str) -> dict:
        paths = self._paths(job_id)
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Unknown job.")
        if job.get("status") not in {"failed", "canceled"}:
            raise ValueError("Only failed or canceled jobs can be cleaned up.")
        archive_root = self.project_root / "job_data" / "archived_jobs"
        archive_root.mkdir(parents=True, exist_ok=True)
        destination = archive_root / job_id
        if destination.exists():
            raise ValueError(f"Archived job already exists: {destination}")
        shutil.move(str(paths.root), str(destination))
        job["archived_path"] = str(destination)
        return job

    def rerun_job(self, job_id: str) -> dict:
        paths = self._paths(job_id)
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Unknown job.")
        if job.get("status") not in {"failed", "canceled"}:
            raise ValueError("Only failed or canceled jobs can be rerun.")
        inputs = job.get("inputs") or {}
        pdb_path = paths.root / inputs.get("pdb", "")
        if not pdb_path.exists():
            raise ValueError(f"Original PDB input was not found: {pdb_path}")
        sdf_payload = None
        if inputs.get("sdf"):
            sdf_path = paths.root / inputs["sdf"]
            if not sdf_path.exists():
                raise ValueError(f"Original SDF input was not found: {sdf_path}")
            sdf_payload = {"name": sdf_path.name, "text": sdf_path.read_text(errors="replace")}
        payload = {
            "target": job.get("target") or "local_cpu",
            "mode": job.get("mode") or ("reference" if sdf_payload else "pocket"),
            "example_id": job.get("example_id"),
            "input_name": f"rerun_{job.get('input_name') or pdb_path.stem}",
            "email": job.get("email") or "",
            "pdb": {"name": pdb_path.name, "text": pdb_path.read_text(errors="replace")},
            "sdf": sdf_payload,
            "slurm": job.get("slurm") or {},
            "postprocess": job.get("postprocess") or {},
            "parameters": job.get("parameters") or {},
        }
        return self.submit(payload)

    def cancel(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Unknown job.")
        if job["status"] in TERMINAL_STATES:
            return job
        if is_slurm_gpu_target(job.get("target")):
            slurm_job_id = (job.get("slurm") or {}).get("job_id")
            scancel = shutil.which(os.environ.get("SCANCEL_BIN", "")) if os.environ.get("SCANCEL_BIN") else shutil.which("scancel")
            if slurm_job_id and scancel:
                subprocess.run([scancel, slurm_job_id], check=False)
        process = self._processes.get(job_id)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        job["status"] = "canceled"
        job["finished_at"] = utc_now()
        job["error_message"] = (
            f"Job canceled by user. See logs: {self._paths(job_id).stderr} and "
            f"{self._paths(job_id).stdout}."
        )
        self._write_job(self._paths(job_id), job)
        self._send_email(job, self._paths(job_id))
        return job

    def _build_command(
        self,
        paths: JobPaths,
        pdb_path: Path,
        sdf_path: Path | None,
        parameters: dict,
        target: str = "local_cpu",
        postprocess: dict | None = None,
    ) -> list[str]:
        if is_slurm_gpu_target(target):
            return self._build_docker_command(paths, pdb_path, sdf_path, parameters, device="cuda:0", gpu=True, postprocess=postprocess)
        if not self.container_runtime:
            if self.container_runtime_kind:
                raise ValueError(
                    f"Unsupported container runtime '{self.container_runtime_kind}'. "
                    "This GUI supports Docker locally and Podman for Slurm GPU jobs."
                )
            raise ValueError(
                "Docker/Podman runtime not found. Install Docker for local CPU runs "
                "or Podman for Slurm GPU runs, then set CONDITAR_RUNTIME, "
                "DOCKER_BIN, or PODMAN_BIN."
            )
        if self.container_runtime_kind in {"docker", "podman"}:
            return self._build_docker_command(paths, pdb_path, sdf_path, parameters, device="cpu", gpu=False, postprocess=postprocess)
        raise ValueError(f"Unsupported container runtime: {self.container_runtime_kind}")

    def _build_docker_command(
        self,
        paths: JobPaths,
        pdb_path: Path,
        sdf_path: Path | None,
        parameters: dict,
        device: str = "cpu",
        gpu: bool = False,
        postprocess: dict | None = None,
    ) -> list[str]:
        tmp_dir = paths.root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        runtime = os.environ.get("PODMAN_BIN", "podman") if gpu else self.container_runtime
        command = [
            runtime,
            "run",
            "--rm",
        ]
        if gpu:
            command.extend(["--device", "nvidia.com/gpu=all"])
        command.extend([
            "-e",
            f"CONDITAR_DEVICE={device}",
            "-v",
            f"{paths.inputs.resolve()}:/inputs:ro",
            "-v",
            f"{paths.outputs.resolve()}:/results",
            "-v",
            f"{tmp_dir.resolve()}:/tmp/conditar",
        ])
        if self.source_mount:
            command.extend(["-v", f"{Path(self.source_mount).expanduser().resolve()}:/opt/conditar/app:ro"])
        command.extend([
            self.docker_image,
            "--pdb",
            f"/inputs/{pdb_path.name}",
            "--out",
            "/results",
            "--tmp-dir",
            "/tmp/conditar",
            "--device",
            device,
        ])
        if sdf_path:
            command.extend(["--sdf", f"/inputs/{sdf_path.name}"])
        for gui_key, cli_key in (
            ("num_samples", "--num-samples"),
            ("batch_size", "--batch-size"),
            ("pocket_radius", "--pocket-radius"),
        ):
            value = parameters.get(gui_key)
            if value not in (None, ""):
                command.extend([cli_key, str(value)])
        self._append_postprocess_args(command, postprocess)
        return command

    def _resolve_container_runtime(self) -> tuple[str | None, str | None]:
        requested = os.environ.get("CONDITAR_RUNTIME", "auto").lower()
        if requested in {"docker", "podman"}:
            return requested, self._resolve_executable(f"{requested.upper()}_BIN", requested)
        if requested != "auto":
            return requested, None

        podman = self._resolve_executable("PODMAN_BIN", "podman")
        if podman:
            return "podman", podman
        docker = self._resolve_executable("DOCKER_BIN", "docker")
        if docker:
            return "docker", docker

        return None, None

    def _resolve_executable(self, env_name: str, fallback: str) -> str | None:
        configured = os.environ.get(env_name)
        if configured:
            return configured if shutil.which(configured) else None
        return shutil.which(fallback)

    def _postprocess_options(self, payload_options: dict) -> dict:
        vina_enabled = bool(payload_options.get("vina"))
        vina_mode = str(payload_options.get("vina_mode") or "vina_score").strip()
        if vina_mode not in {"none", "vina_score", "vina_dock", "qvina", "all"}:
            raise ValueError("Vina mode must be none, vina_score, vina_dock, qvina, or all.")
        return {
            "vina": vina_enabled,
            "vina_mode": vina_mode,
            "vina_exhaustiveness": str(payload_options.get("vina_exhaustiveness") or "8").strip(),
            "vina_cpu": str(payload_options.get("vina_cpu") or "4").strip(),
        }

    def _validated_payload(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Job payload must be a JSON object.")
        payload = dict(payload)
        payload["email"] = self._validated_email(payload.get("email"))
        payload["mode"] = self._validated_choice(payload.get("mode") or "pocket", {"reference", "pocket"}, "mode")
        payload["parameters"] = self._validated_parameters(payload.get("parameters") or {})
        if payload.get("slurm"):
            payload["slurm"] = self._slurm_options(payload["slurm"])
        if payload.get("input_name"):
            payload["input_name"] = safe_name(str(payload["input_name"]), "input")

        pdb = payload.get("pdb") or {}
        if not isinstance(pdb, dict) or not str(pdb.get("text") or "").strip():
            raise ValueError("A PDB input is required.")
        pdb_text = str(pdb["text"])
        if len(pdb_text.encode("utf-8")) > 50 * 1024 * 1024:
            raise ValueError("PDB input is larger than 50 MB.")
        if not self._looks_like_pdb(pdb_text):
            raise ValueError("PDB input does not look like a PDB file.")
        payload["pdb"] = {"name": safe_name(str(pdb.get("name") or "input.pdb"), "input.pdb"), "text": pdb_text}

        sdf = payload.get("sdf")
        if sdf and isinstance(sdf, dict) and str(sdf.get("text") or "").strip():
            sdf_text = str(sdf["text"])
            if len(sdf_text.encode("utf-8")) > 50 * 1024 * 1024:
                raise ValueError("SDF input is larger than 50 MB.")
            if "$$$$" not in sdf_text:
                raise ValueError("Reference ligand input does not look like an SDF file.")
            payload["sdf"] = {"name": safe_name(str(sdf.get("name") or "reference.sdf"), "reference.sdf"), "text": sdf_text}
        else:
            payload["sdf"] = None
        if payload["mode"] == "reference" and not payload["sdf"]:
            raise ValueError("Reference mode requires an SDF ligand input.")
        return payload

    def _validated_email(self, value: str | None) -> str | None:
        email = str(value or "").strip()
        if not email:
            return None
        if not EMAIL_PATTERN.match(email):
            raise ValueError("Email address is not valid.")
        return email

    def _validated_choice(self, value: str, allowed: set[str], label: str) -> str:
        text = str(value).strip()
        if text not in allowed:
            raise ValueError(f"Unsupported {label}: {text}")
        return text

    def _validated_parameters(self, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a JSON object.")
        cleaned = dict(parameters)
        for key, minimum, maximum in (
            ("num_samples", 1, 10000),
            ("batch_size", 1, 10000),
            ("pocket_radius", 1, 1000),
        ):
            if key not in cleaned or cleaned[key] in (None, ""):
                continue
            try:
                value = float(cleaned[key])
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be numeric.")
            if value < minimum or value > maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}.")
            cleaned[key] = int(value) if value.is_integer() else value
        return cleaned

    def _looks_like_pdb(self, text: str) -> bool:
        for line in text.splitlines()[:200]:
            if line.startswith(("ATOM  ", "HETATM", "MODEL ", "HEADER", "CRYST1")):
                return True
        return False

    def _append_postprocess_args(self, command: list[str], postprocess: dict | None) -> None:
        if not postprocess or not postprocess.get("vina"):
            return
        command.extend([
            "--vina-score",
            "--vina-mode",
            postprocess.get("vina_mode") or "vina_score",
            "--vina-exhaustiveness",
            str(postprocess.get("vina_exhaustiveness") or "8"),
            "--vina-cpu",
            str(postprocess.get("vina_cpu") or "4"),
        ])

    def _submit_slurm_job(self, job: dict, paths: JobPaths, pdb_path: Path, sdf_path: Path | None) -> dict:
        if not self.sbatch_bin:
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["exit_code"] = 127
            job["error_message"] = (
                "Slurm submission unavailable: sbatch was not found. Start the GUI where "
                "Slurm is available or set SBATCH_BIN. See job.json and logs under "
                f"{paths.root}."
            )
            (paths.logs / "sbatch.stderr.log").write_text(job["error_message"] + "\n")
            self._write_job(paths, job)
            self._send_email(job, paths)
            return job
        slurm = self._slurm_options(job.get("slurm") or {})
        script_path = paths.root / "run.slurm"
        try:
            script_path.write_text(self._slurm_script(job, paths, pdb_path, sdf_path, slurm))
        except Exception as error:
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["exit_code"] = 1
            job["error_message"] = (
                f"Could not prepare the Slurm submission: {error}. See job metadata and logs "
                f"under {paths.root}."
            )
            (paths.logs / "sbatch.stderr.log").write_text(job["error_message"] + "\n")
            self._write_job(paths, job)
            self._send_email(job, paths)
            return job
        job["slurm"] = {
            **slurm,
            "script": str(script_path.relative_to(paths.root)),
            "job_id": None,
            "state": None,
        }
        self._write_job(paths, job)

        try:
            result = subprocess.run(
                [self.sbatch_bin, str(script_path)],
                cwd=str(self.project_root),
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            result = None
            (paths.logs / "sbatch.stderr.log").write_text(f"{type(error).__name__}: {error}\n")
        (paths.logs / "sbatch.stdout.log").write_text(result.stdout if result is not None else "")
        if result is not None:
            (paths.logs / "sbatch.stderr.log").write_text(result.stderr)
        if result is None or result.returncode != 0:
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["exit_code"] = result.returncode if result is not None else 1
            detail = result.stderr.strip() if result is not None else "sbatch could not be executed."
            job["error_message"] = (
                f"Slurm submission failed: {detail} See logs: {paths.logs / 'sbatch.stderr.log'} "
                f"and {paths.logs / 'sbatch.stdout.log'}."
            )
            self._write_job(paths, job)
            self._send_email(job, paths)
            return job

        slurm_job_id = self._parse_sbatch_job_id(result.stdout)
        if not slurm_job_id:
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["exit_code"] = 1
            job["error_message"] = (
                "Slurm submission returned no job ID. See logs: "
                f"{paths.logs / 'sbatch.stdout.log'} and {paths.logs / 'sbatch.stderr.log'}."
            )
            self._write_job(paths, job)
            self._send_email(job, paths)
            return job
        job["slurm"]["job_id"] = slurm_job_id
        job["status"] = "queued"
        self._write_job(paths, job)
        return job

    def _slurm_options(self, payload_options: dict) -> dict:
        merged = {**self.slurm_defaults, **(payload_options or {})}
        return {
            "account": str(merged.get("account") or "").strip(),
            "partition": str(merged.get("partition") or "").strip(),
            "time": str(merged.get("time") or "04:00:00").strip(),
            "mem": str(merged.get("mem") or "32G").strip(),
            "cpus": str(merged.get("cpus") or "4").strip(),
            "gpus": str(merged.get("gpus") or "1").strip(),
        }

    def _slurm_script(
        self,
        job: dict,
        paths: JobPaths,
        pdb_path: Path,
        sdf_path: Path | None,
        slurm: dict,
    ) -> str:
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name=conditar-{job['id'][-8:]}",
            f"#SBATCH --output={paths.stdout}",
            f"#SBATCH --error={paths.stderr}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={slurm['cpus']}",
            f"#SBATCH --mem={slurm['mem']}",
            f"#SBATCH --time={slurm['time']}",
            f"#SBATCH --gpus={slurm['gpus']}",
        ]
        if slurm["account"]:
            lines.append(f"#SBATCH --account={slurm['account']}")
        if slurm["partition"]:
            lines.append(f"#SBATCH --partition={slurm['partition']}")

        command = self._build_docker_command(
            paths,
            pdb_path,
            sdf_path,
            job["parameters"],
            device="cuda:0",
            gpu=True,
            postprocess=job.get("postprocess"),
        )
        command_text = " ".join(shlex.quote(part) for part in command)
        podman_command = shlex.quote(os.environ.get("PODMAN_BIN", "podman"))
        image_check = ""
        if self.docker_tar:
            image_check = "\n".join([
                f"if ! {podman_command} image exists {shlex.quote(self.docker_image)}; then",
                f"  if [[ ! -f {shlex.quote(self.docker_tar)} ]]; then",
                f"    echo \"Container image archive not found: {shlex.quote(self.docker_tar)}\" >&2",
                "    exit 127",
                "  fi",
                f"  {podman_command} load -i {shlex.quote(self.docker_tar)}",
                "fi",
            ])

        fallback_tmp = shlex.quote(str(paths.root / "tmp"))
        runtime_setup = "\n".join([
            f"export TMPDIR=\"${{TMPDIR:-{fallback_tmp}}}\"",
            "mkdir -p \"$TMPDIR\"",
            "export XDG_RUNTIME_DIR=\"$TMPDIR/xdg_runtime_${SLURM_JOB_ID:-conditar}\"",
            "mkdir -p \"$XDG_RUNTIME_DIR\"",
            "chmod 700 \"$XDG_RUNTIME_DIR\"",
            "echo \"XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR\"",
        ])

        return "\n".join([
            *lines,
            "",
            "set +e",
            "echo \"Starting conDitar Slurm job at $(date)\"",
            runtime_setup,
            image_check,
            f"echo \"$ {command_text}\"",
            command_text,
            "rc=$?",
            f"echo \"$rc\" > {shlex.quote(str(paths.logs / 'exit_code.txt'))}",
            "echo \"Finished conDitar Slurm job at $(date) with exit code $rc\"",
            "exit $rc",
            "",
        ])

    def _parse_sbatch_job_id(self, stdout: str) -> str | None:
        parts = stdout.strip().split()
        return parts[-1] if parts else None

    def _refresh_job(self, job: dict | None) -> dict | None:
        if not job:
            return job
        paths = self._paths(job["id"])
        if job.get("status") in TERMINAL_STATES:
            if is_slurm_gpu_target(job.get("target")):
                self._normalize_terminal_slurm_state(paths, job)
            if (
                not is_slurm_gpu_target(job.get("target"))
                and job.get("status") == "failed"
                and "Server restarted" in (job.get("error_message") or "")
            ):
                self._recover_completed_local_outputs(paths, job)
            return job
        if not is_slurm_gpu_target(job.get("target")):
            return job

        exit_code_path = paths.logs / "exit_code.txt"
        output_sdfs = self._output_sdfs(paths)
        if exit_code_path.exists():
            try:
                exit_code = int(exit_code_path.read_text().strip())
            except ValueError:
                exit_code = 1
            job["exit_code"] = exit_code
            job["finished_at"] = job.get("finished_at") or utc_now()
            job["status"] = "completed" if exit_code == 0 else "failed"
            job.setdefault("slurm", {})["state"] = "COMPLETED" if exit_code == 0 else "FAILED"
            if exit_code != 0:
                job["error_message"] = (
                    f"Slurm container command exited with status {exit_code}. See logs: "
                    f"{paths.stderr} and {paths.stdout}."
                )
            self._write_job(paths, job)
            self._send_email(job, paths)
            return job

        state = self._slurm_state(job)
        if output_sdfs and (not state or state in SLURM_SUCCESS_STATES):
            self._mark_completed_from_outputs(paths, job, output_sdfs)
            return job

        if state:
            job["status_note"] = None
            job.setdefault("slurm", {})["state"] = state
            if state in SLURM_PENDING_STATES:
                job["status"] = "queued"
            elif state in SLURM_RUNNING_STATES:
                if output_sdfs:
                    self._mark_completed_from_outputs(paths, job, output_sdfs)
                    return job
                else:
                    job["status"] = "running"
                    job["started_at"] = job.get("started_at") or utc_now()
            elif state in SLURM_SUCCESS_STATES:
                job["status"] = "completed" if output_sdfs else "failed"
                job["finished_at"] = job.get("finished_at") or utc_now()
                job["exit_code"] = 0 if job["status"] == "completed" else 1
                job.setdefault("slurm", {})["state"] = "COMPLETED" if job["status"] == "completed" else state
                if job["status"] == "failed":
                    job["error_message"] = (
                        "Slurm completed but no SDF outputs were found. See logs: "
                        f"{paths.stderr} and {paths.stdout}."
                    )
                self._send_email(job, paths)
            elif state in SLURM_FAILURE_STATES:
                job["status"] = "failed"
                job["finished_at"] = job.get("finished_at") or utc_now()
                job["exit_code"] = 1
                job["error_message"] = (
                    f"Slurm job ended with state {state}. See logs: {paths.stderr} and {paths.stdout}."
                )
                self._send_email(job, paths)
            self._write_job(paths, job)
        elif is_slurm_gpu_target(job.get("target")):
            if output_sdfs:
                self._mark_completed_from_outputs(paths, job, output_sdfs)
            elif self._job_has_logs(paths, job):
                if job.get("status") == "queued":
                    job["status"] = "running"
                    job["started_at"] = job.get("started_at") or utc_now()
                job["status_note"] = (
                    "Slurm status temporarily unavailable; logs indicate the job has started. "
                    f"See logs: {paths.stderr} and {paths.stdout}."
                )
            else:
                job["status_note"] = (
                    "Slurm status temporarily unavailable; the job may still be queued. "
                    f"See logs: {paths.stderr} and {paths.stdout}."
                )
            self._write_job(paths, job)
        return job

    def _output_sdfs(self, paths: JobPaths) -> list[Path]:
        if not paths.outputs.exists():
            return []
        return sorted(paths.outputs.rglob("*.sdf"))

    def _mark_completed_from_outputs(self, paths: JobPaths, job: dict, output_sdfs: list[Path] | None = None) -> None:
        output_sdfs = output_sdfs if output_sdfs is not None else self._output_sdfs(paths)
        job["status"] = "completed"
        job["finished_at"] = job.get("finished_at") or utc_now()
        job["exit_code"] = 0
        job["error_message"] = None
        job["output_count"] = len(output_sdfs)
        job.setdefault("slurm", {})["state"] = "COMPLETED"
        job["status_note"] = (
            f"Marked completed after finding {len(output_sdfs)} SDF output"
            f"{'' if len(output_sdfs) == 1 else 's'} in the job output directory."
        )
        self._write_job(paths, job)
        self._send_email(job, paths)

    def _normalize_terminal_slurm_state(self, paths: JobPaths, job: dict) -> None:
        slurm = job.setdefault("slurm", {})
        expected = {"completed": "COMPLETED", "failed": "FAILED", "canceled": "CANCELLED"}.get(job.get("status"))
        if expected and slurm.get("state") != expected:
            slurm["state"] = expected
            self._write_job(paths, job)

    def _job_has_logs(self, paths: JobPaths, job: dict) -> bool:
        if any(path.exists() and path.stat().st_size > 0 for path in (paths.stdout, paths.stderr)):
            return True
        return bool(self._related_log_files(paths, job))

    def _extra_log_text(self, paths: JobPaths, job: dict) -> str:
        sections = []
        for path in self._related_log_files(paths, job):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            sections.append(f"{path.name}\n{text}")
        return "\n\n".join(sections)

    def _related_log_files(self, paths: JobPaths, job: dict) -> list[Path]:
        candidates: set[Path] = set()
        for directory in (paths.logs, paths.root):
            if directory.exists():
                for pattern in ("*.log", "*.out", "*.err", "exit_code.txt"):
                    candidates.update(path for path in directory.glob(pattern) if path.is_file())

        slurm = job.get("slurm") or {}
        array_id = slurm.get("array_job_id")
        job_id = slurm.get("job_id")
        if array_id:
            task_id = str(job_id).split("_", 1)[1] if "_" in str(job_id) else "*"
            for pattern in (f"slurm-{array_id}_{task_id}.out", f"slurm-{array_id}_{task_id}.err"):
                candidates.update(path for path in self.project_root.glob(pattern) if path.is_file())
        elif job_id:
            for pattern in (f"slurm-{job_id}.out", f"slurm-{job_id}.err"):
                candidates.update(path for path in self.project_root.glob(pattern) if path.is_file())

        ignored = {paths.stdout, paths.stderr}
        return sorted(path for path in candidates if path not in ignored)

    def _slurm_state(self, job: dict) -> str | None:
        slurm_job_id = (job.get("slurm") or {}).get("job_id")
        if not slurm_job_id:
            return None
        for command in self._slurm_state_commands(slurm_job_id):
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[0].split("|")[0].strip().split()[0]
        return None

    def _slurm_state_commands(self, slurm_job_id: str) -> list[list[str]]:
        commands = []
        if self.squeue_bin:
            commands.append([self.squeue_bin, "-h", "-j", slurm_job_id, "-o", "%T"])
        if self.sacct_bin:
            commands.append([self.sacct_bin, "-n", "-X", "-j", slurm_job_id, "-o", "State", "-P"])
        return commands

    def _paths(self, job_id: str) -> JobPaths:
        root = self.job_root / job_id
        return JobPaths(
            root=root,
            inputs=root / "inputs",
            outputs=root / "outputs",
            logs=root / "logs",
            metadata=root / "job.json",
            stdout=root / "logs" / "stdout.log",
            stderr=root / "logs" / "stderr.log",
        )

    def _write_job(self, paths: JobPaths, job: dict) -> None:
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.metadata.write_text(json.dumps(job, indent=2))

    def _output_count(self, paths: JobPaths) -> int:
        return len(list(paths.outputs.rglob("*.sdf"))) if paths.outputs.exists() else 0

    def _recover_completed_local_outputs(self, paths: JobPaths, job: dict) -> bool:
        output_count = self._output_count(paths)
        if output_count == 0:
            return False
        job.setdefault("outputs", {})["sdf_count"] = output_count
        job["status"] = "completed"
        job["exit_code"] = job.get("exit_code") if job.get("exit_code") is not None else 0
        job["finished_at"] = job.get("finished_at") or utc_now()
        job["error_message"] = None
        job["status_note"] = (
            "Recovered after server restart: SDF outputs were found, so this local CPU job "
            "is available for review."
        )
        self._write_job(paths, job)
        return True

    def _recover_incomplete_jobs(self) -> None:
        for job in self.list_jobs():
            if job["status"] not in TERMINAL_STATES:
                if is_slurm_gpu_target(job.get("target")):
                    continue
                paths = self._paths(job["id"])
                if self._recover_completed_local_outputs(paths, job):
                    continue
                if job.get("status") == "queued" and not job.get("started_at"):
                    job["error_message"] = None
                    job["status_note"] = "Recovered queued local CPU job after server restart."
                    self._write_job(paths, job)
                    self._queue.put(job["id"])
                    continue
                job["status"] = "failed"
                job["finished_at"] = utc_now()
                job["error_message"] = (
                    "Server restarted while this local CPU job was running. See logs: "
                    f"{paths.stderr} and {paths.stdout}."
                )
                self._write_job(paths, job)

    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        paths = self._paths(job_id)
        job = self.get_job(job_id)
        if not job or job["status"] == "canceled":
            return
        job["status"] = "running"
        job["started_at"] = utc_now()
        self._write_job(paths, job)

        env = os.environ.copy()
        env["CONDITAR_DEVICE"] = "cpu"
        paths.logs.mkdir(parents=True, exist_ok=True)
        try:
            with paths.stdout.open("w") as stdout, paths.stderr.open("w") as stderr:
                stdout.write("$ " + " ".join(job["command"]) + "\n\n")
                stdout.flush()
                process = subprocess.Popen(
                    job["command"],
                    stdout=stdout,
                    stderr=stderr,
                    cwd=str(self.project_root),
                    env=env,
                    start_new_session=True,
                )
                with self._lock:
                    self._processes[job_id] = process
                exit_code = process.wait()
                with self._lock:
                    self._processes.pop(job_id, None)
        except OSError as error:
            job = self.get_job(job_id) or job
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["exit_code"] = 1
            job["error_message"] = (
                f"Could not start the Docker/Podman job: {error}. See logs: "
                f"{paths.stderr} and {paths.stdout}."
            )
            self._write_job(paths, job)
            self._send_email(job, paths)
            return

        job = self.get_job(job_id) or job
        if job["status"] == "canceled":
            return
        job["exit_code"] = exit_code
        job["finished_at"] = utc_now()
        output_count = len(list(paths.outputs.rglob("*.sdf"))) if paths.outputs.exists() else 0
        job["outputs"]["sdf_count"] = output_count
        job["status"] = "completed" if exit_code == 0 and output_count > 0 else "failed"
        if exit_code != 0:
            job["error_message"] = (
                f"Docker/Podman command exited with status {exit_code}. See logs: "
                f"{paths.stderr} and {paths.stdout}."
            )
        elif output_count == 0:
            job["exit_code"] = 1
            job["error_message"] = (
                "Docker/Podman command completed but no SDF outputs were found. See logs: "
                f"{paths.stderr} and {paths.stdout}."
            )
        self._write_job(paths, job)
        self._send_email(job, paths)

    def _send_email(self, job: dict, paths: JobPaths) -> None:
        if not job.get("email"):
            return
        if job.get("notification_sent_at"):
            return
        subject = f"conDitar job {job['status']}: {job['id']}"
        body = "\n".join([
            f"Job: {job['id']}",
            f"Status: {job['status']}",
            f"Started: {job.get('started_at')}",
            f"Finished: {job.get('finished_at')}",
            f"Output directory: {paths.outputs}",
            f"Error: {job.get('error_message') or ''}",
        ])
        smtp_host = os.environ.get("CONDITAR_SMTP_HOST")
        if smtp_host:
            self._send_smtp_email(job, paths, subject, body)
            job["notification_sent_at"] = utc_now()
            self._write_job(paths, job)
            return
        sendmail = shutil.which("sendmail")
        if sendmail:
            message = f"Subject: {subject}\nTo: {job['email']}\n\n{body}\n"
            result = subprocess.run([sendmail, "-t"], input=message, text=True, capture_output=True, check=False)
            if result.returncode == 0:
                job["notification_sent_at"] = utc_now()
                self._write_job(paths, job)
                return
            (paths.logs / "email_notice.txt").write_text(
                f"To: {job['email']}\nSubject: {subject}\n\n{body}\n\nsendmail delivery failed:\n{result.stderr}\n"
            )
            job["notification_sent_at"] = utc_now()
            self._write_job(paths, job)
            return
        (paths.logs / "email_notice.txt").write_text(f"To: {job['email']}\nSubject: {subject}\n\n{body}\n")
        job["notification_sent_at"] = utc_now()
        self._write_job(paths, job)

    def _send_smtp_email(self, job: dict, paths: JobPaths, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["To"] = job["email"]
        msg["From"] = os.environ.get("CONDITAR_SMTP_FROM", os.environ.get("CONDITAR_SMTP_USER", "conditar-gui@localhost"))
        msg.set_content(body)

        host = os.environ["CONDITAR_SMTP_HOST"]
        port = int(os.environ.get("CONDITAR_SMTP_PORT", "587"))
        user = os.environ.get("CONDITAR_SMTP_USER")
        password = os.environ.get("CONDITAR_SMTP_PASSWORD")
        use_tls = os.environ.get("CONDITAR_SMTP_TLS", "true").lower() not in {"0", "false", "no"}
        try:
            with smtplib.SMTP(host, port, timeout=30) as server:
                if use_tls:
                    server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(msg)
        except Exception as error:
            (paths.logs / "email_notice.txt").write_text(
                f"To: {job['email']}\nSubject: {subject}\n\n{body}\n\nSMTP delivery failed: {error}\n"
            )
