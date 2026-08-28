from __future__ import annotations

import json
import os
import platform
import queue
import re
import shlex
import shutil
import smtplib
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from .tool_chest import ToolChest


TERMINAL_STATES = {"completed", "failed", "canceled"}
LOCAL_CPU_TARGET = "local_cpu"
OPENSHIFT_JOB_TARGET = "openshift_job"
OPENSHIFT_MOCK_TARGET = "openshift_mock"
SLURM_GPU_TARGET = "slurm_gpu"
LEGACY_SLURM_GPU_TARGET = "osc_gpu"
SLURM_GPU_TARGETS = {SLURM_GPU_TARGET, LEGACY_SLURM_GPU_TARGET}
LOCAL_QUEUE_TARGETS = {LOCAL_CPU_TARGET, OPENSHIFT_JOB_TARGET, OPENSHIFT_MOCK_TARGET}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TRUE_VALUES = {"1", "true", "yes", "on"}
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
        self.job_root = Path(os.environ.get("CONDITAR_JOB_ROOT", project_root / "job_data" / "jobs")).expanduser()
        self.docker_image = os.environ.get("CONDITAR_DOCKER_IMAGE", "osuninglab/conditar-dev:2026-07-10")
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
        self.tool_chest = ToolChest(project_root)
        self._queue: queue.Queue[str] = queue.Queue()
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._recover_incomplete_jobs()
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    def health(self) -> dict:
        image = self._container_image_status()
        archive_path = Path(self.docker_tar).expanduser() if self.docker_tar else None
        archive_exists = bool(archive_path and archive_path.is_file())
        storage = self._job_storage_status()
        tools = self.tool_chest.list_tools()
        available_tools = [tool for tool in tools if tool.get("available")]
        runtime_ok = bool(self.container_runtime)
        image_ok = bool(image.get("exists"))
        slurm_ok = bool(self.sbatch_bin)
        openshift_mode = os.environ.get("KUBERNETES_SERVICE_HOST") or os.environ.get("OPENSHIFT_BUILD_NAME")
        openshift_submit = self._openshift_submit_enabled()
        tool_summary = f"{len(available_tools)}/{len(tools)} optional tools available" if tools else "No optional tools installed"
        checks = [
            {
                "id": "python",
                "label": "Python",
                "status": "ok",
                "detail": f"{platform.python_version()} at {sys.executable}",
                "action": "",
            },
            {
                "id": "container_runtime",
                "label": "Docker or Podman",
                "status": "ok" if runtime_ok else "fail",
                "detail": f"{self.container_runtime_kind}: {self.container_runtime}" if runtime_ok else "No Docker/Podman command found",
                "action": "" if runtime_ok else "Install Docker Desktop for local CPU runs, or load Podman on a Linux/Slurm host.",
            },
            {
                "id": "container_image",
                "label": "conDitar image",
                "status": "ok" if image_ok else "fail",
                "detail": image.get("detail") or f"Image not found: {self.docker_image}",
                "action": "" if image_ok else f"Load or build the image, then check with: docker image inspect {self.docker_image}",
            },
            {
                "id": "slurm",
                "label": "Slurm GPU tools",
                "status": "ok" if slurm_ok else "warn",
                "detail": "sbatch available" if slurm_ok else "sbatch not found; local CPU runs can still work",
                "action": "" if slurm_ok else "Use the local CPU target, or start the GUI from a cluster session with Slurm loaded.",
            },
            {
                "id": "tool_chest",
                "label": "Tool Chest",
                "status": "ok" if len(available_tools) == len(tools) else "warn",
                "detail": tool_summary,
                "action": "" if len(available_tools) == len(tools) else "Run ./setup_tool_chest.sh to enable optional GUI-side tools.",
            },
            {
                "id": "job_storage",
                "label": "Job storage",
                "status": "ok" if storage["writable"] else "fail",
                "detail": storage["detail"],
                "action": "" if storage["writable"] else storage["action"],
            },
            {
                "id": "openshift_mock",
                "label": "OpenShift diagnostics",
                "status": "ok",
                "detail": "Storage, routing, logs, and result loading can be checked without launching conDitar.",
                "action": "Use this only for deployment diagnostics when the generator runtime is not being tested.",
            },
            {
                "id": "openshift_job",
                "label": "OpenShift Job runner",
                "status": "ok",
                "detail": (
                    "Kubernetes Job submission is enabled from the GUI pod."
                    if openshift_submit
                    else "Kubernetes Job manifest generation is available; submission is disabled."
                ),
                "action": (
                    "Use this target to launch conDitar generator pods and poll their Job status."
                    if openshift_submit
                    else "Enable CONDITAR_OPENSHIFT_SUBMIT=true to launch generator pods from the GUI."
                ),
            },
        ]
        return {
            "ok": True,
            "default_target": self._default_target(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "python_executable": sys.executable,
            },
            "environment": {
                "openshift": bool(openshift_mode),
                "openshift_submit": openshift_submit,
                "job_root": str(self.job_root),
                "job_storage_writable": storage["writable"],
            },
            "container_backend": self.container_runtime_kind,
            "container_runtime": self.container_runtime,
            "container_image": image,
            "container_archive": {
                "path": str(archive_path) if archive_path else "",
                "exists": archive_exists,
                "detail": f"Archive available: {archive_path}" if archive_exists else (f"Archive not found: {archive_path}" if archive_path else "No container archive configured"),
            },
            "gpu_available": bool(Path("/dev/nvidia0").exists()),
            "docker_image": self.docker_image,
            "docker_tar": self.docker_tar,
            "slurm": {
                "sbatch": self.sbatch_bin,
                "squeue": self.squeue_bin,
                "sacct": self.sacct_bin,
                "defaults": self.slurm_defaults,
            },
            "tools": {
                "available": len(available_tools),
                "total": len(tools),
                "items": tools,
            },
            "checks": checks,
        }

    def _job_storage_status(self) -> dict:
        try:
            self.job_root.mkdir(parents=True, exist_ok=True)
            probe = self.job_root / ".write-test"
            probe.write_text(utc_now())
            probe.unlink()
            return {
                "writable": True,
                "detail": f"Writable job storage: {self.job_root}",
                "action": "",
            }
        except OSError as error:
            return {
                "writable": False,
                "detail": f"Job storage is not writable: {self.job_root} ({error})",
                "action": "Set CONDITAR_JOB_ROOT to a writable directory or fix the mounted volume permissions.",
            }

    def _default_target(self) -> str:
        requested = os.environ.get("CONDITAR_RUNTIME", "auto").lower()
        if requested in {LOCAL_CPU_TARGET, OPENSHIFT_JOB_TARGET, OPENSHIFT_MOCK_TARGET, SLURM_GPU_TARGET}:
            return requested
        if requested == "docker":
            return LOCAL_CPU_TARGET
        if requested == "podman" and self.sbatch_bin:
            return SLURM_GPU_TARGET
        if os.environ.get("KUBERNETES_SERVICE_HOST") or os.environ.get("OPENSHIFT_BUILD_NAME"):
            return OPENSHIFT_MOCK_TARGET
        if self.sbatch_bin:
            return SLURM_GPU_TARGET
        return LOCAL_CPU_TARGET

    def submit(self, payload: dict, defer_slurm_submit: bool = False) -> dict:
        payload = self._validated_payload(payload)
        target = payload.get("target", LOCAL_CPU_TARGET)
        if target == LEGACY_SLURM_GPU_TARGET:
            target = SLURM_GPU_TARGET
        if target not in {LOCAL_CPU_TARGET, SLURM_GPU_TARGET, OPENSHIFT_JOB_TARGET, OPENSHIFT_MOCK_TARGET}:
            raise ValueError("Only local CPU, Slurm GPU, and OpenShift jobs are supported.")
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
        parameters["device"] = self._target_device(target)
        postprocess = self._postprocess_options(payload.get("postprocess") or {})
        tool_requests = payload.get("tools") or []
        if target == LOCAL_CPU_TARGET:
            image_status = self._container_image_status()
            if image_status.get("checked") and not image_status.get("exists"):
                raise ValueError(
                    f"conDitar container image not found: {self.docker_image}. "
                    f"Load it with `docker load -i /path/to/image.tar.gz`, or build it, then restart the GUI. "
                    f"Details: {image_status.get('detail') or image_status.get('error') or 'image inspect failed'}"
                )
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
            "tools": tool_requests,
            "slurm": slurm_options,
            "container": {
                "backend": self._backend_label(target),
                "runtime": self._runtime_label(target),
                "docker_image": self._job_image_label(target),
                "source_mount": self.source_mount or None,
            },
            "command": command,
            "exit_code": None,
            "error_message": None,
        }
        self._write_job(paths, job)
        if is_slurm_gpu_target(target) and not defer_slurm_submit:
            job = self._submit_slurm_job(job, paths, pdb_path, sdf_path)
        elif target in LOCAL_QUEUE_TARGETS:
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
            "tool_runs": self.tool_chest.read_runs(paths.root),
            "summary": {
                "sdf_count": len(files),
                "artifact_count": len(artifacts),
                "output_directory": str(paths.outputs),
            },
        }

    def list_tools(self) -> list[dict]:
        return self.tool_chest.list_tools()

    def run_tool(self, job_id: str, tool_id: str, options: dict | None = None) -> dict:
        paths = self._paths(job_id)
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Unknown job.")
        if job.get("status") != "completed":
            raise ValueError("Tools can only be run on completed jobs.")
        if not self._output_sdfs(paths):
            raise ValueError("This job has no generated SDF outputs to annotate.")
        run = self.tool_chest.run_tool(tool_id, paths.root, options or {})
        job = self.get_job(job_id) or job
        self._record_tool_run(job, run)
        if run.get("status") == "failed":
            job["status_note"] = f"Tool {run.get('tool_name') or tool_id} failed: {run.get('error')}"
        else:
            job["status_note"] = f"Tool {run.get('tool_name') or tool_id} completed."
        self._write_job(paths, job)
        return {"job": job, "run": run}

    def _record_tool_run(self, job: dict, run: dict) -> None:
        job.setdefault("tool_runs", []).append({
            "id": run.get("id"),
            "tool_id": run.get("tool_id"),
            "tool_name": run.get("tool_name"),
            "status": run.get("status"),
            "finished_at": run.get("finished_at"),
            "result": run.get("result"),
            "error": run.get("error"),
        })

    def _run_requested_tools(self, paths: JobPaths, job: dict) -> None:
        requests = job.get("tools") or []
        if not requests:
            return
        ran_any = False
        for request in requests:
            if request.get("status") in {"completed", "failed"}:
                continue
            tool_id = request.get("id")
            request["status"] = "running"
            self._write_job(paths, job)
            run = self.tool_chest.run_tool(tool_id, paths.root, request.get("options") or {})
            request["status"] = run.get("status")
            request["run_id"] = run.get("id")
            request["finished_at"] = run.get("finished_at")
            request["error"] = run.get("error")
            request["result"] = run.get("result")
            self._record_tool_run(job, run)
            ran_any = True
        if ran_any:
            completed = sum(1 for item in requests if item.get("status") == "completed")
            failed = sum(1 for item in requests if item.get("status") == "failed")
            if failed:
                job["status_note"] = f"Post-run evaluators finished with {failed} failure{'' if failed == 1 else 's'}; {completed} completed."
            else:
                job["status_note"] = f"Post-run evaluators completed: {completed}/{len(requests)}."
            self._write_job(paths, job)

    def export_job(self, job_id: str, payload: dict | None = None) -> dict:
        paths = self._paths(job_id)
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Unknown job.")
        if job.get("status") != "completed":
            raise ValueError("Only completed jobs can be exported.")
        payload = payload or {}
        selected_paths = payload.get("selected_paths") or []
        if selected_paths:
            return self._export_filtered_job(paths, job_id, selected_paths, payload)
        archive = paths.outputs / f"{job_id}_study.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(paths.root.rglob("*")):
                if not path.is_file() or path == archive:
                    continue
                bundle.write(path, path.relative_to(paths.root))
        return {"path": str(archive), "relative_path": str(archive.relative_to(paths.root)), "size": archive.stat().st_size}

    def _export_filtered_job(self, paths: JobPaths, job_id: str, selected_paths: list, payload: dict) -> dict:
        output_sdfs = {str(path.relative_to(paths.root)): path for path in self._output_sdfs(paths)}
        selected = []
        for item in selected_paths[:10000]:
            rel = str(item).strip()
            if rel in output_sdfs:
                selected.append(output_sdfs[rel])
        if not selected:
            raise ValueError("No selected generated SDF files matched this completed job.")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        export_root = paths.root / "filtered_exports" / timestamp
        structures_root = export_root / "generated_structures"
        structures_root.mkdir(parents=True, exist_ok=True)
        for path in selected:
            shutil.copy2(path, structures_root / path.name)
        metadata = {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected_count": len(selected),
            "selected_paths": [str(path.relative_to(paths.root)) for path in selected],
            "filters": payload.get("filters") or [],
            "tool_runs": payload.get("tool_runs") or [],
            "metrics_csv": payload.get("metrics_csv") or "",
            "run_config": payload.get("run_config") or {},
        }
        (export_root / "export_metadata.json").write_text(json.dumps(metadata, indent=2))
        if metadata["metrics_csv"]:
            (export_root / "metrics.csv").write_text(metadata["metrics_csv"])
        archive = export_root.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(export_root.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(export_root.parent))
        return {
            "path": str(archive),
            "relative_path": str(archive.relative_to(paths.root)),
            "directory": str(export_root),
            "relative_directory": str(export_root.relative_to(paths.root)),
            "size": archive.stat().st_size,
            "selected_count": len(selected),
        }

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
            "tools": job.get("tools") or [],
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
        if job.get("target") == OPENSHIFT_JOB_TARGET and (job.get("openshift") or {}).get("submitted"):
            self._delete_openshift_job(str((job.get("openshift") or {}).get("job_name") or ""))
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
        if target == OPENSHIFT_JOB_TARGET:
            return self._build_openshift_job_args(paths, pdb_path, sdf_path, parameters, postprocess)
        if target == OPENSHIFT_MOCK_TARGET:
            return self._build_openshift_mock_command(paths, pdb_path, sdf_path, parameters, postprocess)
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

    def _target_device(self, target: str) -> str:
        if is_slurm_gpu_target(target):
            return "cuda:0"
        if target == OPENSHIFT_JOB_TARGET:
            return os.environ.get("CONDITAR_OPENSHIFT_DEVICE", "cuda:0")
        return "cpu"

    def _backend_label(self, target: str) -> str | None:
        if is_slurm_gpu_target(target):
            return "slurm_podman"
        if target == OPENSHIFT_JOB_TARGET:
            return "openshift_job" if self._openshift_submit_enabled() else "openshift_job_draft"
        if target == OPENSHIFT_MOCK_TARGET:
            return "openshift_mock"
        return self.container_runtime_kind

    def _runtime_label(self, target: str) -> str | None:
        if is_slurm_gpu_target(target):
            return os.environ.get("PODMAN_BIN", "podman")
        if target == OPENSHIFT_JOB_TARGET:
            return "kubernetes Job manifest"
        if target == OPENSHIFT_MOCK_TARGET:
            return "python mock runner"
        return self.container_runtime

    def _job_image_label(self, target: str) -> str | None:
        if target == OPENSHIFT_MOCK_TARGET:
            return None
        if target == OPENSHIFT_JOB_TARGET or is_slurm_gpu_target(target) or self.container_runtime_kind in {"docker", "podman"}:
            return self.docker_image
        return None

    def _build_openshift_job_args(
        self,
        paths: JobPaths,
        pdb_path: Path,
        sdf_path: Path | None,
        parameters: dict,
        postprocess: dict | None = None,
    ) -> list[str]:
        args = [
            "--pdb",
            f"{self._openshift_job_mount_path(paths)}/inputs/{pdb_path.name}",
            "--out",
            f"{self._openshift_job_mount_path(paths)}/outputs",
            "--tmp-dir",
            f"{self._openshift_job_mount_path(paths)}/tmp",
            "--device",
            self._target_device(OPENSHIFT_JOB_TARGET),
        ]
        if sdf_path:
            args.extend(["--sdf", f"{self._openshift_job_mount_path(paths)}/inputs/{sdf_path.name}"])
        for gui_key, cli_key in (
            ("num_samples", "--num-samples"),
            ("batch_size", "--batch-size"),
        ):
            value = parameters.get(gui_key)
            if value not in (None, ""):
                args.extend([cli_key, str(value)])
        if sdf_path:
            value = parameters.get("pocket_radius")
            if value not in (None, ""):
                args.extend(["--pocket-radius", str(value)])
        self._append_postprocess_args(args, postprocess)
        return args

    def _openshift_job_mount_path(self, paths: JobPaths) -> str:
        mount_root = os.environ.get("CONDITAR_OPENSHIFT_JOB_MOUNT", "/data/jobs").rstrip("/")
        return f"{mount_root}/{paths.root.name}"

    def _build_openshift_mock_command(
        self,
        paths: JobPaths,
        pdb_path: Path,
        sdf_path: Path | None,
        parameters: dict,
        postprocess: dict | None = None,
    ) -> list[str]:
        command = [
            "conditar-openshift-mock",
            "--pdb",
            str(pdb_path.relative_to(paths.root)),
            "--out",
            str(paths.outputs.relative_to(paths.root)),
            "--device",
            "cpu",
        ]
        if sdf_path:
            command.extend(["--sdf", str(sdf_path.relative_to(paths.root))])
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
        ):
            value = parameters.get(gui_key)
            if value not in (None, ""):
                command.extend([cli_key, str(value)])
        if sdf_path:
            value = parameters.get("pocket_radius")
            if value not in (None, ""):
                command.extend(["--pocket-radius", str(value)])
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

    def _container_image_status(self) -> dict:
        if not self.container_runtime:
            return {
                "checked": False,
                "exists": False,
                "detail": "Docker/Podman command was not found.",
                "error": None,
            }
        try:
            result = subprocess.run(
                [self.container_runtime, "image", "inspect", self.docker_image],
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return {
                "checked": True,
                "exists": False,
                "detail": f"Could not inspect image with {self.container_runtime}.",
                "error": str(error),
            }
        exists = result.returncode == 0
        detail = f"Image available: {self.docker_image}" if exists else (result.stderr.strip() or result.stdout.strip() or f"Image not found: {self.docker_image}")
        return {
            "checked": True,
            "exists": exists,
            "detail": detail,
            "error": None if exists else detail,
        }

    def _postprocess_options(self, payload_options: dict) -> dict:
        vina_enabled = bool(payload_options.get("vina"))
        vina_mode = str(payload_options.get("vina_mode") or "vina_score").strip()
        if vina_mode not in {"none", "vina_score", "vina_dock", "qvina", "all"}:
            raise ValueError("Vina mode must be none, vina_score, vina_dock, qvina, or all.")
        metrics = payload_options.get("metrics") or []
        if not isinstance(metrics, list):
            raise ValueError("Selected evaluation metrics must be a list.")
        return {
            "vina": vina_enabled,
            "vina_mode": vina_mode,
            "vina_exhaustiveness": str(payload_options.get("vina_exhaustiveness") or "8").strip(),
            "vina_cpu": str(payload_options.get("vina_cpu") or "4").strip(),
            "metrics": [str(item) for item in metrics],
        }

    def _tool_requests(self, payload_tools: list) -> list[dict]:
        if not isinstance(payload_tools, list):
            raise ValueError("Tool selections must be a list.")
        available = {tool["id"]: tool for tool in self.tool_chest.list_tools() if tool.get("available")}
        requests = []
        for item in payload_tools[:20]:
            if not isinstance(item, dict):
                raise ValueError("Each tool selection must be an object.")
            tool_id = str(item.get("id") or "").strip()
            if tool_id not in available:
                raise ValueError(f"Selected tool is not available: {tool_id or 'unknown'}")
            options = item.get("options") or {}
            if not isinstance(options, dict):
                raise ValueError(f"Options for tool {tool_id} must be an object.")
            requests.append({
                "id": tool_id,
                "name": available[tool_id].get("name") or tool_id,
                "options": options,
                "status": "pending",
            })
        return requests

    def _validated_payload(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Job payload must be a JSON object.")
        payload = dict(payload)
        payload["email"] = self._validated_email(payload.get("email"))
        payload["mode"] = self._validated_choice(payload.get("mode") or "pocket", {"reference", "pocket"}, "mode")
        payload["parameters"] = self._validated_parameters(payload.get("parameters") or {})
        payload["tools"] = self._tool_requests(payload.get("tools") or [])
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
        if (postprocess.get("vina_mode") or "vina_score") == "none":
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
        legacy_image = "localhost/conditar-dev:container-dev"
        public_images = {"osuninglab/conditar-dev:2026-07-10", "docker.io/osuninglab/conditar-dev:2026-07-10"}
        allow_legacy_fallback = self.docker_image in public_images
        run_image_setup = "\n".join([
            f"CONDITAR_RUN_IMAGE={shlex.quote(self.docker_image)}",
            f"CONDITAR_LEGACY_IMAGE={shlex.quote(legacy_image)}",
        ])
        image_fallback = "\n".join([
            f"if [[ {str(allow_legacy_fallback).lower()} == true ]] && ! {podman_command} image exists \"$CONDITAR_RUN_IMAGE\" && {podman_command} image exists \"$CONDITAR_LEGACY_IMAGE\"; then",
            "  CONDITAR_RUN_IMAGE=\"$CONDITAR_LEGACY_IMAGE\"",
            "fi",
            f"if ! {podman_command} image exists \"$CONDITAR_RUN_IMAGE\"; then",
            "  echo \"Container image $CONDITAR_RUN_IMAGE is not available on the compute node.\" >&2",
            "  echo \"Set CONDITAR_DOCKER_TAR to a compute-node-visible .tar/.tar.gz archive, or preload/pull the image on the compute node.\" >&2",
            "  exit 125",
            "fi",
        ])
        command_text = command_text.replace(shlex.quote(self.docker_image), '"$CONDITAR_RUN_IMAGE"', 1)
        image_check = ""
        if self.docker_tar:
            image_check = "\n".join([
                f"if ! {podman_command} image exists \"$CONDITAR_RUN_IMAGE\"; then",
                f"  if [[ ! -f {shlex.quote(self.docker_tar)} ]]; then",
                f"    echo \"Container image archive not found: {shlex.quote(self.docker_tar)}\" >&2",
                "    exit 127",
                "  fi",
                f"  {podman_command} load -i {shlex.quote(self.docker_tar)}",
                "fi",
                image_fallback,
            ])
        else:
            image_check = image_fallback

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
            run_image_setup,
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
            if job.get("status") == "completed" and self._output_sdfs(paths):
                self._run_requested_tools(paths, job)
            if (
                not is_slurm_gpu_target(job.get("target"))
                and job.get("status") == "failed"
                and "Server restarted" in (job.get("error_message") or "")
            ):
                self._recover_completed_local_outputs(paths, job)
            return job
        if job.get("target") == OPENSHIFT_JOB_TARGET:
            return self._refresh_openshift_job(job, paths)
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
                job["error_message"] = self._container_failure_message(paths, exit_code)
            self._write_job(paths, job)
            if job["status"] == "completed":
                self._run_requested_tools(paths, job)
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
                reason = self._slurm_pending_reason(job)
                if reason:
                    job["status_note"] = f"Slurm is waiting: {reason}"
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
                else:
                    self._run_requested_tools(paths, job)
                self._send_email(job, paths)
            elif state in SLURM_FAILURE_STATES:
                job["status"] = "failed"
                job["finished_at"] = job.get("finished_at") or utc_now()
                job["exit_code"] = 1
                job["error_message"] = (
                    f"Slurm job ended with state {state}. Check the scheduler reason and logs: "
                    f"{paths.stderr} and {paths.stdout}."
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

    def _refresh_openshift_job(self, job: dict, paths: JobPaths) -> dict:
        output_sdfs = self._output_sdfs(paths)
        openshift = job.get("openshift") or {}
        if output_sdfs:
            self._mark_completed_from_outputs(paths, job, output_sdfs)
            job.setdefault("openshift", {}).setdefault("state", "succeeded")
            self._write_job(paths, job)
            return job
        if not openshift.get("submitted"):
            return job

        state = self._openshift_job_state(str(openshift.get("job_name") or ""))
        if not state:
            job["status_note"] = "OpenShift status is temporarily unavailable; the Job may still be queued."
            self._write_job(paths, job)
            return job

        job.setdefault("openshift", {}).update({
            "state": state.get("state"),
            "active": state.get("active", 0),
            "succeeded": state.get("succeeded", 0),
            "failed": state.get("failed", 0),
            "reason": state.get("reason"),
        })
        if state.get("started_at"):
            job["started_at"] = job.get("started_at") or state["started_at"]

        if state["state"] == "succeeded":
            job["status"] = "failed"
            job["finished_at"] = job.get("finished_at") or utc_now()
            job["exit_code"] = 1
            job["error_message"] = (
                "OpenShift Job completed but no SDF outputs were found. See logs and manifest under "
                f"{paths.root}."
            )
            self._write_job(paths, job)
            self._send_email(job, paths)
            return job
        if state["state"] == "failed":
            job["status"] = "failed"
            job["finished_at"] = job.get("finished_at") or utc_now()
            job["exit_code"] = 1
            reason = state.get("reason") or "OpenShift reported the Job as failed"
            job["error_message"] = f"{reason}. See logs and manifest under {paths.root}."
            self._write_job(paths, job)
            self._send_email(job, paths)
            return job
        if state["state"] == "running":
            job["status"] = "running"
            job["started_at"] = job.get("started_at") or utc_now()
            job["status_note"] = f"OpenShift Job {openshift.get('job_name')} is running."
        else:
            job["status"] = "queued"
            job["status_note"] = f"OpenShift Job {openshift.get('job_name')} is queued."
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
        if is_slurm_gpu_target(job.get("target")):
            job.setdefault("slurm", {})["state"] = "COMPLETED"
        if job.get("target") == OPENSHIFT_JOB_TARGET:
            job.setdefault("openshift", {}).update({
                "state": "succeeded",
                "active": 0,
                "succeeded": 1,
                "failed": 0,
            })
        job["status_note"] = (
            f"Marked completed after finding {len(output_sdfs)} SDF output"
            f"{'' if len(output_sdfs) == 1 else 's'} in the job output directory."
        )
        self._write_job(paths, job)
        if job["status"] == "completed":
            self._run_requested_tools(paths, job)
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
        openshift_logs = self._openshift_log_text(paths, job)
        if openshift_logs:
            sections.append(openshift_logs)
        return "\n\n".join(sections)

    def _openshift_log_text(self, paths: JobPaths, job: dict) -> str:
        openshift = job.get("openshift") or {}
        if job.get("target") != OPENSHIFT_JOB_TARGET or not openshift.get("submitted"):
            return ""
        log_parts = []
        pod_names = self._openshift_job_pod_names(job)
        if not pod_names:
            for log_path in sorted(paths.logs.glob("openshift-*.log")):
                text = log_path.read_text(errors="replace")
                if text.strip():
                    log_parts.append(f"{log_path.name}\n{text}")
            return "\n\n".join(log_parts)
        for pod_name in pod_names:
            log_path = paths.logs / f"openshift-{pod_name}.log"
            fetched = False
            try:
                text = self._openshift_pod_log(pod_name)
                fetched = True
            except urllib.error.HTTPError as error:
                fallback = log_path.read_text(errors="replace") if log_path.exists() else ""
                if error.code == 400:
                    if fallback.startswith("OpenShift pod log unavailable: HTTP Error 400"):
                        fallback = ""
                    text = fallback or "OpenShift pod log is not available yet because the container is still starting.\n"
                else:
                    text = fallback or f"OpenShift pod log unavailable: HTTP {error.code}\n"
            except Exception as error:
                text = log_path.read_text(errors="replace") if log_path.exists() else f"OpenShift pod log unavailable: {error}\n"
            if text:
                if fetched:
                    log_path.write_text(text)
                log_parts.append(f"{log_path.name}\n{text}")
        return "\n\n".join(log_parts)

    def _related_log_files(self, paths: JobPaths, job: dict) -> list[Path]:
        candidates: set[Path] = set()
        for directory in (paths.logs, paths.root):
            if directory.exists():
                for pattern in ("*.log", "*.out", "*.err", "exit_code.txt"):
                    candidates.update(
                        path
                        for path in directory.glob(pattern)
                        if path.is_file() and not path.name.startswith("openshift-")
                    )

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

    def _slurm_pending_reason(self, job: dict) -> str | None:
        job_id = (job.get("slurm") or {}).get("job_id")
        if not job_id or not self.squeue_bin:
            return None
        result = subprocess.run([self.squeue_bin, "-h", "-j", job_id, "-o", "%R"], text=True, capture_output=True, check=False)
        reason = result.stdout.strip()
        if not reason:
            return None
        labels = {
            "MaxGRESPerAccount": "your account has reached its GPU allocation limit; wait for another GPU job to finish or contact the cluster/account administrator",
            "QOSMaxGRESPerUser": "your user/QOS GPU allocation limit has been reached",
            "Resources": "GPU resources are currently unavailable",
            "Priority": "the job is waiting for its scheduler priority",
        }
        return labels.get(reason, reason)

    def _container_failure_message(self, paths: JobPaths, exit_code: int) -> str:
        stderr = paths.stderr.read_text(errors="replace") if paths.stderr.exists() else ""
        if "Trying to pull localhost/" in stderr or "connection refused" in stderr:
            return (
                "Container image was not available on the compute node (exit 125). "
                "Set CONDITAR_DOCKER_TAR to the shared image archive or run podman load "
                "before retrying. See logs: "
                f"{paths.stderr} and {paths.stdout}."
            )
        if "Container image archive not found" in stderr:
            return (
                "The configured container archive was not found on the compute node. "
                "Check CONDITAR_DOCKER_TAR and retry. See logs: "
                f"{paths.stderr} and {paths.stdout}."
            )
        return (
            f"Container command exited with status {exit_code}. Review the container error "
            f"and logs: {paths.stderr} and {paths.stdout}."
        )

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
                if job.get("target") == OPENSHIFT_JOB_TARGET and (job.get("openshift") or {}).get("submitted"):
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
        if job.get("target") == OPENSHIFT_JOB_TARGET:
            self._run_openshift_job_draft(job, paths)
            return
        if job.get("target") == OPENSHIFT_MOCK_TARGET:
            self._run_openshift_mock(job, paths)
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
        if job["status"] == "completed":
            self._run_requested_tools(paths, job)
        self._send_email(job, paths)

    def _run_openshift_job_draft(self, job: dict, paths: JobPaths) -> None:
        job["status"] = "running"
        job["started_at"] = utc_now()
        submit_enabled = self._openshift_submit_enabled()
        job["status_note"] = (
            "Preparing and submitting an OpenShift Job."
            if submit_enabled
            else "Preparing OpenShift Job manifest without submitting it to a cluster."
        )
        self._write_job(paths, job)
        paths.logs.mkdir(parents=True, exist_ok=True)
        paths.outputs.mkdir(parents=True, exist_ok=True)
        (paths.root / "tmp").mkdir(parents=True, exist_ok=True)
        manifest = self._openshift_job_manifest(job, paths)
        manifest_path = paths.outputs / "conditar-openshift-job.yaml"
        manifest_path.write_text(self._yaml_dump(manifest))
        with paths.stdout.open("w") as stdout, paths.stderr.open("w") as stderr:
            stdout.write(f"Wrote manifest: {manifest_path}\n")
            if submit_enabled:
                stdout.write("Submitting manifest through the in-cluster Kubernetes API.\n")
            else:
                stdout.write("OpenShift manifest-only mode: no cluster submission was attempted.\n")
                stdout.write("Apply this manifest only after confirming image, PVC, service account, and GPU policy.\n")
            stderr.write("")
        job = self.get_job(job["id"]) or job
        if job["status"] == "canceled":
            return
        job["outputs"]["manifest"] = str(manifest_path.relative_to(paths.root))
        job["openshift"] = {
            "job_name": manifest["metadata"]["name"],
            "namespace": manifest["metadata"].get("namespace") or self._openshift_namespace(),
            "manifest": str(manifest_path.relative_to(paths.root)),
            "submitted": False,
            "state": "draft",
        }
        if submit_enabled:
            try:
                created = self._submit_openshift_job(manifest)
            except Exception as error:
                job["status"] = "failed"
                job["finished_at"] = utc_now()
                job["exit_code"] = 1
                job["error_message"] = f"OpenShift Job submission failed: {error}. See manifest: {manifest_path}."
                with paths.stderr.open("a") as stderr:
                    stderr.write(job["error_message"] + "\n")
                self._write_job(paths, job)
                self._send_email(job, paths)
                return
            job["status"] = "queued"
            job["finished_at"] = None
            job["exit_code"] = None
            job["error_message"] = None
            job["openshift"]["submitted"] = True
            job["openshift"]["state"] = "submitted"
            job["openshift"]["uid"] = (created.get("metadata") or {}).get("uid")
            job["status_note"] = (
                f"Submitted OpenShift Job {job['openshift']['job_name']}. "
                "The GUI will poll Kubernetes status and read SDF outputs from shared job storage."
            )
            with paths.stdout.open("a") as stdout:
                stdout.write(f"Submitted OpenShift Job: {job['openshift']['job_name']}\n")
            self._write_job(paths, job)
            return
        job["status"] = "completed"
        job["finished_at"] = utc_now()
        job["exit_code"] = 0
        job["outputs"]["sdf_count"] = 0
        job["status_note"] = (
            "OpenShift Job manifest written. No conDitar run was launched; review the generated "
            "manifest artifact before enabling cluster submission."
        )
        job["error_message"] = None
        self._write_job(paths, job)
        self._send_email(job, paths)

    def _openshift_job_manifest(self, job: dict, paths: JobPaths) -> dict:
        job_name = f"conditar-{job['id'][-8:]}"
        namespace = os.environ.get("CONDITAR_OPENSHIFT_NAMESPACE", "")
        pvc_name = os.environ.get("CONDITAR_OPENSHIFT_PVC", "conditar-gui-jobs")
        service_account = os.environ.get("CONDITAR_OPENSHIFT_SERVICE_ACCOUNT", "")
        gpu_resource = os.environ.get("CONDITAR_OPENSHIFT_GPU_RESOURCE", "nvidia.com/gpu")
        gpu_count = os.environ.get("CONDITAR_OPENSHIFT_GPU_COUNT", "1")
        cpu_request = os.environ.get("CONDITAR_OPENSHIFT_CPU_REQUEST", "2")
        memory_request = os.environ.get("CONDITAR_OPENSHIFT_MEMORY_REQUEST", "16Gi")
        memory_limit = os.environ.get("CONDITAR_OPENSHIFT_MEMORY_LIMIT", "32Gi")
        image_pull_policy = os.environ.get("CONDITAR_OPENSHIFT_IMAGE_PULL_POLICY", "IfNotPresent")
        device = self._target_device(OPENSHIFT_JOB_TARGET)
        resources = {
            "requests": {
                "cpu": cpu_request,
                "memory": memory_request,
            },
            "limits": {
                "memory": memory_limit,
            },
        }
        if gpu_count and gpu_count not in {"0", "0.0"} and device != "cpu":
            resources["requests"][gpu_resource] = gpu_count
            resources["limits"][gpu_resource] = gpu_count
        metadata = {
            "name": job_name,
            "labels": {
                "app": "conditar",
                "component": "generator",
                "conditar-gui-job": job["id"],
            },
        }
        if namespace:
            metadata["namespace"] = namespace
        pod_spec = {
            "restartPolicy": "Never",
            "affinity": {
                "podAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {
                                "matchLabels": {
                                    "app": "conditar-gui",
                                },
                            },
                            "topologyKey": "kubernetes.io/hostname",
                        },
                    ],
                },
            },
            "containers": [
                {
                    "name": "conditar",
                    "image": self.docker_image,
                    "imagePullPolicy": image_pull_policy,
                    "args": job.get("command") or [],
                    "env": [
                        {"name": "CONDITAR_DEVICE", "value": device},
                        {"name": "MPLCONFIGDIR", "value": f"{self._openshift_job_mount_path(paths)}/tmp/matplotlib"},
                        {"name": "XDG_CACHE_HOME", "value": f"{self._openshift_job_mount_path(paths)}/tmp/cache"},
                    ],
                    "resources": resources,
                    "volumeMounts": [
                        {
                            "name": "jobs",
                            "mountPath": "/data",
                        },
                    ],
                },
            ],
            "volumes": [
                {
                    "name": "jobs",
                    "persistentVolumeClaim": {
                        "claimName": pvc_name,
                    },
                },
            ],
        }
        if service_account:
            pod_spec["serviceAccountName"] = service_account
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": metadata,
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": metadata["labels"],
                    },
                    "spec": pod_spec,
                },
            },
        }

    def _openshift_submit_enabled(self) -> bool:
        return os.environ.get("CONDITAR_OPENSHIFT_SUBMIT", "").strip().lower() in TRUE_VALUES

    def _openshift_namespace(self) -> str:
        configured = os.environ.get("CONDITAR_OPENSHIFT_NAMESPACE", "").strip()
        if configured:
            return configured
        namespace_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        if namespace_file.exists():
            return namespace_file.read_text().strip()
        return ""

    def _submit_openshift_job(self, manifest: dict) -> dict:
        namespace = manifest["metadata"].get("namespace") or self._openshift_namespace()
        if not namespace:
            raise RuntimeError("OpenShift namespace could not be determined.")
        return self._openshift_api_request(
            "POST",
            f"/apis/batch/v1/namespaces/{namespace}/jobs",
            manifest,
        )

    def _delete_openshift_job(self, job_name: str) -> None:
        if not job_name:
            return
        namespace = self._openshift_namespace()
        if not namespace:
            return
        body = {"propagationPolicy": "Background"}
        try:
            self._openshift_api_request("DELETE", f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}", body)
        except Exception:
            return

    def _openshift_job_state(self, job_name: str) -> dict | None:
        if not job_name:
            return None
        namespace = self._openshift_namespace()
        if not namespace:
            return None
        try:
            data = self._openshift_api_request("GET", f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {
                    "state": "failed",
                    "reason": f"OpenShift Job {job_name} was not found",
                    "active": 0,
                    "succeeded": 0,
                    "failed": 1,
                }
            return None
        except Exception:
            return None

        status = data.get("status") or {}
        conditions = status.get("conditions") or []
        reason = None
        for condition in conditions:
            if condition.get("status") != "True":
                continue
            if condition.get("type") == "Failed":
                reason = condition.get("reason") or condition.get("message") or "OpenShift Job failed"
                return {
                    "state": "failed",
                    "reason": reason,
                    "active": status.get("active", 0),
                    "succeeded": status.get("succeeded", 0),
                    "failed": status.get("failed", 0) or 1,
                    "started_at": status.get("startTime"),
                }
            if condition.get("type") == "Complete":
                return {
                    "state": "succeeded",
                    "reason": condition.get("reason") or condition.get("message"),
                    "active": 0,
                    "succeeded": status.get("succeeded", 1),
                    "failed": status.get("failed", 0),
                    "started_at": status.get("startTime"),
                }
        if status.get("succeeded", 0) > 0:
            return {
                "state": "succeeded",
                "active": status.get("active", 0),
                "succeeded": status.get("succeeded", 0),
                "failed": status.get("failed", 0),
                "started_at": status.get("startTime"),
            }
        if status.get("failed", 0) > 0:
            return {
                "state": "failed",
                "reason": reason or "OpenShift Job pod failed",
                "active": status.get("active", 0),
                "succeeded": status.get("succeeded", 0),
                "failed": status.get("failed", 0),
                "started_at": status.get("startTime"),
            }
        if status.get("active", 0) > 0:
            return {
                "state": "running",
                "active": status.get("active", 0),
                "succeeded": status.get("succeeded", 0),
                "failed": status.get("failed", 0),
                "started_at": status.get("startTime"),
            }
        return {
            "state": "queued",
            "active": 0,
            "succeeded": 0,
            "failed": 0,
            "started_at": status.get("startTime"),
        }

    def _openshift_job_pod_names(self, job: dict) -> list[str]:
        namespace = self._openshift_namespace()
        if not namespace:
            return []
        selector = urllib.parse.quote(f"conditar-gui-job={job['id']}", safe="")
        try:
            data = self._openshift_api_request("GET", f"/api/v1/namespaces/{namespace}/pods?labelSelector={selector}")
        except Exception:
            return []
        names = []
        for item in data.get("items") or []:
            name = (item.get("metadata") or {}).get("name")
            if name:
                names.append(name)
        return sorted(names)

    def _openshift_pod_log(self, pod_name: str) -> str:
        namespace = self._openshift_namespace()
        if not namespace or not pod_name:
            return ""
        query = urllib.parse.urlencode({"container": "conditar", "tailLines": "2000"})
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        if not host:
            return ""
        token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        if not token_path.exists():
            return ""
        request = urllib.request.Request(
            f"https://{host}:{port}/api/v1/namespaces/{namespace}/pods/{pod_name}/log?{query}",
            method="GET",
            headers={"Authorization": f"Bearer {token_path.read_text().strip()}"},
        )
        context = ssl.create_default_context(cafile=str(ca_path)) if ca_path.exists() else ssl.create_default_context()
        with urllib.request.urlopen(request, context=context, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    def _openshift_api_request(self, method: str, path: str, payload: dict | None = None) -> dict:
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        if not host:
            raise RuntimeError("KUBERNETES_SERVICE_HOST is not set; real OpenShift submission only works inside the cluster.")
        token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        if not token_path.exists():
            raise RuntimeError("Service-account token is not mounted in the GUI pod.")
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"https://{host}:{port}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token_path.read_text().strip()}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        context = ssl.create_default_context(cafile=str(ca_path)) if ca_path.exists() else ssl.create_default_context()
        with urllib.request.urlopen(request, context=context, timeout=30) as response:
            text = response.read().decode("utf-8")
        return json.loads(text) if text else {}

    def _yaml_dump(self, value: object, indent: int = 0) -> str:
        prefix = " " * indent
        if isinstance(value, dict):
            lines = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._yaml_dump(item, indent + 2).rstrip())
                else:
                    lines.append(f"{prefix}{key}: {self._yaml_scalar(item)}")
            return "\n".join(lines) + "\n"
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}-")
                    lines.append(self._yaml_dump(item, indent + 2).rstrip())
                elif isinstance(item, list):
                    lines.append(f"{prefix}-")
                    lines.append(self._yaml_dump(item, indent + 2).rstrip())
                else:
                    lines.append(f"{prefix}- {self._yaml_scalar(item)}")
            return "\n".join(lines) + "\n"
        return f"{prefix}{self._yaml_scalar(value)}\n"

    def _yaml_scalar(self, value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value))

    def _run_openshift_mock(self, job: dict, paths: JobPaths) -> None:
        job["status"] = "running"
        job["started_at"] = utc_now()
        job["status_note"] = "OpenShift diagnostics runner is checking job storage, logs, and result loading."
        self._write_job(paths, job)
        paths.logs.mkdir(parents=True, exist_ok=True)
        paths.outputs.mkdir(parents=True, exist_ok=True)
        try:
            requested_samples = int(float(job.get("parameters", {}).get("num_samples") or 3))
        except (TypeError, ValueError):
            requested_samples = 3
        sample_count = max(1, min(requested_samples, 12))
        with paths.stdout.open("w") as stdout, paths.stderr.open("w") as stderr:
            stdout.write("$ " + " ".join(job["command"]) + "\n\n")
            stdout.write("OpenShift diagnostics mode: no conDitar container was launched.\n")
            stdout.write(f"Job storage root: {paths.root}\n")
            stdout.write(f"Generating {sample_count} mock SDF records for UI validation.\n")
            stderr.write("")
        time.sleep(float(os.environ.get("CONDITAR_MOCK_DELAY_SECONDS", "0.2")))
        output_path = paths.outputs / "openshift_mock_results.sdf"
        output_path.write_text(self._mock_sdf(job, sample_count))
        job = self.get_job(job["id"]) or job
        if job["status"] == "canceled":
            return
        job["status"] = "completed"
        job["finished_at"] = utc_now()
        job["exit_code"] = 0
        job["outputs"]["sdf_count"] = 1
        job["status_note"] = (
            "Completed by the OpenShift diagnostics runner. This validates the GUI deployment "
            "path without launching conDitar."
        )
        job["error_message"] = None
        self._write_job(paths, job)
        self._run_requested_tools(paths, job)
        self._send_email(job, paths)

    def _mock_sdf(self, job: dict, sample_count: int) -> str:
        records = []
        for index in range(1, sample_count + 1):
            records.append("\n".join([
                f"OpenShift mock candidate {index}",
                "  conDitar GUI diagnostics",
                "",
                "  3  2  0  0  0  0            999 V2000",
                "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "    1.3000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
                "    0.0000    1.3000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0",
                "  1  2  1  0  0  0  0",
                "  1  3  1  0  0  0  0",
                "M  END",
                "> <conDitar_source>",
                "OpenShift diagnostics mock",
                "",
                "> <SMILES>",
                "CCO",
                "",
                "> <job_id>",
                job["id"],
                "",
                "> <mock_rank>",
                str(index),
                "",
                "$$$$",
                "",
            ]))
        return "".join(records)

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
