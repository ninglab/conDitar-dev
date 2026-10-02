#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.backend.input_processing import parse_residue_spec, pocket_pdb_from_residues
from gui.backend.jobs import CONDITAR_ENGINE, DIFFSMOL_ENGINE, JobPaths, LocalJobManager
from gui.backend.chemistry import sdf_chemistry
from gui.backend.workflow_rules import resolve_workflow, validate_generation_inputs
from gui.tools.medchem_filters import run as run_medchem_filters


PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.000  1.00 20.00           O
END
"""

PDB_WITH_WATER = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.000  1.00 20.00           O
HETATM    5  O   HOH A 201       5.000   5.000   5.000  1.00 20.00           O
HETATM    6  NA   NA A 202       6.000   5.000   5.000  1.00 20.00          NA
END
"""


def sdf_block(name: str, x: float = 0.0) -> str:
    return f"""\
{name}
  conDitar GUI smoke

  1  0  0  0  0  0            999 V2000
{x:10.4f}{0.0:10.4f}{0.0:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
>  <SMILES>
C

$$$$
"""


def expect_value_error(label: str, func, contains: str) -> None:
    try:
        func()
    except ValueError as error:
        message = str(error)
        assert contains in message, f"{label}: expected {contains!r} in {message!r}"
    else:
        raise AssertionError(f"{label}: expected ValueError")


def manager(project_root: Path) -> LocalJobManager:
    os.environ["CONDITAR_JOB_ROOT"] = str(project_root / "job_data" / "jobs")
    return LocalJobManager(project_root)


def test_docker_image_inspection_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mgr = manager(Path(tmp))
        mgr.container_runtime_kind = "docker"
        mgr.container_runtime = "docker"
        inspected = []

        def inspect(command, **_kwargs):
            inspected.append(command[-1])
            return subprocess.CompletedProcess(command, 0 if command[-1].startswith("docker.io/library/") else 1, "", "No such image")

        with patch("gui.backend.jobs.subprocess.run", side_effect=inspect):
            status = mgr._container_image_status("diffsmol:cpu-20261001")
        assert status["exists"] is True
        assert inspected == ["diffsmol:cpu-20261001", "docker.io/library/diffsmol:cpu-20261001"]


def test_diffsmol_input_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mgr = manager(Path(tmp))
        expect_value_error(
            "DiffSMol PDB-only",
            lambda: mgr._validated_payload({
                "engine": DIFFSMOL_ENGINE,
                "target": "local_cpu",
                "mode": "reference",
                "pdb": {"name": "protein.pdb", "text": PDB_TEXT},
                "parameters": {},
            }),
            "DiffSMol requires a 3D reference ligand SDF input.",
        )
        expect_value_error(
            "DiffSMol multi-molecule SDF",
            lambda: mgr._validated_payload({
                "engine": DIFFSMOL_ENGINE,
                "target": "local_cpu",
                "mode": "reference",
                "sdf": {"name": "panel.sdf", "text": sdf_block("mol1") + sdf_block("mol2", 1.0)},
                "parameters": {},
            }),
            "DiffSMol requires one 3D reference ligand SDF.",
        )
        expect_value_error(
            "DiffSMol malformed SDF",
            lambda: mgr._validated_payload({
                "engine": DIFFSMOL_ENGINE,
                "target": "local_cpu",
                "mode": "reference",
                "sdf": {"name": "not_a_ligand.sdf", "text": "This is not an SDF molecule."},
                "parameters": {},
            }),
            "Reference ligand input does not look like an SDF file.",
        )
        payload = mgr._validated_payload({
            "engine": DIFFSMOL_ENGINE,
            "target": "local_cpu",
            "mode": "reference",
            "sdf": {"name": "reference.sdf", "text": sdf_block("reference")},
            "parameters": {},
        })
        assert payload["engine"] == DIFFSMOL_ENGINE
        assert payload["sdf"]["name"] == "reference.sdf"
        assert payload["pdb"] is None


def test_diffsmol_cross_target_commands() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        mgr.container_runtime = "docker"
        paths = mgr._paths("targets")
        paths.inputs.mkdir(parents=True)
        paths.outputs.mkdir()
        sdf_path = paths.inputs / "reference.sdf"
        pdb_path = paths.inputs / "pocket.pdb"
        sdf_path.write_text(sdf_block("reference"))
        pdb_path.write_text(PDB_TEXT)
        options = mgr._postprocess_options({"metrics": ["vina_score", "qed"], "vina_cpu": 2})
        for target in ("local_cpu", "slurm_gpu", "openshift_job"):
            command = mgr._build_command(
                DIFFSMOL_ENGINE, paths, pdb_path, sdf_path,
                {"num_samples": 1, "diffsmol_guidance": True}, target, options,
            )
            assert "/opt/DiffSMol/docker/run.py" in command
            assert command[command.index("--postprocess-metrics") + 1] == "vina_score,qed"
            assert "--protein" in command and "--guidance" in command
            expected_device = "cpu" if target == "local_cpu" else "cuda:0"
            assert command[max(index for index, arg in enumerate(command) if arg == "--device") + 1] == expected_device
            if target == "slurm_gpu":
                assert "--device" in command and "nvidia.com/gpu=all" in command
                with patch.dict(os.environ, {"DIFFSMOL_DOCKER_TAR": "/shared/diffsmol.tar"}):
                    slurm_mgr = manager(root)
                    script = slurm_mgr._slurm_script(
                        {"id": "targets", "engine": DIFFSMOL_ENGINE,
                         "parameters": {"num_samples": 1}, "postprocess": options},
                        paths, pdb_path, sdf_path,
                        {"cpus": "2", "mem": "8G", "time": "00:30:00", "gpus": "1",
                         "account": "", "partition": ""},
                    )
                assert "/opt/DiffSMol/docker/run.py" in script
                assert "--postprocess-metrics" in script
                assert "/shared/diffsmol.tar" in script
                assert "DIFFSMOL_DOCKER_TAR" in script
            if target == "openshift_job":
                manifest = mgr._openshift_job_manifest(
                    {"id": "targets", "engine": DIFFSMOL_ENGINE, "command": command}, paths,
                )
                container = manifest["spec"]["template"]["spec"]["containers"][0]
                assert container["image"] == mgr.diffsmol_image
                assert container["name"] == "diffsmol"
                assert container["args"] == command
                assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
                assert {item["name"]: item["value"] for item in container["env"]}["HOME"] == "/tmp"
        with patch.dict(os.environ, {"CONDITAR_OPENSHIFT_DEVICE": "cpu", "CONDITAR_OPENSHIFT_GPU_COUNT": "0"}):
            command = mgr._build_command(
                DIFFSMOL_ENGINE, paths, pdb_path, sdf_path, {"num_samples": 1},
                "openshift_job", options,
            )
            manifest = mgr._openshift_job_manifest(
                {"id": "targets", "engine": DIFFSMOL_ENGINE, "command": command}, paths,
            )
            assert command[command.index("--device") + 1] == "cpu"
            assert "nvidia.com/gpu" not in manifest["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        with patch.dict(os.environ, {"DIFFSMOL_DOCKER_IMAGE": "ninglab/diffsmol:latest"}):
            old_image_mgr = manager(root)
            expect_value_error(
                "OpenShift rejects old DiffSMol image",
                lambda: old_image_mgr.submit({
                    "engine": DIFFSMOL_ENGINE, "target": "openshift_job", "mode": "reference",
                    "sdf": {"name": "reference.sdf", "text": sdf_block("reference")},
                    "parameters": {"num_samples": 1},
                }),
                "registry-accessible copy of the updated standalone image",
            )
        paths.logs.mkdir(parents=True, exist_ok=True)
        paths.stderr.write_text("Container image archive not found")
        assert "DIFFSMOL_DOCKER_TAR" in mgr._container_failure_message(paths, 127, DIFFSMOL_ENGINE)
        with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "kubernetes.test"}), \
                patch.object(mgr, "_openshift_namespace", return_value="test"), \
                patch.object(Path, "exists", lambda path: str(path).endswith("/token")), \
                patch.object(Path, "read_text", return_value="token"), \
                patch("gui.backend.jobs.ssl.create_default_context"), \
                patch("gui.backend.jobs.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b"DiffSMol log"
            assert mgr._openshift_pod_log("pod-1", DIFFSMOL_ENGINE) == "DiffSMol log"
            assert "container=diffsmol" in urlopen.call_args.args[0].full_url


def test_conditar_input_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mgr = manager(Path(tmp))
        expect_value_error(
            "conDitar missing PDB",
            lambda: mgr._validated_payload({
                "engine": CONDITAR_ENGINE,
                "target": "local_cpu",
                "mode": "pocket",
                "parameters": {},
            }),
            "A PDB or CIF input is required.",
        )
        expect_value_error(
            "conDitar reference missing SDF",
            lambda: mgr._validated_payload({
                "engine": CONDITAR_ENGINE,
                "target": "local_cpu",
                "mode": "reference",
                "pdb": {"name": "protein.pdb", "text": PDB_TEXT},
                "parameters": {},
            }),
            "Reference mode requires an SDF ligand input.",
        )
        pocket_payload = mgr._validated_payload({
            "engine": CONDITAR_ENGINE,
            "target": "local_cpu",
            "mode": "pocket",
            "pdb": {"name": "protein.pdb", "text": PDB_TEXT},
            "parameters": {},
        })
        assert pocket_payload["engine"] == CONDITAR_ENGINE
        assert pocket_payload["mode"] == "pocket"
        assert pocket_payload["pdb"]["name"] == "protein.pdb"
        assert pocket_payload["sdf"] is None

        reference_payload = mgr._validated_payload({
            "engine": CONDITAR_ENGINE,
            "target": "local_cpu",
            "mode": "reference",
            "pdb": {"name": "protein.pdb", "text": PDB_TEXT},
            "sdf": {"name": "reference.sdf", "text": sdf_block("reference")},
            "parameters": {},
        })
        assert reference_payload["engine"] == CONDITAR_ENGINE
        assert reference_payload["mode"] == "reference"
        assert reference_payload["sdf"]["name"] == "reference.sdf"

        sdf_only_payload = mgr._validated_payload({
            "engine": CONDITAR_ENGINE,
            "target": "local_cpu",
            "mode": "reference",
            "sdf": {"name": "reference.sdf", "text": sdf_block("reference")},
            "parameters": {},
        })
        assert sdf_only_payload["engine"] == DIFFSMOL_ENGINE
        assert sdf_only_payload["mode"] == "reference"


def test_structure_cleanup_and_pocket_warnings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mgr = manager(Path(tmp))
        payload = mgr._validated_payload({
            "engine": CONDITAR_ENGINE,
            "target": "local_cpu",
            "mode": "pocket",
            "pdb": {"name": "protein.pdb", "text": PDB_WITH_WATER},
            "parameters": {},
        })
        assert "HOH" not in payload["pdb"]["text"]
        assert " NA A 202" not in payload["pdb"]["text"]
        assert payload["structure_cleaning"]["removed_atom_count"] == 2
        assert payload["structure_cleaning"]["removed_residue_names"] == {"HOH": 1, "NA": 1}

    pocket = pocket_pdb_from_residues(PDB_TEXT, "A:1")
    warnings = " ".join(pocket.get("warnings") or [])
    assert "fewer than 10 complete residues" in warnings
    assert parse_residue_spec("A:1A, A:2-3") == [
        ("A", "1", "A"), ("A", "2", ""), ("A", "3", ""),
    ]


def test_diffsmol_pocket_command_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project_root = root / "gui"
        mgr = manager(project_root)
        mgr.container_runtime = "docker"
        paths = JobPaths(
            root=root / "job",
            inputs=root / "job" / "inputs",
            outputs=root / "job" / "outputs",
            logs=root / "job" / "logs",
            metadata=root / "job" / "job.json",
            stdout=root / "job" / "logs" / "stdout.log",
            stderr=root / "job" / "logs" / "stderr.log",
        )
        paths.inputs.mkdir(parents=True)
        paths.outputs.mkdir(parents=True)
        paths.logs.mkdir(parents=True)
        pdb_path = paths.inputs / "protein.pdb"
        sdf_path = paths.inputs / "reference.sdf"
        pdb_path.write_text(PDB_TEXT)
        sdf_path.write_text(sdf_block("reference"))

        ligand_only = mgr._build_diffsmol_docker_command(
            paths,
            sdf_path,
            {"num_samples": 1, "diffsmol_guidance": True},
        )
        with_pocket = mgr._build_diffsmol_docker_command(
            paths,
            sdf_path,
            {"num_samples": 1, "diffsmol_guidance": True},
            pdb_path=pdb_path,
        )

        assert "/opt/DiffSMol/docker/run.py" in ligand_only
        assert "/opt/DiffSMol/docker/run.py" in with_pocket
        assert not any("/launcher" in part for part in with_pocket)
        assert "--protein" in with_pocket
        assert f"/inputs/{pdb_path.name}" in with_pocket


def test_diffsmol_postprocess_requires_pocket_for_vina() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        options = mgr._postprocess_options({
            "vina": True,
            "vina_mode": "all",
            "metrics": ["qed", "sa", "vina_score", "qvina"],
        })
        ligand_only = mgr._postprocess_for_inputs(DIFFSMOL_ENGINE, options, None)
        with_pocket = mgr._postprocess_for_inputs(DIFFSMOL_ENGINE, options, root / "pocket.pdb")

        assert ligand_only["vina"] is False
        assert ligand_only["vina_mode"] == "none"
        assert ligand_only["metrics"] == ["qed", "sa"]
        assert "Pocket PDB was not provided" in ligand_only["vina_skipped_reason"]

        assert with_pocket["vina"] is True
        assert with_pocket["vina_mode"] == "all"
        assert "vina_score" in with_pocket["metrics"]
        assert "qvina" in with_pocket["metrics"]


def test_postprocess_metrics_are_authoritative() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project_root = root / "gui"
        mgr = manager(project_root)
        mgr.container_runtime = "docker"
        score_only = mgr._postprocess_options({
            "vina": True,
            "vina_mode": "all",
            "metrics": ["vina_score", "qed", "sa"],
        })
        chemistry_only = mgr._postprocess_options({
            "vina": True,
            "vina_mode": "all",
            "metrics": ["qed", "sa"],
        })
        legacy_disabled = mgr._postprocess_options({
            "vina": False,
            "vina_mode": "none",
        })
        expect_value_error(
            "unsupported evaluator rejected before generation",
            lambda: mgr._postprocess_options({"metrics": ["not_a_metric"]}),
            "Unsupported evaluation metrics",
        )

        assert score_only["vina"] is True
        assert score_only["vina_mode"] == "vina_score"
        assert "vina_dock" not in score_only["metrics"]
        assert "qvina" not in score_only["metrics"]
        assert chemistry_only["vina"] is False
        assert chemistry_only["vina_mode"] == "none"
        assert legacy_disabled["metrics"] == []

        score_command = []
        mgr._append_postprocess_args(score_command, score_only)
        assert score_command[score_command.index("--postprocess-metrics") + 1] == "vina_score,qed,sa"
        assert "--vina-score" in score_command

        chemistry_command = []
        mgr._append_postprocess_args(chemistry_command, chemistry_only)
        assert chemistry_command[chemistry_command.index("--postprocess-metrics") + 1] == "qed,sa"
        assert "--vina-score" not in chemistry_command

        paths = JobPaths(
            root=root / "job",
            inputs=root / "job" / "inputs",
            outputs=root / "job" / "outputs",
            logs=root / "job" / "logs",
            metadata=root / "job" / "job.json",
            stdout=root / "job" / "logs" / "stdout.log",
            stderr=root / "job" / "logs" / "stderr.log",
        )
        paths.inputs.mkdir(parents=True)
        paths.outputs.mkdir(parents=True)
        paths.logs.mkdir(parents=True)
        pdb_path = paths.inputs / "pocket.pdb"
        pdb_path.write_text(PDB_TEXT)
        docker_command = mgr._build_docker_command(
            paths,
            pdb_path,
            None,
            {"num_samples": 1},
            postprocess=score_only,
        )
        assert docker_command[docker_command.index("--postprocess-metrics") + 1] == "vina_score,qed,sa"
        mounts = [docker_command[index + 1] for index, value in enumerate(docker_command[:-1]) if value == "-v"]
        assert not any(mount.endswith(":/usr/local/bin/conditar-sample:ro") for mount in mounts)
        assert not any(mount.endswith(":/opt/conditar/app/scripts/conDitar/postprocess_vina.py:ro") for mount in mounts)


def test_conditar_standalone_image_commands_and_legacy_guard() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root / "gui")
        mgr.container_runtime = "docker"
        paths = mgr._paths("standalone")
        paths.inputs.mkdir(parents=True)
        paths.outputs.mkdir(parents=True)
        pdb_path = paths.inputs / "pocket.pdb"
        pdb_path.write_text(PDB_TEXT)
        options = mgr._postprocess_options({"metrics": ["qed"]})
        for target in ("local_cpu", "slurm_gpu", "openshift_job"):
            command = mgr._build_command(
                CONDITAR_ENGINE, paths, pdb_path, None, {"num_samples": 1}, target, options,
            )
            assert command[command.index("--postprocess-metrics") + 1] == "qed"
            assert "/usr/local/bin/conditar-sample:ro" not in " ".join(command)
            assert "/opt/conditar/app/scripts/conDitar/postprocess_vina.py:ro" not in " ".join(command)
            expected_device = "cpu" if target == "local_cpu" else "cuda:0"
            assert command[max(index for index, arg in enumerate(command) if arg == "--device") + 1] == expected_device
            if target == "slurm_gpu":
                assert "nvidia.com/gpu=all" in command
                with patch.dict(os.environ, {"CONDITAR_DOCKER_TAR": "/shared/conditar-standalone.tar"}):
                    slurm_mgr = manager(root / "gui")
                    script = slurm_mgr._slurm_script(
                        {"id": "standalone", "engine": CONDITAR_ENGINE,
                         "parameters": {"num_samples": 1}, "postprocess": options},
                        paths, pdb_path, None,
                        {"cpus": "2", "mem": "8G", "time": "00:30:00", "gpus": "1",
                         "account": "", "partition": ""},
                    )
                assert "/shared/conditar-standalone.tar" in script
                assert "--postprocess-metrics" in script
            if target == "openshift_job":
                manifest = mgr._openshift_job_manifest(
                    {"id": "standalone", "engine": CONDITAR_ENGINE, "command": command}, paths,
                )
                container = manifest["spec"]["template"]["spec"]["containers"][0]
                assert container["image"] == mgr.docker_image
                assert container["name"] == "conditar"
                assert container["args"] == command
                assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
        with patch.dict(os.environ, {"CONDITAR_OPENSHIFT_DEVICE": "cpu", "CONDITAR_OPENSHIFT_GPU_COUNT": "0"}):
            command = mgr._build_command(
                CONDITAR_ENGINE, paths, pdb_path, None, {"num_samples": 1},
                "openshift_job", options,
            )
            manifest = mgr._openshift_job_manifest(
                {"id": "standalone", "engine": CONDITAR_ENGINE, "command": command}, paths,
            )
            assert command[command.index("--device") + 1] == "cpu"
            assert "nvidia.com/gpu" not in manifest["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        with patch.dict(os.environ, {"CONDITAR_DOCKER_IMAGE": "osuninglab/conditar-dev:2026-07-10"}):
            legacy_mgr = manager(root / "gui")
            legacy_mgr.container_runtime = "docker"
            expect_value_error(
                "legacy conDitar image exact metrics",
                lambda: legacy_mgr._build_docker_command(
                    paths, pdb_path, None, {"num_samples": 1}, postprocess=options,
                ),
                "require the refreshed standalone conDitar image",
            )
        existing_jobs = sorted(path.name for path in mgr.job_root.iterdir())
        with patch.dict(os.environ, {"CONDITAR_DOCKER_IMAGE": "osuninglab/conditar-dev:2026-07-10"}):
            legacy_mgr = manager(root / "gui")
            expect_value_error(
                "conDitar metrics rejected before job allocation",
                lambda: legacy_mgr.submit({
                    "engine": CONDITAR_ENGINE,
                    "target": "openshift_job",
                    "mode": "pocket",
                    "pdb": {"name": "pocket.pdb", "text": PDB_TEXT},
                    "postprocess": {"metrics": ["qed"]},
                }),
                "require the refreshed standalone conDitar image",
            )
        assert sorted(path.name for path in mgr.job_root.iterdir()) == existing_jobs


def test_backend_workflow_contract_matrix() -> None:
    cases = [
        ("conDitar pocket", CONDITAR_ENGINE, "pocket", True, False, "conditar_pocket", []),
        ("conDitar reference", CONDITAR_ENGINE, "reference", True, True, "conditar_reference", []),
        ("DiffSMol ligand only", DIFFSMOL_ENGINE, "reference", False, True, "diffsmol_ligand", []),
        ("DiffSMol ligand + pocket", DIFFSMOL_ENGINE, "reference", True, True, "diffsmol_pocket", []),
        ("DiffSMol missing SDF", DIFFSMOL_ENGINE, "reference", True, False, "diffsmol_pocket", ["DiffSMol requires a 3D reference ligand SDF input."]),
        ("conDitar missing PDB", CONDITAR_ENGINE, "pocket", False, False, "conditar_pocket", ["A PDB or CIF input is required."]),
        ("conDitar reference missing SDF", CONDITAR_ENGINE, "reference", True, False, "conditar_reference", ["Reference mode requires an SDF ligand input."]),
    ]
    for label, engine, mode, has_pdb, has_sdf, workflow_id, errors in cases:
        assert resolve_workflow(engine, mode, has_pdb).id == workflow_id, label
        assert validate_generation_inputs(engine, mode, has_pdb, has_sdf) == errors, label


def write_completed_job(root: Path, job_id: str, sdf_name: str, sdf_text: str) -> None:
    job_root = root / "job_data" / "jobs" / job_id
    outputs = job_root / "outputs"
    logs = job_root / "logs"
    outputs.mkdir(parents=True)
    logs.mkdir(parents=True)
    (outputs / sdf_name).write_text(sdf_text)
    (logs / "stdout.log").write_text("")
    (logs / "stderr.log").write_text("")
    job = {
        "id": job_id,
        "target": "local_cpu",
        "status": "completed",
        "created_at": "2026-09-28T00:00:00+00:00",
        "started_at": "2026-09-28T00:00:01+00:00",
        "finished_at": "2026-09-28T00:00:02+00:00",
        "engine": "conditar",
        "mode": "pocket",
        "input_name": job_id,
        "inputs": {"pdb": None, "sdf": None, "preprocess_metadata": None},
        "outputs": {"directory": "outputs", "sdf_count": 1},
        "parameters": {},
        "postprocess": {},
        "command": [],
    }
    (job_root / "job.json").write_text(json.dumps(job))


def test_results_are_job_scoped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "job-a", "a.sdf", sdf_block("job-a", 0.0))
        write_completed_job(root, "job-b", "b.sdf", sdf_block("job-b", 5.0))

        first = mgr.results("job-a")
        second = mgr.results("job-b")

        assert first["job"]["id"] == "job-a"
        assert second["job"]["id"] == "job-b"
        assert first["files"][0]["relative_path"] == "outputs/a.sdf"
        assert second["files"][0]["relative_path"] == "outputs/b.sdf"
        assert first["files"][0]["sha256"] != second["files"][0]["sha256"]


def test_multi_record_outputs_and_chemistry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "multi", "generated.sdf", sdf_block("first") + sdf_block("second", 3.0))
        job_root = root / "job_data" / "jobs" / "multi"
        job = json.loads((job_root / "job.json").read_text())
        job["engine"] = DIFFSMOL_ENGINE
        (job_root / "job.json").write_text(json.dumps(job))
        (job_root / "outputs" / "reference.sdf").write_text(sdf_block("reference"))

        result = mgr.results("multi")
        assert len(result["files"]) == 2
        assert all("record_" in item["name"] for item in result["files"])
        assert (job_root / "raw_outputs" / "generated.sdf").exists()
        assert result["summary"]["sdf_count"] == 2
        assert all(item["chemistry"]["formula"] == "CH4" for item in result["files"])
        assert not any(item["name"] == "reference.sdf" for item in result["files"])

        export = mgr.export_job("multi", {
            "selected_paths": [result["files"][0]["relative_path"]],
        })
        assert export["selected_count"] == 1
        assert sdf_chemistry(result["files"][0]["text"])["molecular_weight"] == 16.0


def test_export_preserves_generated_subdirectories() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "nested", "seed.sdf", sdf_block("seed"))
        paths = mgr._paths("nested")
        (paths.outputs / "seed.sdf").unlink()
        for folder in ("a", "b"):
            destination = paths.outputs / folder
            destination.mkdir()
            (destination / "generated.sdf").write_text(sdf_block(folder))
        result = mgr.results("nested")
        export = mgr.export_job("nested", {
            "selected_paths": [item["relative_path"] for item in result["files"]],
        })
        exported = Path(export["directory"]) / "generated_structures"
        assert (exported / "a" / "generated.sdf").exists()
        assert (exported / "b" / "generated.sdf").exists()


def test_finalization_waits_for_selected_evaluations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "finalizing", "generated.sdf", sdf_block("candidate"))
        paths = mgr._paths("finalizing")
        job = mgr._read_job("finalizing")
        job["status"] = "running"
        job["engine"] = DIFFSMOL_ENGINE
        job["postprocess"] = {"metrics": ["qed"]}
        job["tools"] = [{"id": "medchem_filters", "status": "pending", "options": {}}]
        mgr._write_job(paths, job)
        seen = []

        def postprocess(_paths, current):
            seen.append(mgr._read_job(current["id"])["status"])
            current["postprocess"]["status"] = "completed"
            mgr._write_job(_paths, current)

        def tool(_paths, current):
            seen.append(mgr._read_job(current["id"])["status"])
            current["tools"][0]["status"] = "completed"
            mgr._write_job(_paths, current)

        mgr._run_builtin_postprocess = postprocess
        mgr._run_requested_tools = tool
        mgr._send_email = lambda *_: None
        completed = mgr._finalize_generated_job(paths, job)
        assert seen == ["running", "running"]
        assert completed["status"] == "completed"
        assert completed["postprocess"]["status"] == "completed"
        assert completed["tools"][0]["status"] == "completed"
        manifest = json.loads((paths.outputs / "run_manifest.json").read_text())
        assert manifest["job"]["status"] == "completed"
        assert len(manifest["outputs"]["generated_sdfs"]) == 1
        assert not any(item["name"] == "run_manifest.json" for item in manifest["outputs"]["artifacts"])


def test_final_manifest_separates_generated_molecules_and_reference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "provenance", "generated.sdf", sdf_block("first") + sdf_block("second", 3.0))
        paths = mgr._paths("provenance")
        paths.inputs.mkdir()
        (paths.inputs / "reference.sdf").write_text(sdf_block("reference"))
        job = mgr._read_job("provenance")
        job["status"] = "running"
        job["engine"] = DIFFSMOL_ENGINE
        job["inputs"]["sdf"] = "inputs/reference.sdf"
        mgr._write_job(paths, job)
        mgr._run_builtin_postprocess = lambda *_: None
        mgr._run_requested_tools = lambda *_: None
        mgr._send_email = lambda *_: None

        completed = mgr._finalize_generated_job(paths, job)
        assert completed["status"] == "completed"
        manifest = json.loads((paths.outputs / "run_manifest.json").read_text())
        assert manifest["inputs"]["sdf"]["relative_path"] == "inputs/reference.sdf"
        assert manifest["inputs"]["sdf"]["sha256"]
        generated = manifest["outputs"]["generated_sdfs"]
        assert len(generated) == 2
        assert all("record_" in item["name"] and item["sha256"] for item in generated)
        assert all(item["name"] != "reference.sdf" for item in generated)
        assert any(item["relative_path"] == "raw_outputs/generated.sdf" for item in manifest["outputs"]["artifacts"])


def test_ligand_only_rerun_preserves_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "failed-ligand", "generated.sdf", sdf_block("candidate"))
        paths = mgr._paths("failed-ligand")
        (paths.inputs).mkdir(exist_ok=True)
        (paths.inputs / "reference.sdf").write_text(sdf_block("reference"))
        job = mgr._read_job("failed-ligand")
        job.update(status="failed", engine=DIFFSMOL_ENGINE, preprocess={"workflow": "bound_complex"})
        job["inputs"]["sdf"] = "inputs/reference.sdf"
        job["tools"] = [{"id": "medchem_filters", "options": {}, "status": "failed"}]
        mgr._write_job(paths, job)
        captured = {}

        def submit(payload):
            captured.update(payload)
            return {"id": "new-ligand", "status": "queued"}

        mgr.submit = submit
        rerun = mgr.rerun_job("failed-ligand")
        assert captured["pdb"] is None
        assert captured["engine"] == DIFFSMOL_ENGINE
        assert captured["preprocess"] == job["preprocess"]
        assert captured["tools"] == [{"id": "medchem_filters", "options": {}}]
        assert captured["rerun_of"] == "failed-ligand"
        assert rerun["id"] == "new-ligand"


def test_evaluator_excludes_reference_ligand() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "outputs").mkdir()
        (root / "tool_run").mkdir()
        (root / "outputs" / "generated.sdf").write_text(sdf_block("generated"))
        (root / "outputs" / "reference.sdf").write_text(sdf_block("reference"))
        result = run_medchem_filters(str(root), str(root / "tool_run"), {})
        assert result["molecules"] == 1
        assert result["generated_sdfs"] == ["outputs/generated.sdf"]
        assert "MEDCHEM_STATUS" not in (root / "outputs" / "reference.sdf").read_text()


def test_recovery_keeps_partial_generator_outputs_running() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "partial", "generated.sdf", sdf_block("partial"))
        paths = mgr._paths("partial")
        job = mgr._read_job("partial")
        job["status"] = "running"
        job["phase"] = "generating"
        mgr._write_job(paths, job)
        mgr._local_job_process_running = lambda _job_id: True
        mgr._recover_incomplete_jobs()
        recovered = mgr._read_job("partial")
        assert recovered["status"] == "running"
        assert recovered["phase"] == "generating"
        assert recovered["finished_at"] is None


def test_remote_outputs_wait_for_scheduler_completion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "slurm-partial", "generated.sdf", sdf_block("partial"))
        job = mgr._read_job("slurm-partial")
        job.update(status="running", target="slurm_gpu", slurm={"job_id": "123"})
        mgr._slurm_state = lambda _job: "RUNNING"
        assert mgr._refresh_job(job)["status"] == "running"

        write_completed_job(root, "openshift-partial", "generated.sdf", sdf_block("partial"))
        job = mgr._read_job("openshift-partial")
        job.update(status="running", target="openshift_job", openshift={"submitted": True, "job_name": "partial"})
        mgr._openshift_job_state = lambda _name: {"state": "running"}
        assert mgr._refresh_job(job)["status"] == "running"


def test_diffsmol_postprocessing_remains_running_until_exit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "postprocess", "generated.sdf", sdf_block("candidate"))
        paths = mgr._paths("postprocess")
        job = mgr._read_job("postprocess")
        job.update(status="running", engine=DIFFSMOL_ENGINE, postprocess={"metrics": ["qed"], "vina": False})
        mgr._write_job(paths, job)
        mgr.container_runtime = "docker"
        mgr._container_image_status = lambda _image: {"checked": True, "exists": True}
        seen = []

        class FakeProcess:
            def wait(self):
                current = mgr._read_job("postprocess")
                seen.append((current["status"], current["postprocess"]["status"]))
                return 0

        with patch("gui.backend.jobs.subprocess.Popen", return_value=FakeProcess()) as launch:
            mgr._run_builtin_postprocess(paths, job)
        command = launch.call_args.args[0]
        assert mgr.diffsmol_image in command
        assert mgr.docker_image not in command
        assert "/opt/DiffSMol/docker/postprocess.py" in command
        assert "/workspace" not in command
        assert seen == [("running", "running")]
        assert mgr._read_job("postprocess")["status"] == "running"
        assert mgr._read_job("postprocess")["postprocess"]["status"] == "completed"


def test_diffsmol_in_container_evaluation_status() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "evaluated", "generated.sdf", sdf_block("candidate"))
        paths = mgr._paths("evaluated")
        job = mgr._read_job("evaluated")
        job.update(
            engine=DIFFSMOL_ENGINE,
            postprocess={"metrics": ["qed"], "vina_mode": "none"},
            command=["python", "/opt/DiffSMol/docker/run.py"],
        )
        mgr._write_job(paths, job)
        expect_value_error(
            "missing evaluator status",
            lambda: mgr._run_builtin_postprocess(paths, job),
            "without an evaluation status file",
        )
        status = paths.outputs / "evaluation_status.json"
        status.write_text(json.dumps({"status": "completed", "metrics": ["qed"], "exit_code": 0}))
        mgr._run_builtin_postprocess(paths, job)
        completed = mgr._read_job("evaluated")
        assert completed["postprocess"]["status"] == "completed"
        assert completed["postprocess"]["applied_metrics"] == ["qed"]


def test_conditar_in_container_evaluation_status() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mgr = manager(root)
        write_completed_job(root, "conditar-evaluated", "generated.sdf", sdf_block("candidate"))
        paths = mgr._paths("conditar-evaluated")
        job = mgr._read_job("conditar-evaluated")
        job.update(
            engine=CONDITAR_ENGINE,
            postprocess={"metrics": ["qed"], "vina_mode": "none"},
        )
        mgr._write_job(paths, job)
        expect_value_error(
            "missing conDitar evaluator status",
            lambda: mgr._run_builtin_postprocess(paths, job),
            "without an evaluation status file",
        )
        status = paths.outputs / "evaluation_status.json"
        status.write_text(json.dumps({"status": "completed", "metrics": ["sa"], "exit_code": 0}))
        expect_value_error(
            "mismatched conDitar evaluator status",
            lambda: mgr._run_builtin_postprocess(paths, job),
            "does not match the selected metrics",
        )
        status.write_text(json.dumps({"status": "completed", "metrics": ["qed"], "exit_code": 0}))
        mgr._run_builtin_postprocess(paths, job)
        completed = mgr._read_job("conditar-evaluated")
        assert completed["postprocess"]["status"] == "completed"
        assert completed["postprocess"]["applied_metrics"] == ["qed"]


def main() -> None:
    test_docker_image_inspection_fallback()
    test_diffsmol_input_validation()
    test_diffsmol_cross_target_commands()
    test_conditar_input_validation()
    test_structure_cleanup_and_pocket_warnings()
    test_diffsmol_pocket_command_selection()
    test_diffsmol_postprocess_requires_pocket_for_vina()
    test_postprocess_metrics_are_authoritative()
    test_conditar_standalone_image_commands_and_legacy_guard()
    test_backend_workflow_contract_matrix()
    test_results_are_job_scoped()
    test_multi_record_outputs_and_chemistry()
    test_export_preserves_generated_subdirectories()
    test_finalization_waits_for_selected_evaluations()
    test_final_manifest_separates_generated_molecules_and_reference()
    test_ligand_only_rerun_preserves_context()
    test_evaluator_excludes_reference_ligand()
    test_recovery_keeps_partial_generator_outputs_running()
    test_remote_outputs_wait_for_scheduler_completion()
    test_diffsmol_postprocessing_remains_running_until_exit()
    test_diffsmol_in_container_evaluation_status()
    test_conditar_in_container_evaluation_status()
    print("GUI workflow smoke checks passed.")


if __name__ == "__main__":
    main()
