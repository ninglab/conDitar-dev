#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.backend.input_processing import pocket_pdb_from_residues
from gui.backend.jobs import CONDITAR_ENGINE, DIFFSMOL_ENGINE, JobPaths, LocalJobManager
from gui.backend.workflow_rules import resolve_workflow, validate_generation_inputs


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


def test_diffsmol_pocket_command_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project_root = root / "gui"
        launcher_path = project_root / "backend" / "diffsmol_pocket_generate.py"
        launcher_path.parent.mkdir(parents=True)
        launcher_path.write_text("# test launcher\n")
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

        assert "/opt/DiffSMol/docker/generate.py" in ligand_only
        assert "/launcher/diffsmol_pocket_generate.py" in with_pocket
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
        wrapper_path = root / "scripts" / "container" / "conditar-sample"
        postprocess_path = root / "scripts" / "conDitar" / "postprocess_vina.py"
        wrapper_path.parent.mkdir(parents=True)
        postprocess_path.parent.mkdir(parents=True)
        wrapper_path.write_text("#!/usr/bin/env bash\n")
        postprocess_path.write_text("# test postprocessor\n")
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

        assert mgr._output_completion_blocked_by_postprocess({
            "engine": CONDITAR_ENGINE,
            "postprocess": chemistry_only,
        }) is True

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
        assert any(mount.endswith(":/usr/local/bin/conditar-sample:ro") for mount in mounts)
        assert any(mount.endswith(":/opt/conditar/app/scripts/conDitar/postprocess_vina.py:ro") for mount in mounts)


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


def main() -> None:
    test_diffsmol_input_validation()
    test_conditar_input_validation()
    test_structure_cleanup_and_pocket_warnings()
    test_diffsmol_pocket_command_selection()
    test_diffsmol_postprocess_requires_pocket_for_vina()
    test_postprocess_metrics_are_authoritative()
    test_backend_workflow_contract_matrix()
    test_results_are_job_scoped()
    print("GUI workflow smoke checks passed.")


if __name__ == "__main__":
    main()
