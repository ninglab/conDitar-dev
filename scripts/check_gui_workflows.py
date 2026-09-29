#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.backend.jobs import DIFFSMOL_ENGINE, LocalJobManager


PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.000  1.00 20.00           O
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
    test_results_are_job_scoped()
    print("GUI workflow smoke checks passed.")


if __name__ == "__main__":
    main()
