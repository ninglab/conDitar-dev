#!/usr/bin/env python3
"""Check the standalone DiffSMol container entry point without model inference."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[1] / "runtime" / "run.py"
spec = importlib.util.spec_from_file_location("diffsmol_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdf = root / "reference.sdf"
        pdb = root / "pocket.pdb"
        output = root / "results"
        sdf.write_text("reference\n")
        pdb.write_text("ATOM\n")
        args = ["run.py", "--input", str(sdf), "--protein", str(pdb),
                "--output", str(output), "--device", "cpu",
                "--postprocess-metrics", "vina_score,qed"]
        with patch.object(sys, "argv", args), patch.object(runner.subprocess, "run") as invoke:
            invoke.return_value.returncode = 0
            assert runner.main() == 0
            assert invoke.call_count == 2
            assert "pocket_generate.py" in invoke.call_args_list[0].args[0][1]
            assert "postprocess.py" in invoke.call_args_list[1].args[0][1]
            assert json.loads((output / runner.STATUS_FILE).read_text()) == {
                "status": "completed", "metrics": ["vina_score", "qed"], "exit_code": 0,
            }
        with patch.object(sys, "argv", args), patch.object(runner.subprocess, "run") as invoke:
            invoke.side_effect = [type("Result", (), {"returncode": 0})(),
                                  type("Result", (), {"returncode": 3})()]
            assert runner.main() == 3
            assert json.loads((output / runner.STATUS_FILE).read_text())["status"] == "failed"
        invalid = ["run.py", "--input", str(sdf), "--output", str(output),
                   "--postprocess-metrics", "vina_score"]
        with patch.object(sys, "argv", invalid), patch.object(runner.subprocess, "run") as invoke:
            try:
                runner.main()
            except SystemExit as error:
                assert "require --protein" in str(error)
            else:
                raise AssertionError("Docking without a pocket must fail")
            invoke.assert_not_called()
    print("DiffSMol runner checks passed.")


if __name__ == "__main__":
    main()
