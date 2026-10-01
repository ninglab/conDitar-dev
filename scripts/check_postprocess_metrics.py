#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.conDitar import postprocess_vina


class FakeVinaTask:
    calls: list[str] = []

    def __init__(self, protein: str, mol: Chem.Mol, tmp_dir: str):
        self.protein = protein
        self.mol = mol
        self.tmp_dir = tmp_dir

    def run(self, mode: str, **kwargs):
        self.calls.append(mode)
        return [{"affinity": {"score_only": -1.0, "minimize": -2.0, "dock": -3.0}[mode]}]

    def qvina(self, **kwargs):
        self.calls.append("qvina")
        return [-4.0]


def main() -> None:
    chemistry = {"qed": 0.7, "sa": 2.1, "logp": 1.2, "lipinski": 5}
    with tempfile.TemporaryDirectory() as tmp:
        mol = Chem.MolFromSmiles("CCO")
        with patch.object(postprocess_vina.scoring_func, "get_chem", return_value=chemistry):
            assert postprocess_vina.annotate_molecule(
                mol, None, Path(tmp), "all", 8, 1, metrics={"qed"},
            )
        assert mol.HasProp("QED")
        assert not mol.HasProp("SA")
        assert not mol.HasProp("LOGP")
        assert not mol.HasProp("LIPINSKI")
        assert not mol.HasProp("VINA_DOCK")
        assert not mol.HasProp("QVINA")
        assert mol.GetProp("VINA_STATUS") == "not_run"
        assert mol.GetProp("EVALUATION_METRICS") == "qed"

        FakeVinaTask.calls = []
        mol = Chem.MolFromSmiles("CCO")
        with (
            patch.object(postprocess_vina, "VinaDockingTask", FakeVinaTask),
            patch.object(postprocess_vina.scoring_func, "get_chem", return_value=chemistry),
        ):
            assert postprocess_vina.annotate_molecule(
                mol,
                Path(tmp) / "pocket.pdb",
                Path(tmp),
                "all",
                8,
                1,
                metrics={"vina_score", "qvina", "qed"},
            )
        assert FakeVinaTask.calls == ["score_only", "minimize", "qvina"]
        assert mol.HasProp("VINA_SCORE_ONLY")
        assert mol.HasProp("VINA_MINIMIZE")
        assert mol.HasProp("QVINA")
        assert not mol.HasProp("VINA_DOCK")
        assert mol.HasProp("QED")
        assert not mol.HasProp("SA")
        assert mol.GetProp("EVALUATION_METRICS") == "vina_score,qvina,qed"

        outputs = Path(tmp) / "outputs"
        outputs.mkdir()
        (outputs / "generated.sdf").write_text("generated\n")
        (outputs / "reference.sdf").write_text("reference\n")
        visited = []

        def annotate(path, *_args, **_kwargs):
            visited.append(path.name)
            return 1, 1

        with (
            patch.object(sys, "argv", [
                "postprocess_vina", "--generated-dir", str(outputs), "--tmp-dir", str(Path(tmp) / "work"),
                "--mode", "none", "--metrics", "qed",
                "--status-file", str(outputs / "evaluation_status.json"),
            ]),
            patch.object(postprocess_vina, "annotate_sdf_file", side_effect=annotate),
        ):
            assert postprocess_vina.main() == 0
        assert visited == ["generated.sdf"]
        assert json.loads((outputs / "evaluation_status.json").read_text()) == {
            "status": "completed", "metrics": ["qed"], "exit_code": 0,
            "molecules_total": 1, "molecules_annotated": 1,
        }

    print("Postprocess metric selection checks passed.")


if __name__ == "__main__":
    main()
