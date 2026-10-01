"""Focused checks for the standalone DiffSMol SDF evaluator."""
from pathlib import Path
import sys
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, "/opt/DiffSMol/docker")
import postprocess  # noqa: E402


def molecule(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sdf = work / "generated.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(molecule("CCO"))

        assert postprocess.annotate_file(sdf, {"qed", "sa", "logp", "lipinski"},
                                         None, work, 8, 2, "qvina2.1") == (1, 1)
        first = next(iter(Chem.SDMolSupplier(str(sdf))))
        assert all(first.HasProp(prop) for prop in ("QED", "SA", "LOGP", "LIPINSKI"))
        assert first.GetProp("VINA_STATUS") == "not_run"
        assert not first.HasProp("VINA_SCORE_ONLY")

        assert postprocess.annotate_file(sdf, {"qed"}, None, work, 8, 2, "qvina2.1") == (1, 1)
        second = next(iter(Chem.SDMolSupplier(str(sdf))))
        assert second.HasProp("QED") and not second.HasProp("SA")
        assert second.GetProp("EVALUATION_METRICS") == "qed"

        disconnected = molecule("CC.O")
        assert not postprocess.annotate(disconnected, {"qed"}, None, work, 8, 2, "qvina2.1")
        assert disconnected.GetProp("VINA_STATUS") == "failed"

    print("DiffSMol postprocess checks passed.")


if __name__ == "__main__":
    main()
