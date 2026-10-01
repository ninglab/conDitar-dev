from __future__ import annotations

from io import BytesIO


def sdf_chemistry(text: str) -> dict | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        supplier = Chem.ForwardSDMolSupplier(BytesIO(text.encode("utf-8")))
        mol = next((item for item in supplier if item is not None), None)
        if mol is None:
            return None
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 1),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
        }
    except (ImportError, OSError, ValueError, RuntimeError):
        return None
