from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

from utils import scoring_func
from utils.docking_vina import VinaDockingTask


ALLOWED_VINA_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
DOCKING_MODES = ("none", "vina_score", "vina_dock", "qvina", "all")


def is_vina_compatible(mol: Chem.Mol) -> bool:
    return all(atom.GetSymbol() in ALLOWED_VINA_ELEMENTS for atom in mol.GetAtoms())


def affinity(results: list[dict]) -> float | None:
    if not results:
        return None
    value = results[0].get("affinity")
    return float(value) if value is not None else None


def score_molecule(
    mol: Chem.Mol,
    protein: Path,
    tmp_dir: Path,
    mode: str,
    exhaustiveness: int,
    cpu: int,
    qvina_bin: str | None = None,
) -> dict:
    chem_results = scoring_func.get_chem(mol)
    vina_results = {}
    task = None

    if mode in {"vina_score", "vina_dock", "all"}:
        task = task or VinaDockingTask(str(protein), mol, tmp_dir=str(tmp_dir))
        vina_results["score_only"] = task.run(mode="score_only", exhaustiveness=exhaustiveness, cpu=cpu)
        vina_results["minimize"] = task.run(mode="minimize", exhaustiveness=exhaustiveness, cpu=cpu)
        if mode == "vina_dock":
            vina_results["dock"] = task.run(mode="dock", exhaustiveness=exhaustiveness, cpu=cpu)
    if mode in {"qvina", "all"}:
        task = task or VinaDockingTask(str(protein), mol, tmp_dir=str(tmp_dir))
        vina_results["qvina"] = task.qvina(exhaustiveness=exhaustiveness, qvina_bin=qvina_bin)

    return {
        "chem_results": chem_results,
        "vina": vina_results,
    }


def set_prop(mol: Chem.Mol, name: str, value) -> None:
    if value not in (None, ""):
        mol.SetProp(name, str(value))


def annotate_molecule(
    mol: Chem.Mol,
    protein: Path,
    tmp_dir: Path,
    mode: str,
    exhaustiveness: int,
    cpu: int,
    qvina_bin: str | None = None,
) -> bool:
    for prop in ("VINA_SCORE_ONLY", "VINA_MINIMIZE", "VINA_DOCK", "QVINA", "VINA_ERROR"):
        if mol.HasProp(prop):
            mol.ClearProp(prop)
    mol.SetProp("VINA_MODE", mode)
    mol.SetProp("VINA_EXHAUSTIVENESS", str(exhaustiveness))
    mol.SetProp("VINA_CPU", str(cpu))
    try:
        smiles = Chem.MolToSmiles(mol)
        mol.SetProp("SMILES", smiles)
        if "." in smiles:
            raise ValueError("Molecule has separate fragments.")
        if mode != "none" and not is_vina_compatible(mol):
            raise ValueError("Molecule has atoms unsupported by Vina.")

        result = score_molecule(mol, protein, tmp_dir, mode, exhaustiveness, cpu, qvina_bin=qvina_bin)
        chem_results = result["chem_results"]
        set_prop(mol, "QED", chem_results.get("qed"))
        set_prop(mol, "SA", chem_results.get("sa"))
        set_prop(mol, "LOGP", chem_results.get("logp"))
        set_prop(mol, "LIPINSKI", chem_results.get("lipinski"))
        set_prop(mol, "VINA_SCORE_ONLY", affinity(result["vina"].get("score_only", [])))
        set_prop(mol, "VINA_MINIMIZE", affinity(result["vina"].get("minimize", [])))
        set_prop(mol, "VINA_DOCK", affinity(result["vina"].get("dock", [])))
        set_prop(mol, "QVINA", affinity([{"affinity": score} for score in result["vina"].get("qvina", [])]))
        mol.SetProp("VINA_STATUS", "not_run" if mode == "none" else "ok")
        if mol.HasProp("VINA_ERROR"):
            mol.ClearProp("VINA_ERROR")
        return True
    except Exception as error:
        mol.SetProp("VINA_STATUS", "failed")
        mol.SetProp("VINA_ERROR", str(error))
        return False


def annotate_sdf_file(
    sdf_path: Path,
    protein: Path,
    tmp_dir: Path,
    mode: str,
    exhaustiveness: int,
    cpu: int,
    qvina_bin: str | None = None,
) -> tuple[int, int]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]
    if not molecules:
        return 0, 0

    ok_count = 0
    tmp_path = sdf_path.with_suffix(sdf_path.suffix + ".tmp")
    writer = Chem.SDWriter(str(tmp_path))
    try:
        for mol in molecules:
            if annotate_molecule(mol, protein, tmp_dir, mode, exhaustiveness, cpu, qvina_bin=qvina_bin):
                ok_count += 1
            writer.write(mol)
    finally:
        writer.close()
    tmp_path.replace(sdf_path)
    return len(molecules), ok_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process generated conDitar SDFs with Vina scoring.")
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--tmp-dir", default="/tmp/conditar/vina", type=Path)
    parser.add_argument("--mode", choices=DOCKING_MODES, default="vina_score")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--qvina-bin", default=None, help="QuickVina2 executable for qvina/all modes.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog("rdApp.*")
    if not args.protein.exists():
        raise FileNotFoundError(f"Protein PDB not found: {args.protein}")
    if not args.generated_dir.exists():
        raise FileNotFoundError(f"Generated SDF directory not found: {args.generated_dir}")

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    for sdf_path in sorted(args.generated_dir.rglob("*.sdf")):
        count, ok_count = annotate_sdf_file(
            sdf_path,
            args.protein,
            args.tmp_dir,
            args.mode,
            args.exhaustiveness,
            args.cpu,
            qvina_bin=args.qvina_bin,
        )
        total += count
        ok += ok_count

    print(f"Annotated Vina scores in generated SDFs for {ok}/{total} molecules")
    return 0 if total and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
