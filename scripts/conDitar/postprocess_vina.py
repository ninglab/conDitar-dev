from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

from utils import scoring_func
from utils.docking_vina import VinaDockingTask


ALLOWED_VINA_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


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
) -> dict:
    chem_results = scoring_func.get_chem(mol)
    task = VinaDockingTask(str(protein), mol, tmp_dir=str(tmp_dir))
    vina_results = {}

    if mode in {"vina_score", "vina_dock"}:
        vina_results["score_only"] = task.run(mode="score_only", exhaustiveness=exhaustiveness, cpu=cpu)
        vina_results["minimize"] = task.run(mode="minimize", exhaustiveness=exhaustiveness, cpu=cpu)
        if mode == "vina_dock":
            vina_results["dock"] = task.run(mode="dock", exhaustiveness=exhaustiveness, cpu=cpu)

    return {
        "chem_results": chem_results,
        "vina": vina_results,
    }


def iter_sdf_molecules(generated_dir: Path):
    for sdf_path in sorted(generated_dir.rglob("*.sdf")):
        if "eval_results" in sdf_path.parts:
            continue
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for mol_index, mol in enumerate(supplier):
            yield sdf_path, mol_index, mol


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process generated conDitar SDFs with Vina scoring.")
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tmp-dir", default="/tmp/conditar/vina", type=Path)
    parser.add_argument("--mode", choices=["vina_score", "vina_dock"], default="vina_score")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog("rdApp.*")
    if not args.protein.exists():
        raise FileNotFoundError(f"Protein PDB not found: {args.protein}")
    if not args.generated_dir.exists():
        raise FileNotFoundError(f"Generated SDF directory not found: {args.generated_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sdf_path, mol_index, mol in iter_sdf_molecules(args.generated_dir):
        row = {
            "sdf": str(sdf_path.relative_to(args.generated_dir)),
            "mol_index": mol_index,
            "status": "ok",
            "smiles": "",
            "qed": "",
            "sa": "",
            "logp": "",
            "lipinski": "",
            "vina_score_only": "",
            "vina_minimize": "",
            "vina_dock": "",
            "error": "",
        }
        try:
            if mol is None:
                raise ValueError("RDKit could not parse molecule.")
            smiles = Chem.MolToSmiles(mol)
            row["smiles"] = smiles
            if "." in smiles:
                raise ValueError("Molecule has separate fragments.")
            if not is_vina_compatible(mol):
                raise ValueError("Molecule has atoms unsupported by Vina.")

            result = score_molecule(mol, args.protein, args.tmp_dir, args.mode, args.exhaustiveness, args.cpu)
            chem_results = result["chem_results"]
            row["qed"] = chem_results.get("qed", "")
            row["sa"] = chem_results.get("sa", "")
            row["logp"] = chem_results.get("logp", "")
            row["lipinski"] = chem_results.get("lipinski", "")
            row["vina_score_only"] = affinity(result["vina"].get("score_only", []))
            row["vina_minimize"] = affinity(result["vina"].get("minimize", []))
            row["vina_dock"] = affinity(result["vina"].get("dock", []))
        except Exception as error:
            row["status"] = "failed"
            row["error"] = str(error)
        rows.append(row)

    fieldnames = [
        "sdf",
        "mol_index",
        "status",
        "smiles",
        "qed",
        "sa",
        "logp",
        "lipinski",
        "vina_score_only",
        "vina_minimize",
        "vina_dock",
        "error",
    ]
    csv_path = args.out / "vina_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "mode": args.mode,
        "exhaustiveness": args.exhaustiveness,
        "cpu": args.cpu,
        "protein": str(args.protein),
        "generated_dir": str(args.generated_dir),
        "total": len(rows),
        "ok": sum(1 for row in rows if row["status"] == "ok"),
        "failed": sum(1 for row in rows if row["status"] != "ok"),
        "csv": str(csv_path),
        "rows": rows,
    }
    (args.out / "vina_scores.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote Vina scores for {summary['ok']}/{summary['total']} molecules to {csv_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
