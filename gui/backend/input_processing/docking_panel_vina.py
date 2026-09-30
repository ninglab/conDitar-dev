from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from utils.docking_vina import PrepLig, PrepProt, VinaDock


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small AutoDock Vina panel and emit docked pose centers.")
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--ligands", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--center-z", type=float, default=None)
    parser.add_argument("--size-x", type=float, default=None)
    parser.add_argument("--size-y", type=float, default=None)
    parser.add_argument("--size-z", type=float, default=None)
    parser.add_argument("--protein-buffer", type=float, default=8.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--max-ligands", type=int, default=25)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tmp_root = args.tmp_dir or Path(tempfile.mkdtemp(prefix="vina-panel-"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    if not args.protein.exists():
        raise FileNotFoundError(f"Protein PDB not found: {args.protein}")
    if not args.ligands.exists():
        raise FileNotFoundError(f"Ligand panel SDF not found: {args.ligands}")

    search_center, box_size = search_box(args)
    receptor_pdbqt = prepare_receptor(args.protein, tmp_root)
    poses = []
    warnings = []

    supplier = Chem.SDMolSupplier(str(args.ligands), removeHs=False)
    for index, mol in enumerate(supplier):
        if index >= args.max_ligands:
            warnings.append(f"Stopped after max_ligands={args.max_ligands}.")
            break
        if mol is None:
            warnings.append(f"Ligand {index + 1} could not be parsed.")
            continue
        try:
            ligand_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"ligand_{index + 1}"
            pose = dock_ligand(mol, ligand_name, index + 1, receptor_pdbqt, search_center, box_size, tmp_root, args)
            poses.append(pose)
        except Exception as error:
            warnings.append(f"Ligand {index + 1} failed: {error}")

    result = {
        "status": "ready" if poses else "no_poses",
        "method": "vina_panel",
        "search_box": {
            "center": {"x": search_center[0], "y": search_center[1], "z": search_center[2]},
            "size": {"x": box_size[0], "y": box_size[1], "z": box_size[2]},
        },
        "poses": poses,
        "warnings": warnings,
    }
    (args.out / "vina_panel_poses.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if poses else 1


def search_box(args: argparse.Namespace) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    explicit_center = (args.center_x, args.center_y, args.center_z)
    explicit_size = (args.size_x, args.size_y, args.size_z)
    if all(value is not None for value in explicit_center) and all(value is not None for value in explicit_size):
        return tuple(float(value) for value in explicit_center), tuple(float(value) for value in explicit_size)
    return protein_bounding_box(args.protein, args.protein_buffer)


def protein_bounding_box(protein: Path, buffer: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coords = []
    for line in protein.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    if not coords:
        raise ValueError("Protein PDB contains no dockable atom coordinates.")
    mins = [min(point[axis] for point in coords) for axis in range(3)]
    maxs = [max(point[axis] for point in coords) for axis in range(3)]
    center = tuple((mins[axis] + maxs[axis]) / 2 for axis in range(3))
    size = tuple(max(maxs[axis] - mins[axis] + buffer, 12.0) for axis in range(3))
    return center, size


def prepare_receptor(protein: Path, tmp_root: Path) -> Path:
    dry_pdb = tmp_root / "receptor_dry.pdb"
    pqr = tmp_root / "receptor.pqr"
    pdbqt = tmp_root / "receptor.pdbqt"
    prep = PrepProt(str(protein))
    prep.del_water(str(dry_pdb))
    prep.addH(str(pqr))
    prep.get_pdbqt(str(pdbqt))
    return pdbqt


def dock_ligand(
    mol: Chem.Mol,
    ligand_name: str,
    ligand_index: int,
    receptor_pdbqt: Path,
    search_center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    tmp_root: Path,
    args: argparse.Namespace,
) -> dict:
    mol = Chem.AddHs(mol, addCoords=True)
    if needs_3d_embedding(mol):
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status != 0:
            raise ValueError("RDKit could not generate a 3D conformer for docking.")
    ligand_sdf = tmp_root / f"ligand_{ligand_index}.sdf"
    ligand_pdbqt = tmp_root / f"ligand_{ligand_index}.pdbqt"
    writer = Chem.SDWriter(str(ligand_sdf))
    try:
        writer.write(mol)
    finally:
        writer.close()

    prep_ligand = PrepLig(str(ligand_sdf), "sdf")
    prep_ligand.get_pdbqt(str(ligand_pdbqt))
    source_pdbqt = ligand_pdbqt.read_text(errors="replace")
    dock = VinaDock(str(ligand_pdbqt), str(receptor_pdbqt))
    dock.pocket_center = search_center
    dock.box_size = box_size
    score, pose_text, _ = dock.dock(mode="dock", exhaustiveness=args.exhaustiveness, save_pose=True, cpu=args.cpu)
    center = pose_center_from_pdbqt(pose_text)
    if center is None:
        raise ValueError("Vina returned no pose coordinates.")
    pose_sdf = pose_sdf_from_template(mol, ligand_name, source_pdbqt, pose_text, score)
    return {
        "id": f"{ligand_name}:{ligand_index}",
        "name": ligand_name,
        "ligand_index": ligand_index,
        "score": float(score) if score is not None and math.isfinite(float(score)) else None,
        "center": {"x": round(center[0], 4), "y": round(center[1], 4), "z": round(center[2], 4)},
        "sdf": pose_sdf,
    }


def needs_3d_embedding(mol: Chem.Mol) -> bool:
    if not mol.GetNumConformers():
        return True
    positions = mol.GetConformer(0).GetPositions()
    if len(positions) == 0:
        return True
    z_values = [float(point[2]) for point in positions]
    return max(z_values) - min(z_values) < 0.001


def pose_sdf_from_template(
    mol: Chem.Mol,
    ligand_name: str,
    source_pdbqt: str,
    pose_text: str | None,
    score: float | None,
) -> str:
    source_atoms = pdbqt_atoms(source_pdbqt)
    pose_atoms = pdbqt_atoms(pose_text)
    if not source_atoms or len(source_atoms) != len(pose_atoms):
        raise ValueError("Vina pose atom count did not match the prepared ligand.")

    posed = Chem.Mol(mol)
    if not posed.GetNumConformers():
        raise ValueError("Prepared ligand has no conformer for pose coordinate mapping.")
    conformer = posed.GetConformer(0)
    unmatched_heavy = {atom.GetIdx() for atom in posed.GetAtoms() if atom.GetAtomicNum() > 1}

    for source_atom, pose_atom in zip(source_atoms, pose_atoms):
        if source_atom["is_hydrogen"]:
            continue
        if not unmatched_heavy:
            raise ValueError("Vina pose contains more heavy atoms than the prepared ligand.")
        source_coord = source_atom["coord"]
        matched_index = min(
            unmatched_heavy,
            key=lambda index: squared_distance(conformer.GetAtomPosition(index), source_coord),
        )
        if squared_distance(conformer.GetAtomPosition(matched_index), source_coord) > 0.04:
            raise ValueError("Vina pose atoms could not be mapped back to the prepared ligand.")
        conformer.SetAtomPosition(matched_index, pose_atom["coord"])
        unmatched_heavy.remove(matched_index)

    if unmatched_heavy:
        raise ValueError("Vina pose is missing heavy atoms from the prepared ligand.")
    posed = Chem.RemoveHs(posed)
    validate_pose_geometry(posed)
    posed.SetProp("_Name", ligand_name)
    if score is not None:
        posed.SetProp("VINA_DOCK", str(score))
    posed.SetProp("VINA_STATUS", "ok")
    return Chem.MolToMolBlock(posed) + "\n$$$$\n"


def pose_center_from_pdbqt(text: str | None) -> tuple[float, float, float] | None:
    coords = pose_coordinates_from_pdbqt(text)
    if not coords:
        return None
    count = len(coords)
    return (
        sum(point[0] for point in coords) / count,
        sum(point[1] for point in coords) / count,
        sum(point[2] for point in coords) / count,
    )


def pose_coordinates_from_pdbqt(text: str | None) -> list[tuple[float, float, float]]:
    return [atom["coord"] for atom in pdbqt_atoms(text)]


def pdbqt_atoms(text: str | None) -> list[dict]:
    if not text:
        return []
    atoms = []
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        atom_type = line.split()[-1] if line.split() else ""
        atoms.append({
            "coord": coord,
            "is_hydrogen": atom_type.upper().startswith("H"),
        })
    return atoms


def squared_distance(point, coord: tuple[float, float, float]) -> float:
    return sum((float(point[axis]) - coord[axis]) ** 2 for axis in range(3))


def validate_pose_geometry(mol: Chem.Mol, max_bond_length: float = 3.0) -> None:
    conformer = mol.GetConformer(0)
    for bond in mol.GetBonds():
        start = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        if math.sqrt(squared_distance(start, (end.x, end.y, end.z))) > max_bond_length:
            raise ValueError("Converted Vina pose contains an implausibly long bond.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        raise
