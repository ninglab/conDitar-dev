"""Annotate generated DiffSMol SDFs with selected chemistry and docking metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import AutoDockTools
from rdkit import Chem, RDLogger
from vina import Vina

sys.path.insert(0, "/opt/DiffSMol/source")
from utils import scoring_func  # noqa: E402


METRIC_ORDER = ("vina_score", "vina_dock", "qvina", "qed", "sa", "logp", "lipinski")
VINA_METRICS = {"vina_score", "vina_dock", "qvina"}
CHEMISTRY_METRICS = {"qed", "sa", "logp", "lipinski"}
VINA_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
RESULT_PROPS = (
    "QED", "SA", "LOGP", "LIPINSKI", "VINA_SCORE_ONLY", "VINA_MINIMIZE",
    "VINA_DOCK", "QVINA", "VINA_ERROR", "VINA_EXHAUSTIVENESS", "VINA_CPU",
    "EVALUATION_METRICS", "VINA_STATUS", "VINA_MODE",
)


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{' '.join(command[:2])} failed: {detail[-1200:]}")
    return result.stdout


def prepare_receptor(protein: Path, work: Path) -> Path:
    pqr = work / "receptor.pqr"
    pdbqt = work / "receptor.pdbqt"
    pdb2pqr = shutil.which("pdb2pqr30") or shutil.which("pdb2pqr")
    if not pdb2pqr:
        raise RuntimeError("pdb2pqr is not installed")
    run_command([pdb2pqr, "--ff=AMBER", str(protein), str(pqr)])
    prepare = Path(AutoDockTools.__path__[0]) / "Utilities24" / "prepare_receptor4.py"
    run_command([sys.executable, str(prepare), "-r", str(pqr), "-o", str(pdbqt)])
    if not pdbqt.is_file() or pdbqt.stat().st_size == 0:
        raise RuntimeError("Receptor preparation did not produce a PDBQT file")
    return pdbqt


def docking_box(mol: Chem.Mol) -> tuple[list[float], list[float]]:
    if mol.GetNumConformers() == 0:
        raise ValueError("Generated molecule has no 3D conformer")
    positions = mol.GetConformer().GetPositions()
    center = ((positions.max(axis=0) + positions.min(axis=0)) / 2).tolist()
    size = ((positions.max(axis=0) - positions.min(axis=0)) + 5.0).tolist()
    return center, size


def prepare_ligand(mol: Chem.Mol, work: Path) -> Path:
    sdf = work / "ligand.sdf"
    pdbqt = work / "ligand.pdbqt"
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(mol)
    obabel = shutil.which("obabel")
    if not obabel:
        raise RuntimeError("Open Babel is not installed")
    run_command([obabel, "-isdf", str(sdf), "-opdbqt", "-O", str(pdbqt), "-h"])
    if not pdbqt.is_file() or pdbqt.stat().st_size == 0:
        raise RuntimeError("Ligand preparation did not produce a PDBQT file")
    return pdbqt


def vina_score(receptor: Path, ligand: Path, center: list[float], size: list[float],
               metric: str, exhaustiveness: int, cpu: int) -> float:
    engine = Vina(sf_name="vina", cpu=cpu, seed=0, verbosity=0)
    engine.set_receptor(str(receptor))
    engine.set_ligand_from_file(str(ligand))
    engine.compute_vina_maps(center=center, box_size=size)
    if metric == "vina_score":
        return float(engine.score()[0])
    if metric == "vina_minimize":
        return float(engine.optimize()[0])
    engine.dock(exhaustiveness=exhaustiveness, n_poses=1)
    return float(engine.energies(n_poses=1)[0][0])


def qvina_score(receptor: Path, ligand: Path, center: list[float], size: list[float],
                exhaustiveness: int, cpu: int, qvina_bin: str, work: Path) -> float:
    command = [
        qvina_bin, "--receptor", str(receptor), "--ligand", str(ligand),
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--exhaustiveness", str(exhaustiveness), "--cpu", str(cpu),
        "--out", str(work / "qvina_pose.pdbqt"),
    ]
    output = run_command(command)
    match = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", output, re.MULTILINE)
    if not match:
        raise RuntimeError("QVina did not report a ranked affinity")
    return float(match.group(1))


def annotate(mol: Chem.Mol, selected: set[str], receptor: Path | None,
             work_root: Path, exhaustiveness: int, cpu: int, qvina_bin: str) -> bool:
    for prop in RESULT_PROPS:
        if mol.HasProp(prop):
            mol.ClearProp(prop)
    vina = selected & VINA_METRICS
    mode = "none" if not vina else "vina_score" if vina == {"vina_score"} else "vina_dock" if vina == {"vina_dock"} else "qvina" if vina == {"qvina"} else "all" if vina == VINA_METRICS else "custom"
    mol.SetProp("VINA_MODE", mode)
    mol.SetProp("EVALUATION_METRICS", ",".join(metric for metric in METRIC_ORDER if metric in selected))
    if vina:
        mol.SetProp("VINA_EXHAUSTIVENESS", str(exhaustiveness))
        mol.SetProp("VINA_CPU", str(cpu))
    try:
        smiles = Chem.MolToSmiles(mol)
        mol.SetProp("SMILES", smiles)
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("Molecule has separate fragments")
        if vina and any(atom.GetSymbol() not in VINA_ELEMENTS for atom in mol.GetAtoms()):
            raise ValueError("Molecule has atoms unsupported by Vina")
        if selected & CHEMISTRY_METRICS:
            chemistry = scoring_func.get_chem(mol)
            for metric in CHEMISTRY_METRICS & selected:
                value = chemistry.get(metric)
                if value is not None:
                    mol.SetProp(metric.upper(), str(value))
        if vina:
            if receptor is None:
                raise ValueError("Protein PDB is required for docking metrics")
            with tempfile.TemporaryDirectory(prefix="ligand-", dir=work_root) as tmp:
                work = Path(tmp)
                ligand = prepare_ligand(mol, work)
                center, size = docking_box(mol)
                if "vina_score" in selected:
                    mol.SetProp("VINA_SCORE_ONLY", str(vina_score(receptor, ligand, center, size, "vina_score", exhaustiveness, cpu)))
                    mol.SetProp("VINA_MINIMIZE", str(vina_score(receptor, ligand, center, size, "vina_minimize", exhaustiveness, cpu)))
                if "vina_dock" in selected:
                    mol.SetProp("VINA_DOCK", str(vina_score(receptor, ligand, center, size, "vina_dock", exhaustiveness, cpu)))
                if "qvina" in selected:
                    mol.SetProp("QVINA", str(qvina_score(receptor, ligand, center, size, exhaustiveness, cpu, qvina_bin, work)))
        mol.SetProp("VINA_STATUS", "ok" if vina else "not_run")
        return True
    except Exception as error:
        mol.SetProp("VINA_STATUS", "failed")
        mol.SetProp("VINA_ERROR", str(error))
        return False


def annotate_file(sdf: Path, selected: set[str], receptor: Path | None,
                  work: Path, exhaustiveness: int, cpu: int, qvina_bin: str) -> tuple[int, int]:
    molecules = [mol for mol in Chem.SDMolSupplier(str(sdf), removeHs=False) if mol is not None]
    if not molecules:
        return 0, 0
    replacement = sdf.with_suffix(sdf.suffix + ".tmp")
    success = 0
    with Chem.SDWriter(str(replacement)) as writer:
        for mol in molecules:
            success += annotate(mol, selected, receptor, work, exhaustiveness, cpu, qvina_bin)
            writer.write(mol)
    replacement.replace(sdf)
    return len(molecules), success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--protein", type=Path)
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/diffsmol/postprocess"))
    parser.add_argument("--metrics", required=True, help="Comma-separated evaluator IDs")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--qvina-bin", default="qvina2.1")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    selected = {item.strip() for item in args.metrics.split(",") if item.strip()}
    unknown = selected - set(METRIC_ORDER)
    if not selected or unknown:
        parser.error(f"Select known metrics only: {', '.join(sorted(unknown)) or 'none selected'}")
    if selected & VINA_METRICS and (args.protein is None or not args.protein.is_file()):
        parser.error("Selected docking metrics require a protein PDB")
    if "qvina" in selected and not shutil.which(args.qvina_bin):
        parser.error(f"QVina executable not found: {args.qvina_bin}")
    if args.exhaustiveness < 1 or args.cpu < 1:
        parser.error("--exhaustiveness and --cpu must be positive")
    if not args.generated_dir.is_dir():
        parser.error(f"Generated SDF directory not found: {args.generated_dir}")
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    receptor = prepare_receptor(args.protein, args.tmp_dir) if selected & VINA_METRICS else None
    total = success = 0
    for sdf in sorted(args.generated_dir.rglob("*.sdf")):
        if sdf.name.lower() == "reference.sdf":
            continue
        count, ok = annotate_file(sdf, selected, receptor, args.tmp_dir,
                                  args.exhaustiveness, args.cpu, args.qvina_bin)
        total += count
        success += ok
    print(f"Annotated selected evaluations in generated SDFs for {success}/{total} molecules")
    return 0 if total and success else 1


if __name__ == "__main__":
    raise SystemExit(main())
