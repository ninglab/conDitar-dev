"""Run standalone DiffSMol pocket-conditioned generation from a PDB/SDF pair.

The upstream sampler expects a one-item CrossDocked-style dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import yaml
from rdkit import Chem


ROOT = Path("/opt/DiffSMol")
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE))

from utils import reconstruct, transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="3D SDF reference ligand")
    parser.add_argument("--protein", type=Path, required=True, help="Protein or pocket PDB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--guidance", action="store_true", help="Enable shape guidance with pocket guidance")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.input.is_file() or args.input.suffix.lower() != ".sdf":
        raise SystemExit("--input must be a readable 3D .sdf file")
    if not args.protein.is_file() or args.protein.suffix.lower() != ".pdb":
        raise SystemExit("--protein must be a readable .pdb file")
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        raise SystemExit("--device must be cpu or cuda:N") from error
    if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
        raise SystemExit("--device must be cpu or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; enable GPU access or select --device cpu")
    args.device = str(device)


def prepare_raw_pair(work: Path, sdf_path: Path, protein_path: Path) -> Path:
    raw = work / "crossdocked_test_set"
    pair_dir = raw / "input_pair"
    pair_dir.mkdir(parents=True)
    ligand_name = "reference.sdf"
    protein_name = "protein.pdb"
    (pair_dir / ligand_name).write_text(sdf_path.read_text(errors="replace"))
    (pair_dir / protein_name).write_text(protein_path.read_text(errors="replace"))
    with (raw / "index.pkl").open("wb") as handle:
        pickle.dump([(f"input_pair/{protein_name}", f"input_pair/{ligand_name}")], handle)
    return raw


def build_config(work: Path, raw: Path, args: argparse.Namespace) -> Path:
    suffix = "with_shape_guidance" if args.guidance else "no_shape_guidance"
    template = ROOT / (
        "config/sampling/PMG/"
        f"sample_diff_pos0_10_pos1.e-7_0.01_6_v001_scalar128_vec32_layer8_with_pocket_guidance_{suffix}.yml"
    )
    config = yaml.safe_load(template.read_text())
    processed = work / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    config["data"].update(
        path=str(raw),
        processed_path=str(processed),
        datasize=1,
        chunk_size=1,
        num_workers=1,
        version="input_pair",
    )
    config["data"]["shape"].update(
        batch_size=1,
        num_workers=1,
        shape_parallel=False,
        device=args.device,
        checkpoint=str(ROOT / "models/se.pt"),
    )
    config["model"]["checkpoint"] = str(ROOT / "models/diffusion.pt")
    config["sample"]["num_samples"] = args.num_samples
    config["sample"].setdefault("pocket_threshold", 0.5)
    assert config["sample"]["use_pocket"] is True
    assert config["sample"]["num_steps"] == 1000
    config_path = work / "sampling_pocket.yml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def export_sdf(result_path: Path, output: Path, args: argparse.Namespace, started: float) -> None:
    result_file = result_path / "result_0.pt"
    result = torch.load(result_file, map_location="cpu")
    samples = []
    for index, (positions, types) in enumerate(zip(result["pred_ligand_pos"], result["pred_ligand_v"])):
        entry = {"index": index}
        try:
            if not np.isfinite(positions).all():
                raise ValueError("Non-finite generated coordinates")
            atoms = transforms.get_atomic_number_from_index(types, mode="add_aromatic")
            aromatic = transforms.is_aromatic_from_index(types, mode="add_aromatic")
            entry["atoms"] = len(atoms)
            mol = reconstruct.reconstruct_from_generated(positions, atoms, aromatic)
            Chem.SanitizeMol(mol)
            entry.update(smiles=Chem.MolToSmiles(mol), connected=len(Chem.GetMolFrags(mol)) == 1)
            mol.SetProp("_Name", f"pocket_sample_{index}")
            with Chem.SDWriter(str(output / f"generated_{index:04d}.sdf")) as writer:
                writer.write(mol)
        except (reconstruct.MolReconsError, ValueError, RuntimeError) as error:
            entry["error"] = str(error)
        samples.append(entry)
    summary = {
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "conditioning": "pocket_with_shape" if args.guidance else "pocket_without_shape",
        "guidance": args.guidance,
        "diffusion_steps": 1000,
        "wall_seconds": time.perf_counter() - started,
        "sampling_seconds": result.get("time"),
        "reference_source": {
            "kind": "sdf+pdb",
            "sdf": str(args.input),
            "sdf_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "protein": str(args.protein),
            "protein_sha256": hashlib.sha256(args.protein.read_bytes()).hexdigest(),
        },
        "samples": samples,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not any(sample.get("connected") for sample in samples):
        raise RuntimeError("No connected, sanitized molecule reconstructed; see summary.json")


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pocket-preprocessing-", dir=args.output) as tmp:
        work = Path(tmp)
        raw = prepare_raw_pair(work, args.input, args.protein)
        config_path = build_config(work, raw, args)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.sample_diffusion_with_pocket",
                str(config_path),
                "--device",
                args.device,
                "--data_id",
                "0",
                "--batch_size",
                "1",
                "--result_path",
                str(args.output),
                "--protein_ligand_dist",
                str(ROOT / "data/crossdocked/protein_ligand_dist.txt"),
            ],
            cwd=SOURCE,
            check=True,
        )
    export_sdf(args.output, args.output, args, started)


if __name__ == "__main__":
    main()
