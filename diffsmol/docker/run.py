"""Run standalone DiffSMol generation and selected evaluations in one container."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


DOCKER_DIR = Path(__file__).resolve().parent
STATUS_FILE = "evaluation_status.json"
METRICS = {"vina_score", "vina_dock", "qvina", "qed", "sa", "logp", "lipinski"}
DOCKING_METRICS = {"vina_score", "vina_dock", "qvina"}


def write_status(output: Path, status: str, metrics: list[str], exit_code: int | None = None) -> None:
    payload = {"status": status, "metrics": metrics, "exit_code": exit_code}
    path = output / STATUS_FILE
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="3D reference ligand SDF")
    parser.add_argument("--protein", type=Path, help="Protein/pocket PDB for pocket generation and docking")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--postprocess-metrics", default="", help="Comma-separated evaluator IDs")
    parser.add_argument("--vina-exhaustiveness", type=int, default=8)
    parser.add_argument("--vina-cpu", type=int, default=4)
    parser.add_argument("--qvina-bin", default="qvina2.1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = [item.strip() for item in args.postprocess_metrics.split(",") if item.strip()]
    unknown = set(metrics) - METRICS
    if not args.input.is_file() or args.input.suffix.lower() != ".sdf":
        raise SystemExit("--input must be a readable .sdf file")
    if args.protein and (not args.protein.is_file() or args.protein.suffix.lower() != ".pdb"):
        raise SystemExit("--protein must be a readable .pdb file")
    if unknown:
        raise SystemExit(f"Unknown evaluation metrics: {', '.join(sorted(unknown))}")
    if DOCKING_METRICS.intersection(metrics) and not args.protein:
        raise SystemExit("Docking evaluations require --protein")
    if args.num_samples < 1 or args.vina_exhaustiveness < 1 or args.vina_cpu < 1:
        raise SystemExit("Sample count, Vina exhaustiveness, and Vina CPU count must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    generator = DOCKER_DIR / ("pocket_generate.py" if args.protein else "generate.py")
    command = [
        sys.executable, str(generator), "--input", str(args.input),
        "--output", str(args.output), "--device", args.device,
        "--num-samples", str(args.num_samples),
    ]
    if args.protein:
        command.extend(["--protein", str(args.protein)])
    if args.guidance:
        command.append("--guidance")
    if metrics:
        write_status(args.output, "pending", metrics)
    result = subprocess.run(command, check=False)
    if result.returncode:
        if metrics:
            write_status(args.output, "generation_failed", metrics, result.returncode)
        return result.returncode
    if not metrics:
        write_status(args.output, "skipped", [])
        return 0

    write_status(args.output, "running", metrics)
    command = [
        sys.executable, str(DOCKER_DIR / "postprocess.py"),
        "--generated-dir", str(args.output),
        "--metrics", ",".join(metrics),
        "--exhaustiveness", str(args.vina_exhaustiveness),
        "--cpu", str(args.vina_cpu),
        "--qvina-bin", args.qvina_bin,
    ]
    if args.protein:
        command.extend(["--protein", str(args.protein)])
    result = subprocess.run(command, check=False)
    write_status(args.output, "completed" if result.returncode == 0 else "failed",
                 metrics, result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
