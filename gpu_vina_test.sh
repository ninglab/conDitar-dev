#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

INPUT_DIR="${INPUT_DIR:-/fs/ess/PCON0041/gruoxi/conDitar-dev/examples}"
OUT_DIR="${OUT_DIR:-$PWD/results_docker_gpu_vina}"
IMAGE="${CONDITAR_DOCKER_IMAGE:-localhost/conditar-dev:container-dev}"

mkdir -p "$OUT_DIR"

echo "Using image: $IMAGE"
echo "Using inputs: $INPUT_DIR"
echo "Writing results to: $OUT_DIR"

podman run --rm --device nvidia.com/gpu=all \
  -e CONDITAR_DEVICE=cuda:0 \
  -v "$INPUT_DIR":/inputs:ro \
  -v "$OUT_DIR":/results \
  "$IMAGE" \
  --pdb /inputs/4aua/4aua_protein.pdb \
  --sdf /inputs/4aua/4aua_ligand.sdf \
  --out /results \
  --device cuda:0 \
  --num-samples 1 \
  --batch-size 1 \
  --vina-score \
  --vina-mode vina_score \
  --vina-exhaustiveness 8 \
  --vina-cpu 4

echo
echo "Generated SDF files:"
find "$OUT_DIR" -maxdepth 1 -name "*.sdf" -print

echo
echo "Vina SDF tags:"
grep -n "VINA_SCORE_ONLY\|VINA_STATUS\|QED\|SA" "$OUT_DIR"/*.sdf || true
