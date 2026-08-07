#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${CONDITAR_DOCKER_TAG:-docker.io/osuninglab/conditar-dev:2026-07-10}"
input_dir="${INPUT_DIR:-$repo_root/data/test_data}"
output_dir="${OUTPUT_DIR:-$PWD/results}"
protein_pdb="${PROTEIN_PDB:-4aua/4aua_protein.pdb}"
ligand_sdf="${LIGAND_SDF:-4aua/4aua_ligand.sdf}"
pocket_pdb="${POCKET_PDB:-}"
num_samples="${NUM_SAMPLES:-1}"
batch_size="${BATCH_SIZE:-1}"
vina_mode="${VINA_MODE:-vina_score}"
vina_exhaustiveness="${VINA_EXHAUSTIVENESS:-8}"
vina_cpu="${VINA_CPU:-4}"

usage() {
    cat <<EOF
Usage:
  docker/run-examples.sh COMMAND

Commands:
  cpu-pocket      Docker CPU run with a prepared pocket PDB.
  cpu-ligand      Docker CPU run with a protein PDB and reference ligand SDF.
  gpu             Docker NVIDIA GPU run with a prepared pocket PDB.
  vina            Docker CPU run with Vina/QVina post-processing enabled.
  podman-cpu      Podman CPU run with a prepared pocket PDB.
  podman-gpu      Podman GPU run with a prepared pocket PDB.
  dev             Docker CPU run with the live checkout bind-mounted read-only.

Environment:
  CONDITAR_DOCKER_TAG   Image tag. Default: $image_tag
  INPUT_DIR             Host folder mounted at /inputs. Default: $input_dir
  OUTPUT_DIR            Host results folder. Default: ./results
  POCKET_PDB            Path under INPUT_DIR. Required for pocket-only commands.
  PROTEIN_PDB           Path under INPUT_DIR. Default: $protein_pdb
  LIGAND_SDF            Path under INPUT_DIR. Default: $ligand_sdf
  NUM_SAMPLES           Number of molecules. Default: $num_samples
  BATCH_SIZE            Batch size. Default: $batch_size
  VINA_MODE             none, vina_score, vina_dock, qvina, or all. Default: $vina_mode
  VINA_EXHAUSTIVENESS   Vina exhaustiveness. Default: $vina_exhaustiveness
  VINA_CPU              CPU threads for Vina. Default: $vina_cpu
EOF
}

require_input_dir() {
    if [[ -z "$input_dir" ]]; then
        echo "Set INPUT_DIR to the host folder containing your input PDB/SDF files." >&2
        exit 2
    fi
}

require_pocket_pdb() {
    if [[ -z "$pocket_pdb" ]]; then
        echo "Set POCKET_PDB to a prepared pocket PDB path under INPUT_DIR for pocket-only runs." >&2
        echo "For the included 4aua protein/ligand example, use: docker/run-examples.sh cpu-ligand" >&2
        exit 2
    fi
}

prepare_output_dir() {
    mkdir -p "$output_dir"
}

run_docker_cpu_pocket() {
    require_input_dir
    require_pocket_pdb
    prepare_output_dir
    docker run --rm \
        -e CONDITAR_DEVICE=cpu \
        -v "$input_dir":/inputs:ro \
        -v "$output_dir":/results \
        "$image_tag" \
        --pdb "/inputs/$pocket_pdb" \
        --out /results \
        --device cpu \
        --num-samples "$num_samples" \
        --batch-size "$batch_size"
}

run_docker_cpu_ligand() {
    require_input_dir
    prepare_output_dir
    docker run --rm \
        -e CONDITAR_DEVICE=cpu \
        -v "$input_dir":/inputs:ro \
        -v "$output_dir":/results \
        "$image_tag" \
        --pdb "/inputs/$protein_pdb" \
        --sdf "/inputs/$ligand_sdf" \
        --out /results \
        --device cpu \
        --num-samples "$num_samples" \
        --batch-size "$batch_size"
}

run_docker_gpu() {
    require_input_dir
    require_pocket_pdb
    prepare_output_dir
    docker run --rm --gpus all \
        -e CONDITAR_DEVICE=cuda:0 \
        -v "$input_dir":/inputs:ro \
        -v "$output_dir":/results \
        "$image_tag" \
        --pdb "/inputs/$pocket_pdb" \
        --out /results \
        --device cuda:0 \
        --num-samples "$num_samples" \
        --batch-size "$batch_size"
}

run_docker_vina() {
    require_input_dir
    prepare_output_dir
    docker run --rm \
        -e CONDITAR_DEVICE=cpu \
        -v "$input_dir":/inputs:ro \
        -v "$output_dir":/results \
        "$image_tag" \
        --pdb "/inputs/$protein_pdb" \
        --sdf "/inputs/$ligand_sdf" \
        --out /results \
        --device cpu \
        --num-samples "$num_samples" \
        --batch-size "$batch_size" \
        --vina-score \
        --vina-mode "$vina_mode" \
        --vina-exhaustiveness "$vina_exhaustiveness" \
        --vina-cpu "$vina_cpu"
}

run_podman_cpu() {
    require_input_dir
    require_pocket_pdb
    prepare_output_dir
    podman run --rm \
        -e CONDITAR_DEVICE=cpu \
        -v "$input_dir":/inputs:ro \
        -v "$output_dir":/results \
        "$image_tag" \
        --pdb "/inputs/$pocket_pdb" \
        --out /results \
        --device cpu \
        --num-samples "$num_samples" \
        --batch-size "$batch_size"
}

run_podman_gpu() {
    require_input_dir
    require_pocket_pdb
    prepare_output_dir
    podman run --rm --device nvidia.com/gpu=all \
        -e CONDITAR_DEVICE=cuda:0 \
        -v "$input_dir":/inputs:ro \
        -v "$output_dir":/results \
        "$image_tag" \
        --pdb "/inputs/$pocket_pdb" \
        --out /results \
        --device cuda:0 \
        --num-samples "$num_samples" \
        --batch-size "$batch_size"
}

run_dev() {
    require_input_dir
    require_pocket_pdb
    prepare_output_dir
    docker run --rm \
        -e CONDITAR_DEVICE=cpu \
        -v "$PWD":/opt/conditar/app:ro \
        -v "$input_dir":/inputs:ro \
        -v "$output_dir":/results \
        "$image_tag" \
        --pdb "/inputs/$pocket_pdb" \
        --out /results \
        --device cpu \
        --num-samples "$num_samples" \
        --batch-size "$batch_size"
}

case "${1:-}" in
    cpu-pocket)
        run_docker_cpu_pocket
        ;;
    cpu-ligand)
        run_docker_cpu_ligand
        ;;
    gpu)
        run_docker_gpu
        ;;
    vina)
        run_docker_vina
        ;;
    podman-cpu)
        run_podman_cpu
        ;;
    podman-gpu)
        run_podman_gpu
        ;;
    dev)
        run_dev
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        echo "Unknown command: $1" >&2
        usage >&2
        exit 2
        ;;
esac
