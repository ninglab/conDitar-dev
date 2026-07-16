#!/usr/bin/env bash
set -euo pipefail

image_tag="${CONDITAR_DOCKER_TAG:-localhost/conditar-dev:container-dev}"
input_dir="${INPUT_DIR:-}"
output_dir="${OUTPUT_DIR:-$PWD/results}"
pocket_pdb="${POCKET_PDB:-xxxx/xxxx_pocket.pdb}"
protein_pdb="${PROTEIN_PDB:-4aua/4aua_protein.pdb}"
ligand_sdf="${LIGAND_SDF:-4aua/4aua_ligand.sdf}"
num_samples="${NUM_SAMPLES:-1}"
batch_size="${BATCH_SIZE:-1}"
vina_mode="${VINA_MODE:-vina_score}"
vina_exhaustiveness="${VINA_EXHAUSTIVENESS:-8}"
vina_cpu="${VINA_CPU:-4}"

usage() {
    cat <<EOF
Usage:
  INPUT_DIR=/path/to/input-data docker/run-examples.sh COMMAND

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
  INPUT_DIR             Host folder mounted at /inputs. Required.
  OUTPUT_DIR            Host results folder. Default: ./results
  POCKET_PDB            Path under INPUT_DIR. Default: $pocket_pdb
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

prepare_output_dir() {
    mkdir -p "$output_dir"
}

run_docker_cpu_pocket() {
    require_input_dir
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
