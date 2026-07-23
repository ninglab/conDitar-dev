#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .conditar-slurm.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .conditar-slurm.env
  set +a
fi

export CONDITAR_RUNTIME="${CONDITAR_RUNTIME:-podman}"
export CONDITAR_DOCKER_IMAGE="${CONDITAR_DOCKER_IMAGE:-localhost/conditar-dev:container-dev}"
export CONDITAR_DOCKER_TAR="${CONDITAR_DOCKER_TAR:-}"
if [[ -z "${CONDITAR_SOURCE_MOUNT:-}" && -d ../conDitar-dev ]]; then
  export CONDITAR_SOURCE_MOUNT="$(cd ../conDitar-dev && pwd)"
elif [[ -z "${CONDITAR_SOURCE_MOUNT:-}" && -d ../docker && -d ../scripts ]]; then
  export CONDITAR_SOURCE_MOUNT="$(cd .. && pwd)"
fi
export CONDITAR_SLURM_ACCOUNT="${CONDITAR_SLURM_ACCOUNT:-}"
export CONDITAR_SLURM_TIME="${CONDITAR_SLURM_TIME:-04:00:00}"
export CONDITAR_SLURM_MEM="${CONDITAR_SLURM_MEM:-32G}"
export CONDITAR_SLURM_CPUS="${CONDITAR_SLURM_CPUS:-4}"
export CONDITAR_SLURM_GPUS="${CONDITAR_SLURM_GPUS:-1}"

if [[ -n "$CONDITAR_DOCKER_TAR" && ! -f "$CONDITAR_DOCKER_TAR" ]]; then
  echo "ERROR: GPU container archive not found: $CONDITAR_DOCKER_TAR" >&2
  echo "Set CONDITAR_DOCKER_TAR to a readable .tar/.tar.gz archive, or leave it empty when the image is already available." >&2
  exit 2
fi
for required in python3 podman sbatch; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "ERROR: required Slurm GPU command not found: $required" >&2
    echo "Load the appropriate Python, Podman, and Slurm modules, then retry." >&2
    exit 2
  fi
done

PYTHON_COMMAND=(python3)
if [[ -n "${CONDITAR_GUI_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("$CONDITAR_GUI_PYTHON")
elif command -v conda >/dev/null 2>&1 && conda run -n conditar-gui-dev python -c "import sys" >/dev/null 2>&1; then
  PYTHON_COMMAND=(conda run --no-capture-output -n conditar-gui-dev python)
fi

echo "Starting conDitar GUI"
echo "Container image: $CONDITAR_DOCKER_IMAGE"
echo "Container archive: ${CONDITAR_DOCKER_TAR:-none}"
echo "Source mount: ${CONDITAR_SOURCE_MOUNT:-none}"
echo "Runtime: $CONDITAR_RUNTIME"
echo "GUI Python: ${PYTHON_COMMAND[*]}"
echo "Slurm defaults: account=${CONDITAR_SLURM_ACCOUNT:-none} time=$CONDITAR_SLURM_TIME mem=$CONDITAR_SLURM_MEM cpus=$CONDITAR_SLURM_CPUS gpus=$CONDITAR_SLURM_GPUS"
echo "GPU mode: select Slurm GPU in the Setup panel"
echo

"${PYTHON_COMMAND[@]}" serve.py --host 127.0.0.1 --port "${PORT:-4173}" --open
