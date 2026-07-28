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

# Prefer a nearby exported image archive when one is available. This prevents a
# later Slurm task from trying to pull a localhost image from a registry.
if [[ -z "$CONDITAR_DOCKER_TAR" ]]; then
  shopt -s nullglob
  archive_candidates=(
    "$PWD"/conditar*.tar
    "$PWD"/conditar*.tar.gz
    "$PWD"/localhost_conditar-dev*.tar
    "$PWD"/localhost_conditar-dev*.tar.gz
    "$PWD"/../containers/conditar*.tar
    "$PWD"/../containers/conditar*.tar.gz
    "$PWD"/../containers/localhost_conditar-dev*.tar
    "$PWD"/../containers/localhost_conditar-dev*.tar.gz
    "$HOME"/containers/conditar*.tar
    "$HOME"/containers/conditar*.tar.gz
    "$HOME"/containers/localhost_conditar-dev*.tar
    "$HOME"/containers/localhost_conditar-dev*.tar.gz
  )
  shopt -u nullglob
  for candidate in "${archive_candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      export CONDITAR_DOCKER_TAR="$candidate"
      break
    fi
  done
fi
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

if [[ -z "$CONDITAR_DOCKER_TAR" ]] && command -v podman >/dev/null 2>&1 \
  && ! podman image exists "$CONDITAR_DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: Slurm GPU image is unavailable: $CONDITAR_DOCKER_IMAGE" >&2
  echo "Set CONDITAR_DOCKER_TAR to a readable archive or load the image with podman load." >&2
  echo "Example: podman load -i /shared/path/localhost_conditar-dev_container-dev.tar.gz" >&2
  exit 2
fi
PYTHON_COMMAND=(python3)
if [[ -n "${CONDITAR_GUI_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("$CONDITAR_GUI_PYTHON")
elif command -v conda >/dev/null 2>&1 && conda run -n conditar-gui-dev python -c "import sys" >/dev/null 2>&1; then
  PYTHON_COMMAND=(conda run --no-capture-output -n conditar-gui-dev python)
fi

if ! "${PYTHON_COMMAND[@]}" -c "import sys" >/dev/null 2>&1; then
  echo "ERROR: Python was not found." >&2
  echo "Load Python or install Miniconda/Mambaforge, then retry." >&2
  exit 2
fi

for required in podman sbatch; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "ERROR: required Slurm GPU command not found: $required" >&2
    echo "Load the appropriate Podman and Slurm modules, then retry." >&2
    exit 2
  fi
done

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
