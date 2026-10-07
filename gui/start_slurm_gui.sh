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
DEFAULT_IMAGE="docker.io/osuninglab/conditar-dev:gui-dev-20261005"
if [[ -z "${CONDITAR_DOCKER_IMAGE:-}" ]]; then
  export CONDITAR_DOCKER_IMAGE="$DEFAULT_IMAGE"
fi
export DIFFSMOL_DOCKER_IMAGE="${DIFFSMOL_DOCKER_IMAGE:-docker.io/osuninglab/diffsmol:gui-dev-20261005}"
export DIFFSMOL_DOCKER_TAR="${DIFFSMOL_DOCKER_TAR:-}"
export CONDITAR_DOCKER_TAR="${CONDITAR_DOCKER_TAR:-}"
configured_tar="$CONDITAR_DOCKER_TAR"

# A stale path can remain in .conditar-slurm.env after moving to a new VM.
# Clear it temporarily so the shared/default archive search can recover.
if [[ -n "$CONDITAR_DOCKER_TAR" && ! -f "$CONDITAR_DOCKER_TAR" ]]; then
  echo "WARNING: configured GPU container archive is not readable: $CONDITAR_DOCKER_TAR" >&2
  export CONDITAR_DOCKER_TAR=""
fi

# Prefer a nearby exported image archive when one is available. This prevents a
# later Slurm task from trying to pull a localhost image from a registry.
if [[ -z "$CONDITAR_DOCKER_TAR" ]]; then
  for candidate in \
    "$PWD"/conditar*.tar "$PWD"/conditar*.tar.gz \
    "$PWD"/localhost_conditar-dev*.tar "$PWD"/localhost_conditar-dev*.tar.gz \
    "$PWD"/../containers/conditar*.tar "$PWD"/../containers/conditar*.tar.gz \
    "$PWD"/../containers/localhost_conditar-dev*.tar "$PWD"/../containers/localhost_conditar-dev*.tar.gz \
    "$HOME"/containers/conditar*.tar "$HOME"/containers/conditar*.tar.gz \
    "$HOME"/containers/localhost_conditar-dev*.tar "$HOME"/containers/localhost_conditar-dev*.tar.gz; do
    if [[ -f "$candidate" ]]; then
      export CONDITAR_DOCKER_TAR="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDITAR_DOCKER_TAR" && -n "$configured_tar" ]]; then
  export CONDITAR_DOCKER_TAR="$configured_tar"
fi

for path in "${CONDITAR_JOB_ROOT:-}" "$CONDITAR_DOCKER_TAR" "$DIFFSMOL_DOCKER_TAR"; do
  case "$path" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/run|/run/*)
      echo "ERROR: Slurm jobs cannot use node-local storage: $path" >&2
      echo "Run ./setup_slurm_gui.sh --shared-dir /path/to/shared/storage --account YOUR_ACCOUNT" >&2
      exit 2 ;;
  esac
done

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
if [[ -n "$DIFFSMOL_DOCKER_TAR" && ! -f "$DIFFSMOL_DOCKER_TAR" ]]; then
  echo "ERROR: DiffSMol GPU container archive not found: $DIFFSMOL_DOCKER_TAR" >&2
  echo "Run ./setup_slurm_gui.sh to prepare both images on shared storage." >&2
  exit 2
fi

PODMAN_COMMAND="${PODMAN_BIN:-podman}"
SBATCH_COMMAND="${SBATCH_BIN:-sbatch}"
if [[ -z "$CONDITAR_DOCKER_TAR" ]] && command -v "$PODMAN_COMMAND" >/dev/null 2>&1 \
  && ! "$PODMAN_COMMAND" image exists "$CONDITAR_DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: Slurm GPU image is unavailable: $CONDITAR_DOCKER_IMAGE" >&2
  echo "Pull it with podman, set CONDITAR_DOCKER_TAR to a readable archive, or load the image with podman load." >&2
  echo "Set CONDITAR_DOCKER_IMAGE to a pullable refreshed image or load its archive with podman." >&2
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

for required in "$PODMAN_COMMAND" "$SBATCH_COMMAND"; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "ERROR: required Slurm GPU command not found: $required" >&2
    echo "Load the appropriate Podman and Slurm modules, then retry." >&2
    exit 2
  fi
done

echo "Starting conDitar GUI"
echo "Container image: $CONDITAR_DOCKER_IMAGE"
echo "Container archive: ${CONDITAR_DOCKER_TAR:-none}"
echo "DiffSMol image: $DIFFSMOL_DOCKER_IMAGE"
echo "DiffSMol archive: ${DIFFSMOL_DOCKER_TAR:-none}"
echo "Job storage: ${CONDITAR_JOB_ROOT:-$PWD/job_data/jobs}"
echo "Source mount: ${CONDITAR_SOURCE_MOUNT:-none}"
echo "Runtime: $CONDITAR_RUNTIME"
echo "GUI Python: ${PYTHON_COMMAND[*]}"
echo "Slurm defaults: account=${CONDITAR_SLURM_ACCOUNT:-none} time=$CONDITAR_SLURM_TIME mem=$CONDITAR_SLURM_MEM cpus=$CONDITAR_SLURM_CPUS gpus=$CONDITAR_SLURM_GPUS"
echo "GPU mode: select Slurm GPU in the Setup panel"
echo

"${PYTHON_COMMAND[@]}" serve.py --host 127.0.0.1 --port "${PORT:-4173}" --auto-port --open
