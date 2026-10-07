#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .conditar-slurm.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .conditar-slurm.env
  set +a
fi

SHARED_DIR="${CONDITAR_SLURM_SHARED_DIR:-}"
ACCOUNT="${CONDITAR_SLURM_ACCOUNT:-}"
CONDITAR_IMAGE="${CONDITAR_DOCKER_IMAGE:-docker.io/osuninglab/conditar-dev:gui-dev-20261005}"
DIFFSMOL_IMAGE="${DIFFSMOL_DOCKER_IMAGE:-docker.io/osuninglab/diffsmol:gui-dev-20261005}"
PODMAN_COMMAND="${PODMAN_BIN:-podman}"
SBATCH_COMMAND="${SBATCH_BIN:-sbatch}"

usage() {
  cat <<'EOF'
Prepare the conDitar GUI for Slurm GPU jobs.

Usage: ./setup_slurm_gui.sh --shared-dir PATH --account ACCOUNT [options]

The shared directory must be visible from login and compute nodes. This script
pulls conDitar and DiffSMol with Podman, saves both image archives there, and
records the images, archives, job directory, and account in .conditar-slurm.env.

Options:
  --shared-dir PATH       Shared storage root for image archives and jobs.
  --account ACCOUNT       Slurm account for GPU jobs.
  --conditar-image IMAGE  Override the published conDitar image.
  --diffsmol-image IMAGE  Override the published DiffSMol image.
  --help                  Show this help.

Run ./start_slurm_gui.sh after setup. Rerunning setup refreshes both archives.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shared-dir|--account|--conditar-image|--diffsmol-image)
      [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value."
      case "$1" in
        --shared-dir) SHARED_DIR="$2" ;;
        --account) ACCOUNT="$2" ;;
        --conditar-image) CONDITAR_IMAGE="$2" ;;
        --diffsmol-image) DIFFSMOL_IMAGE="$2" ;;
      esac
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown option: $1. Run --help for usage." ;;
  esac
done

command -v "$PODMAN_COMMAND" >/dev/null 2>&1 || die "Podman not found. Load it or set PODMAN_BIN."
command -v "$SBATCH_COMMAND" >/dev/null 2>&1 || die "sbatch not found. Load Slurm or set SBATCH_BIN."
if ! command -v python3 >/dev/null 2>&1 \
  && ! { command -v conda >/dev/null 2>&1 && conda run -n conditar-gui-dev python -c 'import sys' >/dev/null 2>&1; }; then
  die "Python 3 is needed for the GUI. Load Python or set up the conditar-gui-dev Conda environment."
fi
[[ "$(uname -m)" == "x86_64" ]] || die "The published engine images require an x86_64 host."

if [[ -z "$SHARED_DIR" && -t 0 ]]; then
  read -r -p "Shared directory visible from login and compute nodes: " SHARED_DIR
fi
[[ -n "$SHARED_DIR" ]] || die "Pass --shared-dir with a compute-node-visible absolute path."
[[ "$SHARED_DIR" == /* ]] || die "Shared directory must be an absolute path: $SHARED_DIR"
case "$SHARED_DIR" in
  /tmp|/tmp/*|/var/tmp|/var/tmp/*|/run|/run/*)
    die "$SHARED_DIR is node-local temporary storage. Choose a shared filesystem." ;;
esac
mkdir -p "$SHARED_DIR/images" "$SHARED_DIR/jobs"
SHARED_DIR="$(cd "$SHARED_DIR" && pwd -P)"
case "$SHARED_DIR" in
  /tmp|/tmp/*|/var/tmp|/var/tmp/*|/run|/run/*)
    die "$SHARED_DIR is node-local temporary storage. Choose a shared filesystem." ;;
esac
[[ -w "$SHARED_DIR/images" && -w "$SHARED_DIR/jobs" ]] || die "Shared directory is not writable: $SHARED_DIR"

if [[ -z "$ACCOUNT" && -t 0 ]]; then
  read -r -p "Slurm GPU account: " ACCOUNT
fi
[[ -n "$ACCOUNT" ]] || die "Pass --account with your Slurm GPU account."

CONDITAR_ARCHIVE="$SHARED_DIR/images/conditar.tar"
DIFFSMOL_ARCHIVE="$SHARED_DIR/images/diffsmol.tar"
CONDITAR_TMP="$CONDITAR_ARCHIVE.partial.$$"
DIFFSMOL_TMP="$DIFFSMOL_ARCHIVE.partial.$$"
trap 'rm -f "$CONDITAR_TMP" "$DIFFSMOL_TMP"' EXIT

prepare_image() {
  local image="$1"
  local archive="$2"
  local tmp="$3"
  echo "Pulling $image"
  "$PODMAN_COMMAND" pull "$image"
  echo "Saving $archive"
  "$PODMAN_COMMAND" save --format docker-archive -o "$tmp" "$image"
  [[ -s "$tmp" ]] || die "Podman did not create an image archive: $tmp"
  mv -f "$tmp" "$archive"
}

prepare_image "$CONDITAR_IMAGE" "$CONDITAR_ARCHIVE" "$CONDITAR_TMP"
prepare_image "$DIFFSMOL_IMAGE" "$DIFFSMOL_ARCHIVE" "$DIFFSMOL_TMP"

save_env_value() {
  local key="$1"
  local value="$2"
  local tmp
  tmp="$(mktemp .conditar-slurm.env.XXXXXX)"
  if [[ -f .conditar-slurm.env ]]; then
    grep -v -E "^[[:space:]]*${key}=" .conditar-slurm.env > "$tmp" || true
  fi
  printf '%s=%q\n' "$key" "$value" >> "$tmp"
  mv "$tmp" .conditar-slurm.env
}

save_env_value CONDITAR_SLURM_SHARED_DIR "$SHARED_DIR"
save_env_value CONDITAR_SLURM_ACCOUNT "$ACCOUNT"
save_env_value CONDITAR_JOB_ROOT "$SHARED_DIR/jobs"
save_env_value CONDITAR_DOCKER_IMAGE "$CONDITAR_IMAGE"
save_env_value CONDITAR_DOCKER_TAR "$CONDITAR_ARCHIVE"
save_env_value DIFFSMOL_DOCKER_IMAGE "$DIFFSMOL_IMAGE"
save_env_value DIFFSMOL_DOCKER_TAR "$DIFFSMOL_ARCHIVE"

echo
echo "Slurm GUI setup saved to .conditar-slurm.env"
echo "Images: $CONDITAR_IMAGE and $DIFFSMOL_IMAGE"
echo "Archives: $CONDITAR_ARCHIVE and $DIFFSMOL_ARCHIVE"
echo "Jobs: $SHARED_DIR/jobs"
echo "Confirm this directory is visible from a compute node, then run:"
echo "  ./start_slurm_gui.sh"
