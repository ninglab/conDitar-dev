#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .conditar-slurm.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .conditar-slurm.env
  set +a
fi

IMAGE="${CONDITAR_DOCKER_IMAGE:-localhost/conditar-dev:container-dev}"
ARCHIVE="${CONDITAR_DOCKER_TAR:-}"
PODMAN_COMMAND="${PODMAN_BIN:-podman}"
SBATCH_COMMAND="${SBATCH_BIN:-sbatch}"

# Match the launcher convenience path when the OSC shared filesystem is mounted.
if [[ -z "$ARCHIVE" ]]; then
  for candidate in \
    "/fs/ess/PCON0041/mey200/container_images/localhost_conditar-dev_container-dev-20260710-105038.tar.gz" \
    "$HOME/containers/localhost_conditar-dev_container-dev-20260710-105038.tar.gz"; do
    if [[ -f "$candidate" ]]; then
      ARCHIVE="$candidate"
      break
    fi
  done
fi

echo "conDitar Slurm/GPU setup check"
echo
missing=0

check_command() {
  local name="$1"
  local hint="$2"
  if command -v "$name" >/dev/null 2>&1; then
    echo "OK    $name found: $(command -v "$name")"
  else
    echo "MISS  $name not found"
    echo "      $hint"
    missing=1
  fi
}

check_command python3 "Load Python 3.9 or newer."
check_command "$PODMAN_COMMAND" "Load Podman or set PODMAN_BIN=/path/to/podman."
check_command "$SBATCH_COMMAND" "Load Slurm or set SBATCH_BIN=/path/to/sbatch."

save_env_value() {
  local key="$1"
  local value="$2"
  if [[ -f .conditar-slurm.env ]]; then
    sed -i "/^[[:space:]]*${key}=/d" .conditar-slurm.env
  fi
  printf '%s=%q\n' "$key" "$value" >> .conditar-slurm.env
}

if [[ -n "$ARCHIVE" && ! -r "$ARCHIVE" && -t 0 ]]; then
  echo "Configured archive is not readable: $ARCHIVE"
  read -r -p "Enter a compute-node-visible archive path (or press Enter to keep it): " entered_archive
  [[ -n "$entered_archive" ]] && ARCHIVE="$entered_archive"
fi

if [[ -z "$ARCHIVE" ]] && command -v "$PODMAN_COMMAND" >/dev/null 2>&1 \
  && ! "$PODMAN_COMMAND" image exists "$IMAGE" >/dev/null 2>&1 && -t 0; then
  read -r -p "Path to a shared Docker/OCI archive (or press Enter if preloaded on compute nodes): " entered_archive
  if [[ -n "$entered_archive" ]]; then
    ARCHIVE="$entered_archive"
    save_env_value CONDITAR_DOCKER_TAR "$ARCHIVE"
    echo "Saved container archive path to .conditar-slurm.env"
  fi
fi

if [[ -n "$ARCHIVE" ]]; then
  if [[ -r "$ARCHIVE" ]]; then
    echo "OK    container archive readable: $ARCHIVE"
  else
    echo "MISS  container archive is not readable: $ARCHIVE"
    echo "      Set CONDITAR_DOCKER_TAR to a compute-node-visible .tar/.tar.gz archive."
    missing=1
  fi
elif command -v "$PODMAN_COMMAND" >/dev/null 2>&1 && "$PODMAN_COMMAND" image exists "$IMAGE" >/dev/null 2>&1; then
  echo "OK    preloaded container image found: $IMAGE"
else
  echo "MISS  no container archive or preloaded image found: $IMAGE"
  echo "      Set CONDITAR_DOCKER_TAR to a shared archive, or preload the image with podman load."
  missing=1
fi

if [[ -z "${CONDITAR_SLURM_ACCOUNT:-}" && -t 0 ]]; then
  read -r -p "Slurm account (required for GPU jobs): " entered_account
  if [[ -n "$entered_account" ]]; then
    CONDITAR_SLURM_ACCOUNT="$entered_account"
    if [[ -f .conditar-slurm.env ]]; then
      sed -i '/^[[:space:]]*CONDITAR_SLURM_ACCOUNT=/d' .conditar-slurm.env
    fi
    printf 'CONDITAR_SLURM_ACCOUNT=%q\n' "$CONDITAR_SLURM_ACCOUNT" >> .conditar-slurm.env
    echo "Saved Slurm account to .conditar-slurm.env"
  fi
fi

if [[ -n "${CONDITAR_SLURM_ACCOUNT:-}" ]]; then
  echo "OK    Slurm account configured: $CONDITAR_SLURM_ACCOUNT"
else
  echo "WARN  Slurm account is not configured"
  echo "      Enter it in the GUI before submitting a GPU job, or set CONDITAR_SLURM_ACCOUNT."
fi

echo
if [[ "$missing" -eq 0 ]]; then
  echo "Slurm/GPU setup is ready. Start the GUI with:"
  echo "  ./start_slurm_gui.sh"
else
  echo "Setup check finished with missing requirements. Fix the items above, then rerun:"
  echo "  ./setup_slurm_gui.sh"
  exit 2
fi
