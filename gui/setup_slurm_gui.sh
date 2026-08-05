#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .conditar-slurm.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .conditar-slurm.env
  set +a
fi

ARCHIVE="${CONDITAR_DOCKER_TAR:-}"
PODMAN_COMMAND="${PODMAN_BIN:-podman}"
SBATCH_COMMAND="${SBATCH_BIN:-sbatch}"
PUBLIC_IMAGE="docker.io/averyemeyer/conditar-dev:2026-07-10"
LEGACY_IMAGE="localhost/conditar-dev:container-dev"
IMAGE="${CONDITAR_DOCKER_IMAGE:-$PUBLIC_IMAGE}"

# Match the launcher convenience path search.
if [[ -z "$ARCHIVE" ]]; then
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

check_command "$PODMAN_COMMAND" "Load Podman or set PODMAN_BIN=/path/to/podman."
check_command "$SBATCH_COMMAND" "Load Slurm or set SBATCH_BIN=/path/to/sbatch."

if command -v python3 >/dev/null 2>&1; then
  echo "OK    python3 found: $(command -v python3)"
elif command -v conda >/dev/null 2>&1; then
  echo "OK    conda found; the optional Tool Chest environment can provide GUI Python"
else
  echo "MISS  Python was not found"
  echo "      Load Python 3.9 or newer, or install Miniconda/Mambaforge."
  missing=1
fi

remove_env_value() {
  local key="$1"
  if [[ ! -f .conditar-slurm.env ]]; then
    return
  fi
  local tmp
  tmp="$(mktemp .conditar-slurm.env.XXXXXX)"
  grep -v -E "^[[:space:]]*${key}=" .conditar-slurm.env > "$tmp" || true
  mv "$tmp" .conditar-slurm.env
}

save_env_value() {
  local key="$1"
  local value="$2"
  remove_env_value "$key"
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
elif [[ -z "${CONDITAR_DOCKER_IMAGE:-}" ]] && command -v "$PODMAN_COMMAND" >/dev/null 2>&1 \
  && "$PODMAN_COMMAND" image exists "$LEGACY_IMAGE" >/dev/null 2>&1; then
  echo "OK    preloaded container image found with legacy local tag: $LEGACY_IMAGE"
else
  echo "MISS  no container archive or preloaded image found: $IMAGE"
  echo "      Pull it with: podman pull $PUBLIC_IMAGE"
  echo "      Or set CONDITAR_DOCKER_TAR to a shared archive."
  missing=1
fi

if [[ -z "${CONDITAR_SLURM_ACCOUNT:-}" && -t 0 ]]; then
  read -r -p "Slurm account (required for GPU jobs): " entered_account
  if [[ -n "$entered_account" ]]; then
    CONDITAR_SLURM_ACCOUNT="$entered_account"
    save_env_value CONDITAR_SLURM_ACCOUNT "$CONDITAR_SLURM_ACCOUNT"
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
