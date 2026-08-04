#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="${CONDITAR_DOCKER_IMAGE:-localhost/conditar-dev:container-dev}"
ARCHIVE_NAME="localhost_conditar-dev_container-dev-20260710-105038.tar.gz"

echo "conDitar GUI Mac starter"
echo

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker was not found."
  echo "Install Docker Desktop, open it once, then run this file again."
  echo "https://www.docker.com/products/docker-desktop/"
  read -r -p "Press Enter to close."
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker Desktop is not running."
  echo "Open Docker Desktop, wait until it finishes starting, then run this file again."
  read -r -p "Press Enter to close."
  exit 2
fi

if command -v conda >/dev/null 2>&1; then
  if ! conda run -n conditar-gui-dev python -c "import sys" >/dev/null 2>&1; then
    echo "Setting up optional GUI Tool Chest filters..."
    ./setup_tool_chest.sh
    echo
  fi
else
  echo "Conda was not found. The base GUI can still launch, but optional Lily/MedChem filters may be unavailable."
  echo "Install Miniconda or Miniforge later if you want those filters."
  echo
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "conDitar container image is not loaded yet."
  echo "Looking for $ARCHIVE_NAME..."
  shopt -s nullglob
  candidates=(
    "$PWD"/"$ARCHIVE_NAME"
    "$PWD"/../"$ARCHIVE_NAME"
    "$HOME"/Downloads/"$ARCHIVE_NAME"
    "$HOME"/Desktop/"$ARCHIVE_NAME"
    "$HOME"/containers/"$ARCHIVE_NAME"
  )
  shopt -u nullglob

  archive=""
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      archive="$candidate"
      break
    fi
  done

  if [[ -z "$archive" ]] && command -v osascript >/dev/null 2>&1; then
    archive="$(osascript -e 'POSIX path of (choose file with prompt "Select the conDitar container .tar.gz file")' 2>/dev/null || true)"
  fi

  if [[ -z "$archive" || ! -f "$archive" ]]; then
    echo "ERROR: Container archive was not selected or found."
    echo "Expected file: $ARCHIVE_NAME"
    read -r -p "Press Enter to close."
    exit 2
  fi

  echo "Loading container image from:"
  echo "$archive"
  docker load -i "$archive"
  echo
fi

echo "Starting conDitar GUI..."
./start_cpu_gui.sh
