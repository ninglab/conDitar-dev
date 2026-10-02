#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .conditar-cpu.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .conditar-cpu.env
  set +a
fi

DOCKER_COMMAND="${DOCKER_BIN:-docker}"
DEFAULT_IMAGE="conditar-dev:standalone-20261001"
IMAGE="${CONDITAR_DOCKER_IMAGE:-$DEFAULT_IMAGE}"
DIFFSMOL_IMAGE="${DIFFSMOL_DOCKER_IMAGE:-diffsmol:cpu-20261001}"

image_available() {
  local image="$1"
  "$DOCKER_COMMAND" image inspect "$image" >/dev/null 2>&1 && return 0
  [[ "$image" != */* ]] && "$DOCKER_COMMAND" image inspect "docker.io/library/$image" >/dev/null 2>&1
}

echo "conDitar GUI setup check"
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

check_command "$DOCKER_COMMAND" "Install Docker Desktop, then restart this terminal."

if command -v python3 >/dev/null 2>&1; then
  echo "OK    python3 found: $(command -v python3)"
elif command -v conda >/dev/null 2>&1; then
  echo "OK    conda found; the optional Tool Chest environment can provide GUI Python"
else
  echo "MISS  Python was not found"
  echo "      Install Python 3.9 or newer, Miniconda, or Mambaforge."
  missing=1
fi

if command -v "$DOCKER_COMMAND" >/dev/null 2>&1; then
  if "$DOCKER_COMMAND" info >/dev/null 2>&1; then
    echo "OK    Docker is running"
  else
    echo "MISS  Docker is installed but not running"
    echo "      Start Docker Desktop, then rerun this setup check."
    missing=1
  fi

  if image_available "$IMAGE"; then
    echo "OK    conDitar image found: $IMAGE"
  else
    echo "MISS  conDitar image not found: $IMAGE"
    echo "      Build the refreshed standalone image:"
    echo "        docker build --platform linux/amd64 -f ../docker/Refresh.Dockerfile -t $DEFAULT_IMAGE ../scripts"
    missing=1
  fi

  if image_available "$DIFFSMOL_IMAGE"; then
    echo "OK    DiffSMol image found: $DIFFSMOL_IMAGE"
  else
    echo "WARN  DiffSMol image not found: $DIFFSMOL_IMAGE"
    echo "      Set DIFFSMOL_DOCKER_IMAGE in .conditar-cpu.env after loading or building the image."
    echo "      conDitar-only runs can still work."
  fi
fi

echo
if command -v conda >/dev/null 2>&1; then
  if conda run -n conditar-gui-dev python -c "import sys" >/dev/null 2>&1; then
    echo "OK    optional Tool Chest environment found: conditar-gui-dev"
  else
    echo "SETUP optional Tool Chest environment"
    ./setup_tool_chest.sh
  fi
else
  echo "SKIP  conda not found; optional Tool Chest tools may be unavailable."
  echo "      The basic GUI can still run with system Python."
fi

echo
if [[ "$missing" -eq 0 ]]; then
  echo "Ready. Start the GUI with:"
  echo "  ./start_cpu_gui.sh"
else
  echo "Setup check finished with missing requirements."
  echo "Fix the items above, then rerun:"
  echo "  ./setup_gui.sh"
  exit 2
fi
