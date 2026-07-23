#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE="${CONDITAR_DOCKER_IMAGE:-localhost/conditar-dev:container-dev}"
DOCKER_COMMAND="${DOCKER_BIN:-docker}"

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

check_command python3 "Install Python 3.9 or newer."
check_command "$DOCKER_COMMAND" "Install Docker Desktop, then restart this terminal."

if command -v "$DOCKER_COMMAND" >/dev/null 2>&1; then
  if "$DOCKER_COMMAND" info >/dev/null 2>&1; then
    echo "OK    Docker is running"
  else
    echo "MISS  Docker is installed but not running"
    echo "      Start Docker Desktop, then rerun this setup check."
    missing=1
  fi

  if "$DOCKER_COMMAND" image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "OK    conDitar image found: $IMAGE"
  else
    echo "MISS  conDitar image not found: $IMAGE"
    echo "      Load it with:"
    echo "        docker load -i /path/to/localhost_conditar-dev_container-dev.tar.gz"
    echo "      Or build it from the repository Docker instructions."
    missing=1
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
