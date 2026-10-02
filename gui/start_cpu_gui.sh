#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .conditar-cpu.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .conditar-cpu.env
  set +a
fi

export CONDITAR_RUNTIME="${CONDITAR_RUNTIME:-docker}"
DOCKER_COMMAND="${DOCKER_BIN:-docker}"
DEFAULT_IMAGE="conditar-dev:standalone-20261001"

image_available() {
  local image="$1"
  "$DOCKER_COMMAND" image inspect "$image" >/dev/null 2>&1 && return 0
  [[ "$image" != */* ]] && "$DOCKER_COMMAND" image inspect "docker.io/library/$image" >/dev/null 2>&1
}

if [[ -z "${CONDITAR_DOCKER_IMAGE:-}" ]]; then
  export CONDITAR_DOCKER_IMAGE="$DEFAULT_IMAGE"
fi

PYTHON_COMMAND=(python3)
if [[ -n "${CONDITAR_GUI_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("$CONDITAR_GUI_PYTHON")
elif command -v conda >/dev/null 2>&1 && conda run -n conditar-gui-dev python -c "import sys" >/dev/null 2>&1; then
  PYTHON_COMMAND=(conda run --no-capture-output -n conditar-gui-dev python)
fi

if ! "${PYTHON_COMMAND[@]}" -c "import sys" >/dev/null 2>&1; then
  echo "ERROR: Python was not found." >&2
  echo "Install Python 3 or Miniconda/Mambaforge, then retry." >&2
  echo "Tip: run ./setup_gui.sh for a guided setup check." >&2
  exit 2
fi

if ! command -v "$DOCKER_COMMAND" >/dev/null 2>&1; then
  echo "ERROR: required local CPU command not found: $DOCKER_COMMAND" >&2
  echo "Install Docker Desktop, then retry." >&2
  echo "Tip: run ./setup_gui.sh for a guided setup check." >&2
  exit 2
fi

if ! "$DOCKER_COMMAND" info >/dev/null 2>&1; then
  echo "ERROR: Docker is installed but does not appear to be running." >&2
  echo "Start Docker Desktop, then retry." >&2
  echo "Tip: run ./setup_gui.sh for a guided setup check." >&2
  exit 2
fi

if ! image_available "$CONDITAR_DOCKER_IMAGE"; then
  echo "ERROR: conDitar container image not found: $CONDITAR_DOCKER_IMAGE" >&2
  echo "Build the refreshed image, or set CONDITAR_DOCKER_IMAGE to an available image." >&2
  echo "Example:" >&2
  echo "  docker build --platform linux/amd64 -f ../docker/Refresh.Dockerfile -t $DEFAULT_IMAGE ../scripts" >&2
  echo "Then retry:" >&2
  echo "  ./start_cpu_gui.sh" >&2
  exit 2
fi

echo "Starting conDitar GUI"
echo "Container image: $CONDITAR_DOCKER_IMAGE"
echo "DiffSMol image: ${DIFFSMOL_DOCKER_IMAGE:-diffsmol:cpu-20261001}"
if ! image_available "${DIFFSMOL_DOCKER_IMAGE:-diffsmol:cpu-20261001}"; then
  echo "WARNING: DiffSMol image is unavailable; conDitar runs can still work." >&2
fi
echo "Source mount: ${CONDITAR_SOURCE_MOUNT:-none}"
echo "Runtime: $CONDITAR_RUNTIME"
echo "GUI Python: ${PYTHON_COMMAND[*]}"
echo "CPU mode: select This computer · CPU in the Setup panel"
echo

"${PYTHON_COMMAND[@]}" serve.py --host 127.0.0.1 --port "${PORT:-4173}" --auto-port --open
