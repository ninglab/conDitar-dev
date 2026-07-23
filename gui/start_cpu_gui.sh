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
export CONDITAR_DOCKER_IMAGE="${CONDITAR_DOCKER_IMAGE:-localhost/conditar-dev:container-dev}"
DOCKER_COMMAND="${DOCKER_BIN:-docker}"
if [[ -z "${CONDITAR_SOURCE_MOUNT:-}" && -d ../conDitar-dev ]]; then
  export CONDITAR_SOURCE_MOUNT="$(cd ../conDitar-dev && pwd)"
elif [[ -z "${CONDITAR_SOURCE_MOUNT:-}" && -d ../docker && -d ../scripts ]]; then
  export CONDITAR_SOURCE_MOUNT="$(cd .. && pwd)"
fi

for required in python3 "$DOCKER_COMMAND"; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "ERROR: required local CPU command not found: $required" >&2
    echo "Install Python 3 and Docker Desktop, then retry." >&2
    exit 2
  fi
done

PYTHON_COMMAND=(python3)
if [[ -n "${CONDITAR_GUI_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("$CONDITAR_GUI_PYTHON")
elif command -v conda >/dev/null 2>&1 && conda run -n conditar-gui-dev python -c "import sys" >/dev/null 2>&1; then
  PYTHON_COMMAND=(conda run --no-capture-output -n conditar-gui-dev python)
fi

if ! "$DOCKER_COMMAND" image inspect "$CONDITAR_DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: conDitar container image not found: $CONDITAR_DOCKER_IMAGE" >&2
  echo "Load or build the image first, or set CONDITAR_DOCKER_IMAGE to an available image." >&2
  exit 2
fi

echo "Starting conDitar GUI"
echo "Container image: $CONDITAR_DOCKER_IMAGE"
echo "Source mount: ${CONDITAR_SOURCE_MOUNT:-none}"
echo "Runtime: $CONDITAR_RUNTIME"
echo "GUI Python: ${PYTHON_COMMAND[*]}"
echo "CPU mode: select This computer · CPU in the Setup panel"
echo

"${PYTHON_COMMAND[@]}" serve.py --host 127.0.0.1 --port "${PORT:-4173}" --open
