#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build the conDitar GUI container image for OpenShift.

Usage:
  ./openshift/build_gui_image.sh --image IMAGE [options]

Options:
  --image IMAGE       GUI image tag to build, such as docker.io/org/conditar-gui:tag.
  --platform TARGET   Container platform. Default: linux/amd64.
  --push              Push the image after building.
  --runtime NAME      Container runtime: docker or podman. Defaults to an available runtime.
  --help              Show this help.

Examples:
  ./openshift/build_gui_image.sh --image docker.io/osuninglab/conditar-gui:2026-09-03
  ./openshift/build_gui_image.sh --image docker.io/osuninglab/conditar-gui:2026-09-03 --push
EOF
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${CONDITAR_GUI_IMAGE:-}"
PLATFORM="${CONDITAR_GUI_IMAGE_PLATFORM:-linux/amd64}"
PUSH=0
RUNTIME="${CONTAINER_RUNTIME:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"; shift 2 ;;
    --platform)
      PLATFORM="${2:-}"; shift 2 ;;
    --push)
      PUSH=1; shift ;;
    --runtime)
      RUNTIME="${2:-}"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  echo "ERROR: --image is required." >&2
  usage >&2
  exit 2
fi

if [[ -z "$RUNTIME" ]]; then
  if command -v docker >/dev/null 2>&1; then
    RUNTIME="docker"
  elif command -v podman >/dev/null 2>&1; then
    RUNTIME="podman"
  else
    echo "ERROR: docker or podman was not found on PATH." >&2
    exit 2
  fi
fi

case "$RUNTIME" in
  docker)
    if [[ "$PUSH" -eq 1 ]]; then
      docker buildx build --platform "$PLATFORM" -f Containerfile -t "$IMAGE" --push .
    else
      docker build --platform "$PLATFORM" -f Containerfile -t "$IMAGE" .
    fi
    ;;
  podman)
    podman build --platform "$PLATFORM" -f Containerfile -t "$IMAGE" .
    if [[ "$PUSH" -eq 1 ]]; then
      podman push "$IMAGE"
    fi
    ;;
  *)
    echo "ERROR: --runtime must be docker or podman." >&2
    exit 2
    ;;
esac

echo "Built GUI image: $IMAGE"
if [[ "$PUSH" -eq 0 ]]; then
  echo "Push it with: $RUNTIME push $IMAGE"
fi
