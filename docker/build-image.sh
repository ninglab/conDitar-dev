#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${CONDITAR_DOCKER_TAG:-localhost/conditar-dev:container-dev}"
platform="${CONDITAR_DOCKER_PLATFORM:-linux/amd64}"
checkpoint_src="${CONDITAR_CHECKPOINT_DIR:-/fs/ess/PCON0041/gruoxi/SBDDcode/checkpoints}"
checkpoint_dest="$repo_root/docker/checkpoints"
qvina_src="${CONDITAR_QVINA_BIN:-/fs/ess/PCON0041/gruoxi/qvina/bin/qvina2.1}"
qvina_dest="$repo_root/docker/qvina/qvina2.1"
container_engine="${CONDITAR_CONTAINER_ENGINE:-auto}"
buildah_isolation="${CONDITAR_BUILDAH_ISOLATION:-chroot}"
buildah_userns="${CONDITAR_BUILDAH_USERNS:-host}"

usage() {
    cat <<EOF
Usage:
  docker/build-image.sh [--tag IMAGE_TAG] [--platform PLATFORM] [--checkpoint-dir DIR] [--qvina-bin FILE] [--engine ENGINE]

Defaults:
  IMAGE_TAG       $image_tag
  PLATFORM        $platform
  CHECKPOINT_DIR  $checkpoint_src
  QVINA_BIN       $qvina_src
  ENGINE          $container_engine

Environment overrides:
  CONDITAR_DOCKER_TAG
  CONDITAR_DOCKER_PLATFORM
  CONDITAR_CHECKPOINT_DIR
  CONDITAR_QVINA_BIN
  CONDITAR_CONTAINER_ENGINE  auto, docker, or buildah
  CONDITAR_BUILDAH_ISOLATION default: chroot
  CONDITAR_BUILDAH_USERNS    default: host
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            image_tag="${2:?missing value for $1}"
            shift 2
            ;;
        --platform)
            platform="${2:?missing value for $1}"
            shift 2
            ;;
        --checkpoint-dir)
            checkpoint_src="${2:?missing value for $1}"
            shift 2
            ;;
        --qvina-bin)
            qvina_src="${2:?missing value for $1}"
            shift 2
            ;;
        --engine)
            container_engine="${2:?missing value for $1}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for checkpoint in Diff.pt PocketAE.pt; do
    if [[ ! -f "$checkpoint_src/$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint_src/$checkpoint" >&2
        echo "Set --checkpoint-dir or CONDITAR_CHECKPOINT_DIR to the directory containing Diff.pt and PocketAE.pt." >&2
        exit 2
    fi
done

mkdir -p "$checkpoint_dest"
cp "$checkpoint_src/Diff.pt" "$checkpoint_dest/Diff.pt"
cp "$checkpoint_src/PocketAE.pt" "$checkpoint_dest/PocketAE.pt"

mkdir -p "$(dirname "$qvina_dest")"
if [[ -f "$qvina_src" ]]; then
    cp "$qvina_src" "$qvina_dest"
    chmod 0755 "$qvina_dest"
    echo "Using QuickVina2 from $qvina_src"
elif [[ -f "$qvina_dest" ]]; then
    chmod 0755 "$qvina_dest"
    echo "Using previously staged QuickVina2 at $qvina_dest"
else
    echo "QuickVina2 binary not found at $qvina_src" >&2
    echo "The image will still build, but --vina-mode qvina/all will require CONDITAR_QVINA_BIN or --qvina-bin at runtime." >&2
fi

echo "Building $image_tag for $platform"
echo "Using checkpoints from $checkpoint_src"

if [[ "$container_engine" == "auto" ]]; then
    if command -v buildah >/dev/null 2>&1; then
        container_engine="buildah"
    else
        container_engine="docker"
    fi
fi

case "$container_engine" in
    buildah)
        buildah build \
            --format docker \
            --isolation "$buildah_isolation" \
            --userns "$buildah_userns" \
            --platform "$platform" \
            -f "$repo_root/docker/Dockerfile" \
            -t "$image_tag" \
            "$repo_root"
        ;;
    docker)
        docker build \
            --platform "$platform" \
            -f "$repo_root/docker/Dockerfile" \
            -t "$image_tag" \
            "$repo_root"
        ;;
    *)
        echo "Unsupported engine: $container_engine" >&2
        echo "Use --engine docker, --engine buildah, or --engine auto." >&2
        exit 2
        ;;
esac

echo "Built $image_tag"
