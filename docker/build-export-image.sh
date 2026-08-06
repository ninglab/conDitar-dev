#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${CONDITAR_DOCKER_TAG:-conditar-dev:2026-07-10}"
output_dir="${CONDITAR_IMAGE_OUTPUT_DIR:-$repo_root/container_images}"
stamp="${CONDITAR_IMAGE_STAMP:-$(date +%Y%m%d-%H%M%S)}"

for ((index = 1; index <= $#; index++)); do
    case "${!index}" in
        --tag)
            next=$((index + 1))
            image_tag="${!next:?missing value for --tag}"
            ;;
        -h|--help)
            "$repo_root/docker/build-image.sh" --help
            exit 0
            ;;
    esac
done

mkdir -p "$output_dir"

echo "[$(date)] Building image: $image_tag"
"$repo_root/docker/build-image.sh" "$@"

safe_tag="$(echo "$image_tag" | tr '/:' '__')"
tar_path="$output_dir/${safe_tag}-${stamp}.tar"
gz_path="$tar_path.gz"

save_engine="${CONDITAR_SAVE_ENGINE:-auto}"
if [[ "$save_engine" == "auto" ]]; then
    if command -v podman >/dev/null 2>&1; then
        save_engine="podman"
    else
        save_engine="docker"
    fi
fi

echo "[$(date)] Saving image with $save_engine: $tar_path"
case "$save_engine" in
    podman)
        podman save "$image_tag" -o "$tar_path"
        ;;
    docker)
        docker save "$image_tag" -o "$tar_path"
        ;;
    *)
        echo "Unsupported save engine: $save_engine" >&2
        echo "Use CONDITAR_SAVE_ENGINE=podman, docker, or auto." >&2
        exit 2
        ;;
esac

if command -v pigz >/dev/null 2>&1; then
    echo "[$(date)] Compressing with pigz -9: $gz_path"
    pigz -f -9 "$tar_path"
else
    echo "[$(date)] Compressing with gzip -9: $gz_path"
    gzip -f -9 "$tar_path"
fi

echo "[$(date)] Done"
ls -lh "$gz_path"
