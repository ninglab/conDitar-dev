#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${CONDITAR_IMAGE_OUTPUT_DIR:-/fs/ess/PCON0041/mey200/container_images}"
stamp="${CONDITAR_IMAGE_STAMP:-$(date +%Y%m%d-%H%M%S)}"
log_path="$output_dir/conditar-build-export-$stamp.log"

mkdir -p "$output_dir"
cd "$repo_root"

nohup env \
    CONDITAR_IMAGE_STAMP="$stamp" \
    CONDITAR_CONTAINER_ENGINE="${CONDITAR_CONTAINER_ENGINE:-buildah}" \
    CONDITAR_SAVE_ENGINE="${CONDITAR_SAVE_ENGINE:-podman}" \
    "$repo_root/docker/build-export-image.sh" --engine "${CONDITAR_CONTAINER_ENGINE:-buildah}" \
    > "$log_path" 2>&1 &

pid="$!"
echo "$pid" > "$output_dir/conditar-build-export-$stamp.pid"
echo "PID: $pid"
echo "LOG: $log_path"
