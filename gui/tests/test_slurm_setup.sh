#!/usr/bin/env bash
set -euo pipefail

GUI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEST_ROOT="$(mktemp -d "$GUI_DIR/.slurm-setup-test.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
mkdir -p "$TEST_ROOT/gui" "$TEST_ROOT/shared"
cp "$GUI_DIR/setup_slurm_gui.sh" "$TEST_ROOT/gui/"
cp "$GUI_DIR/start_slurm_gui.sh" "$TEST_ROOT/gui/"
printf 'CONDITAR_SLURM_PARTITION=gpu-test\nCONDITAR_DOCKER_TAR=/tmp/old.tar\n' > "$TEST_ROOT/gui/.conditar-slurm.env"

export MOCK_LOG="$TEST_ROOT/podman.log"
export MOCK_GUI_LOG="$TEST_ROOT/gui-launch.log"
podman() {
  printf '%s\n' "$*" >> "$MOCK_LOG"
  case "$1" in
    pull) return 0 ;;
    save) printf 'archive for %s\n' "$6" > "$5" ;;
    *) return 1 ;;
  esac
}
sbatch() { :; }
python3() {
  if [[ "${1:-}" == "-c" ]]; then
    return 0
  fi
  printf '%s|%s|%s|%s\n' "$CONDITAR_JOB_ROOT" "$CONDITAR_DOCKER_IMAGE" \
    "$DIFFSMOL_DOCKER_IMAGE" "$DIFFSMOL_DOCKER_TAR" > "$MOCK_GUI_LOG"
}
uname() {
  if [[ "$1" == "-m" ]]; then
    printf 'x86_64\n'
  else
    command uname "$@"
  fi
}
export -f podman sbatch python3 uname

"$TEST_ROOT/gui/setup_slurm_gui.sh" \
  --shared-dir "$TEST_ROOT/shared" --account test-account > "$TEST_ROOT/setup.log"

(
  source "$TEST_ROOT/gui/.conditar-slurm.env"
  [[ "$CONDITAR_SLURM_PARTITION" == "gpu-test" ]]
  [[ "$CONDITAR_SLURM_ACCOUNT" == "test-account" ]]
  [[ "$CONDITAR_JOB_ROOT" == "$TEST_ROOT/shared/jobs" ]]
  [[ "$CONDITAR_DOCKER_TAR" == "$TEST_ROOT/shared/images/conditar.tar" ]]
  [[ "$DIFFSMOL_DOCKER_TAR" == "$TEST_ROOT/shared/images/diffsmol.tar" ]]
  [[ "$CONDITAR_DOCKER_IMAGE" == "docker.io/osuninglab/conditar-dev:gui-dev-20261005" ]]
  [[ "$DIFFSMOL_DOCKER_IMAGE" == "docker.io/osuninglab/diffsmol:gui-dev-20261005" ]]
)
[[ -s "$TEST_ROOT/shared/images/conditar.tar" ]]
[[ -s "$TEST_ROOT/shared/images/diffsmol.tar" ]]
[[ "$(grep -c '^pull ' "$MOCK_LOG")" -eq 2 ]]
[[ "$(grep -c '^save ' "$MOCK_LOG")" -eq 2 ]]

CONDITAR_GUI_PYTHON=python3 "$TEST_ROOT/gui/start_slurm_gui.sh" > "$TEST_ROOT/start.log"
grep -Fq "$TEST_ROOT/shared/jobs|docker.io/osuninglab/conditar-dev:gui-dev-20261005|docker.io/osuninglab/diffsmol:gui-dev-20261005|$TEST_ROOT/shared/images/diffsmol.tar" "$MOCK_GUI_LOG"

mv "$TEST_ROOT/shared/images/diffsmol.tar" "$TEST_ROOT/diffsmol.saved"
if CONDITAR_GUI_PYTHON=python3 "$TEST_ROOT/gui/start_slurm_gui.sh" > "$TEST_ROOT/missing-diffsmol.log" 2>&1; then
  echo "Expected missing DiffSMol archive to fail at launch" >&2
  exit 1
fi
grep -q 'DiffSMol GPU container archive not found' "$TEST_ROOT/missing-diffsmol.log"
mv "$TEST_ROOT/diffsmol.saved" "$TEST_ROOT/shared/images/diffsmol.tar"

if "$TEST_ROOT/gui/setup_slurm_gui.sh" \
  --shared-dir /tmp/conditar-slurm-test --account test-account > "$TEST_ROOT/rejected.log" 2>&1; then
  echo "Expected node-local setup path to fail" >&2
  exit 1
fi
grep -q 'node-local temporary storage' "$TEST_ROOT/rejected.log"
[[ "$(grep -c '^pull ' "$MOCK_LOG")" -eq 2 ]]

printf 'CONDITAR_JOB_ROOT=/tmp/jobs\n' > "$TEST_ROOT/gui/.conditar-slurm.env"
if "$TEST_ROOT/gui/start_slurm_gui.sh" > "$TEST_ROOT/start-rejected.log" 2>&1; then
  echo "Expected node-local job root to fail at launch" >&2
  exit 1
fi
if ! grep -q 'node-local storage' "$TEST_ROOT/start-rejected.log"; then
  echo "Unexpected launcher error:" >&2
  sed -n '1,20p' "$TEST_ROOT/start-rejected.log" >&2
  exit 1
fi

echo "Slurm setup tests passed"
