#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Deploy the prebuilt conDitar GUI and CPU engine Jobs to the current OpenShift project.

Usage:
  ./openshift/deploy.sh [options]

Options:
  --project NAME             Switch to an existing OpenShift project first.
  --create-project NAME      Create or switch to an OpenShift project first.
  --runtime MODE             GUI runtime target: openshift_mock or openshift_job.
  --gui-image IMAGE          Override the prebuilt GUI image.
  --runtime-image IMAGE      Registry-pullable refreshed conDitar image for Jobs.
  --diffsmol-image IMAGE     Registry-pullable refreshed DiffSMol image for Jobs.
  --submit                   Allow the GUI pod to create and poll OpenShift Jobs (default).
  --no-submit                Write Job manifests without submitting them.
  --cpu                      Configure generated OpenShift Jobs for CPU-only execution (default).
  --gpu                      Configure generated OpenShift Jobs for one GPU (requires quota).
  --storage SIZE             PVC request size, such as 10Gi or 50Gi.
  --route-host HOST          Optional fixed Route hostname.
  --build-gui                Build the GUI image inside OpenShift instead of pulling the prebuilt image.
  --skip-build               Apply manifests without starting a new OpenShift build.
  --help                     Show this help.

No arguments use the current oc project and three pinned, prebuilt images.
The default is real CPU Jobs; no dependencies are installed in the caller's environment.
EOF
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
. openshift/image-list.txt

PROJECT=""
CREATE_PROJECT=""
RUNTIME="${CONDITAR_RUNTIME:-openshift_job}"
GUI_IMAGE="${CONDITAR_GUI_IMAGE:-$GUI}"
RUNTIME_IMAGE="${CONDITAR_DOCKER_IMAGE:-$CONDITAR}"
DIFFSMOL_IMAGE="${DIFFSMOL_DOCKER_IMAGE:-$DIFFSMOL}"
RUNTIME_EXPLICIT=0
SUBMIT_EXPLICIT=0
STORAGE="${CONDITAR_OPENSHIFT_STORAGE:-10Gi}"
ROUTE_HOST="${CONDITAR_OPENSHIFT_ROUTE_HOST:-}"
SKIP_BUILD=0
OPENSHIFT_SUBMIT="${CONDITAR_OPENSHIFT_SUBMIT:-true}"
OPENSHIFT_DEVICE="${CONDITAR_OPENSHIFT_DEVICE:-cpu}"
OPENSHIFT_GPU_RESOURCE="${CONDITAR_OPENSHIFT_GPU_RESOURCE:-nvidia.com/gpu}"
OPENSHIFT_GPU_COUNT="${CONDITAR_OPENSHIFT_GPU_COUNT:-0}"
OPENSHIFT_CPU_REQUEST="${CONDITAR_OPENSHIFT_CPU_REQUEST:-500m}"
OPENSHIFT_MEMORY_REQUEST="${CONDITAR_OPENSHIFT_MEMORY_REQUEST:-4Gi}"
OPENSHIFT_MEMORY_LIMIT="${CONDITAR_OPENSHIFT_MEMORY_LIMIT:-8Gi}"
BUILD_TIMEOUT_SECONDS="${CONDITAR_OPENSHIFT_BUILD_TIMEOUT_SECONDS:-2400}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --create-project)
      CREATE_PROJECT="${2:-}"; shift 2 ;;
    --runtime)
      RUNTIME="${2:-}"; RUNTIME_EXPLICIT=1; shift 2 ;;
    --gui-image)
      GUI_IMAGE="${2:-}"; shift 2 ;;
    --runtime-image)
      RUNTIME_IMAGE="${2:-}"; shift 2 ;;
    --diffsmol-image)
      DIFFSMOL_IMAGE="${2:-}"; shift 2 ;;
    --submit)
      OPENSHIFT_SUBMIT="true"; SUBMIT_EXPLICIT=1; shift ;;
    --no-submit)
      OPENSHIFT_SUBMIT="false"; SUBMIT_EXPLICIT=1; shift ;;
    --cpu)
      OPENSHIFT_DEVICE="cpu"
      OPENSHIFT_GPU_COUNT="0"
      OPENSHIFT_CPU_REQUEST="${CONDITAR_OPENSHIFT_CPU_REQUEST:-500m}"
      OPENSHIFT_MEMORY_REQUEST="${CONDITAR_OPENSHIFT_MEMORY_REQUEST:-4Gi}"
      OPENSHIFT_MEMORY_LIMIT="${CONDITAR_OPENSHIFT_MEMORY_LIMIT:-8Gi}"
      shift ;;
    --gpu)
      OPENSHIFT_DEVICE="cuda:0"
      OPENSHIFT_GPU_COUNT="${CONDITAR_OPENSHIFT_GPU_COUNT:-1}"
      if [[ "$OPENSHIFT_GPU_COUNT" == "0" ]]; then
        OPENSHIFT_GPU_COUNT="1"
      fi
      OPENSHIFT_CPU_REQUEST="${CONDITAR_OPENSHIFT_CPU_REQUEST:-2}"
      OPENSHIFT_MEMORY_REQUEST="${CONDITAR_OPENSHIFT_MEMORY_REQUEST:-16Gi}"
      OPENSHIFT_MEMORY_LIMIT="${CONDITAR_OPENSHIFT_MEMORY_LIMIT:-32Gi}"
      shift ;;
    --storage)
      STORAGE="${2:-}"; shift 2 ;;
    --route-host)
      ROUTE_HOST="${2:-}"; shift 2 ;;
    --build-gui)
      GUI_IMAGE=""; shift ;;
    --skip-build)
      SKIP_BUILD=1; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$RUNTIME" == "openshift_mock" && "$SUBMIT_EXPLICIT" -eq 0 ]]; then
  OPENSHIFT_SUBMIT="false"
fi

if [[ "$OPENSHIFT_SUBMIT" == "true" && "$RUNTIME_EXPLICIT" -eq 0 ]]; then
  RUNTIME="openshift_job"
fi

if [[ "$RUNTIME" != "openshift_mock" && "$RUNTIME" != "openshift_job" ]]; then
  echo "ERROR: --runtime must be openshift_mock or openshift_job." >&2
  exit 2
fi

if [[ "$OPENSHIFT_SUBMIT" == "true" && "$RUNTIME" == "openshift_mock" ]]; then
  echo "ERROR: --submit cannot be combined with --runtime openshift_mock. Use --runtime openshift_job for real Jobs." >&2
  exit 2
fi

if [[ "$OPENSHIFT_DEVICE" == "cpu" ]]; then
  if [[ "$OPENSHIFT_GPU_COUNT" != "0" ]]; then
    echo "ERROR: CPU Jobs require CONDITAR_OPENSHIFT_GPU_COUNT=0." >&2
    exit 2
  fi
elif [[ "$OPENSHIFT_DEVICE" =~ ^cuda:[0-9]+$ ]]; then
  if [[ ! "$OPENSHIFT_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CUDA Jobs require a positive integer CONDITAR_OPENSHIFT_GPU_COUNT." >&2
    exit 2
  fi
else
  echo "ERROR: CONDITAR_OPENSHIFT_DEVICE must be cpu or cuda:N." >&2
  exit 2
fi

if [[ "$OPENSHIFT_SUBMIT" == "true" ]]; then
  if [[ "$RUNTIME_IMAGE" == *osuninglab/conditar-dev:2026-07-10 || "$RUNTIME_IMAGE" != */* ]]; then
    echo "ERROR: --submit requires a registry-pullable refreshed conDitar image; set --runtime-image or CONDITAR_DOCKER_IMAGE." >&2
    exit 2
  fi
  if [[ "$DIFFSMOL_IMAGE" == *ninglab/diffsmol:latest || "$DIFFSMOL_IMAGE" != */* ]]; then
    echo "ERROR: --submit requires a registry-pullable refreshed DiffSMol image; set --diffsmol-image or DIFFSMOL_DOCKER_IMAGE." >&2
    exit 2
  fi
fi

if [[ -z "$GUI_IMAGE" && "$SKIP_BUILD" -eq 0 ]]; then
  echo "Building the GUI in OpenShift because --build-gui was selected. The default path only pulls prebuilt images."
fi

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: oc was not found on PATH. Run this from the OpenShift Web Terminal, or use a workstation with oc already available." >&2
  exit 2
fi

if ! oc whoami >/dev/null 2>&1; then
  echo "ERROR: oc is not logged in. Run oc login, then retry." >&2
  exit 2
fi

if [[ -n "$CREATE_PROJECT" ]]; then
  oc new-project "$CREATE_PROJECT" >/dev/null 2>&1 || oc project "$CREATE_PROJECT" >/dev/null
elif [[ -n "$PROJECT" ]]; then
  oc project "$PROJECT" >/dev/null
fi

PROJECT_NAME="$(oc project -q)"
echo "Using OpenShift project: $PROJECT_NAME"
echo "GUI image: $GUI_IMAGE"
echo "conDitar image: $RUNTIME_IMAGE"
echo "DiffSMol image: $DIFFSMOL_IMAGE"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

previous_image_ref="$(oc get deployment/conditar-gui -o jsonpath='{.spec.template.spec.containers[?(@.name=="gui")].image}' 2>/dev/null || true)"
if [[ -n "$GUI_IMAGE" ]]; then
  current_image_ref="$GUI_IMAGE"
else
  existing_image_ref="$(oc get istag conditar-gui:latest -o jsonpath='{.image.dockerImageReference}' 2>/dev/null || true)"
  current_image_ref="${previous_image_ref:-$existing_image_ref}"
fi

cp -R openshift "$tmpdir/openshift"
if [[ -n "$GUI_IMAGE" ]]; then
  awk '$1 != "-" || ($2 != "imagestream.yaml" && $2 != "buildconfig.yaml")' \
    "$tmpdir/openshift/kustomization.yaml" > "$tmpdir/kustomization.yaml"
  mv "$tmpdir/kustomization.yaml" "$tmpdir/openshift/kustomization.yaml"
fi

rewrite_file() {
  local file="$1"
  local tmp_file="$file.tmp"
  cat >"$tmp_file"
  mv "$tmp_file" "$file"
}

set_config_value() {
  local file="$1"
  local key="$2"
  local value="$3"
  awk -v key="$key" -v value="$value" '
    $1 == key ":" {
      indent = substr($0, 1, index($0, key) - 1)
      print indent key ": \"" value "\""
      next
    }
    { print }
  ' "$file" | rewrite_file "$file"
}

set_plain_value() {
  local file="$1"
  local key="$2"
  local value="$3"
  awk -v key="$key" -v value="$value" '
    $1 == key ":" {
      indent = substr($0, 1, index($0, key) - 1)
      print indent key ": " value
      next
    }
    { print }
  ' "$file" | rewrite_file "$file"
}

insert_route_host() {
  local file="$1"
  local host="$2"
  if grep -Eq '^[[:space:]]*host:' "$file"; then
    set_plain_value "$file" "host" "$host"
  else
    awk -v host="$host" '
      $1 == "spec:" {
        print
        print "  host: " host
        next
      }
      { print }
    ' "$file" | rewrite_file "$file"
  fi
}

set_deployment_image() {
  local file="$1"
  local image="$2"
  awk -v image="$image" '
    $1 == "image:" && $2 ~ /^conditar-gui:/ {
      indent = substr($0, 1, index($0, "image") - 1)
      print indent "image: " image
      next
    }
    { print }
  ' "$file" | rewrite_file "$file"
}

wait_for_build() {
  local build_ref="$1"
  local phase=""
  local waited=0
  local interval=15

  echo "Waiting for $build_ref to complete. This can take about 10 minutes on a fresh OpenShift node."
  while [[ "$waited" -le "$BUILD_TIMEOUT_SECONDS" ]]; do
    if ! phase="$(oc get "$build_ref" -o jsonpath='{.status.phase}')"; then
      echo "ERROR: could not check $build_ref. Verify your OpenShift login and project, then retry with --skip-build if the build completed." >&2
      return 1
    fi
    case "$phase" in
      Complete)
        echo "$build_ref completed."
        return 0
        ;;
      Failed|Error|Cancelled)
        echo "ERROR: $build_ref ended with status: $phase" >&2
        oc logs "$build_ref" --tail=120 >&2 || true
        return 1
        ;;
      New|Pending|Running)
        echo "$build_ref status: $phase (${waited}s elapsed)"
        ;;
      *)
        echo "$build_ref status: waiting (${waited}s elapsed)"
        ;;
    esac
    sleep "$interval"
    waited=$((waited + interval))
  done

  echo "ERROR: timed out waiting for $build_ref after ${BUILD_TIMEOUT_SECONDS}s." >&2
  echo "The build may still be running. Check it with:" >&2
  echo "  oc get builds" >&2
  echo "  oc logs $build_ref --tail=120" >&2
  echo "If it later completes, finish deployment with:" >&2
  echo "  ./openshift/deploy.sh --build-gui --skip-build" >&2
  return 1
}

set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_RUNTIME" "$RUNTIME"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_DOCKER_IMAGE" "$RUNTIME_IMAGE"
set_config_value "$tmpdir/openshift/configmap.yaml" "DIFFSMOL_DOCKER_IMAGE" "$DIFFSMOL_IMAGE"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_SUBMIT" "$OPENSHIFT_SUBMIT"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_DEVICE" "$OPENSHIFT_DEVICE"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_GPU_RESOURCE" "$OPENSHIFT_GPU_RESOURCE"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_GPU_COUNT" "$OPENSHIFT_GPU_COUNT"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_CPU_REQUEST" "$OPENSHIFT_CPU_REQUEST"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_MEMORY_REQUEST" "$OPENSHIFT_MEMORY_REQUEST"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_MEMORY_LIMIT" "$OPENSHIFT_MEMORY_LIMIT"
set_plain_value "$tmpdir/openshift/pvc.yaml" "storage" "$STORAGE"

if [[ -n "$ROUTE_HOST" ]]; then
  insert_route_host "$tmpdir/openshift/route.yaml" "$ROUTE_HOST"
fi

if [[ -n "$current_image_ref" ]]; then
  set_deployment_image "$tmpdir/openshift/deployment.yaml" "$current_image_ref"
fi

oc apply -k "$tmpdir/openshift"

if [[ -n "$GUI_IMAGE" ]]; then
  echo "Using prebuilt GUI image: $GUI_IMAGE"
  echo "Skipping OpenShift binary build because --gui-image was provided."
elif [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "Starting OpenShift binary build from $(pwd)"
  build_ref="$(oc start-build conditar-gui --from-dir=. --exclude='(^|/)(\.git|job_data|\.tool_chest|__pycache__|\.pytest_cache)(/|$)|(^|/)\.conditar-.*\.env$|\.pyc$|\.tar(\.gz)?$' -o name)"
  echo "Started $build_ref"
  if ! wait_for_build "$build_ref"; then
    if [[ -n "$previous_image_ref" ]]; then
      echo "Build failed; restoring previous GUI image."
      oc set image deployment/conditar-gui "gui=$previous_image_ref" >/dev/null || true
    fi
    exit 1
  fi
else
  echo "Skipping build because --skip-build was provided."
fi

if [[ -n "$GUI_IMAGE" ]]; then
  image_ref="$GUI_IMAGE"
else
  image_ref="$(oc get istag conditar-gui:latest -o jsonpath='{.image.dockerImageReference}' 2>/dev/null || true)"
fi
if [[ -n "$image_ref" ]]; then
  oc set image deployment/conditar-gui "gui=$image_ref" >/dev/null
fi

if [[ -n "$previous_image_ref" ]]; then
  echo "Restarting the GUI pod to apply the current ConfigMap values."
  oc rollout restart deployment/conditar-gui >/dev/null
fi

if ! oc rollout status deployment/conditar-gui --timeout=15m; then
  echo "ERROR: GUI rollout failed. Inspect pods with: oc get pods -l app=conditar-gui" >&2
  echo "If the images are in another OpenShift project, grant image-pull permission to this project's GUI and default service accounts." >&2
  exit 1
fi

if [[ "$RUNTIME" == "openshift_mock" ]]; then
  echo "Runtime: DIAGNOSTICS ONLY (mock outputs; no generator Jobs)."
elif [[ "$OPENSHIFT_SUBMIT" == "true" ]]; then
  echo "Runtime: OpenShift Jobs (real generator submissions)."
else
  echo "Runtime: manifest-only (no generator Jobs submitted)."
fi

route_url="$(oc get route conditar-gui -o jsonpath='{.spec.host}' 2>/dev/null || true)"
if [[ -n "$route_url" ]]; then
  echo "conDitar GUI: https://$route_url"
else
  echo "Deployment is ready. No Route host was reported; inspect with: oc get route conditar-gui"
fi
