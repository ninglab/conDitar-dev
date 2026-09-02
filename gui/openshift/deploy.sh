#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Deploy the conDitar GUI to the current OpenShift project.

Usage:
  ./openshift/deploy.sh [options]

Options:
  --project NAME             Switch to an existing OpenShift project first.
  --create-project NAME      Create or switch to an OpenShift project first.
  --runtime MODE             Default GUI runtime target: openshift_mock or openshift_job.
  --runtime-image IMAGE      conDitar generator image used by OpenShift Jobs.
  --submit                   Allow the GUI pod to create and poll OpenShift Jobs.
  --cpu                      Configure generated OpenShift Jobs for CPU-only execution.
  --storage SIZE             PVC request size, such as 10Gi or 50Gi.
  --route-host HOST          Optional fixed Route hostname.
  --skip-build               Apply manifests without starting a new OpenShift build.
  --minimal-tools            Build the GUI without optional Lilly/MedChem tools.
  --help                     Show this help.

Default runtime is openshift_mock, which is safe for first deployment checks.
EOF
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT=""
CREATE_PROJECT=""
RUNTIME="${CONDITAR_RUNTIME:-openshift_mock}"
RUNTIME_IMAGE="${CONDITAR_DOCKER_IMAGE:-osuninglab/conditar-dev:2026-07-10}"
STORAGE="${CONDITAR_OPENSHIFT_STORAGE:-10Gi}"
ROUTE_HOST="${CONDITAR_OPENSHIFT_ROUTE_HOST:-}"
SKIP_BUILD=0
OPENSHIFT_SUBMIT="${CONDITAR_OPENSHIFT_SUBMIT:-false}"
OPENSHIFT_DEVICE="${CONDITAR_OPENSHIFT_DEVICE:-cuda:0}"
OPENSHIFT_GPU_COUNT="${CONDITAR_OPENSHIFT_GPU_COUNT:-1}"
OPENSHIFT_CPU_REQUEST="${CONDITAR_OPENSHIFT_CPU_REQUEST:-2}"
OPENSHIFT_MEMORY_REQUEST="${CONDITAR_OPENSHIFT_MEMORY_REQUEST:-16Gi}"
OPENSHIFT_MEMORY_LIMIT="${CONDITAR_OPENSHIFT_MEMORY_LIMIT:-32Gi}"
BUILD_TIMEOUT_SECONDS="${CONDITAR_OPENSHIFT_BUILD_TIMEOUT_SECONDS:-2400}"
GUI_TOOL_CHEST="${CONDITAR_GUI_TOOL_CHEST:-full}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:-}"; shift 2 ;;
    --create-project)
      CREATE_PROJECT="${2:-}"; shift 2 ;;
    --runtime)
      RUNTIME="${2:-}"; shift 2 ;;
    --runtime-image)
      RUNTIME_IMAGE="${2:-}"; shift 2 ;;
    --submit)
      OPENSHIFT_SUBMIT="true"; shift ;;
    --cpu)
      OPENSHIFT_DEVICE="cpu"
      OPENSHIFT_GPU_COUNT="0"
      OPENSHIFT_CPU_REQUEST="${CONDITAR_OPENSHIFT_CPU_REQUEST:-500m}"
      OPENSHIFT_MEMORY_REQUEST="${CONDITAR_OPENSHIFT_MEMORY_REQUEST:-4Gi}"
      OPENSHIFT_MEMORY_LIMIT="${CONDITAR_OPENSHIFT_MEMORY_LIMIT:-8Gi}"
      shift ;;
    --storage)
      STORAGE="${2:-}"; shift 2 ;;
    --route-host)
      ROUTE_HOST="${2:-}"; shift 2 ;;
    --skip-build)
      SKIP_BUILD=1; shift ;;
    --minimal-tools)
      GUI_TOOL_CHEST="minimal"; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$RUNTIME" != "openshift_mock" && "$RUNTIME" != "openshift_job" ]]; then
  echo "ERROR: --runtime must be openshift_mock or openshift_job." >&2
  exit 2
fi

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: oc was not found on PATH. Install the OpenShift CLI and run oc login first." >&2
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

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

previous_image_ref="$(oc get deployment/conditar-gui -o jsonpath='{.spec.template.spec.containers[?(@.name=="gui")].image}' 2>/dev/null || true)"
existing_image_ref="$(oc get istag conditar-gui:latest -o jsonpath='{.image.dockerImageReference}' 2>/dev/null || true)"
current_image_ref="${previous_image_ref:-$existing_image_ref}"

cp -R openshift "$tmpdir/openshift"

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

set_build_arg_value() {
  local file="$1"
  local name="$2"
  local value="$3"
  awk -v name="$name" -v value="$value" '
    $1 == "name:" && $2 == name {
      found = 1
      print
      next
    }
    found && $1 == "value:" {
      indent = substr($0, 1, index($0, "value") - 1)
      print indent "value: \"" value "\""
      found = 0
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
    phase="$(oc get "$build_ref" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
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
  echo "  ./openshift/deploy.sh --project $PROJECT_NAME --runtime $RUNTIME --submit --cpu --runtime-image $RUNTIME_IMAGE --skip-build" >&2
  return 1
}

set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_RUNTIME" "$RUNTIME"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_DOCKER_IMAGE" "$RUNTIME_IMAGE"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_SUBMIT" "$OPENSHIFT_SUBMIT"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_DEVICE" "$OPENSHIFT_DEVICE"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_GPU_COUNT" "$OPENSHIFT_GPU_COUNT"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_CPU_REQUEST" "$OPENSHIFT_CPU_REQUEST"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_MEMORY_REQUEST" "$OPENSHIFT_MEMORY_REQUEST"
set_config_value "$tmpdir/openshift/configmap.yaml" "CONDITAR_OPENSHIFT_MEMORY_LIMIT" "$OPENSHIFT_MEMORY_LIMIT"
set_plain_value "$tmpdir/openshift/pvc.yaml" "storage" "$STORAGE"
set_build_arg_value "$tmpdir/openshift/buildconfig.yaml" "CONDITAR_GUI_TOOL_CHEST" "$GUI_TOOL_CHEST"

if [[ "$GUI_TOOL_CHEST" = "minimal" ]]; then
  echo "Building GUI with minimal tools: optional Lilly/MedChem filters will be unavailable in this image."
fi

if [[ -n "$ROUTE_HOST" ]]; then
  insert_route_host "$tmpdir/openshift/route.yaml" "$ROUTE_HOST"
fi

if [[ -n "$current_image_ref" ]]; then
  set_deployment_image "$tmpdir/openshift/deployment.yaml" "$current_image_ref"
fi

oc apply -k "$tmpdir/openshift"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "Starting OpenShift binary build from $(pwd)"
  build_ref="$(oc start-build conditar-gui --from-dir=. -o name)"
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

image_ref="$(oc get istag conditar-gui:latest -o jsonpath='{.image.dockerImageReference}' 2>/dev/null || true)"
if [[ -n "$image_ref" ]]; then
  oc set image deployment/conditar-gui "gui=$image_ref" >/dev/null
fi

oc rollout status deployment/conditar-gui

route_url="$(oc get route conditar-gui -o jsonpath='{.spec.host}' 2>/dev/null || true)"
if [[ -n "$route_url" ]]; then
  echo "conDitar GUI: https://$route_url"
else
  echo "Deployment is ready. No Route host was reported; inspect with: oc get route conditar-gui"
fi
