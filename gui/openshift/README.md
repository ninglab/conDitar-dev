# conDitar GUI on OpenShift

This folder contains the OpenShift deployment files for the conDitar GUI.
It deploys the web service, creates persistent job storage, exposes a Route,
and can launch generator pods as OpenShift Jobs. The GUI image can either be
prebuilt and supplied with `--gui-image`, or built inside the current OpenShift
project.

The GUI image installs the optional Tool Chest dependencies during build. Lilly
Medchem Rules is built from vendored source under `gui/vendor/`, so the
OpenShift build does not need to clone that external repository.

## Fast Start

Prerequisites:

- Access to an OpenShift project.
- The `oc` CLI installed and logged in with `oc login`.
- This repository checked out locally.

From the `gui/` folder:

```bash
./openshift/deploy.sh
```

On Windows PowerShell:

```powershell
.\openshift\deploy.ps1
```

To create or switch to a project first:

```bash
./openshift/deploy.sh --create-project conditar-gui-demo
```

When the script finishes, it prints the HTTPS Route for the GUI.

For a site-facing handoff path, start with
`openshift/SITE_QUICKSTART.md`.

## Prebuilt GUI Image Path

If a GUI image has already been built and pushed to a registry the project can
pull from, deploy it directly:

```bash
./openshift/deploy.sh \
  --project <site-project> \
  --runtime openshift_job \
  --submit \
  --cpu \
  --gui-image <site-conditar-gui-image> \
  --runtime-image <site-conditar-runtime-image>
```

This skips the OpenShift binary build, so the project does not need to download
GUI build dependencies during deployment.

To build that GUI image from a workstation or CI runner with Docker or Podman:

```bash
./openshift/build_gui_image.sh \
  --image docker.io/osuninglab/conditar-gui:<site-version> \
  --push
```

The build defaults to `linux/amd64`, which is the common OpenShift worker-node
platform. Set `--platform` only if the site requires a different target.

## First Click Test

1. Open the printed Route.
2. In **Where should this run?**, choose **OpenShift Job** when submission is
   enabled, or **OpenShift diagnostics** for a storage/logs-only check.
3. Upload or choose input structures.
4. Click **Generate molecules**.
5. Open the completed job and load results.

The diagnostics target writes job metadata, logs, and a small mock SDF output to
the persistent volume. It does not launch the conDitar runtime.

## OpenShift Manifest-Only Test

If `CONDITAR_OPENSHIFT_SUBMIT=false`, choose **OpenShift Job manifest** to write
a Kubernetes `Job` manifest into each job's output folder without submitting it.
This is meant to help site admins review the generator pod shape:

- conDitar runtime image.
- PVC mount and job paths.
- GPU resource key/count.
- CPU and memory requests.
- conDitar command-line arguments.

The artifact is named:

```text
outputs/conditar-openshift-job.yaml
```

## Real OpenShift Job Test

If the project allows the GUI service account to create Jobs, redeploy with:

```bash
./openshift/deploy.sh --runtime openshift_job --submit
```

For CPU-only testing, use:

```bash
./openshift/deploy.sh --runtime openshift_job --submit --cpu
```

The `--cpu` option sets the generated conDitar command to `--device cpu` and
sets `CONDITAR_OPENSHIFT_GPU_COUNT=0`, so the Job does not request a GPU.

The GUI deployment uses a `Recreate` rollout strategy because the default PVC is
`ReadWriteOnce`. This avoids briefly running two GUI pods that both try to mount
the same job-storage volume during upgrades.

## Common Options

```bash
./openshift/deploy.sh \
  --runtime openshift_mock \
  --runtime-image osuninglab/conditar-dev:2026-07-10 \
  --gui-image registry.example.edu/conditar-gui:latest \
  --storage 10Gi
```

Use `--runtime openshift_job` if the site should land on the OpenShift Job
target by default instead of the diagnostics target.

Use `--gui-image` when the GUI image is already available in a registry the
project can pull from. This skips the OpenShift binary build.

Use `--route-host name.apps.example.edu` only when the cluster allows fixed
Route hostnames.

Use `--skip-build` when the `conditar-gui:latest` ImageStreamTag already exists
and only the manifests should be re-applied.

## What Gets Created

- `ImageStream/conditar-gui`
- `BuildConfig/conditar-gui`
- `ServiceAccount/conditar-gui`
- `Role/conditar-gui-job-runner`
- `RoleBinding/conditar-gui-job-runner`
- `ConfigMap/conditar-gui-config`
- `PersistentVolumeClaim/conditar-gui-jobs`
- `Deployment/conditar-gui`
- `Service/conditar-gui`
- `Route/conditar-gui`

The GUI pod stores job data at `/data/jobs`, backed by the PVC.

## Site Settings

See `site.env.example` for configurable environment variables. The most
important ones are:

- `CONDITAR_DOCKER_IMAGE`: conDitar generator image for OpenShift Jobs and
  manifest-only checks.
- `CONDITAR_OPENSHIFT_PVC`: PVC name mounted by generated Jobs.
- `CONDITAR_OPENSHIFT_SUBMIT`: set to `true` to create Jobs from the GUI pod.
- `CONDITAR_OPENSHIFT_DEVICE`: `cuda:0` for GPU clusters or `cpu` for CPU tests.
- `CONDITAR_OPENSHIFT_GPU_RESOURCE`: GPU resource key, often `nvidia.com/gpu`.
- `CONDITAR_OPENSHIFT_GPU_COUNT`: number of GPUs requested by generated Jobs.
- `CONDITAR_OPENSHIFT_SERVICE_ACCOUNT`: optional service account for generated
  Jobs.

## Still To Confirm Before Real Execution

- Whether the GUI is allowed to create Kubernetes Jobs in the site project.
- Which service account and RBAC rules are allowed for job creation and status
  polling.
- Whether GPU resources are available and what resource key they use.
- Whether the conDitar generator image is available from the project.
- Whether the generator pod and GUI pod may share the same PVC.
- Whether the production path should use OpenShift Jobs or continue to hand off
  to Slurm.
