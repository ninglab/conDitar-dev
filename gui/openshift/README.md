# conDitar GUI on OpenShift

This folder contains the OpenShift deployment files for the conDitar GUI.
It deploys the web service, creates persistent job storage, exposes a Route,
and launches generator pods as OpenShift Jobs. The no-argument path pulls
three pinned, prebuilt images; it does not install packages or build an image.

The GUI image installs the optional Tool Chest dependencies during build. Lilly
Medchem Rules is built from vendored source under `gui/vendor/`, so the
OpenShift build does not need to clone that external repository.

## Fast Start

Prerequisites:

- Access to an OpenShift project.
- The `oc` CLI installed and logged in with `oc login`.
- This repository checked out locally.

From the cloned repository:

```bash
cd gui
./openshift/deploy.sh
```

From the repository root, to create or switch to a project first:

```bash
./gui/openshift/deploy.sh --create-project conditar-gui-demo
```

When the script finishes, it prints the HTTPS Route for the GUI. It uses the
current `oc` project and enables real CPU Jobs for conDitar and DiffSMol.
The pinned images live in `avery91-dev` on this OpenShift cluster. A different
project needs permission to pull from that project before running the script;
see `image-list.txt` for the exact references. An owner of `avery91-dev` can
grant its two pod service accounts access:

```bash
oc policy add-role-to-user system:image-puller system:serviceaccount:<partner-project>:conditar-gui -n avery91-dev
oc policy add-role-to-user system:image-puller system:serviceaccount:<partner-project>:default -n avery91-dev
```

The partner does not need permission to build images. The prebuilt path does
not create a BuildConfig or ImageStream. For a different cluster,
mirror all three images to a registry it can pull from and override the image
references with the three image flags or environment variables.

For a site-facing handoff path, start with
`gui/openshift/SITE_QUICKSTART.md`.

## Prebuilt GUI Image Path

If a GUI image has already been built and pushed to a registry the project can
pull from, deploy it directly:

```bash
./gui/openshift/deploy.sh \
  --project <site-project> \
  --submit \
  --cpu \
  --gui-image <site-conditar-gui-image> \
  --runtime-image <site-conditar-runtime-image> \
  --diffsmol-image <site-diffsmol-runtime-image>
```

This is also what the no-argument command does with its pinned images. The
project does not download GUI build dependencies during deployment.

To build that GUI image from a workstation or CI runner with Docker or Podman:

```bash
./gui/openshift/build_gui_image.sh \
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

The optional diagnostics target writes job metadata, logs, and a small mock
SDF output to the persistent volume. It does not launch a generator. Select it
by redeploying with `--runtime openshift_mock`.

## OpenShift Manifest-Only Test

If deployed with `--no-submit`, choose **OpenShift Job manifest** to write
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

Real CPU Job submission is the no-argument default. To override the images,
redeploy with:

```bash
./gui/openshift/deploy.sh --project <site-project> --submit \
  --runtime-image <site-conditar-runtime-image> \
  --diffsmol-image <site-diffsmol-runtime-image>
```

To force CPU when overriding site GPU settings, use:

```bash
./gui/openshift/deploy.sh --project <site-project> --submit --cpu \
  --runtime-image <site-conditar-runtime-image> \
  --diffsmol-image <site-diffsmol-runtime-image>
```

The `--cpu` option sets the generated conDitar command to `--device cpu` and
sets `CONDITAR_OPENSHIFT_GPU_COUNT=0`, so the Job does not request a GPU.
`--submit` selects the real OpenShift Job target. `--no-submit` writes manifests
only.
The `--diffsmol-image` setting must point to the updated standalone DiffSMol
image described in `../../diffsmol/README.md`. That image includes pocket
generation and selected evaluations; the GUI submits one Job running both
stages before results become available. The image must be pushed to a registry
the project can pull from. Site validation is still required for GPU access
and cross-project image pulls. `--cpu` uses the same image without requesting
a GPU. The old `ninglab/diffsmol:latest` image is not suitable for DiffSMol Jobs.

For exact conDitar evaluators, `--runtime-image` must likewise point to a
registry copy of the refreshed conDitar image described in
`../../docker/README.md`. Real submission rejects the older `2026-07-10`
image and local-only tags. Both engine images run their selected evaluations
before the OpenShift Job completes. The Vina preprocessing panel also runs as
an OpenShift Job using the conDitar image, so no Docker socket is needed in
the GUI pod. The Route sets a 35-minute server timeout for this synchronous
panel request; if the site has an external load balancer, its timeout must also
be long enough.

The GUI deployment uses a `Recreate` rollout strategy because the default PVC is
`ReadWriteOnce`. This avoids briefly running two GUI pods that both try to mount
the same job-storage volume during upgrades.

## Common Options

```bash
./gui/openshift/deploy.sh \
  --runtime openshift_mock \
  --storage 10Gi
```

Use `--runtime openshift_mock` only for synthetic infrastructure diagnostics.

Use `--gui-image` when the GUI image is already available in a registry the
project can pull from. This skips the OpenShift binary build.

Use `--route-host name.apps.example.edu` only when the cluster allows fixed
Route hostnames.

Use `--build-gui` only if the site deliberately wants an OpenShift binary build;
that build downloads the GUI's conda dependencies inside the builder. Use
`--build-gui --skip-build` to deploy a completed ImageStream build after a
terminal or network interruption.

## What Gets Created

- `ServiceAccount/conditar-gui`
- `Role/conditar-gui-job-runner`
- `RoleBinding/conditar-gui-job-runner`
- `ConfigMap/conditar-gui-config`
- `PersistentVolumeClaim/conditar-gui-jobs`
- `Deployment/conditar-gui`
- `Service/conditar-gui`
- `Route/conditar-gui`

Only `--build-gui` also creates `ImageStream/conditar-gui` and
`BuildConfig/conditar-gui`.

The GUI pod stores job data at `/data/jobs`, backed by the PVC.

## Site Settings

See `site.env.example` for configurable environment variables. The most
important ones are:

- `CONDITAR_DOCKER_IMAGE`: conDitar generator image for OpenShift Jobs and
  manifest-only checks.
- `CONDITAR_OPENSHIFT_PVC`: PVC name mounted by generated Jobs.
- `CONDITAR_OPENSHIFT_SUBMIT`: defaults to `true` for real Jobs.
- `CONDITAR_OPENSHIFT_DEVICE`: defaults to `cpu`; set `cuda:0` for GPU clusters.
- `CONDITAR_OPENSHIFT_GPU_RESOURCE`: GPU resource key, often `nvidia.com/gpu`.
- `CONDITAR_OPENSHIFT_GPU_COUNT`: number of GPUs requested by generated Jobs.
- `CONDITAR_OPENSHIFT_SERVICE_ACCOUNT`: optional service account for generated
  Jobs.

## Still To Confirm Before Real Execution

- Whether the GUI is allowed to create Kubernetes Jobs in the site project.
- Which service account and RBAC rules are allowed for job creation and status
  polling.
- Whether GPU resources are available and what resource key they use.
- Whether all three pinned images are pullable from the partner project.
- Whether the generator pod and GUI pod may share the same PVC.
- Whether the production path should use OpenShift Jobs or continue to hand off
  to Slurm.
