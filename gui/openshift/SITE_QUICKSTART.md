# conDitar GUI OpenShift Site Quickstart

This branch contains the conDitar GUI plus OpenShift setup files.

The simplest path is:

1. Open the OpenShift Web Terminal.
2. Clone the approved Git repository.
3. Put your OpenShift project name into one deploy command and run it.
4. Open the GUI Route/URL printed by the script.

No Docker Desktop install is required.
No local Git, Helm, Kustomize, or Docker Desktop install is required if the
OpenShift Web Terminal is available.

## What This Deploys

- A conDitar GUI web service.
- A persistent job-storage PVC mounted at `/data/jobs`.
- A Route for browser access.
- RBAC that lets the GUI create and poll Kubernetes Jobs.
- In-cluster OpenShift Job submission for generator runs when enabled with
  `--submit`.

The validated path is:

1. User submits a job in the GUI.
2. GUI writes inputs and a Kubernetes Job manifest under `/data/jobs`.
3. GUI submits the Job through the in-cluster Kubernetes API.
4. Generator pod mounts the same PVC, writes outputs, and exits.
5. GUI polls Job/pod status, captures pod logs, and loads SDF results.

## What You Need

- Access to an OpenShift project/namespace.
- OpenShift Web Terminal access, or the OpenShift command-line tool, `oc`.
- Your OpenShift project name.
- Permission to create the objects in `openshift/`, including:
  - `Deployment`
  - `Service`
  - `Route`
  - `PersistentVolumeClaim`
  - `ServiceAccount`
  - `Role`
  - `RoleBinding`
  - `Job`
  - `Pod/log` reads
- A conDitar runtime image available to the project.

If the site cannot run OpenShift binary builds, build and push the GUI image to
an approved registry first, then deploy with `--gui-image`.

The GUI image can be built from a workstation or CI runner with Docker or
Podman:

```bash
./openshift/build_gui_image.sh \
  --image docker.io/osuninglab/conditar-gui:<site-version> \
  --push
```

If the site uses an internal registry, mirror that image there and use the
internal image reference in the deploy command.

## Preferred Path: OpenShift Web Terminal

If the OpenShift Web Terminal is available, use that first. It avoids installing
OpenShift tools on a local workstation.

1. Log into the OpenShift web console in a browser.
2. Open the Web Terminal.
3. Confirm `oc` is available:

```bash
oc version --client
```

4. Confirm your current project:

```bash
oc project
```

5. Clone the approved repository branch:

```bash
git clone --branch <branch-name> <repo-url>
cd <repo-folder>
```

Use the repository URL and branch name approved for the site environment.

## Get The OpenShift Command

If using OpenShift Web Terminal, skip this section.

If using a local terminal, download `oc` from OpenShift:

1. Log into the OpenShift web console in a browser.
2. Look for **Command line tools** or **Copy login command**. This is often in
   the user menu or help menu.
3. Download the OpenShift CLI for your operating system.
4. Unzip it.
5. Open a terminal in that folder and check:

Windows PowerShell:

```powershell
.\oc.exe version --client
```

macOS/Linux:

```bash
./oc version --client
```

Then copy the full login command from OpenShift and run it. It looks like:

```bash
oc login --token=... --server=...
```

Do not email or share the token. It is personal to your OpenShift account.

## Runtime Image

The runtime image is the conDitar generator container launched by each
OpenShift Job. It is separate from the GUI image.

Start with the public runtime image if the cluster can pull from Docker Hub:

```text
docker.io/osuninglab/conditar-dev:2026-07-10
```

If the cluster cannot pull external images, ask the site admin to mirror that
image into a registry the project can access. In that case, replace
`<site-conditar-runtime-image>` with the mirrored image reference, for example:

```text
image-registry.openshift-image-registry.svc:5000/<site-project>/conditar-runtime:2026-07-10
```

## Fast CPU Validation With A Prebuilt GUI Image

Use this path when a GUI image is already available from a registry your
OpenShift project can pull from. It skips the OpenShift build step.

From the cloned repository folder:

```bash
oc project <site-project>

./openshift/deploy.sh \
  --project <site-project> \
  --runtime openshift_job \
  --submit \
  --cpu \
  --gui-image <site-conditar-gui-image> \
  --runtime-image docker.io/osuninglab/conditar-dev:2026-07-10
```

Replace `<site-project>` with your OpenShift project name. Replace
`<site-conditar-gui-image>` with the GUI image approved for your environment.
Keep the runtime image unchanged unless your site admin gives you an internal
mirror for the runtime image.

The script prints the Route when the deployment is ready.

## Fast CPU Validation With OpenShift Build

From the cloned repository folder:

In OpenShift Web Terminal:

```bash
oc project <site-project>

./openshift/deploy.sh \
  --project <site-project> \
  --runtime openshift_job \
  --submit \
  --cpu \
  --runtime-image docker.io/osuninglab/conditar-dev:2026-07-10
```

On Windows PowerShell:

```powershell
# Paste the full oc login command from OpenShift first:
# oc login --token=... --server=...
oc project <site-project>

.\openshift\deploy.ps1 `
  -Project <site-project> `
  -Runtime openshift_job `
  -Submit `
  -Cpu `
  -RuntimeImage docker.io/osuninglab/conditar-dev:2026-07-10
```

On macOS/Linux/Git Bash:

```bash
# Paste the full oc login command from OpenShift first:
# oc login --token=... --server=...
oc project <site-project>

./openshift/deploy.sh \
  --project <site-project> \
  --runtime openshift_job \
  --submit \
  --cpu \
  --runtime-image docker.io/osuninglab/conditar-dev:2026-07-10
```

Replace only `<site-project>` with your OpenShift project name for the first
test. Keep the runtime image value unchanged unless your site admin tells you
Docker Hub pulls are blocked.

The script prints the Route when the deployment is ready.

If the Web Terminal disconnects while the GUI image is building, log back into
the Web Terminal, return to the repository folder, and finish from the latest
completed image:

```bash
cd conditar_gui_dev

./openshift/deploy.sh \
  --project <site-project> \
  --runtime openshift_job \
  --submit \
  --cpu \
  --runtime-image docker.io/osuninglab/conditar-dev:2026-07-10 \
  --skip-build
```

For a first infrastructure test in the GUI:

- Target: `OpenShift Job`
- Samples: `2`
- Batch size: `100`
- Vina: off
- Optional tools: Lilly Medchem Rules and MedChem Filters

Expected result:

- Job status changes from `running` to `completed`.
- Logs show manifest creation, Job submission, and sampling progress.
- Results show generated SDF output.
- The generated manifest is saved as
  `/data/jobs/<job-id>/outputs/conditar-openshift-job.yaml`.

## GPU Follow-Up

After CPU validation, confirm the site GPU resource details and redeploy
without `--cpu`, using environment variables as needed:

```bash
export CONDITAR_OPENSHIFT_GPU_RESOURCE=nvidia.com/gpu
export CONDITAR_OPENSHIFT_GPU_COUNT=1
export CONDITAR_OPENSHIFT_DEVICE=cuda:0
export CONDITAR_OPENSHIFT_CPU_REQUEST=2
export CONDITAR_OPENSHIFT_MEMORY_REQUEST=16Gi
export CONDITAR_OPENSHIFT_MEMORY_LIMIT=32Gi

./openshift/deploy.sh \
  --project <site-project> \
  --runtime openshift_job \
  --submit \
  --runtime-image <site-conditar-runtime-image>
```

## Storage Notes

The default PVC is `ReadWriteOnce`. The generated generator Jobs include
required pod affinity so they land on the same node as the GUI pod, which allows
the GUI and generator pod to share the same RWO volume.

If the site provides `ReadWriteMany` storage, the same package should still
work. The affinity can be relaxed later if they prefer scheduling flexibility.

## Useful Checks

```bash
oc get deployment conditar-gui
oc get pods -l app=conditar-gui -o wide
oc get route conditar-gui
oc get pvc conditar-gui-jobs
oc get jobs
```

For a specific generated Job:

```bash
oc get job <job-name> -o wide
oc get pods -l job-name=<job-name> -o wide
oc logs job/<job-name>
```

## Known Validation Notes

- The runtime image can be large, so first pull on a fresh node may take several
  minutes.
- Long `oc logs -f` sessions can drop if local credentials expire or the network
  resets; this does not imply the OpenShift Job failed.
- Lilly Medchem Rules is vendored under `gui/vendor/` and built into the GUI
  image without cloning that external repository during the OpenShift build.
- MedChem filters are installed through the GUI conda environment.
- Final production values still need site confirmation for storage class,
  route/TLS policy, image registry, GPU resource name, and build policy.

## Files To Review

- `openshift/README.md`: deployment options and object list.
- `openshift/site.env.example`: tunable environment variables.
- `openshift/deploy.sh`: deploy/build script.
