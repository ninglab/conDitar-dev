# conDitar GUI OpenShift Site Quickstart

This branch contains the conDitar GUI plus OpenShift setup files.

The simplest path is:

1. Open the OpenShift Web Terminal.
2. Clone the approved Git repository.
3. Enter `gui/` and run `./openshift/deploy.sh` in the current `oc` project.
4. Open the GUI Route/URL printed by the script.

No local Helm, Kustomize, Docker Desktop, or Python install is required when
using the OpenShift Web Terminal; it provides `git` and `oc`.
The default path pulls three pinned images from `avery91-dev` and runs real CPU
Jobs. It does not build images or install Python packages in the terminal.

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
- Permission for the project's `conditar-gui` and `default` service accounts to
  pull the three pinned images from `avery91-dev` on this same cluster. An
  owner of `avery91-dev` must grant this once for another project; see
  `README.md`. On a different cluster, mirror the images and override the refs.

If the site cannot pull from `avery91-dev`, mirror the prebuilt images to an
approved registry and pass their references to the launcher.

The GUI image can be built from a workstation or CI runner with Docker or
Podman:

```bash
./gui/openshift/build_gui_image.sh \
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

5. Clone the approved repository branch and deploy:

```bash
git clone --branch <branch-name> <repo-url>
cd <repo-folder>/gui
./openshift/deploy.sh
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

## Engine Images

The conDitar and DiffSMol runtime images are separate from the GUI image.
The no-argument launcher uses the pinned images listed in `image-list.txt`.
For a different OpenShift cluster, publish or mirror all three images to a
registry the project can pull from. The older
`osuninglab/conditar-dev:2026-07-10` image lacks the selected evaluation
support, and `ninglab/diffsmol:latest` is not the refreshed standalone image.
The deploy script rejects those older images in `--submit` mode. Building the
GUI inside OpenShift does not automatically publish either engine image.

If overriding the defaults, get approved references from the site
administrator, for example:

```text
<registry>/<site-project>/conditar-runtime:<version>
<registry>/<site-project>/diffsmol-runtime:<version>
```

Private registries may also require an image pull secret on the project's
default service account. Do not put registry credentials in this repository.

## Optional: Infrastructure Diagnostics

To check the Route and PVC with synthetic results and no engine Job:

```bash
./openshift/deploy.sh --runtime openshift_mock
```

A diagnostics job does not run a generator and its SDFs are synthetic.

## Default CPU Deployment

This is the partner path after image-pull access has been granted. It skips
the OpenShift build step.

From the cloned repository root:

```bash
cd gui
./openshift/deploy.sh
```

The script uses the current `oc` project. To use mirrored images instead, set
`CONDITAR_GUI_IMAGE`, `CONDITAR_DOCKER_IMAGE`, and `DIFFSMOL_DOCKER_IMAGE` or
pass `--gui-image`, `--runtime-image`, and `--diffsmol-image`.

The script prints the Route when the deployment is ready.

## Optional: OpenShift GUI Build

Only use this if the site explicitly permits building and downloading conda
dependencies inside an OpenShift builder. It is not part of the partner path.

From the cloned repository root in OpenShift Web Terminal:

```bash
./gui/openshift/deploy.sh \
  --project <site-project> \
  --submit \
  --cpu \
  --build-gui \
  --runtime-image <site-conditar-runtime-image> \
  --diffsmol-image <site-diffsmol-runtime-image>
```

On Windows PowerShell:

```powershell
# Paste the full oc login command from OpenShift first:
# oc login --token=... --server=...
.\gui\openshift\deploy.ps1 `
  -Project <site-project> `
  -Submit `
  -Cpu `
  -RuntimeImage <site-conditar-runtime-image> `
  -DiffsmolImage <site-diffsmol-runtime-image>
```

On macOS/Linux/Git Bash:

```bash
# Paste the full oc login command from OpenShift first:
# oc login --token=... --server=...
./gui/openshift/deploy.sh \
  --project <site-project> \
  --submit \
  --cpu \
  --build-gui \
  --runtime-image <site-conditar-runtime-image> \
  --diffsmol-image <site-diffsmol-runtime-image>
```

Replace the three placeholders with the project name and two registry image
references. These examples opt into building the GUI image.

The script prints the Route when the deployment is ready.

If the Web Terminal disconnects while the GUI image is building, log back into
the Web Terminal, return to the repository folder, and finish from the latest
completed image:

```bash
cd ~/conDitar-dev

./gui/openshift/deploy.sh \
  --project <site-project> \
  --submit \
  --cpu \
  --build-gui \
  --runtime-image <site-conditar-runtime-image> \
  --diffsmol-image <site-diffsmol-runtime-image> \
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

./gui/openshift/deploy.sh \
  --project <site-project> \
  --submit \
  --runtime-image <site-conditar-runtime-image> \
  --diffsmol-image <site-diffsmol-runtime-image>
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
