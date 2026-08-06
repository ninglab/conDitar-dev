# conDitar GUI

A lightweight browser GUI for running conDitar molecular generation jobs.

This folder contains the conDitar browser GUI. It stages user-selected PDB/SDF
inputs, launches conDitar jobs through the container image built from
`conDitar-dev`, tracks job status, and reads generated SDF outputs.

The generator, container, and GUI are documented separately:

- [`../README.md`](../README.md) — model, sampling, and repository overview.
- [`../docker/README.md`](../docker/README.md) — image build, archive transfer,
  Docker/Podman usage, and container-only runs.

![conDitar GUI overview](media/screenshots/gui-overview.png)

## Architecture overview

```text
conDitar-dev/gui
  Browser GUI + Python backend
  Starts jobs, tracks logs/status, reads generated SDF outputs

conDitar-dev container/source
  Docker/Podman image with conDitar code, dependencies, model files, and runtime entry point
  Default image name: averyemeyer/conditar-dev:2026-07-10
```

Typical local CPU flow:

1. Pull the `conDitar-dev` runtime image.
2. Run `./setup_gui.sh` to check Python, Docker, the image, and optional tools.
3. Start this GUI folder with `./start_cpu_gui.sh`.
4. Open the local URL printed by the launcher.
5. Choose inputs, settings, and target, then click **Generate molecules**.

Job folders are written under `job_data/jobs/<job-id>/`. That directory is
ignored by git and contains staged inputs, logs, metadata, generated SDFs, and
export ZIPs.

## Quick start

For first-time local CPU setup with Docker or Docker Desktop:

```bash
docker pull averyemeyer/conditar-dev:2026-07-10
git clone https://github.com/ninglab/conDitar-dev.git
cd conDitar-dev/gui
./setup_gui.sh
./start_cpu_gui.sh
```

The launcher normally opens `http://127.0.0.1:4173`. If that port is busy, it
automatically tries the next available port and prints the URL to use.

## Local CPU startup

Requirements:

- Git
- Python 3.9 or newer
- Docker Desktop
- The pulled conDitar image `averyemeyer/conditar-dev:2026-07-10`

Linux, macOS, and Windows through WSL2 are supported for local GUI use. On
Windows, run the shell scripts from WSL2, not native PowerShell. Install Docker
Desktop, enable its WSL2 integration, and leave Docker Desktop running while
you use the GUI. On macOS, you can also double-click `START_HERE_MAC.command`
from Finder after Docker Desktop is installed and running.

### Windows CPU setup

1. Install Docker Desktop and WSL2.
2. In Docker Desktop, open **Settings > Resources > WSL Integration** and turn
   on integration for the WSL distribution you will use.
3. Install Miniconda or Miniforge inside WSL, then open a fresh WSL terminal.
4. Clone the repository inside WSL or put the repository somewhere WSL can read,
   such as your WSL home directory.
5. Run the same local CPU commands:

   ```bash
   docker pull averyemeyer/conditar-dev:2026-07-10
   git clone https://github.com/ninglab/conDitar-dev.git
   cd conDitar-dev/gui
   ./setup_gui.sh
   ./start_cpu_gui.sh
   ```

The Setup page includes a **Launch checklist** that reports whether Python,
Docker/Podman, the conDitar image, Slurm, and optional Tool Chest tools are
available. If Docker commands fail inside WSL, first confirm Docker Desktop is
open and WSL integration is enabled for that distribution.

If port `4173` is already in use, the launchers automatically try the next
available port and print the URL they selected. Use the printed URL in your
browser. To request a specific starting port:

```bash
PORT=4174 ./start_cpu_gui.sh
```

On Windows/WSL, seeing "port already in use" usually means another conDitar GUI
terminal or browser session is still running. You can use that existing window,
close the old terminal, or rerun the launcher and follow the newly printed URL.

## Slurm GPU startup

Fresh Slurm/GPU setup:

1. Copy or clone this repository onto the cluster and enter the GUI folder.
2. Make the Docker/OCI image available to compute nodes. If your cluster allows
   registry pulls, preload the image with Podman:

   ```bash
   podman pull docker.io/averyemeyer/conditar-dev:2026-07-10
   ```

   If compute nodes cannot pull from Docker Hub, place the exported
   `.tar`/`.tar.gz` archive on a filesystem visible from compute nodes.
3. Configure the scheduler account in a local `.conditar-slurm.env` file when
   your cluster requires one (this file is ignored and should not be committed):

   ```bash
   CONDITAR_SLURM_ACCOUNT=your_account
   # CONDITAR_SLURM_PARTITION=your_gpu_partition   # if required by your site
   ```

   If compute nodes cannot pull from Docker Hub, also set
   `CONDITAR_DOCKER_TAR=/shared/path/conditar-image.tar.gz`. The archive path
   must resolve from the compute node, not only from the login host. If
   `CONDITAR_DOCKER_TAR` is unset, the launcher checks common nearby
   `conditar*.tar`/`.tar.gz` locations such as the GUI folder, `../containers/`,
   and `$HOME/containers/`.
4. Confirm the cluster tools are available (`python3` or `conda`, `podman`, and `sbatch`),
   then run the GPU setup check and start the GUI:

   ```bash
   ./setup_slurm_gui.sh
   ./start_slurm_gui.sh
   ```

   If no image or archive is detected, `setup_slurm_gui.sh` prompts for the
   compute-node-visible archive path and saves it in `.conditar-slurm.env`.

5. Open the printed GUI URL, choose **Slurm GPU · Podman**, enter/confirm the
   Slurm account, and click **Check again** in Launch readiness before submitting.

The launcher normally opens `http://127.0.0.1:4173`; if that port is busy, it
prints the next available local URL. It validates the image/archive and required
commands before starting.
Each submitted GPU batch is sent to Slurm as an array job; scheduler delays or
account/GPU limits are reported in the Jobs panel with the scheduler reason.

Requirements:

- A cluster session with Slurm available
- Podman available on the login or compute environment
- The conDitar image available as `docker.io/averyemeyer/conditar-dev:2026-07-10`, or a
  shared image archive that can be loaded by the Slurm job
- Any site-specific setup required for remote desktop or web access

The Slurm launcher defaults to:

```bash
CONDITAR_RUNTIME=podman
CONDITAR_DOCKER_IMAGE=docker.io/averyemeyer/conditar-dev:2026-07-10
CONDITAR_DOCKER_TAR=                  # optional archive to load inside the job
CONDITAR_SLURM_ACCOUNT=               # required by many Slurm sites
CONDITAR_SLURM_TIME=04:00:00
CONDITAR_SLURM_MEM=32G
CONDITAR_SLURM_CPUS=4
CONDITAR_SLURM_GPUS=1
```

Override any default inline when needed:

```bash
CONDITAR_SLURM_PARTITION=nextgen \
CONDITAR_SLURM_TIME=08:00:00 \
./start_slurm_gui.sh
```

In the GUI, enter your required Slurm account and choose
**Slurm GPU · Podman** under **Where should this run?** before submitting. The
backend writes `run.slurm`, submits with `sbatch`, and polls Slurm/log files
until outputs are ready.

## Runtime options

The GUI chooses the container runner from environment variables:

- `CONDITAR_RUNTIME=docker` for local Docker Desktop.
- `CONDITAR_RUNTIME=podman` for Linux/cluster Podman.
- `CONDITAR_RUNTIME=auto` to select an available local Docker/Podman runtime.

For local CPU defaults that should not be committed, put environment assignments
in `.conditar-cpu.env`; `start_cpu_gui.sh` loads it automatically.

Use a different image name:

```bash
CONDITAR_DOCKER_IMAGE=my-registry/conditar-dev:tag \
CONDITAR_RUNTIME=docker \
./start_cpu_gui.sh
```

Use a local `conDitar-dev` checkout while keeping the same container
environment/checkpoints:

```bash
CONDITAR_SOURCE_MOUNT=/path/to/conDitar-dev \
CONDITAR_RUNTIME=docker \
./start_cpu_gui.sh
```

This is useful for source-only conDitar edits. Rebuild the container when
dependencies, model/checkpoint files, or container setup changes. The launchers
automatically use the parent `conDitar-dev` checkout as `CONDITAR_SOURCE_MOUNT`
when the GUI is stored at `conDitar-dev/gui`.

If Docker or Podman is installed in a nonstandard location:

```bash
DOCKER_BIN=/path/to/docker ./start_cpu_gui.sh
PODMAN_BIN=/path/to/podman ./start_slurm_gui.sh
```

To copy a shared image archive from a remote cluster to your local machine, run
`rsync` from your local terminal. Replace the placeholders with your cluster
username, login host, and archive path:

```bash
mkdir -p "$HOME/containers"
rsync -avP \
  <CLUSTER_USER>@<CLUSTER_LOGIN_HOST>:/path/to/conditar-dev__2026-07-10.tar.gz \
  "$HOME/containers/"
docker load -i "$HOME/containers/conditar-dev__2026-07-10.tar.gz"
```

The archive is large; `rsync -P` resumes an interrupted transfer. Local NVIDIA
GPU execution additionally requires Docker Desktop GPU support and a compatible
NVIDIA runtime. For normal GPU throughput, use the Slurm/Podman path.

## Using the GUI

1. Choose **Protein + reference ligand** or **Pocket only**.
2. Upload a PDB file; reference mode also requires an SDF ligand.
3. Set **Molecules**, **Batch size**, and **Pocket radius**.
4. Choose **This computer · CPU** or **Slurm GPU · Podman**.
5. Enable Vina scoring if desired, then review Slurm options when using the GPU target.
6. Click **Generate molecules**.
7. Use the **Jobs** tab to monitor status and load completed outputs.
8. Use the **Results** and **Export** tabs to inspect molecules, filter
   candidates, and download SDF/CSV/ZIP artifacts.

CPU email notifications are intentionally disabled in the GUI until a local
SMTP/sendmail path is configured. Slurm GPU jobs can use scheduler email notifications
when an email address is provided.

Filtered exports are saved both by the browser and, for completed backend jobs,
under the job folder at `job_data/jobs/<job-id>/filtered_exports/`. Each
filtered export includes copied SDFs, `metrics.csv`, and `export_metadata.json`
with the active thresholds and tool runs used for that subset.

If a Slurm job is `PENDING`, the scheduler has accepted it but is waiting for account,
partition, or GPU capacity. If it fails before producing container output,
inspect `logs/sbatch.stderr.log` and `logs/stderr.log`; a missing image archive
or unavailable image indicates that the GPU launcher was not used, the image
was not pulled, or the archive path is incorrect.

## Batch folders

The GUI can accept folders of paired inputs.

- Local CPU batches become one job per folder in a serial local worker queue.
- Slurm GPU batches submit one Slurm array with one task per folder, allowing
  Slurm to run them in parallel subject to account, partition, and GPU
  availability.

The browser never passes arbitrary client filesystem paths into the container.
Uploaded files are copied into each job's private `inputs/` directory first.

## Tool Chest

Completed jobs can be annotated with optional molecule-evaluation tools from the
Results tab. Tool runs write logs and summaries under
`job_data/jobs/<job-id>/tool_runs/` and can add new SDF properties that appear
in the table, selected-molecule details, and CSV export.

Tool Chest evaluators can also be selected before submission from
**Advanced run settings** under **Evaluate molecules**. Those evaluators run on
the GUI backend after conDitar generation completes, without rebuilding the
conDitar sampling image.

Included tools currently cover Lilly Medchem Rules plus the medchem tutorial
filter set: Ro5, Ghose, Veber, ZINC, BMS alerts, PAINS alerts, SureChEMBL
alerts, NIBR, complexity, Bredt, molecular graph, and Lilly demerit. Their
dependencies are listed in `gui/environment.yml`, keeping this post-processing
layer separate from the conDitar sampling image. Users can add or update GUI
tools without rebuilding the model container.

The basic GUI only needs Python because structure viewing runs in the browser
with JavaScript libraries. Tool Chest evaluators run on the GUI backend, so
tools with command-line dependencies need those dependencies in the GUI
environment. To enable the included tools, run once:

```bash
./setup_tool_chest.sh
```

After that, `./start_cpu_gui.sh` and `./start_slurm_gui.sh` automatically use
the `conditar-gui-dev` environment when it is available. On macOS,
`START_HERE_MAC.command` will also try to create/update that optional
environment before launching. Without that environment, the GUI still starts
with system Python and marks missing optional tools as unavailable.

The GUI still starts if optional tools are missing; unavailable tools are shown
disabled until their command-line dependency is available in the GUI
environment. See [`tools/README.md`](tools/README.md) for the plug-in contract
for adding custom evaluators.

## Vina post-processing

Vina scoring is optional and lives in the run setup controls. When enabled,
the backend adds Vina arguments to the same container/job after generation. The
Results page reads SDF properties dynamically and can display/export properties
such as:

```text
VINA_SCORE_ONLY
VINA_MINIMIZE
VINA_DOCK
QVINA
QED
SA
```
