# conDitar-dev Container Runtime

This folder defines the Docker/Podman image used to run conDitar sampling with the same dependencies, checkpoints, and launcher across container runtimes.

The container supports:

- CPU sampling with Docker or Podman.
- NVIDIA GPU sampling with Docker `--gpus all` or Podman GPU devices.
- Optional Vina/QVina post-processing after generation.
- A single `conditar-sample` launcher for Docker and Podman runs.

The image uses CUDA-enabled PyTorch wheels, so the same image can run on CPU or GPU. You do not need a GPU to build the image or run CPU sampling. On Apple Silicon Macs, build and run the `linux/amd64` image; Docker Desktop will use emulation, which is compatible but slower.

---

## Quick Start

Build the container from the repository root:

```bash
docker/build-image.sh --checkpoint-dir /path/to/checkpoints
```

The checkpoint directory must contain:

```text
Diff.pt
PocketAE.pt
```

Run a small CPU sampling job:

```bash
INPUT_DIR=/path/to/input-data docker/run-examples.sh cpu-pocket
```

Use the example runner for other common modes:

```bash
INPUT_DIR=/path/to/input-data docker/run-examples.sh cpu-ligand
INPUT_DIR=/path/to/input-data docker/run-examples.sh gpu
INPUT_DIR=/path/to/input-data docker/run-examples.sh vina
INPUT_DIR=/path/to/input-data docker/run-examples.sh podman-cpu
```

The examples write results to `./results` by default. Set `OUTPUT_DIR=/path/to/results` to use a different folder.

---

### Scripts

- `Dockerfile` – Builds the runtime image with conDitar dependencies, checkpoints, and the container launcher.
- `build-image.sh` – Builds the image with Docker or Buildah and stages required checkpoints before the build.
- `build-export-image.sh` – Builds the image and saves it as a `.tar.gz` archive.
- `run-examples.sh` – Runs common Docker/Podman examples for CPU, GPU, Vina/QVina, and development bind mounts.
- `run-build-export-background.sh` – Runs the build/export workflow in the background.
- `qvina/` – Optional QuickVina2 staging location.

---

## Build

From the repository root:

```bash
docker/build-image.sh
```

By default, the build script expects `Diff.pt` and `PocketAE.pt` in the checkpoint directory set by `CONDITAR_CHECKPOINT_DIR`.

```text
/path/to/checkpoints/Diff.pt
/path/to/checkpoints/PocketAE.pt
```

If the checkpoints are somewhere else:

```bash
docker/build-image.sh --checkpoint-dir /path/to/checkpoints
```

The build script also stages QuickVina2 when available. Point it to the executable with `--qvina-bin` or `CONDITAR_QVINA_BIN`.

```bash
docker/build-image.sh \
  --checkpoint-dir /path/to/checkpoints \
  --qvina-bin /path/to/qvina2.1
```

Main build arguments:

- `--tag` – image tag to build. Default: `localhost/conditar-dev:container-dev`.
- `--platform` – target platform. Default: `linux/amd64`.
- `--checkpoint-dir` – folder containing `Diff.pt` and `PocketAE.pt`.
- `--qvina-bin` – optional QuickVina2 executable to include in the image.
- `--engine` – container build engine: `auto`, `docker`, or `buildah`.

Use Buildah explicitly when Docker is not the desired build engine:

```bash
docker/build-image.sh --engine buildah
```

---

## Run Examples

The runnable examples live in [`run-examples.sh`](run-examples.sh). Set `INPUT_DIR` to the host folder containing your target files, then choose a command:

```bash
INPUT_DIR=/path/to/input-data docker/run-examples.sh cpu-pocket
```

Available commands:

- `cpu-pocket` – Docker CPU run with a prepared pocket PDB.
- `cpu-ligand` – Docker CPU run with a protein PDB and reference ligand SDF.
- `gpu` – Docker NVIDIA GPU run with a prepared pocket PDB.
- `vina` – Docker CPU run with Vina/QVina post-processing enabled.
- `podman-cpu` – Podman CPU run with a prepared pocket PDB.
- `podman-gpu` – Podman GPU run with a prepared pocket PDB.
- `dev` – Docker CPU run with the live checkout bind-mounted read-only.

Default example inputs:

- `POCKET_PDB=xxxx/xxxx_pocket.pdb`
- `PROTEIN_PDB=4aua/4aua_protein.pdb`
- `LIGAND_SDF=4aua/4aua_ligand.sdf`

Override those paths when your files have different names:

```bash
INPUT_DIR=/path/to/input-data \
POCKET_PDB=my_target/pocket.pdb \
OUTPUT_DIR=/path/to/results \
docker/run-examples.sh cpu-pocket
```

Use `NUM_SAMPLES`, `BATCH_SIZE`, `VINA_MODE`, `VINA_EXHAUSTIVENESS`, and `VINA_CPU` to adjust the examples. Run `docker/run-examples.sh --help` for the full list.

---

## Launcher Arguments

The image entrypoint is `conditar-sample`.

Main arguments:

- `--pdb` / `--pocket` – protein PDB or prepared pocket PDB.
- `--sdf` / `--ligand` – optional reference ligand SDF.
- `--out` – output directory inside the container. Mount this path to keep results.
- `--config` – sampling config. Default: `/opt/conditar/app/configs/sample_container.yml`.
- `--device` – PyTorch device: `cpu`, `cuda:0`, or `auto`.
- `--num-samples` – number of molecules to generate.
- `--batch-size` – batch size used during sampling.
- `--pocket-radius` – pocket radius passed through to `scripts.conDitar.sample`.

Any unrecognized launcher options are passed through to `scripts.conDitar.sample`.

Show the full launcher help:

```bash
docker run --rm localhost/conditar-dev:container-dev --help
```

---

## Vina and QVina

Add `--vina-score` to run docking/property post-processing after sampling. The example runner includes this in:

```bash
INPUT_DIR=/path/to/input-data docker/run-examples.sh vina
```

Supported modes:

- `none` – skip docking post-processing.
- `vina_score` – run Vina score-only post-processing.
- `vina_dock` – run Vina docking/minimization.
- `qvina` – run QuickVina2.
- `all` – run Vina score/minimize plus QuickVina2.

Post-processing annotates generated SDF files with properties such as `VINA_SCORE_ONLY`, `VINA_MINIMIZE`, `QVINA`, `QED`, and `SA`. QVina is CPU-based; GPU jobs use CUDA for generation and CPU threads for docking post-processing.

---

## Export an Image Archive

Build and save a compressed image archive when you need to move the image to another machine:

```bash
CONDITAR_IMAGE_OUTPUT_DIR=/path/to/container_images docker/build-export-image.sh
```

The archive can be loaded later with Docker or Podman:

```bash
docker load -i /path/to/localhost_conditar-dev_container-dev-YYYYMMDD-HHMMSS.tar.gz
podman load -i /path/to/localhost_conditar-dev_container-dev-YYYYMMDD-HHMMSS.tar.gz
```

If the image has already been built on another machine, copy the archive instead of rebuilding it:

```bash
mkdir -p "$HOME/containers"
rsync -avP \
  <USER>@<HOST>:/path/to/container_images/localhost_conditar-dev_container-dev-YYYYMMDD-HHMMSS.tar.gz \
  "$HOME/containers/"
```

The archive is large, so `-P` allows an interrupted transfer to resume.

---

## Quick Checks

Verify core Python dependencies:

```bash
docker run --rm --entrypoint python localhost/conditar-dev:container-dev - <<'PY'
import torch
import torch_geometric
from rdkit import Chem

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
assert Chem.MolFromSmiles("CCO") is not None
PY
```

Rebuild the image when dependencies, checkpoints, or anything installed into the image changes.
