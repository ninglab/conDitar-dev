# conDitar-dev Docker Container

This image is the supported runtime for local Docker and OSC Podman. It keeps the same launcher behavior:

- `CONDITAR_DEVICE=cpu` or `--device cpu` for local CPU.
- `CONDITAR_DEVICE=cuda:0` or `--device cuda:0` plus Docker `--gpus all` for NVIDIA GPU.
- `CONDITAR_DEVICE=auto` uses CUDA when visible and falls back to CPU.

The image uses CUDA-enabled PyTorch wheels so one image can run on GPU or CPU. On Apple Silicon Macs, build/run as `linux/amd64`; Docker Desktop will use emulation, so it is compatible but slow.

## Build

From the repository root:

```bash
docker/build-image.sh
```

On OSC, use Buildah explicitly. This matches OSC's Docker/Podman build guidance and avoids Docker Desktop assumptions:

```bash
docker/build-image.sh --engine buildah
```

The helper stages the trained checkpoints into `docker/checkpoints/` before building. By default it expects:

```text
/path/to/checkpoints/Diff.pt
/path/to/checkpoints/PocketAE.pt
```

If the checkpoints are somewhere else:

```bash
docker/build-image.sh --checkpoint-dir /path/to/checkpoints
```

The helper also stages QuickVina2 into the image when available. By default it checks:

```text
/path/to/qvina2.1
```

If the QuickVina2 binary is somewhere else:

```bash
docker/build-image.sh --qvina-bin /path/to/qvina2.1
```

Inside the image, QVina is exposed as:

```text
CONDITAR_QVINA_BIN=/opt/conditar/qvina/qvina2.1
```

Override the tag or platform:

```bash
docker/build-image.sh --tag localhost/conditar-dev:latest --platform linux/amd64
```

## Run On Local CPU

Set `INPUT_DIR` to the folder containing `xxxx/xxxx_pocket.pdb` and/or `4aua/4aua_protein.pdb`. On OSC, use:

```bash
INPUT_DIR=/path/to/input-data
```

On a local machine, use the path where you copied or cloned the examples.

```bash
mkdir -p results

docker run --rm \
  -e CONDITAR_DEVICE=cpu \
  -v "${INPUT_DIR:?set INPUT_DIR to your examples/input directory}":/inputs:ro \
  -v "$PWD/results":/results \
  localhost/conditar-dev:container-dev \
  --pdb /inputs/xxxx/xxxx_pocket.pdb \
  --out /results \
  --device cpu \
  --num-samples 1 \
  --batch-size 1
```

With a reference ligand:

```bash
docker run --rm \
  -e CONDITAR_DEVICE=cpu \
  -v "${INPUT_DIR:?set INPUT_DIR to your examples/input directory}":/inputs:ro \
  -v "$PWD/results":/results \
  localhost/conditar-dev:container-dev \
  --pdb /inputs/4aua/4aua_protein.pdb \
  --sdf /inputs/4aua/4aua_ligand.sdf \
  --out /results \
  --device cpu \
  --num-samples 1 \
  --batch-size 1
```

Add docking/chemistry post-processing by passing `--vina-score`. This runs
after sampling inside the same container and annotates each generated SDF with
properties such as `VINA_SCORE_ONLY`, `VINA_MINIMIZE`, `QVINA`, `QED`, and `SA`.
Supported modes are `none`, `vina_score`, `vina_dock`, `qvina`, and `all`:

```bash
docker run --rm \
  -e CONDITAR_DEVICE=cpu \
  -v "${INPUT_DIR:?set INPUT_DIR to your examples/input directory}":/inputs:ro \
  -v "$PWD/results":/results \
  localhost/conditar-dev:container-dev \
  --pdb /inputs/4aua/4aua_protein.pdb \
  --sdf /inputs/4aua/4aua_ligand.sdf \
  --out /results \
  --device cpu \
  --num-samples 1 \
  --batch-size 1 \
  --vina-score \
  --vina-mode vina_score \
  --vina-exhaustiveness 8 \
  --vina-cpu 4
```

Use `--vina-mode qvina` for QuickVina2 only, or `--vina-mode all` for Vina
score/minimize plus QVina. QVina is CPU-based; GPU jobs use CUDA for generation
and CPU threads for docking post-processing.

## Run On OSC With Podman

OSC's Docker-compatible runtime is Podman/Buildah. For Slurm jobs, use a shared archive so compute nodes can load the image:

```bash
IMAGE_TAR=/fs/ess/PCON0041/mey200/container_images/localhost_conditar-dev_container-dev-20260710-105038.tar.gz
podman load -i "$IMAGE_TAR"
podman image exists localhost/conditar-dev:container-dev
```

The GUI's `start_gpu_gui.sh` and generated Slurm scripts perform this load when
the image is missing. Do not let Podman fall through to a registry pull for the
`localhost/...` image.

CPU:

```bash
mkdir -p results
INPUT_DIR=/path/to/input-data

podman run --rm \
  -e CONDITAR_DEVICE=cpu \
  -v "$INPUT_DIR":/inputs:ro \
  -v "$PWD/results":/results \
  localhost/conditar-dev:container-dev \
  --pdb /inputs/xxxx/xxxx_pocket.pdb \
  --out /results \
  --device cpu \
  --num-samples 1 \
  --batch-size 1
```

GPU:

```bash
salloc -n 1 -G 1
mkdir -p results
INPUT_DIR=/path/to/input-data

podman run --rm --device nvidia.com/gpu=all \
  -e CONDITAR_DEVICE=cuda:0 \
  -v "$INPUT_DIR":/inputs:ro \
  -v "$PWD/results":/results \
  localhost/conditar-dev:container-dev \
  --pdb /inputs/xxxx/xxxx_pocket.pdb \
  --out /results \
  --device cuda:0 \
  --num-samples 1 \
  --batch-size 1
```

The same `--vina-score` flags can be added to Podman GPU runs. Generation uses
CUDA when requested; Vina post-processing uses CPU threads inside the same
container/job.

## Run On NVIDIA GPU

```bash
mkdir -p results

docker run --rm --gpus all \
  -e CONDITAR_DEVICE=cuda:0 \
  -v "${INPUT_DIR:?set INPUT_DIR to your examples/input directory}":/inputs:ro \
  -v "$PWD/results":/results \
  localhost/conditar-dev:container-dev \
  --pdb /inputs/xxxx/xxxx_pocket.pdb \
  --out /results \
  --device cuda:0 \
  --num-samples 10
```

## Quick Checks

```bash
docker run --rm localhost/conditar-dev:container-dev --help

docker run --rm --entrypoint python localhost/conditar-dev:container-dev - <<'PY'
import torch
import torch_geometric
from rdkit import Chem
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
assert Chem.MolFromSmiles("CCO") is not None
PY
```

## Development

For quick code/config iteration without rebuilding, bind the live checkout over the image app directory:

```bash
docker run --rm \
  -e CONDITAR_DEVICE=cpu \
  -v "$PWD":/opt/conditar/app:ro \
  -v "${INPUT_DIR:?set INPUT_DIR to your examples/input directory}":/inputs:ro \
  -v "$PWD/results":/results \
  localhost/conditar-dev:container-dev \
  --pdb /inputs/xxxx/xxxx_pocket.pdb \
  --out /results \
  --device cpu \
  --num-samples 1 \
  --batch-size 1
```

Rebuild when dependencies, checkpoints, or anything installed into the image changes.
