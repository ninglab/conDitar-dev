# conDitar-dev

conDitar generates ligand structures conditioned on a protein or prepared
pocket. This branch supports CPU/GPU execution, Docker/Podman containers, and
optional Vina/QVina post-processing.

## Environment Setup

```bash
conda create -n conDitar-dev python=3.10 -y
conda activate conDitar-dev

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric==2.7.0
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
conda install -c conda-forge rdkit openbabel tensorboard pyyaml easydict python-lmdb
pip install meeko==0.1.dev3 pdb2pqr tqdm vina cvxpy admet-ai
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3  
```

---

## Data

- **Preprocessed training data**:
[https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link](https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link)

- **Curated test data**:
[`test_data/`](./data/test_data/)

---
## Training

### Pocket Encoder Pretraining

```bash
python -m scripts.conDitar.train_pocketAE configs/pretrain_pocket.yml
```

---

### Diffusion Model Training

```bash
python -m scripts.conDitar.train_diffusion configs/train_diffusion.yml
```

### Trained Model Checkpoints
[https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link](https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link)

---

## Sampling

### Sampling without Optimization
```bash
python -m scripts.conDitar.sample configs/sample.yml
```

### Sampling with Optimization
```bash
python -m scripts.paOPT.sample_with_opt configs/sample.yml
```

### Docker Container

For local Docker CPU/GPU runs and OSC Podman/Slurm runs, see [`docker/README.md`](docker/README.md). The image uses the same `conditar-sample` launcher everywhere.

```bash
docker/build-image.sh
INPUT_DIR=/fs/ess/PCON0041/gruoxi/conDitar-dev/examples
docker run --rm -e CONDITAR_DEVICE=cpu -v "$INPUT_DIR":/inputs:ro -v "$PWD/results":/results localhost/conditar-dev:container-dev --pdb /inputs/xxxx/xxxx_pocket.pdb --out /results --device cpu --num-samples 1 --batch-size 1
docker run --rm --gpus all -e CONDITAR_DEVICE=cuda:0 -v "$INPUT_DIR":/inputs:ro -v "$PWD/results":/results localhost/conditar-dev:container-dev --pdb /inputs/xxxx/xxxx_pocket.pdb --out /results --device cuda:0 --num-samples 10
```

Add optional Vina score/minimize post-processing with `--vina-score`. Vina runs
after generation inside the same container and annotates each generated SDF with
properties such as `VINA_SCORE_ONLY`, `VINA_MINIMIZE`, `QED`, and `SA`.

---

## GUI integration

The GUI stages uploaded inputs and invokes this same Docker/Podman container.
Local jobs use Docker; OSC jobs submit Podman commands through Slurm. See the
GUI repository for setup, batch behavior, logs, and export instructions.
