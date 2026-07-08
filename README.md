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

### Apptainer / Singularity Container

The `container_dev` branch includes an Apptainer definition that packages the conda environment, conDitar-dev code, and trained checkpoints into a single image. See [`apptainer/README.md`](apptainer/README.md).

```bash
apptainer build conditar-dev.sif apptainer/conditar.def
apptainer run --nv conditar-dev.sif --pdb /path/to/pocket.pdb --out /path/to/results
apptainer run conditar-dev.sif --pdb /path/to/pocket.pdb --out /path/to/cpu_results
CONDITAR_DEVICE=cpu apptainer run conditar-dev.sif --pdb /path/to/pocket.pdb --out /path/to/results
apptainer run --nv conditar-dev.sif --pdb /path/to/protein.pdb --sdf /path/to/ligand.sdf --out /path/to/results
```

### Docker Container

For Docker Desktop local CPU runs and Docker/NVIDIA GPU runs, see [`docker/README.md`](docker/README.md). The Docker image keeps the same `conditar-sample` launcher and `--device` / `CONDITAR_DEVICE` CPU-GPU behavior as the Apptainer image.

```bash
docker/build-image.sh
INPUT_DIR=/fs/ess/PCON0041/gruoxi/conDitar-dev/examples
docker run --rm -e CONDITAR_DEVICE=cpu -v "$INPUT_DIR":/inputs:ro -v "$PWD/results":/results localhost/conditar-dev:container-dev --pdb /inputs/xxxx/xxxx_pocket.pdb --out /results --device cpu --num-samples 1 --batch-size 1
docker run --rm --gpus all -e CONDITAR_DEVICE=cuda:0 -v "$INPUT_DIR":/inputs:ro -v "$PWD/results":/results localhost/conditar-dev:container-dev --pdb /inputs/xxxx/xxxx_pocket.pdb --out /results --device cuda:0 --num-samples 10
```

Add optional Vina score/minimize post-processing with `--vina-score`. Vina runs
after generation inside the same container and writes `eval_results/vina_scores.csv`
and `eval_results/vina_scores.json` under the output directory.

---

## Evaluation

```bash
python -m scripts.conDitar.evaluate_mol
```

---

### Generation Results

[https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link](https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link)
