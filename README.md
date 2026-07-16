# Generating Developable 3D Molecules via Pocket-Conditioned Diffusion and Property-Aware Optimization

This is the implementation of our conDitar-dev model: https://arxiv.org/abs/2607.12349.

The repository has two parts:

- **conDitar** – the generator: a pocket encoder plus a diffusion model that generates molecules conditioned on the pocket.
- **paOPT** – a module on top of generation that steers the samples toward better ADMET endpoints while they are being generated.

The two parts are released under separate licenses; see [`NOTICE.txt`](./NOTICE.txt) and the `LICENCE.txt` file in each `scripts/` subfolder.

---

### Scripts

**conDitar**

- `train_pocketAE.py` – Pretrains the pocket encoder, which learns to represent a pocket. Produces the `PocketAE.pt` checkpoint that the diffusion model builds on.
- `train_diffusion.py` – Trains the diffusion model that generates molecules, using the pretrained pocket encoder as its condition. Produces the `Diff.pt` checkpoint.
- `sample.py` – Generates molecules for a single target and writes them out as SDF files. Works both with a reference ligand (to define the pocket) and without one, from a pocket structure alone.
- `evaluate_mol.py` – Evaluates a folder of generated molecules.

**paOPT**

- `sample_with_opt.py` – Generates molecules while optimizing them toward chosen properties (e.g. ADMET endpoints).

### Configs

- `configs/pretrain_pocket.yml` – Settings for pocket encoder pretraining.
- `configs/train_diffusion.yml` – Settings for diffusion model training.
- `configs/sample.yml` – Settings for sampling.

---

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

## Containerized Usage

There is Docker/Podman container support for CPU/GPU sampling and optional post-processing. See [`docker/README.md`](docker/README.md) for build, run, and development instructions.

---

## Data

We train our model on CrossDocked2020 v1.1 (https://bits.csb.pitt.edu/files/crossdock2020/). If you want to use your own dataset, you need to prepare paired `.pdb` (protein) and `.sdf` (ligand) data. Point the data paths in `configs/train_diffusion.yml` to your data, and it will be preprocessed automatically before training starts.

- **Curated test data**:
[`test_data/`](./data/test_data/) – the protein–ligand complexes we used for sampling and evaluation in our paper.

---

## Training

If you just want to generate molecules, you can skip this section and use the released checkpoints (linked below) with the [Sampling](#sampling) commands.

### Pocket Encoder Pretraining

Train the pocket encoder first; the diffusion model depends on it.

```bash
python -m scripts.conDitar.train_pocketAE configs/pretrain_pocket.yml
```

---

### Diffusion Model Training

Train the diffusion on top of the pretrained pocket encoder (set its path in `configs/train_diffusion.yml`).

```bash
python -m scripts.conDitar.train_diffusion configs/train_diffusion.yml
```

### Trained Model Checkpoints
[https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link](https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link)

---

## Sampling

Given a target, the model generates molecules and saves them as SDF files. Set the checkpoint paths in `configs/sample.yml` before running.

### Sampling without Optimization

Generates molecules that fit the pocket based on the pocket representations.

```bash
python -m scripts.conDitar.sample configs/sample.yml \
    --protein_root data/test_data \
    --pdb_filename 4aua/4aua_protein.pdb \
    --sdf_filename 4aua/4aua_ligand.sdf \
    --pocket_radius 10 \
    --num_samples 100 \
    --result_path results
```

Generated molecules are written to `--result_path` as `<pdb>_generated_<i>.sdf`.

Main arguments:

- `--protein_root` – base folder that holds your targets.
- `--pdb_filename` – protein (or pocket) `.pdb`, relative to `--protein_root`.
- `--sdf_filename` – reference ligand `.sdf` that determines the pocket. Omit it to sample from the pocket structure alone.
- `--pocket_radius` – radius (Å) around the reference ligand used to define the pocket.
- `--num_samples` – number of molecules to generate.
- `--result_path` – output folder for the SDF files.

### Sampling with Optimization

Generates molecules that fit the pocket while optimizing the ADMET endpoints you specify.

```bash
python -m scripts.paOPT.sample_with_opt configs/sample.yml \
    --protein_root data/test_data \
    --pdb_filename 4aua/4aua_protein.pdb \
    --sdf_filename 4aua/4aua_ligand.sdf \
    --num_samples 100 \
    --result_path outputs \
    --opt_keys Carcinogenicity \
    --opt_keys_min Carcinogenicity
```

Uses the same target/output arguments as above, plus:

- `--opt_keys` – one or more ADMET endpoints to optimize (space-separated).
- `--opt_keys_min` – the endpoints among `--opt_keys` to push lower; any endpoint not listed here is pushed higher.

---

## Evaluation

Evaluate a folder of generated molecules (molecular properties and binding affinities).

```bash
python -m scripts.conDitar.evaluate_mol \
    --sample_path results \
    --protein_root data/test_data \
    --docking_mode vina_score
```

Main arguments:

- `--sample_path` – folder of generated SDF files to evaluate (the `--result_path` you sampled into). Results are written to `<sample_path>/eval_results/`.
- `--protein_root` – base folder having the target proteins, used for docking.
- `--docking_mode` – how to predict binding: `vina_score`, `vina_dock`, `qvina`, `none`, or `all`.

---

### Generation Results

[https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link](https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link)
