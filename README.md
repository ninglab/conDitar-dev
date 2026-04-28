## Environment Setup

### Create Conda Environment
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
  ```
  crossdocked_train_processed.lmdb
  ```

- **Curated test data**:
  ```
  test_data/
  ```

---

## Pocket Encoder Pretraining

```bash
python -m scripts.train_pocketAE configs/pretrain_pocket.yml
```

---

## Diffusion Model Training

```bash
python -m scripts.train_diffusion configs/train_diffusion.yml
```

---

## Sampling

### Sampling without Optimization
```bash
python -m scripts.sample configs/sample.yml
```

### Sampling with Optimization
```bash
python -m scripts.sample_with_opt configs/sample.yml
```

---

## Evaluation

```bash
python -m scripts.evaluate_mol
```

---

### Trained Model Checkpoints and Generation Results

[https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link](https://drive.google.com/drive/folders/158A-cQKIF-x_-ewrf7jPGdFew005I3W0?usp=drive_link)
