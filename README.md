# SBDD Data and Code

This repository contains code and data for xxxx, including pocket representation pretraining, pocket-conditional diffusion training, diffusion-based ligand generation, optional optimization, and evaluation.

---

## 1. Environment Setup

### 1.1 Create Conda Environment
```bash
conda create -n SBDD python=3.10 -y
conda activate SBDD
```

### 1.2 Install PyTorch
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 1.3 Install PyG
```bash
pip install torch-geometric==2.7.0
```

```bash
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

### 1.4 Install Chemistry and Utility Libraries
```bash
conda install -c conda-forge rdkit openbabel tensorboard pyyaml easydict python-lmdb
```

```bash
pip install meeko==0.1.dev3 pdb2pqr tqdm vina cvxpy admet-ai 
```

```bash
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3 
```


---

## 2. Data

- **Preprocessed training data**:
  ```
  crossdocked_train_processed.lmdb
  ```

- **Curated test data**:
  ```
  test_data/
  ```

---

## 3. Pocket Encoder Pretraining

```bash
python -m scripts.train_pocketAE configs/pretrain_pocket.yml
```

---

## 4. Diffusion Model Training

```bash
python -m scripts.train_diffusion configs/train_diffusion.yml
```

---

## 5. Sampling

### 5.1 Sampling without Optimization
```bash
python -m scripts.sample configs/sample.yml
```

### 5.2 Sampling with Optimization
```bash
python -m scripts.sample_with_opt configs/sample.yml
```

---

## 6. Evaluation

```bash
python -m scripts.evaluate_mol
```
