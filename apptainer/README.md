# conDitar-dev Apptainer Container

This image packages the conDitar-dev code, its conda environment, and the trained checkpoints from:

- `/fs/ess/PCON0041/gruoxi/SBDDcode/checkpoints/Diff.pt`
- `/fs/ess/PCON0041/gruoxi/SBDDcode/checkpoints/PocketAE.pt`

Build from the repository root:

```bash
apptainer build conditar-dev.sif apptainer/conditar.def
```

Run pocket-only sampling:

```bash
apptainer run --nv conditar-dev.sif \
  --pdb /path/to/pocket_or_protein.pdb \
  --out /path/to/results \
  --num-samples 100
```

Run with a reference ligand:

```bash
apptainer run --nv conditar-dev.sif \
  --pdb /path/to/protein.pdb \
  --sdf /path/to/reference_ligand.sdf \
  --out /path/to/results \
  --num-samples 100
```

By default, the launcher uses CUDA when it is visible and falls back to CPU otherwise. You can also select CPU or GPU explicitly with `--device` or `CONDITAR_DEVICE`:

```bash
CONDITAR_DEVICE=cpu apptainer run conditar-dev.sif \
  --pdb /path/to/pocket_or_protein.pdb \
  --out /path/to/results

CONDITAR_DEVICE=cuda:0 apptainer run --nv conditar-dev.sif \
  --pdb /path/to/pocket_or_protein.pdb \
  --out /path/to/results
```

The image exposes the same command as `conditar-sample`. Extra options are passed through to `scripts.conDitar.sample`.

```bash
apptainer exec --nv conditar-dev.sif conditar-sample --help
```

## Development Changes

For baked-in code, config, or dependency changes, edit the repo and rebuild:

```bash
apptainer build --force conditar-dev.sif apptainer/conditar.def
```

For quick Python/config iteration without rebuilding, bind the live checkout over the in-image app directory:

```bash
apptainer run --nv \
  --bind "$PWD":/opt/conditar/app \
  conditar-dev.sif \
  --pdb /path/to/pocket_or_protein.pdb \
  --out /path/to/results
```

Rebuild when dependencies, checkpoints, or anything installed in `%post` changes.
