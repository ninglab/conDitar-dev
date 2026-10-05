# Standalone DiffSMol image

`Containerfile` extends the existing DiffSMol inference image with a standalone
`run.py` entry point. It accepts a 3D ligand SDF, an optional pocket PDB, and
optional chemistry/docking evaluations. Generation and evaluations finish in
the same container invocation. The underlying `generate.py`,
`pocket_generate.py`, and `postprocess.py` remain usable independently. None of
these commands depends on the GUI at run time.

Build the image from the repository root (the pinned base image is amd64):

```bash
docker build --platform linux/amd64 -f docker/diffsmol/Containerfile -t diffsmol:cpu-20261001 docker/diffsmol
```

The base DiffSMol image is pinned by digest. The QuickVina2 binary is copied
from the existing conDitar image at build time; the resulting DiffSMol image
does not need conDitar to run. Both base images must be available to the build.

Standalone commands (bind `inputs/` and `results/` first):

```bash
docker run --rm -v "$PWD/inputs:/inputs:ro" -v "$PWD/results:/results" diffsmol:cpu-20261001 \
  python /opt/DiffSMol/docker/run.py --input /inputs/reference.sdf --output /results --device cpu --num-samples 1

docker run --rm -v "$PWD/inputs:/inputs:ro" -v "$PWD/results:/results" diffsmol:cpu-20261001 \
  python /opt/DiffSMol/docker/run.py --input /inputs/reference.sdf \
  --protein /inputs/pocket.pdb --output /results --device cpu --num-samples 1 \
  --postprocess-metrics vina_score,qed

docker run --rm -v "$PWD/inputs:/inputs:ro" -v "$PWD/results:/results" diffsmol:cpu-20261001 \
  python /opt/DiffSMol/docker/run.py --input /inputs/reference.sdf \
  --output /results --device cpu --num-samples 1 --postprocess-metrics qed,sa,logp,lipinski
```

`--protein` is required only for `vina_score`, `vina_dock`, or `qvina`.
SDF-only metrics are `qed`, `sa`, `logp`, and `lipinski`. The postprocessor
annotates generated SDFs in place with the same property names the GUI reads,
skips `reference.sdf`, and returns a failure if no molecule can be annotated.
`run.py` writes `evaluation_status.json` alongside the output SDFs; the GUI
checks that file before showing completed results. Use `--device cuda:0` on a
GPU host with a working NVIDIA runtime, or use the same image in an OpenShift
Job with a shared writable output mount. The image must be pushed to a registry
accessible to the cluster. A CPU run is not evidence that GPU drivers or a
particular OpenShift security context are configured correctly.

These evaluators are not a scientific guarantee that a generated pose is good.
