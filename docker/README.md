# Container Images

- [`conditar_dev/`](conditar_dev/) builds the standalone conDitar image from
  this repository's model code and checkpoints.
- [`diffsmol/`](diffsmol/) builds the standalone DiffSMol image and contains
  its generation and evaluation adapters.

These images are built independently. The GUI references published images; it
does not supply their runtime scripts to engine Jobs.
