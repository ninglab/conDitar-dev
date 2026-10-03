# Input Processing

This package contains deterministic GUI-side preparation steps for conDitar inputs.
It should stay separate from the conDitar model/runtime code.

## Current Scope

- Accept uploaded PDB/ENT protein-ligand complexes.
- Detect bound ligand candidates from `HETATM` records.
- Filter common water, solvent, and ion records away from automatic selection.
- Extract a protein PDB and selected reference ligand SDF for the existing
  protein + reference ligand workflow.
- Extract ATOM-only pocket PDBs using the lab convention of whole protein
  residues within 10 A of a selected ligand or pocket center.
- Summarize docked ligand poses into recurring candidate pocket centers.

## Design Boundary

This code should prepare and validate inputs in auditable, reproducible ways.
It should not silently decide uncertain biology for the user.

Good deterministic tasks:

- parse structure files
- identify ligand/cofactor candidates
- extract selected ligand and protein records
- compute ligand-derived pocket metadata
- return warnings and provenance

Future exploratory tasks, such as homolog searches, literature-backed ligand
selection, or docking-panel pocket discovery, should produce candidate options
that the user can review before this package prepares final conDitar inputs.

## No Reference Ligand Workflow

The non-agent path should be deterministic:

1. Prepare or accept a protein structure.
2. Generate pocket proposals from explicit sources:
   - known bound ligands in homologous/family structures
   - cavity-detection tools such as fpocket/P2Rank
   - docking pose clusters from a documented ligand panel
3. Return ranked pocket candidates with centers, radii, support counts, scores,
   warnings, and provenance.
4. Let the user pick one candidate before conDitar input files are staged.

The GUI should present these as hypotheses, not as an automatic binding-site
decision.

## Vina Panel v1

The first docking-panel backend uses AutoDock Vina through the conDitar
container image. The GUI backend stages a protein PDB, ligand-panel SDF, and a
copy of `docking_panel_vina.py` in the preprocessing run directory. It runs
that adapter using the image's own docking dependencies, then clusters the
returned pose centers. No source-checkout mount is needed.

If no search box is supplied, v1 docks against a whole-protein bounding box.
That is intentionally conservative for unknown-pocket exploration, but it can
be slow and noisy. A user-selected or tool-proposed search box should be
preferred whenever available.
