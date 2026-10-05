# Input Processing

This package prepares structures for the GUI without changing the standalone
conDitar or DiffSMol engines. It validates uploads, converts CIF structures,
identifies bound ligands, and records provenance for generated candidates.

For bound complexes, the user selects a ligand before extracting a cleaned
protein PDB, reference-ligand SDF, or pocket. Pocket extraction includes whole
protein residues within the selected distance of the ligand. For a protein
without a reference ligand, users can select residues directly or run the Vina
docking panel to obtain pose-derived pocket candidates.

The docking-panel adapter uses dependencies inside the conDitar image. On
OpenShift it runs as a separate Job with shared PVC inputs and outputs; local
container runs use the same adapter. Its returned poses and pocket centers are
candidates for user review, not automatic binding-site assignments. A
whole-protein search box may be slow or ambiguous, so a constrained search box
is preferable when the site is known.
