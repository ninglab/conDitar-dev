# OpenShift Acceptance Checks

Run these checks after deploying from a clean clone using
`SITE_QUICKSTART.md`. Record the project, image references, job IDs, and
result for each run outside the repository.

1. Confirm `oc get deployment conditar-gui`, `oc get route conditar-gui`, and
   `oc get pvc conditar-gui-jobs` show one available GUI replica, an HTTPS
   Route, and bound storage. The GUI health endpoint should respond.
2. Run a small conDitar pocket-only job with **OpenShift Job** selected. Confirm
   that an OpenShift Job completes, the GUI shows generated SDFs, and the study
   export contains the inputs and provenance manifest.
3. Run DiffSMol with a 3D ligand SDF, then with a ligand and pocket. For the
   second run, enable chemistry and Vina evaluation. Confirm Results wait for
   post-processing and the exported SDF has the selected annotations.
4. Run conDitar with a protein and reference ligand, then use the Vina docking
   panel in Preprocess. Confirm the panel creates a separate OpenShift Job and
   produces docked poses and pocket candidates.
5. Check malformed PDB/SDF rejection, PDB/CIF complex splitting, residue-based
   pocket selection, candidate staging/removal, and the 3D/2D Results viewer.
6. Restart the GUI Deployment and confirm completed jobs, logs, annotations,
   and study exports remain available on the PVC.
7. In a separate project, verify that all three images pull successfully. If
   GPU quota is available, deploy with GPU settings and repeat a small engine
   job. A CPU pass does not establish GPU compatibility.

Inspect a generated Job with:

```bash
oc get jobs
oc get pods -l job-name=<job-name>
oc logs job/<job-name>
```

The first pull of a large image can take several minutes. For a pod stuck in
`Pending` or `ContainerCreating`, inspect pod events and check image-pull
permission, quota, and PVC scheduling before treating the application run as
failed.
