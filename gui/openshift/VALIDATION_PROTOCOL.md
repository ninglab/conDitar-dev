# conDitar GUI OpenShift Validation Protocol

Use this protocol after the GUI deploy script prints a Route URL.

## 1. Confirm Deployment

```bash
oc get deployment conditar-gui
oc get pods -l app=conditar-gui -o wide
oc get route conditar-gui
oc get pvc conditar-gui-jobs
```

Expected:

- Deployment is available.
- One GUI pod is running.
- A Route exists.
- PVC is bound.

## 2. Open The GUI

Open the Route URL in a browser.

Expected:

- The GUI loads.
- The run target is `OpenShift Job`.
- Setup/readiness shows OpenShift Job submission is enabled.

## 3. Run Infrastructure Diagnostics

For an optional infrastructure-only check, deploy with `--runtime openshift_mock --cpu`
and submit a diagnostics job. Confirm the synthetic SDFs appear in
Results, export a study ZIP, restart the GUI Deployment, and confirm the job
still loads. This proves the Route and PVC path, not conDitar or DiffSMol.

## 4. Run A Small CPU Engine Job

The no-argument launcher already selects real CPU Jobs and the pinned conDitar
and DiffSMol images. If overriding those images, use `--submit --cpu` with
registry-pullable refreshed images. Do not use local Docker tags or older images.
The project must allow CPU and memory requests for the generator Jobs.

Recommended first settings:

- Target: `OpenShift Job`
- Samples: `2`
- Batch size: `100`
- Vina: off
- Optional tools: Lilly Medchem Rules and MedChem Filters

Expected:

- Job status changes to `running`.
- Logs show manifest creation and OpenShift Job submission.
- Additional logs show sampling progress.
- Job status changes to `completed`.
- Results show generated SDF output.

## 5. Check OpenShift Objects

For the generated Job name shown in the GUI logs:

```bash
oc get job <job-name> -o wide
oc get pods -l job-name=<job-name> -o wide
oc logs job/<job-name>
```

Expected:

- Job shows `Complete`.
- Pod shows `Completed`.
- Logs include sampling progress and normal completion output.

## 6. Follow-Up Tests

After the first CPU test passes:

- Run DiffSMol ligand-only and ligand-plus-pocket jobs with selected
  evaluations, checking that Results wait for post-processing.
- Run Vina docking-panel preprocessing and verify its separate Kubernetes Job
  produces docked poses and pocket candidates without a Docker socket.
- Repeat conDitar with Vina enabled and a larger sample count.
- If GPUs are available, redeploy with site GPU settings and run a small GPU
  test.

## Known Notes

- First runtime image pull can take several minutes.
- `ContainerCreating` can be normal while the image is pulling.
- If a pod reports PVC multi-attach errors, confirm storage mode and scheduling.
  The generated Jobs include pod affinity for `ReadWriteOnce` storage.
- Long `oc logs -f` sessions can drop if local credentials expire; this does
  not imply the OpenShift Job failed.

## 2026-10-04 CPU Rehearsal (`avery91-dev`)

Passed with real OpenShift Jobs and the separate conDitar and DiffSMol images:

- conDitar pocket-only job `20261004-155333-6b6fa9d3` and DiffSMol ligand-only
  job `20261004-155939-6fab107b` completed with generated SDFs and study ZIPs.
- DiffSMol ligand plus pocket, shape guidance, chemistry and Vina evaluation
  job `20261004-161939-85a5bfda` completed; the SDF contains evaluation
  properties and Results appeared after post-processing.
- conDitar protein plus reference ligand with QED and Vina evaluation job
  `20261004-161951-df380626` completed with annotated SDF output.
- Vina docking-panel preprocessing Job `vina-panel-20261004-161736-bd4e09a1`
  completed with two poses and pocket candidates.
- PDB and CIF complex splitting, selected-residue pocket generation, invalid
  input rejection, and export manifest/input hashes were checked.

Completed on 2026-10-05: the GUI image with pinned MedChem 2.1.1 was built
in OpenShift and deployed with `--submit --cpu`. The first MedChem API run
exposed a native-process exit in the threaded GUI server. Tool Chest runs now
execute in child processes, and the revised image passed both optional-tool
API calls on DiffSMol job `20261004-155939-6fab107b` without a GUI pod
restart. Results returned one SDF with MedChem and Lilly annotations. A GUI
Deployment restart preserved that job, its annotations, and its study ZIP.
The one-command build/deploy path and `--skip-build` recovery path both worked;
the latter was used after a transient local OpenShift API DNS failure while
the cluster-side build continued to completion.

Not yet covered: GPU Jobs (no GPU quota in this project) and image pulls from
a different OpenShift project. The engine images used here live in
`avery91-dev`; a partner project needs pull permission or mirrored images.
