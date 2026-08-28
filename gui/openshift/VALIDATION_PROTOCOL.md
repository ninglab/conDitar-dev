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

## 3. Run A Small CPU Job

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

## 4. Check OpenShift Objects

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

## 5. Optional Follow-Up Tests

After the first CPU test passes:

- Repeat with Vina enabled.
- Repeat with a larger sample count.
- If GPUs are available, redeploy with site GPU settings and run a small GPU
  test.

## Known Notes

- First runtime image pull can take several minutes.
- `ContainerCreating` can be normal while the image is pulling.
- If a pod reports PVC multi-attach errors, confirm storage mode and scheduling.
  The generated Jobs include pod affinity for `ReadWriteOnce` storage.
- Long `oc logs -f` sessions can drop if local credentials expire; this does
  not imply the OpenShift Job failed.
