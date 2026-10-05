# OpenShift Quickstart

Use an OpenShift Web Terminal with access to the deployment project. It provides
`git` and `oc`; no local Python, Docker, or image build is required.

```bash
oc project <partner-project>
git clone --branch gui-dev https://github.com/ninglab/conDitar-dev.git
cd conDitar-dev/gui
./openshift/deploy.sh
```

Open the HTTPS Route printed by the script. The GUI runs conDitar and DiffSMol
as OpenShift Jobs on CPU by default. The three prebuilt images are pinned in
`image-list.txt`; deployment does not install dependencies or start a build.

## Image Access

The current images are stored in `avery91-dev` on the same cluster. Before
deploying into another project, an owner of `avery91-dev` must grant pull
access to both pod service accounts:

```bash
oc policy add-role-to-user system:image-puller system:serviceaccount:<partner-project>:conditar-gui -n avery91-dev
oc policy add-role-to-user system:image-puller system:serviceaccount:<partner-project>:default -n avery91-dev
```

Alternatively, mirror all three images to a registry the project can pull
from and set `CONDITAR_GUI_IMAGE`, `CONDITAR_DOCKER_IMAGE`, and
`DIFFSMOL_DOCKER_IMAGE` before running the script. Do not put registry
credentials in the repository.

## Verify

```bash
oc get deployment conditar-gui
oc get route conditar-gui
oc get pvc conditar-gui-jobs
```

The Deployment should have one available replica and the PVC should be bound.
In the GUI, select **OpenShift Job** and run a small job. Results should appear
after generation and selected post-processing complete. See
`VALIDATION_PROTOCOL.md` for the full acceptance checks and `README.md` for
deployment options.

## GPU

OpenShift Job is the run target; CPU or GPU is the resource requested by that
Job. The default is CPU. To use a GPU, first confirm the project has GPU quota
and the correct resource key, then set the device, GPU count, and memory values
documented in `site.env.example` before redeploying. GPU execution has not yet
been validated in this project.
