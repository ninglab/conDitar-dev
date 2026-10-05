# OpenShift Deployment Reference

For the partner's clone-and-deploy procedure, start with
`SITE_QUICKSTART.md`. The launcher uses the current `oc` project, three
pinned prebuilt images, and real CPU OpenShift Jobs by default:

```bash
cd gui
./openshift/deploy.sh
```

The script creates the GUI Deployment, Service, Route, service account and Job
RBAC, ConfigMap, and a persistent job-storage PVC. It waits for the GUI rollout
and prints the Route. It does not install packages, start a build, or require a
Docker socket. conDitar and DiffSMol run in separate engine images as
OpenShift Jobs; generation and selected evaluations finish before Results are
shown. The Vina preprocessing panel also runs in a Job using the conDitar
image.

The tested image references are in `image-list.txt`. They currently point to
`avery91-dev` on this cluster. For a different project, grant pull access as
shown in `SITE_QUICKSTART.md` or mirror all three images. For a different
cluster, use a registry reachable from that cluster.

## Overrides

From the repository root, pass registry-pullable images explicitly:

```bash
./gui/openshift/deploy.sh --project <project> \
  --gui-image <gui-image> \
  --runtime-image <conditar-image> \
  --diffsmol-image <diffsmol-image>
```

The equivalent environment variables are `CONDITAR_GUI_IMAGE`,
`CONDITAR_DOCKER_IMAGE`, and `DIFFSMOL_DOCKER_IMAGE`. Other supported settings
are listed in `site.env.example` and `./openshift/deploy.sh --help`.

- `--no-submit` writes Job manifests without submitting them.
- `--runtime openshift_mock` runs synthetic infrastructure diagnostics.
- `--cpu` forces CPU settings when overriding GPU environment variables.
- `--storage 50Gi` changes the PVC request at first creation. Resizing an
  existing PVC depends on the storage class.
- `--route-host HOST` requests a fixed hostname, if permitted by the cluster.
- `--build-gui` opts into an OpenShift binary build and creates a BuildConfig
  and ImageStream. This path downloads image-build dependencies inside the
  builder; it is not required for partner deployment. `--build-gui --skip-build`
  resumes after a completed build.

`--submit` is the default. The launcher rejects known outdated engine images
in real-submission mode. The GUI image is built from `gui/Containerfile`; the
conDitar and DiffSMol images are built independently using the instructions in
`../../docker/conditar_dev/README.md` and `../../docker/diffsmol/README.md`.

## Storage And Permissions

Job files live at `/data/jobs` on the `conditar-gui-jobs` PVC. The default PVC
uses `ReadWriteOnce`, so generated Jobs have required pod affinity to the GUI
pod's node. The GUI Deployment uses a `Recreate` rollout to avoid simultaneous
mounts during upgrades. The project must permit the GUI service account to
create and monitor Jobs and read pod logs, and both the GUI and Job service
accounts must be able to pull their images.

The Route has a 35-minute timeout for synchronous docking-panel preprocessing.
Any external load balancer must permit an adequate request duration too.
GPU execution requires a project with GPU quota and site-specific resource
settings; CPU validation alone does not establish GPU readiness.

Use `VALIDATION_PROTOCOL.md` for the acceptance checks. Inspect a failed
rollout or Job with `oc describe pod`, `oc get events`, and `oc logs` before
changing resources or image references.
