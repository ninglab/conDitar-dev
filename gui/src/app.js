import { ADVANCED_PARAMETERS, EXAMPLES, PARAMETERS } from "./config.js?v=20260723-theme-1";
import { drawCategoryChart, drawHistogram } from "./charts.js?v=20260723-theme-1";
import { ExampleDataService } from "./data-service.js?v=20260723-results-fix-1";
import { vinaWasRun } from "./sdf.js?v=20260723-theme-1";
import { render2D, render3D } from "./viewers.js?v=20261002-viewer-controls-1";
import {
  CHEMISTRY_METRICS as WORKFLOW_CHEMISTRY_METRICS,
  VINA_METRICS as WORKFLOW_VINA_METRICS,
  filterMetricsForWorkflow,
  normalizeWorkflowWarning,
  requiredInputs,
  resolveGenerationWorkflow,
  setupWarningsForWorkflow,
  validateGenerationSetup,
  vinaModeForMetrics,
  warningText,
} from "./workflow-rules.js?v=20260930-evaluation-contract-1";

const service = new ExampleDataService();
const ACTIVE_JOB_STATUSES = new Set(["queued", "running"]);
const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "canceled"]);
const CLEANUP_JOB_STATUSES = new Set(["failed", "canceled"]);
const OPENSHIFT_JOB_TARGET = "openshift_job";
const OPENSHIFT_MOCK_TARGET = "openshift_mock";
const SLURM_GPU_TARGET = "slurm_gpu";
const LEGACY_SLURM_GPU_TARGET = "osc_gpu";
const MAX_CATEGORICAL_FILTER_VALUES = 24;
const THEME_STORAGE_KEY = "conditar-theme";
const VINA_EVALUATIONS = WORKFLOW_VINA_METRICS;
const CHEMISTRY_EVALUATIONS = WORKFLOW_CHEMISTRY_METRICS;
const DIFFSMOL_SHAPE_HEAVY_ATOM_WARNING = 35;
const DEFAULT_BUILTIN_EVALUATIONS = new Set(["vina_score", "qed", "sa", "logp", "lipinski"]);
const EVALUATION_LABELS = {
  vina_score: "Vina score + minimize",
  vina_dock: "Vina redock",
  qvina: "QVina",
  qed: "QED",
  sa: "SA",
  logp: "LogP",
  lipinski: "Lipinski",
};

const state = {
  study: null,
  selected: null,
  exampleId: "custom",
  engine: "conditar",
  mode: "reference",
  view: "3d",
  showReferenceLigand: false,
  workspaceViewerMolecule: "selected",
  parameters: Object.fromEntries([...PARAMETERS, ...ADVANCED_PARAMETERS].map((item) => [item.key, item.value])),
  customPdb: null,
  customSdf: null,
  preprocessComplex: null,
  preprocessComplexResult: null,
  preprocessComplexResultSignature: null,
  preprocessTarget: null,
  preparedStructures: [],
  selectedPreparedStructureId: null,
  stagedPreparedStructureId: null,
  stagedPreparedVariantId: null,
  stagedPreprocessManifest: null,
  preprocessProtein: null,
  preprocessPanel: null,
  vinaPocketResult: null,
  vinaPocketResultSignature: null,
  selectedVinaPocketId: null,
  batchInputs: [],
  currentJob: null,
  selectedJob: null,
  jobs: [],
  jobFilter: "all",
  jobPollTimer: null,
  jobsRefreshTimer: null,
  activeTab: "setup",
  resultSource: "upload",
  runtimeHealth: null,
  targetTouched: false,
  histogramThreshold: null,
  exportFilters: {},
  exportSelection: new Set(),
  notifiedJobs: new Set(),
  watchedJobs: new Set(),
  thresholdFrame: null,
  exportFilterTimer: null,
  tools: [],
  toolsLoaded: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function initialize() {
  initializeTheme();
  const commonKeys = new Set(["num_samples", "pocket_radius", "diffsmol_guidance"]);
  renderParameterFields(PARAMETERS.filter((parameter) => commonKeys.has(parameter.key)), $("#parameter-fields"));
  renderParameterFields(
    [...PARAMETERS.filter((parameter) => !commonKeys.has(parameter.key)), ...ADVANCED_PARAMETERS],
    $("#advanced-fields"),
  );
  bindEvents();
  setMode("reference", false);
  updateEngineControls();
  updateInputLabels(null);
  updateRunSummary();
  refreshJobs(false);
  setActiveTab("preprocess");
}

function renderParameterFields(parameters, container) {
  container.innerHTML = parameters.map((parameter) => {
    const control = parameter.type === "select"
      ? `<select id="param-${parameter.key}">${parameter.options.map((option) => `<option ${option === parameter.value ? "selected" : ""}>${option}</option>`).join("")}</select>`
      : parameter.type === "checkbox"
        ? `<input id="param-${parameter.key}" type="checkbox" ${parameter.value ? "checked" : ""}>`
        : `<input id="param-${parameter.key}" type="${parameter.type}" value="${parameter.value}" ${parameter.min !== undefined ? `min="${parameter.min}"` : ""} ${parameter.max !== undefined ? `max="${parameter.max}"` : ""} ${parameter.step ? `step="${parameter.step}"` : ""}>`;
    return `<div class="parameter-field" data-parameter-key="${escapeHtml(parameter.key)}"><label for="param-${parameter.key}">${parameter.label}${tooltip(parameter.tooltip)}${parameter.suffix ? `<span>${parameter.suffix}</span>` : ""}</label>${control}<small>${parameter.help || ""}</small></div>`;
  }).join("");
}

function tooltip(text) {
  return text ? `<span class="info-tip" tabindex="0" aria-label="${escapeHtml(text)}" data-tip="${escapeHtml(text)}">?</span>` : "";
}

function bindEvents() {
  $("#example-select").addEventListener("change", (event) => {
    if (event.target.value === "custom") {
      state.exampleId = "custom";
      state.study = null;
      state.selected = null;
      state.exportFilters = {};
      state.resultSource = "upload";
      updateInputLabels(null);
      renderSummary();
      renderExportFilters();
      renderResultsTable();
      showToast("Upload custom input structures, then submit a local CPU job.");
      return;
    }
    loadExample(event.target.value);
  });
  $$(".engine-option").forEach((button) => button.addEventListener("click", () => setEngine(button.dataset.engine)));
  $$(".mode-toggle button").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $$(".view-toggle button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $("#viewer-reference-toggle").addEventListener("change", (event) => {
    state.showReferenceLigand = event.target.checked;
    renderSelectedStructure();
  });
  $$(".workflow-step").forEach((button) => button.addEventListener("click", () => setActiveTab(button.dataset.section)));
  [...PARAMETERS, ...ADVANCED_PARAMETERS].forEach((parameter) => {
    $(`#param-${parameter.key}`).addEventListener("input", (event) => {
      state.parameters[parameter.key] = parameter.type === "number" ? Number(event.target.value) : parameter.type === "checkbox" ? event.target.checked : event.target.value;
      updateEngineControls();
      updateRunSummary();
    });
  });
  $("#reset-params").addEventListener("click", resetParameters);
  $("#preview-run").addEventListener("click", submitGenerationJob);
  $("#refresh-health").addEventListener("click", () => refreshRuntime(true));
  $("#job-target").addEventListener("change", () => {
    state.targetTouched = true;
    updateJobTargetControls();
    if (state.runtimeHealth) {
      renderRuntimeStatus(state.runtimeHealth);
      renderSetupHealth(state.runtimeHealth);
    }
  });
  $("#refresh-pdb").addEventListener("click", () => chooseFileAgain("#pdb-input"));
  $("#refresh-sdf").addEventListener("click", () => chooseFileAgain("#sdf-input"));
  $("#preprocess-refresh-complex").addEventListener("click", () => chooseFileAgain("#preprocess-complex-input"));
  $("#preprocess-refresh-target").addEventListener("click", () => chooseFileAgain("#preprocess-target-input"));
  $("#preprocess-target-input").addEventListener("change", handlePreprocessTargetUpload);
  $("#preprocess-preview-residue-pocket").addEventListener("click", previewResiduePocket);
  $("#preprocess-build-residue-pocket").addEventListener("click", buildResiduePocket);
  $("#preprocess-complex-input").addEventListener("change", handlePreprocessComplexUpload);
  $("#preprocess-apply-ligand").addEventListener("click", applyPreprocessSelectedLigand);
  $("#preprocess-complex-radius").addEventListener("input", renderPreparedStructureTray);
  $("#preprocess-preview-reference").addEventListener("click", () => previewPreprocessedComplexVariant("reference"));
  $("#preprocess-preview-ligand").addEventListener("click", () => previewPreprocessedComplexVariant("shape"));
  $("#preprocess-preview-pocket").addEventListener("click", () => previewPreprocessedComplexVariant("pocket"));
  $("#preprocess-download-complex").addEventListener("click", downloadPreprocessedComplex);
  $("#preprocess-use-reference").addEventListener("click", () => usePreprocessedComplex("reference"));
  $("#preprocess-use-ligand").addEventListener("click", () => usePreprocessedComplex("shape"));
  $("#preprocess-use-pocket").addEventListener("click", () => usePreprocessedComplex("pocket"));
  $("#preprocess-refresh-protein").addEventListener("click", () => chooseFileAgain("#preprocess-protein-input"));
  $("#preprocess-protein-input").addEventListener("change", handlePreprocessProteinUpload);
  $("#preprocess-refresh-panel").addEventListener("click", () => chooseFileAgain("#preprocess-panel-input"));
  $("#preprocess-panel-input").addEventListener("change", handlePreprocessPanelUpload);
  $("#preprocess-run-vina-panel").addEventListener("click", runVinaPanelPreprocessing);
  $("#preprocess-download-vina-pocket").addEventListener("click", downloadSelectedVinaPocket);
  $("#preprocess-use-vina-pocket").addEventListener("click", useSelectedVinaPocket);
  $$(".builtin-evaluation-toggle").forEach((input) => input.addEventListener("change", updateVinaControls));
  ["#vina-exhaustiveness", "#vina-cpu"].forEach((selector) => {
    $(selector).addEventListener("input", updateRunSummary);
  });
  ["#vina-panel-center-x", "#vina-panel-center-y", "#vina-panel-center-z", "#vina-panel-size-x", "#vina-panel-size-y", "#vina-panel-size-z", "#vina-panel-max-ligands", "#vina-panel-exhaustiveness"].forEach((selector) => {
    $(selector).addEventListener("input", renderPreparedStructureTray);
  });
  $("#refresh-jobs").addEventListener("click", () => refreshJobs(true));
  $("#job-filter").addEventListener("change", (event) => {
    state.jobFilter = event.target.value;
    renderJobsTable();
  });
  $("#result-search").addEventListener("input", renderResultsTable);
  $("#result-sort").addEventListener("change", renderResultsTable);
  $("#histogram-metric").addEventListener("change", () => {
    state.histogramThreshold = null;
    renderCharts();
  });
  $("#histogram-threshold").addEventListener("input", (event) => {
    state.histogramThreshold = Number(event.target.value);
    $("#histogram-threshold-value").textContent = Number(event.target.value).toFixed(1);
    syncHistogramThresholdToExportFilter({ defer: true });
    scheduleThresholdRender();
  });
  $("#histogram-threshold").addEventListener("change", () => syncHistogramThresholdToExportFilter());
  $("#histogram").addEventListener("mousemove", handleHistogramHover);
  $("#histogram").addEventListener("mouseleave", () => { $("#histogram-tooltip").textContent = "Hover a bar to see its range and count."; });
  $(".analytics-details").addEventListener("toggle", (event) => {
    if (event.target.open) requestAnimationFrame(renderCharts);
  });
  $("#viewer-molecule-select").addEventListener("change", (event) => {
    state.workspaceViewerMolecule = event.target.value;
    renderMoleculeViewerWorkspace();
  });
  $("#viewer-use-selected").addEventListener("click", () => {
    state.workspaceViewerMolecule = "selected";
    $("#viewer-molecule-select").value = "selected";
    renderMoleculeViewerWorkspace();
  });
  $("#download-selected").addEventListener("click", downloadSelected);
  $("#download-csv").addEventListener("click", downloadCsv);
  $("#download-config").addEventListener("click", downloadConfig);
  $("#download-all").addEventListener("click", downloadAll);
  $("#export-filtered").addEventListener("change", updateExportScope);
  $("#reset-export-filters").addEventListener("click", resetExportFilters);
  $("#select-all-candidates").addEventListener("click", () => { state.exportSelection = new Set(state.study?.candidates?.map((item) => item.id) || []); renderResultsTable(); });
  $("#clear-candidate-selection").addEventListener("click", () => { state.exportSelection.clear(); renderResultsTable(); });
  $("#theme-toggle").addEventListener("click", toggleTheme);
  $("#pdb-input").addEventListener("change", handlePdbUpload);
  $("#sdf-input").addEventListener("change", handleSdfUpload);
  $("#pdb-input").addEventListener("click", clearFileInputBeforeChoose);
  $("#sdf-input").addEventListener("click", clearFileInputBeforeChoose);
  $("#folder-input").addEventListener("change", handleFolderUpload);
  $("#clear-batch").addEventListener("click", clearBatchSelection);
  window.addEventListener("resize", debounce(renderCharts, 120));
  updateJobTargetControls();
  updateVinaControls();
  refreshRuntime(false);
  loadTools();
}

async function loadTools() {
  const status = $("#tool-chest-status");
  if (status) status.textContent = "Loading tools";
  try {
    state.tools = await service.listTools();
    state.toolsLoaded = true;
  } catch (error) {
    state.tools = [];
    state.toolsLoaded = false;
    if (status) status.textContent = "Tools unavailable";
    console.warn("Tool Chest failed to load", error);
  }
  renderEvaluationTools();
  renderToolChest();
  renderExportFilters();
  renderHistogramMetricOptions();
  if (state.study?.candidates?.length) {
    applyExportFilters(false);
    renderResultsTable();
  }
}

async function refreshRuntime(showMessage = false) {
  const status = $("#runtime-status");
  const detail = $("#runtime-detail");
  status.textContent = "Checking runtime…";
  detail.textContent = "Detecting available container and scheduler tools.";
  try {
    const health = await service.health();
    state.runtimeHealth = health;
    const slurmAvailable = Boolean(health.slurm?.sbatch);
    if (!state.targetTouched) {
      const defaultTarget = health.default_target || (slurmAvailable ? SLURM_GPU_TARGET : "local_cpu");
      if ($(`#job-target option[value="${defaultTarget}"]`)) {
        $("#job-target").value = defaultTarget;
      } else {
        $("#job-target").value = slurmAvailable ? SLURM_GPU_TARGET : "local_cpu";
      }
      updateJobTargetControls();
    }
    updateTargetOptionLabels(health);
    renderRuntimeStatus(health);
    renderSetupHealth(health);
    if (showMessage) showToast("Setup checklist refreshed.");
  } catch (error) {
    status.textContent = "Backend unavailable";
    detail.textContent = error.message;
    renderSetupHealth(null, error);
    if (showMessage) showToast(`Runtime check failed: ${error.message}`);
  }
}

function renderRuntimeStatus(health) {
  const status = $("#runtime-status");
  const detail = $("#runtime-detail");
  if (!status || !detail || !health) return;
  const target = resolvedTarget();
  const slurmAvailable = Boolean(health.slurm?.sbatch);
  const slurm = slurmAvailable ? "sbatch available" : "sbatch not found";
  const isSlurmGpu = isSlurmGpuTarget(target);
  const isOpenShift = isOpenShiftTarget(target);
  // Slurm tasks can load the image from the configured shared archive on the
  // compute node; it does not need to be pre-loaded in the GUI host's image
  // store.
  const image = state.engine === "diffsmol" ? health.diffsmol_image : health.container_image;
  const archiveReady = Boolean((state.engine === "diffsmol" ? health.diffsmol_archive : health.container_archive)?.exists);
  const imageReady = Boolean(image?.exists) || (isSlurmGpu && archiveReady);
  if (isOpenShift) {
    status.textContent = `${targetLabel({ target })} available`;
    detail.textContent = isOpenShiftJobTarget(target)
      ? openShiftSubmissionEnabled(health)
        ? `Job storage: ${health.environment?.job_root || "configured path"}. Generator pods launch as OpenShift Jobs and write results back to shared storage.`
        : `Job storage: ${health.environment?.job_root || "configured path"}. Manifest-only mode is active; enable submission to launch generator pods.`
      : `Job storage: ${health.environment?.job_root || "configured path"}. Diagnostics mode validates deployment storage, logs, and result loading.`;
    return;
  }
  status.textContent = isSlurmGpu ? (slurmAvailable ? "Slurm GPU available" : "Slurm setup needs attention") : imageReady ? "Local CPU available" : "Local setup needs attention";
  detail.textContent = isSlurmGpu
    ? `${slurm}; ${image?.exists ? "container image found" : archiveReady ? "shared container archive ready" : "container image not confirmed"}. Selected target: Slurm GPU.`
    : `${state.engine === "diffsmol" ? "DiffSMol" : "conDitar"} ${imageReady ? "image found" : "image missing"}. Selected target: Local CPU.`;
}

function openShiftSubmissionEnabled(health = state.runtimeHealth) {
  return Boolean(health?.environment?.openshift_submit);
}

function openShiftJobLabel(health = state.runtimeHealth) {
  return openShiftSubmissionEnabled(health) ? "OpenShift Job" : "OpenShift Job manifest";
}

function updateTargetOptionLabels(health = state.runtimeHealth) {
  const jobOption = $('#job-target option[value="openshift_job"]');
  const mockOption = $('#job-target option[value="openshift_mock"]');
  if (jobOption) jobOption.textContent = openShiftJobLabel(health);
  if (mockOption) mockOption.textContent = "OpenShift diagnostics";
}

function renderSetupHealth(health, error = null) {
  const panel = $("#setup-health-panel");
  const status = $("#setup-health-status");
  const detail = $("#setup-health-detail");
  const list = $("#setup-health-list");
  if (!status || !detail || !list) return;
  if (error) {
    panel?.setAttribute("data-status", "fail");
    if (panel) panel.open = true;
    status.textContent = "Backend unavailable";
    detail.textContent = error.message;
    list.innerHTML = "";
    return;
  }
  const checks = targetAwareHealthChecks(health);
  const failing = checks.filter((check) => check.status === "fail").length;
  const warnings = checks.filter((check) => check.status === "warn").length;
  panel?.setAttribute("data-status", failing ? "fail" : warnings ? "warn" : "ready");
  if (panel && failing) panel.open = true;
  const target = resolvedTarget();
  const isSlurmGpu = isSlurmGpuTarget(target);
  const label = targetLabel({ target });
  status.textContent = failing ? "Setup needs attention" : warnings ? "Ready with optional warnings" : "Ready to run";
  detail.textContent = failing
    ? `Fix the missing required items before submitting a ${label} job.`
    : warnings
      ? `${label} runs can work; optional Tool Chest items may need setup.`
      : `${label} launch requirements look ready.`;
  list.innerHTML = checks.map((check) => `
    <div class="setup-health-item" data-status="${escapeHtml(check.status)}">
      <i aria-hidden="true"></i>
      <div>
        <strong>${escapeHtml(check.label)}</strong>
        <span>${escapeHtml(check.detail || "")}</span>
        ${check.action ? `<small>${escapeHtml(check.action)}</small>` : ""}
      </div>
    </div>
  `).join("");
}

function targetAwareHealthChecks(health) {
  const target = resolvedTarget();
  const isSlurmGpu = isSlurmGpuTarget(target);
  const checks = (health?.checks || []).map((check) => {
    if (check.id !== "container_image" || state.engine !== "diffsmol") return { ...check };
    const image = health.diffsmol_image;
    return {
      ...check,
      label: "DiffSMol image",
      status: image?.exists ? "ok" : "fail",
      detail: image?.detail || `Image not found: ${health.diffsmol_image_name || "configured DiffSMol image"}`,
      action: image?.exists ? "" : "Load or build the DiffSMol image, then check again.",
    };
  });
  if (isOpenShiftTarget(target)) {
    const ids = isOpenShiftJobTarget(target)
      ? ["python", "job_storage", "openshift_job", "tool_chest"]
      : ["python", "job_storage", "openshift_mock", "tool_chest"];
    return checks.filter((check) => ids.includes(check.id));
  }
  if (!isSlurmGpu) {
    return checks.filter((check) => !["slurm", "openshift_mock", "openshift_job"].includes(check.id));
  }
  return checks.filter((check) => !["openshift_mock", "openshift_job"].includes(check.id)).map((check) => {
    const archive = state.engine === "diffsmol" ? health?.diffsmol_archive : health?.container_archive;
    if (check.id === "container_image" && check.status !== "ok" && archive?.exists) {
      return {
        ...check,
        status: "ok",
        detail: "Shared container archive is ready; the image will be loaded on the Slurm compute node.",
        action: "",
      };
    }
    if (check.id !== "slurm" || check.status === "ok") return check;
    return {
      ...check,
      status: "fail",
      detail: "sbatch not found for the selected Slurm GPU target",
      action: "Start the GUI from a cluster session with Slurm loaded, or switch to This computer · CPU.",
    };
  });
}

function resolvedTarget() {
  return $("#job-target").value;
}

function chooseFileAgain(selector) {
  const input = $(selector);
  input.value = "";
  input.click();
}

function clearFileInputBeforeChoose(event) {
  event.currentTarget.value = "";
}

async function setActiveTab(tab) {
  state.activeTab = tab;
  $$(".workflow-step").forEach((button) => button.classList.toggle("active", button.dataset.section === tab));
  $$(".workspace-section").forEach((section) => {
    section.hidden = section.id !== `${tab}-section`;
  });
  if (tab === "jobs") {
    await refreshJobs(false);
  }
  if (tab === "results") {
    renderCharts();
    renderSelectedStructure();
  }
  if (tab === "viewer") {
    renderMoleculeViewerWorkspace();
  }
}

async function loadExample(exampleId) {
  setLoading(true);
  state.exampleId = exampleId;
  state.customPdb = null;
  state.customSdf = null;
  state.stagedPreprocessManifest = null;
  renderSetupPreprocessSummary();
  state.batchInputs = [];
  updateCustomOptionLabel("");
  updateBatchLabel();
  const example = EXAMPLES[exampleId];
  if (!example) {
    state.exampleId = "custom";
    $("#example-select").value = "custom";
    updateInputLabels(null);
    showToast("No bundled inputs are included. Upload PDB/SDF inputs to run a job.");
    setLoading(false);
    return;
  }
  $("#example-select").value = exampleId;
  setMode(example.mode, false);
  updateInputLabels(example);
  try {
    state.study = await service.loadStudy(exampleId, (loaded, total) => {
      $("#hero-status").textContent = `${Math.round((loaded / total) * 100)}%`;
    });
    state.showReferenceLigand = false;
    state.selected = state.study.candidates[0] || null;
    state.exportSelection = new Set();
    state.exportFilters = {};
    state.resultSource = "example";
    state.selectedJob = null;
    $("#hero-candidate-count").textContent = state.study.candidates.length;
    $("#hero-status").textContent = "Ready";
    renderStudy();
  } catch (error) {
    showToast(error.message);
    $("#hero-status").textContent = "Error";
  } finally {
    setLoading(false);
  }
}

function setMode(mode, updateSelect = true) {
  if (state.engine === "diffsmol") mode = "reference";
  state.mode = mode;
  $$(".mode-toggle button").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $("#sdf-dropzone").hidden = mode === "pocket" && state.engine !== "diffsmol";
  const pocketRadiusField = $("[data-parameter-key='pocket_radius']");
  if (pocketRadiusField) pocketRadiusField.hidden = state.engine === "diffsmol" || mode === "pocket";
  const guidanceField = $("[data-parameter-key='diffsmol_guidance']");
  if (guidanceField) guidanceField.hidden = state.engine !== "diffsmol";
  $("#mode-note").textContent = mode === "reference"
    ? "The reference ligand defines the generation center. Pocket radius controls the surrounding protein context."
    : "The prepared pocket PDB supplies the generation region without a reference ligand.";
  $("#hero-input-mode").textContent = mode === "reference" ? "Ligand" : "Pocket";
  if (updateSelect) {
    if (state.exampleId !== "custom") {
      state.exampleId = "custom";
      state.study = null;
      state.selected = null;
      state.exportFilters = {};
      state.resultSource = "upload";
      updateInputLabels(null);
    }
    $("#example-select").value = "custom";
  }
  if (state.engine === "diffsmol") {
    $("#mode-note").textContent = "DiffSMol uses a 3D reference ligand SDF; add a pocket PDB for pocket-conditioned generation.";
    $("#hero-input-mode").textContent = "Shape";
  }
  updateRunSummary();
}

function updateInputLabels(example) {
  $("#pdb-name").textContent = example ? example.pdb.split("/").pop() : "Choose a PDB file";
  $("#pdb-detail").textContent = example ? `${example.pdbRecords} · bundled input` : "Required · uploaded with this job";
  $("#sdf-name").textContent = example?.sdf ? example.sdf.split("/").pop() : "Choose a reference SDF";
  $("#sdf-detail").textContent = example?.sdf ? "Reference ligand · bundled input" : "Required for protein + ligand mode";
}

function clearLoadedStudyForCustomInput() {
  if (!state.study && !state.selected) return;
  state.study = null;
  state.selected = null;
  state.exportSelection = new Set();
  state.exportFilters = {};
  state.resultSource = "upload";
  renderExportFilters();
  renderResultsTable();
}

function setupStudyInput() {
  return state.resultSource === "example" ? state.study : null;
}

function setupPdbInput(inputOverride = null) {
  if (inputOverride) return inputOverride.pdb || null;
  const study = setupStudyInput();
  return state.customPdb || (study?.pdbText ? {
    name: study.example?.pdb?.split("/").pop() || "input.pdb",
    text: study.pdbText,
  } : null);
}

function setupSdfInput(inputOverride = null) {
  if (inputOverride) return inputOverride.sdf || null;
  const study = setupStudyInput();
  return state.customSdf || (study?.referenceSdf ? {
    name: study.example?.sdf?.split("/").pop() || "reference.sdf",
    text: study.referenceSdf,
  } : null);
}

function currentGenerationInput(inputOverride = null) {
  const engine = state.engine === "diffsmol" ? "diffsmol" : "conditar";
  const mode = engine === "diffsmol" ? "reference" : state.mode;
  const pdb = setupPdbInput(inputOverride);
  const sdf = engine === "diffsmol" || mode === "reference"
    ? setupSdfInput(inputOverride)
    : null;
  const workflow = resolveGenerationWorkflow({ engine, mode, pdb, sdf });
  return {
    engine,
    mode,
    workflow,
    pdb,
    sdf,
    hasPdb: Boolean(pdb?.text),
    hasSdf: Boolean(sdf?.text),
    batchCount: state.batchInputs.length,
    inputName: inputOverride?.name || (engine === "diffsmol" ? sdf?.name : pdb?.name) || state.exampleId,
    preprocess: inputOverride ? null : state.stagedPreprocessManifest,
  };
}

function renderStudy() {
  renderSummary();
  renderToolChest();
  renderExportFilters();
  renderHistogramMetricOptions();
  applyExportFilters(false);
  renderResultsTable();
  renderCharts();
  renderSelectedStructure();
  renderViewerSelectors();
  updateResultsSource();
  updateRunSummary();
}

function setEngine(engine) {
  state.engine = engine === "diffsmol" ? "diffsmol" : "conditar";
  if (state.engine === "diffsmol" && state.mode !== "reference") {
    setMode("reference", false);
  }
  updateEngineControls();
  if (state.runtimeHealth) {
    renderRuntimeStatus(state.runtimeHealth);
    renderSetupHealth(state.runtimeHealth);
  }
  updateVinaControls();
  updateRunSummary();
}

function updateEngineControls() {
  const isDiffSmol = state.engine === "diffsmol";
  const generationInput = currentGenerationInput();
  const requirements = requiredInputs(state.engine, state.mode, generationInput);
  const hasPocketContext = generationInput.workflow.id === "diffsmol_pocket";
  document.body.classList.toggle("diffsmol-engine", isDiffSmol);
  $$(".engine-option").forEach((button) => button.classList.toggle("active", button.dataset.engine === state.engine));
  $(".mode-toggle")?.classList.toggle("is-disabled", isDiffSmol);
  $("#pdb-dropzone").hidden = false;
  $("#sdf-dropzone").hidden = state.mode === "pocket" && !isDiffSmol;
  $("#pdb-detail").textContent = isDiffSmol
    ? (hasPocketContext ? "Pocket-conditioned generation will use this pocket" : "Optional: add a pocket PDB for pocket-conditioned generation")
    : "Required · uploaded with this job";
  $("#sdf-detail").textContent = isDiffSmol
    ? "Required 3D reference ligand for shape expansion"
    : (state.customSdf ? "Reference ligand · local upload" : "Required for protein + ligand mode");
  const guidanceField = $("[data-parameter-key='diffsmol_guidance']");
  if (guidanceField) guidanceField.hidden = !isDiffSmol;
  const pocketRadiusField = $("[data-parameter-key='pocket_radius']");
  if (pocketRadiusField) pocketRadiusField.hidden = isDiffSmol || state.mode === "pocket";
  const batchSizeField = $("[data-parameter-key='batch_size']");
  if (batchSizeField) batchSizeField.querySelector("small").textContent = isDiffSmol
    ? "Samples processed per DiffSMol batch"
    : "Samples processed per conDitar batch";
  const headingLabel = $(".input-panel .required-label");
  if (headingLabel) {
    headingLabel.textContent = requirements.sdf === "required" && requirements.pdb === "required"
      ? "PDB + SDF required"
      : requirements.sdf === "required"
      ? "SDF required"
      : "PDB required";
  }
  if (isDiffSmol) {
    $("#mode-note").textContent = hasPocketContext
      ? "DiffSMol will run pocket-conditioned generation from the attached pocket and 3D ligand shape."
      : "DiffSMol ligand-only mode will not produce reliable docking scores. Attach a pocket PDB if you want pocket-conditioned generation and Vina/QVina post-processing.";
    $("#hero-input-mode").textContent = hasPocketContext ? "Shape + Pocket" : "Ligand Shape";
  } else {
    setMode(state.mode, false);
  }
  updateBuiltinEvaluationAvailability();
  updateBatchLabel();
}

async function submitGenerationJob() {
  if (shouldTreatCurrentInputAsDiffSmol()) {
    setEngine("diffsmol");
    showToast("SDF-only input selected; using DiffSMol shape generation.");
  }
  const validation = validateGenerationSetup({
    ...currentGenerationInput(),
    batchCount: state.batchInputs.length,
  });
  if (!validation.ready) {
    showToast(validation.errors[0]);
    return;
  }
  const button = $("#preview-run");
  button.disabled = true;
  button.querySelector("span").textContent = "Submitting";
  try {
    const payload = state.batchInputs.length ? buildBatchPayload() : buildJobPayload();
    await prepareLocalNotifications(payload);
    const response = state.batchInputs.length ? await service.submitBatch(payload) : { jobs: [await service.submitJob(payload)], errors: [] };
    response.jobs.forEach((item) => {
      if (usesBrowserNotifications(item.target) && !isTerminalJob(item)) state.watchedJobs.add(item.id);
    });
    const job = response.jobs[0];
    state.currentJob = job;
    state.selectedJob = job;
    if (response.jobs.some((item) => item.status !== "failed")) resetEvaluationSelections();
    const failedJobs = response.jobs.filter((item) => item.status === "failed");
    const queuedJobs = response.jobs.length - failedJobs.length;
    const message = failedJobs.length
      ? `${queuedJobs} queued, ${failedJobs.length} failed. ${failedJobs[0].error_message || "See the selected job logs."}`
      : response.jobs.length > 1
        ? `${response.jobs.length} ${batchTargetNoun(job?.target)}${response.errors.length ? `, ${response.errors.length} skipped: ${response.errors[0].error}` : ""}.`
        : "Job queued.";
    updateJobPanel(job, message);
    updateJobDetail(job, message);
    await refreshJobs(false);
    setActiveTab("jobs");
    showToast(failedJobs.length ? message : response.jobs.length > 1 ? message : `${targetLabel(job)} job queued.`);
    if (response.errors.length) console.warn("Batch submission errors", response.errors);
    scheduleJobsRefresh();
    if (job.status !== "failed") pollJob(job.id);
  } catch (error) {
    showToast(error.message);
    updateJobPanel(null, error.message);
  } finally {
    button.disabled = false;
    updateBatchLabel();
  }
}

function shouldTreatCurrentInputAsDiffSmol() {
  return state.engine !== "diffsmol"
    && !state.batchInputs.length
    && Boolean(setupSdfInput())
    && !setupPdbInput();
}

function buildJobPayload(inputOverride = null) {
  const generationInput = currentGenerationInput(inputOverride);
  return {
    engine: generationInput.engine,
    target: resolvedTarget(),
    mode: generationInput.mode,
    example_id: state.exampleId,
    input_name: generationInput.inputName,
    email: $("#job-email").disabled ? "" : $("#job-email").value.trim(),
    pdb: generationInput.pdb,
    sdf: generationInput.sdf,
    preprocess: generationInput.preprocess,
    slurm: buildSlurmPayload(),
    postprocess: buildPostprocessPayload(inputOverride),
    tools: buildEvaluationToolsPayload(),
    parameters: {
      ...state.parameters,
      device: isSlurmGpuTarget(resolvedTarget()) || isOpenShiftJobTarget(resolvedTarget()) ? "cuda:0" : "cpu",
    },
  };
}

function buildBatchPayload() {
  return {
    jobs: state.batchInputs.map((input) => buildJobPayload(input)),
  };
}

function buildSlurmPayload() {
  return {
    time: $("#slurm-time").value.trim(),
    mem: $("#slurm-mem").value.trim(),
    cpus: $("#slurm-cpus").value,
    gpus: $("#slurm-gpus").value,
    partition: $("#slurm-partition").value.trim(),
    account: $("#slurm-account").value.trim(),
  };
}

function buildPostprocessPayload(inputOverride = null) {
  const generationInput = currentGenerationInput(inputOverride);
  let selected = selectedBuiltinEvaluations({ includeDisabled: true });
  selected = filterMetricsForWorkflow(selected, generationInput.workflow);
  const vinaMode = selectedVinaMode(selected);
  return {
    vina: vinaMode !== "none",
    vina_mode: vinaMode,
    vina_exhaustiveness: $("#vina-exhaustiveness").value,
    vina_cpu: $("#vina-cpu").value,
    metrics: selected,
  };
}

function buildEvaluationToolsPayload() {
  return $$(".evaluation-tool-toggle:checked").map((input) => ({
    id: input.dataset.toolId,
    options: {},
  }));
}

function selectedBuiltinEvaluations(options = {}) {
  const selector = options.includeDisabled ? ".builtin-evaluation-toggle:checked" : ".builtin-evaluation-toggle:checked:not(:disabled)";
  return $$(selector).map((input) => input.value);
}

function hasDiffSmolPocketContext(inputOverride = null) {
  return currentGenerationInput(inputOverride).workflow.id === "diffsmol_pocket";
}

function selectedVinaMode(selected = selectedBuiltinEvaluations()) {
  return vinaModeForMetrics(selected);
}

function updateRunEvaluationSummary() {
  const container = $("#run-evaluations");
  if (!container) return;
  const jobs = state.batchInputs.length ? buildBatchPayload().jobs : [buildJobPayload()];
  const summaries = jobs.map((job) => [
    ...(job.postprocess.metrics || []).map((metric) => EVALUATION_LABELS[metric] || metric),
    ...job.tools.map((request) => state.tools.find((tool) => tool.id === request.id)?.name || request.id),
  ].join(", "));
  const unique = [...new Set(summaries)];
  container.textContent = unique.length > 1
    ? "Evaluations vary by input; each job will use its eligible selections."
    : unique[0]
      ? `This job will run: ${unique[0]}.`
      : "This job will run generation only; no evaluators are selected.";
}

function resetEvaluationSelections() {
  $$(".builtin-evaluation-toggle").forEach((input) => {
    input.checked = DEFAULT_BUILTIN_EVALUATIONS.has(input.value);
  });
  $$(".evaluation-tool-toggle").forEach((input) => {
    input.checked = false;
  });
  updateVinaControls();
}

function selectedEvaluationLabel(selected = selectedBuiltinEvaluations()) {
  if (!selected.length) return "None";
  const mode = selectedVinaMode(selected);
  const labels = [];
  if (mode === "all") labels.push("Vina + QVina");
  else if (mode === "vina_dock") labels.push("Vina redock");
  else if (mode === "qvina") labels.push("QVina");
  else if (mode === "vina_score") labels.push("Vina score");
  const chemistry = selected.filter((item) => ["qed", "sa", "logp", "lipinski"].includes(item)).length;
  if (chemistry) labels.push(`${chemistry} chemistry metric${chemistry === 1 ? "" : "s"}`);
  return labels.join(" + ") || "Chemistry only";
}

async function pollJob(jobId) {
  clearTimeout(state.jobPollTimer);
  try {
    const job = await service.getJob(jobId);
    const logs = await service.getJobLogs(jobId).catch(() => ({ stdout: "", stderr: "" }));
    const logText = combineLogs(logs);
    state.currentJob = job;
    // Keep polling the running job, but do not steal the user's selected log
    // view when they are inspecting a different job.
    const isSelected = !state.selectedJob || state.selectedJob.id === job.id;
    if (isSelected) {
      state.selectedJob = job;
      updateJobPanel(job, logText || "Waiting for job output.");
      updateJobDetail(job, logText || "Waiting for job output.", logs);
    }
    renderJobsTable();
    if (isResultsReadyJob(job)) {
      notifyJobTerminal(job, "completed");
      await refreshJobs(false);
      if (isSelected) await loadCompletedJob(job);
      return;
    }
    if (CLEANUP_JOB_STATUSES.has(job.status)) {
      notifyJobTerminal(job, job.status);
      showToast(job.error_message || `Job ${job.status}.`);
      return;
    }
    state.jobPollTimer = setTimeout(() => pollJob(jobId), 5000);
  } catch (error) {
    updateJobPanel(state.currentJob, error.message);
    state.jobPollTimer = setTimeout(() => pollJob(jobId), 5000);
  }
}

async function loadCompletedJob(job) {
  const result = await service.loadJobResults(job);
  const resultLogText = combineLogs(result.logs || {});
  const candidates = result.candidates || [];
  const loadedJob = result.job || job;
  if (!candidates.length) {
    state.selectedJob = loadedJob;
    updateJobDetail(state.selectedJob, resultLogText || "No SDF files were found in the job output directory.", result.logs || {});
    showToast(state.selectedJob?.error_message || "No SDF results were found for this job.");
    setActiveTab("jobs");
    return;
  }
  const vinaFailures = candidates.filter((item) => String(item.properties?.VINA_STATUS || "").toLowerCase() === "failed");
  const pdbInput = result.inputs?.pdb || null;
  const sdfInput = result.inputs?.sdf || null;
  state.study = {
    example: {
      id: loadedJob.id,
      label: loadedJob.id,
      pdb: pdbInput?.name || null,
      sdf: sdfInput?.name || null,
    },
    pdbText: pdbInput?.text || "",
    referenceSdf: sdfInput?.text || null,
    candidates,
    artifacts: result.artifacts || [],
    logs: result.logs || {},
    summary: result.summary || {},
    toolRuns: result.toolRuns || [],
    loadedJob,
  };
  state.showReferenceLigand = false;
  state.currentJob = loadedJob;
  state.selectedJob = loadedJob;
  state.resultSource = "job";
  state.selected = candidates[0];
  state.exportSelection = new Set();
  state.exportFilters = {};
  $("#hero-candidate-count").textContent = candidates.length;
  $("#hero-status").textContent = "Completed";
  renderStudy();
  setActiveTab("results");
  showToast(vinaFailures.length
    ? `${candidates.length} result${candidates.length === 1 ? "" : "s"} loaded; ${vinaFailures.length} molecule${vinaFailures.length === 1 ? "" : "s"} had incomplete Vina annotations.`
    : `Job completed with ${candidates.length} result${candidates.length === 1 ? "" : "s"}.`);
}

function updateJobPanel(job, logText) {
  $("#job-status").textContent = job?.status || "Idle";
  $("#job-id").textContent = job?.id || "None";
  $("#job-log").textContent = trimLog(logText || "No job submitted.");
}

function updateJobDetail(job, logText, logs = null) {
  $("#job-detail-status").textContent = job?.status || "None";
  $("#job-detail-status").dataset.status = job?.status || "none";
  $("#job-detail-id").textContent = job?.id || "None";
  $("#job-detail-target").textContent = targetLabel(job);
  $("#job-detail-started").textContent = formatDate(job?.started_at || job?.created_at);
  renderJobProvenance(job);
  const note = job?.status_note ? `${job.status_note}\n\n` : "";
  const error = job?.error_message ? `Error: ${job.error_message}\n\n` : "";
  const fallback = job ? jobPaths(job) : null;
  const pathText = fallback ? `Paths:\nstdout: ${fallback.stdout}\nstderr: ${fallback.stderr}\noutputs: ${fallback.outputs}\n\n` : "";
  const renderedLog = logText || (logs && (logs.stdout || logs.stderr || logs.extra) ? combineLogs(logs) : "");
  $("#job-detail-log").textContent = trimLog(note + error + pathText + (renderedLog || "Select a job to view logs."));
  renderJobAlert(job, fallback);
}

async function refreshJobs(showMessage = false) {
  try {
    state.jobs = await service.listJobs();
    notifyWatchedTerminalJobs(state.jobs);
    renderJobsTable();
    scheduleJobsRefresh();
    if (showMessage) showToast(`Loaded ${state.jobs.length} job${state.jobs.length === 1 ? "" : "s"}.`);
  } catch (error) {
    if (showMessage) showToast(error.message);
  }
}

function scheduleJobsRefresh() {
  clearTimeout(state.jobsRefreshTimer);
  if (!state.jobs.some(isActiveJob)) return;
  state.jobsRefreshTimer = setTimeout(() => refreshJobs(false), 7000);
}

function renderJobsTable() {
  const jobs = [...state.jobs]
    .filter((job) => {
      if (state.jobFilter === "all") return true;
      if (state.jobFilter === "active") return isActiveJob(job);
      return job.status === state.jobFilter;
    })
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  $("#jobs-table").innerHTML = jobs.length ? jobs.map((job) => `
    <tr data-job-id="${escapeHtml(job.id)}" class="${state.selectedJob?.id === job.id ? "active" : ""}">
      <td>${escapeHtml(shortJobId(job.id))}<br><small title="${escapeHtml(job.id)}">${escapeHtml(engineLabel(job))} · ${escapeHtml(conditioningLabel(job))}</small></td>
      <td><span class="status-badge" data-status="${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>${job.status_note ? `<br><small>${escapeHtml(job.status_note)}</small>` : ""}</td>
      <td>${escapeHtml(targetLabel(job))}<br><small>${escapeHtml(inputLabel(job))}</small></td>
      <td>${formatDate(job.created_at)}<br><small>${escapeHtml(slurmLabel(job))}</small></td>
      <td>
        ${isResultsReadyJob(job) ? `<button class="secondary-button compact-action load-job-results">Results</button>` : ""}
        ${isActiveJob(job) ? `<button class="secondary-button compact-action cancel-job">Cancel</button>` : ""}
        ${CLEANUP_JOB_STATUSES.has(job.status) ? `<button class="secondary-button compact-action rerun-job">Rerun</button>` : ""}
        ${CLEANUP_JOB_STATUSES.has(job.status) ? `<button class="secondary-button compact-action danger-action cleanup-job">Clean up</button>` : ""}
      </td>
    </tr>`).join("") : `<tr><td colspan="5">No jobs yet.</td></tr>`;

  $$("#jobs-table tr[data-job-id]").forEach((row) => row.addEventListener("click", async (event) => {
    const job = state.jobs.find((item) => item.id === row.dataset.jobId);
    if (!job) return;
    state.selectedJob = job;
    renderJobsTable();
    if (event.target.closest(".cancel-job")) {
      await cancelJob(job.id);
      return;
    }
    if (event.target.closest(".cleanup-job")) {
      await cleanupJob(job.id);
      return;
    }
    if (event.target.closest(".rerun-job")) {
      await rerunJob(job.id);
      return;
    }
    const logs = await service.getJobLogs(job.id).catch(() => ({ stdout: "", stderr: "" }));
    updateJobDetail(job, combineLogs(logs) || "Logs are not available for this job yet.", logs);
    if (event.target.closest(".load-job-results")) {
      await loadSelectedJobResults(job.id);
    }
  }));
}

async function cancelJob(jobId) {
  try {
    const body = await service.cancelJob(jobId);
    state.selectedJob = body.job;
    await refreshJobs(false);
    updateJobDetail(body.job, "Cancel requested.");
    showToast("Job canceled.");
  } catch (error) {
    showToast(error.message);
  }
}

async function cleanupJob(jobId) {
  try {
    const body = await service.archiveJob(jobId);
    if (state.selectedJob?.id === jobId) {
      state.selectedJob = null;
      updateJobDetail(null, "Cleaned up failed/canceled job.");
    }
    await refreshJobs(false);
    showToast(`Cleaned up ${shortJobId(body.job?.id || jobId)}.`);
  } catch (error) {
    showToast(error.message);
  }
}

async function rerunJob(jobId) {
  try {
    let body;
    try {
      body = await service.rerunJob(jobId);
    } catch (error) {
      if (!/Unknown API endpoint/i.test(error.message)) throw error;
      body = { job: await submitRerunFromSavedInputs(jobId) };
    }
    const job = body.job;
    state.currentJob = job;
    state.selectedJob = job;
    if (usesBrowserNotifications(job.target)) state.watchedJobs.add(job.id);
    await refreshJobs(false);
    updateJobPanel(job, "Rerun queued.");
    updateJobDetail(job, `Rerun created from ${jobId}.`);
    showToast(`Rerun queued as ${shortJobId(job.id)}.`);
    scheduleJobsRefresh();
    if (!isTerminalJob(job)) pollJob(job.id);
  } catch (error) {
    showToast(error.message);
  }
}

async function submitRerunFromSavedInputs(jobId) {
  const original = await service.getJob(jobId);
  const saved = await service.loadJobResults(original);
  if (original.inputs?.pdb && !saved.inputs?.pdb?.text) throw new Error("Original PDB input was not found for rerun.");
  if (original.inputs?.sdf && !saved.inputs?.sdf?.text) throw new Error("Original SDF input was not found for rerun.");
  const payload = {
    target: original.target || "local_cpu",
    engine: original.engine || "conditar",
    mode: original.mode || (saved.inputs.sdf ? "reference" : "pocket"),
    example_id: original.example_id || "custom",
    input_name: `rerun_${original.input_name || saved.inputs.pdb?.name || saved.inputs.sdf?.name || jobId}`,
    email: original.email || "",
    pdb: saved.inputs.pdb?.text ? { name: saved.inputs.pdb.name, text: saved.inputs.pdb.text } : null,
    sdf: saved.inputs.sdf?.text ? { name: saved.inputs.sdf.name, text: saved.inputs.sdf.text } : null,
    preprocess: original.preprocess || null,
    slurm: original.slurm || {},
    postprocess: original.postprocess || {},
    tools: (original.tools || []).map(({ id, options }) => ({ id, options })),
    parameters: original.parameters || {},
    rerun_of: jobId,
  };
  return service.submitJob(payload);
}

async function loadSelectedJobResults(jobId) {
  const job = await service.getJob(jobId);
  state.selectedJob = job;
  updateJobDetail(job, "Loading results...");
  if (!isResultsReadyJob(job)) {
    showToast(isBuiltinPostprocessRunning(job) ? "Post-processing is still running. Results will be ready after annotation finishes." : "Only completed jobs have results to load.");
    return;
  }
  await loadCompletedJob(job);
}

function trimLog(text) {
  return text.length > 5000 ? `…\n${text.slice(-5000)}` : text;
}

function combineLogs(logs = {}) {
  const sections = [];
  if (logs.stdout) sections.push(`STDOUT\n${logs.stdout}`);
  if (logs.stderr) sections.push(`STDERR\n${logs.stderr}`);
  if (logs.extra) sections.push(`ADDITIONAL LOGS\n${logs.extra}`);
  return sections.join("\n\n");
}

function jobPaths(job) {
  if (!job?.id) return null;
  const base = `job_data/jobs/${job.id}`;
  const outputDirectory = job.outputs?.directory || "outputs";
  return {
    stdout: `${base}/logs/stdout.log`,
    stderr: `${base}/logs/stderr.log`,
    outputs: `${base}/${outputDirectory}`,
  };
}

function renderJobAlert(job, paths) {
  const alert = $("#job-detail-alert");
  if (!alert) return;
  if (!job || !["failed", "canceled"].includes(job.status)) {
    alert.hidden = true;
    alert.innerHTML = "";
    return;
  }
  const title = job.status === "failed" ? "Run failed" : "Run canceled";
  alert.hidden = false;
  alert.innerHTML = `
    <strong>${title}</strong>
    <div>${escapeHtml(job.error_message || job.status_note || "Review the logs below for details.")}</div>
    ${paths ? `<code>${escapeHtml(paths.stderr)}</code><code>${escapeHtml(paths.stdout)}</code><code>${escapeHtml(paths.outputs)}</code>` : ""}
  `;
}

async function prepareLocalNotifications(payload) {
  const jobs = payload.jobs || [payload];
  if (!jobs.some((job) => usesBrowserNotifications(job.target))) return;
  if (!("Notification" in window) || Notification.permission !== "default") return;
  try {
    await Notification.requestPermission();
  } catch {
    // Browser notifications are optional; toast updates still work.
  }
}

function notifyJobTerminal(job, status) {
  if (!job?.id || state.notifiedJobs.has(job.id)) return;
  if (!usesBrowserNotifications(job.target)) return;
  state.notifiedJobs.add(job.id);
  const title = status === "completed" ? "conDitar run completed" : `conDitar run ${status}`;
  const body = `${inputLabel(job)} · ${targetLabel(job)} · ${shortJobId(job.id)}`;
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification(title, { body });
  }
}

function notifyWatchedTerminalJobs(jobs) {
  jobs.forEach((job) => {
    if (!state.watchedJobs.has(job.id) || !["completed", "failed", "canceled"].includes(job.status)) return;
    notifyJobTerminal(job, job.status);
    state.watchedJobs.delete(job.id);
  });
}

function updateRunEstimate() {
  const estimate = $("#run-estimate");
  if (!estimate) return;
  const job = state.batchInputs.length ? buildBatchPayload().jobs[0] : buildJobPayload();
  const hasSingleInput = job.engine === "diffsmol" ? Boolean(job.sdf) : Boolean(job.pdb);
  const inputs = state.batchInputs.length || (hasSingleInput ? 1 : 0);
  const samples = Math.max(1, Number(job.parameters.num_samples) || 1);
  const totalSamples = inputs * samples;
  const target = job.target;
  const isGpu = isSlurmGpuTarget(target);
  const isOpenShift = isOpenShiftTarget(target);
  if (!inputs) {
    estimate.textContent = "Estimate updates after you choose inputs.";
    return;
  }
  if (isOpenShift) {
    estimate.textContent = isOpenShiftJobTarget(target)
      ? openShiftSubmissionEnabled()
        ? `OpenShift will launch ${inputs} generator Job${inputs === 1 ? "" : "s"} for ${totalSamples} requested sample${totalSamples === 1 ? "" : "s"}.`
        : `OpenShift manifest mode will write Kubernetes Job artifacts for ${inputs} input${inputs === 1 ? "" : "s"} without submitting them.`
      : `OpenShift diagnostics will create mock outputs for ${inputs} input${inputs === 1 ? "" : "s"} and validate deployment plumbing.`;
    return;
  }
  const minutesPerSample = isGpu ? 1.5 : 5.5;
  const concurrencyNote = isGpu
    ? "Slurm GPU jobs can run in parallel once scheduled."
    : "Local CPU jobs run serially; keep this server window open.";
  const engineLabel = job.engine === "diffsmol" ? "DiffSMol" : "conDitar";
  estimate.textContent = `Rule-of-thumb runtime: about ${formatDuration(totalSamples * minutesPerSample)} for ${inputs} ${engineLabel} input${inputs === 1 ? "" : "s"} × ${samples} sample${samples === 1 ? "" : "s"} on ${isGpu ? "Slurm GPU" : "local CPU"}. ${concurrencyNote}`;
}

function formatDuration(minutes) {
  if (minutes < 90) return `${Math.round(minutes)} min`;
  return `${(minutes / 60).toFixed(minutes < 600 ? 1 : 0)} hr`;
}

function renderSummary() {
  const candidates = state.study?.candidates || [];
  const average = (key) => {
    const values = numericValues(candidates, key);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const cards = [
    ["Loaded structures", candidates.length, "SDF"],
    ["Mean molecular weight", formatMetric(average("molecularWeight"), 1), "Da"],
    ["Mean heavy atoms", formatMetric(average("heavyAtoms"), 1), "atoms"],
  ];
  const vinaScoreValues = candidates.map((item) => propertyMetric(item, "VINA_SCORE_ONLY")).filter(Number.isFinite);
  if (vinaScoreValues.length) cards[2] = ["Mean Vina score", formatMetric(mean(vinaScoreValues)), "kcal/mol"];
  $("#metric-strip").innerHTML = cards.map(([label, value, unit]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong><small>${unit}</small></div>`).join("");
  renderResultsProvenance();
  renderQualitySummary(candidates);
}

function renderQualitySummary(candidates) {
  const vinaScoreValues = candidates.map((item) => propertyMetric(item, "VINA_SCORE_ONLY")).filter(Number.isFinite);
  const vinaMinimizeValues = candidates.map((item) => propertyMetric(item, "VINA_MINIMIZE")).filter(Number.isFinite);
  const vinaDockValues = candidates.map((item) => propertyMetric(item, "VINA_DOCK")).filter(Number.isFinite);
  const qedValues = propertyValues(candidates, "QED");
  const saValues = propertyValues(candidates, "SA");
  const logpValues = propertyValues(candidates, "LOGP");
  const lipinskiValues = propertyValues(candidates, "LIPINSKI");
  const scored = candidates.filter((item) => propertyMetric(item, "VINA_SCORE_ONLY") !== null).length;
  const minimized = candidates.filter((item) => propertyMetric(item, "VINA_MINIMIZE") !== null).length;
  const docked = candidates.filter((item) => propertyMetric(item, "VINA_DOCK") !== null).length;
  const lipinskiPasses = lipinskiValues.filter((value) => value >= 4).length;
  const rows = [
    ["Vina scored", scored || minimized || docked ? `${scored} score · ${minimized} min · ${docked} redock` : "Not run"],
    ["Best affinity", bestValueLabel(vinaDockValues.length ? vinaDockValues : vinaMinimizeValues.length ? vinaMinimizeValues : vinaScoreValues)],
    ["QED", rangeLabel(qedValues)],
    ["SA", rangeLabel(saValues)],
    ["LogP", rangeLabel(logpValues)],
    ["Lipinski >=4", lipinskiValues.length ? `${lipinskiPasses}/${lipinskiValues.length}` : "n/a"],
    ...toolQualityRows(candidates).slice(0, 2),
  ].filter(([, value]) => value !== "n/a" && value !== "Not run");
  if (!rows.length) rows.push(["Summary", candidates.length ? `${candidates.length} candidates loaded` : "No results loaded"]);
  $("#quality-summary").innerHTML = rows.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderToolChest() {
  const list = $("#tool-list");
  const status = $("#tool-chest-status");
  if (!list || !status) return;
  const tools = state.tools || [];
  const availableTools = tools.filter((tool) => tool.available).length;
  const job = loadedResultsJob();
  const isCompletedJob = isResultsReadyJob(job);
  const runs = state.study?.toolRuns || job?.tool_runs || [];
  status.textContent = !state.toolsLoaded
    ? "Tools unavailable"
    : isCompletedJob
      ? `${availableTools} tool${availableTools === 1 ? "" : "s"} available`
      : "Load a completed job";
  if (!tools.length) {
    list.innerHTML = `<p class="tool-empty">${state.toolsLoaded ? "No tools are installed in gui/tools." : "Tool discovery failed."}</p>`;
    return;
  }
  list.innerHTML = tools.map((tool) => {
    const toolRuns = runs.filter((run) => run.tool_id === tool.id);
    const lastRun = toolRuns[toolRuns.length - 1];
    const disabledReason = !isCompletedJob
      ? "Load a completed job to run tools."
      : !tool.available
        ? (tool.error || "This tool is unavailable.")
        : "";
    return `
      <div class="tool-card" data-tool-id="${escapeHtml(tool.id)}">
        <div class="tool-card-main">
          <div>
            <h4>${escapeHtml(tool.name || tool.id)}</h4>
            <p>${escapeHtml(tool.description || "")}</p>
          </div>
          <span class="status-badge" data-status="${tool.available ? "completed" : "failed"}">${tool.available ? "Ready" : "Unavailable"}</span>
        </div>
        ${disabledReason ? `<p class="tool-warning">${escapeHtml(disabledReason)}</p>` : ""}
        ${tool.inputs?.length ? `<div class="tool-options">${tool.inputs.map((input) => toolOptionControl(tool, input, "results")).join("")}</div>` : ""}
        <div class="tool-card-actions">
          <button class="secondary-button compact-action tool-run-button" data-tool-id="${escapeHtml(tool.id)}" ${disabledReason ? "disabled" : ""}>Run tool</button>
          <span>${lastRun ? toolRunSummary(lastRun) : "Not run on this job"}</span>
        </div>
      </div>
    `;
  }).join("");
  $$(".tool-run-button").forEach((button) => button.addEventListener("click", () => runTool(button.dataset.toolId)));
}

function renderEvaluationTools() {
  const list = $("#evaluation-tool-list");
  const status = $("#evaluation-tool-status");
  if (!list || !status) return;
  const tools = state.tools || [];
  status.textContent = !state.toolsLoaded
    ? "Tools unavailable"
    : tools.length
      ? `${tools.length} tool${tools.length === 1 ? "" : "s"}`
      : "No tools";
  if (!tools.length) {
    list.innerHTML = `<p class="tool-empty">${state.toolsLoaded ? "No Tool Chest evaluators are installed." : "Tool discovery failed."}</p>`;
    return;
  }
  list.innerHTML = tools.map((tool) => `
    <div class="evaluation-tool-card" data-tool-id="${escapeHtml(tool.id)}">
      <label class="check-control">
        <input class="evaluation-tool-toggle" type="checkbox" data-tool-id="${escapeHtml(tool.id)}" ${tool.available ? "" : "disabled"}>
        <span>${escapeHtml(tool.name || tool.id)}</span>
      </label>
      <small>${escapeHtml(tool.available ? (tool.description || "") : (tool.error || "Tool unavailable."))}</small>
    </div>
  `).join("");
  $$(".evaluation-tool-toggle").forEach((input) => input.addEventListener("change", updateRunEvaluationSummary));
  updateRunEvaluationSummary();
}

function toolOptionControl(tool, input, context = "results") {
  const inputId = `tool-${context}-${tool.id}-${input.name}`;
  if (input.type === "boolean") {
    return `
      <label class="check-control" for="${escapeHtml(inputId)}">
        <input id="${escapeHtml(inputId)}" data-tool-option="${escapeHtml(input.name)}" type="checkbox" ${input.default ? "checked" : ""}>
        ${escapeHtml(input.label || input.name)}
      </label>
    `;
  }
  if (input.type === "number") {
    return `
      <label class="tool-field" for="${escapeHtml(inputId)}">${escapeHtml(input.label || input.name)}
        <input id="${escapeHtml(inputId)}" data-tool-option="${escapeHtml(input.name)}" type="number" value="${escapeHtml(input.default ?? "")}">
      </label>
    `;
  }
  return `
    <label class="tool-field" for="${escapeHtml(inputId)}">${escapeHtml(input.label || input.name)}
      <input id="${escapeHtml(inputId)}" data-tool-option="${escapeHtml(input.name)}" type="text" value="${escapeHtml(input.default ?? "")}">
    </label>
  `;
}

function toolRunSummary(run) {
  const result = run.result || {};
  if (run.status === "completed" && state.study?.loadedJob?.engine === "diffsmol"
    && state.study?.referenceSdf && Number.isFinite(Number(result.molecules))
    && !Array.isArray(result.generated_sdfs)) {
    return "Earlier run may include the reference ligand; rerun for corrected totals.";
  }
  if (run.tool_id === "medchem_filters" && Number.isFinite(Number(result.molecules))) {
    const molecules = Number(result.molecules);
    const allPassed = Number(result.all_passed ?? 0);
    const filterCount = Array.isArray(result.filters) ? result.filters.length : Number(result.filters_total || 0);
    const filterText = filterCount ? ` all ${filterCount} filters` : " all filters";
    return `${run.status === "completed" ? "Last run" : "Last attempt"}: ${escapeHtml(allPassed)}/${escapeHtml(molecules)} passed${filterText}`;
  }
  const totals = Number.isFinite(Number(result.molecules))
    ? `${result.passed ?? 0}/${result.molecules} passed`
    : run.status;
  return `${run.status === "completed" ? "Last run" : "Last attempt"}: ${escapeHtml(totals)}`;
}

function collectToolOptions(toolId, context = "results") {
  const selector = context === "evaluation" ? ".evaluation-tool-card" : ".tool-card";
  const card = $(`${selector}[data-tool-id="${CSS.escape(toolId)}"]`);
  const options = {};
  if (!card) return options;
  card.querySelectorAll("[data-tool-option]").forEach((input) => {
    options[input.dataset.toolOption] = input.type === "checkbox" ? input.checked : input.value;
  });
  return options;
}

async function runTool(toolId) {
  const job = loadedResultsJob();
  if (!job?.id) {
    showToast("Load a completed job before running a tool.");
    return;
  }
  const button = $(`.tool-run-button[data-tool-id="${CSS.escape(toolId)}"]`);
  if (button) {
    button.disabled = true;
    button.textContent = "Running...";
  }
  try {
    const options = collectToolOptions(toolId);
    const body = await service.runTool(job.id, toolId, options);
    showToast(`${body.run?.tool_name || "Tool"} ${body.run?.status || "completed"}. Reloading results...`);
    await loadCompletedJob(body.job || job);
  } catch (error) {
    showToast(error.message, 7000);
    renderToolChest();
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Run tool";
    }
  }
}

function toolOutputDefinitions() {
  const outputs = [];
  const seen = new Set();
  (state.tools || []).forEach((tool) => (tool.outputs || []).forEach((output) => {
    if (!output?.name || seen.has(output.name)) return;
    seen.add(output.name);
    outputs.push({
      ...output,
      label: output.label || propertyLabel(output.name),
    });
  }));
  return outputs;
}

function toolQualityRows(candidates) {
  return toolOutputDefinitions()
    .filter((output) => output.type === "boolean")
    .filter((output) => output.viewer !== "hidden")
    .map((output) => {
      const annotated = candidates.filter((item) => item.properties?.[output.name]);
      const passing = annotated.filter((item) => isTruthyProperty(item.properties?.[output.name])).length;
      return [output.label, annotated.length ? `${passing}/${annotated.length}` : "Not run"];
    });
}

function toolPropertyBadges(item) {
  const summaries = toolSummaryBadges(item);
  const values = toolOutputDefinitions()
    .filter((output) => shouldShowToolOutput(output))
    .map((output) => ({ output, value: item.properties?.[output.name] }))
    .filter((entry) => entry.value);
  if (!values.length && !summaries.length) return "-";
  return [
    ...summaries,
    ...values.map(({ output, value }) => {
      const text = toolPropertyDisplay(value);
      const title = output.description || output.label;
      if (output.type === "boolean") {
        const passed = isTruthyProperty(value);
        return `<span class="status-badge compact-badge" data-status="${passed ? "completed" : "failed"}" title="${escapeHtml(title)}">${escapeHtml(output.label)}: ${passed ? "Pass" : "Fail"}</span>`;
      }
      return `<span class="tool-property-chip" title="${escapeHtml(`${title}: ${text}`)}">${escapeHtml(output.label)}: ${escapeHtml(text)}</span>`;
    }),
  ].filter(Boolean).join(" ");
}

function viewerToolMetricRows(item) {
  const rows = toolSummaryRows(item);
  toolOutputDefinitions()
    .filter((output) => shouldShowToolOutput(output))
    .forEach((output) => {
      const value = item.properties?.[output.name];
      if (value) rows.push([output.label || propertyLabel(output.name), toolPropertyDisplay(value)]);
    });
  return rows;
}

function shouldShowToolOutput(output) {
  if (output.viewer === "hidden" || output.viewer === "summary") return false;
  return !/_STATUS$|_REASONS$|_FAILURES$|_OUTPUT$/i.test(output.name || "");
}

function toolSummaryRows(item) {
  return toolOutputDefinitions()
    .filter((output) => output.viewer === "summary")
    .map((output) => {
      const value = propertyMetric(item, output.name);
      const total = output.summary_total ? propertyMetric(item, output.summary_total) : null;
      if (!Number.isFinite(value)) return null;
      const suffix = output.summary_suffix ? ` ${output.summary_suffix}` : "";
      const display = Number.isFinite(total)
        ? `${formatMetric(value, 0)}/${formatMetric(total, 0)}${suffix}`
        : `${formatMetric(value)}${suffix}`;
      return [output.summary_label || output.label || propertyLabel(output.name), display, output.description || ""];
    })
    .filter(Boolean);
}

function toolSummaryBadges(item) {
  return toolSummaryRows(item).map(([label, value, title]) => (
    `<span class="tool-property-chip" title="${escapeHtml(title || label)}">${escapeHtml(label)}: ${escapeHtml(value)}</span>`
  ));
}

function toolPropertyDisplay(value) {
  const text = String(value ?? "");
  if (/^(true|yes|pass)$/i.test(text)) return "Pass";
  if (/^(false|no|fail)$/i.test(text)) return "Fail";
  return text || "-";
}

function isTruthyProperty(value) {
  return /^(true|yes|pass|1)$/i.test(String(value ?? "").trim());
}

function isFalseyProperty(value) {
  return /^(false|no|fail|0)$/i.test(String(value ?? "").trim());
}

function propertyLabel(name) {
  return String(name || "")
    .toLowerCase()
    .split("_")
    .filter(Boolean)
    .map((part) => part === "qvina" ? "QVina" : part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderExportFilters() {
  const list = $("#export-filter-list");
  const status = $("#export-filter-status");
  if (!list || !status) return;
  const definitions = availableExportMetrics();
  if (!state.study?.candidates?.length) {
    list.innerHTML = `<p class="tool-empty">Load results to build export requirements.</p>`;
    status.textContent = "No job loaded";
    return;
  }
  if (!definitions.length) {
    list.innerHTML = `<p class="tool-empty">No filterable metrics are available for this job.</p>`;
    status.textContent = "No metrics";
    return;
  }
  const groups = exportFilterGroups(definitions);
  list.innerHTML = groups.map((group) => `
    <section class="export-filter-group" aria-label="${escapeHtml(group.title)}">
      <div class="export-filter-group-heading">
        <strong>${escapeHtml(group.title)}</strong>
        <span>${escapeHtml(group.description)}</span>
      </div>
      <div class="export-filter-group-grid">
        ${group.metrics.map((metric) => exportFilterControl(metric)).join("")}
      </div>
    </section>
  `).join("");
  $$(".export-filter-row").forEach((row) => {
    const id = row.dataset.metricId;
    row.querySelectorAll("input, select").forEach((control) => control.addEventListener("input", () => {
      const filter = state.exportFilters[id] || defaultExportFilter(definitions.find((item) => item.id === id));
      const operatorChanged = control.dataset.filterField === "operator" && filter.operator !== control.value;
      if (control.dataset.filterField === "enabled") filter.enabled = control.checked;
      if (control.dataset.filterField === "operator") filter.operator = control.value;
      if (control.dataset.filterField === "value") filter.value = control.type === "number" ? Number(control.value) : control.value;
      if (control.dataset.filterField === "min") filter.min = Number(control.value);
      if (control.dataset.filterField === "max") filter.max = Number(control.value);
      state.exportFilters[id] = filter;
      if (operatorChanged) renderExportFilters();
      applyExportFilters();
      renderCharts();
    }));
  });
  updateExportFilterStatus();
}

function exportFilterGroups(definitions) {
  const values = definitions.filter((metric) => metric.type === "number");
  const filters = definitions.filter((metric) => metric.type !== "number");
  return [
    { title: "Values", description: "Numeric thresholds and ranges", metrics: values },
    { title: "Filters", description: "Pass/fail and category requirements", metrics: filters },
  ].filter((group) => group.metrics.length);
}

function exportFilterControl(metric) {
  const filter = ensureExportFilter(metric);
  const active = filter.enabled ? "checked" : "";
  const disabled = filter.enabled ? "" : "disabled";
  const valueLabel = metric.type === "number" ? rangeLabel(metric.values) : `${metric.values.length} value${metric.values.length === 1 ? "" : "s"}`;
  if (metric.type === "number") {
    return `
      <div class="export-filter-row" data-metric-id="${escapeHtml(metric.id)}" title="${escapeHtml(exportFilterHelp(metric, filter))}">
        <label class="check-control">
          <input type="checkbox" data-filter-field="enabled" ${active}>
          <span>${escapeHtml(metric.label)}</span>
        </label>
        <div class="export-filter-controls">
          <select class="compact-select operator-select" data-filter-field="operator" ${disabled}>
            <option value="<=" ${filter.operator === "<=" ? "selected" : ""}>&le;</option>
            <option value=">=" ${filter.operator === ">=" ? "selected" : ""}>&ge;</option>
            <option value="range" ${filter.operator === "range" ? "selected" : ""}>range</option>
          </select>
          ${filter.operator === "range"
            ? `<input type="number" data-filter-field="min" value="${escapeHtml(formatFilterValue(filter.min))}" step="${escapeHtml(metric.step || "0.1")}" aria-label="${escapeHtml(metric.label)} minimum" ${disabled}>
               <input type="number" data-filter-field="max" value="${escapeHtml(formatFilterValue(filter.max))}" step="${escapeHtml(metric.step || "0.1")}" aria-label="${escapeHtml(metric.label)} maximum" ${disabled}>`
            : `<input type="number" data-filter-field="value" value="${escapeHtml(formatFilterValue(filter.value))}" step="${escapeHtml(metric.step || "0.1")}" ${disabled}>`}
        </div>
        <small>${escapeHtml(valueLabel)}</small>
      </div>
    `;
  }
  if (metric.type === "boolean") {
    return `
      <div class="export-filter-row boolean-filter-row" data-metric-id="${escapeHtml(metric.id)}" title="${escapeHtml(exportFilterHelp(metric, filter))}">
        <label class="check-control">
          <input type="checkbox" data-filter-field="enabled" ${active}>
          <span>${escapeHtml(metric.label)}</span>
        </label>
        <small>Require pass</small>
      </div>
    `;
  }
  return `
    <div class="export-filter-row" data-metric-id="${escapeHtml(metric.id)}" title="${escapeHtml(exportFilterHelp(metric, filter))}">
      <label class="check-control">
        <input type="checkbox" data-filter-field="enabled" ${active}>
        <span>${escapeHtml(metric.label)}</span>
      </label>
      <div class="export-filter-controls">
        <select class="compact-select wide-filter-select" data-filter-field="value" ${disabled}>
          ${metric.values.map((value) => `<option value="${escapeHtml(value)}" ${String(filter.value) === String(value) ? "selected" : ""}>${escapeHtml(metricValueLabel(metric, value))}</option>`).join("")}
        </select>
      </div>
      <small>${escapeHtml(valueLabel)}</small>
    </div>
  `;
}

function ensureExportFilter(metric) {
  if (!state.exportFilters[metric.id]) state.exportFilters[metric.id] = defaultExportFilter(metric);
  return state.exportFilters[metric.id];
}

function defaultExportFilter(metric) {
  if (!metric) return { enabled: false, operator: ">=", value: "" };
  if (metric.type === "number") {
    const values = metric.values.filter(Number.isFinite);
    const fallback = values.length ? (metric.defaultOperator === "<=" ? Math.max(...values) : Math.min(...values)) : 0;
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 0;
    return {
      enabled: false,
      operator: metric.defaultOperator || ">=",
      value: Number(fallback.toFixed(metric.digits ?? 2)),
      min: Number(min.toFixed(metric.digits ?? 2)),
      max: Number(max.toFixed(metric.digits ?? 2)),
    };
  }
  const value = metric.defaultValue && metric.values.includes(metric.defaultValue) ? metric.defaultValue : metric.values[0] || "";
  return { enabled: false, operator: "=", value };
}

function applyExportFilters(render = true) {
  const candidates = state.study?.candidates || [];
  const definitions = availableExportMetrics();
  const definitionById = new Map(definitions.map((metric) => [metric.id, metric]));
  const activeFilters = Object.entries(state.exportFilters)
    .map(([id, filter]) => ({ metric: definitionById.get(id), filter }))
    .filter((entry) => entry.metric && entry.filter.enabled);
  if (!activeFilters.length) {
    state.exportSelection = new Set(candidates.map((item) => item.id));
  } else {
    state.exportSelection = new Set(candidates
      .filter((item) => activeFilters.every(({ metric, filter }) => candidatePassesExportFilter(item, metric, filter)))
      .map((item) => item.id));
  }
  const checkbox = $("#export-filtered");
  if (checkbox) checkbox.checked = true;
  updateExportFilterStatus();
  updateExportScope();
  if (render) renderResultsTable();
}

function activeExportFilters() {
  const definitionById = new Map(availableExportMetrics().map((metric) => [metric.id, metric]));
  return Object.entries(state.exportFilters)
    .map(([id, filter]) => ({ metric: definitionById.get(id), filter }))
    .filter((entry) => entry.metric && entry.filter.enabled)
    .map(({ metric, filter }) => ({
      id: metric.id,
      label: metric.label,
      type: metric.type,
      operator: filter.operator,
      value: filter.value,
      min: filter.min,
      max: filter.max,
      text: exportFilterText(metric, filter),
    }));
}

function exportFilterText(metric, filter) {
  if (metric.type === "number" && filter.operator === "range") {
    return `${metric.label} between ${formatFilterValue(filter.min)} and ${formatFilterValue(filter.max)}`;
  }
  if (metric.type === "number") {
    return `${metric.label} ${filter.operator || ">="} ${formatFilterValue(filter.value)}`;
  }
  if (metric.type === "boolean") {
    return `${metric.label} must pass`;
  }
  return `${metric.label} = ${metricValueLabel(metric, filter.value)}`;
}

function exportFilterHelp(metric, filter) {
  const description = metric.description ? `${metric.description} ` : "";
  if (filter.enabled) return exportFilterText(metric, filter);
  if (metric.type === "number") return `${description}Enable to filter exported molecules by ${metric.label}.`;
  if (metric.type === "boolean") return `${description}Enable to require ${metric.label} to pass for exported molecules.`;
  return `${description}Enable to require a ${metric.label} value for exported molecules.`;
}

function updateExportFilterStatus() {
  const status = $("#export-filter-status");
  if (!status) return;
  const activeCount = Object.values(state.exportFilters).filter((filter) => filter.enabled).length;
  const selectedCount = state.exportSelection?.size || 0;
  const total = state.study?.candidates?.length || 0;
  status.textContent = activeCount
    ? `${selectedCount}/${total} passed all ${activeCount} filter${activeCount === 1 ? "" : "s"}`
    : total ? `No filters · ${selectedCount}/${total} selected` : "No job loaded";
  $$(".export-filter-row").forEach((row) => {
    const filter = state.exportFilters[row.dataset.metricId];
    row.classList.toggle("active", Boolean(filter?.enabled));
    row.querySelectorAll("select, input[type='number']").forEach((control) => {
      control.disabled = !filter?.enabled;
    });
  });
}

function resetExportFilters(event = null) {
  event?.preventDefault();
  event?.stopPropagation();
  state.exportFilters = {};
  state.histogramThreshold = null;
  renderExportFilters();
  renderHistogramMetricOptions();
  applyExportFilters();
  renderCharts();
  showToast("Export filters reset.");
}

function candidatePassesExportFilter(item, metric, filter) {
  const value = exportMetricValue(item, metric);
  if (metric.type === "number") {
    if (value == null || value === "") return false;
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) return false;
    if (filter.operator === "range") {
      const min = Number(filter.min);
      const max = Number(filter.max);
      if (!Number.isFinite(min) || !Number.isFinite(max)) return false;
      return numericValue >= Math.min(min, max) && numericValue <= Math.max(min, max);
    }
    const threshold = Number(filter.value);
    if (!Number.isFinite(threshold)) return false;
    return filter.operator === "<=" ? numericValue <= threshold : numericValue >= threshold;
  }
  if (metric.type === "boolean") {
    return isTruthyProperty(value);
  }
  return String(value ?? "") === String(filter.value ?? "");
}

function availableExportMetrics() {
  const candidates = state.study?.candidates || [];
  if (!candidates.length) return [];
  const builtIn = [
    { id: "field:molecularWeight", label: "Molecular weight", type: "number", field: "molecularWeight", defaultOperator: "<=", digits: 1, step: "0.1" },
    { id: "field:heavyAtoms", label: "Heavy atoms", type: "number", field: "heavyAtoms", defaultOperator: "<=", digits: 0, step: "1" },
    { id: "field:heteroAtoms", label: "Hetero atoms", type: "number", field: "heteroAtoms", defaultOperator: "<=", digits: 0, step: "1" },
    { id: "field:rings", label: "Ring estimate", type: "number", field: "rings", defaultOperator: "<=", digits: 0, step: "1" },
    { id: "prop:VINA_SCORE_ONLY", label: "Vina score", type: "number", property: "VINA_SCORE_ONLY", defaultOperator: "<=", digits: 2, step: "0.1" },
    { id: "prop:VINA_MINIMIZE", label: "Vina minimize", type: "number", property: "VINA_MINIMIZE", defaultOperator: "<=", digits: 2, step: "0.1" },
    { id: "prop:VINA_DOCK", label: "Vina redock", type: "number", property: "VINA_DOCK", defaultOperator: "<=", digits: 2, step: "0.1" },
    { id: "prop:QVINA", label: "QVina", type: "number", property: "QVINA", defaultOperator: "<=", digits: 2, step: "0.1" },
    { id: "prop:QED", label: "QED", type: "number", property: "QED", defaultOperator: ">=", digits: 2, step: "0.01" },
    { id: "prop:SA", label: "SA", type: "number", property: "SA", defaultOperator: "<=", digits: 2, step: "0.1" },
    { id: "prop:LOGP", label: "LogP", type: "number", property: "LOGP", defaultOperator: "<=", digits: 2, step: "0.1" },
    { id: "prop:LIPINSKI", label: "Lipinski", type: "number", property: "LIPINSKI", defaultOperator: ">=", digits: 0, step: "1" },
  ];
  const knownProperties = new Set(builtIn.filter((metric) => metric.property).map((metric) => metric.property));
  const metrics = builtIn.map((metric) => metricWithValues(metric, candidates)).filter(Boolean);
  toolOutputDefinitions().forEach((output) => {
    if (knownProperties.has(output.name)) return;
    if (!isExportFilterOutput(output)) return;
    const type = output.type === "boolean" ? "boolean" : inferPropertyMetricType(candidates, output.name);
    const metric = metricWithValues({
      id: `tool:${output.name}`,
      label: output.label || propertyLabel(output.name),
      description: output.description || "",
      type,
      property: output.name,
      defaultOperator: type === "number" ? ">=" : "=",
      defaultValue: type === "boolean" ? "true" : undefined,
      digits: 2,
      step: "0.1",
    }, candidates);
    if (metric) metrics.push(metric);
  });
  return metrics;
}

function isExportFilterOutput(output) {
  if (output.filterable === false) return false;
  if (output.type === "boolean") return true;
  return !/_STATUS$|_REASONS$|_OUTPUT$/i.test(output.name || "");
}

function metricWithValues(metric, candidates) {
  const rawValues = candidates.map((item) => exportMetricValue(item, metric)).filter((value) => value !== null && value !== undefined && value !== "");
  if (!rawValues.length) return null;
  const values = metric.type === "number"
    ? rawValues.map(Number).filter(Number.isFinite)
    : uniqueValues(rawValues.map((value) => metric.type === "boolean" ? booleanFilterValue(value) : String(value)));
  if (!values.length) return null;
  if (metric.type === "text" && values.length > MAX_CATEGORICAL_FILTER_VALUES) return null;
  return { ...metric, values };
}

function exportMetricValue(item, metric) {
  if (metric.field) return item[metric.field] == null ? null : Number(item[metric.field]);
  if (!metric.property) return null;
  if (metric.property === "VINA_DOCK" || metric.property === "QVINA" || metric.property === "VINA_SCORE_ONLY" || metric.property === "VINA_MINIMIZE") {
    return propertyMetric(item, metric.property);
  }
  const value = item.properties?.[metric.property];
  return metric.type === "number" ? Number.parseFloat(value) : value;
}

function inferPropertyMetricType(candidates, property) {
  const values = candidates.map((item) => item.properties?.[property]).filter((value) => value !== null && value !== undefined && value !== "");
  if (values.length && values.every((value) => isTruthyProperty(value) || isFalseyProperty(value))) return "boolean";
  if (values.length && values.every((value) => Number.isFinite(Number.parseFloat(value)))) return "number";
  return "text";
}

function booleanFilterValue(value) {
  return isTruthyProperty(value) ? "true" : "false";
}

function metricValueLabel(metric, value) {
  if (metric.type !== "boolean") return value;
  return value === "true" ? "Pass" : "Fail";
}

function uniqueValues(values) {
  return [...new Set(values.map((value) => String(value)))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function formatFilterValue(value) {
  return Number.isFinite(Number(value)) ? String(value) : "";
}

function filteredCandidates() {
  const candidates = [...(state.study?.candidates || [])];
  const query = $("#result-search").value.trim().toLowerCase();
  const sort = $("#result-sort").value;
  return candidates
    .filter((item) => {
      if (!query) return true;
      return [
        item.id,
        item.name,
        item.formula,
        item.smiles,
        item.properties?.SMILES,
        item.properties?.VINA_STATUS,
        ...toolOutputDefinitions().map((output) => item.properties?.[output.name]),
      ].some((value) => String(value || "").toLowerCase().includes(query));
    })
    .sort((a, b) => sort === "index" ? a.index - b.index : compareMetric(candidateMetric(b, sort), candidateMetric(a, sort)));
}

function renderResultsTable() {
  const candidates = filteredCandidates();
  $("#visible-count").textContent = `${candidates.length} shown`;
  updateExportScope();
  $("#result-table").innerHTML = candidates.map((item) => `
    <tr data-index="${item.index}" class="${state.selected?.index === item.index ? "active" : ""}">
      <td><input class="candidate-export-toggle" type="checkbox" data-candidate-id="${escapeHtml(item.id)}" ${state.exportSelection.has(item.id) ? "checked" : ""} aria-label="Export ${escapeHtml(item.id)}"></td>
      <td>${escapeHtml(item.id)}<br><small>${escapeHtml(item.formula)}</small></td>
      <td class="smiles-cell" title="${escapeHtml(item.smiles || item.properties?.SMILES || "")}">${escapeHtml(item.smiles || item.properties?.SMILES || "-")}</td>
      <td>${formatMetric(propertyMetric(item, "VINA_SCORE_ONLY"))}</td>
      <td>${formatMetric(propertyMetric(item, "VINA_MINIMIZE"))}</td>
      <td>${formatMetric(propertyMetric(item, "VINA_DOCK") ?? propertyMetric(item, "QVINA"))}</td>
      <td>${toolPropertyBadges(item)}</td>
    </tr>`).join("");
  $$("#result-table .candidate-export-toggle").forEach((input) => input.addEventListener("click", (event) => {
    event.stopPropagation();
    if (input.checked) state.exportSelection.add(input.dataset.candidateId); else state.exportSelection.delete(input.dataset.candidateId);
    updateExportScope();
  }));
  $$("#result-table tr").forEach((row) => row.addEventListener("click", () => {
    state.selected = state.study.candidates.find((item) => item.index === Number(row.dataset.index));
    renderResultsTable();
    renderSelectedStructure();
    renderViewerSelectors();
    renderMoleculeViewerWorkspace();
    renderCharts();
  }));
}

function propertyMetric(item, key) {
  if (["VINA_SCORE_ONLY", "VINA_MINIMIZE", "VINA_DOCK", "QVINA"].includes(key) && !vinaWasRun(item.properties)) return null;
  const value = Number.parseFloat(item.properties?.[key]);
  return Number.isFinite(value) ? value : null;
}

function candidateMetric(item, key) {
  if (key === "vina_score_only") return propertyMetric(item, "VINA_SCORE_ONLY");
  if (key === "vina_minimize") return propertyMetric(item, "VINA_MINIMIZE");
  if (key === "vina_dock") return propertyMetric(item, "VINA_DOCK");
  if (key === "qvina") return propertyMetric(item, "QVINA");
  return item[key];
}

function renderHistogramMetricOptions() {
  const select = $("#histogram-metric");
  if (!select) return;
  const current = select.value;
  const metrics = availableExportMetrics();
  if (!metrics.length) {
    select.innerHTML = `<option value="">No metrics</option>`;
    select.disabled = true;
    return;
  }
  select.disabled = false;
  select.innerHTML = metrics.map((metric) => `<option value="${escapeHtml(metric.id)}" title="${escapeHtml(metric.description || metric.label)}">${escapeHtml(metric.label)}</option>`).join("");
  select.value = metrics.some((metric) => metric.id === current) ? current : metrics[0].id;
}

function renderCharts({ refreshMetricOptions = true } = {}) {
  if (!state.study || state.activeTab !== "results" || !$(".analytics-details").open) return;
  if (refreshMetricOptions) renderHistogramMetricOptions();
  const metric = exportMetricForHistogram($("#histogram-metric").value);
  const label = metric?.label || $("#histogram-metric").selectedOptions[0]?.textContent || "Metric";
  const slider = $("#histogram-threshold");
  if (metric && metric.type !== "number") {
    slider.disabled = true;
    $("#histogram-threshold-value").textContent = "—";
    const entries = categoricalDistribution(state.study.candidates, metric);
    $("#histogram-tooltip").textContent = `${entries.reduce((sum, entry) => sum + entry.count, 0)}/${state.study.candidates.length} molecules have ${label} annotations.`;
    drawCategoryChart($("#histogram"), entries, label);
    return;
  }
  const values = metric ? state.study.candidates.map((item) => exportMetricValue(item, metric)).filter(Number.isFinite) : [];
  if (!values.length) {
    slider.disabled = true;
    $("#histogram-threshold-value").textContent = "—";
    drawHistogram($("#histogram"), values, label);
    return;
  }
  const min = Math.min(...values); const max = Math.max(...values);
  slider.disabled = false; slider.min = min; slider.max = max; slider.step = (max - min || 1) / 100;
  const exportFilter = metric ? state.exportFilters[metric.id] : null;
  if (exportFilter?.enabled && Number.isFinite(Number(exportFilter.value))) {
    state.histogramThreshold = Number(exportFilter.value);
  } else if (exportFilter?.enabled && exportFilter.operator === "range" && Number.isFinite(Number(exportFilter.max))) {
    state.histogramThreshold = Number(exportFilter.max);
  }
  if (state.histogramThreshold === null || state.histogramThreshold < min || state.histogramThreshold > max) state.histogramThreshold = min;
  slider.value = state.histogramThreshold;
  $("#histogram-threshold-value").textContent = Number(state.histogramThreshold).toFixed(1);
  const lowerIsBetter = exportFilter?.operator ? exportFilter.operator === "<=" : metric.defaultOperator === "<=";
  const passing = exportFilter?.enabled && exportFilter.operator === "range"
    ? values.filter((value) => value >= Math.min(Number(exportFilter.min), Number(exportFilter.max)) && value <= Math.max(Number(exportFilter.min), Number(exportFilter.max))).length
    : values.filter((value) => lowerIsBetter ? value <= state.histogramThreshold : value >= state.histogramThreshold).length;
  $("#histogram-tooltip").textContent = `${passing}/${values.length} molecules meet the displayed ${exportFilter?.operator === "range" ? "range" : "threshold"}.`;
  drawHistogram($("#histogram"), values, label, state.histogramThreshold);
}

function categoricalDistribution(candidates, metric) {
  const counts = new Map();
  candidates.forEach((item) => {
    const value = exportMetricValue(item, metric);
    if (value === null || value === undefined || value === "") return;
    const label = metric.type === "boolean" ? metricValueLabel(metric, booleanFilterValue(value)) : String(value);
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, undefined, { numeric: true }));
}

function syncHistogramThresholdToExportFilter({ defer = false } = {}) {
  const metric = exportMetricForHistogram($("#histogram-metric").value);
  if (!metric) return;
  const filter = state.exportFilters[metric.id] || defaultExportFilter(metric);
  filter.enabled = true;
  filter.operator = filter.operator === "range" ? metric.defaultOperator || ">=" : filter.operator || metric.defaultOperator || ">=";
  filter.value = Number(state.histogramThreshold);
  state.exportFilters[metric.id] = filter;
  if (defer) {
    scheduleExportFilterApply();
    return;
  }
  renderExportFilters();
  applyExportFilters();
}

function scheduleExportFilterApply() {
  if (state.exportFilterTimer) clearTimeout(state.exportFilterTimer);
  state.exportFilterTimer = setTimeout(() => {
    state.exportFilterTimer = null;
    renderExportFilters();
    applyExportFilters();
  }, 140);
}

function exportMetricForHistogram(histogramMetric) {
  const direct = {
    molecularWeight: "field:molecularWeight",
    heavyAtoms: "field:heavyAtoms",
    heteroAtoms: "field:heteroAtoms",
    rings: "field:rings",
  }[histogramMetric];
  const definitions = availableExportMetrics();
  if (direct) return definitions.find((metric) => metric.id === direct) || null;
  if (histogramMetric === "vinaScore") {
    return definitions.find((metric) => ["prop:VINA_SCORE_ONLY", "prop:VINA_MINIMIZE", "prop:VINA_DOCK", "prop:QVINA"].includes(metric.id)) || null;
  }
  return definitions.find((metric) => metric.id === histogramMetric) || null;
}

function scheduleThresholdRender() {
  if (state.thresholdFrame) cancelAnimationFrame(state.thresholdFrame);
  state.thresholdFrame = requestAnimationFrame(() => {
    state.thresholdFrame = null;
    renderCharts({ refreshMetricOptions: false });
  });
}

function handleHistogramHover(event) {
  const chart = $("#histogram")._histogram;
  if (!chart) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const index = Math.max(0, Math.min(chart.bins - 1, Math.floor((x - chart.pad.left) / (chart.plotW / chart.bins))));
  const step = (chart.max - chart.min || 1) / chart.bins;
  const start = chart.min + index * step;
  $("#histogram-tooltip").textContent = `${start.toFixed(1)}–${(start + step).toFixed(1)}: ${chart.counts[index]} molecule${chart.counts[index] === 1 ? "" : "s"}`;
}

function renderSelectedStructure() {
  const molecule = state.selected;
  if (!molecule || !state.study || state.activeTab !== "results") return;
  const hasReference = Boolean(state.study.referenceSdf);
  const referenceControl = $("#viewer-reference-control");
  const referenceToggle = $("#viewer-reference-toggle");
  referenceControl.hidden = !hasReference;
  referenceToggle.disabled = !hasReference;
  referenceToggle.checked = hasReference && state.showReferenceLigand;
  $("#selected-name").textContent = molecule.id;
  const metrics = [
    ["Formula", molecule.formula || "-"],
    ["MW", Number.isFinite(molecule.molecularWeight) ? `${molecule.molecularWeight} Da` : "-"],
    ["Heavy atoms", molecule.heavyAtoms],
    ["Rings", molecule.rings],
  ];
  if (molecule.smiles) {
    metrics.push(["SMILES", molecule.smiles]);
  }
  const vinaScore = propertyMetric(molecule, "VINA_SCORE_ONLY");
  if (vinaScore !== null) metrics.push(["Vina score", formatMetric(vinaScore)]);
  const vinaMinimized = propertyMetric(molecule, "VINA_MINIMIZE");
  if (vinaMinimized !== null) metrics.push(["Vina minimized", formatMetric(vinaMinimized)]);
  const vinaDock = propertyMetric(molecule, "VINA_DOCK");
  if (vinaDock !== null) metrics.push(["Vina redocked", formatMetric(vinaDock)]);
  const qvina = propertyMetric(molecule, "QVINA");
  if (qvina !== null) metrics.push(["QVina", formatMetric(qvina)]);
  metrics.push(...viewerToolMetricRows(molecule));
  $("#selected-metrics").innerHTML = metrics.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  render2D($("#viewer-2d"), molecule);
  $("#viewer-loading").hidden = false;
  render3D($("#viewer-3d"), molecule, state.study.pdbText, {
    referenceText: hasReference && state.showReferenceLigand ? state.study.referenceSdf : null,
    controlsHost: $("#results-viewer-controls"),
    hideReferenceVisibility: true,
  }).finally(() => {
    $("#viewer-loading").hidden = true;
  });
  setView(state.view);
}

function setView(view) {
  state.view = view;
  $$(".view-toggle button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $("#viewer-3d").hidden = view !== "3d";
  $("#viewer-2d").hidden = view !== "2d";
  $("#results-viewer-controls").hidden = view !== "3d";
}

function renderViewerSelectors() {
  const select = $("#viewer-molecule-select");
  if (!select) return;
  const options = viewerMoleculeOptions();
  const optionHtml = options.map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`).join("");
  select.innerHTML = optionHtml;
  state.workspaceViewerMolecule = options.some((option) => option.value === state.workspaceViewerMolecule)
    ? state.workspaceViewerMolecule
    : "selected";
  select.value = state.workspaceViewerMolecule;
}

function viewerMoleculeOptions() {
  const options = [{ value: "selected", label: `Selected: ${state.selected?.id || "none"}` }];
  if (state.study?.referenceSdf) options.push({ value: "reference", label: "Reference ligand" });
  (state.study?.candidates || []).forEach((candidate) => {
    options.push({ value: `candidate:${candidate.index}`, label: candidate.id });
  });
  return options;
}

function renderMoleculeViewerWorkspace() {
  if (!state.study || !state.selected) return;
  renderViewerSelectors();
  renderViewerSlot(resolveViewerMolecule(state.workspaceViewerMolecule));
}

function renderViewerSlot(item) {
  const title = $("#viewer-workspace-title");
  const container = $("#viewer-workspace-3d");
  const loading = $("#viewer-workspace-loading");
  if (!title || !container || !loading) return;
  if (!item) {
    title.textContent = "No molecule";
    container.innerHTML = "<div class='viewer-error'>Load a completed job before using the viewer.</div>";
    loading.hidden = true;
    return;
  }
  title.textContent = item.label;
  loading.hidden = false;
  render3D(container, item.molecule, state.study.pdbText, {
    controlsHost: $("#workspace-viewer-controls"),
    referenceText: state.study.referenceSdf && item.label !== "Reference ligand" ? state.study.referenceSdf : null,
    defaultHiddenKinds: ["reference"],
  }).finally(() => {
    loading.hidden = true;
  });
}

function resolveViewerMolecule(value) {
  if (value === "reference") {
    if (!state.study?.referenceSdf) return null;
    return {
      label: "Reference ligand",
      molecule: {
        id: "Reference ligand",
        name: state.study.example?.sdf || "reference.sdf",
        text: state.study.referenceSdf,
      },
    };
  }
  if (value?.startsWith("candidate:")) {
    const index = Number(value.split(":")[1]);
    const candidate = state.study?.candidates?.find((item) => item.index === index);
    return candidate ? { label: candidate.id, molecule: candidate } : null;
  }
  return state.selected ? { label: `Selected: ${state.selected.id}`, molecule: state.selected } : null;
}

function updateResultsSource() {
  const job = loadedResultsJob();
  if (job) {
    const count = state.study?.summary?.sdf_count ?? state.study?.candidates?.length ?? 0;
    $("#results-source").textContent = `Loaded ${count} generated SDF${count === 1 ? "" : "s"} from ${engineLabel(job)} job ${job.id}.`;
    return;
  }
  $("#results-source").textContent = "Upload input structures and submit a job to review generated outputs here.";
}

function renderJobProvenance(job) {
  const container = $("#job-detail-provenance");
  if (!container) return;
  const rows = job ? provenanceRows(job, { includeOutput: false }) : [];
  container.hidden = !rows.length;
  container.innerHTML = rows.map(([label, value]) => provenanceItem(label, value)).join("");
}

function renderResultsProvenance() {
  const container = $("#results-provenance");
  if (!container) return;
  const job = loadedResultsJob();
  const rows = job ? provenanceRows(job, {
    includeOutput: true,
    candidateCount: state.study?.summary?.sdf_count ?? state.study?.candidates?.length ?? null,
  }) : [];
  container.hidden = !rows.length;
  container.innerHTML = rows.map(([label, value]) => provenanceItem(label, value)).join("");
  const warning = $("#results-evaluation-warning");
  if (warning) {
    const failures = resultEvaluationFailures(job);
    warning.hidden = !failures.length;
    warning.innerHTML = failures.length
      ? `<strong>Some evaluations failed</strong>Generated molecules are available, but ${escapeHtml(failures.join(" and "))} did not complete successfully. Check the job logs before interpreting those annotations.`
      : "";
  }
}

function resultEvaluationFailures(job) {
  if (!job) return [];
  const failures = [];
  if (job.postprocess?.status === "failed") failures.push("built-in postprocessing");
  if ((job.tools || []).some((tool) => tool.status === "failed")) failures.push("a selected Tool Chest evaluator");
  return failures;
}

function provenanceItem(label, value) {
  return `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function provenanceRows(job, { includeOutput = false, candidateCount = null } = {}) {
  const preprocess = job?.preprocess || {};
  const sourceFiles = preprocess.source_files || {};
  const rows = [
    ["Engine", engineLabel(job)],
    ["Conditioning", conditioningLabel(job)],
    ["Input", inputLabel(job)],
    ["PDB", provenanceFilename(job?.inputs?.pdb || sourceFiles.pdb || sourceFiles.protein || sourceFiles.pocket)],
    ["SDF", provenanceFilename(job?.inputs?.sdf || sourceFiles.sdf || sourceFiles.ligand)],
    ["Target", targetLabel(job)],
  ];
  const prepSource = preprocess.source || preprocess.workflow || preprocess.preprocessing?.method;
  if (prepSource) rows.push(["Preprocess", workflowLabel(prepSource)]);
  if (includeOutput) rows.push(["Generated", candidateCount == null ? "n/a" : `${candidateCount} SDF${candidateCount === 1 ? "" : "s"}`]);
  return rows.filter(([, value]) => value !== null && value !== undefined && value !== "");
}

function provenanceFilename(path) {
  if (!path) return "none";
  return String(path).split("/").pop() || "none";
}

function engineLabel(job) {
  return job?.engine === "diffsmol" ? "DiffSMol" : "conDitar";
}

function conditioningLabel(job) {
  if (job?.engine === "diffsmol") return job?.inputs?.pdb ? "Ligand + pocket" : "Ligand only";
  if (job?.mode === "reference") return "Protein + ligand";
  if (job?.mode === "pocket") return "Pocket";
  return job?.mode || "Run";
}

function targetLabel(job) {
  if (!job) return "Local CPU";
  if (job.target === "local_cpu") return "Local CPU";
  if (isOpenShiftJobTarget(job.target)) return openShiftSubmissionEnabled() ? "OpenShift Job" : "OpenShift Job manifest";
  if (isOpenShiftMockTarget(job.target)) return "OpenShift diagnostics";
  if (isSlurmGpuTarget(job.target)) return "Slurm GPU";
  return job.target || "Local CPU";
}

function isOpenShiftJobTarget(target) {
  return target === OPENSHIFT_JOB_TARGET;
}

function isOpenShiftMockTarget(target) {
  return target === OPENSHIFT_MOCK_TARGET;
}

function isOpenShiftTarget(target) {
  return isOpenShiftJobTarget(target) || isOpenShiftMockTarget(target);
}

function isSlurmGpuTarget(target) {
  return target === SLURM_GPU_TARGET || target === LEGACY_SLURM_GPU_TARGET;
}

function usesBrowserNotifications(target) {
  return target === "local_cpu" || isOpenShiftTarget(target);
}

function batchTargetNoun(target) {
  if (isSlurmGpuTarget(target)) return "parallel GPU tasks";
  if (isOpenShiftJobTarget(target)) return openShiftSubmissionEnabled() ? "OpenShift Jobs queued" : "OpenShift Job manifests queued";
  if (isOpenShiftMockTarget(target)) return "OpenShift diagnostic jobs queued";
  return "CPU jobs queued";
}

function isActiveJob(job) {
  return ACTIVE_JOB_STATUSES.has(job?.status) || isBuiltinPostprocessRunning(job);
}

function isTerminalJob(job) {
  return TERMINAL_JOB_STATUSES.has(job?.status) && !isBuiltinPostprocessRunning(job);
}

function isBuiltinPostprocessRunning(job) {
  return (job?.postprocess || {}).status === "running";
}

function isResultsReadyJob(job) {
  return job?.status === "completed"
    && job.outputs?.sdf_count !== 0
    && !isBuiltinPostprocessRunning(job)
    && !(job.tools || []).some((tool) => ["pending", "running"].includes(tool.status));
}

function shortJobId(jobId) {
  const text = String(jobId || "");
  return text.length > 22 ? `${text.slice(0, 15)}…${text.slice(-6)}` : text;
}

function slurmLabel(job) {
  const slurm = job?.slurm || {};
  const terminalState = { completed: "COMPLETED", failed: "FAILED", canceled: "CANCELLED" }[job?.status];
  if (slurm.job_id && terminalState) return `Slurm ${slurm.job_id} · ${terminalState}`;
  if (slurm.job_id && slurm.state) return `Slurm ${slurm.job_id} · ${slurm.state}`;
  if (slurm.job_id) return `Slurm ${slurm.job_id}`;
  return isSlurmGpuTarget(job?.target) ? "Slurm pending" : "";
}

function inputLabel(job) {
  return job?.input_name || filenameOnly(job?.inputs?.pdb || "") || job?.example_id || "custom";
}

function updateJobTargetControls() {
  const target = resolvedTarget();
  const isSlurmGpu = isSlurmGpuTarget(target);
  const isOpenShift = isOpenShiftTarget(target);
  $("#slurm-controls").hidden = !isSlurmGpu;
  $("#job-runtime-label").textContent = targetLabel({ target });
  const emailInput = $("#job-email");
  const emailNote = $("#email-note");
  emailInput.disabled = !isSlurmGpu;
  emailInput.closest(".job-controls").classList.toggle("is-disabled", !isSlurmGpu);
  if (!isSlurmGpu) {
    emailInput.value = "";
    emailNote.textContent = isOpenShift
      ? `${targetLabel({ target })} runs use browser/system notifications when this page is allowed to notify you.`
      : "Local CPU runs use browser/system notifications when this page is allowed to notify you.";
  } else {
    emailNote.textContent = "Slurm can send completion/failure notifications when an email is provided.";
  }
  state.parameters.device = isSlurmGpu ? "auto" : "cpu";
  updateBatchLabel();
  updateRunSummary();
}

function updateBuiltinEvaluationAvailability() {
  const isDiffSmol = state.engine === "diffsmol";
  const hasPocket = hasDiffSmolPocketContext();
  $$(".builtin-evaluation-toggle").forEach((input) => {
    const disabled = isDiffSmol && VINA_EVALUATIONS.has(input.value) && !hasPocket;
    input.disabled = disabled;
    input.closest(".check-control")?.classList.toggle("is-disabled", disabled);
  });
  const note = $("#builtin-evaluation-note");
  if (note) {
    note.textContent = !isDiffSmol
      ? "Select generated-molecule properties to compute with each run."
      : hasPocket
        ? "DiffSMol is in ligand + pocket mode. Generation uses the pocket, and Vina/QVina can score the outputs afterward."
        : "DiffSMol is ligand-only. Vina/QVina is disabled because docking scores require a pocket PDB.";
  }
}

function updateVinaControls() {
  updateBuiltinEvaluationAvailability();
  const selected = selectedBuiltinEvaluations();
  const vinaSelected = selected.some((item) => VINA_EVALUATIONS.has(item));
  const enabled = vinaSelected;
  $("#vina-options").classList.toggle("is-disabled", !enabled);
  $("#vina-exhaustiveness").disabled = !enabled;
  $("#vina-cpu").disabled = !enabled;
  $("#vina-mode-summary").textContent = selectedEvaluationLabel(selected);
  updateRunEvaluationSummary();
  updateRunSummary();
}

function setupWarnings() {
  const generationInput = currentGenerationInput();
  if (generationInput.batchCount) return [];
  const sdfHeavyAtoms = generationInput.sdf?.text ? sdfHeavyAtomCount(generationInput.sdf.text) : 0;
  return setupWarningsForWorkflow({
    workflow: generationInput.workflow,
    sdf: generationInput.sdf,
    sdfHeavyAtoms,
    heavyAtomWarningThreshold: DIFFSMOL_SHAPE_HEAVY_ATOM_WARNING,
    selectedMetrics: selectedBuiltinEvaluations({ includeDisabled: true }),
    stagedPocketWarnings: stagedPocketWarnings(),
  });
}

function stagedPocketWarnings() {
  const manifest = state.stagedPreprocessManifest;
  if (!manifest) return [];
  const prep = manifest.preprocessing || {};
  const staged = manifest.staged_inputs || {};
  const mode = staged.mode || manifest.selected_variant?.mode || "";
  const hasPocketInput = Boolean(staged.pdb) && (mode === "pocket" || state.engine === "diffsmol");
  if (!hasPocketInput) return [];
  const warnings = [...(prep.structured_warnings || []), ...(prep.warnings || [])];
  const residueCount = Number(prep.residue_count);
  if (Number.isFinite(residueCount) && residueCount > 0 && !warnings.some((item) => String(item).includes("fewer than"))) {
    if (residueCount < 10) {
      warnings.push(`Pocket has ${residueCount} complete residues. More than 10-20 residues is advised for stable generation/scoring.`);
    } else if (residueCount < 20) {
      warnings.push(`Pocket has ${residueCount} complete residues. More than 20 residues is advised when possible.`);
    }
  }
  const seen = new Set();
  return warnings
    .map((item) => normalizeWorkflowWarning(item, { code: "pocket_size", title: "Pocket size", appliesTo: "pocket" }))
    .filter((item) => {
      const key = `${item.code}:${item.text}`;
      if (!item.text || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function updateSetupWarningBanner() {
  const banner = $("#setup-warning-banner");
  if (!banner) return;
  const warnings = setupWarnings();
  if (!warnings.length) {
    banner.hidden = true;
    banner.innerHTML = "";
    return;
  }
  banner.hidden = false;
  banner.innerHTML = `
    <strong>Check before generating</strong>
    <ul>${warnings.map((warning) => `<li><b>${escapeHtml(warning.title)}:</b> ${escapeHtml(warningText(warning))}</li>`).join("")}</ul>
  `;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatMetric(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function bestValueLabel(values) {
  return values.length ? `${formatMetric(Math.min(...values))} kcal/mol` : "n/a";
}

function numericValues(items, key) {
  return items.map((item) => item[key] == null ? null : Number(item[key])).filter(Number.isFinite);
}

function propertyValues(items, key) {
  return items.map((item) => Number.parseFloat(item.properties?.[key])).filter(Number.isFinite);
}

function rangeLabel(values) {
  if (!values.length) return "n/a";
  return `${formatMetric(Math.min(...values))} to ${formatMetric(Math.max(...values))}`;
}

function compareMetric(a, b) {
  const left = Number(a);
  const right = Number(b);
  const leftValid = Number.isFinite(left);
  const rightValid = Number.isFinite(right);
  if (!leftValid && !rightValid) return 0;
  if (!leftValid) return -1;
  if (!rightValid) return 1;
  return left - right;
}

function updateRunSummary() {
  updateRunEvaluationSummary();
  updateRunEstimate();
  updateSetupWarningBanner();
}

function resetParameters() {
  [...PARAMETERS, ...ADVANCED_PARAMETERS].forEach((parameter) => {
    state.parameters[parameter.key] = parameter.value;
    $(`#param-${parameter.key}`).value = parameter.value;
  });
  updateRunSummary();
  showToast("Sampling defaults restored.");
}

async function handlePdbUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await readValidatedTextFile(file, "pdb");
  if (!text) {
    event.target.value = "";
    updateRunSummary();
    return;
  }
  clearLoadedStudyForCustomInput();
  state.customPdb = { name: file.name, text };
  state.stagedPreprocessManifest = null;
  renderSetupPreprocessSummary();
  state.batchInputs = [];
  updateBatchLabel();
  $("#pdb-name").textContent = file.name;
  $("#pdb-detail").textContent = `${formatBytes(file.size)} · local upload`;
  if (!state.customSdf) {
    $("#sdf-name").textContent = "Choose a reference SDF";
    $("#sdf-detail").textContent = state.engine === "diffsmol"
      ? "Required 3D reference ligand for shape expansion"
      : "Required for protein + ligand mode";
  }
  $("#example-select").value = "custom";
  state.exampleId = "custom";
  updateCustomOptionLabel(file.name);
  updateVinaControls();
  updateRunSummary();
}

async function handleSdfUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await readValidatedTextFile(file, "sdf");
  if (!text) {
    event.target.value = "";
    updateRunSummary();
    return;
  }
  const moleculeCount = countSdfMolecules(text);
  if (moleculeCount !== 1) {
    showToast(`${file.name} has ${moleculeCount || "no"} molecules. Upload one 3D reference ligand SDF here; use Docking panel for ligand databases.`);
    event.target.value = "";
    return;
  }
  const heavyAtoms = sdfHeavyAtomCount(text);
  clearLoadedStudyForCustomInput();
  state.customSdf = { name: file.name, text };
  const useSdfOnlyDiffSmol = state.engine !== "diffsmol" && !setupPdbInput();
  if (useSdfOnlyDiffSmol) setEngine("diffsmol");
  if (state.engine === "diffsmol") setMode("reference", false);
  state.stagedPreprocessManifest = null;
  renderSetupPreprocessSummary();
  state.batchInputs = [];
  updateBatchLabel();
  $("#sdf-name").textContent = file.name;
  $("#sdf-detail").textContent = [
    `${formatBytes(file.size)} · local upload`,
    heavyAtoms ? `${heavyAtoms} heavy atoms` : "",
  ].filter(Boolean).join(" · ");
  if (!state.customPdb) {
    $("#pdb-name").textContent = "Choose a PDB file";
    $("#pdb-detail").textContent = state.engine === "diffsmol"
      ? "Optional: add a pocket PDB for pocket-conditioned generation"
      : "Required · uploaded with this job";
  }
  $("#example-select").value = "custom";
  state.exampleId = "custom";
  updateCustomOptionLabel(state.customPdb?.name || file.name);
  updateVinaControls();
  updateRunSummary();
  const message = useSdfOnlyDiffSmol ? `${file.name} loaded for DiffSMol shape generation.` : `${file.name} loaded as reference ligand.`;
  showToast(message);
}

async function handlePreprocessTargetUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await readValidatedTextFile(file, "pdb");
  if (!text) return;
  state.preprocessTarget = { name: file.name, text };
  state.preprocessProtein = state.preprocessProtein || state.preprocessTarget;
  $("#preprocess-target-name").textContent = file.name;
  $("#preprocess-target-detail").textContent = `${formatBytes(file.size)} · active target`;
  if ($("#preprocess-protein-name").textContent === "Choose a protein PDB") {
    $("#preprocess-protein-name").textContent = file.name;
    $("#preprocess-protein-detail").textContent = `${formatBytes(file.size)} · receptor`;
  }
  addPreparedStructure({
    mode: "pocket",
    label: file.name,
    source: "Uploaded target structure",
    groupLabel: "Target upload",
    sourceFiles: { protein: file.name },
    pdb: state.preprocessTarget,
    sdf: null,
    metadata: {},
    detail: "Uploaded target structure",
  });
  await renderPreprocessViewer({
    title: file.name,
    pdb: state.preprocessTarget,
    sdf: centerMarkerMolecule({ center: { x: 0, y: 0, z: 0 } }),
  });
  showToast("Target loaded for preprocessing.");
}

function residuePocketPayload() {
  const source = state.preprocessTarget || state.preprocessProtein || state.customPdb;
  const residues = $("#manual-residue-spec").value.trim();
  if (!source?.text) {
    throw new Error("Choose a target protein first.");
  }
  if (!residues) {
    throw new Error("Enter residues like A:45-62, A:88.");
  }
  const radiusText = $("#manual-residue-radius").value;
  return {
    source,
    residues,
    payload: {
      pdb: source,
      residues,
      radius: radiusText === "" ? null : Number(radiusText),
    },
  };
}

async function runResiduePocketAction({ save }) {
  const button = save ? $("#preprocess-build-residue-pocket") : $("#preprocess-preview-residue-pocket");
  button.disabled = true;
  $("#preprocess-target-detail").textContent = save ? "Building residue pocket..." : "Previewing residue pocket...";
  try {
    const { source, residues, payload } = residuePocketPayload();
    const result = await service.preprocessPocketFromResidues(payload);
    const meta = result.metadata || {};
    $("#preprocess-target-detail").textContent = `${meta.residue_count || 0} residues · ${meta.atom_count || 0} atoms`;
    await renderPreprocessViewer({
      title: result.pdb?.name || "Residue pocket preview",
      pdb: result.pdb,
      sdf: centerMarkerMolecule({ center: { x: 0, y: 0, z: 0 } }),
    });
    if (!save) {
      showToast("Residue pocket preview updated.");
      return;
    }
    const entry = addPreparedStructure({
      mode: "pocket",
      label: result.pdb.name,
      source: `Manual residues · ${residues}`,
      groupLabel: `Residue pocket from ${source.name}`,
      sourceFiles: { protein: source.name, residues },
      pdb: result.pdb,
      sdf: null,
      metadata: meta,
      detail: `Residue pocket · ${meta.residue_count || "n/a"} residues`,
    });
    await previewPreparedStructure(entry);
    showToast("Residue pocket added to prepared structures.");
  } catch (error) {
    $("#preprocess-target-detail").textContent = save ? "Residue pocket failed" : "Residue preview failed";
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

async function previewResiduePocket() {
  await runResiduePocketAction({ save: false });
}

async function buildResiduePocket() {
  await runResiduePocketAction({ save: true });
}

async function handlePreprocessComplexUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await readValidatedTextFile(file, "pdb");
  if (!text) return;
  state.preprocessComplex = { name: file.name, text };
  state.preprocessComplexResult = null;
  state.preprocessComplexResultSignature = null;
  renderPreparedStructureTray();
  $("#preprocess-complex-name").textContent = file.name;
  $("#preprocess-complex-detail").textContent = "Detecting ligands and preparing pocket...";
  $("#preprocess-complex-picker").hidden = true;
  $("#preprocess-complex-output").hidden = true;
  try {
    const result = await service.preprocessComplex({ name: file.name, text, pocket_radius: preprocessComplexRadius() });
    handlePreprocessComplexResult(result);
  } catch (error) {
    $("#preprocess-complex-detail").textContent = "Preprocessing failed";
    showToast(error.message);
  }
}

async function applyPreprocessSelectedLigand() {
  if (!state.preprocessComplex) return;
  const ligandId = $("#preprocess-ligand-select").value;
  if (!ligandId) {
    showToast("Choose a ligand first.");
    return;
  }
  $("#preprocess-complex-detail").textContent = "Preparing selected ligand and pocket...";
  try {
    const result = await service.preprocessComplex({
      name: state.preprocessComplex.name,
      text: state.preprocessComplex.text,
      ligand_id: ligandId,
      pocket_radius: preprocessComplexRadius(),
    });
    handlePreprocessComplexResult(result);
  } catch (error) {
    $("#preprocess-complex-detail").textContent = "Preprocessing failed";
    showToast(error.message);
  }
}

function handlePreprocessComplexResult(result) {
  if (result.status === "needs_ligand") {
    const candidates = result.candidates || [];
    renderPreprocessLigandPicker(candidates, result.selected?.id || "");
    $("#preprocess-complex-detail").textContent = `${candidates.length} ligands/cofactors found`;
    showToast(result.message || "Choose which ligand should define the pocket.");
    return;
  }
  if (result.status !== "ready" || !result.pdb?.text || !result.sdf?.text) {
    showToast("Preprocessing did not return prepared files.");
    return;
  }
  state.preprocessComplexResult = result;
  renderPreprocessLigandPicker(result.candidates || [], result.selected?.id || "");
  state.preprocessComplexResultSignature = boundComplexSettingsSignature();
  $("#preprocess-complex-detail").textContent = `Prepared using ${result.selected?.label || "selected ligand"}`;
  renderPreparedComplexOutput(result);
  const complexGroup = `${state.preprocessComplex?.name || result.pdb.name} · ${result.selected?.label || "selected ligand"}`;
  const complexSourceFiles = {
    complex: state.preprocessComplex?.name || null,
    protein: result.pdb?.name || null,
    ligand: result.sdf?.name || null,
    pocket: result.pocket_pdb?.name || null,
  };
  addPreparedStructure({
    mode: "context",
    contextType: "bound-complex",
    settingsSignature: state.preprocessComplexResultSignature,
    selectedVariantId: "pocket",
    label: `Bound complex: ${result.selected?.resname || result.selected?.label || "selected ligand"}`,
    source: "Bound-ligand complex split",
    groupLabel: complexGroup,
    sourceFiles: complexSourceFiles,
    pdb: result.pdb,
    sdf: result.sdf,
    metadata: result.pocket_pdb?.metadata || {},
    detail: `Reference ligand · ${result.selected?.label || "selected ligand"}`,
    variants: [
      {
        id: "reference",
        mode: "reference",
        label: "Full protein + reference ligand",
        pdb: result.pdb,
        sdf: result.sdf,
        metadata: result.pocket_pdb?.metadata || {},
        detail: `Reference ligand · ${result.selected?.label || "selected ligand"}`,
      },
      {
        id: "shape",
        mode: "shape",
        label: "Reference ligand only",
        pdb: null,
        sdf: result.sdf,
        metadata: {
          method: "bound_ligand_shape",
          ligand: result.selected || null,
        },
        detail: `DiffSMol shape input · ${result.selected?.label || "selected ligand"}`,
      },
      ...(result.pocket_pdb?.text ? [{
        id: "pocket",
        mode: "pocket",
        label: "Cropped pocket",
        pdb: result.pocket_pdb,
        sdf: null,
        metadata: result.pocket_pdb.metadata || {},
        detail: `${result.pocket_pdb.metadata?.radius || preprocessComplexRadius()} A pocket · ${result.pocket_pdb.metadata?.residue_count || "n/a"} residues`,
      }] : []),
    ],
  });
  renderPreprocessedComplexPreview(result, "pocket");
}

function renderPreprocessLigandPicker(candidates, selectedId = "") {
  const ligandCandidates = (candidates || []).filter((item) => item.kind === "ligand");
  const options = ligandCandidates.length ? ligandCandidates : (candidates || []);
  if (!options.length) {
    $("#preprocess-complex-picker").hidden = true;
    return;
  }
  $("#preprocess-ligand-select").innerHTML = options
    .map((item) => `<option value="${escapeHtml(item.id)}" ${item.id === selectedId ? "selected" : ""}>${escapeHtml(item.label)}</option>`)
    .join("");
  $("#preprocess-complex-picker").hidden = false;
}

function preprocessComplexRadius() {
  const value = Number($("#preprocess-complex-radius")?.value || 10);
  return Number.isFinite(value) && value > 0 ? value : 10;
}

function boundComplexSettingsSignature() {
  return JSON.stringify({
    complex: state.preprocessComplex?.name || null,
    radius: preprocessComplexRadius(),
    ligand: $("#preprocess-ligand-select")?.value
      || state.preprocessComplexResult?.selected?.id
      || state.preprocessComplexResult?.selected?.label
      || null,
  });
}

function preprocessedComplexVariant(result, variantId = "pocket") {
  if (!result) return null;
  if (variantId === "shape" && result.sdf?.text) {
    return {
      id: "shape",
      mode: "shape",
      pdb: null,
      sdf: result.sdf,
      title: "Reference ligand only",
      detail: `DiffSMol shape input · ${result.selected?.label || "selected ligand"}`,
      metadata: {
        method: "bound_ligand_shape",
        ligand: result.selected || null,
      },
    };
  }
  if (variantId === "pocket" && result.pocket_pdb?.text) {
    return {
      id: "pocket",
      mode: "pocket",
      pdb: result.pocket_pdb,
      sdf: null,
      title: "Cropped pocket",
      detail: `${result.pocket_pdb.metadata?.radius || preprocessComplexRadius()} A pocket · ${result.pocket_pdb.metadata?.residue_count || "n/a"} residues`,
    };
  }
  return {
    id: "reference",
    mode: "reference",
    pdb: result.pdb,
    sdf: result.sdf,
    title: "Full protein + reference ligand",
    detail: `Reference ligand · ${result.selected?.label || "selected ligand"}`,
  };
}

function renderPreparedComplexOutput(result) {
  const pocketMeta = result.pocket_pdb?.metadata || {};
  $("#preprocess-complex-output-title").textContent = "Bound-complex context saved to Candidate inputs";
  $("#preprocess-complex-output-grid").innerHTML = [
    ["Protein", result.pdb?.name || "protein.pdb"],
    ["Reference ligand", result.sdf?.name || "reference.sdf"],
    ["Pocket", result.pocket_pdb?.name || "pocket.pdb"],
    ["Pocket residues", pocketMeta.residue_count ? String(pocketMeta.residue_count) : "n/a"],
    ["Pocket atoms", pocketMeta.atom_count ? String(pocketMeta.atom_count) : "n/a"],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  renderPreparedMetadata("#preprocess-complex-meta", [
    pocketMeta.provenance,
    ...(pocketMeta.warnings || []),
  ]);
  $("#preprocess-complex-output").hidden = false;
}

async function renderPreprocessedComplexPreview(result, variantId = "pocket") {
  const variant = preprocessedComplexVariant(result, variantId);
  if (!variant?.pdb?.text && !variant?.sdf?.text) return;
  await renderPreprocessViewer({
    title: variant.title,
    pdb: variant.pdb,
    sdf: variant.sdf,
  });
}

async function previewPreprocessedComplexVariant(variantId) {
  if (!state.preprocessComplexResult) {
    showToast("Preprocess a complex first.");
    return;
  }
  const contextEntry = state.preparedStructures.find((item) => item.contextType === "bound-complex" && item.sourceFiles?.complex === state.preprocessComplex?.name);
  if (contextEntry) {
    setPreparedContextVariant(contextEntry.id, variantId);
    state.selectedPreparedStructureId = contextEntry.id;
    renderPreparedStructureTray();
  }
  await renderPreprocessedComplexPreview(state.preprocessComplexResult, variantId);
}

function renderPreparedMetadata(selector, lines) {
  const container = $(selector);
  if (!container) return;
  const items = [...new Set((lines || []).filter(Boolean))];
  container.innerHTML = items.map((line, index) => `
    <p><strong>${index === 0 ? "Source" : "Note"}:</strong> ${escapeHtml(line)}</p>
  `).join("");
}

function addPreparedStructure(entry) {
  const id = entry.id || `prepared-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const normalized = { ...entry, id, createdAt: new Date().toISOString() };
  state.preparedStructures = [normalized, ...state.preparedStructures.filter((item) => item.id !== id)].slice(0, 12);
  state.selectedPreparedStructureId = id;
  renderPreparedStructureTray();
  renderPreprocessStagingStatus();
  return normalized;
}

function renderPreparedStructureTray() {
  const tray = $("#prepared-structure-tray");
  const count = state.preparedStructures.length;
  $("#prepared-structure-count").textContent = `${count} candidate${count === 1 ? "" : "s"}`;
  if (!count) {
    tray.innerHTML = "<p class='field-note'>Candidate protein, pocket, and ligand sets appear here.</p>";
    renderPreprocessStagingStatus();
    return;
  }
  tray.innerHTML = state.preparedStructures.map((item) => {
    const variant = selectedPreparedVariant(item);
    const meta = variant.metadata || item.metadata || {};
    const active = item.id === state.selectedPreparedStructureId ? " active" : "";
    const isStagedView = item.id === state.stagedPreparedStructureId && (!item.variants?.length || item.selectedVariantId === state.stagedPreparedVariantId);
    const staged = isStagedView ? " staged" : "";
    const cardClass = `prepared-structure-card${active}${staged}`;
    const mode = variant.mode || item.mode;
    const modeLabel = preparedModeLabel(mode);
    const residues = meta.residue_count ? `${meta.residue_count} residues` : "residues n/a";
    const atoms = meta.atom_count ? `${meta.atom_count} atoms` : "atoms n/a";
    const files = preparedStructureFileLines(item);
    const sourceLabel = item.groupLabel || item.source || meta.provenance || "Prepared structure";
    const statusLabel = isStagedView ? "Staged" : active ? "Previewing" : modeLabel;
    const warningLines = preprocessWarningObjects(meta).map(warningText);
    const variantSelector = item.variants?.length ? `
          <label class="prepared-variant-control">
            <span>View</span>
            <select data-prepared-action="variant">
              ${item.variants.map((option) => `<option value="${escapeHtml(option.id)}" ${option.id === item.selectedVariantId ? "selected" : ""}>${escapeHtml(option.label)}</option>`).join("")}
            </select>
          </label>
    ` : "";
    return `
      <article class="${cardClass}" data-prepared-id="${escapeHtml(item.id)}">
        <div>
          <div class="prepared-structure-title-row">
            <strong>${escapeHtml(item.label || item.pdb?.name || "Prepared structure")}</strong>
            <span>${escapeHtml(statusLabel)}</span>
          </div>
          <small>${escapeHtml(item.variants?.length ? `Selected view: ${variant.label || modeLabel}` : modeLabel)} · ${escapeHtml(residues)} · ${escapeHtml(atoms)}</small>
          <small class="prepared-source-line">${escapeHtml(sourceLabel)}</small>
          ${warningLines.map((line) => `<small class="prepared-warning-line">${escapeHtml(line)}</small>`).join("")}
          ${variantSelector}
          <details class="prepared-provenance">
            <summary>Source details</summary>
            ${item.groupLabel ? `<p><b>Set</b>${escapeHtml(item.groupLabel)}</p>` : ""}
            <p><b>Source</b>${escapeHtml(item.source || meta.provenance || "Prepared structure")}</p>
            ${files.map(([label, value]) => `<p><b>${escapeHtml(label)}</b>${escapeHtml(value)}</p>`).join("")}
          </details>
        </div>
        <div class="prepared-structure-actions">
          <button class="secondary-button compact-action" type="button" data-prepared-action="preview">Preview</button>
          <button class="secondary-button compact-action" type="button" data-prepared-action="download">Download</button>
          <button class="secondary-button compact-action danger-action" type="button" data-prepared-action="remove">Remove</button>
          <button class="primary-button compact-action" type="button" data-prepared-action="use">${isStagedView ? "Staged" : "Stage view"}</button>
        </div>
      </article>
    `;
  }).join("");
  $$("#prepared-structure-tray button[data-prepared-action]").forEach((button) => button.addEventListener("click", handlePreparedStructureAction));
  $$("#prepared-structure-tray select[data-prepared-action]").forEach((select) => select.addEventListener("change", handlePreparedStructureAction));
  renderPreprocessStagingStatus();
}

function boundComplexResultNeedsUpdate() {
  return Boolean(
    state.preprocessComplexResultSignature
    && state.preprocessComplexResultSignature !== boundComplexSettingsSignature(),
  );
}

function vinaPanelResultNeedsUpdate() {
  return Boolean(
    state.vinaPocketResultSignature
    && state.vinaPocketResultSignature !== vinaPanelSettingsSignature(),
  );
}

function selectedPreparedVariant(entry) {
  if (!entry?.variants?.length) return entry || {};
  return entry.variants.find((variant) => variant.id === entry.selectedVariantId) || entry.variants[0] || entry;
}

function preparedVariantById(entry, variantId) {
  if (!entry?.variants?.length) return entry || {};
  return entry.variants.find((variant) => variant.id === variantId) || selectedPreparedVariant(entry);
}

function preparedVariantPdb(entry, variant) {
  return Object.prototype.hasOwnProperty.call(variant || {}, "pdb") ? variant.pdb : entry?.pdb;
}

function preparedVariantSdf(entry, variant) {
  return Object.prototype.hasOwnProperty.call(variant || {}, "sdf") ? variant.sdf : entry?.sdf || null;
}

function renderPreprocessStagingStatus() {
  const status = $("#preprocess-staged-status");
  if (!status) return;
  const entry = state.preparedStructures.find((item) => item.id === state.stagedPreparedStructureId);
  status.classList.toggle("is-staged", Boolean(entry));
  const variant = preparedVariantById(entry, state.stagedPreparedVariantId);
  const mode = variant.mode || entry?.mode;
  status.textContent = entry ? `Staged: ${preparedModeLabel(mode).toLowerCase()}` : "Nothing staged";
  status.title = entry ? `${entry.label || ""}${variant.label ? ` · ${variant.label}` : ""}` : "";
}

function preparedModeLabel(mode) {
  if (mode === "reference") return "Protein + ligand";
  if (mode === "pocket") return "Pocket";
  if (mode === "shape") return "Shape ligand";
  return "Context";
}

function preparedStructureFileLines(item) {
  const files = item.sourceFiles || {};
  return [
    ["Complex", files.complex],
    ["Protein", files.protein || item.pdb?.name],
    ["Ligand", files.ligand || item.sdf?.name],
    ["Pocket", files.pocket],
    ["Panel", files.panel],
    ["Residues", files.residues],
  ].filter(([, value]) => value);
}

async function handlePreparedStructureAction(event) {
  const card = event.target.closest("[data-prepared-id]");
  let entry = state.preparedStructures.find((item) => item.id === card?.dataset.preparedId);
  if (!entry) return;
  const action = event.target.dataset.preparedAction;
  if (action === "variant") {
    await updatePreparedStructureVariant(entry.id, event.target.value);
    return;
  }
  if (action === "remove") {
    removePreparedStructure(entry.id);
    return;
  }
  state.selectedPreparedStructureId = entry.id;
  renderPreparedStructureTray();
  entry = await resolvePreparedStructure(entry);
  if (!entry) return;
  const variant = selectedPreparedVariant(entry);
  const pdb = preparedVariantPdb(entry, variant);
  const sdf = preparedVariantSdf(entry, variant);
  const stageMode = variant.mode || entry.mode;
  if (!pdb?.text && !sdf?.text) return;
  if (action === "use") {
    state.stagedPreparedStructureId = entry.id;
    state.stagedPreparedVariantId = variant.id || null;
    renderPreparedStructureTray();
    stagePreparedInputs({
      mode: stageMode,
      pdb,
      sdf: stageMode === "reference" || stageMode === "shape" ? sdf : null,
      label: entry.label || pdb?.name || sdf?.name,
      detail: variant.detail || entry.detail || entry.source || "Prepared structure",
      preprocessManifest: buildPreprocessManifest(entry, variant, {
        mode: stageMode,
        pdb,
        sdf: stageMode === "reference" || stageMode === "shape" ? sdf : null,
      }),
    });
    return;
  }
  if (action === "download") {
    await downloadPreparedStructure(entry);
    return;
  }
  await previewPreparedStructure(entry);
}

async function updatePreparedStructureVariant(id, variantId) {
  let updatedEntry = null;
  state.preparedStructures = state.preparedStructures.map((item) => {
    if (item.id !== id) return item;
    updatedEntry = { ...item, selectedVariantId: variantId };
    return updatedEntry;
  });
  state.selectedPreparedStructureId = id;
  if (updatedEntry?.contextType === "vina-panel") {
    state.selectedVinaPocketId = variantId;
    renderVinaPocketCandidates();
  }
  renderPreparedStructureTray();
  if (updatedEntry) {
    const resolved = await resolvePreparedStructure(updatedEntry);
    if (resolved) await previewPreparedStructure(resolved);
  }
}

function removePreparedStructure(id) {
  const removed = state.preparedStructures.find((item) => item.id === id);
  const wasStaged = state.stagedPreparedStructureId === id;
  state.preparedStructures = state.preparedStructures.filter((item) => item.id !== id);
  if (state.selectedPreparedStructureId === id) {
    state.selectedPreparedStructureId = state.preparedStructures[0]?.id || null;
  }
  if (wasStaged) clearStagedPreparedInputs();
  renderPreparedStructureTray();
  renderPreprocessStagingStatus();
  showToast(wasStaged
    ? "Removed staged input from Generate setup."
    : removed?.label ? `Removed ${removed.label}.` : "Prepared structure removed.");
}

function clearStagedPreparedInputs() {
  state.stagedPreparedStructureId = null;
  state.stagedPreparedVariantId = null;
  state.stagedPreprocessManifest = null;
  state.customPdb = null;
  state.customSdf = null;
  state.batchInputs = [];
  updateInputLabels(null);
  updateCustomOptionLabel("");
  $("#example-select").value = "custom";
  state.exampleId = "custom";
  renderSetupPreprocessSummary();
  updateBatchLabel();
  updateVinaControls();
  updateRunSummary();
}

async function resolvePreparedStructure(entry) {
  if (!entry) return null;
  if (entry.variants?.length) {
    const variant = selectedPreparedVariant(entry);
    if (variant.pdb?.text) return entry;
    if (variant.prepare === "vina-pocket" && variant.center) {
      const sourceProtein = entry.sourceProtein || state.preprocessProtein;
      if (!sourceProtein) {
        showToast("The source protein for this Vina pocket is no longer loaded.");
        return null;
      }
      try {
        const result = await service.preprocessPocketFromCenter({
          pdb: sourceProtein,
          center: variant.center,
          radius: variant.radius || variant.metadata?.radius || 10,
        });
        const updatedVariant = {
          ...variant,
          pdb: result.pdb,
          metadata: { ...(variant.metadata || {}), ...(result.metadata || {}) },
          detail: `Vina cluster · ${result.metadata?.residue_count || "n/a"} residues`,
        };
        const updated = {
          ...entry,
          sourceFiles: { ...(entry.sourceFiles || {}), pocket: result.pdb?.name || null },
          variants: entry.variants.map((item) => item.id === variant.id ? updatedVariant : item),
        };
        state.preparedStructures = state.preparedStructures.map((item) => item.id === entry.id ? updated : item);
        renderPreparedStructureTray();
        return updated;
      } catch (error) {
        showToast(error.message);
        return null;
      }
    }
  }
  if (entry.pdb?.text) return entry;
  if (entry.prepare === "vina-pocket") {
    if (!state.preprocessProtein || !entry.center) {
      showToast("The source protein for this Vina pocket is no longer loaded.");
      return null;
    }
    try {
      const result = await service.preprocessPocketFromCenter({
        pdb: state.preprocessProtein,
        center: entry.center,
        radius: entry.metadata?.radius || 10,
      });
      const updated = {
        ...entry,
        pdb: result.pdb,
        metadata: { ...(entry.metadata || {}), ...(result.metadata || {}) },
        sourceFiles: { ...(entry.sourceFiles || {}), pocket: result.pdb?.name || null },
        detail: `Vina cluster · ${result.metadata?.residue_count || "n/a"} residues`,
      };
      state.preparedStructures = state.preparedStructures.map((item) => item.id === entry.id ? updated : item);
      renderPreparedStructureTray();
      return updated;
    } catch (error) {
      showToast(error.message);
      return null;
    }
  }
  showToast("This prepared structure is missing a PDB.");
  return null;
}

async function previewPreparedStructure(entry) {
  const variant = selectedPreparedVariant(entry);
  const pdb = preparedVariantPdb(entry, variant);
  const sdf = preparedVariantSdf(entry, variant);
  if (!pdb?.text && !sdf?.text) return;
  await renderPreprocessViewer({
    title: variant.label || entry.label || pdb?.name || sdf?.name,
    pdb,
    sdf: sdf || (entry.center ? centerMarkerMolecule({ center: entry.center }) : null),
  });
}

async function renderPreprocessViewer({ title, pdb, sdf }) {
  const container = $("#preprocess-main-viewer-3d");
  const loading = $("#preprocess-main-viewer-loading");
  const heading = $("#preprocess-main-viewer-title");
  if (!container || (!pdb?.text && !sdf?.text)) return;
  container.closest(".preprocess-main-viewer-panel")?.classList.remove("empty-viewer");
  heading.textContent = title || pdb?.name || sdf?.name || "Prepared structure";
  loading.hidden = false;
  try {
    container.innerHTML = "";
    await render3D(container, sdf || null, pdb?.text || "", {
      controlsHost: $("#preprocess-viewer-controls"),
      ligandLabel: sdf?.id === "Pocket center" ? "Pocket center" : "Ligand",
      onUseResidues: (spec) => {
        state.preprocessTarget = pdb;
        $("#preprocess-target-name").textContent = pdb.name || "Viewed protein";
        $("#preprocess-target-detail").textContent = "Selected from the active viewer";
        $("#manual-residue-spec").value = spec;
        $("#manual-residue-spec").closest("details").open = true;
        showToast("Residues added to the known-site pocket input. Preview before saving.");
      },
    });
  } catch (error) {
    container.innerHTML = `<div class="viewer-error">${escapeHtml(error.message)}</div>`;
  } finally {
    loading.hidden = true;
  }
}

async function downloadPreparedStructure(entry) {
  const variant = selectedPreparedVariant(entry);
  const pdb = preparedVariantPdb(entry, variant);
  const sdf = preparedVariantSdf(entry, variant);
  await downloadPreparedArchive(`${filenameStem(pdb?.name || sdf?.name || "prepared_structure")}_preprocess.zip`, [
    pdb,
    sdf,
    {
      name: "preprocess_metadata.json",
      text: JSON.stringify({
        label: entry.label,
        mode: variant.mode || entry.mode,
        variant: variant.label || null,
        source: entry.source,
        files: {
          pdb: pdb?.name || null,
          sdf: sdf?.name || null,
        },
        source_files: entry.sourceFiles || {},
        set: entry.groupLabel || null,
        metadata: variant.metadata || entry.metadata || {},
      }, null, 2),
    },
  ]);
  showToast("Prepared structure downloaded.");
}

async function downloadPreprocessedComplex() {
  const result = state.preprocessComplexResult;
  if (!result?.pdb?.text || !result?.sdf?.text) {
    showToast("Preprocess a complex first.");
    return;
  }
  await downloadPreparedArchive("conditar_complex_inputs.zip", [
    result.pdb,
    result.sdf,
    result.pocket_pdb,
    {
      name: "preprocess_metadata.json",
      text: JSON.stringify(preprocessComplexMetadata(result), null, 2),
    },
  ]);
  showToast("Prepared complex inputs downloaded.");
}

function preprocessComplexMetadata(result) {
  return {
    workflow: "complex_with_bound_ligand",
    selected_ligand: result.selected || null,
    files: {
      protein: result.pdb?.name || null,
      reference_ligand: result.sdf?.name || null,
      pocket: result.pocket_pdb?.name || null,
    },
    pocket: result.pocket_pdb?.metadata || {},
  };
}

function usePreprocessedComplex(mode) {
  const result = state.preprocessComplexResult;
  if (!result) {
    showToast("Preprocess a complex first.");
    return;
  }
  const contextEntry = state.preparedStructures.find((item) => item.contextType === "bound-complex" && item.sourceFiles?.complex === state.preprocessComplex?.name);
  if (boundComplexResultNeedsUpdate()) {
    showToast("Bound-complex settings changed. Regenerate the context before staging.");
    return;
  }
  if (mode === "shape") {
    const shapeSdf = boundComplexShapeSdf(result, contextEntry);
    if (!shapeSdf?.text) {
      showToast("No reference ligand SDF is available.");
      return;
    }
    if (contextEntry) setPreparedContextVariant(contextEntry.id, "shape");
    stagePreparedInputs({
      mode: "shape",
      pdb: null,
      sdf: shapeSdf,
      label: shapeSdf.name,
      detail: `DiffSMol shape input · ${result.selected?.label || "selected ligand"}`,
      preprocessManifest: buildPreprocessManifest(contextEntry || {
        contextType: "bound-complex",
        source: "Bound-ligand complex split",
        groupLabel: state.preprocessComplex?.name || null,
        sourceFiles: { complex: state.preprocessComplex?.name || null, ligand: shapeSdf?.name || null },
        metadata: { method: "bound_ligand_shape", ligand: result.selected || null },
      }, preprocessedComplexVariant(result, "shape"), {
        mode: "shape",
        pdb: null,
        sdf: shapeSdf,
      }),
    });
    markPreparedStructureStaged((item) => item.id === contextEntry?.id, "shape");
    return;
  }
  if (mode === "pocket") {
    if (!result.pocket_pdb?.text) {
      showToast("No cropped pocket PDB is available.");
      return;
    }
    if (contextEntry) setPreparedContextVariant(contextEntry.id, "pocket");
    stagePreparedInputs({
      mode: "pocket",
      pdb: result.pocket_pdb,
      sdf: null,
      label: result.pocket_pdb.name,
      detail: `${result.pocket_pdb.metadata?.radius || preprocessComplexRadius()} A pocket · ${result.pocket_pdb.metadata?.residue_count || "n/a"} residues`,
      preprocessManifest: buildPreprocessManifest(contextEntry || {
        contextType: "bound-complex",
        source: "Bound-ligand complex split",
        groupLabel: state.preprocessComplex?.name || null,
        sourceFiles: { complex: state.preprocessComplex?.name || null },
        metadata: result.pocket_pdb?.metadata || {},
      }, preprocessedComplexVariant(result, "pocket"), {
        mode: "pocket",
        pdb: result.pocket_pdb,
        sdf: null,
      }),
    });
    markPreparedStructureStaged((item) => item.id === contextEntry?.id, "pocket");
    return;
  }
  if (contextEntry) setPreparedContextVariant(contextEntry.id, "reference");
  stagePreparedInputs({
    mode: "reference",
    pdb: result.pdb,
    sdf: result.sdf,
    label: state.preprocessComplex?.name || result.pdb.name,
    detail: `Reference ligand · ${result.selected?.label || "selected ligand"}`,
    preprocessManifest: buildPreprocessManifest(contextEntry || {
      contextType: "bound-complex",
      source: "Bound-ligand complex split",
      groupLabel: state.preprocessComplex?.name || null,
      sourceFiles: { complex: state.preprocessComplex?.name || null },
      metadata: result.pocket_pdb?.metadata || {},
    }, preprocessedComplexVariant(result, "reference"), {
      mode: "reference",
      pdb: result.pdb,
      sdf: result.sdf,
    }),
  });
  markPreparedStructureStaged((item) => item.id === contextEntry?.id, "reference");
}

function boundComplexShapeSdf(result, contextEntry) {
  if (result?.sdf?.text) return result.sdf;
  const variants = contextEntry?.variants || [];
  return variants.find((variant) => variant.id === "shape")?.sdf
    || variants.find((variant) => variant.id === "reference")?.sdf
    || contextEntry?.sdf
    || null;
}

function setPreparedContextVariant(id, variantId) {
  state.preparedStructures = state.preparedStructures.map((item) => item.id === id ? { ...item, selectedVariantId: variantId } : item);
}

function markPreparedStructureStaged(predicate, variantId = null) {
  const entry = state.preparedStructures.find(predicate);
  if (!entry) return;
  state.stagedPreparedStructureId = entry.id;
  state.stagedPreparedVariantId = variantId || entry.selectedVariantId || null;
  state.selectedPreparedStructureId = entry.id;
  renderPreparedStructureTray();
  renderPreprocessStagingStatus();
}

async function handlePreprocessProteinUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await readValidatedTextFile(file, "pdb");
  if (!text) return;
  state.preprocessProtein = { name: file.name, text };
  state.vinaPocketResult = null;
  state.vinaPocketResultSignature = null;
  state.selectedVinaPocketId = null;
  renderPreparedStructureTray();
  $("#preprocess-protein-name").textContent = file.name;
  $("#preprocess-protein-detail").textContent = `${formatBytes(file.size)} · receptor`;
  $("#vina-pocket-output").hidden = true;
  await renderPreprocessViewer({
    title: file.name,
    pdb: state.preprocessProtein,
    sdf: centerMarkerMolecule({ center: { x: 0, y: 0, z: 0 } }),
  });
}

async function handlePreprocessPanelUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  try {
    $("#preprocess-panel-detail").textContent = "Preparing ligand panel...";
    const panel = await readLigandPanelFile(file);
    if (!panel) return;
    state.preprocessPanel = panel;
    state.vinaPocketResult = null;
    state.vinaPocketResultSignature = null;
    state.selectedVinaPocketId = null;
    renderPreparedStructureTray();
    $("#preprocess-panel-name").textContent = file.name;
    $("#preprocess-panel-detail").textContent = `${panel.count || "SDF"} ligand${panel.count === 1 ? "" : "s"} · ${panel.source || "ligand panel"}`;
    $("#vina-pocket-output").hidden = true;
  } catch (error) {
    $("#preprocess-panel-detail").textContent = "Ligand panel failed";
    showToast(error.message);
  }
}

async function runVinaPanelPreprocessing() {
  if (!state.preprocessProtein || !state.preprocessPanel) {
    showToast("Choose a protein structure and ligand panel SDF or CSV first.");
    return;
  }
  const button = $("#preprocess-run-vina-panel");
  button.disabled = true;
  $("#preprocess-panel-detail").textContent = "Running Vina panel...";
  try {
    const options = vinaPanelOptions();
    const result = await service.preprocessVinaPanelPockets({
      pdb: state.preprocessProtein,
      ligands: state.preprocessPanel,
      options,
    });
    state.vinaPocketResult = result;
    state.vinaPocketResultSignature = vinaPanelSettingsSignature();
    state.selectedVinaPocketId = result.candidates?.[0]?.id || null;
    renderVinaPocketCandidates();
    saveVinaPanelContext(result);
    renderPreparedMetadata("#vina-pocket-meta", [
      result.provenance,
      ...(result.warnings || []),
      ...(result.vina_panel?.warnings || []),
    ]);
    renderSelectedVinaPocketPreview();
    $("#preprocess-panel-detail").textContent = `${result.vina_panel?.pose_count || 0} docked poses clustered`;
    showToast(`${result.candidates?.length || 0} candidate pocket${result.candidates?.length === 1 ? "" : "s"} saved to Candidate inputs.`);
  } catch (error) {
    $("#preprocess-panel-detail").textContent = "Vina panel failed";
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

function saveVinaPanelContext(result) {
  const candidates = result.candidates || [];
  if (!candidates.length) return;
  const proteinName = state.preprocessProtein?.name || "protein";
  const panelName = state.preprocessPanel?.name || "panel";
  addPreparedStructure({
    id: `vina-context-${filenameStem(proteinName)}-${filenameStem(panelName)}`,
    mode: "context",
    contextType: "vina-panel",
    settingsSignature: state.vinaPocketResultSignature,
    selectedVariantId: state.selectedVinaPocketId || candidates[0].id,
    label: `Docking panel: ${panelName}`,
    source: "Docking-panel pocket search",
    groupLabel: `${proteinName} · ${panelName}`,
    sourceProtein: state.preprocessProtein,
    sourceFiles: {
      protein: proteinName,
      panel: panelName,
    },
    metadata: result.vina_panel || {},
    detail: `${candidates.length} pocket candidate${candidates.length === 1 ? "" : "s"}`,
    variants: candidates.map((candidate) => {
      const pose = candidate.representative_pose;
      return {
        id: candidate.id,
        mode: "pocket",
        prepare: "vina-pocket",
        label: candidate.label || candidate.id,
        center: candidate.center,
        radius: candidate.radius || 10,
        sdf: pose?.sdf ? { name: `${filenameStem(pose.name || candidate.id)}.sdf`, text: pose.sdf } : null,
        metadata: candidate,
        detail: `Vina cluster · ${candidate.supporting_ligand_count || 0} ligand${candidate.supporting_ligand_count === 1 ? "" : "s"}`,
      };
    }),
  });
}

function vinaPanelSettingsSignature() {
  return JSON.stringify({
    protein: state.preprocessProtein?.name || null,
    panel: state.preprocessPanel?.name || null,
    options: vinaPanelOptions(),
  });
}

function vinaPanelOptions() {
  const optionIds = {
    center_x: "#vina-panel-center-x",
    center_y: "#vina-panel-center-y",
    center_z: "#vina-panel-center-z",
    size_x: "#vina-panel-size-x",
    size_y: "#vina-panel-size-y",
    size_z: "#vina-panel-size-z",
    max_ligands: "#vina-panel-max-ligands",
    exhaustiveness: "#vina-panel-exhaustiveness",
  };
  const options = { cpu: 1, pocket_radius: 10, cluster_distance: 6 };
  Object.entries(optionIds).forEach(([key, selector]) => {
    const value = $(selector).value;
    if (value !== "") options[key] = Number(value);
  });
  return options;
}

function renderVinaPocketCandidates() {
  const candidates = state.vinaPocketResult?.candidates || [];
  $("#vina-pocket-candidates").innerHTML = candidates.length ? candidates.map((candidate) => `
    <label class="pocket-candidate ${candidate.id === state.selectedVinaPocketId ? "active" : ""}">
      <input type="radio" name="vina-pocket-candidate" value="${escapeHtml(candidate.id)}" ${candidate.id === state.selectedVinaPocketId ? "checked" : ""}>
      <span><strong>${escapeHtml(candidate.label || candidate.id)}</strong><small>${escapeHtml(pocketCandidateSummary(candidate))}</small></span>
    </label>
  `).join("") : "<p class='field-note'>No pocket candidates were found.</p>";
  $$("#vina-pocket-candidates input").forEach((input) => input.addEventListener("change", (event) => {
    state.selectedVinaPocketId = event.target.value;
    const contextId = `vina-context-${filenameStem(state.preprocessProtein?.name || "protein")}-${filenameStem(state.preprocessPanel?.name || "panel")}`;
    setPreparedContextVariant(contextId, state.selectedVinaPocketId);
    renderVinaPocketCandidates();
    renderPreparedStructureTray();
    renderSelectedVinaPocketPreview();
  }));
  $("#vina-pocket-output").hidden = false;
}

function pocketCandidateSummary(candidate) {
  const center = candidate.center || {};
  const score = candidate.best_score == null ? "n/a" : Number(candidate.best_score).toFixed(2);
  return `center ${center.x}, ${center.y}, ${center.z} · ${candidate.supporting_ligand_count || 0} ligands · best ${score}`;
}

async function useSelectedVinaPocket() {
  const prepared = await prepareSelectedVinaPocket();
  if (!prepared) return;
  const pose = prepared.candidate.representative_pose;
  const contextId = `vina-context-${filenameStem(state.preprocessProtein?.name || "protein")}-${filenameStem(state.preprocessPanel?.name || "panel")}`;
  let contextEntry = state.preparedStructures.find((item) => item.id === contextId);
  if (vinaPanelResultNeedsUpdate()) {
    showToast("Docking-panel settings changed. Find candidate pockets again before staging.");
    return;
  }
  if (!contextEntry && state.vinaPocketResult) {
    saveVinaPanelContext(state.vinaPocketResult);
    contextEntry = state.preparedStructures.find((item) => item.id === contextId);
  }
  if (contextEntry) {
    state.preparedStructures = state.preparedStructures.map((item) => {
      if (item.id !== contextEntry.id) return item;
      return {
        ...item,
        selectedVariantId: prepared.candidate.id,
        sourceFiles: { ...(item.sourceFiles || {}), ligand: pose?.name || null, pocket: prepared.pdb?.name || null },
        variants: (item.variants || []).map((variant) => variant.id === prepared.candidate.id ? {
          ...variant,
          pdb: prepared.pdb,
          sdf: pose?.sdf ? { name: `${filenameStem(pose.name || prepared.candidate.id)}.sdf`, text: pose.sdf } : variant.sdf || null,
          metadata: { ...(variant.metadata || {}), ...(prepared.metadata || {}) },
          detail: `Vina cluster · ${prepared.metadata?.residue_count || "n/a"} residues`,
        } : variant),
      };
    });
    state.stagedPreparedStructureId = contextEntry.id;
    state.stagedPreparedVariantId = prepared.candidate.id;
    state.selectedPreparedStructureId = contextEntry.id;
  }
  renderPreparedStructureTray();
    stagePreparedInputs({
      mode: "pocket",
      pdb: prepared.pdb,
      sdf: null,
      label: prepared.pdb.name,
      detail: `Vina cluster · ${prepared.metadata?.residue_count || "n/a"} residues`,
      preprocessManifest: buildPreprocessManifest(contextEntry || {
        contextType: "vina-panel",
        source: "Docking-panel pocket search",
        groupLabel: `${state.preprocessProtein?.name || "protein"} · ${state.preprocessPanel?.name || "panel"}`,
        sourceFiles: { protein: state.preprocessProtein?.name || null, panel: state.preprocessPanel?.name || null },
        metadata: state.vinaPocketResult?.vina_panel || {},
      }, {
        id: prepared.candidate.id,
        mode: "pocket",
        label: prepared.candidate.label || prepared.candidate.id,
        center: prepared.candidate.center,
        metadata: { ...(prepared.candidate || {}), ...(prepared.metadata || {}) },
      }, {
        mode: "pocket",
        pdb: prepared.pdb,
        sdf: null,
      }),
    });
}

async function downloadSelectedVinaPocket() {
  const prepared = await prepareSelectedVinaPocket();
  if (!prepared) return;
  const pose = prepared.candidate.representative_pose;
  await downloadPreparedArchive(`${filenameStem(prepared.pdb.name)}_preprocess.zip`, [
    prepared.pdb,
    pose?.sdf ? { name: `${filenameStem(pose.name || "representative_pose")}.sdf`, text: pose.sdf } : null,
    {
      name: "preprocess_metadata.json",
      text: JSON.stringify(preprocessVinaPocketMetadata(prepared), null, 2),
    },
  ]);
  showToast("Selected pocket downloaded.");
}

async function prepareSelectedVinaPocket() {
  const candidate = (state.vinaPocketResult?.candidates || []).find((item) => item.id === state.selectedVinaPocketId);
  if (!candidate || !state.preprocessProtein) {
    showToast("Select a Vina pocket candidate first.");
    return null;
  }
  try {
    const result = await service.preprocessPocketFromCenter({
      pdb: state.preprocessProtein,
      center: candidate.center,
      radius: candidate.radius || 10,
    });
    return { ...result, candidate };
  } catch (error) {
    showToast(error.message);
    return null;
  }
}

function preprocessVinaPocketMetadata(prepared) {
  const result = state.vinaPocketResult || {};
  return {
    workflow: "protein_without_reference_ligand_vina_panel",
    source_protein: state.preprocessProtein?.name || null,
    source_panel: state.preprocessPanel?.name || null,
    selected_candidate: prepared.candidate,
    pocket: prepared.metadata || {},
    vina_panel: result.vina_panel || {},
    warnings: result.warnings || [],
  };
}

async function renderSelectedVinaPocketPreview() {
  const candidate = (state.vinaPocketResult?.candidates || []).find((item) => item.id === state.selectedVinaPocketId);
  if (!candidate || !state.preprocessProtein) return;
  try {
    const pocket = await service.preprocessPocketFromCenter({
      pdb: state.preprocessProtein,
      center: candidate.center,
      radius: candidate.radius || 10,
    });
    const poseText = candidate.representative_pose?.sdf;
    const molecule = poseText
      ? { id: candidate.representative_pose.name || "Representative pose", name: candidate.representative_pose.name || "pose.sdf", text: poseText }
      : centerMarkerMolecule(candidate);
    await renderPreprocessViewer({
      title: candidate.label || candidate.id,
      pdb: pocket.pdb,
      sdf: molecule,
    });
  } catch (error) {
    const container = $("#preprocess-main-viewer-3d");
    if (container) container.innerHTML = `<div class="viewer-error">${escapeHtml(error.message)}</div>`;
  }
}

function centerMarkerMolecule(candidate) {
  const center = candidate.center || { x: 0, y: 0, z: 0 };
  const sdf = `Pocket center\n  conDitar GUI\n\n  1  0  0  0  0  0            999 V2000\n${sdfCoord(center.x)}${sdfCoord(center.y)}${sdfCoord(center.z)} C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n`;
  return { id: "Pocket center", name: "pocket_center.sdf", text: sdf };
}

function sdfCoord(value) {
  return Number(value || 0).toFixed(4).padStart(10, " ");
}

function buildPreprocessManifest(entry, variant, staged) {
  const sourceFiles = entry?.sourceFiles || {};
  const metadata = variant?.metadata || entry?.metadata || {};
  const structuredWarnings = preprocessWarningObjects(metadata);
  return {
    schema_version: 1,
    created_at: new Date().toISOString(),
    workflow: entry?.contextType || entry?.mode || "manual",
    source: entry?.source || metadata.provenance || null,
    set: entry?.groupLabel || null,
    selected_variant: variant ? {
      id: variant.id || null,
      label: variant.label || null,
      mode: variant.mode || staged?.mode || null,
      center: variant.center || null,
      radius: variant.radius || metadata.radius || null,
    } : null,
    source_files: {
      complex: sourceFiles.complex || null,
      protein: sourceFiles.protein || null,
      ligand: sourceFiles.ligand || null,
      pocket: sourceFiles.pocket || null,
      panel: sourceFiles.panel || null,
      residues: sourceFiles.residues || null,
    },
    staged_inputs: {
      mode: staged?.mode || null,
      pdb: staged?.pdb?.name || null,
      sdf: staged?.sdf?.name || null,
    },
    preprocessing: {
      method: metadata.method || entry?.contextType || entry?.mode || null,
      radius: metadata.radius || variant?.radius || null,
      residue_count: metadata.residue_count || null,
      atom_count: metadata.atom_count || null,
      residues: metadata.residues || null,
      warnings: structuredWarnings.map(warningText),
      structured_warnings: structuredWarnings,
      details: metadata,
    },
  };
}

function preprocessWarningObjects(metadata = {}) {
  const warnings = [
    ...(metadata.structured_warnings || []),
    ...(metadata.warnings || []),
  ];
  const residueCount = Number(metadata.residue_count);
  if (Number.isFinite(residueCount) && residueCount > 0 && !warnings.some((item) => warningText(item).includes("fewer than"))) {
    if (residueCount < 10) {
      warnings.push({
        code: "pocket_lt_10_residues",
        severity: "warning",
        title: "Pocket size",
        text: `Pocket has ${residueCount} complete residues. More than 10-20 residues is advised for stable generation/scoring.`,
        appliesTo: "pocket",
      });
    } else if (residueCount < 20) {
      warnings.push({
        code: "pocket_lt_20_residues",
        severity: "warning",
        title: "Pocket size",
        text: `Pocket has ${residueCount} complete residues. More than 20 residues is advised when possible.`,
        appliesTo: "pocket",
      });
    }
  }
  const seen = new Set();
  return warnings
    .map((warning) => normalizeWorkflowWarning(warning, { code: "pocket_warning", title: "Pocket size", appliesTo: "pocket" }))
    .filter((warning) => {
      const key = `${warning.code}:${warning.text}`;
      if (!warning.text || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function renderSetupPreprocessSummary() {
  const panel = $("#setup-staged-summary");
  if (!panel) return;
  const manifest = state.stagedPreprocessManifest;
  if (!manifest) {
    panel.hidden = true;
    $("#setup-staged-grid").innerHTML = "";
    $("#setup-staged-provenance").textContent = "";
    return;
  }
  const workflow = manifest.workflow || "preprocess";
  const staged = manifest.staged_inputs || {};
  const selected = manifest.selected_variant || {};
  const prep = manifest.preprocessing || {};
  $("#setup-staged-title").textContent = `Staged from ${workflowLabel(workflow)}`;
  $("#setup-staged-mode").textContent = preparedModeLabel(staged.mode);
  const rows = [
    ["View", selected.label || null],
    ["Source", manifest.set || manifest.source || null],
    ["PDB", staged.pdb || null],
    ["SDF", staged.sdf || "none"],
    ["Radius", selected.radius || prep.radius ? `${selected.radius || prep.radius} A` : null],
    ["Residues", prep.residue_count ? String(prep.residue_count) : null],
    ["Warnings", (prep.structured_warnings || prep.warnings || []).length ? (prep.structured_warnings || prep.warnings || []).map(warningText).join("; ") : null],
  ].filter(([, value]) => value !== null && value !== "");
  $("#setup-staged-grid").innerHTML = rows.map(([label, value]) => `
    <div><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>
  `).join("");
  $("#setup-staged-provenance").textContent = JSON.stringify(manifest, null, 2);
  panel.hidden = false;
}

function workflowLabel(workflow) {
  return String(workflow || "preprocess")
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function stagePreparedInputs({ mode, pdb, sdf, label, detail, preprocessManifest = null }) {
  const isShapeInput = mode === "shape";
  clearLoadedStudyForCustomInput();
  state.customPdb = pdb;
  state.customSdf = sdf;
  state.stagedPreprocessManifest = preprocessManifest;
  state.batchInputs = [];
  if (isShapeInput) setEngine("diffsmol");
  setMode(isShapeInput ? "reference" : mode, false);
  $("#pdb-name").textContent = pdb?.name || "No protein staged";
  $("#pdb-detail").textContent = pdb ? (detail || "Prepared input") : "DiffSMol ligand-shape input";
  $("#sdf-name").textContent = sdf?.name || "No reference SDF";
  const heavyAtoms = sdf?.text ? sdfHeavyAtomCount(sdf.text) : 0;
  $("#sdf-detail").textContent = isShapeInput
    ? `Prepared shape reference ligand${heavyAtoms ? ` · ${heavyAtoms} heavy atoms` : ""}`
    : sdf
      ? `Prepared reference ligand${heavyAtoms ? ` · ${heavyAtoms} heavy atoms` : ""}`
      : "Pocket mode uses the cropped PDB";
  $("#example-select").value = "custom";
  state.exampleId = "custom";
  updateCustomOptionLabel(label || pdb?.name || sdf?.name);
  updateBatchLabel();
  renderSetupPreprocessSummary();
  updateRunSummary();
  renderPreparedStructureTray();
  setActiveTab("setup");
  renderPreprocessStagingStatus();
  showToast(isShapeInput
    ? (pdb ? "Pocket and reference ligand staged for DiffSMol." : "Reference ligand staged for DiffSMol.")
    : mode === "reference" ? "Prepared protein and reference ligand staged." : "Prepared pocket staged.");
}

async function handleFolderUpload(event) {
  const files = [...event.target.files];
  if (!files.length) return;
  try {
    const grouped = await groupBatchFiles(files);
    state.batchInputs = grouped;
    state.stagedPreprocessManifest = null;
    renderSetupPreprocessSummary();
    $("#example-select").value = "custom";
    state.exampleId = "custom";
    updateCustomOptionLabel(grouped.length === 1 ? grouped[0].name : `${grouped.length} folders`);
    updateBatchLabel();
    updateRunSummary();
    showToast(`${grouped.length} folder${grouped.length === 1 ? "" : "s"} ready for batch submission.`);
  } catch (error) {
    showToast(error.message);
    state.batchInputs = [];
    updateBatchLabel();
  }
}

function clearBatchSelection() {
  state.batchInputs = [];
  $("#folder-input").value = "";
  updateBatchLabel();
  updateRunSummary();
  showToast("Batch selection cleared. You can choose another folder or upload one input.");
}

async function groupBatchFiles(files) {
  const byFolder = new Map();
  files.forEach((file) => {
    const relative = file.webkitRelativePath || file.name;
    const parts = relative.split("/");
    const folder = parts.length > 1 ? parts.slice(0, -1).join("/") : "Selected files";
    if (!byFolder.has(folder)) byFolder.set(folder, []);
    byFolder.get(folder).push(file);
  });
  const jobs = [];
  const skipped = [];
  for (const [folder, folderFiles] of byFolder) {
    const pdbFile = chooseStructureInputFile(folderFiles, ["protein", "pocket"]);
    const sdfFile = chooseInputFile(folderFiles, ".sdf", ["ligand", "reference", "ref"]);
    if (state.engine !== "diffsmol" && !pdbFile) {
      skipped.push(`${folder}: no PDB/CIF`);
      continue;
    }
    const pdbText = pdbFile ? await readValidatedTextFile(pdbFile, "pdb", false) : null;
    const sdfText = sdfFile ? await readValidatedTextFile(sdfFile, "sdf", false) : null;
    if (sdfText && countSdfMolecules(sdfText) !== 1) {
      skipped.push(`${folder}: SDF must contain one reference ligand, not a ligand database`);
      continue;
    }
    if (state.engine !== "diffsmol" && !pdbText) {
      skipped.push(`${folder}: invalid PDB/CIF`);
      continue;
    }
    if ((state.mode === "reference" || state.engine === "diffsmol") && !sdfText) {
      skipped.push(`${folder}: no valid SDF`);
      continue;
    }
    jobs.push({
      name: folder,
      pdb: pdbText ? { name: pdbFile.name, text: pdbText } : null,
      sdf: sdfText ? { name: sdfFile.name, text: sdfText } : null,
    });
  }
  if (!jobs.length) {
    throw new Error(state.engine === "diffsmol"
      ? "No valid batch folders found. Each DiffSMol folder needs a 3D SDF reference ligand."
      : state.mode === "reference"
      ? "No valid batch folders found. Each folder needs a PDB/CIF and SDF in reference mode."
      : "No valid batch folders found. Each folder needs a PDB/CIF.");
  }
  if (skipped.length) {
    showToast(`${jobs.length} folder${jobs.length === 1 ? "" : "s"} ready; ${skipped.length} skipped.`);
    console.warn("Skipped batch folders", skipped);
  }
  return jobs;
}

function chooseInputFile(files, extension, preferredTokens = []) {
  const candidates = files.filter((file) => file.name.toLowerCase().endsWith(extension));
  if (!candidates.length) return null;
  const preferred = candidates.find((file) => {
    const name = file.name.toLowerCase();
    return preferredTokens.some((token) => name.includes(token)) && !name.includes("generated");
  });
  return preferred || candidates.find((file) => !file.name.toLowerCase().includes("generated")) || candidates[0];
}

function chooseStructureInputFile(files, preferredTokens = []) {
  const candidates = files.filter((file) => isStructureFilename(file.name));
  if (!candidates.length) return null;
  const preferred = candidates.find((file) => {
    const name = file.name.toLowerCase();
    return preferredTokens.some((token) => name.includes(token)) && !name.includes("generated");
  });
  return preferred || candidates.find((file) => !file.name.toLowerCase().includes("generated")) || candidates[0];
}

async function readValidatedTextFile(file, kind, showError = true) {
  const text = await file.text();
  const lower = file.name.toLowerCase();
  const validExtension = kind === "pdb" ? isStructureFilename(lower) : lower.endsWith(".sdf");
  const validContent = kind === "pdb"
    ? looksLikeStructureText(text)
    : looksLikeSdfText(text);
  if (!validExtension || !validContent) {
    if (showError) showToast(`${file.name} does not look like a valid ${kind === "pdb" ? "PDB/CIF" : kind.toUpperCase()} file.`);
    return null;
  }
  return kind === "sdf" ? normalizeSdfText(text) : text;
}

function isStructureFilename(name) {
  const lower = String(name || "").toLowerCase();
  return lower.endsWith(".pdb") || lower.endsWith(".ent") || lower.endsWith(".cif") || lower.endsWith(".mmcif");
}

function looksLikeStructureText(text) {
  return text.split(/\r?\n/, 200).some((line) => {
    const trimmed = line.trim();
    return /^(ATOM  |HETATM|MODEL |HEADER|CRYST1)/.test(line)
      || trimmed.startsWith("data_")
      || trimmed.startsWith("_atom_site.")
      || trimmed === "loop_";
  });
}

function looksLikeSdfText(text) {
  const body = String(text || "");
  if (body.includes("$$$$")) return true;
  return /\bV(2000|3000)\b/.test(body) && /^\s*M\s+END\s*$/m.test(body);
}

function normalizeSdfText(text) {
  const body = String(text || "").replace(/\s+$/g, "");
  return body.includes("$$$$") ? `${body}\n` : `${body}\n$$$$\n`;
}

async function readLigandPanelFile(file) {
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".sdf")) {
    const text = await readValidatedTextFile(file, "sdf");
    return text ? { name: file.name, text, count: countSdfMolecules(text), source: "SDF panel" } : null;
  }
  if (lower.endsWith(".csv")) {
    const text = await file.text();
    const rows = parseCsvRows(text);
    if (!rows.length) throw new Error("CSV ligand panel is empty.");
    const headers = rows[0].map((item) => item.trim());
    const smilesIndex = headers.findIndex((header) => ["smiles", "smile", "canonical_smiles", "isomeric_smiles"].includes(header.toLowerCase()));
    if (smilesIndex < 0) throw new Error("CSV ligand panel must include a smiles column.");
    const nameIndex = headers.findIndex((header) => ["name", "id", "ligand", "compound", "compound_id"].includes(header.toLowerCase()));
    const ligands = rows.slice(1)
      .map((row, index) => ({
        smiles: (row[smilesIndex] || "").trim(),
        name: (nameIndex >= 0 ? row[nameIndex] : "")?.trim() || `ligand_${index + 1}`,
      }))
      .filter((item) => item.smiles);
    if (!ligands.length) throw new Error("CSV ligand panel has no SMILES values.");
    const sdf = await smilesRowsToSdf(ligands);
    return {
      name: file.name.replace(/\.csv$/i, ".sdf"),
      text: sdf.text,
      count: sdf.count,
      source: `${file.name} CSV`,
    };
  }
  showToast(`${file.name} must be an SDF or CSV ligand panel.`);
  return null;
}

function parseCsvRows(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (char === "\"" && next === "\"") {
        cell += "\"";
        i += 1;
      } else if (char === "\"") {
        quoted = false;
      } else {
        cell += char;
      }
    } else if (char === "\"") {
      quoted = true;
    } else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (char !== "\r") {
      cell += char;
    }
  }
  row.push(cell);
  rows.push(row);
  return rows.filter((items) => items.some((item) => item.trim()));
}

async function smilesRowsToSdf(ligands) {
  const RDKit = await waitForRDKit();
  const blocks = [];
  const failures = [];
  ligands.slice(0, 250).forEach((ligand, index) => {
    let mol = null;
    try {
      mol = RDKit.get_mol(ligand.smiles);
      if (!mol) throw new Error("RDKit could not parse SMILES.");
      const molblock = mol.get_molblock();
      if (!molblock || !molblock.includes("M  END")) throw new Error("RDKit could not create a mol block.");
      blocks.push(`${ligand.name || `ligand_${index + 1}`}\n${molblock.split(/\r?\n/).slice(1).join("\n")}\n>  <SMILES>\n${ligand.smiles}\n\n$$$$`);
    } catch (error) {
      failures.push(`${ligand.name || `ligand_${index + 1}`}: ${error.message}`);
    } finally {
      if (mol?.delete) mol.delete();
    }
  });
  if (!blocks.length) throw new Error(`No CSV SMILES could be converted to SDF. ${failures[0] || ""}`.trim());
  if (failures.length) showToast(`${failures.length} CSV ligand${failures.length === 1 ? "" : "s"} could not be converted.`);
  return { text: `${blocks.join("\n")}\n`, count: blocks.length };
}

async function waitForRDKit() {
  if (typeof window.initRDKitModule !== "function") {
    throw new Error("RDKit is still loading. Try the CSV panel again in a moment.");
  }
  if (!window._conditarRDKitPromise) {
    window._conditarRDKitPromise = window.initRDKitModule();
  }
  return window._conditarRDKitPromise;
}

function countSdfMolecules(text) {
  return text.split("$$$$").filter((block) => block.trim()).length;
}

function sdfHeavyAtomCount(text) {
  const block = String(text || "").split("$$$$")[0] || "";
  const lines = block.split(/\r?\n/);
  const counts = lines[3] || "";
  const atomCount = Number.parseInt(counts.slice(0, 3).trim(), 10);
  if (!Number.isFinite(atomCount) || atomCount <= 0) return 0;
  let heavyAtoms = 0;
  for (const line of lines.slice(4, 4 + atomCount)) {
    const symbol = line.slice(31, 34).trim() || line.trim().split(/\s+/)[3] || "";
    if (symbol && symbol.toUpperCase() !== "H") heavyAtoms += 1;
  }
  return heavyAtoms;
}

function updateBatchLabel() {
  const count = state.batchInputs.length;
  const target = resolvedTarget();
  const isSlurmGpu = isSlurmGpuTarget(target);
  const isOpenShiftJob = isOpenShiftJobTarget(target);
  const isOpenShiftMock = isOpenShiftMockTarget(target);
  const cpuBatchWarning = count > 10
    ? "Large local CPU batches can take many hours. Keep this server window open, or use Slurm GPU for durable parallel queueing."
    : "Local CPU batches run one at a time. Keep this server window open until the queued jobs finish.";
  const openshiftSubmit = openShiftSubmissionEnabled();
  const batchKind = isSlurmGpu ? "independent Slurm" : isOpenShiftJob ? (openshiftSubmit ? "OpenShift" : "OpenShift manifest") : isOpenShiftMock ? "OpenShift diagnostic" : "serial queued CPU";
  const batchTitle = isSlurmGpu ? "Parallel GPU batch" : isOpenShiftJob ? (openshiftSubmit ? "OpenShift Job batch" : "OpenShift manifest batch") : isOpenShiftMock ? "OpenShift diagnostics batch" : "Queued CPU batch";
  const batchMessage = isSlurmGpu
    ? "Slurm will process folders concurrently when capacity is available."
    : isOpenShiftJob
      ? openshiftSubmit
        ? "OpenShift will launch one generator Job per folder when capacity is available."
        : "Manifest mode writes Kubernetes Job artifacts for inspection; no cluster submission is attempted."
      : isOpenShiftMock
      ? "Diagnostics jobs create mock outputs so OpenShift routing, storage, logs, and results can be checked."
      : cpuBatchWarning;
  $("#folder-name").textContent = count ? `${count} batch folder${count === 1 ? "" : "s"}` : "Batch folders";
  $("#folder-detail").textContent = count
    ? `Generate will submit ${count} ${batchKind} job${count === 1 ? "" : "s"}`
    : state.engine === "diffsmol" ? "Optional: one SDF per folder" : "Optional: one PDB and optional SDF per folder";
  $("#batch-mode-banner").hidden = !count;
  $("#batch-mode-banner").classList.toggle("is-warning", Boolean(count && !isSlurmGpu && !isOpenShiftJob && !isOpenShiftMock));
  $("#batch-mode-title").textContent = batchTitle;
  $("#batch-mode-message").textContent = count
    ? `${count} folder${count === 1 ? "" : "s"} ready. ${batchMessage} Each folder is processed as its own job; inputs are never mixed.`
    : "Each selected folder will submit as a separate job.";
  $("#preview-run span").textContent = count
    ? `Submit ${count} batch job${count === 1 ? "" : "s"}`
    : state.engine === "diffsmol" ? "Generate shape analogs" : "Generate molecules";
  updateRunEstimate();
}

function updateCustomOptionLabel(label) {
  const option = $("#example-select option[value='custom']");
  option.textContent = label ? `Custom · ${label}` : "Custom upload";
}

function downloadSelected() {
  if (!state.selected) return;
  downloadBlob(state.selected.text, state.selected.name, "chemical/x-mdl-sdfile");
}

function downloadCsv() {
  if (!state.study) return;
  downloadBlob(csvText(exportCandidates()), `${studyName()}_metrics.csv`, "text/csv");
}

function downloadConfig() {
  downloadBlob(JSON.stringify(buildConfiguration(), null, 2), `${studyName()}_config.json`, "application/json");
}

async function downloadAll() {
  if (!state.study) return;
  if (!window.JSZip) {
    showToast("ZIP support could not load. Download the CSV and selected SDF individually.");
    return;
  }
  const button = $("#download-all");
  const candidates = exportCandidates();
  if (!candidates.length) {
    showToast("No candidates match the current export filters. Reset or loosen filters before downloading.", 8000);
    return;
  }
  button.disabled = true;
  button.textContent = "Packaging...";
  const exportMetadata = buildExportMetadata(candidates);
  let archiveNotice = "The archive will be saved by your browser to its Downloads folder.";
  const job = loadedResultsJob();
  if (job?.id) {
    try {
      const saved = await service.exportJob(job.id, {
        selected_paths: candidates.map((item) => item.path).filter(Boolean),
        filters: exportMetadata.filters,
        tool_runs: exportMetadata.tool_runs,
        run_config: exportMetadata.run_config,
        metrics_csv: csvText(candidates),
      });
      archiveNotice = `A filtered server copy was saved to ${saved.relative_directory || saved.relative_path}. The browser copy will be saved to its Downloads folder.`;
    } catch (error) {
      archiveNotice = `Browser copy will be saved to Downloads. Server archive was not created: ${error.message}`;
    }
  }
  const zip = new window.JSZip();
  const structures = zip.folder("generated_structures");
  candidates.forEach((item) => {
    const path = job && item.path?.startsWith("outputs/") ? item.path.slice("outputs/".length) : item.name;
    structures.file(path, item.text);
  });
  zip.file("metrics.csv", csvText(candidates));
  zip.file("run_config.json", JSON.stringify(exportMetadata.run_config, null, 2));
  zip.file("export_metadata.json", JSON.stringify(exportMetadata, null, 2));
  zip.file("run_manifest.json", JSON.stringify(buildRunManifest(candidates, exportMetadata), null, 2));
  if (state.study.logs?.stdout) zip.file("logs/stdout.log", state.study.logs.stdout);
  if (state.study.logs?.stderr) zip.file("logs/stderr.log", state.study.logs.stderr);
  if (state.study.logs?.extra) zip.file("logs/additional_logs.txt", state.study.logs.extra);
  if (state.study.summary) zip.file("job_summary.json", JSON.stringify(state.study.summary, null, 2));
  if (state.study.pdbText) zip.file(filenameOnly(state.study.example.pdb || "input.pdb"), state.study.pdbText);
  if (state.study.referenceSdf) zip.file(filenameOnly(state.study.example.sdf || "reference.sdf"), state.study.referenceSdf);
  const blob = await zip.generateAsync({ type: "blob" });
  downloadBlob(blob, `${studyName()}_study.zip`, "application/zip");
  showToast(archiveNotice, 10000);
  button.disabled = false;
  button.innerHTML = "Download all <b>↓</b>";
  updateExportScope();
}

function loadedResultsJob() {
  return state.resultSource === "job" ? state.study?.loadedJob || null : null;
}

function buildConfiguration() {
  const job = loadedResultsJob();
  const mode = job?.mode || state.mode;
  const pdbInput = setupPdbInput();
  const sdfInput = setupSdfInput();
  return {
    interface_version: "0.1.0",
    backend_connected: true,
    job_id: job?.id || null,
    conditioning_mode: mode,
    inputs: {
      pdb_filename: job ? (job.inputs?.pdb ? filenameOnly(job.inputs.pdb) : null) : pdbInput?.name || null,
      sdf_filename: mode === "reference"
        ? (job ? (job.inputs?.sdf ? filenameOnly(job.inputs.sdf) : null) : sdfInput?.name || null)
        : null,
    },
    parameters: { ...(job?.parameters || state.parameters) },
    export: {
      selected_count: exportCandidates().length,
      total_count: state.study?.candidates?.length || 0,
      filters: activeExportFilters(),
    },
  };
}

function buildExportMetadata(candidates = exportCandidates()) {
  return {
    created_at: new Date().toISOString(),
    job_id: loadedResultsJob()?.id || null,
    selected_count: candidates.length,
    total_count: state.study?.candidates?.length || 0,
    selected_candidates: candidates.map((item) => ({ id: item.id, name: item.name, path: item.path || item.name })),
    filters: activeExportFilters(),
    tool_runs: state.study?.toolRuns || loadedResultsJob()?.tool_runs || [],
    run_config: buildConfiguration(),
  };
}

function buildRunManifest(candidates = exportCandidates(), exportMetadata = buildExportMetadata(candidates)) {
  const job = loadedResultsJob();
  const inputPdb = job?.inputs?.pdb ? filenameOnly(job.inputs.pdb) : state.study?.example?.pdb || null;
  const inputSdf = job?.inputs?.sdf ? filenameOnly(job.inputs.sdf) : state.study?.example?.sdf || null;
  return {
    schema_version: 1,
    created_at: new Date().toISOString(),
    job: job ? {
      id: job.id || null,
      engine: job.engine || "conditar",
      mode: job.mode || null,
      target: job.target || null,
      status: job.status || null,
      created_at: job.created_at || null,
      started_at: job.started_at || null,
      finished_at: job.finished_at || null,
    } : null,
    inputs: {
      pdb: inputPdb,
      sdf: inputSdf,
      preprocess_metadata: job?.inputs?.preprocess_metadata ? filenameOnly(job.inputs.preprocess_metadata) : null,
    },
    preprocess: job?.preprocess || null,
    parameters: job?.parameters || state.parameters || {},
    postprocess: job?.postprocess || {},
    container: job?.container || {},
    command: job?.command || [],
    outputs: {
      generated_sdfs: candidates.map((item) => ({
        id: item.id,
        name: item.name,
        relative_path: item.path || item.name,
      })),
      artifacts: state.study?.artifacts || [],
    },
    export: {
      selected_count: candidates.length,
      total_count: state.study?.candidates?.length || 0,
      filters: exportMetadata.filters || [],
      metadata: exportMetadata,
    },
  };
}

function csvText(candidates = filteredCandidates()) {
  const toolOutputs = toolOutputDefinitions();
  const header = [
    "candidate",
    "source_file",
    "smiles",
    "formula",
    "molecular_weight",
    "atom_count",
    "heavy_atoms",
    "hetero_atoms",
    "ring_estimate",
    "vina_score_only",
    "vina_minimize",
    "vina_dock",
    "qvina",
    ...toolOutputs.map((output) => output.name.toLowerCase()),
  ];
  const rows = candidates.map((item) => [
    item.id,
    item.name,
    item.smiles || item.properties?.SMILES || "",
    item.formula,
    item.molecularWeight,
    item.atomCount,
    item.heavyAtoms,
    item.heteroAtoms,
    item.rings,
    item.properties?.VINA_SCORE_ONLY || "",
    item.properties?.VINA_MINIMIZE || "",
    item.properties?.VINA_DOCK || "",
    item.properties?.QVINA || "",
    ...toolOutputs.map((output) => item.properties?.[output.name] || ""),
  ]);
  return [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
}

function exportCandidates() {
  if (!$("#export-filtered")?.checked) return state.study?.candidates || [];
  return (state.study?.candidates || []).filter((item) => state.exportSelection.has(item.id));
}

function updateExportScope() {
  const total = state.study?.candidates?.length || 0;
  const count = exportCandidates().length;
  const filtered = $("#export-filtered")?.checked;
  if ($("#export-scope-count")) $("#export-scope-count").textContent = `(${count} of ${total} candidates)`;
  if ($("#download-all")) $("#download-all").firstChild.textContent = filtered ? "Download filtered " : "Download all ";
  renderExportManifestSummary(count, total);
  updateExportFilterStatus();
}

function renderExportManifestSummary(selectedCount = exportCandidates().length, total = state.study?.candidates?.length || 0) {
  const container = $("#export-manifest-summary");
  if (!container) return;
  const job = loadedResultsJob();
  const rows = state.study ? [
    ["Manifest", "run_manifest.json"],
    ["Engine", job ? engineLabel(job) : state.engine === "diffsmol" ? "DiffSMol" : "conDitar"],
    ["Selected", `${selectedCount}/${total} candidates`],
    ["Includes", "inputs, command, container, logs"],
  ] : [];
  container.hidden = !rows.length;
  container.innerHTML = rows.map(([label, value]) => provenanceItem(label, value)).join("");
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function studyName() {
  return state.study?.example?.id || "conditar";
}

function filenameOnly(path) {
  return String(path || "").split("/").pop() || "conditar_input";
}

function filenameStem(path) {
  return filenameOnly(path).replace(/\.[^.]+$/, "") || "conditar_input";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[char]));
}

async function downloadPreparedArchive(filename, files) {
  const validFiles = files.filter((file) => file?.text);
  if (!validFiles.length) {
    showToast("No prepared files are available to download.");
    return;
  }
  if (!window.JSZip) {
    validFiles.forEach((file) => downloadBlob(file.text, file.name, contentTypeForFilename(file.name)));
    return;
  }
  const zip = new window.JSZip();
  validFiles.forEach((file) => zip.file(filenameOnly(file.name), file.text));
  const blob = await zip.generateAsync({ type: "blob" });
  downloadBlob(blob, filename, "application/zip");
}

function contentTypeForFilename(filename) {
  const lower = String(filename || "").toLowerCase();
  if (lower.endsWith(".pdb") || lower.endsWith(".ent")) return "chemical/x-pdb";
  if (lower.endsWith(".sdf")) return "chemical/x-mdl-sdfile";
  if (lower.endsWith(".json")) return "application/json";
  return "text/plain";
}

function downloadBlob(content, filename, type) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setLoading(loading) {
  $("#viewer-loading").hidden = !loading;
  if (loading) $("#hero-status").textContent = "Loading";
}

function showToast(message, duration = 3400) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("show");
  void toast.offsetWidth;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), duration);
}

function formatBytes(bytes) {
  return bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`;
}

function debounce(fn, wait) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), wait);
  };
}

function initializeTheme() {
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  setTheme(savedTheme === "dark" ? "dark" : "light");
}

function toggleTheme() {
  setTheme(document.body.classList.contains("dark-mode") ? "light" : "dark", { persist: true });
}

function setTheme(theme, options = {}) {
  const isDark = theme === "dark";
  document.body.classList.toggle("dark-mode", isDark);
  const toggle = $("#theme-toggle");
  if (toggle) {
    toggle.textContent = isDark ? "☼" : "◐";
    toggle.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
    toggle.title = isDark ? "Switch to light mode" : "Switch to dark mode";
  }
  if (options.persist) localStorage.setItem(THEME_STORAGE_KEY, isDark ? "dark" : "light");
}

initialize();
