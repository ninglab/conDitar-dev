import { ADVANCED_PARAMETERS, EXAMPLES, PARAMETERS } from "./config.js?v=20260723-theme-1";
import { drawCategoryChart, drawHistogram } from "./charts.js?v=20260723-theme-1";
import { ExampleDataService } from "./data-service.js?v=20260723-theme-1";
import { vinaWasRun } from "./sdf.js?v=20260723-theme-1";
import { render2D, render3D } from "./viewers.js?v=20260723-theme-1";

const service = new ExampleDataService();
const ACTIVE_JOB_STATUSES = new Set(["queued", "running"]);
const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "canceled"]);
const CLEANUP_JOB_STATUSES = new Set(["failed", "canceled"]);
const SLURM_GPU_TARGET = "slurm_gpu";
const LEGACY_SLURM_GPU_TARGET = "osc_gpu";
const MAX_CATEGORICAL_FILTER_VALUES = 24;
const THEME_STORAGE_KEY = "conditar-theme";

const state = {
  study: null,
  selected: null,
  exampleId: "custom",
  mode: "reference",
  view: "3d",
  parameters: Object.fromEntries([...PARAMETERS, ...ADVANCED_PARAMETERS].map((item) => [item.key, item.value])),
  customPdb: null,
  customSdf: null,
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
  const commonKeys = new Set(["num_samples", "pocket_radius"]);
  renderParameterFields(PARAMETERS.filter((parameter) => commonKeys.has(parameter.key)), $("#parameter-fields"));
  renderParameterFields(
    [...PARAMETERS.filter((parameter) => !commonKeys.has(parameter.key)), ...ADVANCED_PARAMETERS],
    $("#advanced-fields"),
  );
  bindEvents();
  setMode("reference", false);
  updateInputLabels(null);
  updateCommand();
  refreshJobs(false);
  setActiveTab("setup");
}

function renderParameterFields(parameters, container) {
  container.innerHTML = parameters.map((parameter) => {
    const control = parameter.type === "select"
      ? `<select id="param-${parameter.key}">${parameter.options.map((option) => `<option ${option === parameter.value ? "selected" : ""}>${option}</option>`).join("")}</select>`
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
  $$(".mode-toggle button").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $$(".view-toggle button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $$(".workflow-step").forEach((button) => button.addEventListener("click", () => setActiveTab(button.dataset.section)));
  [...PARAMETERS, ...ADVANCED_PARAMETERS].forEach((parameter) => {
    $(`#param-${parameter.key}`).addEventListener("input", (event) => {
      state.parameters[parameter.key] = parameter.type === "number" ? Number(event.target.value) : event.target.value;
      updateCommand();
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
  $$(".builtin-evaluation-toggle").forEach((input) => input.addEventListener("change", updateVinaControls));
  ["#vina-exhaustiveness", "#vina-cpu"].forEach((selector) => {
    $(selector).addEventListener("input", updateCommand);
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
  $("#protein-style").addEventListener("change", renderSelectedStructure);
  $("#ligand-style").addEventListener("change", renderSelectedStructure);
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
      $("#job-target").value = slurmAvailable ? SLURM_GPU_TARGET : "local_cpu";
      updateJobTargetControls();
    }
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
  const slurmAvailable = Boolean(health.slurm?.sbatch);
  const slurm = slurmAvailable ? "sbatch available" : "sbatch not found";
  const imageReady = Boolean(health.container_image?.exists);
  const isSlurmGpu = isSlurmGpuTarget(resolvedTarget());
  status.textContent = isSlurmGpu ? (slurmAvailable ? "Slurm GPU available" : "Slurm setup needs attention") : imageReady ? "Local CPU available" : "Local setup needs attention";
  detail.textContent = isSlurmGpu
    ? `${slurm}; ${imageReady ? "container image found" : "container image not confirmed"}. Selected target: Slurm GPU.`
    : `${imageReady ? "container image found" : "container image missing"}. Selected target: Local CPU.`;
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
  const isSlurmGpu = isSlurmGpuTarget(resolvedTarget());
  status.textContent = failing ? "Setup needs attention" : warnings ? "Ready with optional warnings" : "Ready to run";
  detail.textContent = failing
    ? `Fix the missing required items before submitting a ${isSlurmGpu ? "Slurm GPU" : "local CPU"} job.`
    : warnings
      ? `${isSlurmGpu ? "Slurm GPU" : "Local CPU"} runs can work; optional Tool Chest items may need setup.`
      : `${isSlurmGpu ? "Slurm GPU" : "Local CPU"} launch requirements look ready.`;
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
  const checks = (health?.checks || []).map((check) => ({ ...check }));
  if (!isSlurmGpu) {
    return checks.filter((check) => check.id !== "slurm");
  }
  return checks.map((check) => {
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
}

async function loadExample(exampleId) {
  setLoading(true);
  state.exampleId = exampleId;
  state.customPdb = null;
  state.customSdf = null;
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
  state.mode = mode;
  $$(".mode-toggle button").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $("#sdf-dropzone").hidden = mode === "pocket";
  const pocketRadiusField = $("[data-parameter-key='pocket_radius']");
  if (pocketRadiusField) pocketRadiusField.hidden = mode === "pocket";
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
  updateCommand();
}

function updateInputLabels(example) {
  $("#pdb-name").textContent = example ? example.pdb.split("/").pop() : "Choose a PDB file";
  $("#pdb-detail").textContent = example ? `${example.pdbRecords} · bundled input` : "Required · uploaded with this job";
  $("#sdf-name").textContent = example?.sdf ? example.sdf.split("/").pop() : "Choose a reference SDF";
  $("#sdf-detail").textContent = example?.sdf ? "Reference ligand · bundled input" : "Required for protein + ligand mode";
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
  updateResultsSource();
  updateCommand();
}

async function submitGenerationJob() {
  if (!state.batchInputs.length && !state.study && !state.customPdb) {
    showToast("Load or upload a PDB before submitting a job.");
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
      if (item.target === "local_cpu" && !isTerminalJob(item)) state.watchedJobs.add(item.id);
    });
    const job = response.jobs[0];
    state.currentJob = job;
    state.selectedJob = job;
    const failedJobs = response.jobs.filter((item) => item.status === "failed");
    const queuedJobs = response.jobs.length - failedJobs.length;
    const message = failedJobs.length
      ? `${queuedJobs} queued, ${failedJobs.length} failed. ${failedJobs[0].error_message || "See the selected job logs."}`
      : response.jobs.length > 1
        ? `${response.jobs.length} ${isSlurmGpuTarget(job?.target) ? "parallel GPU tasks" : "CPU jobs queued"}${response.errors.length ? `, ${response.errors.length} skipped: ${response.errors[0].error}` : ""}.`
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

function buildJobPayload(inputOverride = null) {
  const example = EXAMPLES[state.exampleId];
  const pdb = inputOverride?.pdb || state.customPdb || (state.study?.pdbText ? {
    name: example?.pdb.split("/").pop() || "input.pdb",
    text: state.study.pdbText,
  } : null);
  const sdf = state.mode === "reference"
    ? (inputOverride?.sdf || state.customSdf || (state.study?.referenceSdf ? {
      name: example?.sdf?.split("/").pop() || "reference.sdf",
      text: state.study.referenceSdf,
    } : null))
    : null;
  return {
    target: resolvedTarget(),
    mode: state.mode,
    example_id: state.exampleId,
    input_name: inputOverride?.name || pdb?.name || state.exampleId,
    email: $("#job-email").disabled ? "" : $("#job-email").value.trim(),
    pdb,
    sdf,
    slurm: buildSlurmPayload(),
    postprocess: buildPostprocessPayload(),
    tools: buildEvaluationToolsPayload(),
    parameters: {
      ...state.parameters,
      device: isSlurmGpuTarget(resolvedTarget()) ? "cuda:0" : "cpu",
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

function buildPostprocessPayload() {
  const selected = selectedBuiltinEvaluations();
  return {
    vina: selected.length > 0,
    vina_mode: selectedVinaMode(selected),
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

function selectedBuiltinEvaluations() {
  return $$(".builtin-evaluation-toggle:checked").map((input) => input.value);
}

function selectedVinaMode(selected = selectedBuiltinEvaluations()) {
  const chosen = new Set(selected);
  const wantsScore = chosen.has("vina_score");
  const wantsDock = chosen.has("vina_dock");
  const wantsQvina = chosen.has("qvina");
  if ((wantsScore || wantsDock) && wantsQvina) return "all";
  if (wantsDock) return "vina_dock";
  if (wantsQvina) return "qvina";
  if (wantsScore) return "vina_score";
  return "none";
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
    if (job.status === "completed") {
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
  if (!candidates.length) {
    state.selectedJob = result.job || job;
    updateJobDetail(state.selectedJob, resultLogText || "No SDF files were found in the job output directory.", result.logs || {});
    showToast(state.selectedJob?.error_message || "No SDF results were found for this job.");
    setActiveTab("jobs");
    return;
  }
  const vinaFailures = candidates.filter((item) => String(item.properties?.VINA_STATUS || "").toLowerCase() === "failed");
  const fallbackExample = state.study?.example || EXAMPLES[state.exampleId] || {};
  const pdbInput = result.inputs?.pdb || null;
  const sdfInput = result.inputs?.sdf || null;
  state.study = {
    ...state.study,
    example: {
      ...fallbackExample,
      id: job.id,
      label: job.id,
      pdb: pdbInput?.name || fallbackExample.pdb || "input.pdb",
      sdf: sdfInput?.name || fallbackExample.sdf || null,
    },
    pdbText: pdbInput?.text || state.study?.pdbText || "",
    referenceSdf: sdfInput?.text || state.study?.referenceSdf || null,
    candidates,
    artifacts: result.artifacts || [],
    logs: result.logs || {},
    summary: result.summary || {},
    toolRuns: result.toolRuns || [],
  };
  state.currentJob = job;
  state.selectedJob = job;
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
      <td>${escapeHtml(shortJobId(job.id))}<br><small title="${escapeHtml(job.id)}">${escapeHtml(job.mode || "run")}</small></td>
      <td><span class="status-badge" data-status="${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>${job.status_note ? `<br><small>${escapeHtml(job.status_note)}</small>` : ""}</td>
      <td>${escapeHtml(targetLabel(job))}<br><small>${escapeHtml(inputLabel(job))}</small></td>
      <td>${formatDate(job.created_at)}<br><small>${escapeHtml(slurmLabel(job))}</small></td>
      <td>
        ${job.status === "completed" ? `<button class="secondary-button compact-action load-job-results">Results</button>` : ""}
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
    if (job.target === "local_cpu") state.watchedJobs.add(job.id);
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
  if (!saved.inputs?.pdb?.text) throw new Error("Original PDB input was not found for rerun.");
  const payload = {
    target: original.target || "local_cpu",
    mode: original.mode || (saved.inputs.sdf ? "reference" : "pocket"),
    example_id: original.example_id || "custom",
    input_name: `rerun_${original.input_name || saved.inputs.pdb.name || jobId}`,
    email: original.email || "",
    pdb: { name: saved.inputs.pdb.name, text: saved.inputs.pdb.text },
    sdf: saved.inputs.sdf?.text ? { name: saved.inputs.sdf.name, text: saved.inputs.sdf.text } : null,
    slurm: original.slurm || {},
    postprocess: original.postprocess || {},
    parameters: original.parameters || {},
  };
  return service.submitJob(payload);
}

async function loadSelectedJobResults(jobId) {
  const job = await service.getJob(jobId);
  state.selectedJob = job;
  updateJobDetail(job, "Loading results...");
  if (job.status !== "completed") {
    showToast("Only completed jobs have results to load.");
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
  if (!jobs.some((job) => job.target === "local_cpu")) return;
  if (!("Notification" in window) || Notification.permission !== "default") return;
  try {
    await Notification.requestPermission();
  } catch {
    // Browser notifications are optional; toast updates still work.
  }
}

function notifyJobTerminal(job, status) {
  if (!job?.id || state.notifiedJobs.has(job.id)) return;
  if (job.target !== "local_cpu") return;
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
  const inputs = Math.max(1, state.batchInputs.length || (state.customPdb || state.study ? 1 : 0));
  const samples = Math.max(1, Number(state.parameters.num_samples) || 1);
  const totalSamples = inputs * samples;
  const isGpu = isSlurmGpuTarget(resolvedTarget());
  if (!inputs) {
    estimate.textContent = "Estimate updates after you choose inputs.";
    return;
  }
  const minutesPerSample = isGpu ? 1.5 : 5.5;
  const concurrencyNote = isGpu
    ? "Slurm GPU jobs can run in parallel once scheduled."
    : "Local CPU jobs run serially; keep this server window open.";
  estimate.textContent = `Rule-of-thumb runtime: about ${formatDuration(totalSamples * minutesPerSample)} for ${inputs} input${inputs === 1 ? "" : "s"} × ${samples} sample${samples === 1 ? "" : "s"} on ${isGpu ? "Slurm GPU" : "local CPU"}. ${concurrencyNote}`;
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
  const job = state.selectedJob || state.currentJob;
  const isCompletedJob = state.resultSource === "job" && job?.status === "completed";
  const runs = state.study?.toolRuns || job?.tool_runs || [];
  status.textContent = !state.toolsLoaded
    ? "Tools unavailable"
    : isCompletedJob
      ? `${tools.length} tool${tools.length === 1 ? "" : "s"} available`
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
  const job = state.selectedJob || state.currentJob;
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
    .map((output) => {
      const annotated = candidates.filter((item) => item.properties?.[output.name]);
      const passing = annotated.filter((item) => isTruthyProperty(item.properties?.[output.name])).length;
      return [output.label, annotated.length ? `${passing}/${annotated.length}` : "Not run"];
    });
}

function toolPropertyBadges(item) {
  const values = toolOutputDefinitions()
    .map((output) => ({ output, value: item.properties?.[output.name] }))
    .filter((entry) => entry.value);
  if (!values.length) return "-";
  return values.map(({ output, value }) => {
    const text = toolPropertyDisplay(value);
    if (output.type === "boolean") {
      const passed = isTruthyProperty(value);
      return `<span class="status-badge compact-badge" data-status="${passed ? "completed" : "failed"}" title="${escapeHtml(output.label)}">${escapeHtml(output.label)}: ${passed ? "Pass" : "Fail"}</span>`;
    }
    return `<span class="tool-property-chip" title="${escapeHtml(text)}">${escapeHtml(output.label)}: ${escapeHtml(text)}</span>`;
  }).join(" ");
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
  list.innerHTML = definitions.map((metric) => exportFilterControl(metric)).join("");
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
  return `${metric.label} = ${metricValueLabel(metric, filter.value)}`;
}

function exportFilterHelp(metric, filter) {
  if (filter.enabled) return exportFilterText(metric, filter);
  if (metric.type === "number") return `Enable to filter exported molecules by ${metric.label}.`;
  return `Enable to require a ${metric.label} value for exported molecules.`;
}

function updateExportFilterStatus() {
  const status = $("#export-filter-status");
  if (!status) return;
  const activeCount = Object.values(state.exportFilters).filter((filter) => filter.enabled).length;
  const selectedCount = state.exportSelection?.size || 0;
  const total = state.study?.candidates?.length || 0;
  status.textContent = activeCount ? `${activeCount} active · ${selectedCount}/${total} selected` : total ? `No filters · ${selectedCount}/${total} selected` : "No job loaded";
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
    return String(filter.value) === "true" ? isTruthyProperty(value) : isFalseyProperty(value);
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
  if (metric.field) return Number(item[metric.field]);
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
  select.innerHTML = metrics.map((metric) => `<option value="${escapeHtml(metric.id)}">${escapeHtml(metric.label)}</option>`).join("");
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
  $("#selected-name").textContent = molecule.id;
  const metrics = [
    ["Formula", molecule.formula],
    ["MW", `${molecule.molecularWeight} Da`],
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
  toolOutputDefinitions().forEach((output) => {
    const value = molecule.properties?.[output.name];
    if (value) metrics.push([output.label || propertyLabel(output.name), toolPropertyDisplay(value)]);
  });
  $("#selected-metrics").innerHTML = metrics.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  render2D($("#viewer-2d"), molecule);
  render3D($("#viewer-3d"), molecule, state.study.pdbText, {
    proteinStyle: $("#protein-style").value,
    ligandStyle: $("#ligand-style").value,
  });
  $("#viewer-loading").hidden = true;
}

function setView(view) {
  state.view = view;
  $$(".view-toggle button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $("#viewer-3d").hidden = view !== "3d";
  $("#viewer-2d").hidden = view !== "2d";
}

function updateResultsSource() {
  if (state.resultSource === "job" && state.selectedJob) {
    const count = state.study?.summary?.sdf_count ?? state.study?.candidates?.length ?? 0;
    $("#results-source").textContent = `Loaded ${count} generated SDF${count === 1 ? "" : "s"} from job ${state.selectedJob.id}.`;
    return;
  }
  $("#results-source").textContent = "Upload input structures and submit a job to review generated outputs here.";
}

function targetLabel(job) {
  if (!job) return "Local CPU";
  if (job.target === "local_cpu") return "Local CPU";
  if (isSlurmGpuTarget(job.target)) return "Slurm GPU";
  return job.target || "Local CPU";
}

function isSlurmGpuTarget(target) {
  return target === SLURM_GPU_TARGET || target === LEGACY_SLURM_GPU_TARGET;
}

function isActiveJob(job) {
  return ACTIVE_JOB_STATUSES.has(job?.status);
}

function isTerminalJob(job) {
  return TERMINAL_JOB_STATUSES.has(job?.status);
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
  $("#slurm-controls").hidden = !isSlurmGpu;
  $("#job-runtime-label").textContent = isSlurmGpu ? "Slurm GPU" : "Local CPU";
  const emailInput = $("#job-email");
  const emailNote = $("#email-note");
  emailInput.disabled = !isSlurmGpu;
  emailInput.closest(".job-controls").classList.toggle("is-disabled", !isSlurmGpu);
  if (!isSlurmGpu) {
    emailInput.value = "";
    emailNote.textContent = "Local CPU runs use browser/system notifications when this page is allowed to notify you.";
  } else {
    emailNote.textContent = "Slurm can send completion/failure notifications when an email is provided.";
  }
  state.parameters.device = isSlurmGpu ? "auto" : "cpu";
  updateBatchLabel();
  updateCommand();
}

function updateVinaControls() {
  const selected = selectedBuiltinEvaluations();
  const enabled = selected.length > 0;
  $("#vina-options").classList.toggle("is-disabled", !enabled);
  $("#vina-exhaustiveness").disabled = !enabled;
  $("#vina-cpu").disabled = !enabled;
  $("#vina-mode-summary").textContent = selectedEvaluationLabel(selected);
  updateCommand();
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
  return items.map((item) => Number(item[key])).filter(Number.isFinite);
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

function updateCommand() {
  const pdbName = state.batchInputs.length
    ? `${state.batchInputs.length} folders`
    : (state.customPdb?.name || EXAMPLES[state.exampleId]?.pdb || "<choose PDB>");
  const sdfName = state.customSdf?.name || EXAMPLES[state.exampleId]?.sdf;
  const args = [
    "conditar-sample",
    `--device ${isSlurmGpuTarget(resolvedTarget()) ? "cuda:0" : "cpu"}`,
    `--num_samples ${state.parameters.num_samples}`,
    `--batch_size ${state.parameters.batch_size}`,
    `--pdb_filename ${pdbName}`,
  ];
  if (state.mode !== "pocket") args.splice(4, 0, `--pocket_radius ${state.parameters.pocket_radius}`);
  if (state.mode === "reference" && sdfName) args.push(`--sdf_filename ${sdfName}`);
  const selected = selectedBuiltinEvaluations();
  if (selected.length) {
    args.push(`--vina_score --vina_mode ${selectedVinaMode(selected)} --vina_exhaustiveness ${$("#vina-exhaustiveness").value}`);
  }
  $("#command-preview").textContent = args.join(" ");
  updateRunEstimate();
}

function resetParameters() {
  [...PARAMETERS, ...ADVANCED_PARAMETERS].forEach((parameter) => {
    state.parameters[parameter.key] = parameter.value;
    $(`#param-${parameter.key}`).value = parameter.value;
  });
  updateCommand();
  showToast("Sampling defaults restored.");
}

async function handlePdbUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await readValidatedTextFile(file, "pdb");
  if (!text) return;
  state.customPdb = { name: file.name, text };
  state.batchInputs = [];
  updateBatchLabel();
  $("#pdb-name").textContent = file.name;
  $("#pdb-detail").textContent = `${formatBytes(file.size)} · local upload`;
  $("#example-select").value = "custom";
  state.exampleId = "custom";
  updateCustomOptionLabel(file.name);
  if (state.study) {
    state.study.pdbText = state.customPdb.text;
    renderSelectedStructure();
  }
}

async function handleSdfUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await readValidatedTextFile(file, "sdf");
  if (!text) return;
  state.customSdf = { name: file.name, text };
  state.batchInputs = [];
  updateBatchLabel();
  $("#sdf-name").textContent = file.name;
  $("#sdf-detail").textContent = `${formatBytes(file.size)} · local upload`;
  $("#example-select").value = "custom";
  state.exampleId = "custom";
  updateCustomOptionLabel(state.customPdb?.name || file.name);
}

async function handleFolderUpload(event) {
  const files = [...event.target.files];
  if (!files.length) return;
  try {
    const grouped = await groupBatchFiles(files);
    state.batchInputs = grouped;
    $("#example-select").value = "custom";
    state.exampleId = "custom";
    updateCustomOptionLabel(grouped.length === 1 ? grouped[0].name : `${grouped.length} folders`);
    updateBatchLabel();
    updateCommand();
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
  updateCommand();
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
    const pdbFile = chooseInputFile(folderFiles, ".pdb", ["protein", "pocket"]);
    const sdfFile = chooseInputFile(folderFiles, ".sdf", ["ligand", "reference", "ref"]);
    if (!pdbFile) {
      skipped.push(`${folder}: no PDB`);
      continue;
    }
    const pdbText = await readValidatedTextFile(pdbFile, "pdb", false);
    const sdfText = sdfFile ? await readValidatedTextFile(sdfFile, "sdf", false) : null;
    if (!pdbText) {
      skipped.push(`${folder}: invalid PDB`);
      continue;
    }
    if (state.mode === "reference" && !sdfText) {
      skipped.push(`${folder}: no valid SDF`);
      continue;
    }
    jobs.push({
      name: folder,
      pdb: { name: pdbFile.name, text: pdbText },
      sdf: sdfText ? { name: sdfFile.name, text: sdfText } : null,
    });
  }
  if (!jobs.length) {
    throw new Error(state.mode === "reference"
      ? "No valid batch folders found. Each folder needs a PDB and SDF in reference mode."
      : "No valid batch folders found. Each folder needs a PDB.");
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

async function readValidatedTextFile(file, kind, showError = true) {
  const text = await file.text();
  const lower = file.name.toLowerCase();
  const validExtension = kind === "pdb" ? lower.endsWith(".pdb") : lower.endsWith(".sdf");
  const validContent = kind === "pdb"
    ? text.split(/\r?\n/, 200).some((line) => /^(ATOM  |HETATM|MODEL |HEADER|CRYST1)/.test(line))
    : text.includes("$$$$");
  if (!validExtension || !validContent) {
    if (showError) showToast(`${file.name} does not look like a valid ${kind.toUpperCase()} file.`);
    return null;
  }
  return text;
}

function updateBatchLabel() {
  const count = state.batchInputs.length;
  const target = resolvedTarget();
  const isSlurmGpu = isSlurmGpuTarget(target);
  const cpuBatchWarning = count > 10
    ? "Large local CPU batches can take many hours. Keep this server window open, or use Slurm GPU for durable parallel queueing."
    : "Local CPU batches run one at a time. Keep this server window open until the queued jobs finish.";
  $("#folder-name").textContent = count ? `${count} batch folder${count === 1 ? "" : "s"}` : "Batch folders";
  $("#folder-detail").textContent = count
    ? `Generate will submit ${count} ${isSlurmGpu ? "independent Slurm" : "serial queued CPU"} job${count === 1 ? "" : "s"}`
    : "Optional: one PDB and optional SDF per folder";
  $("#batch-mode-banner").hidden = !count;
  $("#batch-mode-banner").classList.toggle("is-warning", Boolean(count && !isSlurmGpu));
  $("#batch-mode-title").textContent = isSlurmGpu ? "Parallel GPU batch" : "Queued CPU batch";
  $("#batch-mode-message").textContent = count
    ? `${count} folder${count === 1 ? "" : "s"} ready. ${isSlurmGpu ? "Slurm will process folders concurrently when capacity is available." : cpuBatchWarning} Each folder is processed as its own job; inputs are never mixed.`
    : "Each selected folder will submit as a separate job.";
  $("#preview-run span").textContent = count
    ? `Submit ${count} batch job${count === 1 ? "" : "s"}`
    : "Generate molecules";
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
  if (state.resultSource === "job" && state.selectedJob?.id) {
    try {
      const saved = await service.exportJob(state.selectedJob.id, {
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
  candidates.forEach((item) => structures.file(item.name, item.text));
  zip.file("metrics.csv", csvText(candidates));
  zip.file("run_config.json", JSON.stringify(exportMetadata.run_config, null, 2));
  zip.file("export_metadata.json", JSON.stringify(exportMetadata, null, 2));
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

function buildConfiguration() {
  const job = state.resultSource === "job" ? state.selectedJob || state.currentJob : null;
  const example = state.study?.example || EXAMPLES[state.exampleId];
  const mode = job?.mode || state.mode;
  return {
    interface_version: "0.1.0",
    backend_connected: true,
    job_id: job?.id || null,
    conditioning_mode: mode,
    inputs: {
      pdb_filename: job ? (job.inputs?.pdb ? filenameOnly(job.inputs.pdb) : example?.pdb || null) : state.customPdb?.name || example?.pdb || null,
      sdf_filename: mode === "reference"
        ? (job ? (job.inputs?.sdf ? filenameOnly(job.inputs.sdf) : example?.sdf || null) : state.customSdf?.name || example?.sdf || null)
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
    job_id: state.selectedJob?.id || null,
    selected_count: candidates.length,
    total_count: state.study?.candidates?.length || 0,
    selected_candidates: candidates.map((item) => ({ id: item.id, name: item.name, path: item.path || item.name })),
    filters: activeExportFilters(),
    tool_runs: state.study?.toolRuns || state.selectedJob?.tool_runs || [],
    run_config: buildConfiguration(),
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
  updateExportFilterStatus();
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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[char]));
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
