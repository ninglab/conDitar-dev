import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import vm from "node:vm";

const modulePath = resolve("gui/src/workflow-rules.js");
const source = await readFile(modulePath, "utf8");
const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}#${pathToFileURL(modulePath)}`;
const {
  filterMetricsForWorkflow,
  resolveGenerationWorkflow,
  setupWarningsForWorkflow,
  validateGenerationSetup,
  vinaModeForMetrics,
} = await import(moduleUrl);
const viewerPath = resolve("gui/src/viewers.js");
const viewerSource = await readFile(viewerPath, "utf8");
const viewerUrl = `data:text/javascript;base64,${Buffer.from(viewerSource).toString("base64")}#${pathToFileURL(viewerPath)}`;
const { prepareSdfForViewer } = await import(viewerUrl);

const pdb = { name: "pocket.pdb", text: "ATOM\n" };
const sdf = { name: "reference.sdf", text: "mol\n$$$$\n" };

function validation(payload) {
  return validateGenerationSetup(payload);
}

const workflowCases = [
  {
    name: "conDitar pocket",
    payload: { engine: "conditar", mode: "pocket", pdb, sdf: null },
    workflow: "conditar_pocket",
    ready: true,
  },
  {
    name: "conDitar reference",
    payload: { engine: "conditar", mode: "reference", pdb, sdf },
    workflow: "conditar_reference",
    ready: true,
  },
  {
    name: "DiffSMol ligand only",
    payload: { engine: "diffsmol", mode: "reference", pdb: null, sdf },
    workflow: "diffsmol_ligand",
    ready: true,
  },
  {
    name: "DiffSMol ligand + pocket",
    payload: { engine: "diffsmol", mode: "reference", pdb, sdf },
    workflow: "diffsmol_pocket",
    ready: true,
  },
];

for (const item of workflowCases) {
  const result = validation(item.payload);
  assert.equal(result.ready, item.ready, item.name);
  assert.equal(result.workflow.id, item.workflow, item.name);
  assert.equal(resolveGenerationWorkflow(item.payload).id, item.workflow, item.name);
}

assert.deepEqual(validation({ engine: "diffsmol", mode: "reference", pdb, sdf: null }).errors, [
  "DiffSMol needs a 3D reference ligand SDF.",
]);
assert.deepEqual(validation({ engine: "conditar", mode: "pocket", pdb: null, sdf: null }).errors, [
  "Load or upload a PDB before submitting a job.",
]);
assert.deepEqual(validation({ engine: "conditar", mode: "reference", pdb, sdf: null }).errors, [
  "Protein + reference ligand mode needs an SDF ligand.",
]);
assert.equal(validation({ engine: "conditar", mode: "reference", pdb: null, sdf: null, batchCount: 2 }).ready, true);

const ligandOnly = resolveGenerationWorkflow({ engine: "diffsmol", mode: "reference", pdb: null, sdf });
assert.deepEqual(
  filterMetricsForWorkflow(["qed", "sa", "vina_score", "qvina"], ligandOnly),
  ["qed", "sa"],
);

const ligandPocket = resolveGenerationWorkflow({ engine: "diffsmol", mode: "reference", pdb, sdf });
assert.deepEqual(
  filterMetricsForWorkflow(["qed", "sa", "vina_score", "qvina"], ligandPocket),
  ["qed", "sa", "vina_score", "qvina"],
);
assert.equal(vinaModeForMetrics(["vina_score", "qed", "sa"]), "vina_score");
assert.equal(vinaModeForMetrics(["vina_dock", "qed"]), "vina_dock");
assert.equal(vinaModeForMetrics(["qvina", "qed"]), "qvina");
assert.equal(vinaModeForMetrics(["vina_score", "vina_dock", "qvina"]), "all");
assert.equal(vinaModeForMetrics(["qed", "sa"]), "none");

const warnings = setupWarningsForWorkflow({
  workflow: ligandOnly,
  sdf: { name: "large.sdf" },
  sdfHeavyAtoms: 78,
  selectedMetrics: ["qed", "vina_score"],
  stagedPocketWarnings: [],
});
assert.deepEqual(warnings.map((item) => item.code), [
  "diffsmol_large_ligand",
  "diffsmol_vina_requires_pocket",
]);

const pocketWarnings = setupWarningsForWorkflow({
  workflow: ligandPocket,
  sdf,
  sdfHeavyAtoms: 29,
  selectedMetrics: ["qed", "vina_score"],
  stagedPocketWarnings: [{ code: "pocket_lt_10", title: "Pocket size", text: "Pocket contains fewer than 10 complete residues." }],
});
assert.deepEqual(pocketWarnings.map((item) => item.code), ["pocket_lt_10"]);

const explicitHydrogenSdf = `Viewer test
  conDitar GUI

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   20.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
>  <SOURCE>
unchanged

$$$$
`;
const displaySdf = prepareSdfForViewer(explicitHydrogenSdf);
const displayLines = displaySdf.split("\n");
assert.equal(Number.parseInt(displayLines[3].slice(0, 3), 10), 2);
assert.equal(Number.parseInt(displayLines[3].slice(3, 6), 10), 1);
assert.equal(displayLines.slice(4, 6).some((line) => line.slice(31, 34).trim() === "H"), false);
assert.match(displaySdf, />  <SOURCE>\nunchanged/);

const corruptTopologySdf = `Viewer topology test
  conDitar GUI

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.7000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$
`;
const repairedSdf = prepareSdfForViewer(corruptTopologySdf);
const repairedLines = repairedSdf.split("\n");
assert.equal(Number.parseInt(repairedLines[3].slice(3, 6), 10), 2);
assert.deepEqual(repairedLines.slice(7, 9).map((line) => line.slice(0, 6)), ["  2  3", "  1  2"]);

const appSource = await readFile(resolve("gui/src/app.js"), "utf8");
const appContext = {
  state: {
    customPdb: { name: "old-pocket.pdb", text: "ATOM" },
    customSdf: { name: "old-reference.sdf", text: "SDF" },
    resultSource: "job",
    study: { loadedJob: { id: "job-a", mode: "reference", engine: "diffsmol", inputs: {}, parameters: {} }, candidates: [] },
    selectedJob: { id: "job-b", inputs: {}, parameters: {} },
    mode: "reference",
    parameters: {},
  },
  setupStudyInput: () => null,
  exportCandidates: () => [],
  activeExportFilters: () => [],
  isBuiltinPostprocessRunning: () => false,
};
vm.createContext(appContext);
vm.runInContext(appSource.slice(appSource.indexOf("function setupPdbInput("), appSource.indexOf("function currentGenerationInput(")), appContext);
vm.runInContext(appSource.slice(appSource.indexOf("function loadedResultsJob("), appSource.indexOf("function buildRunManifest(")), appContext);
vm.runInContext(appSource.slice(appSource.indexOf("function isResultsReadyJob("), appSource.indexOf("function shortJobId(")), appContext);
vm.runInContext(appSource.slice(appSource.indexOf("function toolRunSummary("), appSource.indexOf("function collectToolOptions(")), appContext);
assert.equal(appContext.setupPdbInput({ pdb: null, sdf }), null);
assert.equal(appContext.setupSdfInput({ pdb, sdf: null }), null);
assert.equal(appContext.buildConfiguration().job_id, "job-a");
assert.equal(appContext.buildExportMetadata().job_id, "job-a");
assert.equal(appContext.isResultsReadyJob({ status: "completed", tools: [{ status: "pending" }] }), false);
assert.equal(appContext.isResultsReadyJob({ status: "completed", outputs: { sdf_count: 0 } }), false);
appContext.state.study.referenceSdf = "reference";
assert.match(appContext.toolRunSummary({ status: "completed", tool_id: "medchem_filters", result: { molecules: 2 } }), /rerun for corrected totals/);

const toolList = { innerHTML: "" };
const toolStatus = { textContent: "" };
appContext.state.toolsLoaded = true;
appContext.state.tools = [
  { id: "lilly_medchem", name: "Lilly", available: false, error: "Not installed" },
  { id: "medchem_filters", name: "MedChem", available: true },
];
appContext.state.study.loadedJob.status = "completed";
appContext.state.study.loadedJob.outputs = { sdf_count: 1 };
appContext.$ = (selector) => ({ "#tool-list": toolList, "#tool-chest-status": toolStatus })[selector];
appContext.$$ = () => [];
appContext.escapeHtml = (value) => String(value);
vm.runInContext(appSource.slice(appSource.indexOf("function renderToolChest("), appSource.indexOf("function renderEvaluationTools(")), appContext);
appContext.renderToolChest();
assert.equal(toolStatus.textContent, "1 tool available");
assert.match(toolList.innerHTML, /Unavailable/);
vm.runInContext(appSource.slice(appSource.indexOf("function resultEvaluationFailures("), appSource.indexOf("function provenanceItem(")), appContext);
assert.deepEqual(Array.from(appContext.resultEvaluationFailures({ postprocess: { status: "completed" }, tools: [] })), []);
assert.deepEqual(Array.from(appContext.resultEvaluationFailures({ postprocess: { status: "failed" }, tools: [{ status: "failed" }] })), [
  "built-in postprocessing",
  "a selected Tool Chest evaluator",
]);

const sdfSource = await readFile(resolve("gui/src/sdf.js"), "utf8");
const { parseSdf, splitSdfRecords } = await import(`data:text/javascript;base64,${Buffer.from(sdfSource).toString("base64")}`);
const unprofiled = parseSdf(explicitHydrogenSdf);
assert.equal(unprofiled.molecularWeight, null);
assert.equal(unprofiled.formula, null);
assert.equal(splitSdfRecords(explicitHydrogenSdf + explicitHydrogenSdf).length, 2);

console.log("Frontend workflow rule checks passed.");
