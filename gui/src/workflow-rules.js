export const VINA_METRICS = new Set(["vina_score", "vina_dock", "qvina"]);
export const CHEMISTRY_METRICS = new Set(["qed", "sa", "logp", "lipinski"]);

export const WORKFLOWS = {
  conditar_pocket: {
    id: "conditar_pocket",
    engine: "conditar",
    mode: "pocket",
    label: "conDitar pocket",
    conditioningLabel: "Pocket",
    required: { pdb: "required", sdf: "none" },
    metrics: { chemistry: true, vina: true },
  },
  conditar_reference: {
    id: "conditar_reference",
    engine: "conditar",
    mode: "reference",
    label: "conDitar protein + ligand",
    conditioningLabel: "Protein + ligand",
    required: { pdb: "required", sdf: "required" },
    metrics: { chemistry: true, vina: true },
  },
  diffsmol_ligand: {
    id: "diffsmol_ligand",
    engine: "diffsmol",
    mode: "reference",
    label: "DiffSMol ligand only",
    conditioningLabel: "Ligand only",
    required: { pdb: "optional", sdf: "required" },
    metrics: { chemistry: true, vina: false },
  },
  diffsmol_pocket: {
    id: "diffsmol_pocket",
    engine: "diffsmol",
    mode: "reference",
    label: "DiffSMol ligand + pocket",
    conditioningLabel: "Ligand + pocket",
    required: { pdb: "optional", sdf: "required" },
    metrics: { chemistry: true, vina: true },
  },
};

export function resolveGenerationWorkflow({ engine, mode, pdb, sdf }) {
  const hasPdb = Boolean(pdb?.text);
  const normalizedEngine = engine === "diffsmol" ? "diffsmol" : "conditar";
  if (normalizedEngine === "diffsmol") {
    return hasPdb ? WORKFLOWS.diffsmol_pocket : WORKFLOWS.diffsmol_ligand;
  }
  return mode === "pocket" ? WORKFLOWS.conditar_pocket : WORKFLOWS.conditar_reference;
}

export function requiredInputs(engine, mode, input = {}) {
  if (engine === "diffsmol" && !input.pdb && !input.sdf) {
    return { ...WORKFLOWS.diffsmol_ligand.required, label: "Shape ligand" };
  }
  const workflow = resolveGenerationWorkflow({ engine, mode, pdb: input.pdb, sdf: input.sdf });
  return { ...workflow.required, label: workflow.conditioningLabel };
}

export function validateGenerationSetup({ engine, mode, pdb, sdf, batchCount = 0 }) {
  const workflow = resolveGenerationWorkflow({ engine, mode, pdb, sdf });
  if (batchCount > 0) return { ready: true, errors: [], required: workflow.required, workflow };

  const hasPdb = Boolean(pdb?.text);
  const hasSdf = Boolean(sdf?.text);
  const errors = [];

  if (workflow.engine === "diffsmol") {
    if (!hasSdf) errors.push("DiffSMol needs a 3D reference ligand SDF.");
  } else {
    if (!hasPdb) errors.push("Load or upload a PDB before submitting a job.");
    if (workflow.mode === "reference" && !hasSdf) errors.push("Protein + reference ligand mode needs an SDF ligand.");
  }

  return {
    ready: errors.length === 0,
    errors,
    required: workflow.required,
    workflow,
  };
}

export function metricAllowedForWorkflow(metric, workflow) {
  if (!workflow) return true;
  if (VINA_METRICS.has(metric)) return Boolean(workflow.metrics.vina);
  if (CHEMISTRY_METRICS.has(metric)) return Boolean(workflow.metrics.chemistry);
  return true;
}

export function filterMetricsForWorkflow(metrics, workflow) {
  return (metrics || []).filter((metric) => metricAllowedForWorkflow(metric, workflow));
}

export function vinaModeForMetrics(metrics = []) {
  const selected = new Set(metrics);
  const wantsScore = selected.has("vina_score");
  const wantsDock = selected.has("vina_dock");
  const wantsQvina = selected.has("qvina");
  if ((wantsScore || wantsDock) && wantsQvina) return "all";
  if (wantsDock) return "vina_dock";
  if (wantsQvina) return "qvina";
  if (wantsScore) return "vina_score";
  return "none";
}

export function setupWarningsForWorkflow({
  workflow,
  sdf,
  sdfHeavyAtoms = 0,
  heavyAtomWarningThreshold = 35,
  selectedMetrics = [],
  stagedPocketWarnings = [],
}) {
  const warnings = [];
  stagedPocketWarnings.forEach((warning) => {
    warnings.push(normalizeWorkflowWarning(warning, {
      code: "pocket_size",
      severity: "warning",
      title: "Pocket size",
      appliesTo: "pocket",
    }));
  });
  if (workflow?.id?.startsWith("diffsmol") && sdfHeavyAtoms > heavyAtomWarningThreshold) {
    warnings.push({
      code: "diffsmol_large_ligand",
      severity: "warning",
      title: "Large DiffSMol ligand",
      text: `${sdf?.name || "Reference SDF"} has ${sdfHeavyAtoms} heavy atoms. Less than ~${heavyAtomWarningThreshold} atoms is advised.`,
      appliesTo: "sdf",
    });
  }
  const wantsVina = selectedMetrics.some((item) => VINA_METRICS.has(item));
  if (workflow?.id === "diffsmol_ligand" && wantsVina) {
    warnings.push({
      code: "diffsmol_vina_requires_pocket",
      severity: "warning",
      title: "Vina/QVina requires a pocket",
      text: "This is ligand-only DiffSMol. Vina/QVina will be disabled from the submitted job unless you add a pocket PDB.",
      appliesTo: "postprocess",
    });
  }
  return dedupeWarnings(warnings);
}

export function normalizeWorkflowWarning(warning, defaults = {}) {
  if (warning && typeof warning === "object") {
    return {
      code: warning.code || defaults.code || "warning",
      severity: warning.severity || defaults.severity || "warning",
      title: warning.title || defaults.title || "Warning",
      text: warning.text || warning.message || String(warning),
      appliesTo: warning.appliesTo || warning.applies_to || defaults.appliesTo || "workflow",
    };
  }
  return {
    code: defaults.code || "warning",
    severity: defaults.severity || "warning",
    title: defaults.title || "Warning",
    text: String(warning || ""),
    appliesTo: defaults.appliesTo || "workflow",
  };
}

export function warningText(warning) {
  return normalizeWorkflowWarning(warning).text;
}

function dedupeWarnings(warnings) {
  const seen = new Set();
  return warnings.filter((warning) => {
    const normalized = normalizeWorkflowWarning(warning);
    const key = `${normalized.code}:${normalized.text}`;
    if (seen.has(key)) return false;
    seen.add(key);
    Object.assign(warning, normalized);
    return true;
  });
}
