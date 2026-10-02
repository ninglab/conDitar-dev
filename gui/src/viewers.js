const ELEMENT_COLORS = {
  C: "#263632", N: "#3f67b1", O: "#c95b50", S: "#d6a431", F: "#4a9a6a",
  Cl: "#4a9a6a", Br: "#9c5f45", I: "#72559b", H: "#b9c1be",
};
const COVALENT_RADII = {
  B: 0.85, C: 0.76, N: 0.71, O: 0.66, F: 0.57, P: 1.07, S: 1.05,
  Cl: 1.02, Br: 1.20, I: 1.39,
};
const DISPLAY_VALENCE = {
  B: 3, C: 4, N: 3, O: 2, F: 1, P: 5, S: 6, Cl: 1, Br: 1, I: 1,
};
const MAX_DISPLAY_BOND_ANGSTROMS = 2.6;

const molstarViewers = new WeakMap();
const renderTokens = new WeakMap();
const viewerControls = new WeakMap();

function createViewerControls(container, options) {
  const previous = viewerControls.get(container);
  previous?.clickSubscription?.unsubscribe();
  const host = options.controlsHost;
  if (!host) return null;
  const kinds = [
    ...(options.receptorText ? ["protein"] : []),
    ...(options.referenceText ? ["reference"] : []),
    ...(options.ligandText ? ["ligand"] : []),
  ];
  host.hidden = kinds.length === 0;
  if (!kinds.length) {
    host.innerHTML = "";
    viewerControls.delete(container);
    return null;
  }
  const state = {
    container, host, kinds, viewer: null, structures: {}, mode: "none", residues: new Set(),
    firstDistanceLoci: null, clickSubscription: null, operation: Promise.resolve(), measurementRefs: [],
  };
  viewerControls.set(container, state);
  const visibility = kinds.filter((kind) => kind !== "reference" || !options.hideReferenceVisibility)
    .map((kind) => `<label class="viewer-check"><input type="checkbox" data-visibility="${kind}" ${options.defaultHiddenKinds?.includes(kind) ? "" : "checked"}>${kind === "protein" ? "Protein" : kind === "reference" ? "Reference" : options.ligandLabel || "Ligand"}</label>`).join("");
  const swatches = kinds.map((kind) => `<input type="color" data-uniform-color="${kind}" value="${kind === "protein" ? "#1b8063" : kind === "reference" ? "#9a637d" : "#e07832"}" aria-label="${kind} color" title="${kind} color" hidden>`).join("");
  host.innerHTML = `
    <div class="viewer-control-row">
      <div class="viewer-control-group">${visibility}</div>
      ${kinds.includes("protein") ? `<label>Protein <select data-style="protein" title="Cartoon is Mol*'s ribbon view. Cropped pockets may appear discontinuous."><option value="cartoon">Cartoon</option><option value="ball-and-stick">Sticks</option><option value="molecular-surface">Surface</option></select></label>` : ""}
      ${kinds.includes("ligand") ? `<label>${options.ligandLabel || "Ligand"} <select data-style="ligand"><option value="ball-and-stick">Sticks</option><option value="spacefill">Spacefill</option></select></label>` : ""}
      ${kinds.includes("reference") ? `<label>Reference <select data-style="reference"><option value="ball-and-stick">Sticks</option><option value="spacefill">Spacefill</option></select></label>` : ""}
      <label>Color <select data-color-scheme><option value="default">Default</option><option value="uniform">Custom</option></select></label>
      ${swatches}
      <button type="button" data-viewer-action="fit" title="Fit structures in view">Fit</button>
      ${kinds.includes("protein") ? `<button type="button" data-viewer-action="select" aria-pressed="false" title="Click protein residues to collect a selection">Select residues</button>` : ""}
      <button type="button" data-viewer-action="distance" aria-pressed="false" title="Click two atoms in the same structure to measure distance">Distance</button>
      <button type="button" data-viewer-action="clear" title="Clear picked residues and measurement mode">Clear</button>
      <button type="button" data-viewer-action="advanced" aria-pressed="false" title="Show Mol* advanced structure tools">Advanced</button>
    </div>
    <div class="viewer-control-feedback"><span data-viewer-status>Select or measure directly on the structure.</span>${options.onUseResidues ? `<button type="button" data-viewer-action="use-residues" hidden>Use residues for pocket</button>` : ""}</div>`;
  host.onchange = (event) => {
    const target = event.target;
    if (target.dataset.visibility) runControl(state, () => toggleStructure(state, target.dataset.visibility));
    if (target.dataset.style) runControl(state, () => setStructureStyle(state, target.dataset.style, target.value));
    if ("colorScheme" in target.dataset || "uniformColor" in target.dataset) {
      host.querySelectorAll("[data-uniform-color]").forEach((swatch) => { swatch.hidden = host.querySelector("[data-color-scheme]").value !== "uniform"; });
      runControl(state, () => setStructureColor(state));
    }
  };
  host.onclick = (event) => {
    const action = event.target.closest("[data-viewer-action]")?.dataset.viewerAction;
    if (!action) return;
    if (action === "fit") state.viewer?.plugin.canvas3d?.requestCameraReset();
    if (action === "select" || action === "distance") setPickingMode(state, state.mode === action ? "none" : action);
    if (action === "clear") {
      state.residues.clear();
      setPickingMode(state, "none");
      setViewerStatus(state, "Selection cleared.");
      state.viewer?.plugin.managers.interactivity.lociSelects.deselectAll();
      runControl(state, async () => {
        const refs = state.measurementRefs.splice(0);
        if (!refs.length) return;
        const update = state.viewer.plugin.state.data.build();
        refs.forEach((ref) => update.delete(ref));
        await update.commit();
      });
    }
    if (action === "advanced") {
      const pressed = event.target.getAttribute("aria-pressed") !== "true";
      event.target.setAttribute("aria-pressed", String(pressed));
      state.viewer?.plugin.layout.setProps({ showControls: pressed });
      state.viewer?.plugin.layout.events.updated.next(void 0);
    }
    if (action === "use-residues" && state.residues.size) options.onUseResidues?.([...state.residues].sort().join(", "));
  };
  return state;
}

function setViewerStatus(state, message) {
  state.host.querySelector("[data-viewer-status]").textContent = message;
  const useButton = state.host.querySelector('[data-viewer-action="use-residues"]');
  if (useButton) useButton.hidden = !state.residues.size;
}

function runControl(state, callback) {
  state.operation = state.operation.then(() => {
    if (viewerControls.get(state.container) === state) return callback();
  }).catch((error) => {
    if (viewerControls.get(state.container) === state) setViewerStatus(state, `Viewer control: ${error.message || error}`);
  });
}

function componentsFor(state, kind) {
  const ref = state.structures[kind];
  return state.viewer?.plugin.managers.structure.hierarchy.current.structures
    .find((structure) => structure.cell.transform.ref === ref)?.components || [];
}

async function toggleStructure(state, kind) {
  const components = componentsFor(state, kind);
  if (components.length) await state.viewer.plugin.managers.structure.component.toggleVisibility(components);
}

async function setStructureStyle(state, kind, style) {
  const components = componentsFor(state, kind);
  if (!components.length) return;
  const manager = state.viewer.plugin.managers.structure.component;
  await manager.removeRepresentations(components);
  const current = componentsFor(state, kind);
  const preferred = kind === "protein" && style === "cartoon" ? "Polymer" : "All";
  const target = current.find((component) => component.cell.obj?.label === preferred) || current[0];
  if (target) await manager.addRepresentation([target], style);
  await setStructureColor(state);
}

async function setStructureColor(state) {
  const manager = state.viewer?.plugin.managers.structure.component;
  if (!manager) return;
  const uniform = state.host.querySelector("[data-color-scheme]").value === "uniform";
  for (const kind of state.kinds) {
    const components = componentsFor(state, kind);
    if (!components.length) continue;
    const value = Number.parseInt(state.host.querySelector(`[data-uniform-color="${kind}"]`).value.slice(1), 16);
    await manager.updateRepresentationsTheme(components, uniform
      ? { color: "uniform", colorParams: { value } }
      : { color: kind === "protein" ? "chain-id" : "element-symbol" });
  }
}

function setPickingMode(state, mode) {
  state.mode = mode;
  state.firstDistanceLoci = null;
  for (const action of ["select", "distance"]) {
    state.host.querySelector(`[data-viewer-action="${action}"]`)?.setAttribute("aria-pressed", String(mode === action));
  }
  state.viewer?.plugin.behaviors.interaction.selectionMode.next(mode !== "none");
  setViewerStatus(state, mode === "select" ? "Click protein residues to select them." : mode === "distance" ? "Click the first atom to measure." : "Select or measure directly on the structure.");
}

function residueFromLoci(loci) {
  const element = loci.elements?.[0];
  if (!element?.unit?.model?.atomicHierarchy) return null;
  const indices = element.indices;
  let offset;
  if (typeof indices === "number") {
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, indices, true);
    offset = view.getInt32(0, true);
  } else {
    offset = indices?.[0];
  }
  const atom = element.unit.elements[offset];
  const hierarchy = element.unit.model.atomicHierarchy;
  const residue = hierarchy.residueAtomSegments.index[atom];
  const chain = hierarchy.chainAtomSegments.index[atom];
  const chainId = hierarchy.chains.auth_asym_id.value(chain);
  const residueId = hierarchy.residues.auth_seq_id.value(residue);
  if (!chainId || !Number.isFinite(residueId)) return null;
  const insertion = hierarchy.residues.pdbx_PDB_ins_code?.value(residue);
  return `${chainId}:${residueId}${insertion && insertion !== "?" && insertion !== "." ? insertion : ""}`;
}

function handleViewerClick(state, event) {
  const loci = event.current?.loci;
  if (loci?.kind !== "element-loci" || state.mode === "none") return;
  if (state.mode === "select") {
    const protein = state.viewer?.plugin.managers.structure.hierarchy.current.structures
      .find((structure) => structure.cell.transform.ref === state.structures.protein)?.cell?.obj?.data;
    if (!protein?.models?.includes(loci.elements?.[0]?.unit?.model)) return;
    const residue = residueFromLoci(loci);
    if (!residue) return;
    if (state.residues.has(residue)) state.residues.delete(residue);
    else state.residues.add(residue);
    setViewerStatus(state, state.residues.size ? `Residues: ${[...state.residues].sort().join(", ")}` : "Click protein residues to select them.");
    return;
  }
  if (!state.firstDistanceLoci) {
    state.firstDistanceLoci = loci;
    setViewerStatus(state, "Click the second atom in the same structure.");
    return;
  }
  const first = state.firstDistanceLoci;
  state.firstDistanceLoci = null;
  if (first.structure !== loci.structure) {
    setViewerStatus(state, "Distance requires two atoms in the same structure.");
    return;
  }
  runControl(state, async () => {
    const added = await state.viewer.plugin.managers.structure.measurement.addDistance(first, loci);
    if (added?.selection?.ref) state.measurementRefs.push(added.selection.ref);
    setViewerStatus(state, "Distance added. Click another pair to measure again.");
  });
}

export async function render3D(container, molecule, receptorText, options = {}) {
  const token = (renderTokens.get(container) || 0) + 1;
  renderTokens.set(container, token);
  container.innerHTML = "";
  const controls = createViewerControls(container, {
    ...options, receptorText, ligandText: molecule?.text,
  });
  molstarViewers.get(container)?.dispose?.();
  molstarViewers.delete(container);

  try {
    const molstar = await waitForMolstar();
    if (token !== renderTokens.get(container)) return;
    const viewer = await molstar.Viewer.create(container, {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowRemoteState: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      viewportShowExpand: true,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      extensions: [],
      pdbProvider: "rcsb",
      emdbProvider: "rcsb",
    });
    molstarViewers.set(container, viewer);
    if (controls) {
      controls.viewer = viewer;
      controls.clickSubscription = viewer.plugin.behaviors.interaction.click.subscribe((event) => handleViewerClick(controls, event));
    }
    const loadKind = async (kind, data, format, label) => {
      const count = viewer.plugin.managers.structure.hierarchy.current.structures.length;
      await loadStructure(viewer, data, format, label);
      if (controls && viewer.plugin.managers.structure.hierarchy.current.structures.length > count) {
        controls.structures[kind] = viewer.plugin.managers.structure.hierarchy.current.structures.at(-1).cell.transform.ref;
      }
    };
    if (receptorText) {
      await loadKind("protein", receptorText, "pdb", "Input protein");
    }
    if (options.referenceText) {
      await loadKind("reference", prepareSdfForViewer(options.referenceText), "sdf", "Reference ligand");
    }
    if (molecule?.text) {
      await loadKind("ligand", prepareSdfForViewer(molecule.text), "sdf", molecule.id || molecule.name || "Generated ligand");
    }
    if (token !== renderTokens.get(container)) return;
    for (const kind of options.defaultHiddenKinds || []) {
      if (controls?.structures[kind]) await toggleStructure(controls, kind);
    }
    viewer.plugin.canvas3d?.requestCameraReset();
  } catch (error) {
    console.warn("Mol* viewer failed", error);
    if (controls) controls.host.hidden = true;
    container.innerHTML = `<div class="viewer-error">Mol* could not load this structure: ${escapeHtml(error.message || String(error))}</div>`;
  }
}

export function prepareSdfForViewer(text) {
  const normalized = String(text || "").replace(/\r/g, "");
  const delimiter = normalized.includes("$$$$") ? "$$$$" : "";
  const blocks = delimiter ? normalized.split(delimiter) : [normalized];
  const cleaned = blocks.map((block) => stripHydrogensFromMolBlock(block));
  return delimiter ? cleaned.join(delimiter) : cleaned[0];
}

function stripHydrogensFromMolBlock(block) {
  if (!block.trim()) return block;
  const lines = block.split("\n");
  const countLine = lines[3] || "";
  if (!/V2000/.test(countLine)) return block;
  const atomCount = Number.parseInt(countLine.slice(0, 3), 10);
  const bondCount = Number.parseInt(countLine.slice(3, 6), 10);
  if (!Number.isInteger(atomCount) || !Number.isInteger(bondCount)) return block;

  const atomLines = lines.slice(4, 4 + atomCount);
  const bondLines = lines.slice(4 + atomCount, 4 + atomCount + bondCount);
  if (atomLines.length !== atomCount || bondLines.length !== bondCount) return block;

  const indexMap = new Map();
  const heavyAtomLines = [];
  atomLines.forEach((line, index) => {
    if (line.slice(31, 34).trim().toUpperCase() === "H") return;
    indexMap.set(index + 1, heavyAtomLines.length + 1);
    heavyAtomLines.push(line);
  });
  const heavyBondLines = [];
  bondLines.forEach((line) => {
    const begin = indexMap.get(Number.parseInt(line.slice(0, 3), 10));
    const end = indexMap.get(Number.parseInt(line.slice(3, 6), 10));
    if (!begin || !end) return;
    heavyBondLines.push(`${String(begin).padStart(3)}${String(end).padStart(3)}${line.slice(6)}`);
  });
  const atoms = heavyAtomLines.map(parseMolAtomLine);
  const displayBondLines = heavyBondLines.some((line) => molBondLength(line, atoms) > MAX_DISPLAY_BOND_ANGSTROMS)
    ? inferredDisplayBondLines(atoms)
    : heavyBondLines;
  const updatedCountLine = `${String(heavyAtomLines.length).padStart(3)}${String(displayBondLines.length).padStart(3)}${countLine.slice(6)}`;
  return [
    ...lines.slice(0, 3),
    updatedCountLine,
    ...heavyAtomLines,
    ...displayBondLines,
    ...lines.slice(4 + atomCount + bondCount),
  ].join("\n");
}

function parseMolAtomLine(line) {
  return {
    x: Number.parseFloat(line.slice(0, 10)),
    y: Number.parseFloat(line.slice(10, 20)),
    z: Number.parseFloat(line.slice(20, 30)),
    element: line.slice(31, 34).trim(),
  };
}

function molBondLength(line, atoms) {
  const begin = atoms[Number.parseInt(line.slice(0, 3), 10) - 1];
  const end = atoms[Number.parseInt(line.slice(3, 6), 10) - 1];
  if (!begin || !end) return Number.POSITIVE_INFINITY;
  return Math.hypot(begin.x - end.x, begin.y - end.y, begin.z - end.z);
}

function inferredDisplayBondLines(atoms) {
  const candidates = [];
  for (let begin = 0; begin < atoms.length; begin += 1) {
    for (let end = begin + 1; end < atoms.length; end += 1) {
      const distance = Math.hypot(
        atoms[begin].x - atoms[end].x,
        atoms[begin].y - atoms[end].y,
        atoms[begin].z - atoms[end].z,
      );
      const radius = (COVALENT_RADII[atoms[begin].element] || 0.77)
        + (COVALENT_RADII[atoms[end].element] || 0.77);
      if (distance > 0.45 && distance <= radius + 0.45) {
        candidates.push({ begin, end, distance, ratio: distance / radius });
      }
    }
  }
  candidates.sort((a, b) => a.ratio - b.ratio || a.distance - b.distance);
  const degrees = Array(atoms.length).fill(0);
  return candidates.flatMap(({ begin, end }) => {
    const beginLimit = DISPLAY_VALENCE[atoms[begin].element] || 4;
    const endLimit = DISPLAY_VALENCE[atoms[end].element] || 4;
    if (degrees[begin] >= beginLimit || degrees[end] >= endLimit) return [];
    degrees[begin] += 1;
    degrees[end] += 1;
    return [`${String(begin + 1).padStart(3)}${String(end + 1).padStart(3)}  1  0  0  0  0`];
  });
}

export function render2D(container, molecule) {
  const smiles = molecule.smiles || molecule.properties?.SMILES || "";
  if (smiles && typeof window.initRDKitModule === "function") {
    container.innerHTML = "<div class='viewer-loading'>Generating 2D depiction...</div>";
    window.initRDKitModule().then((RDKit) => {
      const mol = RDKit.get_mol(smiles);
      if (!mol) throw new Error("Invalid SMILES");
      try {
        container.innerHTML = mol.get_svg(720, 480);
      } finally {
        mol.delete();
      }
    }).catch(() => renderCoordinate2D(container, molecule));
    return;
  }
  renderCoordinate2D(container, molecule);
}

async function waitForMolstar() {
  for (let attempts = 0; attempts < 80; attempts += 1) {
    if (window.molstar?.Viewer?.create) return window.molstar;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Mol* did not finish loading.");
}

async function loadStructure(viewer, data, preferredFormat, label) {
  const formats = preferredFormat === "sdf" ? ["sdf", "mol"] : [preferredFormat];
  let lastError = null;
  for (const format of formats) {
    try {
      await viewer.loadStructureFromData(data, format, { dataLabel: label });
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error(`Unable to load ${label}.`);
}

function renderCoordinate2D(container, molecule) {
  const width = 720;
  const height = 480;
  const pad = 45;
  const xs = molecule.atoms.map((atom) => atom.x);
  const ys = molecule.atoms.map((atom) => atom.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = Math.min((width - pad * 2) / (maxX - minX || 1), (height - pad * 2) / (maxY - minY || 1));
  const project = (atom) => ({
    x: pad + (atom.x - minX) * scale + (width - pad * 2 - (maxX - minX) * scale) / 2,
    y: height - pad - (atom.y - minY) * scale - (height - pad * 2 - (maxY - minY) * scale) / 2,
  });
  const coords = molecule.atoms.map(project);
  const bonds = molecule.bonds.map((bond) => bondSvg(coords[bond.a], coords[bond.b], bond.order)).join("");
  const atoms = molecule.atoms.map((atom, index) => {
    const point = coords[index];
    if (atom.element === "C" || atom.element === "H") return "";
    return `<g><circle cx="${point.x}" cy="${point.y}" r="12" fill="#f7f9f8"/><text x="${point.x}" y="${point.y + 5}" text-anchor="middle" fill="${ELEMENT_COLORS[atom.element] || "#263632"}">${atom.element}</text></g>`;
  }).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="2D structure of ${escapeHtml(molecule.id)}"><rect width="${width}" height="${height}" fill="#f7f9f8"/>${bonds}${atoms}</svg>`;
}

function bondSvg(a, b, order) {
  if (!a || !b) return "";
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  const offsetX = (-dy / length) * 3;
  const offsetY = (dx / length) * 3;
  const line = (offset = 0) => `<line x1="${a.x + offsetX * offset}" y1="${a.y + offsetY * offset}" x2="${b.x + offsetX * offset}" y2="${b.y + offsetY * offset}" stroke="#40514c" stroke-width="2.3" stroke-linecap="round"/>`;
  if (order === 2) return line(-1) + line(1);
  if (order >= 3) return line(-1.7) + line(0) + line(1.7);
  return line(0);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;",
  }[char]));
}
