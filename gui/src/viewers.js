const ELEMENT_COLORS = {
  C: "#263632", N: "#3f67b1", O: "#c95b50", S: "#d6a431", F: "#4a9a6a",
  Cl: "#4a9a6a", Br: "#9c5f45", I: "#72559b", H: "#b9c1be",
};

const molstarViewers = new WeakMap();
const renderTokens = new WeakMap();

export async function render3D(container, molecule, receptorText, options = {}) {
  const token = (renderTokens.get(container) || 0) + 1;
  renderTokens.set(container, token);
  container.innerHTML = "";
  molstarViewers.get(container)?.dispose?.();
  molstarViewers.delete(container);

  try {
    const molstar = await waitForMolstar();
    if (token !== renderTokens.get(container)) return;
    const viewer = await molstar.Viewer.create(container, {
      layoutIsExpanded: false,
      layoutShowControls: true,
      layoutShowRemoteState: false,
      layoutShowSequence: true,
      layoutShowLog: false,
      layoutShowLeftPanel: true,
      viewportShowExpand: true,
      viewportShowSelectionMode: true,
      viewportShowAnimation: false,
      extensions: [],
      pdbProvider: "rcsb",
      emdbProvider: "rcsb",
    });
    molstarViewers.set(container, viewer);

    const loads = [];
    if (receptorText) {
      loads.push(loadStructure(viewer, receptorText, "pdb", "Input protein"));
    }
    if (options.referenceText) {
      loads.push(loadStructure(viewer, options.referenceText, "sdf", "Reference ligand"));
    }
    if (molecule?.text) {
      loads.push(loadStructure(viewer, molecule.text, "sdf", molecule.id || molecule.name || "Generated ligand"));
    }
    await Promise.all(loads);
  } catch (error) {
    console.warn("Mol* viewer failed", error);
    container.innerHTML = `<div class="viewer-error">Mol* could not load this structure: ${escapeHtml(error.message || String(error))}</div>`;
  }
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
