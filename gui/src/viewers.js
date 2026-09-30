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
      loads.push(loadStructure(viewer, prepareSdfForViewer(options.referenceText), "sdf", "Reference ligand"));
    }
    if (molecule?.text) {
      loads.push(loadStructure(viewer, prepareSdfForViewer(molecule.text), "sdf", molecule.id || molecule.name || "Generated ligand"));
    }
    await Promise.all(loads);
  } catch (error) {
    console.warn("Mol* viewer failed", error);
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
