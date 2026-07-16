const ELEMENT_COLORS = {
  C: "#263632", N: "#3f67b1", O: "#c95b50", S: "#d6a431", F: "#4a9a6a",
  Cl: "#4a9a6a", Br: "#9c5f45", I: "#72559b", H: "#b9c1be",
};

let viewer = null;

export function render3D(container, molecule, receptorText, options = {}) {
  container.innerHTML = "";
  if (!window.$3Dmol) {
    container.innerHTML = "<div class='viewer-error'>3Dmol.js could not load. The 2D structure view remains available.</div>";
    return;
  }
  viewer = window.$3Dmol.createViewer(container, { backgroundColor: "#f7f9f8", antialias: true });
  if (receptorText && options.proteinStyle !== "hidden") {
    const receptor = viewer.addModel(receptorText, "pdb");
    if (options.proteinStyle === "surface") {
      receptor.setStyle({}, { line: { color: "#a7b3af", opacity: 0.25 } });
      viewer.addSurface(window.$3Dmol.SurfaceType.VDW, { opacity: 0.17, color: "#8ba69d" }, { model: receptor });
    } else if (options.proteinStyle === "line") {
      receptor.setStyle({}, { line: { color: "#aab6b2", opacity: 0.42 } });
    } else {
      receptor.setStyle({}, { cartoon: { color: "#a9bbb5", opacity: 0.72 } });
    }
  }
  const ligand = viewer.addModel(molecule.text, "sdf");
  const ligandStyle = options.ligandStyle || "stick";
  if (ligandStyle === "sphere") ligand.setStyle({}, { sphere: { scale: 0.28, colorscheme: "Jmol" } });
  else if (ligandStyle === "line") ligand.setStyle({}, { line: { linewidth: 2, colorscheme: "Jmol" } });
  else ligand.setStyle({}, { stick: { radius: 0.18, colorscheme: "Jmol" } });
  viewer.zoomTo({ model: ligand });
  viewer.zoom(0.9);
  viewer.render();
}

export function render2D(container, molecule) {
  const smiles = molecule.smiles || molecule.properties?.SMILES || "";
  if (smiles && typeof window.initRDKitModule === "function") {
    container.innerHTML = "<div class='viewer-loading'>Generating 2D depiction…</div>";
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
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="2D structure of ${molecule.id}"><rect width="${width}" height="${height}" fill="#f7f9f8"/>${bonds}${atoms}</svg>`;
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
