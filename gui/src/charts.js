const COLORS = {
  ink: "#172321",
  muted: "#77837f",
  line: "#dbe2df",
  accent: "#2f7d68",
  accentSoft: "rgba(47, 125, 104, .16)",
};

export function drawHistogram(canvas, values, label, threshold = null) {
  const { ctx, width, height } = prepare(canvas);
  ctx.clearRect(0, 0, width, height);
  canvas._histogram = null;
  if (!values.length) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bins = 12;
  const step = (max - min || 1) / bins;
  const counts = Array(bins).fill(0);
  values.forEach((value) => {
    const index = Math.min(bins - 1, Math.floor((value - min) / step));
    counts[index] += 1;
  });
  const pad = { left: 40, right: 12, top: 16, bottom: 34 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxCount = Math.max(...counts);
  drawAxes(ctx, width, height, pad, min.toFixed(0), max.toFixed(0), label);
  counts.forEach((count, index) => {
    const gap = 3;
    const barW = plotW / bins - gap;
    const barH = (count / maxCount) * plotH;
    ctx.fillStyle = index === counts.indexOf(maxCount) ? COLORS.accent : COLORS.accentSoft;
    ctx.fillRect(pad.left + index * (plotW / bins) + gap / 2, pad.top + plotH - barH, barW, barH);
  });
  if (threshold !== null && threshold >= min && threshold <= max) {
    const x = pad.left + ((threshold - min) / (max - min || 1)) * plotW;
    ctx.strokeStyle = "#b56b36";
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
    ctx.setLineDash([]);
  }
  canvas._histogram = { min, max, counts, pad, plotW, bins };
}

export function drawCategoryChart(canvas, entries, label) {
  const { ctx, width, height } = prepare(canvas);
  ctx.clearRect(0, 0, width, height);
  canvas._histogram = null;
  if (!entries.length) return;
  const pad = { left: 40, right: 12, top: 16, bottom: 40 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxCount = Math.max(...entries.map((entry) => entry.count));
  drawAxes(ctx, width, height, pad, "0", String(maxCount), label);
  entries.forEach((entry, index) => {
    const gap = 8;
    const slot = plotW / entries.length;
    const barW = Math.max(10, slot - gap);
    const barH = (entry.count / maxCount) * plotH;
    const x = pad.left + index * slot + gap / 2;
    ctx.fillStyle = COLORS.accentSoft;
    ctx.fillRect(x, pad.top + plotH - barH, barW, barH);
    ctx.fillStyle = COLORS.muted;
    ctx.font = "10px DM Mono, monospace";
    ctx.textAlign = "center";
    const name = String(entry.label).slice(0, 12);
    ctx.fillText(name, x + barW / 2, height - 20);
    ctx.fillText(String(entry.count), x + barW / 2, pad.top + plotH - barH - 5);
  });
  ctx.textAlign = "left";
}

function prepare(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(Number(canvas.getAttribute("height")) || rect.height || 220));
  const backingWidth = Math.max(1, Math.round(width * ratio));
  const backingHeight = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== backingWidth) canvas.width = backingWidth;
  if (canvas.height !== backingHeight) canvas.height = backingHeight;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return { ctx, width, height };
}

function drawAxes(ctx, width, height, pad, min, max, label) {
  ctx.strokeStyle = COLORS.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
  ctx.fillStyle = COLORS.muted;
  ctx.font = "11px DM Mono, monospace";
  ctx.fillText(min, pad.left, height - 12);
  ctx.textAlign = "right";
  ctx.fillText(max, width - pad.right, height - 12);
  ctx.textAlign = "center";
  ctx.fillText(label, pad.left + (width - pad.left - pad.right) / 2, height - 12);
  ctx.textAlign = "left";
}
