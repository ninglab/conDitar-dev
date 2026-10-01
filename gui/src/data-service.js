import { EXAMPLES } from "./config.js?v=20260723-theme-1";
import { candidateId, parseSdf, splitSdfRecords } from "./sdf.js?v=20260723-theme-1";

export class ExampleDataService {
  async loadStudy(exampleId, onProgress = () => {}) {
    const example = EXAMPLES[exampleId];
    if (!example) throw new Error(`Unknown example: ${exampleId}`);

    const [pdbText, referenceSdf] = await Promise.all([
      fetchText(example.pdb),
      example.sdf ? fetchText(example.sdf) : Promise.resolve(null),
    ]);

    let loaded = 0;
    const candidates = [];
    const indexes = Array.from({ length: example.count }, (_, index) => index);
    for (let offset = 0; offset < indexes.length; offset += 12) {
      const batch = indexes.slice(offset, offset + 12);
      const batchCandidates = await Promise.all(batch.map(async (index) => {
        const preferred = `${example.outputRoot}/${example.outputStem}${index}.sdf`;
        const fallback = example.outputFallbackStem
          ? `${example.outputRoot}/${example.outputFallbackStem}${index}.sdf`
          : null;
        let path = preferred;
        let text = await fetchText(preferred, false);
        if (!text && fallback) {
          path = fallback;
          text = await fetchText(fallback, false);
        }
        loaded += 1;
        onProgress(loaded, example.count);
        if (!text) return [];
        const records = splitSdfRecords(text);
        return records.map((record, recordIndex) => {
          const name = records.length === 1
            ? path.split("/").pop()
            : `${path.split("/").pop().replace(/\.sdf$/i, "")}_record_${recordIndex}.sdf`;
          return { ...parseSdf(record, name), path };
        });
      }));
      const available = batchCandidates.flat();
      const profiles = await this.chemistryProfiles(available.map((item) => item.text));
      candidates.push(...available.map((item, index) => ({
        ...withChemistry(item, profiles[index]),
        index: candidates.length + index,
        id: candidateId(candidates.length + index),
      })));
      await yieldToBrowser();
    }
    candidates.sort((a, b) => a.index - b.index);

    return { example, pdbText, referenceSdf, candidates };
  }

  async loadUploadedOutputs(files) {
    const fileCandidates = await Promise.all([...files].map(async (file) => {
      const text = await file.text();
      const records = splitSdfRecords(text);
      return records.map((record, recordIndex) => {
        const name = records.length === 1 ? file.name : `${file.name.replace(/\.sdf$/i, "")}_record_${recordIndex}.sdf`;
        return { ...parseSdf(record, name), path: file.name };
      });
    }));
    const candidates = fileCandidates.flat();
    const profiles = await this.chemistryProfiles(candidates.map((item) => item.text));
    return candidates.map((item, index) => ({
      ...withChemistry(item, profiles[index]),
      index,
      id: candidateId(index),
    }));
  }

  async chemistryProfiles(sdfs) {
    const profiles = [];
    for (let offset = 0; offset < sdfs.length; offset += 20) {
      const response = await fetchJson("/api/molecules/chemistry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdfs: sdfs.slice(offset, offset + 20) }),
      });
      profiles.push(...response.profiles);
    }
    return profiles;
  }

  async submitJob(payload) {
    const response = await fetchJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return response.job;
  }

  async submitBatch(payload) {
    return fetchJson("/api/jobs/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async listJobs() {
    const response = await fetchJson("/api/jobs");
    return response.jobs;
  }

  async health() {
    return fetchJson("/api/health");
  }

  async getJob(jobId) {
    const response = await fetchJson(`/api/jobs/${jobId}`);
    return response.job;
  }

  async getJobLogs(jobId) {
    return fetchJson(`/api/jobs/${jobId}/logs`);
  }

  async cancelJob(jobId) {
    return fetchJson(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  }

  async loadJobResults(job) {
    const response = await fetchJson(`/api/jobs/${job.id}/results`);
    return {
      job: response.job || job,
      inputs: response.inputs || {},
      artifacts: response.artifacts || [],
      logs: response.logs || {},
      summary: response.summary || {},
      toolRuns: response.tool_runs || [],
      candidates: (response.files || []).map((file, index) => ({
        ...withChemistry(parseSdf(file.text, file.name), file.chemistry),
        index,
        id: candidateId(index),
        path: file.relative_path,
        sha256: file.sha256 || null,
      })),
    };
  }

  async exportJob(jobId, payload = {}) {
    return fetchJson(`/api/jobs/${jobId}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async archiveJob(jobId) {
    return fetchJson(`/api/jobs/${jobId}/archive`, { method: "POST" });
  }

  async rerunJob(jobId) {
    return fetchJson(`/api/jobs/${jobId}/rerun`, { method: "POST" });
  }

  async listTools() {
    const response = await fetchJson("/api/tools");
    return response.tools || [];
  }

  async runTool(jobId, toolId, options = {}) {
    return fetchJson(`/api/jobs/${jobId}/tools/${toolId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    });
  }

  async preprocessComplex(payload) {
    return fetchJson("/api/preprocess/complex", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async preprocessDockedPockets(payload) {
    return fetchJson("/api/preprocess/pockets/docked", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async preprocessVinaPanelPockets(payload) {
    return fetchJson("/api/preprocess/pockets/vina-panel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async preprocessPocketFromCenter(payload) {
    return fetchJson("/api/preprocess/pockets/center", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async preprocessPocketFromResidues(payload) {
    return fetchJson("/api/preprocess/pockets/residues", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }
}

function withChemistry(candidate, chemistry) {
  return {
    ...candidate,
    molecularWeight: Number.isFinite(chemistry?.molecular_weight) ? chemistry.molecular_weight : null,
    formula: chemistry?.formula || null,
  };
}

async function fetchText(path, required = true) {
  const response = await fetch(path);
  if (!response.ok) {
    if (required) throw new Error(`Unable to load ${path}`);
    return null;
  }
  return response.text();
}

function yieldToBrowser() {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => resolve());
      return;
    }
    setTimeout(resolve, 0);
  });
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed: ${path}`);
  return body;
}
