export const PARAMETERS = [
  { key: "num_samples", label: "Molecules", type: "number", value: 100, min: 1, max: 1000, step: 1, help: "Number of candidates to generate" },
  { key: "batch_size", label: "Batch size", type: "number", value: 100, min: 1, max: 500, step: 1, help: "Samples processed per batch" },
  { key: "pocket_radius", label: "Pocket radius", type: "number", value: 10, min: 4, max: 20, step: 1, suffix: "Å", help: "Protein context around the ligand" },
];

export const ADVANCED_PARAMETERS = [];

export const EXAMPLES = {};
