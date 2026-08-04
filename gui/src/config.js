export const PARAMETERS = [
  { key: "num_samples", label: "Molecules", type: "number", value: 100, min: 1, max: 1000, step: 1, help: "Number of candidates to generate", tooltip: "Total number of molecule candidates requested for each input structure." },
  { key: "batch_size", label: "Batch size", type: "number", value: 100, min: 1, max: 500, step: 1, help: "Samples processed per batch", tooltip: "How many samples conDitar processes together internally. Larger values may use more memory." },
  { key: "pocket_radius", label: "Pocket radius", type: "number", value: 10, min: 4, max: 20, step: 1, suffix: "Å", help: "Protein context around the ligand", tooltip: "Radius around the reference ligand or prepared pocket used to define the protein context for generation." },
];

export const ADVANCED_PARAMETERS = [];

export const EXAMPLES = {};
