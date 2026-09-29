from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CONDITAR_ENGINE = "conditar"
DIFFSMOL_ENGINE = "diffsmol"
VINA_METRICS = {"vina_score", "vina_dock", "qvina"}
CHEMISTRY_METRICS = {"qed", "sa", "logp", "lipinski"}


@dataclass(frozen=True)
class WorkflowRule:
    id: str
    engine: str
    mode: str
    label: str
    pdb: str
    sdf: str
    allow_vina: bool
    allow_chemistry: bool = True


WORKFLOWS = {
    "conditar_pocket": WorkflowRule(
        id="conditar_pocket",
        engine=CONDITAR_ENGINE,
        mode="pocket",
        label="conDitar pocket",
        pdb="required",
        sdf="none",
        allow_vina=True,
    ),
    "conditar_reference": WorkflowRule(
        id="conditar_reference",
        engine=CONDITAR_ENGINE,
        mode="reference",
        label="conDitar protein + ligand",
        pdb="required",
        sdf="required",
        allow_vina=True,
    ),
    "diffsmol_ligand": WorkflowRule(
        id="diffsmol_ligand",
        engine=DIFFSMOL_ENGINE,
        mode="reference",
        label="DiffSMol ligand only",
        pdb="optional",
        sdf="required",
        allow_vina=False,
    ),
    "diffsmol_pocket": WorkflowRule(
        id="diffsmol_pocket",
        engine=DIFFSMOL_ENGINE,
        mode="reference",
        label="DiffSMol ligand + pocket",
        pdb="optional",
        sdf="required",
        allow_vina=True,
    ),
}


def resolve_workflow(engine: str, mode: str, has_pdb: bool = False) -> WorkflowRule:
    if engine == DIFFSMOL_ENGINE:
        return WORKFLOWS["diffsmol_pocket"] if has_pdb else WORKFLOWS["diffsmol_ligand"]
    return WORKFLOWS["conditar_pocket"] if mode == "pocket" else WORKFLOWS["conditar_reference"]


def validate_generation_inputs(engine: str, mode: str, has_pdb: bool, has_sdf: bool) -> list[str]:
    workflow = resolve_workflow(engine, mode, has_pdb=has_pdb)
    errors: list[str] = []
    if workflow.engine == DIFFSMOL_ENGINE:
        if not has_sdf:
            errors.append("DiffSMol requires a 3D reference ligand SDF input.")
        return errors
    if not has_pdb:
        errors.append("A PDB or CIF input is required.")
    if workflow.mode == "reference" and not has_sdf:
        errors.append("Reference mode requires an SDF ligand input.")
    return errors


def filter_postprocess_for_workflow(engine: str, mode: str, pdb_path: Path | None, postprocess: dict) -> dict:
    workflow = resolve_workflow(engine, mode, has_pdb=bool(pdb_path))
    if workflow.allow_vina:
        return postprocess
    cleaned = dict(postprocess)
    original_metrics = [str(item) for item in cleaned.get("metrics", [])]
    requested_vina = bool(cleaned.get("vina")) or any(item in VINA_METRICS for item in original_metrics)
    cleaned["metrics"] = [item for item in original_metrics if item not in VINA_METRICS]
    cleaned["vina"] = False
    cleaned["vina_mode"] = "none"
    if requested_vina:
        cleaned["vina_skipped_reason"] = "Pocket PDB was not provided; Vina/QVina requires a pocket."
    return cleaned
