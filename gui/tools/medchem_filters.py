from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable


FILTERS = [
    ("MEDCHEM_RO5_PASS", "Ro5", "rule", "Lipinski rule-of-five check: pass means MW <= 500, LogP <= 5, HBD <= 5, and HBA <= 10."),
    ("MEDCHEM_GHOSE_PASS", "Ghose", "rule", "Ghose drug-likeness check: pass means MW, LogP, atom count, and molar refractivity are within MedChem's Ghose ranges."),
    ("MEDCHEM_VEBER_PASS", "Veber", "rule", "Veber oral-drug-likeness check: pass means rotatable bonds <= 10 and TPSA < 140."),
    ("MEDCHEM_ZINC_PASS", "ZINC", "rule", "ZINC drug-likeness rule of thumb covering MW, LogP, HBD/HBA, TPSA, rotatable bonds, rings, carbon count, ratio, and charge limits."),
    ("MEDCHEM_BMS_ALERTS_PASS", "BMS Alerts", "functional", "Structural-alert filter using the BMS alert collection; pass means no matching BMS alert was found."),
    ("MEDCHEM_PAINS_ALERTS_PASS", "PAINS Alerts", "functional", "Structural-alert filter using the PAINS alert collection; pass means no matching PAINS alert was found."),
    ("MEDCHEM_SURECHEMBL_ALERTS_PASS", "SureChEMBL Alerts", "functional", "Structural-alert filter using the SureChEMBL alert collection; pass means no matching SureChEMBL alert was found."),
    ("MEDCHEM_NIBR_PASS", "NIBR", "functional", "Novartis screening-deck curation filter; pass means accumulated alert severity is below the configured cutoff."),
    ("MEDCHEM_COMPLEXITY_PASS", "Complexity", "functional", "Complexity filter using MedChem's configured Bertz complexity metric against ZINC reference statistics."),
    ("MEDCHEM_BREDT_PASS", "Bredt", "functional", "Bredt-rule filter; pass means the molecule does not violate Bredt's rules."),
    ("MEDCHEM_GRAPH_PASS", "Molecular Graph", "functional", "Unstable molecular-graph filter; pass means no disallowed graph pattern exceeds the configured severity cutoff."),
    ("MEDCHEM_LILLY_DEMERIT_PASS", "Lilly Demerit", "functional", "Eli Lilly demerit filter; pass means the molecule does not violate MedChem's Lilly demerit rules at the configured cutoff."),
]

PROPERTY_NAMES = [name for name, _, _, _ in FILTERS] + [
    "MEDCHEM_FILTERS_PASSED",
    "MEDCHEM_FILTERS_TOTAL",
    "MEDCHEM_STATUS",
    "MEDCHEM_FAILURES",
]


def describe() -> dict:
    available, error = _dependency_status()
    return {
        "id": "medchem_filters",
        "name": "MedChem Filters",
        "description": "Annotates generated molecules with the 12 medchem tutorial drug-likeness and alert filters.",
        "available": available,
        "error": error,
        "inputs": [],
        "outputs": [
            *[{"name": name, "label": label, "type": "boolean", "description": description, "viewer": "hidden"} for name, label, _, description in FILTERS],
            {
                "name": "MEDCHEM_FILTERS_PASSED",
                "label": "MedChem Passed",
                "type": "number",
                "description": "Number of MedChem filters passed by this molecule.",
                "viewer": "summary",
                "summary_label": "MedChem",
                "summary_total": "MEDCHEM_FILTERS_TOTAL",
                "summary_suffix": "passed",
            },
            {"name": "MEDCHEM_FILTERS_TOTAL", "label": "MedChem Total", "type": "number", "filterable": False},
            {"name": "MEDCHEM_STATUS", "type": "text"},
            {"name": "MEDCHEM_FAILURES", "type": "text", "filterable": False},
        ],
    }


def run(job_root: str, run_root: str, options: dict) -> dict:
    mc, chem = _import_dependencies()
    evaluators = _build_evaluators(mc)
    job_path = Path(job_root)
    run_path = Path(run_root)
    sdf_paths = sorted((job_path / "outputs").rglob("*.sdf"))
    if not sdf_paths:
        raise RuntimeError("No generated SDF files were found for this job.")

    records = []
    annotations = {}
    errors = {}
    for index, sdf_path in enumerate(sdf_paths):
        sdf_text = sdf_path.read_text(errors="replace")
        mol = _mol_from_sdf_text(sdf_text, chem)
        smiles = _property(sdf_text, "SMILES")
        if mol is None and smiles:
            mol = chem.MolFromSmiles(smiles)
        record_id = _safe_id(sdf_path, index)
        if mol is None:
            props = _invalid_properties("invalid molecule")
            annotations[str(sdf_path)] = props
            errors[record_id] = "Could not parse molecule from SDF block or SMILES property."
            continue

        results: dict[str, bool] = {}
        failures = []
        for name, label, _, _ in FILTERS:
            try:
                passed = bool(evaluators[name](mol))
            except Exception as error:  # Keep one problematic filter/molecule from killing the whole run.
                passed = False
                errors.setdefault(record_id, f"{label}: {error}")
            results[name] = passed
            if not passed:
                failures.append(label)

        passed_count = sum(1 for passed in results.values() if passed)
        props = {name: _bool_text(value) for name, value in results.items()}
        props.update({
            "MEDCHEM_FILTERS_PASSED": str(passed_count),
            "MEDCHEM_FILTERS_TOTAL": str(len(FILTERS)),
            "MEDCHEM_STATUS": "pass" if passed_count == len(FILTERS) else "fail",
            "MEDCHEM_FAILURES": "; ".join(failures),
        })
        annotations[str(sdf_path)] = props
        records.append({
            "id": record_id,
            "path": str(sdf_path),
            "passed": passed_count,
            "failed_filters": failures,
        })

    for sdf_path_text, props in annotations.items():
        sdf_path = Path(sdf_path_text)
        sdf_path.write_text(_replace_properties(sdf_path.read_text(errors="replace"), props))

    summary = {
        "molecules": len(sdf_paths),
        "parsed": len(records),
        "filters": [{"property": name, "label": label} for name, label, _, _ in FILTERS],
        "all_passed": sum(1 for item in records if item["passed"] == len(FILTERS)),
        "mean_filters_passed": round(sum(item["passed"] for item in records) / len(records), 3) if records else 0,
        "errors": errors,
        "options": options,
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _dependency_status() -> tuple[bool, str | None]:
    try:
        _import_dependencies()
        return True, None
    except Exception as error:
        return False, f"Install the GUI Tool Chest environment with ./setup_tool_chest.sh. Missing medchem dependency: {error}"


def _import_dependencies():
    import medchem as mc
    from rdkit import Chem

    return mc, Chem


def _build_evaluators(mc) -> dict[str, Callable]:
    def functional(fn: Callable, **kwargs) -> Callable:
        return lambda mol: _first_bool(fn(mols=[mol], n_jobs=1, progress=False, return_idx=False, **kwargs))

    return {
        "MEDCHEM_RO5_PASS": mc.rules.basic_rules.rule_of_five,
        "MEDCHEM_GHOSE_PASS": mc.rules.basic_rules.rule_of_ghose,
        "MEDCHEM_VEBER_PASS": mc.rules.basic_rules.rule_of_veber,
        "MEDCHEM_ZINC_PASS": mc.rules.basic_rules.rule_of_zinc,
        "MEDCHEM_BMS_ALERTS_PASS": functional(mc.functional.alert_filter, alerts=["BMS"]),
        "MEDCHEM_PAINS_ALERTS_PASS": functional(mc.functional.alert_filter, alerts=["PAINS"]),
        "MEDCHEM_SURECHEMBL_ALERTS_PASS": functional(mc.functional.alert_filter, alerts=["SureChEMBL"]),
        "MEDCHEM_NIBR_PASS": functional(mc.functional.nibr_filter),
        "MEDCHEM_COMPLEXITY_PASS": functional(mc.functional.complexity_filter, complexity_metric="bertz", threshold_stats_file="zinc_15_available"),
        "MEDCHEM_BREDT_PASS": functional(mc.functional.bredt_filter),
        "MEDCHEM_GRAPH_PASS": functional(mc.functional.molecular_graph_filter, max_severity=5),
        "MEDCHEM_LILLY_DEMERIT_PASS": functional(mc.functional.lilly_demerit_filter),
    }


def _first_bool(value) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else False
    return bool(value)


def _mol_from_sdf_text(sdf_text: str, chem):
    mol_block = sdf_text.split("$$$$", 1)[0]
    try:
        return chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=False)
    except Exception:
        return None


def _invalid_properties(reason: str) -> dict[str, str]:
    props = {name: "false" for name, _, _, _ in FILTERS}
    props.update({
        "MEDCHEM_FILTERS_PASSED": "0",
        "MEDCHEM_FILTERS_TOTAL": str(len(FILTERS)),
        "MEDCHEM_STATUS": "invalid",
        "MEDCHEM_FAILURES": reason,
    })
    return props


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _property(sdf_text: str, name: str) -> str:
    match = re.search(rf"^>\s*<{re.escape(name)}>[^\n]*\n(.*?)(?:\n\n|\Z)", sdf_text, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip().splitlines()[0] if match else ""


def _safe_id(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return f"mol_{index:04d}_{stem}"


def _replace_properties(sdf_text: str, properties: dict[str, str]) -> str:
    text = sdf_text
    for name in PROPERTY_NAMES:
        text = re.sub(rf"\n?>\s*<{re.escape(name)}>\s*\n.*?(?=\n>\s*<|\nM  END|\n\$\$\$\$|\Z)", "\n", text, flags=re.DOTALL)
    block = "".join(f"> <{name}>\n{value}\n\n" for name, value in properties.items())
    if "$$$$" in text:
        return text.replace("$$$$", f"{block}$$$$", 1)
    return f"{text.rstrip()}\n{block}$$$$\n"
