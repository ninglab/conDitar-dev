from __future__ import annotations

import json
import re
from pathlib import Path


# Copy this file to a new name such as my_filter.py.
# Files that start with "_" are ignored by the GUI.
#
# Then edit:
# 1. TOOL_ID, TOOL_NAME, and OUTPUTS
# 2. calculate_properties()
#
# The GUI will find the new file after restart/refresh and will show the
# outputs as filterable metrics in Results.

TOOL_ID = "my_filter"
TOOL_NAME = "My Filter"
OUTPUTS = [
    {"name": "MY_FILTER_PASS", "label": "My Filter", "type": "boolean"},
    {"name": "MY_FILTER_SCORE", "label": "My Score", "type": "number"},
    {"name": "MY_FILTER_NOTE", "label": "My Note", "type": "text", "filterable": False},
]


def describe() -> dict:
    return {
        "id": TOOL_ID,
        "name": TOOL_NAME,
        "description": "Example molecule evaluator template.",
        "available": True,
        "inputs": [],
        "outputs": OUTPUTS,
    }


def run(job_root: str, run_root: str, options: dict) -> dict:
    job_path = Path(job_root)
    run_path = Path(run_root)
    sdf_paths = sorted((job_path / "outputs").rglob("*.sdf"))
    if not sdf_paths:
        raise RuntimeError("No generated SDF files were found for this job.")

    annotated = 0
    passed = 0
    for sdf_path in sdf_paths:
        sdf_text = sdf_path.read_text(errors="replace")
        properties = calculate_properties(sdf_text, options)
        sdf_path.write_text(replace_sdf_properties(sdf_text, properties))
        annotated += 1
        if properties.get("MY_FILTER_PASS") == "true":
            passed += 1

    summary = {
        "molecules": annotated,
        "passed": passed,
        "failed": annotated - passed,
        "options": options,
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def calculate_properties(sdf_text: str, options: dict) -> dict[str, str]:
    """Replace this example with your own scoring/filtering code."""
    smiles = sdf_property(sdf_text, "SMILES")
    score = len(smiles)
    threshold = float(options.get("max_smiles_length", 80))
    passes = score <= threshold
    return {
        "MY_FILTER_PASS": "true" if passes else "false",
        "MY_FILTER_SCORE": f"{score:.0f}",
        "MY_FILTER_NOTE": "Example score is SMILES length.",
    }


def sdf_property(sdf_text: str, name: str) -> str:
    match = re.search(rf"^>\s*<{re.escape(name)}>[^\n]*\n(.*?)(?:\n\n|\Z)", sdf_text, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip().splitlines()[0] if match else ""


def replace_sdf_properties(sdf_text: str, properties: dict[str, str]) -> str:
    text = sdf_text
    for name in properties:
        text = re.sub(rf"\n?>\s*<{re.escape(name)}>\s*\n.*?(?=\n>\s*<|\nM  END|\n\$\$\$\$|\Z)", "\n", text, flags=re.DOTALL)
    block = "".join(f"> <{name}>\n{value}\n\n" for name, value in properties.items())
    if "$$$$" in text:
        return text.replace("$$$$", f"{block}$$$$", 1)
    return f"{text.rstrip()}\n{block}$$$$\n"
