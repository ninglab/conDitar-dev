from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


PROPERTY_NAMES = ("LILLY_PASS", "LILLY_STATUS", "LILLY_REASONS", "LILLY_OUTPUT")


def describe() -> dict:
    command = _resolve_command()
    return {
        "id": "lilly_medchem",
        "name": "Lilly Medchem Rules",
        "description": "Annotates generated molecules with Lilly medicinal chemistry pass/fail rules.",
        "available": bool(command),
        "error": None if command else "Install the GUI environment from gui/environment.yml, or set LILLY_MEDCHEM_RULES_BIN.",
        "inputs": [],
        "outputs": [
            {"name": "LILLY_PASS", "label": "Lily Filter", "type": "boolean"},
            {"name": "LILLY_STATUS", "type": "text"},
            {"name": "LILLY_REASONS", "type": "text"},
        ],
    }


def run(job_root: str, run_root: str, options: dict) -> dict:
    job_path = Path(job_root)
    run_path = Path(run_root)
    command = _resolve_command()
    if not command:
        raise RuntimeError("Lilly Medchem Rules was not found. Install the GUI environment from gui/environment.yml or set LILLY_MEDCHEM_RULES_BIN.")

    sdf_paths = sorted((job_path / "outputs").rglob("*.sdf"))
    if not sdf_paths:
        raise RuntimeError("No generated SDF files were found for this job.")

    records = []
    input_smi = run_path / "lilly_input.smi"
    for index, sdf_path in enumerate(sdf_paths):
        text = sdf_path.read_text(errors="replace")
        smiles = _property(text, "SMILES")
        molecule_id = _safe_id(sdf_path, index)
        if smiles:
            records.append({"id": molecule_id, "smiles": smiles, "path": str(sdf_path)})
    if not records:
        raise RuntimeError("No SMILES properties were found in generated SDF files.")

    input_smi.write_text("".join(f"{item['smiles']} {item['id']}\n" for item in records))
    with (run_path / "lilly_stdout.smi").open("w") as stdout, (run_path / "lilly_stderr.log").open("w") as stderr:
        result = subprocess.run(_command_line(command, input_smi), cwd=str(run_path), text=True, stdout=stdout, stderr=stderr, check=False)

    passed, output_lines = _read_passed(_passed_output_path(command, run_path))
    failed = _read_failed(run_path)
    annotations = {}
    for item in records:
        molecule_id = item["id"]
        if molecule_id in passed:
            annotations[item["path"]] = {
                "LILLY_PASS": "true",
                "LILLY_STATUS": "pass",
                "LILLY_REASONS": "",
                "LILLY_OUTPUT": passed[molecule_id],
            }
        else:
            reasons = failed.get(molecule_id) or ["Rejected by Lilly Medchem Rules or not returned in the pass list."]
            annotations[item["path"]] = {
                "LILLY_PASS": "false",
                "LILLY_STATUS": "rejected",
                "LILLY_REASONS": "; ".join(reasons),
                "LILLY_OUTPUT": " | ".join(reasons),
            }

    for sdf_path_text, props in annotations.items():
        sdf_path = Path(sdf_path_text)
        sdf_path.write_text(_replace_properties(sdf_path.read_text(errors="replace"), props))

    summary = {
        "command": command,
        "exit_code": result.returncode,
        "molecules": len(records),
        "passed": sum(1 for props in annotations.values() if props["LILLY_PASS"] == "true"),
        "failed": sum(1 for props in annotations.values() if props["LILLY_PASS"] == "false"),
        "options": options,
    }
    summary["output_lines"] = output_lines
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2))
    if result.returncode != 0:
        raise RuntimeError(f"Lilly Medchem Rules exited with status {result.returncode}. See {run_path / 'lilly_stderr.log'}.")
    return summary


def _resolve_command() -> dict[str, str] | None:
    configured = os.environ.get("LILLY_MEDCHEM_RULES_BIN", "").strip()
    if configured:
        return {"kind": "wrapper", "path": configured} if Path(configured).exists() else None
    wrapper = shutil.which("Lilly_Medchem_Rules.sh") or shutil.which("Lilly_Medchem_Rules.rb")
    if wrapper:
        return {"kind": "wrapper", "path": wrapper}
    iwdemerit = shutil.which("iwdemerit")
    if iwdemerit:
        return {"kind": "iwdemerit", "path": iwdemerit}
    return None


def _command_line(command: dict[str, str], input_smi: Path) -> list[str]:
    if command["kind"] == "iwdemerit":
        return [command["path"], "-i", "smi", "-o", "smi", "-G", "good", "-R", "bad", str(input_smi)]
    return [command["path"], str(input_smi)]


def _passed_output_path(command: dict[str, str], run_path: Path) -> Path:
    return run_path / "good.smi" if command["kind"] == "iwdemerit" else run_path / "lilly_stdout.smi"


def _property(sdf_text: str, name: str) -> str:
    match = re.search(rf"^>\s*<{re.escape(name)}>[^\n]*\n(.*?)(?:\n\n|\Z)", sdf_text, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip().splitlines()[0] if match else ""


def _safe_id(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return f"mol_{index:04d}_{stem}"


def _read_passed(path: Path) -> tuple[dict[str, str], list[str]]:
    passed = {}
    lines = path.read_text(errors="replace").splitlines() if path.exists() else []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            passed[parts[1]] = line.strip()
    return passed, lines


def _read_failed(run_path: Path) -> dict[str, list[str]]:
    failed: dict[str, list[str]] = {}
    for path in sorted(run_path.glob("bad*.smi")):
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            molecule_id = parts[1]
            reason = line.split(molecule_id, 1)[1].strip(" :") if molecule_id in line else line
            failed.setdefault(molecule_id, []).append(reason or "Listed in bad.smi by Lilly Medchem Rules.")
    return failed


def _replace_properties(sdf_text: str, properties: dict[str, str]) -> str:
    text = sdf_text
    for name in PROPERTY_NAMES:
        text = re.sub(rf"\n?>\s*<{re.escape(name)}>\s*\n.*?(?=\n>\s*<|\nM  END|\n\$\$\$\$|\Z)", "\n", text, flags=re.DOTALL)
    block = "".join(f"> <{name}>\n{value}\n\n" for name, value in properties.items())
    if "$$$$" in text:
        return text.replace("$$$$", f"{block}$$$$", 1)
    return f"{text.rstrip()}\n{block}$$$$\n"
