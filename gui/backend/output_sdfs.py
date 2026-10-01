from __future__ import annotations

import re
from pathlib import Path


def generated_sdf_paths(outputs: Path) -> list[Path]:
    if not outputs.exists():
        return []
    return [path for path in sorted(outputs.rglob("*.sdf")) if path.name.lower() != "reference.sdf"]


def sdf_records(text: str) -> list[str]:
    records = []
    for block in re.split(r"(?m)^\$\$\$\$[ \t]*(?:\r?\n|$)", text):
        if block.strip():
            records.append(block.rstrip() + "\n$$$$\n")
    return records
