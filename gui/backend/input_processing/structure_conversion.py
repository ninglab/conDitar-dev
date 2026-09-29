from __future__ import annotations

from io import StringIO
from pathlib import Path

from .constants import SOLVENT_OR_ION_RESNAMES


def looks_like_cif(text: str) -> bool:
    for line in text.splitlines()[:200]:
        stripped = line.strip()
        if stripped.startswith(("data_", "_atom_site.", "loop_")):
            return True
    return False


def normalize_structure_payload(payload: dict, fallback_name: str = "input.pdb") -> tuple[dict, dict | None]:
    if not isinstance(payload, dict):
        raise ValueError("Structure payload must be a JSON object.")
    text = str(payload.get("text") or "")
    name = str(payload.get("name") or fallback_name)
    lower_name = name.lower()
    if not text.strip():
        raise ValueError("A structure file is required.")
    is_cif = lower_name.endswith((".cif", ".mmcif")) or looks_like_cif(text)
    if not is_cif:
        return payload, None

    pdb_text = cif_to_pdb(text, structure_id=Path(name).stem or "structure")
    pdb_name = f"{Path(name).stem or Path(fallback_name).stem or 'input'}.pdb"
    return {
        **payload,
        "name": pdb_name,
        "text": pdb_text,
    }, {
        "source_format": "cif",
        "source_name": name,
        "normalized_name": pdb_name,
    }


def clean_structure_payload(payload: dict) -> tuple[dict, dict | None]:
    if not isinstance(payload, dict):
        raise ValueError("Structure payload must be a JSON object.")
    cleaned_text, report = clean_pdb_solvent_records(str(payload.get("text") or ""))
    if not report:
        return payload, None
    return {**payload, "text": cleaned_text}, report


def clean_pdb_solvent_records(text: str) -> tuple[str, dict | None]:
    removed = {}
    kept_lines = []
    for line in str(text or "").splitlines():
        if line.startswith(("HETATM", "ANISOU")):
            resname = line[17:20].strip().upper()
            if resname in SOLVENT_OR_ION_RESNAMES:
                removed[resname] = removed.get(resname, 0) + 1
                continue
        kept_lines.append(line)
    if not removed:
        return text, None
    if kept_lines and not kept_lines[-1].startswith("END"):
        kept_lines.append("END")
    return "\n".join(kept_lines).rstrip() + "\n", {
        "method": "remove_solvent_and_ions",
        "removed_atom_count": sum(removed.values()),
        "removed_residue_names": removed,
    }


def cif_to_pdb(text: str, structure_id: str = "structure") -> str:
    try:
        from Bio.PDB import MMCIFParser, PDBIO
    except ImportError as error:
        raise ValueError(
            "CIF input requires Biopython in the GUI environment. "
            "Install the GUI environment or convert the structure to PDB first."
        ) from error

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_id, StringIO(text))
    except Exception as error:
        raise ValueError(f"CIF input could not be parsed: {error}") from error

    output = StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    try:
        writer.save(output)
    except Exception as error:
        raise ValueError(f"CIF input could not be converted to PDB: {error}") from error
    pdb_text = output.getvalue()
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in pdb_text.splitlines()):
        raise ValueError("CIF input did not contain ATOM/HETATM coordinates.")
    if not pdb_text.rstrip().endswith("END"):
        pdb_text = pdb_text.rstrip() + "\nEND\n"
    return pdb_text
