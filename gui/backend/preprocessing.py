from __future__ import annotations

from pathlib import Path


SOLVENT_OR_ION_RESNAMES = {
    "HOH", "WAT", "SOL", "DOD",
    "NA", "K", "CL", "CA", "MG", "MN", "ZN", "FE", "CU", "CO", "NI",
    "CD", "HG", "BR", "IOD", "SO4", "PO4", "NO3", "ACT", "EDO", "GOL",
}


def preprocess_complex_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Complex preprocessing payload must be a JSON object.")
    name = _safe_name(str(payload.get("name") or "complex.pdb"), "complex.pdb")
    text = str(payload.get("text") or "")
    if not text.strip():
        raise ValueError("A protein-ligand complex PDB is required.")
    if len(text.encode("utf-8")) > 50 * 1024 * 1024:
        raise ValueError("Complex PDB is larger than 50 MB.")
    if not looks_like_pdb(text):
        raise ValueError("Complex input does not look like a PDB file.")

    candidates = ligand_candidates_from_complex(text)
    if not candidates:
        raise ValueError("No ligand-like HETATM residues were detected in the complex PDB.")

    selected_id = str(payload.get("ligand_id") or "").strip()
    ligand_like = [item for item in candidates if item["kind"] == "ligand"]
    if not selected_id:
        if not ligand_like:
            raise ValueError("Only solvent or ion HETATM records were detected; no reference ligand candidate was found.")
        if len(ligand_like) == 1:
            selected_id = ligand_like[0]["id"]
        else:
            return {
                "status": "needs_ligand",
                "candidates": [_public_candidate(item) for item in candidates],
                "message": "Choose which ligand from the complex should define the reference pocket.",
            }

    selected = next((item for item in candidates if item["id"] == selected_id), None)
    if not selected:
        raise ValueError("Selected ligand was not found in the complex PDB.")

    protein_text = protein_from_complex(text)
    ligand_pdb = ligand_pdb_from_complex(text, selected)
    sdf_text = ligand_sdf_from_pdb(ligand_pdb, selected)
    stem = Path(name).stem or "complex"
    return {
        "status": "ready",
        "candidates": [_public_candidate(item) for item in candidates],
        "selected": _public_candidate(selected),
        "pdb": {
            "name": f"{stem}_protein.pdb",
            "text": protein_text,
        },
        "sdf": {
            "name": f"{stem}_{selected['resname']}_{selected['chain'] or 'chain'}_{selected['resseq']}.sdf",
            "text": sdf_text,
        },
    }


def looks_like_pdb(text: str) -> bool:
    for line in text.splitlines()[:200]:
        if line.startswith(("ATOM  ", "HETATM", "MODEL ", "HEADER", "CRYST1")):
            return True
    return False


def ligand_candidates_from_complex(text: str) -> list[dict]:
    groups: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip() or "UNK"
        chain = line[21:22].strip()
        resseq = line[22:26].strip()
        icode = line[26:27].strip()
        key = f"{resname}:{chain}:{resseq}:{icode}"
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        serial = line[6:11].strip()
        group = groups.setdefault(key, {
            "id": key,
            "resname": resname,
            "chain": chain,
            "resseq": resseq,
            "icode": icode,
            "atom_count": 0,
            "heavy_atom_count": 0,
            "serials": [],
        })
        group["atom_count"] += 1
        if element != "H":
            group["heavy_atom_count"] += 1
        if serial:
            group["serials"].append(serial)

    candidates = []
    for group in groups.values():
        is_solvent_or_ion = group["resname"].upper() in SOLVENT_OR_ION_RESNAMES or group["heavy_atom_count"] <= 1
        candidates.append({
            **group,
            "kind": "solvent_or_ion" if is_solvent_or_ion else "ligand",
            "label": ligand_candidate_label(group, is_solvent_or_ion),
        })
    candidates.sort(key=lambda item: (item["kind"] != "ligand", -item["heavy_atom_count"], item["resname"], item["chain"], item["resseq"]))
    return candidates


def ligand_candidate_label(group: dict, is_solvent_or_ion: bool) -> str:
    chain = group.get("chain") or "-"
    suffix = "solvent/ion" if is_solvent_or_ion else "ligand"
    return f"{group['resname']} chain {chain} residue {group['resseq']} ({group['heavy_atom_count']} heavy atoms, {suffix})"


def protein_from_complex(text: str) -> str:
    keep_prefixes = ("HEADER", "TITLE ", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR", "REVDAT", "JRNL  ", "REMARK", "CRYST1", "SCALE", "MODEL ", "ATOM  ", "TER", "ENDMDL")
    lines = [line for line in text.splitlines() if line.startswith(keep_prefixes)]
    if not any(line.startswith("ATOM  ") for line in lines):
        raise ValueError("The complex PDB does not contain protein ATOM records to stage for conDitar.")
    if not lines[-1].startswith("END"):
        lines.append("END")
    return "\n".join(lines) + "\n"


def ligand_pdb_from_complex(text: str, selected: dict) -> str:
    serials = set(selected.get("serials") or [])
    ligand_lines = []
    conect_lines = []
    for line in text.splitlines():
        if line.startswith("HETATM") and line[6:11].strip() in serials:
            ligand_lines.append(line)
        elif line.startswith("CONECT"):
            linked = [line[index:index + 5].strip() for index in range(6, len(line), 5)]
            if linked and linked[0] in serials:
                kept = [item for item in linked if item in serials]
                if len(kept) > 1:
                    conect_lines.append(f"CONECT{''.join(item.rjust(5) for item in kept)}")
    if not ligand_lines:
        raise ValueError("Selected ligand records could not be extracted from the complex PDB.")
    return "\n".join(ligand_lines + conect_lines + ["END"]) + "\n"


def ligand_sdf_from_pdb(ligand_pdb: str, selected: dict) -> str:
    try:
        from rdkit import Chem
    except ImportError as error:
        raise ValueError(
            "Complex splitting requires RDKit in the GUI environment. "
            "Install or start the GUI with the provided environment.yml."
        ) from error
    mol = Chem.MolFromPDBBlock(ligand_pdb, removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBBlock(ligand_pdb, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError("The selected ligand could not be converted from PDB records to SDF.")
    mol.SetProp("_Name", selected.get("label") or selected.get("resname") or "reference_ligand")
    return Chem.MolToMolBlock(mol) + "\n$$$$\n"


def _safe_name(name: str, fallback: str) -> str:
    cleaned = "".join(char for char in name if char.isalnum() or char in "._-")
    return cleaned or fallback


def _public_candidate(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key != "serials"}
