from __future__ import annotations

from typing import Any


DEFAULT_POCKET_RADIUS = 10.0
HEADER_PREFIXES = ("HEADER", "TITLE ", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR", "REVDAT", "JRNL  ", "REMARK", "CRYST1", "SCALE", "MODEL ")


def pocket_pdb_from_ligand(
    pdb_text: str,
    selected_ligand: dict[str, Any],
    radius: float = DEFAULT_POCKET_RADIUS,
) -> dict[str, Any]:
    ligand_serials = set(str(item) for item in selected_ligand.get("serials") or [])
    ligand_atoms = [
        atom for atom in _pdb_atoms(pdb_text, record_type="HETATM")
        if atom["serial"] in ligand_serials
    ]
    if not ligand_atoms:
        raise ValueError("Selected ligand has no coordinates for pocket extraction.")
    return pocket_pdb_from_points(
        pdb_text,
        [atom["coord"] for atom in ligand_atoms],
        radius=radius,
        method="bound_ligand_distance",
        provenance=f"Protein residues within {radius:g} A of selected ligand.",
    )


def pocket_pdb_from_center(
    pdb_text: str,
    center: dict[str, Any] | tuple[float, float, float] | list[float],
    radius: float = DEFAULT_POCKET_RADIUS,
) -> dict[str, Any]:
    point = _normalize_center(center)
    if point is None:
        raise ValueError("Pocket center must include numeric x, y, and z coordinates.")
    return pocket_pdb_from_points(
        pdb_text,
        [point],
        radius=radius,
        method="center_distance",
        provenance=f"Protein residues within {radius:g} A of selected pocket center.",
    )


def pocket_pdb_from_residues(
    pdb_text: str,
    residue_spec: str,
    radius: float | None = None,
) -> dict[str, Any]:
    requested = parse_residue_spec(residue_spec)
    if not requested:
        raise ValueError("Residues must use entries like A:45-62, A:88, or 45-62.")
    atoms = _pdb_atoms(pdb_text, record_type="ATOM")
    if not atoms:
        raise ValueError("Protein PDB contains no ATOM records for residue pocket extraction.")
    available = {atom["residue_key"] for atom in atoms}
    selected_residues = {key for key in requested if key in available}
    missing = [key for key in requested if key not in available]
    if not selected_residues:
        raise ValueError("None of the requested residues were found in the protein structure.")

    if radius and radius > 0:
        selected_points = [atom["coord"] for atom in atoms if atom["residue_key"] in selected_residues]
        expanded = {
            atom["residue_key"]
            for atom in atoms
            if _min_distance(atom["coord"], selected_points) <= radius
        }
        selected_residues = expanded

    pocket = pocket_pdb_from_residue_keys(
        pdb_text,
        selected_residues,
        method="residue_selection",
        provenance=(
            f"Protein residues selected from manual residue list plus residues within {radius:g} A."
            if radius and radius > 0
            else "Protein residues selected from manual residue list."
        ),
    )
    pocket["requested_residue_count"] = len(requested)
    pocket["matched_residue_count"] = len(set(requested) & available)
    pocket["missing_residues"] = [_format_residue_key(key) for key in missing]
    if missing:
        pocket.setdefault("warnings", []).append(
            f"{len(missing)} requested residue{'' if len(missing) == 1 else 's'} not found: "
            + ", ".join(_format_residue_key(key) for key in missing[:12])
            + ("..." if len(missing) > 12 else "")
        )
    return pocket


def pocket_pdb_from_points(
    pdb_text: str,
    points: list[tuple[float, float, float]],
    radius: float = DEFAULT_POCKET_RADIUS,
    method: str = "distance",
    provenance: str | None = None,
) -> dict[str, Any]:
    if radius <= 0:
        raise ValueError("Pocket radius must be positive.")
    atoms = _pdb_atoms(pdb_text, record_type="ATOM")
    if not atoms:
        raise ValueError("Protein PDB contains no ATOM records for pocket extraction.")
    selected_residues = {
        atom["residue_key"]
        for atom in atoms
        if _min_distance(atom["coord"], points) <= radius
    }
    if not selected_residues:
        raise ValueError("No protein residues were found within the requested pocket radius.")

    return pocket_pdb_from_residue_keys(
        pdb_text,
        selected_residues,
        method=method,
        provenance=provenance or f"Protein residues within {radius:g} A of selected points.",
        radius=radius,
    )


def pocket_pdb_from_residue_keys(
    pdb_text: str,
    selected_residues: set[tuple[str, str, str]],
    method: str = "residue_selection",
    provenance: str | None = None,
    radius: float | None = None,
) -> dict[str, Any]:
    atom_lines_by_residue: dict[tuple[str, str, str], list[tuple[dict[str, Any], str]]] = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM  "):
            atom = _parse_atom_line(line)
            if atom and atom["residue_key"] in selected_residues:
                atom_lines_by_residue.setdefault(atom["residue_key"], []).append((atom, line))

    complete_residues = []
    skipped_residue_count = 0
    for residue_key, atom_lines in atom_lines_by_residue.items():
        names = {atom["atom_name"] for atom, _ in atom_lines}
        if not {"N", "CA", "C", "O"}.issubset(names):
            skipped_residue_count += 1
            continue
        complete_residues.append((residue_key, sorted(atom_lines, key=lambda item: _atom_order(item[0]["atom_name"]))))
    complete_residues.sort(key=lambda item: _residue_order(item[0]))

    lines = [line for line in pdb_text.splitlines() if line.startswith(HEADER_PREFIXES)]
    for _, atom_lines in complete_residues:
        lines.extend(line for _, line in atom_lines)

    if not lines or not any(line.startswith("ATOM  ") for line in lines):
        raise ValueError("Pocket extraction produced no protein atoms.")
    if not lines[-1].startswith("END"):
        lines.append("END")
    selected_atom_count = sum(1 for line in lines if line.startswith("ATOM  "))
    kept_residues = {key for key, _ in complete_residues}
    chains = sorted({key[0] or "-" for key in kept_residues})
    warnings = []
    if skipped_residue_count:
        warnings.append(f"Skipped {skipped_residue_count} incomplete residue{'' if skipped_residue_count == 1 else 's'} without N/CA/C/O backbone atoms.")
    if len(kept_residues) < 10:
        warnings.append(
            "Pocket contains fewer than 10 complete residues; generation and docking scores may be unreliable."
        )
    elif len(kept_residues) < 20:
        warnings.append(
            "Pocket contains fewer than 20 complete residues; expect more variability and review results carefully."
        )
    return {
        "text": "\n".join(lines) + "\n",
        "method": method,
        "radius": radius,
        "residue_count": len(kept_residues),
        "atom_count": selected_atom_count,
        "chains": chains,
        "residues": [_format_residue_key(key) for key in sorted(kept_residues, key=_residue_order)],
        "provenance": provenance or "Protein residues selected from residue keys.",
        "warnings": warnings,
    }


def parse_residue_spec(spec: str) -> list[tuple[str, str, str]]:
    residues: list[tuple[str, str, str]] = []
    seen = set()
    default_chain = ""
    for raw_part in str(spec or "").replace(";", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        chain = default_chain
        residue_part = part
        if ":" in part:
            chain, residue_part = part.split(":", 1)
            chain = chain.strip()
            default_chain = chain
        residue_part = residue_part.strip()
        if not residue_part:
            continue
        for residue in _expand_residue_token(chain, residue_part):
            if residue not in seen:
                seen.add(residue)
                residues.append(residue)
    return residues


def _expand_residue_token(chain: str, token: str) -> list[tuple[str, str, str]]:
    if "-" in token:
        start, end = [item.strip() for item in token.split("-", 1)]
        if start.lstrip("-").isdigit() and end.lstrip("-").isdigit():
            start_number = int(start)
            end_number = int(end)
            step = 1 if end_number >= start_number else -1
            return [(chain, str(number), "") for number in range(start_number, end_number + step, step)]
    return [(chain, token, "")]


def _pdb_atoms(pdb_text: str, record_type: str) -> list[dict[str, Any]]:
    return [
        atom for atom in (_parse_atom_line(line) for line in pdb_text.splitlines() if line.startswith(record_type))
        if atom is not None
    ]


def _parse_atom_line(line: str) -> dict[str, Any] | None:
    try:
        chain = line[21:22].strip()
        resseq = line[22:26].strip()
        icode = line[26:27].strip()
        return {
            "serial": line[6:11].strip(),
            "atom_name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": chain,
            "resseq": resseq,
            "icode": icode,
            "residue_key": (chain, resseq, icode),
            "coord": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        }
    except (IndexError, ValueError):
        return None


def _atom_order(atom_name: str) -> tuple[int, str]:
    backbone_order = {"N": 0, "CA": 1, "C": 2, "O": 3}
    return (backbone_order.get(atom_name, 10), atom_name)


def _residue_order(key: tuple[str, str, str]) -> tuple[str, int, str]:
    chain, resseq, icode = key
    try:
        number = int(resseq)
    except ValueError:
        number = 0
    return (chain, number, icode)


def _format_residue_key(key: tuple[str, str, str]) -> str:
    chain, resseq, icode = key
    prefix = f"{chain}:" if chain else ""
    return f"{prefix}{resseq}{icode or ''}"


def _min_distance(point: tuple[float, float, float], others: list[tuple[float, float, float]]) -> float:
    return min(_distance(point, other) for other in others)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _normalize_center(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        items = (value.get("x"), value.get("y"), value.get("z"))
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        items = (value[0], value[1], value[2])
    else:
        return None
    try:
        return (float(items[0]), float(items[1]), float(items[2]))
    except (TypeError, ValueError):
        return None
