from __future__ import annotations

import math
from statistics import median
from typing import Any


DEFAULT_CLUSTER_DISTANCE = 6.0
DEFAULT_POCKET_RADIUS = 10.0


def pocket_candidates_from_docked_sdfs(
    docked_sdfs: list[dict[str, Any]],
    cluster_distance: float = DEFAULT_CLUSTER_DISTANCE,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
) -> dict[str, Any]:
    """Cluster docked ligand poses into candidate pocket regions.

    This does not run docking. It deterministically summarizes docking output
    from a panel of ligands into reviewable pocket candidates.
    """
    poses = []
    for index, item in enumerate(docked_sdfs):
        name = str(item.get("name") or f"pose_{index + 1}.sdf")
        text = str(item.get("text") or "")
        score = _coerce_optional_float(item.get("score"))
        for pose_index, block in enumerate(_sdf_blocks(text), start=1):
            atoms = _mol_block_atoms(block)
            if not atoms:
                continue
            centroid = _centroid(atoms)
            block_score = _coerce_optional_float(_sdf_property(block, "VINA_DOCK"))
            if block_score is None:
                block_score = _coerce_optional_float(_sdf_property(block, "VINA_MINIMIZE"))
            if block_score is None:
                block_score = _coerce_optional_float(_sdf_property(block, "VINA_SCORE_ONLY"))
            poses.append({
                "id": f"{name}:{pose_index}",
                "name": name,
                "pose_index": pose_index,
                "center": centroid,
                "score": block_score if block_score is not None else score,
                "atom_count": len(atoms),
            })

    clusters: list[dict[str, Any]] = []
    for pose in poses:
        cluster = _nearest_cluster(pose["center"], clusters, cluster_distance)
        if cluster is None:
            clusters.append({
                "poses": [pose],
                "center": pose["center"],
            })
        else:
            cluster["poses"].append(pose)
            cluster["center"] = _mean_point([item["center"] for item in cluster["poses"]])

    candidates = []
    for index, cluster in enumerate(clusters, start=1):
        scores = [pose["score"] for pose in cluster["poses"] if pose["score"] is not None]
        best_score = min(scores) if scores else None
        median_score = median(scores) if scores else None
        candidates.append({
            "id": f"docking_cluster_{index}",
            "label": f"Docking cluster {index}",
            "method": "docking_pose_cluster",
            "center": _round_point(cluster["center"]),
            "radius": pocket_radius,
            "supporting_pose_count": len(cluster["poses"]),
            "supporting_ligand_count": len({pose["name"] for pose in cluster["poses"]}),
            "best_score": best_score,
            "median_score": median_score,
            "pose_ids": [pose["id"] for pose in cluster["poses"]],
            "provenance": "Clustered from docked ligand pose centroids.",
            "warnings": [],
        })
    candidates.sort(key=lambda item: (-item["supporting_ligand_count"], -item["supporting_pose_count"], item["best_score"] if item["best_score"] is not None else math.inf))
    return {
        "status": "ready" if candidates else "no_candidates",
        "method": "docking_pose_cluster",
        "candidates": candidates,
        "warnings": [] if poses else ["No docked ligand coordinates were found."],
    }


def pocket_candidates_from_pose_centers(
    poses: list[dict[str, Any]],
    cluster_distance: float = DEFAULT_CLUSTER_DISTANCE,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
    method: str = "vina_panel_pose_cluster",
) -> dict[str, Any]:
    valid_poses = []
    for index, pose in enumerate(poses):
        center = _normalize_center(pose.get("center"))
        if center is None:
            continue
        valid_poses.append({
            "id": str(pose.get("id") or f"pose_{index + 1}"),
            "name": str(pose.get("name") or pose.get("ligand") or f"ligand_{index + 1}"),
            "center": center,
            "score": _coerce_optional_float(pose.get("score")),
            "sdf": str(pose.get("sdf") or ""),
        })

    clusters: list[dict[str, Any]] = []
    for pose in valid_poses:
        cluster = _nearest_cluster(pose["center"], clusters, cluster_distance)
        if cluster is None:
            clusters.append({
                "poses": [pose],
                "center": pose["center"],
            })
        else:
            cluster["poses"].append(pose)
            cluster["center"] = _mean_point([item["center"] for item in cluster["poses"]])

    candidates = []
    for index, cluster in enumerate(clusters, start=1):
        scores = [pose["score"] for pose in cluster["poses"] if pose["score"] is not None]
        representative = _representative_pose(cluster["poses"])
        candidates.append({
            "id": f"{method}_{index}",
            "label": f"Vina panel cluster {index}" if method.startswith("vina") else f"Pose cluster {index}",
            "method": method,
            "center": _round_point(cluster["center"]),
            "radius": pocket_radius,
            "supporting_pose_count": len(cluster["poses"]),
            "supporting_ligand_count": len({pose["name"] for pose in cluster["poses"]}),
            "best_score": min(scores) if scores else None,
            "median_score": median(scores) if scores else None,
            "pose_ids": [pose["id"] for pose in cluster["poses"]],
            "representative_pose": {
                "id": representative["id"],
                "name": representative["name"],
                "score": representative["score"],
                "sdf": representative["sdf"],
            } if representative else None,
            "provenance": "Clustered from docking-panel pose centers.",
            "warnings": [],
        })
    candidates.sort(key=lambda item: (-item["supporting_ligand_count"], -item["supporting_pose_count"], item["best_score"] if item["best_score"] is not None else math.inf))
    return {
        "status": "ready" if candidates else "no_candidates",
        "method": method,
        "candidates": candidates,
        "warnings": [] if valid_poses else ["No valid docked pose centers were found."],
    }


def _sdf_blocks(text: str) -> list[str]:
    return [block.strip("\n") for block in text.split("$$$$") if block.strip()]


def _mol_block_atoms(block: str) -> list[tuple[float, float, float]]:
    lines = block.splitlines()
    if len(lines) < 4:
        return []
    try:
        atom_count = int(lines[3][0:3])
    except ValueError:
        return []
    atoms = []
    for line in lines[4:4 + atom_count]:
        try:
            atoms.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))
        except ValueError:
            continue
    return atoms


def _sdf_property(block: str, key: str) -> str | None:
    lines = block.splitlines()
    header = f">  <{key}>"
    for index, line in enumerate(lines):
        if line.strip() == header and index + 1 < len(lines):
            return lines[index + 1].strip()
    return None


def _centroid(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return _mean_point(points)


def _mean_point(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = len(points) or 1
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _nearest_cluster(
    center: tuple[float, float, float],
    clusters: list[dict[str, Any]],
    max_distance: float,
) -> dict[str, Any] | None:
    nearest = None
    nearest_distance = math.inf
    for cluster in clusters:
        distance = _distance(center, cluster["center"])
        if distance <= max_distance and distance < nearest_distance:
            nearest = cluster
            nearest_distance = distance
    return nearest


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _round_point(point: tuple[float, float, float]) -> dict[str, float]:
    return {
        "x": round(point[0], 3),
        "y": round(point[1], 3),
        "z": round(point[2], 3),
    }


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _representative_pose(poses: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not poses:
        return None
    return sorted(poses, key=lambda pose: pose["score"] if pose["score"] is not None else math.inf)[0]


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
