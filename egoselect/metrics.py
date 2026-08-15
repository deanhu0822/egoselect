"""Proxy metrics over a kept subset. SQL labels are retrospective only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from egoselect.features import MOTION_COLUMNS
from egoselect.scoring import l2_normalize


def _subset(features: pd.DataFrame, hashes: list[str]) -> pd.DataFrame:
    idx = features.set_index("episode_hash").loc[list(hashes)]
    return idx.reset_index()


def region_coverage(features: pd.DataFrame, hashes: list[str]) -> float:
    all_r = set(features["behavioral_region"].astype(int))
    kept = _subset(features, hashes)
    hit = set(kept["behavioral_region"].astype(int))
    return float(len(hit) / max(len(all_r), 1))


def average_quality(features: pd.DataFrame, hashes: list[str]) -> float:
    kept = _subset(features, hashes)
    return float(kept["quality_score"].mean())


def nn_redundancy(features: pd.DataFrame, hashes: list[str]) -> float:
    kept = _subset(features, hashes)
    if len(kept) < 2:
        return 0.0
    z = l2_normalize(np.stack(kept["z"].to_numpy()).astype(np.float64))
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    return float(sim.max(axis=1).mean())


def stationary_content(features: pd.DataFrame, hashes: list[str]) -> float:
    kept = _subset(features, hashes)
    return float(kept["stationary_ratio"].mean())


def _mean_max_cosine(universe: np.ndarray, kept: np.ndarray) -> float:
    if len(kept) == 0:
        return 0.0
    u = l2_normalize(universe.astype(np.float64))
    k = l2_normalize(kept.astype(np.float64))
    return float((u @ k.T).max(axis=1).mean())


def visual_coverage(features: pd.DataFrame, hashes: list[str]) -> float:
    vis_all = np.stack(features["vis_emb"].to_numpy())
    kept = _subset(features, hashes)
    vis_keep = np.stack(kept["vis_emb"].to_numpy())
    return _mean_max_cosine(vis_all, vis_keep)


def motion_coverage(features: pd.DataFrame, hashes: list[str]) -> float:
    mot_all = features.loc[:, list(MOTION_COLUMNS)].to_numpy(dtype=np.float64)
    kept = _subset(features, hashes)
    mot_keep = kept.loc[:, list(MOTION_COLUMNS)].to_numpy(dtype=np.float64)
    return _mean_max_cosine(mot_all, mot_keep)


def _label_diversity(features: pd.DataFrame, hashes: list[str], col: str) -> float | None:
    if col not in features.columns:
        return None
    universe = features[col].dropna().astype(str)
    universe = universe[universe.str.strip() != ""]
    if universe.empty:
        return None
    kept = _subset(features, hashes)[col].dropna().astype(str)
    kept = kept[kept.str.strip() != ""]
    return float(kept.nunique() / universe.nunique())


def evaluate_keep(
    features: pd.DataFrame,
    hashes: list[str],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_keep": len(hashes),
        "behavioral_region_coverage": region_coverage(features, hashes),
        "average_quality": average_quality(features, hashes),
        "nn_redundancy": nn_redundancy(features, hashes),
        "stationary_content_ratio": stationary_content(features, hashes),
        "visual_coverage": visual_coverage(features, hashes),
        "motion_coverage": motion_coverage(features, hashes),
        "task_diversity": _label_diversity(features, hashes, "task"),
        "scene_diversity": _label_diversity(features, hashes, "scene"),
        "lab_diversity": _label_diversity(features, hashes, "lab"),
        "operator_diversity": _label_diversity(features, hashes, "operator"),
    }
    if extra:
        out.update(extra)
    return out
