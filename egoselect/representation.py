"""Multimodal representation z_i, 2D PCA layout, and coverage regions.

Behavioral / coverage / representation-space regions are KMeans partitions of
z_i. They are not ground-truth semantic skills.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from egoselect.features import MOTION_COLUMNS, QUALITY_COLUMNS

VIS_PCA_DIM = 16
N_REGIONS = 6
Z_RANDOM_STATE = 42


def _stack_vis(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([np.asarray(r["vis_emb"], dtype=np.float32) for r in rows])


def _motion_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(r[c]) for c in MOTION_COLUMNS] for r in rows], dtype=np.float64
    )


def fit_representation(
    rows: list[dict[str, Any]],
    *,
    vis_pca_dim: int = VIS_PCA_DIM,
    n_regions: int = N_REGIONS,
    random_state: int = Z_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(rows)
    if n == 0:
        raise ValueError("No feature rows to represent")

    vis = _stack_vis(rows)
    motion = _motion_matrix(rows)
    if not np.isfinite(motion).all():
        bad = ~np.isfinite(motion)
        raise ValueError(f"Motion features contain NaN/Inf at {int(bad.sum())} cells")

    vis_k = int(min(vis_pca_dim, n - 1, vis.shape[1]))
    vis_pca = PCA(n_components=vis_k, random_state=random_state)
    vis_reduced = vis_pca.fit_transform(vis)
    vis_reduced = StandardScaler().fit_transform(vis_reduced)
    motion_std = StandardScaler().fit_transform(motion)
    z = np.concatenate([vis_reduced, motion_std], axis=1)
    z = StandardScaler().fit_transform(z)

    xy_pca = PCA(n_components=2, random_state=random_state)
    xy = xy_pca.fit_transform(z)

    k = int(min(n_regions, n))
    regions = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(z)

    records = []
    for i, row in enumerate(rows):
        rec = {
            "episode_hash": row["episode_hash"],
            "visual_model": row["visual_model"],
            "visual_dim": int(row["visual_dim"]),
            "visual_pca_dim": vis_k,
            "n_rgb_sampled": int(row["n_rgb_sampled"]),
            "n_rgb_decoded": int(row["n_rgb_decoded"]),
            "vis_emb": np.asarray(row["vis_emb"], dtype=np.float32),
            "z": z[i].astype(np.float32),
            "pca_x": float(xy[i, 0]),
            "pca_y": float(xy[i, 1]),
            "behavioral_region": int(regions[i]),
            "z_dim": int(z.shape[1]),
            "n_regions": k,
        }
        for col in MOTION_COLUMNS:
            rec[col] = float(row[col])
        for col in QUALITY_COLUMNS:
            rec[col] = float(row[col])
        rec["fps"] = float(row.get("fps") or 0.0)
        records.append(rec)

    meta = {
        "n_episodes": n,
        "visual_dim": int(vis.shape[1]),
        "visual_pca_dim": vis_k,
        "visual_pca_explained_variance_ratio": vis_pca.explained_variance_ratio_.tolist(),
        "motion_columns": list(MOTION_COLUMNS),
        "z_dim": int(z.shape[1]),
        "xy_explained_variance_ratio": xy_pca.explained_variance_ratio_.tolist(),
        "n_regions": k,
        "region_counts": {int(i): int((regions == i).sum()) for i in range(k)},
        "interaction": "not used",
    }
    return pd.DataFrame.from_records(records), meta
