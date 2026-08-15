"""Feature-layer corruption. Does not modify original EgoVerse files."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clone_row(row: pd.Series, new_hash: str) -> dict:
    rec = row.to_dict()
    rec["episode_hash"] = new_hash
    rec["z"] = np.asarray(row["z"], dtype=np.float32).copy()
    rec["vis_emb"] = np.asarray(row["vis_emb"], dtype=np.float32).copy()
    rec["is_corruption"] = True
    rec["source_episode"] = str(row["episode_hash"])
    return rec


def _split_counts(k: int) -> tuple[int, int, int]:
    a, b, c = k // 3, k // 3, k - 2 * (k // 3)
    return a, b, c


def inject_corruption(
    features: pd.DataFrame,
    *,
    rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Append synthetic duplicates, idle-heavy, and overrepresented-region rows.

    Original rows are copied, never mutated. `rate` is relative to the real N.
    """
    base = features.copy()
    base["is_corruption"] = False
    base["corruption_type"] = ""
    base["source_episode"] = ""
    n = len(base)
    k = int(round(rate * n))
    if k <= 0:
        return base.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_dup, n_idle, n_over = _split_counts(k)
    extras: list[dict] = []

    dup_idx = rng.choice(n, size=n_dup, replace=False) if n_dup else []
    for i, idx in enumerate(dup_idx):
        rec = _clone_row(base.iloc[int(idx)], f"{base.iloc[int(idx)]['episode_hash']}::dup{i}")
        rec["corruption_type"] = "duplicate"
        extras.append(rec)

    idle_idx = rng.choice(n, size=n_idle, replace=False) if n_idle else []
    for i, idx in enumerate(idle_idx):
        rec = _clone_row(base.iloc[int(idx)], f"{base.iloc[int(idx)]['episode_hash']}::idle{i}")
        rec["corruption_type"] = "idle"
        rec["stationary_ratio"] = 0.95
        rec["quality_score"] = 0.40
        rec["quality_nonstationary"] = 0.05
        rec["quality_score"] = 0.40
        z = rec["z"]
        # stationary_ratio is motion dim 7 → z index 16+7=23 when vis_pca_dim=16
        if z.shape[0] > 23:
            z = z.copy()
            z[23] = z[23] + 3.0
            rec["z"] = z
        extras.append(rec)

    region_counts = base["behavioral_region"].value_counts()
    over_region = int(region_counts.idxmax())
    pool = base.index[base["behavioral_region"] == over_region].to_numpy()
    over_idx = rng.choice(pool, size=n_over, replace=len(pool) < n_over) if n_over else []
    for i, idx in enumerate(over_idx):
        rec = _clone_row(base.loc[int(idx)], f"{base.loc[int(idx)]['episode_hash']}::over{i}")
        rec["corruption_type"] = "overrepresented_region"
        rec["behavioral_region"] = over_region
        noise = rng.normal(0.0, 0.05, size=rec["z"].shape).astype(np.float32)
        rec["z"] = rec["z"] + noise
        extras.append(rec)

    extra_df = pd.DataFrame(extras)
    return pd.concat([base, extra_df], ignore_index=True)
