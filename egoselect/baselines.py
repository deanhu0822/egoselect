"""Equal-budget baselines: Random, Dedup-only, Diversity-only, EgoSelect."""

from __future__ import annotations

import numpy as np
import pandas as pd

from egoselect.explain import explain_step
from egoselect.selector import greedy_select, result_to_frame
from egoselect.scoring import Weights


def budget_count(n: int, fraction: float) -> int:
    k = int(round(n * fraction))
    return max(1, min(n, k))


def random_ranking(features: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hashes = features["episode_hash"].astype(str).to_numpy()
    perm = rng.permutation(len(hashes))
    rows = []
    for rank, idx in enumerate(perm, start=1):
        rows.append(
            {
                "episode_hash": str(hashes[idx]),
                "selection_rank": rank,
                "quality": float(features.iloc[idx]["quality_score"]),
                "quality_norm": np.nan,
                "coverage_gain": np.nan,
                "redundancy": np.nan,
                "value": np.nan,
                "nearest_selected_episode": "",
                "nearest_similarity": np.nan,
                "new_region": np.nan,
                "distance_bonus": np.nan,
                "region_balance": np.nan,
                "behavioral_region": int(features.iloc[idx]["behavioral_region"]),
                "stationary_ratio": float(features.iloc[idx]["stationary_ratio"]),
                "reason": "Random permutation",
            }
        )
    return pd.DataFrame(rows)


def greedy_ranking(
    features: pd.DataFrame,
    *,
    objective: str,
    weights: Weights | None = None,
) -> tuple[pd.DataFrame, object]:
    result = greedy_select(features, weights=weights, objective=objective)
    frame = result_to_frame(result)
    frame["reason"] = [explain_step(rec) for rec in result.order]
    return frame, result


def rankings(features: pd.DataFrame, *, seed: int = 42, weights: Weights | None = None):
    ego_frame, ego_result = greedy_ranking(
        features, objective="egoselect", weights=weights
    )
    dedup_frame, _ = greedy_ranking(features, objective="dedup", weights=weights)
    div_frame, _ = greedy_ranking(features, objective="diversity", weights=weights)
    rnd = random_ranking(features, seed=seed)
    return {
        "EgoSelect": (ego_frame, ego_result),
        "Dedup-only": (dedup_frame, None),
        "Diversity-only": (div_frame, None),
        "Random": (rnd, None),
    }


def keep_prefix(ranking: pd.DataFrame, k: int) -> list[str]:
    ordered = ranking.sort_values("selection_rank")
    return ordered["episode_hash"].astype(str).head(k).tolist()
