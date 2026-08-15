"""Greedy marginal selection. Remaining candidates are rescored after every pick."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from egoselect.scoring import (
    Weights,
    coverage_gain,
    l2_normalize,
    minmax_quality,
    pairwise_l2_median,
    redundancy_to_selected,
    value_of,
)


@dataclass
class StepRecord:
    episode_hash: str
    selection_rank: int
    quality: float
    quality_norm: float
    coverage_gain: float
    redundancy: float
    value: float
    nearest_selected_episode: str | None
    nearest_similarity: float
    new_region: float
    distance_bonus: float
    region_balance: float
    behavioral_region: int
    stationary_ratio: float
    n_selected_before: int
    remaining_value_fingerprint: str


@dataclass
class GreedyResult:
    order: list[StepRecord]
    n_rescore_passes: int
    first_pick_hash: str
    max_quality_hash: str
    second_pick_hash: str
    second_quality_hash: str
    max_abs_value_shift_after_first: float
    ranking_changed_after_first: bool

    @property
    def greedy_recomputed(self) -> bool:
        n = len(self.order)
        return self.n_rescore_passes == n


def _fingerprint(values: np.ndarray) -> str:
    return format(abs(hash(np.round(values, 6).tobytes())), "x")[:16]


def greedy_select(
    features: pd.DataFrame,
    *,
    weights: Weights | None = None,
    objective: str = "egoselect",
) -> GreedyResult:
    """Select every episode. `objective` is egoselect | diversity | dedup.

    Forbidden path (score once → sort once) is not implemented. Each remaining
    candidate is rescored against the updated S.
    """
    weights = weights or Weights()
    hashes = features["episode_hash"].astype(str).to_numpy()
    z = np.stack(features["z"].to_numpy()).astype(np.float64)
    z_unit = l2_normalize(z)
    regions = features["behavioral_region"].to_numpy(dtype=int)
    quality_raw = features["quality_score"].to_numpy(dtype=np.float64)
    quality_norm = minmax_quality(quality_raw)
    stationary = features["stationary_ratio"].to_numpy(dtype=np.float64)
    d_scale = pairwise_l2_median(z)

    n = len(features)
    remaining = list(range(n))
    selected: list[int] = []
    order: list[StepRecord] = []
    n_rescore_passes = 0
    quality_order = np.argsort(-quality_norm, kind="stable")
    max_quality_hash = str(hashes[int(quality_order[0])])
    second_quality_hash = (
        str(hashes[int(quality_order[1])]) if n > 1 else max_quality_hash
    )

    while remaining:
        values = np.empty(len(remaining), dtype=np.float64)
        meta: list[tuple[float, float, str | None, float, dict[str, float]]] = []
        for k, idx in enumerate(remaining):
            red, nearest, _sim = redundancy_to_selected(z_unit, hashes, idx, selected)
            gain, parts = coverage_gain(
                regions=regions,
                z=z,
                d_scale=d_scale,
                candidate=idx,
                selected=selected,
                weights=weights,
            )
            qn = float(quality_norm[idx])
            if objective == "egoselect":
                val = value_of(qn, gain, red, weights)
            elif objective == "diversity":
                val = gain
            elif objective == "dedup":
                val = -red + 1e-6 * qn
            else:
                raise ValueError(f"Unknown objective {objective!r}")
            values[k] = val
            meta.append((qn, gain, nearest, red, parts))
        n_rescore_passes += 1

        best_k = int(
            sorted(
                range(len(remaining)),
                key=lambda k: (-values[k], str(hashes[remaining[k]])),
            )[0]
        )
        idx = remaining[best_k]
        qn, gain, nearest, red, parts = meta[best_k]
        order.append(
            StepRecord(
                episode_hash=str(hashes[idx]),
                selection_rank=len(selected) + 1,
                quality=float(quality_raw[idx]),
                quality_norm=qn,
                coverage_gain=float(gain),
                redundancy=float(red),
                value=float(values[best_k]),
                nearest_selected_episode=nearest,
                nearest_similarity=float(red),
                new_region=float(parts["new_region"]),
                distance_bonus=float(parts["distance"]),
                region_balance=float(parts["balance"]),
                behavioral_region=int(regions[idx]),
                stationary_ratio=float(stationary[idx]),
                n_selected_before=len(selected),
                remaining_value_fingerprint=_fingerprint(values),
            )
        )
        selected.append(idx)
        remaining.pop(best_k)

    second_pick_hash = order[1].episode_hash if len(order) > 1 else order[0].episode_hash
    ranking_changed = second_pick_hash != second_quality_hash
    empty_vals = weights.alpha * quality_norm + weights.beta * 1.0
    after_map = {rec.episode_hash: rec.value for rec in order[1:]}
    diffs = [
        abs(after_map[str(hashes[i])] - float(empty_vals[i]))
        for i in range(n)
        if str(hashes[i]) in after_map
    ]
    shift = float(max(diffs)) if diffs else 0.0

    return GreedyResult(
        order=order,
        n_rescore_passes=n_rescore_passes,
        first_pick_hash=order[0].episode_hash,
        max_quality_hash=max_quality_hash,
        second_pick_hash=second_pick_hash,
        second_quality_hash=second_quality_hash,
        max_abs_value_shift_after_first=shift,
        ranking_changed_after_first=ranking_changed,
    )


def result_to_frame(result: GreedyResult) -> pd.DataFrame:
    rows = []
    for rec in result.order:
        rows.append(
            {
                "episode_hash": rec.episode_hash,
                "selection_rank": rec.selection_rank,
                "quality": rec.quality,
                "quality_norm": rec.quality_norm,
                "coverage_gain": rec.coverage_gain,
                "redundancy": rec.redundancy,
                "value": rec.value,
                "nearest_selected_episode": rec.nearest_selected_episode or "",
                "nearest_similarity": rec.nearest_similarity,
                "new_region": rec.new_region,
                "distance_bonus": rec.distance_bonus,
                "region_balance": rec.region_balance,
                "behavioral_region": rec.behavioral_region,
                "stationary_ratio": rec.stationary_ratio,
            }
        )
    return pd.DataFrame(rows)
