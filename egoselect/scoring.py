"""Transparent Quality, CoverageGain, and Redundancy scores.

Scores that depend on S are recomputed from scratch for every remaining
candidate. Quality is independent of S and is min-max normalized so that
higher is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA = 0.35
BETA = 0.45
GAMMA = 0.20

NEW_REGION_W = 0.50
DISTANCE_W = 0.30
BALANCE_W = 0.20


@dataclass(frozen=True)
class Weights:
    alpha: float = ALPHA
    beta: float = BETA
    gamma: float = GAMMA
    new_region: float = NEW_REGION_W
    distance: float = DISTANCE_W
    balance: float = BALANCE_W


def l2_normalize(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(nrm, eps)


def minmax_quality(quality: np.ndarray) -> np.ndarray:
    q = np.asarray(quality, dtype=np.float64)
    lo, hi = float(np.min(q)), float(np.max(q))
    if hi - lo < 1e-12:
        return np.ones_like(q)
    return (q - lo) / (hi - lo)


def pairwise_l2_median(z: np.ndarray) -> float:
    n = len(z)
    if n < 2:
        return 1.0
    d = np.sqrt(np.maximum(0.0, ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)))
    iu = np.triu_indices(n, k=1)
    med = float(np.median(d[iu]))
    return med if med > 1e-12 else 1.0


def redundancy_to_selected(
    z_unit: np.ndarray,
    hashes: np.ndarray,
    candidate: int,
    selected: list[int],
) -> tuple[float, str | None, float]:
    """Max cosine similarity of candidate to S. Empty S → 0."""
    if not selected:
        return 0.0, None, 0.0
    sims = z_unit[candidate] @ z_unit[selected].T
    j = int(np.argmax(sims))
    sim = float(sims[j])
    return sim, str(hashes[selected[j]]), sim


def coverage_gain(
    *,
    regions: np.ndarray,
    z: np.ndarray,
    d_scale: float,
    candidate: int,
    selected: list[int],
    weights: Weights,
) -> tuple[float, dict[str, float]]:
    region_i = int(regions[candidate])
    if not selected:
        parts = {"new_region": 1.0, "distance": 1.0, "balance": 1.0}
    else:
        selected_regions = regions[selected]
        new_region = 0.0 if np.any(selected_regions == region_i) else 1.0
        delta = z[candidate] - z[selected]
        d_min = float(np.sqrt(np.maximum(0.0, (delta * delta).sum(axis=1))).min())
        distance = float(np.clip(d_min / d_scale, 0.0, 1.0))
        count_r = int(np.sum(selected_regions == region_i))
        balance = float(1.0 - count_r / max(len(selected), 1))
        parts = {
            "new_region": float(new_region),
            "distance": distance,
            "balance": balance,
        }
    gain = (
        weights.new_region * parts["new_region"]
        + weights.distance * parts["distance"]
        + weights.balance * parts["balance"]
    )
    return float(gain), parts


def value_of(
    quality_norm: float,
    gain: float,
    redundancy: float,
    weights: Weights,
) -> float:
    return (
        weights.alpha * quality_norm
        + weights.beta * gain
        - weights.gamma * redundancy
    )
