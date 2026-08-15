#!/usr/bin/env python3
"""Export a static JSON payload for the EgoSelect frontend. No API server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from egoselect.baselines import budget_count  # noqa: E402
from egoselect.metrics import evaluate_keep  # noqa: E402

FEATURES = REPO_ROOT / "outputs" / "episode_features.parquet"
SELECTIONS = REPO_ROOT / "outputs" / "selections.csv"
EXPERIMENT = REPO_ROOT / "outputs" / "experiment_results.csv"
AUDIT = REPO_ROOT / "outputs" / "greedy_audit.json"
COHORT = REPO_ROOT / "outputs" / "cohort_summary.json"
OUT_A = REPO_ROOT / "outputs" / "demo_payload.json"
OUT_B = REPO_ROOT / "web" / "public" / "data" / "demo_payload.json"

METHODS = ("Random", "Dedup-only", "Diversity-only", "EgoSelect")
PRIMARY = 0.3


def _r(x, n: int = 4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    return round(float(x), n)


def _s(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x)
    return "" if text in {"nan", "None"} else text


def _method_block(row: pd.Series, *, stress: bool) -> dict:
    block = {
        "name": str(row["method"]),
        "coverage": _r(row["behavioral_region_coverage"]),
        "quality": _r(row["average_quality"]),
        "redundancy": _r(row["nn_redundancy"]),
        "visual_coverage": _r(row["visual_coverage"]),
        "stationary": _r(row["stationary_content_ratio"]),
    }
    if stress:
        block["corrupt_retained"] = int(row["n_corrupt_retained"])
        block["corrupt_pool"] = int(row["n_corrupt_in_pool"])
    return block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--selections", type=Path, default=SELECTIONS)
    args = parser.parse_args()

    for path in (args.features, args.selections, EXPERIMENT, AUDIT, COHORT):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    features = pd.read_parquet(args.features)
    sel = pd.read_csv(args.selections)
    exp = pd.read_csv(EXPERIMENT)
    audit = json.loads(AUDIT.read_text())
    cohort = json.loads(COHORT.read_text())

    n = len(sel)
    keep30 = budget_count(n, PRIMARY)
    keep_id = str(audit["keep_example"]["episode"])
    drop_id = str(audit["drop_example"]["episode"])

    episodes = []
    for _, rec in sel.sort_values("selection_rank").iterrows():
        hid = str(rec["episode_hash"])
        role = None
        if hid == keep_id:
            role = "keep_example"
        elif hid == drop_id:
            role = "drop_example"
        episodes.append(
            {
                "id": hid,
                "x": _r(rec["pca_x"], 6),
                "y": _r(rec["pca_y"], 6),
                "region": int(rec["behavioral_region"]),
                "rank": int(rec["selection_rank"]),
                "quality": _r(rec["quality"], 6),
                "coverage_gain": _r(rec["coverage_gain"], 6),
                "redundancy": _r(rec["redundancy"], 6),
                "value": _r(rec["value"], 6),
                "nearest": _s(rec["nearest_selected_episode"]),
                "nearest_similarity": _r(rec["nearest_similarity"], 6),
                "stationary": _r(rec["stationary_ratio"], 6),
                "reason": _s(rec["reason"]),
                "task": _s(rec.get("task")),
                "lab": _s(rec.get("lab")),
                "role": role,
            }
        )

    ordered = sel.sort_values("selection_rank")["episode_hash"].astype(str).tolist()
    curve = []
    for k in range(1, n + 1):
        ev = evaluate_keep(features, ordered[:k])
        curve.append(
            {
                "k": k,
                "fraction": _r(k / n, 4),
                "coverage": _r(ev["behavioral_region_coverage"]),
                "quality": _r(ev["average_quality"]),
                "redundancy": _r(ev["nn_redundancy"]),
                "visual_coverage": _r(ev["visual_coverage"]),
                "stationary": _r(ev["stationary_content_ratio"]),
            }
        )

    clean = exp[(exp["corruption_rate"] == 0.0) & (exp["budget"] == PRIMARY)]
    stress = exp[(exp["corruption_rate"] == 0.30) & (exp["budget"] == PRIMARY)]
    payload = {
        "meta": {
            "title": "EgoSelect",
            "subtitle": "Capability-aware training-value curation for EgoVerse",
            "n_episodes": n,
            "n_regions": int(features["behavioral_region"].nunique()),
            "primary_budget": PRIMARY,
            "primary_keep": keep30,
            "formula": "Value(i | S) = 0.35·Quality + 0.45·CoverageGain − 0.20·Redundancy",
            "weights": {"alpha": 0.35, "beta": 0.45, "gamma": 0.20},
            "greedy_recomputed": bool(audit.get("greedy_recomputed")),
            "keep_example": keep_id,
            "drop_example": drop_id,
            "keep_reason": audit["keep_example"]["reason"],
            "drop_reason": audit["drop_example"]["reason"],
            "headline": (
                "At 30% keep, EgoSelect matches full region coverage with the "
                "fewest injected corruptions; Dedup-only has lower redundancy."
            ),
            "tasks": cohort.get("tasks", []),
            "labs": cohort.get("labs", []),
            "n_rescore_passes": int(audit.get("n_rescore_passes", n)),
        },
        "episodes": episodes,
        "retention_curve": curve,
        "benchmark": {
            "budget": PRIMARY,
            "n_keep": keep30,
            "methods": [
                _method_block(clean[clean["method"] == name].iloc[0], stress=False)
                for name in METHODS
            ],
        },
        "stress": {
            "inject_rate": 0.3,
            "n_injected": int(stress["n_corrupt_in_pool"].iloc[0]),
            "n_keep": keep30,
            "methods": [
                _method_block(stress[stress["method"] == name].iloc[0], stress=True)
                for name in METHODS
            ],
        },
    }

    text = json.dumps(payload, indent=2)
    for dest in (OUT_A, OUT_B):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n")
        print(f"wrote {dest} bytes={dest.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
