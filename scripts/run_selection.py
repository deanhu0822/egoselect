#!/usr/bin/env python3
"""Phase 3: greedy EgoSelect ranking over the full cached cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from egoselect.baselines import greedy_ranking  # noqa: E402
from egoselect.dataset import dump_json  # noqa: E402

FEATURES_PATH = REPO_ROOT / "outputs" / "episode_features.parquet"
SELECTIONS_PATH = REPO_ROOT / "outputs" / "selections.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=SELECTIONS_PATH)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    ranking, result = greedy_ranking(features, objective="egoselect")
    meta_cols = [
        c
        for c in (
            "episode_hash",
            "pca_x",
            "pca_y",
            "task",
            "lab",
            "scene",
            "operator",
            "embodiment",
            "rig_name",
        )
        if c in features.columns
    ]
    out = ranking.merge(features[meta_cols], on="episode_hash", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    audit = {
        "n_episodes": int(len(out)),
        "n_rescore_passes": int(result.n_rescore_passes),
        "greedy_recomputed": bool(result.greedy_recomputed),
        "first_pick": result.first_pick_hash,
        "max_quality": result.max_quality_hash,
        "first_pick_is_max_quality": result.first_pick_hash == result.max_quality_hash,
        "second_pick": result.second_pick_hash,
        "second_by_quality": result.second_quality_hash,
        "ranking_changed_after_first": bool(result.ranking_changed_after_first),
        "max_abs_value_shift_after_first": result.max_abs_value_shift_after_first,
        "keep_example": {
            "episode": out.iloc[0]["episode_hash"],
            "reason": out.iloc[0]["reason"],
        },
        "drop_example": {
            "episode": out.iloc[-1]["episode_hash"],
            "reason": out.iloc[-1]["reason"],
        },
    }
    dump_json(audit, REPO_ROOT / "outputs" / "greedy_audit.json")
    if not result.greedy_recomputed:
        print("ERROR: greedy rescore count != n_episodes", file=sys.stderr)
        return 1
    print(json.dumps(audit, indent=2))
    print(f"wrote {args.out} ranks=1..{len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
