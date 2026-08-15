#!/usr/bin/env python3
"""Phase 3: equal-budget baselines, metrics, and controlled corruption."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from egoselect.baselines import budget_count, keep_prefix, rankings  # noqa: E402
from egoselect.corruption import inject_corruption  # noqa: E402
from egoselect.dataset import dump_json  # noqa: E402
from egoselect.metrics import evaluate_keep  # noqa: E402

FEATURES_PATH = REPO_ROOT / "outputs" / "episode_features.parquet"
METRICS_PATH = REPO_ROOT / "outputs" / "metrics.json"
EXPERIMENT_PATH = REPO_ROOT / "outputs" / "experiment_results.csv"
BUDGETS = (0.10, 0.30, 0.50)
CORRUPTION_RATES = (0.0, 0.10, 0.20, 0.30)
PRIMARY_BUDGET = 0.30
METHODS = ("EgoSelect", "Dedup-only", "Diversity-only", "Random")


def _corrupt_retained(pool: pd.DataFrame, kept: list[str]) -> dict:
    sub = pool.set_index("episode_hash").loc[kept]
    flag = sub["is_corruption"].astype(bool)
    types = sub.loc[flag, "corruption_type"] if "corruption_type" in sub.columns else []
    n_ret = int(flag.sum())
    n_pool = int(pool["is_corruption"].astype(bool).sum())
    return {
        "n_corrupt_in_pool": n_pool,
        "n_corrupt_retained": n_ret,
        "corrupt_retained_frac": float(n_ret / n_pool) if n_pool else 0.0,
        "n_dup_retained": int((types == "duplicate").sum()) if n_pool else 0,
        "n_idle_retained": int((types == "idle").sum()) if n_pool else 0,
        "n_over_retained": int((types == "overrepresented_region").sum())
        if n_pool
        else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    n_real = len(features)
    clean_ranks = rankings(features, seed=args.seed)

    metrics: dict = {"n_real": n_real, "budgets": {}, "primary_budget": PRIMARY_BUDGET}
    rows = []
    for frac in BUDGETS:
        k = budget_count(n_real, frac)
        metrics["budgets"][str(frac)] = {"n_keep": k, "methods": {}}
        sizes = set()
        for name in METHODS:
            ranking, extra = clean_ranks[name]
            kept = keep_prefix(ranking, k)
            sizes.add(len(kept))
            if len(kept) != k:
                print(f"ERROR: {name} keep {len(kept)} != {k}", file=sys.stderr)
                return 1
            ev = evaluate_keep(features, kept, extra={"method": name, "budget": frac})
            if name == "EgoSelect" and extra is not None:
                ev["greedy_recomputed"] = bool(extra.greedy_recomputed)
                ev["n_rescore_passes"] = int(extra.n_rescore_passes)
            metrics["budgets"][str(frac)]["methods"][name] = ev
        if sizes != {k}:
            print(f"ERROR: unequal budgets at {frac}: {sizes}", file=sys.stderr)
            return 1

    for rate in CORRUPTION_RATES:
        pool = inject_corruption(features, rate=rate, seed=args.seed)
        k = budget_count(n_real, PRIMARY_BUDGET)
        ranked = rankings(pool, seed=args.seed)
        for name in METHODS:
            ranking, _ = ranked[name]
            kept = keep_prefix(ranking, k)
            ev = evaluate_keep(pool, kept)
            cr = _corrupt_retained(pool, kept)
            rows.append(
                {
                    "method": name,
                    "budget": PRIMARY_BUDGET,
                    "n_keep": k,
                    "corruption_rate": rate,
                    **ev,
                    **cr,
                }
            )

    exp = pd.DataFrame(rows)
    args.experiment.parent.mkdir(parents=True, exist_ok=True)
    exp.to_csv(args.experiment, index=False)
    dump_json(metrics, args.metrics)

    print("\nMethod | Keep | Coverage | Quality | Redundancy | Corruption retained")
    primary = exp[exp["corruption_rate"] == 0.0]
    for name in METHODS:
        r = primary[primary["method"] == name].iloc[0]
        c30 = exp[(exp["method"] == name) & (exp["corruption_rate"] == 0.30)].iloc[0]
        print(
            f"{name:16s} | {int(r['n_keep']):4d} | "
            f"{r['behavioral_region_coverage']:.3f}    | "
            f"{r['average_quality']:.3f}  | "
            f"{r['nn_redundancy']:.3f}      | "
            f"{int(c30['n_corrupt_retained']):d}/{int(c30['n_corrupt_in_pool'])} "
            f"@30% inject"
        )
    print(f"wrote {args.metrics}")
    print(f"wrote {args.experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
