#!/usr/bin/env python3
"""Phase 1: EgoVerse SQL audit, cohort sampling, and runtime-schema inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from egoselect.dataset import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_COHORT_PATH,
    REQUIRED_RUNTIME_KEYS,
    build_multidataset,
    current_embodiment_note,
    dump_json,
    inspect_unique_episodes,
    load_episode_table,
    raw_episode_keys,
    sample_cohort,
    save_cohort,
    usable_human_bimanual,
)


def _summarize_cohort(cohort) -> dict:
    def _nunique(col: str) -> int:
        if col not in cohort.columns:
            return 0
        return int(cohort[col].nunique(dropna=True))

    return {
        "episode_count": int(len(cohort)),
        "tasks": sorted(cohort["task"].dropna().unique().tolist()),
        "task_counts": cohort.groupby("task").size().to_dict(),
        "scenes": _nunique("scene"),
        "labs": sorted(cohort["lab"].dropna().unique().tolist()),
        "operators": _nunique("operator"),
        "embodiments": sorted(cohort["embodiment"].dropna().unique().tolist()),
        "num_frames": {
            "min": float(cohort["num_frames"].min()),
            "median": float(cohort["num_frames"].median()),
            "max": float(cohort["num_frames"].max()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Write sampled_episodes.csv")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load real episodes via MultiDataset and record runtime schema",
    )
    parser.add_argument("--inspect-n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--sql-cache",
        type=Path,
        default=REPO_ROOT / "data" / "episode_table.parquet",
    )
    parser.add_argument("--cohort-path", type=Path, default=DEFAULT_COHORT_PATH)
    args = parser.parse_args()

    if not args.sample and not args.inspect:
        args.sample = True
        args.inspect = True

    print("Loading SQL episode table via create_default_engine / episode_table_to_df ...")
    df = load_episode_table(cache_path=args.sql_cache)
    print(f"SQL rows={len(df)} columns={list(df.columns)}")
    usable = usable_human_bimanual(df)
    print(f"usable human_bimanual={len(usable)}")

    schema_path = REPO_ROOT / "outputs" / "sql_schema.json"
    dump_json(
        {
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "embodiment_counts": df["embodiment"].value_counts(dropna=False).to_dict(),
            "lab_counts": df["lab"].value_counts(dropna=False).to_dict(),
            "current_embodiments": current_embodiment_note(),
        },
        schema_path,
    )
    print(f"wrote {schema_path}")

    if args.sample:
        cohort = sample_cohort(df, seed=args.seed)
        save_cohort(cohort, args.cohort_path)
        summary = _summarize_cohort(cohort)
        dump_json(summary, REPO_ROOT / "outputs" / "cohort_summary.json")
        print(f"wrote {args.cohort_path} n={len(cohort)}")
        print(json.dumps(summary, indent=2, default=str))
    else:
        import pandas as pd

        cohort = pd.read_csv(args.cohort_path)

    if args.inspect:
        inspect_n = min(args.inspect_n, len(cohort))
        # Prefer one episode from three different task groups.
        hashes: list[str] = []
        for _, group in cohort.groupby("cohort_group", sort=True):
            hashes.append(str(group.iloc[0]["episode_hash"]))
            if len(hashes) >= inspect_n:
                break
        while len(hashes) < inspect_n:
            candidate = str(cohort.iloc[len(hashes)]["episode_hash"])
            if candidate not in hashes:
                hashes.append(candidate)
        print(f"Inspecting {len(hashes)} episodes: {hashes}")
        multi_ds = build_multidataset(hashes, cache_dir=args.cache_dir)
        print(f"resolved episodes={list(multi_ds.datasets.keys())} len={len(multi_ds)}")
        raw = raw_episode_keys(multi_ds)
        reports = inspect_unique_episodes(multi_ds, n_episodes=inspect_n)
        payload = {
            "inspect_hashes": hashes,
            "resolved": list(multi_ds.datasets.keys()),
            "raw_zarr": raw,
            "batches": reports,
            "required_runtime_keys": list(REQUIRED_RUNTIME_KEYS),
        }
        out = dump_json(payload, REPO_ROOT / "outputs" / "runtime_schema.json")
        print(f"wrote {out}")
        print(f"inspected {len(reports)} unique episodes")
        for report in reports:
            print(
                f"  hash={report['episode_hash']} embodiment={report['embodiment']} "
                f"keys={report['keys']}"
            )
            print(f"  required={report['required_present']}")
        if len(reports) < 3:
            print("ERROR: need at least 3 successfully inspected episodes", file=sys.stderr)
            return 1
        missing_any = [
            key
            for report in reports
            for key, present in report["required_present"].items()
            if not present
        ]
        if missing_any:
            print(f"WARNING: missing required keys in some batches: {missing_any}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
