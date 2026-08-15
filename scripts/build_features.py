#!/usr/bin/env python3
"""Phase 2: extract cached multimodal features and representation-space regions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from egoselect.dataset import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_COHORT_PATH,
    dump_json,
    ensure_episodes_local,
)
from egoselect.features import (  # noqa: E402
    DINO_MODEL_ID,
    DinoEncoder,
    N_RGB_FRAMES,
    extract_episode_features,
    load_visual_cache,
    save_visual_cache,
)
from egoselect.representation import (  # noqa: E402
    N_REGIONS,
    VIS_PCA_DIM,
    fit_representation,
)

VISUAL_CACHE_PATH = REPO_ROOT / "data" / "visual_embeddings.parquet"
FEATURES_PATH = REPO_ROOT / "outputs" / "episode_features.parquet"


def _present_hashes(cache_dir: Path) -> set[str]:
    if not cache_dir.exists():
        return set()
    names = set()
    for p in cache_dir.iterdir():
        if p.is_dir():
            names.add(p.name[:-5] if p.name.endswith(".zarr") else p.name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=FEATURES_PATH)
    parser.add_argument("--visual-cache", type=Path, default=VISUAL_CACHE_PATH)
    parser.add_argument("--n-rgb", type=int, default=N_RGB_FRAMES)
    parser.add_argument("--n-regions", type=int, default=N_REGIONS)
    parser.add_argument("--vis-pca-dim", type=int, default=VIS_PCA_DIM)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    hashes = [str(h) for h in cohort["episode_hash"]]
    print(f"cohort episodes={len(hashes)}")

    do_sync = args.sync or not args.no_sync
    if do_sync:
        missing = [h for h in hashes if h not in _present_hashes(args.cache_dir)]
        print(f"local cache missing={len(missing)}; syncing via S3EpisodeResolver")
        ensure_episodes_local(hashes, cache_dir=args.cache_dir)

    present = _present_hashes(args.cache_dir)
    runnable = [h for h in hashes if h in present]
    skipped = [h for h in hashes if h not in present]
    if skipped:
        print(f"WARNING: {len(skipped)} hashes still missing locally")
    if len(runnable) < 50:
        print(
            f"ERROR: only {len(runnable)} local episodes; need ~50–100",
            file=sys.stderr,
        )
        return 1

    vis_cache = load_visual_cache(args.visual_cache)
    print(f"visual cache hits={sum(h in vis_cache for h in runnable)} model={DINO_MODEL_ID}")
    encoder = DinoEncoder(model_id=DINO_MODEL_ID)

    rows = []
    frame_validation = []
    for i, h in enumerate(tqdm(runnable, desc="featurize")):
        row = extract_episode_features(
            h,
            cache_dir=args.cache_dir,
            encoder=encoder,
            n_rgb=args.n_rgb,
            visual_cache=vis_cache if h in vis_cache else None,
        )
        rows.append(row)
        if i < 4:
            frame_validation.append(
                {
                    "episode_hash": h,
                    "indices": row["sampled_indices"],
                    "decoded": row["n_rgb_decoded"],
                    "frames": row["_frame_report"],
                }
            )
        row.pop("_frame_report", None)

    save_visual_cache(rows, args.visual_cache)
    features, meta = fit_representation(
        rows, vis_pca_dim=args.vis_pca_dim, n_regions=args.n_regions
    )
    meta_cols = [
        c
        for c in (
            "episode_hash",
            "cohort_group",
            "task",
            "lab",
            "scene",
            "operator",
            "embodiment",
            "rig_name",
        )
        if c in cohort.columns
    ]
    features = features.merge(cohort[meta_cols], on="episode_hash", how="left")

    motion_cols = [
        "path_length",
        "displacement",
        "mean_speed",
        "speed_std",
        "max_speed",
        "mean_abs_accel",
        "direction_change_ratio",
        "stationary_ratio",
        "duration_s",
        "n_frames",
    ]
    nan_motion = int(features[motion_cols].isna().sum().sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    dump_json(
        {
            **meta,
            "visual_model": DINO_MODEL_ID,
            "n_rgb": args.n_rgb,
            "nan_motion_cells": nan_motion,
            "skipped_hashes": skipped,
            "quality_score": {
                "min": float(features["quality_score"].min()),
                "median": float(features["quality_score"].median()),
                "max": float(features["quality_score"].max()),
            },
            "stationary_ratio": {
                "min": float(features["stationary_ratio"].min()),
                "median": float(features["stationary_ratio"].median()),
                "max": float(features["stationary_ratio"].max()),
            },
        },
        REPO_ROOT / "outputs" / "representation_meta.json",
    )
    dump_json({"episodes": frame_validation}, REPO_ROOT / "outputs" / "frame_validation.json")

    print("\n=== Phase 2 validation ===")
    print(f"episodes featurized: {len(features)}")
    print(f"visual: {DINO_MODEL_ID} dim={meta['visual_dim']} pca={meta['visual_pca_dim']}")
    print(f"z_dim={meta['z_dim']} regions={meta['n_regions']} counts={meta['region_counts']}")
    print(f"quality median={features['quality_score'].median():.3f}")
    print(f"stationary median={features['stationary_ratio'].median():.3f}")
    print(f"motion NaNs={nan_motion}")
    print(f"wrote {args.out}")
    print(f"wrote {args.visual_cache} (DINO cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
