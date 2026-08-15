"""Thin EgoSelect adapter over current EgoVerse APIs.

SQL labels (task, scene, lab, operator) are for sampling, stratification,
and retrospective evaluation only. They are not inputs to the behavioral
representation.

Episode bytes stay on disk. Callers read frames through EgoVerse's
DatasetFilter → Resolver → MultiDataset → DataLoader path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment_id
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    S3EpisodeResolver,
)
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "egoverse_cache"
DEFAULT_COHORT_PATH = REPO_ROOT / "outputs" / "sampled_episodes.csv"

# Deliberate 80-episode plan: 5 repeated task groups, 16 each.
# Mixes two labs on the short tasks so SQL sanity checks have lab diversity
# without putting those labels into the representation.
COHORT_PLAN: list[tuple[str, list[tuple[str, int]]]] = [
    ("fold_clothes", [("mecka", 12), ("rl2", 4)]),
    ("cup_on_saucer", [("mecka", 12), ("rl2", 4)]),
    ("dishwashing", [("mecka", 16)]),
    ("potting_plants", [("mecka", 16)]),
    ("ironing_clothes", [("mecka", 16)]),
]

REQUIRED_RUNTIME_KEYS = (
    "observations.images.front_img_1",
    "actions_cartesian",
    "observations.state.ee_pose",
    "intrinsics",
    "embodiment",
    "episode_hash",
)

OPTIONAL_ZARR_KEYS = (
    "left.obs_keypoints",
    "right.obs_keypoints",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "left.obs_aria_keypoints",
    "right.obs_aria_keypoints",
    "obs_head_pose",
    "obs_eye_gaze",
    "annotations",
    "images.right_wrist",
    "images.left_wrist",
    "right.cmd_ee_pose",
    "left.cmd_ee_pose",
)

SQL_EVAL_COLUMNS = ("task", "scene", "lab", "operator")

CURRENT_EMBODIMENTS = tuple(member.name.lower() for member in EMBODIMENT)


def load_episode_table(cache_path: Path | None = None) -> pd.DataFrame:
    """Load `app.episodes` through official EgoVerse SQL helpers."""
    load_env()
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def usable_human_bimanual(df: pd.DataFrame) -> pd.DataFrame:
    """Current-schema human episodes with a processed zarr path."""
    out = df.copy()
    out = out[~out["is_deleted"].fillna(False).astype(bool)]
    out = out[out["embodiment"].astype(str) == "human_bimanual"]
    path = out["zarr_processed_path"].fillna("").astype(str).str.strip()
    out = out[path != ""]
    out = out[out["num_frames"].fillna(0) > 0]
    return out


def hashes_to_filter(episode_hashes: Sequence[str]) -> DatasetFilter:
    dumped = "{" + ", ".join(repr(str(h)) for h in episode_hashes) + "}"
    return DatasetFilter(
        filter_lambdas=[f"lambda row: row.get('episode_hash') in {dumped}"]
    )


def _take_round_robin(
    sub: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Spread across scenes, then operators, while allowing repeats for redundancy."""
    if len(sub) <= n:
        return sub.copy()
    work = sub.copy()
    work["_scene"] = work["scene"].fillna("__na__").astype(str)
    work["_operator"] = work["operator"].fillna("__na__").astype(str)
    work = work.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000_000)))
    buckets: dict[tuple[str, str], list[int]] = {}
    order: list[tuple[str, str]] = []
    for idx, row in work.iterrows():
        key = (row["_scene"], row["_operator"])
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(idx)
    rng.shuffle(order)
    picked: list[int] = []
    while len(picked) < n:
        progressed = False
        for key in order:
            if buckets[key] and len(picked) < n:
                picked.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return work.loc[picked].drop(columns=["_scene", "_operator"])


def sample_cohort(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    plan: Sequence[tuple[str, Sequence[tuple[str, int]]]] = COHORT_PLAN,
) -> pd.DataFrame:
    """Build a 50–100 episode cohort with repeated task groups."""
    rng = np.random.default_rng(seed)
    pool = usable_human_bimanual(df)
    parts: list[pd.DataFrame] = []
    for task, lab_quotas in plan:
        for lab, n in lab_quotas:
            sub = pool[(pool["task"] == task) & (pool["lab"] == lab)]
            if sub.empty:
                raise ValueError(f"No usable episodes for task={task!r} lab={lab!r}")
            if len(sub) < n:
                raise ValueError(
                    f"Need {n} episodes for task={task!r} lab={lab!r}, found {len(sub)}"
                )
            taken = _take_round_robin(sub, n, rng)
            taken = taken.copy()
            taken["cohort_group"] = task
            parts.append(taken)
    cohort = pd.concat(parts, ignore_index=True)
    keep = [
        "episode_hash",
        "cohort_group",
        "task",
        "lab",
        "scene",
        "operator",
        "embodiment",
        "rig_name",
        "num_frames",
        "zarr_processed_path",
        "is_eval",
    ]
    keep = [c for c in keep if c in cohort.columns]
    return cohort[keep].reset_index(drop=True)


def save_cohort(cohort: pd.DataFrame, path: Path = DEFAULT_COHORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(path, index=False)
    return path


def human_cartesian_maps(*, stride: int = 1, has_head_pose: bool = True):
    """Official Human cartesian keymap + transform list (current embodiments)."""
    key_map = Human.get_keymap(
        keymap_mode="cartesian", has_head_pose=has_head_pose
    )
    transform_list = Human.get_transform_list(mode="cartesian", stride=stride)
    return key_map, transform_list


def ensure_episodes_local(
    episode_hashes: Sequence[str],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    numworkers: int = 16,
) -> Path:
    """Download missing zarrs through official S3EpisodeResolver.sync_from_filters."""
    load_env()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters=hashes_to_filter(list(episode_hashes)),
        local_dir=cache_dir,
        numworkers=numworkers,
    )
    return cache_dir


def local_episode_path(episode_hash: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    cache_dir = Path(cache_dir)
    direct = cache_dir / episode_hash
    if direct.is_dir():
        return direct
    dotted = cache_dir / f"{episode_hash}.zarr"
    if dotted.is_dir():
        return dotted
    raise FileNotFoundError(f"Episode {episode_hash} not in {cache_dir}")


def build_multidataset(
    episode_hashes: Sequence[str],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    resolver_kind: str = "s3",
    stride: int = 1,
    has_head_pose: bool = True,
    sync_from_s3: bool = True,
) -> MultiDataset:
    """Resolve selected hashes with official EgoVerse abstractions."""
    load_env()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_map, transform_list = human_cartesian_maps(
        stride=stride, has_head_pose=has_head_pose
    )
    filters = hashes_to_filter(list(episode_hashes))
    if resolver_kind == "s3":
        resolver = S3EpisodeResolver(
            cache_dir, key_map=key_map, transform_list=transform_list
        )
        return MultiDataset._from_resolver(
            resolver, filters=filters, mode="total"
        )
    if resolver_kind == "local":
        resolver = LocalEpisodeResolver(
            cache_dir, key_map=key_map, transform_list=transform_list
        )
        return MultiDataset._from_resolver(
            resolver,
            filters=filters,
            sync_from_s3=False,
            mode="total",
        )
    raise ValueError(f"Unknown resolver_kind={resolver_kind!r}")


def make_loader(
    multi_ds: MultiDataset, *, batch_size: int = 1, num_workers: int = 0
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        multi_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def describe_value(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "len": len(value),
            "item": describe_value(value[0]) if value else None,
        }
    return {"kind": type(value).__name__, "value": repr(value)[:200]}


def inspect_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    fields = {key: describe_value(val) for key, val in batch.items()}
    episode_hash = batch.get("episode_hash")
    if isinstance(episode_hash, (list, tuple)):
        episode_hash = episode_hash[0]
    embodiment = batch.get("embodiment")
    if isinstance(embodiment, torch.Tensor):
        embodiment = int(embodiment.reshape(-1)[0].item())
    return {
        "keys": list(batch.keys()),
        "fields": fields,
        "episode_hash": episode_hash,
        "embodiment": embodiment,
        "required_present": {
            key: key in batch for key in REQUIRED_RUNTIME_KEYS
        },
    }


def raw_episode_keys(multi_ds: MultiDataset) -> dict[str, dict[str, Any]]:
    """List on-disk zarr keys without copying episode arrays into memory."""
    out: dict[str, dict[str, Any]] = {}
    for name, ds in multi_ds.datasets.items():
        keys = sorted(getattr(ds, "keys_dict", {}).keys())
        optional = {k: k in keys for k in OPTIONAL_ZARR_KEYS}
        out[name] = {
            "embodiment": getattr(ds, "embodiment", None),
            "total_frames": getattr(ds, "total_frames", None),
            "n_keys": len(keys),
            "keys": keys,
            "optional_present": optional,
        }
    return out


def inspect_unique_episodes(
    multi_ds: MultiDataset,
    *,
    n_episodes: int = 3,
) -> list[dict[str, Any]]:
    """Load one real batch from each of N distinct resolved episodes."""
    seen: set[str] = set()
    reports: list[dict[str, Any]] = []
    loader = make_loader(multi_ds, batch_size=1)
    for batch in loader:
        report = inspect_batch(batch)
        ep = str(report["episode_hash"])
        if ep in seen:
            continue
        seen.add(ep)
        reports.append(report)
        if len(reports) >= n_episodes:
            break
    return reports


def current_embodiment_note() -> dict[str, Any]:
    return {
        "enum": list(CURRENT_EMBODIMENTS),
        "human_bimanual_id": get_embodiment_id("human_bimanual"),
        "eva_bimanual_id": get_embodiment_id("eva_bimanual"),
        "sql_eval_columns": list(SQL_EVAL_COLUMNS),
        "human_cartesian_zarr_keys": [
            spec["zarr_key"]
            for spec in Human.get_keymap(keymap_mode="cartesian").values()
        ],
        "eva_cartesian_zarr_keys": [
            spec["zarr_key"]
            for spec in Eva.get_keymap(keymap_mode="cartesian").values()
        ],
    }


def dump_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
