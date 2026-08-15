# Build state

Current phase: **Phase 1 complete**

## Completed

- Cloned current EgoVerse (`third_party/EgoVerse`, gitignored)
- Slim Python 3.11 env with `egomimic` installed `--no-deps`
- Public EgoVerse AWS/R2 + readonly SQL credentials via `setup_secret.sh`
- SQL metadata loaded through `create_default_engine()` → `episode_table_to_df(engine)`
- 80-episode cohort written
- Thin adapter `egoselect/dataset.py` wrapping official `DatasetFilter` → `S3EpisodeResolver`/`LocalEpisodeResolver` → `MultiDataset` → `DataLoader`
- 4 real episodes downloaded and inspected (3 Mecka + 1 RL2/Aria)

## Sample cohort

- episode count: **80**
- tasks (16 each): `fold_clothes`, `cup_on_saucer`, `dishwashing`, `potting_plants`, `ironing_clothes`
- labs: `mecka` (72), `rl2` (8)
- scenes: 24 nonempty (+ 17 blank scene rows)
- operators: 70
- embodiments: `human_bimanual` only
- rigs: `mecka`, `aria_gen1`
- SQL labels stored in the CSV for later eval only — **not** used in the behavioral representation

## Confirmed runtime fields

Cartesian `Human` batch (official transform list, stride=1):

| key | shape | dtype |
| --- | --- | --- |
| `observations.images.front_img_1` | Mecka `[1,3,360,640]`; Aria `[1,3,480,640]` | float32, range `[0,1]` |
| `actions_cartesian` | `[1,100,12]` | float32, finite |
| `observations.state.ee_pose` | `[1,12]` | float32, finite |
| `intrinsics` | `[1,3,4]` | float32, real K |
| `embodiment` | `[1]` | int64, value `3` (`HUMAN_BIMANUAL`) |
| `episode_hash` | str | |

On-disk zarr (not in cartesian batch): `left/right.obs_keypoints`, `left/right.obs_wrist_pose`, `obs_head_pose`, `annotations`. Aria additionally has `left/right.obs_aria_keypoints`, `obs_eye_gaze`, `obs_rgb_timestamps_ns`. Absent on inspected episodes: wrist RGB, `cmd_ee_pose`.

Current `EMBODIMENT` enum: `human_right_arm`, `human_left_arm`, `human_bimanual`, `eva_*`. SQL also has `yam_bimanual` (not in enum; excluded). Source is SQL `lab` / `rig_name`, not vendor embodiment strings.

SQL columns (446,957 rows): `episode_hash`, `operator`, `lab`, `num_frames`, `task`, `task_description`, `scene`, `objects`, `is_deleted`, `embodiment`, `is_eval`, `eval_score`, `eval_success`, `zarr_processed_path`, `zarr_processing_error`, `zarr_mp4_path`, `license`, `segments`, `created_at`, `updated_at`, `rig_name`.

## Exact commands that work

```bash
git clone --depth 1 https://github.com/GaTech-RL2/EgoVerse.git third_party/EgoVerse
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e third_party/EgoVerse --no-deps
# aws configure as egoverse-public, region us-east-2
bash third_party/EgoVerse/egomimic/utils/aws/setup_secret.sh
python scripts/audit_data.py --sample
python scripts/audit_data.py --inspect --inspect-n 3
```

Adapter load path:

```python
from egoselect.dataset import build_multidataset, make_loader
ds = build_multidataset(["692e9858af371a654ce7fc3f"])
batch = next(iter(make_loader(ds)))
```

## Artifacts

- `outputs/sampled_episodes.csv`
- `outputs/cohort_summary.json`
- `outputs/sql_schema.json`
- `outputs/runtime_schema.json`
- `egoselect/dataset.py`
- `scripts/audit_data.py`
- local cache (gitignored): `data/egoverse_cache/` (4 episodes), `data/episode_table.parquet`

## Known limitations

- Remaining 76 cohort hashes are not cached; they are loadable on demand via `S3EpisodeResolver` (do not use stale `sync_s3.py` presets that filter `embodiment=='aria'|'mecka'`).
- Cartesian batches omit keypoints/gaze even when present on disk. Interaction features stay optional.
- Mixed Mecka/Aria loaded with Human stride=1 (Aria configs often use 3). Fine for Phase 1 load; revisit for motion stats.
- `S3EpisodeResolver` re-queries the full SQL table on every resolve (~20s).
- Slim env only — not the full EgoVerse training stack.
- No DINOv2 / PCA / selector / frontend in this phase.

## Next phase

Phase 2 — feature extraction and representation
