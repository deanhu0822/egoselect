# Build state

Current phase: **0**

## Completed

- Repository inspected

## Existing reusable code

None. This repo is a fresh `git init` on `master`: no commits, no remote, no source files, no `.gitignore`.

## Environment

- Machine: Apple M4 Max, 36 GB RAM, ~416 GB free
- Python 3.11.4 (system Frameworks). `uv` 0.10.7 present. No conda, no venv, no EgoVerse env
- Installed: numpy, Pillow. Missing: torch, pandas, sklearn, zarr, sqlalchemy, opencv, transformers, DINOv2
- Node 22.15.1 / npm 10.9.2 (frontend-ready)
- No AWS CLI, no `~/.aws`, no `s5cmd`, no ffmpeg
- No EgoVerse clone on this machine (checked Desktop / Documents / expected `~/Documents/rl2/projects/EgoVerse`)

## Known EgoVerse setup

Official access (do not invent a parallel storage layer):

```
create_default_engine() → episode_table_to_df(engine) → episode_hash
DatasetFilter → S3EpisodeResolver | LocalEpisodeResolver → MultiDataset → DataLoader
```

- SQL lives in PostgreSQL `app.episodes`. `create_default_engine()` requires `SECRETS_ARN` via EgoVerse `setup_secret.sh` (Secrets Manager), then `postgresql+psycopg`.
- `TableRow` fields include: `episode_hash`, `operator`, `lab`, `task`, `embodiment`, `rig_name`, `num_frames`, `task_description`, `scene`, `objects`, `zarr_processed_path`, `zarr_mp4_path`, `is_deleted`, `is_eval`, …
- `DatasetFilter.matches()` always drops `is_deleted` rows.
- Cartesian tutorial batch keys: `observations.images.front_img_1`, `actions_cartesian`, `observations.state.ee_pose`, `embodiment`, `intrinsics`, `episode_hash`.
- Human cartesian vs keypoints/gaze/annotations require different `Human.get_keymap` / `get_transform_list` modes. The keypoints cell in the official notebook currently fails (`filters matched no episodes`). Treat interaction signals as optional until audited.
- 2026-07-08 collapse: human data is `human_*`; source is SQL `lab`, not embodiment. Intrinsics are mandatory. Stale local zarrs (`aria_bimanual`, etc.) hard-crash (`KeyError: 'ARIA_BIMANUAL'`). Re-download required.
- `sync_s3.py` named presets still filter `embodiment == 'aria'|'mecka'`. Those presets are likely stale; prefer `lab` + `human_*` filters after SQL audit.
- Manual download: `S3EpisodeResolver.sync_from_filters(bucket_name="rldb", ...)`.

## Blocking issues

- No EgoVerse checkout, no cloud credentials, no SQL access, no local episodes
- `sync_s3.py` presets vs embodiment collapse: do not use named filters blindly
- Interaction/keypoint availability is unproven; cartesian RGB + motion is the safe default
- System Python lacks the ML/data stack; Phase 1 should use a dedicated `uv` env (EgoVerse wants 3.11)

## Next phase

Phase 1 — EgoVerse acquisition and cohort construction
