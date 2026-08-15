# Build state

Current phase: **Phase 2 complete**

## Completed

- Sparse RGB sampling (8 uniform frames) via official `ZarrEpisode.read`; JPEGs decoded only at those indices
- Frozen `facebook/dinov2-small` CLS embeddings, mean-pooled and L2-normalized
- Motion stats from full `left/right.obs_ee_pose` xyz (source of cartesian `ee_pose`)
- Transparent quality score + components
- Visual embeddings cached separately from selector weights
- Multimodal `z_i`, 2D PCA layout, KMeans coverage regions
- Modal skipped (not already configured)

## Feature artifact

`outputs/episode_features.parquet`

## Episodes successfully featurized

**80 / 80** cohort episodes (all 8 RGB frames decoded)

## Visual embedding

- model: `facebook/dinov2-small` (no training)
- dimensions: 384
- PCA dimensions: 16 (before concat)

Cache: `data/visual_embeddings.parquet` (gitignored). Re-running representation does not require DINO.

## Motion features

`path_length`, `displacement`, `mean_speed`, `speed_std`, `max_speed`, `mean_abs_accel`, `direction_change_ratio`, `stationary_ratio`, `duration_s`, `n_frames`

From bimanual xyz at metadata `fps` (30 on inspected episodes). Structured motion/quality/z/xy: **0 NaNs**. The only parquet NaNs are 17 empty SQL `scene` labels (eval metadata, not in `z_i`).

## Optional interaction

**Not used.** Keypoints/wrist/gaze exist on disk but are absent from the cartesian runtime batch. No grasp/contact signals were fabricated.

## Quality definition

```
0.25·valid_frame_ratio
+ 0.25·trajectory_completeness
+ 0.20·finite_pose_ratio
+ 0.15·(1 − stationary_ratio)
+ 0.15·temporal_validity
```

Median quality ≈ 0.998 (cohort already filtered to complete human_bimanual zarrs). Stationary ratio range 0.00–0.19.

## Representation dimensions

`z_i` is 26-D: 16 visual-PCA + 10 standardized motion, then re-standardized. Frontend: `pca_x`, `pca_y`.

## Behavioral region count

**6** KMeans regions on `z_i` (representation-space coverage regions, not semantic skills). Sizes: 3, 17, 12, 8, 24, 16.

## Exact commands

```bash
source .venv/bin/activate
python scripts/build_features.py          # sync missing zarrs, then featurize
python scripts/build_features.py --no-sync  # local cache already complete
```

## Known limitations

- DINOv2-small, not giant/base; 8 frames, not the full video
- Human cartesian stride not applied to raw ee_pose motion (full-rate xyz)
- Interaction omitted on purpose
- Quality is weakly discriminative on this already-clean cohort
- Region 0 has only 3 episodes
- Modal not used
- SQL task/lab/scene/operator stored for later eval only; not inside `z_i`

## Next phase

Phase 3 — selection, baselines, validation, corruption
