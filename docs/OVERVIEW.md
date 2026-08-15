# EgoSelect

Capability-aware training-value curation for EgoVerse.

**Question:** which EgoVerse demonstrations are worth spending training compute on?

EgoSelect estimates **marginal training value** using transparent proxy metrics over quality, behavioral coverage, and redundancy. It does not claim semantic skill discovery, downstream training improvement, optimality, or 90%+ coverage unless those are actually measured.

## Core algorithm

```
Value(i | S) = α · Quality(i) + β · CoverageGain(i | S) − γ · Redundancy(i | S)
```

`S` is the subset already selected. This is **greedy marginal selection**, not a static ranking: after each episode is chosen, remaining candidates are rescored against the updated `S`.

Default weights: `α = 0.35`, `β = 0.45`, `γ = 0.20`.

- **Redundancy:** max cosine similarity to the selected set.
- **Coverage gain:** new-region bonus + distance to selected examples + underrepresented-region bonus.
- Store the **complete** greedy ordering (powers the frontend retention slider). Do not store only a 30% subset.

## Source of truth

Dataset access and schema follow the current EgoVerse repository and official tutorials:

1. https://github.com/GaTech-RL2/EgoVerse/
2. https://partners.mecka.ai/egoverse
3. `egomimic/scripts/tutorials/zarr_data_viz.ipynb`
4. `egomimic/scripts/tutorials/sql_tutorial.ipynb`

Do not invent a custom EgoVerse storage abstraction before checking that implementation.

**Metadata**

```
create_default_engine() → episode_table_to_df(engine) → select episode_hash
```

**Episodes**

```
DatasetFilter → S3EpisodeResolver / LocalEpisodeResolver → MultiDataset → DataLoader
```

Known cartesian runtime keys (inspect before assuming optional modalities exist):

`observations.images.front_img_1`, `actions_cartesian`, `observations.state.ee_pose`, `embodiment`, `intrinsics`, `episode_hash`

## Boundaries

Target: ~50–100 real EgoVerse episodes.

Selector inputs come primarily from demonstration content: RGB, motion/state/action, quality signals, and optional interaction signals **when actually present**. Never fabricate unavailable signals.

SQL labels (`task`, `scene`, `lab`, `operator`) are for sampling, stratification, and retrospective evaluation. They are **not** inputs to the main behavioral coverage score.

## Features and representation

| Stream | Method |
| --- | --- |
| Visual | ~8–12 frames → pretrained DINOv2 → mean-pooled episode embedding |
| Motion | Compact trajectory stats from `actions_cartesian` and/or `ee_pose` (path length, displacement, mean speed, speed variance, acceleration, direction changes, stationary ratio) |
| Interaction | Optional; only if left/right/keypoint/contact fields exist in real data |
| Quality | Transparent score from valid-frame ratio, trajectory completeness, missingness, stationary content, temporal validity |

```
visual embedding → PCA
+ normalized motion
+ optional normalized interaction
→ multimodal z_i
```

PCA → 2D for frontend visualization. KMeans or density grouping over representation space defines **behavioral / coverage / representation-space regions**. Do not call these ground-truth semantic skills.

## Validation

Four equal-budget strategies: **Random**, **Dedup-only**, **Diversity-only**, **EgoSelect**.

Primary proxies: behavioral coverage, average quality, nearest-neighbor redundancy, stationary-content ratio, visual coverage, motion coverage.

Secondary SQL sanity checks when available: task / scene / lab / operator diversity retained.

Controlled corruption (duplicates, idle-heavy examples, overrepresented-region examples) measures how many corrupted examples each strategy retains. Do not modify original EgoVerse files.

## Frontend

React + Vite + TypeScript, native SVG, custom CSS. No backend API.

Python exports `web/public/data/demo_payload.json`; the frontend reads only that artifact.

Primary interaction: retention 100% → 30%. Episodes fade by greedy rank while metrics update.

Main view: 2D behavior-space episode field. Secondary: KEEP/DROP inspector, benchmark comparison, stress-test toggle.

Visual direction: minimal, modern, editorial research instrument. Large negative space, circular episode geometry, restrained typography. Not a SaaS dashboard, sidebar, card grid, rainbow plot, glassmorphism, or heavy component library.

## Three-hour priority

1. Real EgoVerse data → 2. Features → 3. Representation → 4. Greedy selector → 5. Baselines → 6. Evaluation → 7. Corruption test → 8. JSON artifact → 9. Frontend → 10. README/demo

If behind: cut animations, optional interaction features, and Modal. Never cut the greedy selector, equal-budget baselines, real measurements, or controlled corruption.

## Repository layout

```
egoselect/     dataset.py, features.py, representation.py, scoring.py,
               selector.py, explain.py, baselines.py, metrics.py, corruption.py
scripts/       audit_data.py, build_features.py, run_selection.py,
               run_experiment.py, export_demo.py
outputs/       sampled_episodes.csv, episode_features.parquet, selections.csv,
               metrics.json, experiment_results.csv, demo_payload.json
web/           React/Vite/TypeScript frontend
docs/          OVERVIEW.md, BUILD_STATE.md
```
