# EgoSelect

Capability-aware training-value curation for EgoVerse.

## Problem

Which demonstrations are worth spending training compute on? EgoVerse mixes useful motion with near-duplicates and idle-heavy episodes. This project ranks episodes by **marginal training value**. It does not train a policy and does not claim downstream improvement.

## Method

Greedy selection over multimodal episode embeddings `z_i` (DINOv2 visual PCA + standardized motion). After each pick, remaining candidates are rescored against the updated set `S`. SQL labels are not used for scoring.

## Formula

```
Value(i | S) = 0.35·Quality(i) + 0.45·CoverageGain(i | S) − 0.20·Redundancy(i | S)
```

- **Quality:** valid frames, completeness, finite pose, non-stationary content, temporal validity
- **CoverageGain:** new region + distance to `S` + underrepresented-region bonus
- **Redundancy:** max cosine similarity to `S`

## Architecture

```
EgoVerse SQL/S3 → local zarr cache
  → features (DINOv2 + motion + quality)
  → z_i + 2D PCA + representation-space regions
  → greedy EgoSelect ranking
  → equal-budget baselines + feature-layer corruption
  → demo_payload.json
  → one-screen React / Vite / SVG frontend (no API)
```

## Validation

Four methods, same keep counts (8 / 24 / 40). Primary budget **24 / 80 (30%)**. Corruption injects duplicates, idle-heavy, and overrepresented-region rows at the feature layer only.

## Measured result

At 30% keep, EgoSelect is **not uniformly best**.

| Method | Coverage | Quality | Redundancy | Injected retained |
| --- | ---: | ---: | ---: | ---: |
| EgoSelect | 1.000 | 0.999 | 0.406 | **3 / 24** |
| Dedup | 1.000 | 0.995 | **0.377** | 4 / 24 |
| Diversity | 1.000 | 0.993 | 0.485 | 7 / 24 |
| Random | 0.833 | 0.995 | 0.534 | 8 / 24 |

EgoSelect matches full region coverage, highest mean quality, and fewest injected items. Dedup has lower redundancy. Random misses a region.

## How to run

```bash
source .venv/bin/activate
python scripts/audit_data.py --sample          # SQL cohort
python scripts/build_features.py --no-sync     # cached zarrs
python scripts/run_selection.py
python scripts/run_experiment.py
python scripts/export_demo.py
cd web && npm install && npm run dev           # http://localhost:5173
```

Frontend reads only `/data/demo_payload.json`. If that file is missing, generate it with `python scripts/export_demo.py`.

## Limitations

Quality is weakly discriminative on this clean cohort (0.971–1.000). Motion-coverage saturates. Region 0 has only 3 episodes. Corruption is synthetic. Coverage here is representation-space region coverage, not semantic skill coverage, and not a claim about trained-policy performance.
