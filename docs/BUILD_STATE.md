# Build state

Current phase: **Phase 3 complete**

## Algorithm

Greedy marginal selection over multimodal `z_i`. After every addition to `S`, remaining candidates are rescored. Verified: 80 rescore passes for 80 episodes; second pick ≠ second-highest quality; max |value shift| after first pick = 0.537.

```
Value(i | S) = 0.35·Q + 0.45·CoverageGain(i|S) − 0.20·Redundancy(i|S)
CoverageGain = 0.50·new-region + 0.30·min-distance + 0.20·region-balance
Redundancy = max cosine(z_i, z_S)  (0 if S empty)
```

Quality is min-max normalized `quality_score` (higher = better). SQL labels are not used in scoring.

## Weights

α=0.35, β=0.45, γ=0.20. Coverage mix 0.50 / 0.30 / 0.20. Untuned.

## Validation

Four methods, identical keep counts: 8 / 24 / 40 at 10% / 30% / 50%. Primary budget **24**. Corruption injected at the feature layer only (0/10/20/30% of N); keep stays 24.

## Actual headline result

At 30% keep (24/80), **EgoSelect is not uniformly best**.

| Method | Keep | Coverage | Quality | Redundancy | Corruption retained |
| --- | --- | --- | --- | --- | --- |
| EgoSelect | 24 | 1.000 | 0.999 | 0.406 | 3/24 @30% inject |
| Dedup-only | 24 | 1.000 | 0.995 | 0.377 | 4/24 @30% inject |
| Diversity-only | 24 | 1.000 | 0.993 | 0.485 | 7/24 @30% inject |
| Random | 24 | 0.833 | 0.995 | 0.534 | 8/24 @30% inject |

Measured: EgoSelect matches full region coverage, has the highest mean quality, and retains the fewest injected items at 30% inject. Dedup-only has lower nearest-neighbor redundancy. Random misses a region (coverage 0.833). Quality gaps are small because cached quality is already compressed (0.971–1.000).

## Artifacts

- `outputs/selections.csv` (complete ranks 1–80)
- `outputs/metrics.json`
- `outputs/experiment_results.csv`
- `outputs/greedy_audit.json`

## Known weaknesses

- Quality is weakly discriminative on this clean cohort
- Motion-coverage proxy saturates at 1.0 for all methods
- Dedup beats EgoSelect on redundancy
- Region 0 has only 3 episodes, so coverage is easy to fill
- Corruption is synthetic at the feature layer, not real bad demos

## Best KEEP example

- episode: `692e90f07641010d043544f1`
- reason: High quality (1.000); Underrepresented region 4

## Best DROP example

- episode: `692f13d3f59fd218b1be026e`
- reason: Region already represented; High stationary content (19%)

## Exact commands

```bash
source .venv/bin/activate
python scripts/run_selection.py
python scripts/run_experiment.py
```

## Next phase

Phase 4 — export, frontend, smoke test
