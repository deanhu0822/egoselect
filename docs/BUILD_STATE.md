# Build state

Status: **DEMO READY**

Current phase: **Phase 4 complete**

## Algorithm

Unchanged from Phase 3. Greedy marginal selection over multimodal `z_i`. 80 rescore passes; second pick ≠ second-highest quality; max |value shift| after first pick = 0.537.

```
Value(i | S) = 0.35·Q + 0.45·CoverageGain(i|S) − 0.20·Redundancy(i|S)
```

## Measured headline

At 30% keep (24/80), EgoSelect matches full region coverage (1.000), highest mean quality (0.999), and fewest injected corruptions (3/24). Dedup-only still wins redundancy (0.377 vs 0.406). Random misses a region (0.833). EgoSelect is not uniformly best.

## Commands

```bash
source .venv/bin/activate
python scripts/audit_data.py --sample
python scripts/build_features.py --no-sync
python scripts/run_selection.py
python scripts/run_experiment.py
python scripts/export_demo.py
cd web && npm install && npm run build && npm run dev
```

## Artifacts

- `outputs/sampled_episodes.csv`
- `outputs/episode_features.parquet`
- `outputs/selections.csv` (ranks 1–80)
- `outputs/metrics.json`
- `outputs/experiment_results.csv`
- `outputs/greedy_audit.json`
- `outputs/demo_payload.json`
- `web/public/data/demo_payload.json`

## Known limitations

- Quality is weakly discriminative (0.971–1.000)
- Motion-coverage proxy saturates at ~1.0
- Dedup beats EgoSelect on redundancy
- Region 0 has only 3 episodes
- Corruption is synthetic at the feature layer
- No downstream policy training

## 60-second demo order

1. Default training budget 30% (24/80). Retained circles stay strong; the rest remain visible at low opacity.
2. Drag to 100% to show useful + repetitive demonstrations mixed together.
3. Drag back to 30%. Coverage stays 1.000. Live retained / coverage / quality / redundancy update from the greedy prefix curve.
4. Click the solid ringed KEEP: `692e90f07641010d` — high quality, underrepresented region 4.
5. Click the faded ringed DROP: `692f13d3f59fd218` — region already represented, 19% stationary.
6. Read the four-method row: Random misses a region; Dedup has lower redundancy.
7. Toggle Stress test: EgoSelect retains 3/24 injected items vs Random 8/24.
8. Close on the measured headline: full coverage, highest quality, fewest injected items — not uniformly best.

## Next phase

None. Feature freeze.
