"""Deterministic KEEP/DROP reasons from measured score parts. No LLM."""

from __future__ import annotations

from egoselect.selector import StepRecord


def explain_step(rec: StepRecord) -> str:
    bits: list[str] = []
    if rec.redundancy >= 0.70 and rec.nearest_selected_episode:
        pct = 100.0 * rec.redundancy
        bits.append(f"{pct:.0f}% similar to {rec.nearest_selected_episode}")
    if rec.n_selected_before > 0 and rec.new_region < 0.5:
        bits.append("Region already represented")
    if rec.stationary_ratio >= 0.10:
        bits.append(f"High stationary content ({100.0 * rec.stationary_ratio:.0f}%)")
    if rec.quality_norm >= 0.80:
        bits.append(f"High quality ({rec.quality:.3f})")
    if rec.new_region >= 0.5:
        bits.append(f"Underrepresented region {rec.behavioral_region}")
    if rec.n_selected_before > 0 and rec.redundancy < 0.35:
        bits.append("Low redundancy")
    if rec.n_selected_before > 0 and rec.region_balance >= 0.75:
        bits.append(f"Region-balance bonus (region {rec.behavioral_region})")
    if not bits:
        bits.append(
            f"Value={rec.value:.3f} (Q={rec.quality:.3f}, "
            f"C={rec.coverage_gain:.3f}, R={rec.redundancy:.3f})"
        )
    return "; ".join(bits)
