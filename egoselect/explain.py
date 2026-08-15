"""Deterministic KEEP/DROP reasons from measured score parts. No LLM."""

from __future__ import annotations

from egoselect.selector import StepRecord


def explain_step(rec: StepRecord) -> str:
    bits: list[str] = []
    if rec.new_region >= 0.5:
        bits.append("Underrepresented behavior")
    elif rec.n_selected_before > 0:
        bits.append("Behavior already represented")
    if rec.quality_norm >= 0.80:
        bits.append("High quality")
    if rec.redundancy >= 0.70:
        bits.append("High redundancy")
    elif rec.n_selected_before > 0 and rec.redundancy < 0.35:
        bits.append("Low redundancy")
    if rec.stationary_ratio >= 0.10:
        bits.append(f"{100.0 * rec.stationary_ratio:.0f}% stationary")
    if not bits:
        bits.append(
            f"Value={rec.value:.3f} (Q={rec.quality:.3f}, "
            f"C={rec.coverage_gain:.3f}, R={rec.redundancy:.3f})"
        )
    return "; ".join(bits)
