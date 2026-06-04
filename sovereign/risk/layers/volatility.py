"""Layer 3 — VOLATILITY TARGETING. factor = clamp(baseline/estimate, floor, 1.0).

When current vol > baseline → factor < 1 (shrink). When vol <= baseline → 1.0 (NEVER amplify).
ATR/EWMA-agnostic: state supplies vol_estimates & vol_baseline; swap the estimator behind this
interface later. Missing vol data → 1.0 (no vol-based reduction; other layers still bind). Invalid
(<=0) vol → raise (fail loud; never silently size on garbage).
"""
from __future__ import annotations


def factor(signal, state, cfg) -> float:
    floor = cfg["volatility"]["factor_floor"]
    inst = signal.instrument
    est = state.vol_estimates.get(inst)
    base = state.vol_baseline.get(inst)
    if est is None or base is None:
        return 1.0
    if est <= 0 or base <= 0:
        raise ValueError(f"volatility layer: invalid vol for {inst} (est={est}, base={base})")
    return max(floor, min(base / est, 1.0))
