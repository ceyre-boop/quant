"""Exchange mechanics as theorems. LULD bands (Reg NMS Plan) are deterministic; encode them in Z3 and
prove per-entry bounds instead of simulating them.

max_loss_before_halt(entry, t, tier): the largest adverse move a position can suffer before a LULD
halt MUST have triggered, given the band at that price/time and the reference-price rule (trailing
5-minute mean). Bound = band(entry) applied to the least favourable reference within the window, i.e.
the reference can lag a fast move by up to the 5-bar mean — the formal answer to HYP-097's "declared
worst case beyond the stop". SSR (Rule 201) is stated as a constraint: after a −10% day, short sales
must execute above the NBB — encoded as 'no short fill at or below the bid'."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

import z3

from backtester.luld import (luld_band, PRICE_TIER_HIGH, PRICE_TIER_LOW, TIER1_BAND, TIER2_BAND, MID_BAND, LOW_ABS, LOW_PCT,
                             OPEN_DOUBLE, CLOSE_DOUBLE, REFERENCE_LOOKBACK)


def _z3_band(price: z3.ArithRef, doubled: bool, tier: int) -> z3.ArithRef:
    base = z3.If(price >= PRICE_TIER_HIGH, z3.RealVal(TIER1_BAND if tier == 1 else TIER2_BAND),
                 z3.If(price >= PRICE_TIER_LOW, z3.RealVal(MID_BAND),
                       z3.If(LOW_ABS / price < LOW_PCT, LOW_ABS / price, z3.RealVal(LOW_PCT))))
    return base * 2 if doubled else base


def _doubled(t: dtime) -> bool:
    return OPEN_DOUBLE[0] <= t < OPEN_DOUBLE[1] or CLOSE_DOUBLE[0] <= t < CLOSE_DOUBLE[1]


@dataclass
class HaltBound:
    entry: float
    band: float
    max_adverse_frac: float      # proven: adverse excursion before a halt must trigger, as a fraction of entry
    doubled: bool
    tier: int
    proof: str


def max_loss_before_halt(entry: float, t: dtime, tier: int = 2, prior_closes: list[float] | None = None) -> HaltBound:
    """Prove: for any path, the price cannot trade more than (1+band)·ref_max above / (1−band)·ref_min below
    without a halt, where ref is the trailing 5-bar mean. With the 5 prior closes given, the worst reference
    is bounded by their extremes; without them, the reference is taken as the entry (tight case) and the
    bound is band(entry) — the classic 'one band' worst case, plus the lag term."""
    doubled = _doubled(t)
    band_val = luld_band(entry, t, tier)
    s = z3.Solver()
    p = z3.Real("p"); ref = z3.Real("ref"); e = z3.RealVal(entry)
    band = _z3_band(ref, doubled, tier)
    # a long is halted (limit-down) when p <= ref*(1-band); before that, p > ref*(1-band)
    if prior_closes:
        lo, hi = min(prior_closes + [entry]), max(prior_closes + [entry])
        s.add(ref >= lo, ref <= hi)
    else:
        s.add(ref == e)
    s.add(p > ref * (1 - band))                  # not yet halted
    s.add(p < e)                                 # adverse
    loss = (e - p) / e
    # maximise loss: binary search on a bound b with Z3 (exact rationals)
    lo_b, hi_b = 0.0, 1.0
    for _ in range(40):
        mid = (lo_b + hi_b) / 2
        s.push(); s.add(loss >= mid)
        if s.check() == z3.sat: lo_b = mid
        else: hi_b = mid
        s.pop()
    return HaltBound(entry, band_val, hi_b, doubled, tier,
                     f"Z3: sup of (entry−p)/entry subject to p > ref·(1−band(ref)), ref ∈ [min,max] of trailing {REFERENCE_LOOKBACK} closes → {hi_b:.4f}")


def ssr_constraint(prev_day_return: float) -> dict:
    """Rule 201: if the security fell ≥10% from the prior close, short sales for the rest of the day and
    the next day must be at a price above the current national best bid."""
    active = prev_day_return <= -0.10
    return {"ssr_active": active, "constraint": "short fills must be > NBB (no hitting the bid)" if active else "none"}
