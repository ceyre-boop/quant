"""Realistic halt + spread + size fill model for parabolic-gapper intraday trades.

The bias-free engine already avoids exact-trigger stop fills, but for ≥100%
microcap gappers three real frictions dominate and were NOT modelled:
  1. LULD halts — detected from the minute tape as (a) a timestamp GAP before a
     bar (missing RTH minutes) or (b) a single-minute move beyond the LULD band.
     You cannot trade during a halt; you fill at the RESUME bar, which for a long
     entry is adverse (reopen pops) and for an exit is adverse (reopen dumps).
  2. Quoted spread — charged ROUND-TRIP (entry + exit).
     CORRECTED 2026-07-18 (TICK-039). This docstring previously asserted "real
     bid/ask on these names is 1–15%". That was never measured and is wrong.
     313 real NBBO observations at the 09:31 entry instant give p10 0.13% /
     median 0.55% / p90 2.06% / p99 5.01%. The old scenario model charged a
     median 6.21% round-trip against a measured 0.55% — an 11.3x overcharge that
     biased every gapper backtest PESSIMISTIC. See MEASURED_SPREAD below.
  3. Size/impact — you can fill at most a fraction of the entry minute's volume;
     beyond that, market impact walks the book.

`realistic_long_return(bars, entry_time, exit_time, stop_pct, scenario, notional)`
returns the net long % after all three, or None if unfillable (halted at entry
with no resume, or size impossible).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.luld import halt_flags as _tiered_halt_flags

# spread scenarios: (vol_coef on bar-range, inverse-$vol coef, floor, cap)
SCENARIOS = {
    "optimistic": dict(k_range=0.15, k_illiq=0.02, floor=0.003, cap=0.03),
    "base":       dict(k_range=0.30, k_illiq=0.05, floor=0.005, cap=0.08),
    "pessimistic":dict(k_range=0.50, k_illiq=0.10, floor=0.010, cap=0.15),
}
LULD_BAND = 0.10          # DEPRECATED — see backtester/luld.py. Retained only so
                          # external importers do not break at import time. The
                          # flat band is NO LONGER USED by _halt_flags().
HALT_RESUME_SLIP = 0.02   # extra adverse slip filling on a halt-resume bar
MAX_VOL_FRAC = 0.10       # can fill ≤10% of the entry minute's share volume


def _idx_at(bars, hhmm):
    w = np.where(bars["time"].to_numpy() >= hhmm)[0]
    return int(w[0]) if len(w) else None


def _halt_flags(bars, tier: int = 2):
    """Per-bar: True if this bar is a halt-resume.

    Delegates to the tiered LULD implementation. The previous flat 10% band
    flagged some microcap opening minutes as halts and charged HALT_RESUME_SLIP
    against them, biasing backtests PESSIMISTIC.

    Measured magnitude: ~0.071% per trade at the 09:31 entry bar (old 3.6% of
    events flagged vs 0.0% under the tiered rule, n=56), and no measurable effect
    at 10:30 (n=59). Real, small, and directionally optimistic when corrected —
    not a result-changing fix. See backtester/luld.py for the rule and
    data/agent/param_change_log.jsonl for the rationale.
    """
    return _tiered_halt_flags(bars, tier=tier)


# ── Measured spread model (TICK-039) ──────────────────────────────────────────
# Fitted on 313 REAL NBBO observations at the 09:31 entry instant, drawn from
# MINING-period gapper events only (holdout untouched). Harness commit 36b3902;
# collection research/gapper/tick039_collect.py, fit research/gapper/tick039_fit.py.
#
#   log(half_spread) = a + b*log(price) + c*log(minute_$vol) + d*log(bar_range)
#   R^2(log) = 0.404, residual sd(log) = 0.878
#
# Calibration at the entry leg (mining sample, n=313):
#   measured median round-trip   0.5510%
#   this model                   0.5099%
#   LEGACY SCENARIOS model       6.2060%   <-- 11.3x overcharge
#
# The legacy model assumed "1-15%" quoted spreads (see module docstring) and
# saturated its 8% round-trip cap on gapper opens. Real quoted spreads on these
# names are p10 0.13% / median 0.55% / p90 2.06% / p99 5.01%.
#
# Caps are OBSERVED percentiles, not hand-picked: floor = p1, cap = p99. The
# model is therefore not uniformly optimistic — genuinely wide events still get
# charged up to 5.4% round-trip.
MEASURED_SPREAD = {
    "intercept": 0.324468,
    "log_price": 0.193418,
    "log_dollar_vol": -0.383361,
    "log_bar_range": 0.368907,
    "floor": 0.00009,      # p1 half-spread
    "cap": 0.02707,        # p99 half-spread (5.41% round-trip)
    "n_obs": 313,
    "r2_log": 0.404,
    "source": "TICK-039 / harness 36b3902",
}


def _half_spread_measured(price, bar_range_pct, minute_dollar_vol):
    """Half-spread from the fitted measured-quote model. See MEASURED_SPREAD."""
    m = MEASURED_SPREAD
    lv = (m["intercept"]
          + m["log_price"] * np.log(max(float(price), 1e-6))
          + m["log_dollar_vol"] * np.log(max(float(minute_dollar_vol), 1e3))
          + m["log_bar_range"] * np.log(max(float(bar_range_pct), 1e-4)))
    return float(np.clip(np.exp(lv), m["floor"], m["cap"]))


def _half_spread_legacy(price, bar_range_pct, minute_dollar_vol, sc):
    """The pre-TICK-039 scenario model. Retained for A/B only — it overcharges
    by ~11x against measured quotes. Do not use for new work."""
    illiq = sc["k_illiq"] / np.sqrt(max(minute_dollar_vol, 1e4) / 1e6)
    s = sc["k_range"] * bar_range_pct + illiq
    return float(np.clip(s, sc["floor"], sc["cap"]) / 2.0)


def _half_spread(price, bar_range_pct, minute_dollar_vol, sc, measured=True):
    """Half-spread charged per leg.

    Defaults to the measured model. Pass measured=False for the legacy scenario
    model when reproducing a pre-TICK-039 number.
    """
    if measured:
        return _half_spread_measured(price, bar_range_pct, minute_dollar_vol)
    return _half_spread_legacy(price, bar_range_pct, minute_dollar_vol, sc)


def realistic_long_return(bars, entry_time, exit_time, stop_pct,
                          scenario="base", notional=None):
    sc = SCENARIOS[scenario]
    ei = _idx_at(bars, entry_time)
    xi = _idx_at(bars, exit_time)
    if ei is None or ei == 0:
        return None
    if xi is None or xi <= ei:
        xi = len(bars) - 1
    halts = _halt_flags(bars)
    o = bars["open"].to_numpy(); h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy(); c = bars["close"].to_numpy()
    v = bars["volume"].to_numpy()

    # --- entry: if the entry bar is a halt-resume, fill worse (or push to next) ---
    e = ei
    entry_px = o[e]
    if halts[e]:
        entry_px = o[e] * (1 + HALT_RESUME_SLIP)   # buy into a pop
    # size feasibility
    if notional is not None:
        shares = notional / max(entry_px, 1e-6)
        if shares > MAX_VOL_FRAC * v[e]:
            # cannot fill full size at this minute — model partial as extra impact
            impact = min((shares / max(MAX_VOL_FRAC * v[e], 1) - 1) * 0.02, 0.10)
            entry_px *= (1 + impact)

    br = (h[e] - lo[e]) / o[e] if o[e] else 0.0
    hs_entry = _half_spread(entry_px, br, o[e] * v[e], sc)

    # --- stop / exit walk with halt-aware fills ---
    trigger = entry_px * (1 - stop_pct)
    stop_px = None
    for j in range(e + 1, xi + 1):
        if lo[j] <= trigger:
            # gap-through OR halt-resume below trigger ⇒ fill at that bar open
            stop_px = min(o[j], trigger)
            if halts[j]:
                stop_px = min(stop_px, o[j]) * (1 - HALT_RESUME_SLIP)
            break
    if stop_px is not None:
        exit_px = stop_px
        hs_exit = _half_spread(exit_px, (h[j]-lo[j])/max(o[j],1e-6), o[j]*v[j], sc)
    else:
        exit_px = c[xi]
        if halts[xi]:
            exit_px *= (1 - HALT_RESUME_SLIP)     # forced exit into a halt-resume
        hs_exit = _half_spread(exit_px, (h[xi]-lo[xi])/max(o[xi],1e-6),
                               o[xi]*v[xi], sc)

    gross = exit_px / entry_px - 1
    return gross - hs_entry - hs_exit             # round-trip spread charged
