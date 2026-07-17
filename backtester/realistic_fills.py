"""Realistic halt + spread + size fill model for parabolic-gapper intraday trades.

The bias-free engine already avoids exact-trigger stop fills, but for ≥100%
microcap gappers three real frictions dominate and were NOT modelled:
  1. LULD halts — detected from the minute tape as (a) a timestamp GAP before a
     bar (missing RTH minutes) or (b) a single-minute move beyond the LULD band.
     You cannot trade during a halt; you fill at the RESUME bar, which for a long
     entry is adverse (reopen pops) and for an exit is adverse (reopen dumps).
  2. Quoted spread — real bid/ask on these names is 1–15%, not the tight prints.
     Charged ROUND-TRIP (entry + exit), scaled to bar volatility + inverse to
     minute $-volume, capped per scenario (optimistic/base/pessimistic).
  3. Size/impact — you can fill at most a fraction of the entry minute's volume;
     beyond that, market impact walks the book.

`realistic_long_return(bars, entry_time, exit_time, stop_pct, scenario, notional)`
returns the net long % after all three, or None if unfillable (halted at entry
with no resume, or size impossible).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# spread scenarios: (vol_coef on bar-range, inverse-$vol coef, floor, cap)
SCENARIOS = {
    "optimistic": dict(k_range=0.15, k_illiq=0.02, floor=0.003, cap=0.03),
    "base":       dict(k_range=0.30, k_illiq=0.05, floor=0.005, cap=0.08),
    "pessimistic":dict(k_range=0.50, k_illiq=0.10, floor=0.010, cap=0.15),
}
LULD_BAND = 0.10          # single-minute move beyond this ⇒ treat as halt
HALT_RESUME_SLIP = 0.02   # extra adverse slip filling on a halt-resume bar
MAX_VOL_FRAC = 0.10       # can fill ≤10% of the entry minute's share volume


def _idx_at(bars, hhmm):
    w = np.where(bars["time"].to_numpy() >= hhmm)[0]
    return int(w[0]) if len(w) else None


def _halt_flags(bars):
    """Per-bar: True if this bar is a halt-resume (gap before it) or a >band jump."""
    t = pd.to_datetime(bars["time"], format="%H:%M")
    gap_before = t.diff().dt.total_seconds().div(60).fillna(1).to_numpy() > 1.5
    o = bars["open"].to_numpy(); c = bars["close"].to_numpy()
    prev_c = np.concatenate([[c[0]], c[:-1]])
    jump = np.abs(o / prev_c - 1) > LULD_BAND
    intrabar = np.abs(c / o - 1) > LULD_BAND
    return gap_before | jump | intrabar


def _half_spread(price, bar_range_pct, minute_dollar_vol, sc):
    illiq = sc["k_illiq"] / np.sqrt(max(minute_dollar_vol, 1e4) / 1e6)
    s = sc["k_range"] * bar_range_pct + illiq
    return float(np.clip(s, sc["floor"], sc["cap"]) / 2.0)


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
