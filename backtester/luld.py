"""Tiered LULD (Limit Up-Limit Down) bands and halt detection.

Replaces the flat 10% band previously hardcoded in `realistic_fills._halt_flags`,
which was catastrophically wrong for the securities this repo actually trades.

THE BUG THIS FIXES
------------------
The old rule was::

    jump     = abs(open / prev_close - 1) > 0.10
    intrabar = abs(close / open   - 1) > 0.10

On a microcap that just gapped +100%, a 10-15% move inside the opening minute is
ORDINARY, not a halt. The flat band therefore flagged some opening-minute bars as
halt-resumes and charged `HALT_RESUME_SLIP = 0.02` against them, for halts that
never happened. The bug biases results PESSIMISTIC; correcting it moves prior
net-return figures UP.

MEASURED MAGNITUDE — the bug is real but SMALL. Do not overstate it.
Across 60 randomly sampled events from data/research/gapper/per_candidate_enriched.csv
(seed 20260718), at the bars where entries actually occur:

    09:31 (HYP-107 LONG entry) : old flagged 2/56 = 3.6%, new 0/56 = 0.0%
                                 -> avg spurious cost removed 0.071% per trade
    10:30 (HYP-093 SHORT entry): old flagged 0/59,        new 0/59
                                 -> no measurable effect

Across the whole 09:30-09:45 window the old rule flagged 4.30% of minutes vs
2.32% for the tiered rule (46% of those flags were false positives, n=40 events).

So this fix is worth ~7 basis points on the long leg and nothing on the short. It
is a correctness fix, not a rescue: it does NOT change HYP-107's viability
question. An earlier draft of this docstring claimed "nearly every opening minute"
and "~2% per trade" — both were wrong, asserted from plausibility rather than
measurement, and are corrected here. The upward direction of any correction is
exactly where motivated reasoning enters; see the param_change_log entry.

THE ACTUAL RULE (Reg NMS Plan to Address Extraordinary Volatility)
------------------------------------------------------------------
Price bands are a percentage of a reference price (the trailing 5-minute average
of eligible prints), and depend on the security's tier and the time of day:

    Tier 1 (S&P 500, Russell 1000, select ETPs), px >= $3.00 :  5%
    Tier 2 (all other NMS securities),           px >= $3.00 : 10%
    Any tier, $0.75 <= px < $3.00                            : 20%
    Any tier, px < $0.75                        : lesser of 75% or $0.15

Bands are DOUBLED during the opening period (09:30-09:45) and the closing period
(15:35-16:00).

Every gapper this repo studies is Tier 2 and is entered at 09:31 — i.e. inside
the doubling window. The correct band at entry is therefore 20% (or 40% for
sub-$3 names, or up to 150% for sub-$0.75 names), NOT 10%.

WHAT IS AND IS NOT EVIDENCE OF A HALT
-------------------------------------
`gap_before` — a missing RTH minute in the tape — is genuine evidence: if the
security did not print for a minute during regular hours, something stopped it.
A large single-minute EXCURSION is only evidence when measured against the
correct tiered band. Both are retained here; only the band changed.

Reference price is approximated as the trailing 5x 1-minute bar mean of closes.
True LULD reference is a 5-minute average of prints. This is adequate for
detection and imprecise at exact band edges — stated, not papered over.

This module lives under `backtester/` deliberately: `backtester/` imports neither
`execution/` nor `sovereign/`, and that leaf position is preserved. `execution/`
imports THIS, never the reverse.
"""
from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd

# ── Band constants ────────────────────────────────────────────────────────────
TIER1_BAND = 0.05      # S&P500 / Russell 1000 / select ETPs, >= $3.00
TIER2_BAND = 0.10      # all other NMS securities, >= $3.00
MID_BAND = 0.20        # $0.75 <= price < $3.00, any tier
LOW_ABS = 0.15         # price < $0.75: lesser of 75% or $0.15 (absolute)
LOW_PCT = 0.75         # price < $0.75: the 75% leg

PRICE_TIER_HIGH = 3.00
PRICE_TIER_LOW = 0.75

# Doubling windows (inclusive start, exclusive end)
OPEN_DOUBLE = (dtime(9, 30), dtime(9, 45))
CLOSE_DOUBLE = (dtime(15, 35), dtime(16, 0))

REFERENCE_LOOKBACK = 5   # 1-minute bars averaged for the reference price
GAP_MINUTES = 1.5        # tape gap beyond this many minutes => halt evidence


def _in_window(t: dtime, window: tuple[dtime, dtime]) -> bool:
    """True if t falls in [start, end). Boundary: 09:45:00 is NOT doubled."""
    start, end = window
    return start <= t < end


def is_doubled(t: dtime) -> bool:
    """True during the opening (09:30-09:45) or closing (15:35-16:00) period."""
    return _in_window(t, OPEN_DOUBLE) or _in_window(t, CLOSE_DOUBLE)


def luld_band(price: float, t: dtime, tier: int = 2) -> float:
    """Fractional LULD band for `price` at time `t`.

    Returns e.g. 0.20 for a +/-20% band. Doubled in the opening/closing periods.

    tier=2 is the correct default: every parabolic gapper in this repo's universe
    is a Tier 2 NMS security, not an S&P 500 / Russell 1000 name.
    """
    if price >= PRICE_TIER_HIGH:
        band = TIER1_BAND if tier == 1 else TIER2_BAND
    elif price >= PRICE_TIER_LOW:
        band = MID_BAND
    else:
        # "lesser of 75% or $0.15" — as a fraction of the reference price
        band = min(LOW_PCT, LOW_ABS / max(price, 1e-6))

    if is_doubled(t):
        band *= 2.0
    return band


def reference_price(closes: np.ndarray, i: int,
                    lookback: int = REFERENCE_LOOKBACK) -> float:
    """Approximate the LULD reference price at bar i.

    True reference is a 5-minute average of eligible prints; this uses the mean
    of the trailing `lookback` 1-minute closes (inclusive of bar i-1). At i=0 the
    bar's own close is used, since no history exists.
    """
    if i <= 0:
        return float(closes[0])
    lo = max(0, i - lookback)
    window = closes[lo:i]
    if window.size == 0:
        return float(closes[i])
    return float(np.mean(window))


def _bar_times(bars) -> list[dtime]:
    """Extract per-bar ET wall-clock times from a bars frame with a 'time' column
    formatted '%H:%M' (the convention `realistic_fills` already uses)."""
    ts = pd.to_datetime(bars["time"], format="%H:%M")
    return [t.time() for t in ts]


def halt_flags(bars, tier: int = 2) -> np.ndarray:
    """Per-bar boolean: True if this bar is a halt-resume.

    Two independent pieces of evidence, OR'd:
      1. `gap_before` — a missing RTH minute immediately before this bar.
      2. An open-vs-reference or close-vs-open excursion beyond the TIERED band
         for this bar's price and time of day.

    `bars` is a frame with columns time/open/high/low/close/volume, matching
    `realistic_fills`.
    """
    t = pd.to_datetime(bars["time"], format="%H:%M")
    gap_before = t.diff().dt.total_seconds().div(60).fillna(1).to_numpy() > GAP_MINUTES

    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    times = _bar_times(bars)

    n = len(o)
    excursion = np.zeros(n, dtype=bool)
    for i in range(n):
        ref = reference_price(c, i)
        band = luld_band(ref, times[i], tier=tier)
        if ref <= 0:
            continue
        jump = abs(o[i] / ref - 1) > band
        intrabar = abs(c[i] / o[i] - 1) > band if o[i] else False
        excursion[i] = bool(jump or intrabar)

    return gap_before | excursion


def bands_for(bars, tier: int = 2) -> np.ndarray:
    """Per-bar band actually applied — recorded in the fill log for auditability."""
    c = bars["close"].to_numpy(dtype=float)
    times = _bar_times(bars)
    return np.array([luld_band(reference_price(c, i), times[i], tier=tier)
                     for i in range(len(c))], dtype=float)
