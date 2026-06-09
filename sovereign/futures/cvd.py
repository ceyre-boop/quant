"""Cumulative Volume Delta proxy for the futures sandbox (Increment 4).

Sandbox-local: no forex/ICT/intelligence imports. We have no tick data, so per-bar delta is a
PROXY: (close-open)/(high-low+eps) * volume — the fraction of the bar's range closed in,
scaled by volume. Real order-flow delta needs a CME feed (Databento/CQG). On yfinance the
volume itself is unreliable, so this is telemetry there, not truth — fail loud on thin volume.

The 1% confirmation: don't just enter at a level — enter when delta shows someone defending it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sovereign.futures.config import futures_params


def cvd_state(df) -> Optional[dict]:
    """Running-CVD snapshot: {cvd, slope, slope_prev, avg_abs_slope, n}. None if too few
    bars have volume (so yfinance noise can't masquerade as an order-flow signal)."""
    if df is None or len(df) == 0:
        return None
    c = futures_params()["cvd"]
    lb = c["slope_lookback"]
    o = df["Open"].to_numpy(float); h = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float); cl = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    if int(np.sum(np.isfinite(v) & (v > 0))) < c["min_vol_bars"]:
        return None
    rng = (h - lo) + 1e-9
    delta = np.nan_to_num((cl - o) / rng) * np.nan_to_num(v)
    cvd = np.cumsum(delta)
    if len(cvd) < lb + 2:
        return None
    slope = float(cvd[-1] - cvd[-1 - lb])
    slope_prev = float(cvd[-2] - cvd[-2 - lb])
    steps = np.abs(np.diff(cvd))
    avg_abs_slope = float(np.mean(steps[-(lb * 4):])) * lb if len(steps) else 0.0
    return {"cvd": float(cvd[-1]), "slope": slope, "slope_prev": slope_prev,
            "avg_abs_slope": avg_abs_slope, "n": int(len(cvd))}


def cvd_confirms(setup: str, direction: str, state: Optional[dict]) -> Optional[bool]:
    """Does delta confirm this entry? None if state is unavailable (thin volume = unknown).

    ORB/MICRO: slope must agree with direction (buyers stepping in for longs).
    VWAP_MR: fade needs absorption — for a LONG at the lower band, selling pressure must be
    decelerating (slope improving vs the prior slope); mirror for SHORT at the upper band."""
    if state is None:
        return None
    s = setup.upper()
    slope, slope_prev = state["slope"], state["slope_prev"]
    if s in ("ORB", "MICRO"):
        return slope > 0 if direction == "LONG" else slope < 0
    if s == "VWAP_MR":
        return slope > slope_prev if direction == "LONG" else slope < slope_prev
    return None


def is_strong(state: Optional[dict]) -> bool:
    """True if the CVD slope is strong vs recent activity (tier-3 scale-in gate)."""
    if state is None or state["avg_abs_slope"] <= 0:
        return False
    return abs(state["slope"]) >= futures_params()["cvd"]["strong_slope_mult"] * state["avg_abs_slope"]
