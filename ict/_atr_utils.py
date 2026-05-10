"""
ict/_atr_utils.py
=================
Shared ATR computation helper for the ICT subsystem.

All ICT detectors must use this helper to ensure consistent ATR calculation
and a price-relative fallback (avoids hardcoded 1.0 that doesn't scale across
instruments at different price levels, e.g. USDJPY ≈150 vs AUDUSD ≈0.65).

ISOLATION: No imports from sovereign/, layer1/, layer2/, layer3/.
"""
from __future__ import annotations

import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute the 14-period Average True Range on *df*.

    Falls back to ``0.0001 × last_close`` when the rolling ATR is zero or NaN
    (e.g. completely flat data, or fewer bars than the period).  This is
    instrument-relative and avoids the misleading hardcoded constant of 1.0.

    Args:
        df: DataFrame with uppercase High, Low, Close columns (already normalised).
        period: ATR rolling window.

    Returns:
        ATR as a float > 0.
    """
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    val = float(tr.rolling(period).mean().iloc[-1])
    if val > 0 and not (val != val):  # guard NaN and zero
        return val
    # Price-relative fallback: 1 bp of the last close price
    last_close = float(df["Close"].iloc[-1])
    return max(0.0001 * last_close, 1e-9)
