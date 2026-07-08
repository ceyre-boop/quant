"""
sovereign/research/ohlc_quartile.py
===================================
OHLC-quartile feature encoding for the candle-structure prediction hypothesis
(queued to the autonomous research factory 2026-06-15).

THE IDEA (user's): break each candle into clean numbers — the 25/50/75/100% price
levels of its constituent sub-bars — on dual timeframes (4H + 5min), and let a
SIMPLE model see if that structure predicts the next 4H bar's direction.

DISCIPLINE (read before trusting anything here)
-----------------------------------------------
Mechanism: NONE STATED — exploratory. There is no identified reason these levels
carry information the market hasn't already priced. This is the same CLASS of
hypothesis that has been killed repeatedly on this data (ES/NQ bias p=0.567, ICT
p=0.52, momentum p=0.85). It is built ONLY to be tested cheaply and honestly through
the factory in shadow mode. The expected base-case verdict is NOT_SIGNIFICANT.

This module is pure feature math — no I/O, no model, no live wiring. The validator
(scripts/validate_ohlc_quartile.py) consumes it; nothing in the live path imports it.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# Quartile cut points: 25 / 50 / 75 / 100 percentile price levels of the sub-bars.
_QUANTILES = (0.25, 0.50, 0.75, 1.00)
_N_TRAIL_5M = 6      # trailing 5min bars summarized into each 4H feature row
_REF_LOOKBACK = 20   # bars used for the normalizing reference range


def encode_bar_quartiles(prices: np.ndarray, ref_low: float, ref_range: float) -> List[float]:
    """
    Encode one bar as 4 numbers: its 25/50/75/100% sub-bar price levels, normalized
    to a reference (ref_low, ref_range) so the features are scale-free and comparable
    across regimes.

    Args:
        prices:    1-D array of the bar's constituent sub-bar prices (e.g. the closes
                   of the 5min bars inside a 4H bar, or the 1min closes inside a 5min).
        ref_low:   reference low for normalization (e.g. trailing window min).
        ref_range: reference range for normalization (e.g. trailing window max-min).

    Returns:
        [q25, q50, q75, q100] normalized levels. Deterministic. Empty/zero-range
        inputs return [0,0,0,0] rather than raising.
    """
    p = np.asarray(prices, dtype=float)
    p = p[~np.isnan(p)]
    if p.size == 0 or ref_range <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    levels = np.quantile(p, _QUANTILES)
    return [float((lv - ref_low) / ref_range) for lv in levels]


def _reference(window_prices: np.ndarray) -> tuple[float, float]:
    """Normalizing (low, range) from a trailing price window."""
    w = np.asarray(window_prices, dtype=float)
    w = w[~np.isnan(w)]
    if w.size == 0:
        return 0.0, 0.0
    lo, hi = float(w.min()), float(w.max())
    return lo, (hi - lo)


def resample_4h(df5m: pd.DataFrame) -> pd.DataFrame:
    """
    5min → 4H bars (label='left', closed='left' — same convention as the ES/NQ 5min
    resampler). Keeps a list of constituent 5min closes per 4H bar for quartile encoding.
    """
    if df5m is None or len(df5m) == 0:
        raise ValueError("resample_4h: empty input")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    out = df5m.resample("240min", label="left", closed="left").agg(agg).dropna(subset=["Open"])
    # constituent 5min closes per 4H bin (for sub-bar quartiles)
    closes = df5m["Close"].resample("240min", label="left", closed="left").apply(
        lambda s: list(s.dropna().values))
    out = out.join(closes.rename("sub_closes"))
    return out


def build_feature_matrix(df5m: pd.DataFrame,
                         n_trail_5m: int = _N_TRAIL_5M,
                         ref_lookback: int = _REF_LOOKBACK) -> Optional[pd.DataFrame]:
    """
    Assemble the dual-timeframe quartile feature matrix, predicting the NEXT 4H bar.

    For each completed 4H bar t (using only bars ≤ t — NO lookahead):
      - 4H quartiles of bar t (sub-bars = the 5min closes inside it)
      - 4H quartiles of bar t-1
      - aggregated quartiles of the last `n_trail_5m` 5min bars
    Label y = sign(next 4H bar close - open)  ∈ {0,1} (1 = up).

    Returns a DataFrame of features + 'y' + 'ret' (next-bar open→close return),
    indexed by the decision-time 4H bar timestamp. None if insufficient data.
    """
    if df5m is None or len(df5m) < ref_lookback + 10:
        return None
    df4h = resample_4h(df5m)
    if len(df4h) < ref_lookback + 3:
        return None

    five_close = df5m["Close"].to_numpy()
    five_index = df5m.index
    four_close = df4h["Close"].to_numpy()

    rows = []
    idx = []
    for t in range(ref_lookback, len(df4h) - 1):
        # Normalizing reference: trailing 4H closes (completed bars only).
        ref_lo, ref_rng = _reference(four_close[t - ref_lookback:t + 1])
        if ref_rng <= 0:
            continue

        q_t = encode_bar_quartiles(df4h["sub_closes"].iloc[t] or [four_close[t]], ref_lo, ref_rng)
        q_tm1 = encode_bar_quartiles(df4h["sub_closes"].iloc[t - 1] or [four_close[t - 1]], ref_lo, ref_rng)

        # Trailing 5min: the last n_trail_5m 5min closes completed by 4H bar t's close.
        t_end = df4h.index[t] + pd.Timedelta(minutes=240)
        mask = five_index < t_end
        trail = five_close[mask][-n_trail_5m:]
        q_5m = encode_bar_quartiles(trail if trail.size else [four_close[t]], ref_lo, ref_rng)

        nxt_open = float(df4h["Open"].iloc[t + 1])
        nxt_close = float(df4h["Close"].iloc[t + 1])
        if nxt_open <= 0:
            continue
        ret = (nxt_close - nxt_open) / nxt_open
        y = 1 if nxt_close > nxt_open else 0

        rows.append(q_t + q_tm1 + q_5m + [y, ret])
        idx.append(df4h.index[t])

    if not rows:
        return None
    cols = ([f"q4h_t_{q}" for q in _QUANTILES] +
            [f"q4h_tm1_{q}" for q in _QUANTILES] +
            [f"q5m_{q}" for q in _QUANTILES] + ["y", "ret"])
    return pd.DataFrame(rows, columns=cols, index=pd.DatetimeIndex(idx))


FEATURE_COLS = ([f"q4h_t_{q}" for q in _QUANTILES] +
                [f"q4h_tm1_{q}" for q in _QUANTILES] +
                [f"q5m_{q}" for q in _QUANTILES])
