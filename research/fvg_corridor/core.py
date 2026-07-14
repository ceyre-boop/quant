"""FVG × Fractal-Corridor core — fenced loader, causal features (TICK-035/HYP-098).

Causality contract (charter): every feature at bar t uses bars <= t only.
- ATR: trailing mean of true range, computed on PRIOR bars (shifted).
- FVG: 3-bar gap known at the close of its third bar.
- Fractal pivots: Pine `pvts()` semantics — a depth-d pivot at index i is
  CONFIRMED only at index i + d (strict on the older side, non-strict on the
  newer side, matching the Pine loops); corridor lines join the two most recent
  confirmed pivots per side and extrapolate to the current bar.
"""
from __future__ import annotations

from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MINING_END = pd.Timestamp("2024-06-30 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2024-07-01", tz="UTC")

DEPTH_BASE, FACTOR = 10, 5          # Pine defaults: depths 10 / 50 / 250
DEPTHS = (DEPTH_BASE, DEPTH_BASE * FACTOR, DEPTH_BASE * FACTOR * FACTOR)

# ict/session_classifier.py windows, verbatim (UTC)
KILLZONES = {"LONDON": (dtime(2, 0), dtime(5, 0)),
             "NY_OPEN": (dtime(7, 0), dtime(10, 0)),
             "NY_PM": (dtime(13, 30), dtime(16, 0))}


def load_nq_5min(segment: str = "mining") -> pd.DataFrame:
    """5-min Globex bars from the 1-min parquet, fenced by segment."""
    df = pd.read_parquet(REPO / "data/es_nq/nq_globex_1min.parquet")
    if segment == "mining":
        df = df[df.index <= MINING_END]
    elif segment == "holdout":
        df = df[df.index >= HOLDOUT_START]
    else:
        raise ValueError("segment must be 'mining' or 'holdout'")
    o = df["Open"].resample("5min").first()
    h = df["High"].resample("5min").max()
    l = df["Low"].resample("5min").min()
    c = df["Close"].resample("5min").last()
    v = df["Volume"].resample("5min").sum()
    out = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}).dropna()
    return out


def causal_atr(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    """Trailing ATR known at bar t (uses bars < t only, via shift)."""
    pc = df["c"].shift(1)
    tr = np.maximum(df["h"] - df["l"],
                    np.maximum((df["h"] - pc).abs(), (df["l"] - pc).abs()))
    return tr.rolling(n).mean().shift(1).to_numpy()


def detect_fvgs(df: pd.DataFrame, atr: np.ndarray, min_size_atr: float):
    """3-bar FVGs. Bullish at i: l[i] > h[i-2]; bearish: h[i] < l[i-2].
    Known at close of bar i. Returns arrays (idx, side, top, bottom)."""
    h, l = df["h"].to_numpy(), df["l"].to_numpy()
    i = np.arange(2, len(df))
    bull = l[i] > h[i - 2]
    bear = h[i] < l[i - 2]
    size = np.where(bull, l[i] - h[i - 2], np.where(bear, l[i - 2] - h[i], 0.0))
    ok = (bull | bear) & np.isfinite(atr[i]) & (size >= min_size_atr * atr[i])
    idx = i[ok]
    side = np.where(bull[ok], 1, -1)
    top = np.where(side == 1, l[idx], l[idx - 2])
    bot = np.where(side == 1, h[idx - 2], h[idx])
    return idx, side, top, bot


def confirmed_pivots(x: np.ndarray, depth: int, is_high: bool):
    """Pine pvts() port, vectorized. Candidate at i is a pivot if the depth bars
    AFTER it are <= (high) / >= (low) [non-strict] and the depth bars BEFORE it
    are strictly < / >. Returns (pivot_index, confirm_index=i+depth, value)."""
    s = pd.Series(x)
    y = x if is_high else -x            # unify: pivot = local max of y
    ys = pd.Series(y)
    before_max = ys.rolling(depth).max().shift(1).to_numpy()      # bars i-depth..i-1
    after_max = ys[::-1].rolling(depth).max().shift(1).to_numpy()[::-1]  # i+1..i+depth
    ok = (before_max < y) & (after_max <= y)
    ok &= np.isfinite(before_max) & np.isfinite(after_max)
    piv_i = np.where(ok)[0]
    piv_i = piv_i[(piv_i >= depth) & (piv_i < len(x) - depth)]
    return piv_i, piv_i + depth, x[piv_i]


def corridor_lines(df: pd.DataFrame, depth: int):
    """Per-bar upper/lower corridor values from the two most recent CONFIRMED
    pivots per side, linearly extrapolated to the current bar. NaN until two
    pivots are confirmed. O(n) sweep."""
    n = len(df)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for is_high, out in ((True, upper), (False, lower)):
        x = (df["h"] if is_high else df["l"]).to_numpy()
        piv_i, conf_i, val = confirmed_pivots(x, depth, is_high)
        if len(piv_i) < 2:
            continue
        t = np.arange(n)
        k = np.searchsorted(conf_i, t, side="right") - 1   # last confirmed pivot at t
        m = k >= 1
        i0, i1 = piv_i[k[m] - 1], piv_i[k[m]]
        v0, v1 = val[k[m] - 1], val[k[m]]
        slope = (v1 - v0) / (i1 - i0)
        out[m] = v1 + slope * (t[m] - i1)
    return upper, lower


def corridor_features(df: pd.DataFrame):
    """Per-bar features per depth: position in corridor (0..1, outside <0/>1),
    mid-line slope sign. dict keyed by depth."""
    feats = {}
    c = df["c"].to_numpy()
    for d in DEPTHS:
        up, lo = corridor_lines(df, d)
        width = up - lo
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = np.where(np.isfinite(width) & (width > 0), (c - lo) / width, np.nan)
        mid = (up + lo) / 2
        slope = np.full(len(df), np.nan)
        slope[1:] = mid[1:] - mid[:-1]
        feats[d] = {"pos": pos, "slope_sign": np.sign(slope)}
    return feats


def killzone_mask(df: pd.DataFrame, zone: str) -> np.ndarray:
    if zone == "ALL":
        return np.ones(len(df), bool)
    lo, hi = KILLZONES[zone]
    t = df.index.time
    return np.array([(lo <= x < hi) for x in t])
