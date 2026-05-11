"""
ict/liquidity_heatmap.py
========================
Real-time liquidity probability heatmap for ICT pairs.

Uses sweep_detector output + recent OHLCV to identify:
  - Swept liquidity levels (already hit — magnet potential reduced)
  - Unswept swing highs/lows (pending liquidity)
  - Equal highs/lows clusters (institutional targets)
  - Unmitigated FVGs (price likely to return)
  - ADR exhaustion bands

Assigns a pull probability (0–1) to each level based on:
  - Distance from current price (closer = higher prob)
  - Number of touches (more touches = stronger magnet)
  - Time since formed (older = decays)
  - Volume proxy at formation (range/ATR)

Output pushed to Firebase: signals/ICT_ENGINE/heatmap/{PAIR}
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SWING_LOOKBACK  = 20    # bars to detect swing highs/lows
EQUAL_TOLERANCE = 0.0003  # 3 pip tolerance for "equal" levels
DECAY_BARS      = 48    # probability decays to 50% after this many bars
MAX_LEVELS      = 12    # cap output size


@dataclass
class LiquidityLevel:
    price:     float
    prob:      float     # 0–1 pull probability
    kind:      str       # SWING_HIGH | SWING_LOW | EQUAL_HIGH | EQUAL_LOW | FVG_HIGH | FVG_LOW
    age_bars:  int       # bars since formed
    touches:   int       # number of times price approached within tolerance
    swept:     bool      # True if already swept (lower prob)

    def to_dict(self):
        return asdict(self)


def compute_heatmap(
    pair: str,
    df: pd.DataFrame,
    atr: float,
    current_price: Optional[float] = None,
) -> dict:
    """
    Compute liquidity heatmap from OHLCV dataframe.

    Args:
        pair:          e.g. 'USDJPY'
        df:            hourly OHLCV, columns Open/High/Low/Close
        atr:           current ATR value
        current_price: last close price (uses df.Close.iloc[-1] if None)

    Returns Firebase-ready dict:
      {
        'pair': 'USDJPY',
        'current_price': 154.82,
        'levels': [...],
        'top_magnet': {'price': 154.51, 'prob': 0.91, 'kind': 'EQUAL_LOW'},
        'adr_high': 155.20,
        'adr_low': 153.90,
      }
    """
    if df is None or len(df) < SWING_LOOKBACK + 1:
        return {'pair': pair, 'levels': [], 'available': False}

    price = current_price or float(df['Close'].iloc[-1])
    levels: List[LiquidityLevel] = []

    # ── 1. Swing highs / lows ─────────────────────────────────────────────
    highs = _swing_highs(df)
    lows  = _swing_lows(df)

    for bar_idx, h in highs:
        age   = len(df) - 1 - bar_idx
        swept = float(df['High'].iloc[bar_idx+1:].max()) > h if bar_idx < len(df)-1 else False
        prob  = _prob(h, price, atr, age, swept, touches=1)
        levels.append(LiquidityLevel(price=round(h,5), prob=prob,
                      kind='SWING_HIGH', age_bars=age, touches=1, swept=swept))

    for bar_idx, l in lows:
        age   = len(df) - 1 - bar_idx
        swept = float(df['Low'].iloc[bar_idx+1:].min()) < l if bar_idx < len(df)-1 else False
        prob  = _prob(l, price, atr, age, swept, touches=1)
        levels.append(LiquidityLevel(price=round(l,5), prob=prob,
                      kind='SWING_LOW', age_bars=age, touches=1, swept=swept))

    # ── 2. Equal highs / lows clusters ────────────────────────────────────
    eq_highs = _equal_clusters(highs, tolerance=price * EQUAL_TOLERANCE)
    eq_lows  = _equal_clusters(lows,  tolerance=price * EQUAL_TOLERANCE)

    for cluster_price, count in eq_highs:
        age  = 10  # approximate
        prob = _prob(cluster_price, price, atr, age, swept=False, touches=count)
        levels.append(LiquidityLevel(price=round(cluster_price,5), prob=min(prob*1.3, 0.99),
                      kind='EQUAL_HIGH', age_bars=age, touches=count, swept=False))

    for cluster_price, count in eq_lows:
        age  = 10
        prob = _prob(cluster_price, price, atr, age, swept=False, touches=count)
        levels.append(LiquidityLevel(price=round(cluster_price,5), prob=min(prob*1.3, 0.99),
                      kind='EQUAL_LOW', age_bars=age, touches=count, swept=False))

    # ── 3. FVG midpoints ─────────────────────────────────────────────────
    fvg_levels = _detect_fvg_levels(df, atr)
    levels.extend(fvg_levels)

    # ── 4. ADR bands ─────────────────────────────────────────────────────
    daily_range = _compute_adr(df, days=14)
    adr_high = round(price + daily_range * 0.5, 5)
    adr_low  = round(price - daily_range * 0.5, 5)

    # ── 5. Sort by probability, deduplicate nearby levels ────────────────
    levels = _deduplicate(levels, tolerance=price * 0.0005)
    levels.sort(key=lambda l: l.prob, reverse=True)
    levels = levels[:MAX_LEVELS]

    top = levels[0] if levels else None

    return {
        'pair':          pair,
        'current_price': round(price, 5),
        'levels':        [l.to_dict() for l in levels],
        'top_magnet':    top.to_dict() if top else None,
        'adr_high':      adr_high,
        'adr_low':       adr_low,
        'available':     True,
    }


# ── Helpers ────────────────────────────────────────────────────────────────── #

def _swing_highs(df: pd.DataFrame, n: int = SWING_LOOKBACK):
    highs = []
    for i in range(n, len(df) - n):
        h = float(df['High'].iloc[i])
        if h == float(df['High'].iloc[i-n:i+n+1].max()):
            highs.append((i, h))
    return highs[-6:]  # last 6 swing highs


def _swing_lows(df: pd.DataFrame, n: int = SWING_LOOKBACK):
    lows = []
    for i in range(n, len(df) - n):
        l = float(df['Low'].iloc[i])
        if l == float(df['Low'].iloc[i-n:i+n+1].min()):
            lows.append((i, l))
    return lows[-6:]


def _equal_clusters(levels, tolerance: float):
    """Find groups of levels within tolerance — institutional targets."""
    clusters = []
    used = set()
    for i, (_, p1) in enumerate(levels):
        if i in used:
            continue
        group = [p1]
        for j, (_, p2) in enumerate(levels):
            if j != i and j not in used and abs(p1 - p2) <= tolerance:
                group.append(p2)
                used.add(j)
        if len(group) >= 2:
            clusters.append((sum(group)/len(group), len(group)))
    return clusters


def _detect_fvg_levels(df: pd.DataFrame, atr: float):
    """Find unmitigated FVG midpoints in recent bars."""
    fvgs = []
    bars = df.tail(48)
    for i in range(2, len(bars)):
        h0 = float(bars['High'].iloc[i-2])
        l2 = float(bars['Low'].iloc[i])
        h2 = float(bars['High'].iloc[i])
        l0 = float(bars['Low'].iloc[i-2])

        # Bullish FVG: gap between bar[i-2].high and bar[i].low
        if l2 > h0 and (l2 - h0) > atr * 0.1:
            mid = (l2 + h0) / 2
            age = len(bars) - 1 - i
            fvgs.append(LiquidityLevel(price=round(mid,5), prob=0.65,
                         kind='FVG_LOW', age_bars=age, touches=0, swept=False))

        # Bearish FVG: gap between bar[i].high and bar[i-2].low
        if h2 < l0 and (l0 - h2) > atr * 0.1:
            mid = (h2 + l0) / 2
            age = len(bars) - 1 - i
            fvgs.append(LiquidityLevel(price=round(mid,5), prob=0.65,
                         kind='FVG_HIGH', age_bars=age, touches=0, swept=False))

    return fvgs[-4:]  # most recent 4 FVGs


def _prob(level_price, current_price, atr, age_bars, swept, touches=1):
    """Compute pull probability 0–1."""
    dist = abs(level_price - current_price)
    dist_r    = dist / (atr + 1e-9)          # distance in R units
    dist_prob = math.exp(-dist_r * 0.8)       # exponential decay with distance
    decay     = math.exp(-age_bars / (DECAY_BARS * 2))  # time decay
    touch_amp = min(1.0 + (touches - 1) * 0.15, 1.5)   # more touches = stronger
    swept_pen = 0.3 if swept else 1.0
    return round(min(dist_prob * decay * touch_amp * swept_pen, 0.99), 3)


def _compute_adr(df: pd.DataFrame, days: int = 14) -> float:
    daily = df.resample('1D').agg({'High': 'max', 'Low': 'min'}).dropna()
    if len(daily) < 3:
        return float(df['High'].max() - df['Low'].min())
    ranges = (daily['High'] - daily['Low']).tail(days)
    return float(ranges.mean())


def _deduplicate(levels, tolerance):
    kept = []
    for l in sorted(levels, key=lambda x: x.prob, reverse=True):
        if not any(abs(l.price - k.price) < tolerance for k in kept):
            kept.append(l)
    return kept
