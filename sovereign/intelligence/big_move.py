"""
sovereign/intelligence/big_move.py
==================================
Big-Move-of-the-Day classifier — a deterministic daily estimate of whether
*today* is an institutional displacement day, which direction, and how big.

WHY THIS EXISTS
---------------
The system models per-bar setups (sweep -> displacement -> FVG -> entry) and uses
ADR only as a *veto*. It never models the thing that is actually more predictable
than candle noise: the daily institutional displacement. This module fuses signals
the system already computes into one estimate per pair:

    P(big move) x direction x expected magnitude x session

ISOLATION
---------
This lives in sovereign/ (full access) and computes its own daily displacement
features from daily OHLCV. It does NOT import ict/ internals, so the
`ict/ -/-> sovereign/` boundary is untouched. It is invoked from the zero-cost
pulse (every 2h), never from ict/pipeline.py.

DISCIPLINE (read before trusting the numbers)
---------------------------------------------
The weights below are hand-set PRIORS, not fitted coefficients. This classifier
has NOT earned live influence. `scripts/validate_big_move.py` is the gate: until it
clears OOS (Sharpe + permutation + Benjamini-Hochberg), the estimate is
display-only. This mirrors the ES/NQ bias-engine lesson (killed 2026-06-10 at
p=0.57 when its inputs proved sub-base-rate) — an estimate is a hypothesis until the
harness says otherwise.

Pure function, no I/O, no LLM. Robust to missing context (degrades to lower
confidence rather than failing).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Tunable PRIORS (hand-set; calibrated/justified only by the validation gate) ─ #

# Minimum daily bars required for a confident estimate.
_MIN_BARS = 60

# Trailing windows (daily bars).
_ATR_SHORT = 5     # recent volatility
_ATR_LONG = 20     # baseline volatility
_ADR_LOOKBACK = 20  # average daily range
_RANGE_PCTILE_LOOKBACK = 60  # distribution window for the "big-move day" label

# "Big-move day" label threshold: daily true range in the top tercile of the
# trailing distribution. This is the target the validation harness scores against.
_BIG_MOVE_PERCENTILE = 0.667

# Magnitude-feature weights -> logit for P(big move). Compression (coiled-spring /
# volatility-clustering) and event risk are the load-bearing priors; macro
# |z-score| and |momentum| contribute less; VIX is a mild amplifier.
_W_BIAS = -0.40           # base rate skews toward "not a big day"
_W_COMPRESSION = 1.30     # low recent/baseline ATR -> expansion more likely
_W_EVENT = 0.90           # high-impact event today -> bigger expected range
_W_RATE_DIFF_MAG = 0.55   # |rate-differential z| -> macro pressure
_W_MOMENTUM_MAG = 0.45    # |5d momentum| -> trend energy
_W_VIX = 0.35             # elevated VIX -> larger ranges

# Directional-driver weights -> signed score for LONG/SHORT.
_W_DIR_MOMENTUM = 1.0
_W_DIR_RATE_DIFF = 1.0
_W_DIR_DISPLACEMENT = 0.6
_DIRECTION_DEADBAND = 0.15  # |signed score| below this -> NEUTRAL

# Expected magnitude: predicted daily range as a multiple of ADR, scaled by p_big.
_MAG_BASE = 0.80
_MAG_SPAN = 0.90  # expected_vs_adr in [_MAG_BASE, _MAG_BASE + _MAG_SPAN]


@dataclass
class BigMoveEstimate:
    """One pair's big-move estimate for the current session."""
    pair: str
    p_big: float                      # 0–1 probability today is a top-tercile range day
    direction: str                    # LONG | SHORT | NEUTRAL
    expected_magnitude_pct: float     # predicted daily range as % of price
    expected_vs_adr: float            # predicted range / trailing ADR (multiple)
    session: str                      # LONDON | NY_AM | NY_PM | OFF_HOURS
    confidence: float                 # 0–1, scales with data + driver agreement
    drivers: List[Tuple[str, float]] = field(default_factory=list)  # (name, signed contribution)
    timestamp: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "p_big": round(self.p_big, 4),
            "direction": self.direction,
            "expected_magnitude_pct": round(self.expected_magnitude_pct, 4),
            "expected_vs_adr": round(self.expected_vs_adr, 3),
            "session": self.session,
            "confidence": round(self.confidence, 3),
            "drivers": [[name, round(val, 4)] for name, val in self.drivers],
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


# ── Math helpers (daily granularity, self-contained) ──────────────────────────── #

def _true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder true range on daily bars: max(H-L, |H-prevC|, |L-prevC|)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(df: pd.DataFrame, n: int) -> float:
    tr = _true_range(df).dropna()
    if len(tr) < n:
        return float(tr.mean()) if len(tr) else 0.0
    return float(tr.iloc[-n:].mean())


def _adr(df: pd.DataFrame, n: int = _ADR_LOOKBACK) -> float:
    """Average daily range over the last n COMPLETED bars (excludes the latest)."""
    rng = (df["High"] - df["Low"]).dropna()
    if len(rng) < 3:
        return float(rng.mean()) if len(rng) else 0.0
    window = rng.iloc[-(n + 1):-1]
    return float(window.mean()) if len(window) else float(rng.mean())


def _compression(df: pd.DataFrame) -> float:
    """
    Coiled-spring feature in [0, 1]: 1.0 = maximally compressed (recent ATR far
    below baseline -> expansion likely), 0.0 = recent ATR >= baseline. Volatility
    clustering means low-range days cluster, and the breakout out of compression
    tends to be the big move.
    """
    short_atr = _atr(df, _ATR_SHORT)
    long_atr = _atr(df, _ATR_LONG)
    if long_atr <= 0:
        return 0.0
    ratio = short_atr / long_atr
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def _momentum_z(df: pd.DataFrame) -> float:
    """5-day close-to-close momentum, normalized by long ATR. Signed."""
    close = df["Close"].dropna()
    long_atr = _atr(df, _ATR_LONG)
    if len(close) < 6 or long_atr <= 0:
        return 0.0
    move = float(close.iloc[-1] - close.iloc[-6])
    return float(np.clip(move / (long_atr * math.sqrt(5)), -3.0, 3.0))


def _displacement_dir(df: pd.DataFrame) -> float:
    """
    Signed continuous displacement strength of the most recent completed bar:
    body / ATR, signed by candle direction, clipped. Daily analogue of the ICT
    per-bar displacement check — committed directional close.
    """
    long_atr = _atr(df, _ATR_LONG)
    if len(df) < 2 or long_atr <= 0:
        return 0.0
    bar = df.iloc[-1]
    body = float(bar["Close"]) - float(bar["Open"])
    return float(np.clip(body / long_atr, -2.0, 2.0))


def _percentile_of_last_range(df: pd.DataFrame) -> float:
    """Where the latest daily true range sits in the trailing distribution (0–1)."""
    tr = _true_range(df).dropna()
    if len(tr) < 10:
        return 0.5
    window = tr.iloc[-_RANGE_PCTILE_LOOKBACK:]
    last = float(tr.iloc[-1])
    return float((window < last).mean())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _norm_context(context: Optional[dict]) -> Tuple[Dict[str, Optional[float]], int]:
    """
    Pull optional macro context, returning normalized values and a count of how
    many were actually supplied (drives confidence). Missing -> None (neutral).
    Expected keys: rate_diff_z, vix, cot_percentile, high_impact_event_today.
    """
    ctx = context or {}
    present = 0
    out: Dict[str, Optional[float]] = {}

    rd = ctx.get("rate_diff_z")
    out["rate_diff_z"] = float(np.clip(rd, -3.0, 3.0)) if _is_num(rd) else None
    present += out["rate_diff_z"] is not None

    vix = ctx.get("vix")
    # Map VIX ~[10, 40] -> [0, 1] amplifier; <12 calm, >28 stressed.
    out["vix_amp"] = float(np.clip((vix - 12.0) / 18.0, 0.0, 1.0)) if _is_num(vix) else None
    present += out["vix_amp"] is not None

    cot = ctx.get("cot_percentile")
    out["cot_extreme"] = float(abs(cot - 0.5) * 2.0) if _is_num(cot) else None  # 0=neutral,1=extreme
    present += out["cot_extreme"] is not None

    ev = ctx.get("high_impact_event_today")
    out["event"] = 1.0 if ev else (0.0 if ev is not None else None)
    present += out["event"] is not None

    return out, present


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


# ── Public API ────────────────────────────────────────────────────────────────── #

def estimate_big_move(
    pair: str,
    df_daily: pd.DataFrame,
    context: Optional[dict] = None,
    session: str = "OFF_HOURS",
    now: Optional[datetime] = None,
) -> BigMoveEstimate:
    """
    Produce a deterministic big-move estimate for `pair` from daily OHLCV.

    Args:
        pair:     symbol, e.g. "GBPUSD".
        df_daily: daily OHLCV with columns Open/High/Low/Close, chronological.
        context:  optional macro dict — keys: rate_diff_z, vix, cot_percentile,
                  high_impact_event_today. Missing keys degrade confidence, not crash.
        session:  current killzone label for display.
        now:      timestamp override (defaults to UTC now).

    Returns:
        BigMoveEstimate. On insufficient data, returns a low-confidence NEUTRAL
        estimate with a note rather than raising.
    """
    ts = (now or datetime.now(tz=timezone.utc)).isoformat()

    if df_daily is None or len(df_daily) < 10 or not _has_ohlc(df_daily):
        return BigMoveEstimate(
            pair=pair, p_big=0.0, direction="NEUTRAL", expected_magnitude_pct=0.0,
            expected_vs_adr=0.0, session=session, confidence=0.0, drivers=[],
            timestamp=ts, notes=["insufficient daily data for estimate"],
        )

    df = df_daily.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    # ── Features ──────────────────────────────────────────────────────────────
    compression = _compression(df)                  # [0,1] magnitude
    momentum = _momentum_z(df)                       # signed
    displacement = _displacement_dir(df)             # signed
    ctx, n_ctx = _norm_context(context)

    rate_diff_z = ctx["rate_diff_z"]
    vix_amp = ctx["vix_amp"]
    cot_extreme = ctx["cot_extreme"]
    event = ctx["event"]

    # ── P(big move): logit over magnitude features ────────────────────────────
    logit = _W_BIAS
    logit += _W_COMPRESSION * compression
    logit += _W_MOMENTUM_MAG * min(abs(momentum), 2.0) / 2.0
    if rate_diff_z is not None:
        logit += _W_RATE_DIFF_MAG * min(abs(rate_diff_z), 3.0) / 3.0
    if vix_amp is not None:
        logit += _W_VIX * vix_amp
    if event is not None:
        logit += _W_EVENT * event
    p_big = _sigmoid(logit)

    # ── Direction: signed driver score ────────────────────────────────────────
    dir_score = _W_DIR_MOMENTUM * (momentum / 3.0) + _W_DIR_DISPLACEMENT * (displacement / 2.0)
    if rate_diff_z is not None:
        dir_score += _W_DIR_RATE_DIFF * (rate_diff_z / 3.0)
    if dir_score > _DIRECTION_DEADBAND:
        direction = "LONG"
    elif dir_score < -_DIRECTION_DEADBAND:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # ── Expected magnitude: range as multiple of ADR, then as % of price ──────
    adr = _adr(df)
    last_close = float(df["Close"].iloc[-1])
    expected_vs_adr = _MAG_BASE + _MAG_SPAN * p_big
    if vix_amp is not None:
        expected_vs_adr += 0.30 * vix_amp
    predicted_range = adr * expected_vs_adr
    expected_magnitude_pct = (predicted_range / last_close * 100.0) if last_close > 0 else 0.0

    # ── Confidence: data sufficiency x context coverage x driver agreement ─────
    data_conf = float(np.clip(len(df) / _MIN_BARS, 0.3, 1.0))
    ctx_conf = 0.5 + 0.5 * (n_ctx / 4.0)               # 0.5 with no context, 1.0 with all four
    dir_conf = float(np.clip(abs(dir_score), 0.0, 1.0))  # stronger signed score -> more confident
    if event:
        dir_conf *= 0.75  # event days raise magnitude but blur direction
    confidence = float(np.clip(data_conf * ctx_conf * (0.4 + 0.6 * dir_conf), 0.0, 1.0))

    # ── Drivers (signed contributions, largest |magnitude| first) ─────────────
    drivers: List[Tuple[str, float]] = [
        ("compression", _W_COMPRESSION * compression),
        ("momentum", _W_DIR_MOMENTUM * (momentum / 3.0)),
        ("displacement", _W_DIR_DISPLACEMENT * (displacement / 2.0)),
    ]
    if rate_diff_z is not None:
        drivers.append(("rate_diff_z", _W_DIR_RATE_DIFF * (rate_diff_z / 3.0)))
    if vix_amp is not None:
        drivers.append(("vix", _W_VIX * vix_amp))
    if event is not None:
        drivers.append(("event_risk", _W_EVENT * event))
    if cot_extreme is not None:
        drivers.append(("cot_extreme", cot_extreme))
    drivers.sort(key=lambda d: abs(d[1]), reverse=True)

    notes: List[str] = []
    if n_ctx == 0:
        notes.append("no macro context supplied — price-only estimate, reduced confidence")

    return BigMoveEstimate(
        pair=pair,
        p_big=p_big,
        direction=direction,
        expected_magnitude_pct=expected_magnitude_pct,
        expected_vs_adr=expected_vs_adr,
        session=session,
        confidence=confidence,
        drivers=drivers,
        timestamp=ts,
        notes=notes,
    )


# ── Label helper (shared with the validation harness) ─────────────────────────── #

def big_move_label(df_daily: pd.DataFrame, idx: int) -> Optional[Tuple[bool, int]]:
    """
    Ground-truth label for bar `idx`: was that day a big-move day, and which way?

    Returns (is_big, direction_sign) or None if there isn't enough trailing data.
      is_big          — daily true range in the top tercile of the trailing window.
      direction_sign  — +1 if close > open, -1 if close < open, 0 if flat.

    Used ONLY by the validation harness to score the classifier against realized
    outcomes — never imported by the live path.
    """
    if df_daily is None or idx < _RANGE_PCTILE_LOOKBACK or idx >= len(df_daily):
        return None
    tr = _true_range(df_daily).dropna()
    if idx >= len(tr):
        return None
    window = tr.iloc[idx - _RANGE_PCTILE_LOOKBACK:idx]
    if len(window) < 20:
        return None
    threshold = float(window.quantile(_BIG_MOVE_PERCENTILE))
    today_tr = float(tr.iloc[idx])
    bar = df_daily.iloc[idx]
    sign = 1 if float(bar["Close"]) > float(bar["Open"]) else (-1 if float(bar["Close"]) < float(bar["Open"]) else 0)
    return (today_tr >= threshold, sign)


def _has_ohlc(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ("Open", "High", "Low", "Close"))
