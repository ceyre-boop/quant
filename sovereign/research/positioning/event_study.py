"""Event-study core for the positioning family (HYP-072/073/081 + 077's crossing logic).

Pure functions — no I/O, no network, no clock — so every semantic is offline-testable.

Locked-protocol semantics implemented here (each stamped verbatim on the seals):
- De-overlap = a hysteresis state machine: one Event per distinct crossing; re-armed only
  when the metric re-enters the re-arm region (per-HYP thresholds from the prereg).
- Effective date t0 = first trading day of the spot index >= publish_date; window return is
  close(t0) -> close(t0+h) log return. The CFTC 15:30 ET release precedes the ~17:00 ET FX
  daily close, so close(t0) is executable post-information — zero look-ahead.
- USDJPY inversion applied ONCE at Event construction (6J is JPY/USD: crowd-long-JPY fade
  means USDJPY RISES, so the pair-space side flips for USDJPY only).
- Null (locked: "event-label shuffle preserving per-pair event counts") read as PER-PAIR
  DATE-SHUFFLE: the k observed (side, strength) labels are reassigned without replacement to
  random eligible dates within the pair. Sign-flip nulls are rejected (destroyed by carry
  drift). Event-level shuffling is also the stated defense for overlapping forward windows —
  the null statistic inherits the same weekly-grid autocorrelation and overlap structure.
- One-sided Knuth p in the pre-registered direction: (n_ge + 1) / (N + 1), seed 42.
- ex-2020 = drop an event if ANY day of [t0, t0+h] falls in calendar 2020.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence

import numpy as np
import pandas as pd

PAIR_FLIP = {"USDJPY": -1}  # 6J quotes JPY/USD — pair-space direction flips for USDJPY only


@dataclass(frozen=True)
class Event:
    pair: str
    publish_date: date
    side: int          # ±1 in PAIR-quote space (inversion already applied)
    strength: float    # distance past the entry threshold (IC input)


@dataclass(frozen=True)
class PermResult:
    obs: float
    p: float
    n_perm: int
    n_events: int


def detect_crossings(dates: Sequence[date], values: Sequence[float],
                     enter: Callable[[float], int],
                     rearm: Callable[[float], bool]) -> list[tuple[date, int, float]]:
    """One (date, metric_side, value) per distinct crossing, hysteresis re-armed.

    ARMED -> enter(v) != 0 emits an event and DISARMS; re-ARMS only when rearm(v) is True.
    NaNs neither trigger nor re-arm.
    """
    out, armed = [], True
    for d, v in zip(dates, values):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if armed:
            side = enter(v)
            if side != 0:
                out.append((d, side, float(v)))
                armed = False
        elif rearm(v):
            armed = True
    return out


def make_events(pair: str, crossings: list[tuple[date, int, float]],
                metric_to_pair_side: Callable[[int], int],
                strength_fn: Callable[[float], float]) -> list[Event]:
    """Metric-space crossings -> pair-space Events (USDJPY inversion applied HERE, once)."""
    flip = PAIR_FLIP.get(pair, 1)
    return [Event(pair, d, metric_to_pair_side(side) * flip, strength_fn(v))
            for d, side, v in crossings]


def effective_t0(publish_date: date, trading_index: pd.DatetimeIndex,
                 max_lag_days: int = 7) -> pd.Timestamp | None:
    """First trading day >= publish_date, REQUIRED within max_lag_days of it.

    The lag guard prevents events older than the spot series (e.g. 1990s COT rows vs a
    2014-start yfinance index) from silently collapsing onto the index's first bar —
    they are unscoreable and must be dropped/counted, not remapped.
    """
    ts = pd.Timestamp(publish_date)
    pos = trading_index.searchsorted(ts)
    if pos >= len(trading_index):
        return None
    t0 = trading_index[pos]
    if (t0 - ts).days > max_lag_days:
        return None
    return t0


def forward_signed_return(closes: pd.Series, t0: pd.Timestamp, h: int, side: int) -> float | None:
    """side * ln(C[t0+h]/C[t0]); None when fewer than h closes exist after t0."""
    idx = closes.index
    pos = idx.get_indexer([t0])[0]
    if pos < 0 or pos + h >= len(idx):
        return None
    c0, ch = float(closes.iloc[pos]), float(closes.iloc[pos + h])
    if not (np.isfinite(c0) and np.isfinite(ch) and c0 > 0 and ch > 0):
        return None
    return side * math.log(ch / c0)


def window_touches_2020(t0: pd.Timestamp, h: int, trading_index: pd.DatetimeIndex) -> bool:
    pos = trading_index.get_indexer([t0])[0]
    end = trading_index[min(pos + h, len(trading_index) - 1)]
    return t0.year == 2020 or end.year == 2020 or (t0.year < 2020 < end.year)


def signed_returns(events: list[Event], closes_by_pair: dict[str, pd.Series], h: int,
                   ex2020: bool = False) -> tuple[list[float], list[Event], int]:
    """Direction-adjusted forward returns for scoreable events. Returns (rets, kept, dropped)."""
    rets, kept, dropped = [], [], 0
    for e in events:
        closes = closes_by_pair.get(e.pair)
        if closes is None or closes.empty:
            dropped += 1
            continue
        t0 = effective_t0(e.publish_date, closes.index)
        if t0 is None:
            dropped += 1
            continue
        if ex2020 and window_touches_2020(t0, h, closes.index):
            continue
        r = forward_signed_return(closes, t0, h, e.side)
        if r is None:
            dropped += 1
            continue
        rets.append(r)
        kept.append(e)
    return rets, kept, dropped


def eligible_dates(closes: pd.Series, weekly_dates: Sequence[date], h: int) -> list[date]:
    """Weekly publish dates whose forward h-day window fits inside the spot index."""
    idx = closes.index
    out = []
    for d in weekly_dates:
        t0 = effective_t0(d, idx)
        if t0 is None:
            continue
        pos = idx.get_indexer([t0])[0]
        if pos + h < len(idx):
            out.append(d)
    return out


def pooled_primary_p(events_by_pair: dict[str, list[Event]],
                     closes_by_pair: dict[str, pd.Series],
                     eligible_by_pair: dict[str, list[date]],
                     h: int, rng: np.random.Generator,
                     n_perm: int = 10000) -> PermResult:
    """Pooled median signed forward return + per-pair date-shuffle permutation p (one-sided).

    Null: within each pair, the k observed (side, strength) labels land on k random eligible
    dates (without replacement); pooled median recomputed per draw. p = (n_ge+1)/(N+1).
    """
    obs_rets, counts = [], {}
    for pair, evs in events_by_pair.items():
        rets, kept, _ = signed_returns(evs, closes_by_pair, h)
        obs_rets.extend(rets)
        counts[pair] = [e.side for e in kept]
    if not obs_rets:
        return PermResult(float("nan"), float("nan"), n_perm, 0)
    obs = float(np.median(obs_rets))

    # Precompute each pair's full eligible-date signed-return lookup per side (side just signs r)
    raw_by_pair: dict[str, np.ndarray] = {}
    for pair in counts:
        closes = closes_by_pair[pair]
        rs = []
        for d in eligible_by_pair.get(pair, []):
            t0 = effective_t0(d, closes.index)
            r = forward_signed_return(closes, t0, h, +1) if t0 is not None else None
            rs.append(np.nan if r is None else r)
        raw_by_pair[pair] = np.asarray(rs, dtype=float)

    n_ge = 0
    for _ in range(n_perm):
        draw = []
        for pair, sides in counts.items():
            raw = raw_by_pair[pair]
            ok = np.flatnonzero(np.isfinite(raw))
            if len(sides) == 0 or len(ok) < len(sides):
                continue
            pick = rng.choice(ok, size=len(sides), replace=False)
            draw.extend(np.asarray(sides, dtype=float) * raw[pick])
        if draw and float(np.median(draw)) >= obs:
            n_ge += 1
    return PermResult(obs, (n_ge + 1) / (n_perm + 1), n_perm, len(obs_rets))


def spearman_ic(strengths: Sequence[float], rets: Sequence[float]) -> float | None:
    if len(strengths) < 3 or len(strengths) != len(rets):
        return None
    from scipy.stats import spearmanr
    rho = spearmanr(strengths, rets).statistic
    return None if not np.isfinite(rho) else float(rho)


def binomial_hit(rets: Sequence[float]) -> dict:
    """Hit rate vs 0.5, one-sided greater, at the REAL de-overlapped N."""
    n = len(rets)
    wins = sum(1 for r in rets if r > 0)
    if n == 0:
        return {"n": 0, "wins": 0, "hit": None, "binom_p": None}
    from scipy.stats import binomtest
    return {"n": n, "wins": wins, "hit": wins / n,
            "binom_p": float(binomtest(wins, n, 0.5, alternative="greater").pvalue)}
