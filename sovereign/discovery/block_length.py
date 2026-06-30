#!/usr/bin/env python3
"""sovereign/discovery/block_length.py — locked stationary-block-bootstrap block length L.

Computes the expected geometric block length ``L`` for the regime-conditional stationary block
bootstrap (Politis-Romano 1994) used by the HYP-071 exit value function. See the pre-registration:
``data/research/preregister/HYP-071_tabular_exit_value.yaml``.

LOCKED RULE (pre-registered — DO NOT TUNE):
    L = round( mean over the four pairs of [ first lag k>=1 where the daily ATR% INNOVATION
        (first difference) autocorrelation drops below 1/e ] ),  clamped to [5, 60] trading days.

v2 CORRECTION: the crossing is measured on the ATR% INNOVATION (first difference), not the LEVEL. The
level is a 14-day SMA — autocorrelated to ~lag 14 by construction — so measuring it double-counts the
mechanical smoothing on top of genuine vol persistence (v1 clamped to its ceiling for this reason). The
first difference Δa_t = (TR_t - TR_{t-14})/(14·C) strips that artifact, leaving genuine TR dynamics. The
level path remains available (series="level") but UNUSED, for audit. Clamp widened to [5, 60] so it only
catches degenerate cases; if the innovation crossing still exceeds 60 on most pairs the helper RAISES
BlockLengthDegenerate rather than clamp-and-proceed.

This reads daily ATR% — a property of the data-generating process, not a strategy result — to set
ONE hyperparameter that is then frozen into the prereg. It computes NO labels, values, or Sharpe.

ATR% matches the live exit machine EXACTLY by reusing ``ForexSignalEngine._compute_atr_pct``
(True Range -> 14-day SMA -> / Close), so there is one source of truth — NOT discovery/features.py's
Wilder-EMA variant.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── repo root importable when run by absolute path (sys.path[0] = discovery/) ──────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.forex.pair_universe import MAJOR_PAIRS
from sovereign.forex.signal_engine import ForexSignalEngine

for _lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

INV_E = 1.0 / np.e          # ≈ 0.367879441 — the 1/e autocorrelation crossing threshold
BLOCK_MIN = 5               # locked clamp floor (trading days)
BLOCK_MAX = 60              # locked clamp ceiling (trading days) — wide so it only catches degenerate cases
ATR_PERIOD = 14             # matches ForexSignalEngine._compute_atr_pct default
DEFAULT_MAX_LAG = 250       # ~1 trading year; cap on the crossing search
SERIES_LOCKED = "innovations"   # v2 locked choice: measure on Δ(ATR%), not the SMA-autocorrelated level


class BlockLengthDegenerate(RuntimeError):
    """Raised when the chosen-series crossing exceeds BLOCK_MAX on most pairs — daily-bar vol persistence
    beyond what this bootstrap can represent. A design conversation, not a parameter to clamp."""


# ── pure numerics (network-free, unit-tested) ──────────────────────────────────────────────────────
def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation r[0..max_lag] of a 1-D series (r[0] == 1.0)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.array([1.0])
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:                       # constant series → no decay information
        return np.array([1.0])
    max_lag = int(min(max_lag, n - 1))
    r = np.empty(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        r[k] = float(np.dot(x[: n - k], x[k:]) / denom)
    return r


def _innovations(x: np.ndarray) -> np.ndarray:
    """First difference (innovation) of a 1-D series — strips the level's mechanical SMA autocorrelation,
    leaving the genuine vol dynamics. Finite values only; returns length max(0, n-1)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return np.diff(x)


def _series_crossing(atr_level: np.ndarray, series: str, max_lag: int = DEFAULT_MAX_LAG) -> int:
    """1/e crossing lag for the requested view of an ATR% level series.

    series="innovations" (locked): crossing of the first difference Δ(ATR%).
    series="level"      (audit):   crossing of the raw ATR% level.
    """
    if series == "innovations":
        return _crossing_lag(_innovations(atr_level), max_lag)
    if series == "level":
        return _crossing_lag(np.asarray(atr_level, dtype=np.float64), max_lag)
    raise ValueError(f"unknown series {series!r} — expected 'innovations' or 'level'")


def _crossing_lag(atr_pct: np.ndarray, max_lag: int = DEFAULT_MAX_LAG) -> int:
    """First lag k>=1 where ATR% autocorrelation drops below 1/e.

    If it never crosses within ``max_lag`` (slow-decay / highly persistent series — the case the
    clamp ceiling exists to cap), return the largest lag examined.
    """
    r = _autocorr(atr_pct, max_lag)
    for k in range(1, r.size):
        if r[k] < INV_E:
            return k
    return max(1, r.size - 1)


def _aggregate_clamp(lags) -> int:
    """Mean of per-pair crossing lags, rounded, clamped to [BLOCK_MIN, BLOCK_MAX]."""
    vals = [int(l) for l in lags if l is not None]
    if not vals:
        raise ValueError("no crossing lags to aggregate")
    mean_lag = float(np.mean(vals))
    return int(min(BLOCK_MAX, max(BLOCK_MIN, round(mean_lag))))


# ── data access (the one permitted data touch) ─────────────────────────────────────────────────────
def _default_loader(pair: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLC from yfinance, matching the house download call (auto_adjust, flattened columns)."""
    import yfinance as yf

    df = yf.download(pair, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def atr_pct_for_pair(pair: str, start: str, end: str, loader=None) -> pd.Series:
    """Daily ATR% series for one pair, via the live exit-machine ATR% definition."""
    loader = loader or _default_loader
    df = loader(pair, start, end)
    series = ForexSignalEngine._compute_atr_pct(df["Close"], df, period=ATR_PERIOD)
    if series is None:
        raise ValueError(f"ATR% computation returned None for {pair}")
    return series.dropna()


def compute_atr_block_length(
    pairs=None,
    start: str = "2015-01-01",
    end: str = "2026-06-30",
    *,
    series: str = SERIES_LOCKED,
    loader=None,
    return_detail: bool = False,
):
    """Compute the locked stationary-block-bootstrap block length L.

    Args:
        pairs: iterable of yfinance tickers; defaults to ``pair_universe.MAJOR_PAIRS`` (the four pairs).
        start, end: date window for the ATR% autocorrelation read.
        series: "innovations" (locked) measures Δ(ATR%); "level" (audit) measures the raw level.
        loader: optional ``(pair, start, end) -> DataFrame`` injection (tests pass synthetic data).
        return_detail: if True, also return an audit dict with BOTH innovation and level crossings.

    Returns:
        int L in [BLOCK_MIN, BLOCK_MAX], or ``(L, detail)`` when ``return_detail`` is True.

    Raises:
        BlockLengthDegenerate: the chosen-series crossing exceeds BLOCK_MAX on most pairs.
    """
    pairs = list(pairs) if pairs is not None else list(MAJOR_PAIRS)
    innovation_crossings: dict[str, int] = {}
    level_crossings: dict[str, int] = {}
    for pair in pairs:
        atr = atr_pct_for_pair(pair, start, end, loader=loader).to_numpy()
        innovation_crossings[pair] = _series_crossing(atr, "innovations")
        level_crossings[pair] = _series_crossing(atr, "level")

    chosen = innovation_crossings if series == "innovations" else level_crossings

    # Stop-and-report: if the chosen series still overshoots the (wide) ceiling on most pairs, this is a
    # design problem, not a clamp candidate — vol persistence exceeds what this bootstrap can represent.
    overshoot = sum(1 for c in chosen.values() if c > BLOCK_MAX)
    if overshoot > len(chosen) / 2:
        raise BlockLengthDegenerate(
            f"{overshoot}/{len(chosen)} {series} crossings exceed BLOCK_MAX={BLOCK_MAX}; refusing to "
            f"clamp-and-proceed. crossings={chosen}. This is a design conversation (daily-bar vol "
            f"persistence beyond the bootstrap's representable range), not a parameter."
        )

    L = _aggregate_clamp(chosen.values())
    if return_detail:
        detail = {
            "series_locked": series,
            "innovation_crossings": innovation_crossings,
            "level_crossings": level_crossings,
            "mean_innovation_crossing": round(float(np.mean(list(innovation_crossings.values()))), 1),
            "mean_level_crossing": round(float(np.mean(list(level_crossings.values()))), 1),
            "clamp": [BLOCK_MIN, BLOCK_MAX],
            "L_locked": L,
        }
        return L, detail
    return L


if __name__ == "__main__":
    try:
        L, detail = compute_atr_block_length(return_detail=True)
    except BlockLengthDegenerate as exc:
        print("BLOCK LENGTH DEGENERATE — halting:\n ", exc)
        raise SystemExit(1)
    print("ATR% INNOVATION (Δ) 1/e crossing per pair:", detail["innovation_crossings"],
          "| mean", detail["mean_innovation_crossing"])
    print("ATR% LEVEL 1/e crossing per pair (audit) :", detail["level_crossings"],
          "| mean", detail["mean_level_crossing"])
    print(f"clamp {detail['clamp']}  ->  block_length_L = {L}")
    print("block_length_L =", L)
