"""
Fast Vectorized Backtest Engine — M4 Pro Optimized

Replaces iterrows()-based loops with:
  1. Pre-loaded numpy float32 arrays (no pandas overhead per bar)
  2. Numba JIT-compiled inner loop (compiles once, ~cached 0.1ms/backtest)
  3. Vectorized ATR & signal computation

Usage:
    from backtest.fast_engine import FastBacktestEngine, SweepParams

    engine = FastBacktestEngine.from_dataframe(df)               # default SMA signal
    engine = FastBacktestEngine.from_signals(df, signal_array)   # your own signals

    result = engine.run_single(SweepParams(stop_atr_mult=2.0, tp_rr=2.5))

Pair with sweep.py for multi-core parameter sweeps.
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Tuple

# ── Numba import with graceful fallback ──────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:  # pragma: no cover
    def _njit(*args, **kwargs):           # type: ignore[misc]
        """No-op decorator when numba is not installed."""
        def _wrap(fn):
            return fn
        return _wrap if args and callable(args[0]) else _wrap
    _NUMBA = False


# ── Parameter & result containers ───────────────────────────────────────────

@dataclass
class SweepParams:
    """All knobs that can be swept over. Keep flat for easy pickling."""
    stop_atr_mult: float = 2.0        # ATR multiples for stop distance
    tp_rr: float = 2.0                # Reward:Risk for take-profit
    atr_period: int = 14              # ATR lookback bars
    signal_min_confidence: float = 0.55  # Threshold on signal array (0–1)
    commission_per_side: float = 2.5  # $ per side; round-trip = ×2
    slippage_pct: float = 0.0001      # 1 bp per entry
    initial_capital: float = 50_000.0

    def as_tuple(self) -> tuple:
        """Ordered tuple for numba/pickle passing."""
        return (
            self.stop_atr_mult,
            self.tp_rr,
            self.atr_period,
            self.signal_min_confidence,
            self.commission_per_side,
            self.slippage_pct,
            self.initial_capital,
        )

    @classmethod
    def from_tuple(cls, t: tuple) -> "SweepParams":
        return cls(*t)


@dataclass
class FastResult:
    """Lean result — no equity curve stored per-run (saves memory for 1000s of runs)."""
    total_pnl: float
    total_return_pct: float
    trade_count: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    params: SweepParams

    def to_dict(self) -> dict:
        d = {
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 3),
            "trades": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_dd_pct": round(self.max_drawdown_pct, 3),
        }
        d.update(asdict(self.params))
        return d


# ── Immutable data bundle shared across all sweep workers ───────────────────

@dataclass
class DataArrays:
    """Pre-cast float32 numpy arrays. Shared (via pickle) to worker processes once."""
    opens: np.ndarray    # shape (n,) float32
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    signals: np.ndarray  # shape (n,) int8 — 1=long, -1=short, 0=flat
    # Optional confidence weights (0.0–1.0) — used with signal_min_confidence
    confidence: np.ndarray  # float32, same length


# ── Numba JIT kernels ────────────────────────────────────────────────────────

@_njit(cache=True)
def _jit_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int) -> np.ndarray:
    """Wilder-smoothed ATR in a single JIT pass. Returns float32 array."""
    n = len(closes)
    atr = np.empty(n, dtype=np.float32)

    # Seed with simple average of first `period` true ranges
    seed = 0.0
    prev_close = closes[0]
    for i in range(period):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - prev_close)
        lc = abs(lows[i] - prev_close)
        tr = hl if (hl >= hc and hl >= lc) else (hc if hc >= lc else lc)
        seed += tr
        prev_close = closes[i]
    seed /= period
    for i in range(period):
        atr[i] = seed

    alpha = 1.0 / period
    prev = seed
    prev_close = closes[period - 1]
    for i in range(period, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - prev_close)
        lc = abs(lows[i] - prev_close)
        tr = hl if (hl >= hc and hl >= lc) else (hc if hc >= lc else lc)
        prev = prev * (1.0 - alpha) + tr * alpha
        atr[i] = prev
        prev_close = closes[i]

    return atr


@_njit(cache=True)
def _simulate_core(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,       # int8
    confidence: np.ndarray,    # float32
    atrs: np.ndarray,          # float32
    stop_atr_mult: float,
    tp_rr: float,
    min_confidence: float,
    commission_rt: float,      # round-trip commission in $
    slippage_pct: float,
    initial_capital: float,
) -> Tuple[float, int, float, float, float]:
    """
    Core trade-simulation loop — compiled once, runs in ~0.05–0.2 ms.

    Returns:
        (total_pnl, trade_count, win_rate, profit_factor, max_drawdown_pct)
    """
    n = len(closes)
    capital = initial_capital
    peak_capital = initial_capital

    in_trade = False
    direction = 0         # 1=long  -1=short
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0

    trade_count = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    max_dd = 0.0

    for i in range(n):

        if in_trade:
            # ── Check exits ──────────────────────────────────────
            if direction == 1:  # long
                if lows[i] <= stop_price:
                    ep = opens[i] if opens[i] < stop_price else stop_price
                    pnl = (ep - entry_price) - commission_rt
                    capital += pnl
                    trade_count += 1
                    if pnl > 0.0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss -= pnl
                    in_trade = False
                elif highs[i] >= tp_price:
                    pnl = (tp_price - entry_price) - commission_rt
                    capital += pnl
                    trade_count += 1
                    wins += 1
                    gross_profit += pnl
                    in_trade = False

            else:  # short  (direction == -1)
                if highs[i] >= stop_price:
                    ep = opens[i] if opens[i] > stop_price else stop_price
                    pnl = (entry_price - ep) - commission_rt
                    capital += pnl
                    trade_count += 1
                    if pnl > 0.0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss -= pnl
                    in_trade = False
                elif lows[i] <= tp_price:
                    pnl = (entry_price - tp_price) - commission_rt
                    capital += pnl
                    trade_count += 1
                    wins += 1
                    gross_profit += pnl
                    in_trade = False

            # Update drawdown after any exit or equity move
            if capital > peak_capital:
                peak_capital = capital
            else:
                dd = (peak_capital - capital) / peak_capital * 100.0
                if dd > max_dd:
                    max_dd = dd

        elif signals[i] != 0 and confidence[i] >= min_confidence:
            # ── Open trade ───────────────────────────────────────
            direction = int(signals[i])
            atr = atrs[i]
            slip = closes[i] * slippage_pct
            entry_price = closes[i] + direction * slip
            stop_dist = stop_atr_mult * atr
            stop_price = entry_price - direction * stop_dist
            tp_price = entry_price + direction * stop_dist * tp_rr
            in_trade = True

    win_rate = wins / trade_count if trade_count > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0.0 else 999.0
    return capital - initial_capital, trade_count, win_rate, pf, max_dd


# ── Detailed simulation JIT (per-trade records) ─────────────────────────────

@_njit(cache=True)
def _simulate_detailed(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,
    confidence: np.ndarray,
    atrs: np.ndarray,
    stop_atr_mult: float,
    tp_rr: float,
    min_confidence: float,
    commission_rt: float,
    slippage_pct: float,
    initial_capital: float,
):
    """Same trade loop as _simulate_core but records per-trade arrays.

    Returns:
        (entry_idx, exit_idx, pnl, pnl_r, direction, is_stop,
         total_pnl, trade_count, win_rate, profit_factor, max_dd_pct)
    """
    max_trades = 1000
    t_entry  = np.full(max_trades, -1, dtype=np.int32)
    t_exit   = np.full(max_trades, -1, dtype=np.int32)
    t_pnl    = np.zeros(max_trades, dtype=np.float32)
    t_pnl_r  = np.zeros(max_trades, dtype=np.float32)
    t_dir    = np.zeros(max_trades, dtype=np.int8)
    t_stop   = np.zeros(max_trades, dtype=np.int8)   # 1 = stopped out

    n = len(closes)
    capital = initial_capital
    peak_capital = initial_capital
    in_trade = False
    direction = 0
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    risk_dist = 0.0
    entry_bar = 0
    tc = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    max_dd = 0.0

    for i in range(n):
        if in_trade:
            exited = False
            is_stop_hit = False
            ep = 0.0

            if direction == 1:
                if lows[i] <= stop_price:
                    ep = opens[i] if opens[i] < stop_price else stop_price
                    exited = True
                    is_stop_hit = True
                elif highs[i] >= tp_price:
                    ep = tp_price
                    exited = True
            else:
                if highs[i] >= stop_price:
                    ep = opens[i] if opens[i] > stop_price else stop_price
                    exited = True
                    is_stop_hit = True
                elif lows[i] <= tp_price:
                    ep = tp_price
                    exited = True

            if exited and tc < max_trades:
                raw_pnl = direction * (ep - entry_price) - commission_rt
                raw_pnl_r = raw_pnl / (risk_dist + 1e-8)
                capital += raw_pnl
                t_exit[tc]  = i
                t_pnl[tc]   = raw_pnl
                t_pnl_r[tc] = raw_pnl_r
                t_stop[tc]  = 1 if is_stop_hit else 0
                if raw_pnl > 0.0:
                    wins += 1
                    gross_profit += raw_pnl
                else:
                    gross_loss -= raw_pnl
                tc += 1
                in_trade = False
                if capital > peak_capital:
                    peak_capital = capital
                else:
                    dd = (peak_capital - capital) / peak_capital * 100.0
                    if dd > max_dd:
                        max_dd = dd

        elif signals[i] != 0 and confidence[i] >= min_confidence and tc < max_trades:
            direction = int(signals[i])
            atr = atrs[i]
            slip = closes[i] * slippage_pct
            entry_price = closes[i] + direction * slip
            risk_dist = stop_atr_mult * atr
            stop_price = entry_price - direction * risk_dist
            tp_price = entry_price + direction * risk_dist * tp_rr
            t_entry[tc] = i
            t_dir[tc] = np.int8(direction)
            entry_bar = i
            in_trade = True

    win_rate = wins / tc if tc > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0.0 else 999.0

    return (
        t_entry[:tc], t_exit[:tc], t_pnl[:tc], t_pnl_r[:tc],
        t_dir[:tc], t_stop[:tc],
        capital - initial_capital, tc, win_rate, pf, max_dd,
    )


# ── Default signal generators (pure numpy, fast) ────────────────────────────

def _sma_crossover_signals(closes: np.ndarray,
                            fast: int = 9, slow: int = 21
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Simple SMA crossover: +1 when fast > slow, -1 when fast < slow.

    Returns (signal_int8, confidence_float32).
    """
    def _sma(arr, w):
        out = np.empty_like(arr)
        out[:w] = arr[:w].mean()
        cs = np.cumsum(arr)
        out[w:] = (cs[w:] - cs[:-w]) / w
        return out

    fast_ma = _sma(closes, fast)
    slow_ma = _sma(closes, slow)
    diff = fast_ma - slow_ma
    prev_diff = np.roll(diff, 1)
    prev_diff[0] = diff[0]

    signals = np.zeros(len(closes), dtype=np.int8)
    # Long: crossover up
    signals[(diff > 0) & (prev_diff <= 0)] = 1
    # Short: crossover down
    signals[(diff < 0) & (prev_diff >= 0)] = -1

    # Confidence: normalised absolute slope
    slope = np.abs(diff - prev_diff) / (slow_ma + 1e-9)
    conf = np.clip(slope * 1000.0, 0.0, 1.0).astype(np.float32)
    return signals, conf


# ── Engine class ─────────────────────────────────────────────────────────────

class FastBacktestEngine:
    """Thin wrapper around pre-loaded DataArrays + JIT kernels.

    One instance per dataset. Call run_single() many times with different
    SweepParams to sweep parameters without re-loading data.
    """

    def __init__(self, arrays: DataArrays):
        self.arrays = arrays
        self._n_bars = len(arrays.closes)

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        signal_fn: Optional[Callable] = None,
        fast_sma: int = 9,
        slow_sma: int = 21,
    ) -> "FastBacktestEngine":
        """Build from a standard OHLCV DataFrame (columns: open/high/low/close/volume).

        Args:
            df: OHLCV DataFrame, any DatetimeIndex.
            signal_fn: Optional callable(df) → (signal_int8_array, confidence_float32_array).
                       If None, uses built-in SMA crossover for demo purposes.
            fast_sma: Fast SMA period (ignored when signal_fn provided).
            slow_sma: Slow SMA period (ignored when signal_fn provided).
        """
        opens  = df["open"].to_numpy(dtype=np.float32)
        highs  = df["high"].to_numpy(dtype=np.float32)
        lows   = df["low"].to_numpy(dtype=np.float32)
        closes = df["close"].to_numpy(dtype=np.float32)

        if signal_fn is not None:
            result = signal_fn(df)
            if isinstance(result, tuple):
                signals, confidence = result
            else:
                signals = result
                confidence = np.ones(len(signals), dtype=np.float32)
            signals = np.asarray(signals, dtype=np.int8)
            confidence = np.asarray(confidence, dtype=np.float32)
        else:
            signals, confidence = _sma_crossover_signals(closes, fast_sma, slow_sma)

        arrays = DataArrays(
            opens=opens, highs=highs, lows=lows, closes=closes,
            signals=signals, confidence=confidence,
        )
        return cls(arrays)

    @classmethod
    def from_signals(
        cls,
        df: pd.DataFrame,
        signals: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> "FastBacktestEngine":
        """Build from pre-computed signal array (e.g. from your 3-layer system).

        Args:
            df: OHLCV DataFrame.
            signals: int8 array — 1=long, -1=short, 0=flat.
            confidence: float32 array (0–1). Defaults to all-ones (always trade).
        """
        opens  = df["open"].to_numpy(dtype=np.float32)
        highs  = df["high"].to_numpy(dtype=np.float32)
        lows   = df["low"].to_numpy(dtype=np.float32)
        closes = df["close"].to_numpy(dtype=np.float32)

        signals_arr = np.asarray(signals, dtype=np.int8)
        if confidence is None:
            conf_arr = np.ones(len(signals_arr), dtype=np.float32)
        else:
            conf_arr = np.asarray(confidence, dtype=np.float32)

        arrays = DataArrays(
            opens=opens, highs=highs, lows=lows, closes=closes,
            signals=signals_arr, confidence=conf_arr,
        )
        return cls(arrays)

    # ── Warm-up (force JIT compilation) ──────────────────────────────────

    def warmup(self) -> float:
        """Trigger JIT compilation on a tiny slice. Returns compile time in seconds.

        Call this once at startup so the first real sweep runs at full speed.
        """
        tiny = DataArrays(
            opens=self.arrays.opens[:50].copy(),
            highs=self.arrays.highs[:50].copy(),
            lows=self.arrays.lows[:50].copy(),
            closes=self.arrays.closes[:50].copy(),
            signals=self.arrays.signals[:50].copy(),
            confidence=self.arrays.confidence[:50].copy(),
        )
        eng = FastBacktestEngine(tiny)
        t0 = time.perf_counter()
        eng.run_single(SweepParams())
        return time.perf_counter() - t0

    # ── Single run ────────────────────────────────────────────────────────

    def run_single(self, params: SweepParams) -> FastResult:
        """Run one backtest with the given parameters.

        On a warm JIT cache (~0.1–0.5 ms for 1 year of 5-min data).
        """
        atrs = _jit_atr(
            self.arrays.highs, self.arrays.lows, self.arrays.closes,
            params.atr_period,
        )

        total_pnl, trade_count, win_rate, pf, max_dd = _simulate_core(
            self.arrays.opens,
            self.arrays.highs,
            self.arrays.lows,
            self.arrays.closes,
            self.arrays.signals,
            self.arrays.confidence,
            atrs,
            float(params.stop_atr_mult),
            float(params.tp_rr),
            float(params.signal_min_confidence),
            float(params.commission_per_side * 2),   # round-trip
            float(params.slippage_pct),
            float(params.initial_capital),
        )

        ret_pct = (total_pnl / params.initial_capital) * 100.0

        return FastResult(
            total_pnl=total_pnl,
            total_return_pct=ret_pct,
            trade_count=trade_count,
            win_rate=win_rate,
            profit_factor=pf,
            max_drawdown_pct=max_dd,
            params=params,
        )

    # ── Convenience ───────────────────────────────────────────────────────

    def benchmark(self, n: int = 500) -> None:
        """Print how many backtests/second this engine achieves on one core."""
        params = SweepParams()
        # Ensure JIT is warm
        self.run_single(params)

        t0 = time.perf_counter()
        for _ in range(n):
            self.run_single(params)
        elapsed = time.perf_counter() - t0
        rate = n / elapsed
        print(
            f"[FastEngine] {n:,} runs on 1 core in {elapsed*1000:.1f} ms "
            f"→ {rate:,.0f} backtests/sec  ({elapsed/n*1000:.3f} ms each)"
        )

    # ── Detailed run (per-trade records for failure analysis) ─────────────────

    def run_detailed(self, params: SweepParams) -> dict:
        """Run one backtest and return per-trade arrays alongside summary metrics.

        Returns dict with keys:
            entry_idx, exit_idx, pnl, pnl_r, direction, is_stop  ← per-trade arrays
            total_pnl, trade_count, win_rate, profit_factor, max_drawdown_pct  ← summary
        """
        atrs = _jit_atr(
            self.arrays.highs, self.arrays.lows, self.arrays.closes,
            params.atr_period,
        )

        (entry_idx, exit_idx, pnl_arr, pnl_r_arr,
         dir_arr, stop_arr,
         total_pnl, tc, win_rate, pf, max_dd) = _simulate_detailed(
            self.arrays.opens,
            self.arrays.highs,
            self.arrays.lows,
            self.arrays.closes,
            self.arrays.signals,
            self.arrays.confidence,
            atrs,
            float(params.stop_atr_mult),
            float(params.tp_rr),
            float(params.signal_min_confidence),
            float(params.commission_per_side * 2),
            float(params.slippage_pct),
            float(params.initial_capital),
        )

        ret_pct = (total_pnl / params.initial_capital) * 100.0

        return {
            # Per-trade (trimmed to actual trade count)
            "entry_idx":  entry_idx,
            "exit_idx":   exit_idx,
            "pnl":        pnl_arr,
            "pnl_r":      pnl_r_arr,
            "direction":  dir_arr,
            "is_stop":    stop_arr,
            # Summary
            "total_pnl":        total_pnl,
            "total_return_pct": ret_pct,
            "trade_count":      tc,
            "win_rate":         win_rate,
            "profit_factor":    pf,
            "max_drawdown_pct": max_dd,
        }
