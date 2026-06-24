"""
sovereign/discovery/equity_adapter.py
=====================================
Equity-index discovery substrate (NQ) on a CLEAN source, with a generic bar
evaluator so the same machinery (features/candidates/gate/validation/visuals)
that ran on forex runs here unchanged.

Two backends for the same asset, so the discovery delta isolates the DATA SOURCE:
  • parquet  — the clean on-disk NQ daily RTH bars (data/es_nq/nq_daily.parquet),
               broker-grade, non-drifting.
  • yfinance — NQ=F continuous front-month (the drifting source), for the
               apples-to-apples comparison.

The evaluator reuses fast_backtester.simulate_forex_trades_arrays (the kernel is
asset-agnostic: opens/closes/signals/hold/ATR-stop) with a flat per-trade cost
instead of the forex spread/swap model. Sharpe is computed with the same
empirical-annualization formula as ForexBacktester._compute_stats so the gate's
numbers are directly comparable to the forex run.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

from sovereign.discovery.data_adapter import EvalResult

ROOT = Path(__file__).resolve().parents[2]
NQ_DAILY = ROOT / "data" / "es_nq" / "nq_daily.parquet"


class EquityIndexAdapter:
    track = "equity"

    def __init__(self, source: str = "parquet", symbol: str = "NQ",
                 start: str = "2018-01-01", end: str = "2026-06-09",
                 hold: int = 10, stop_pct: float = 0.04, cost_bps: float = 2.0):
        self.source = source
        self.symbol = symbol
        self.start, self.end = start, end
        self.hold, self.stop_pct, self.cost_bps = hold, stop_pct, cost_bps
        self._df: Optional[pd.DataFrame] = None

    # -- loading --
    def _load(self) -> pd.DataFrame:
        if self.source == "parquet":
            df = pd.read_parquet(NQ_DAILY)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.rename(columns={"rth_open": "Open", "rth_high": "High",
                                    "rth_low": "Low", "rth_close": "Close"})
            df = df[["Open", "High", "Low", "Close"]].dropna()
        elif self.source == "yfinance":
            import yfinance as yf
            tick = "NQ=F" if self.symbol == "NQ" else self.symbol
            df = yf.download(tick, start=self.start, end=self.end, progress=False, auto_adjust=True)
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close"]].dropna()
        else:
            raise ValueError(f"unknown source {self.source!r} (parquet|yfinance)")
        return df[(df.index >= pd.Timestamp(self.start)) & (df.index <= pd.Timestamp(self.end))]

    def preload(self) -> "EquityIndexAdapter":
        self._df = self._load()
        return self

    @property
    def pairs(self) -> list[str]:
        return [self.symbol]

    def price_df(self, pair: str = "NQ") -> Optional[pd.DataFrame]:
        return self._df

    def index(self, pair: str = "NQ") -> Optional[pd.Index]:
        return None if self._df is None else self._df.index

    def dataset(self, pair: str = "NQ"):
        """Momentum base signal — only used by the look-ahead canary's 'real' control
        (the generic discovery candidates generate their own signals from features)."""
        close = self._df["Close"]
        sig = np.sign(close.pct_change(20).fillna(0.0)).astype(np.int8).to_numpy()
        return SimpleNamespace(signals=sig, hold_days=np.full(len(self._df), self.hold, dtype=np.int32),
                               index=self._df.index)

    # -- generic costed evaluator --
    def eval_signals(self, pair: str, signals: np.ndarray,
                     window: Optional[tuple[str, str]] = None) -> EvalResult:
        from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
        df = self._df
        opens = df["Open"].to_numpy(dtype=np.float64)
        closes = df["Close"].to_numpy(dtype=np.float64)
        idx = df.index
        sig = np.asarray(signals).astype(np.int8)
        if window is not None:
            m = (idx >= pd.Timestamp(window[0])) & (idx <= pd.Timestamp(window[1]))
            opens, closes, idx, sig = opens[m], closes[m], idx[m], sig[m]
        if len(idx) < 50:
            return EvalResult(0.0, 0, [])
        hold = np.full(len(idx), self.hold, dtype=np.int32)
        trades = simulate_forex_trades_arrays(
            opens=opens, closes=closes, signals=sig, hold_days=hold,
            stop_pct=self.stop_pct, index=idx)
        if not trades:
            return EvalResult(0.0, 0, [])
        rt_cost = 2.0 * self.cost_bps / 10000.0  # round-trip flat cost on the fractional return
        for t in trades:
            t["pnl_pct"] = float(t.get("pnl_pct", 0.0)) - rt_cost
            t["risk_adjusted_pnl_pct"] = t["pnl_pct"] * float(t.get("risk_pct", 0.0075))
            t.setdefault("pair", pair)
        sharpe, n = self._stats(trades, len(idx))
        return EvalResult(sharpe, n, trades)

    @staticmethod
    def _stats(trades: list, n_bars: int) -> tuple[float, int]:
        """Sharpe + n with the same empirical annualization as ForexBacktester._compute_stats."""
        pnls = np.array([t["pnl_pct"] for t in trades], dtype=float)
        n = len(pnls)
        if n < 2:
            return 0.0, n
        years = max(n_bars / 252.0, 1e-9)
        equity = np.cumprod(1.0 + pnls)
        returns = np.diff(np.log(equity), prepend=0.0)
        ann = np.sqrt(max(n, 1) / years)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * ann)
        return round(sharpe, 3), n
