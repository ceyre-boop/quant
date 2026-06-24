"""
sovereign/discovery/data_adapter.py
====================================
Per-track data + evaluation substrate for the discovery pipeline.

A *candidate* is a function that emits a signal array (+1/-1/0 per bar). The
adapter supplies (a) the OHLCV price frame for feature computation and (b) a
costed evaluator that runs any signal array through the SAME engine + cost model
+ stats the live system uses — so discovery results are methodologically
identical to `permutation_test_forex.py` / `holdout_validation_v014.py`.

Tracks:
  ForexDailyAdapter  — wraps ForexBatchBacktester (yfinance daily). FULLY built.
  NQIntradayAdapter  — loads data/es_nq/*.parquet. Price + features ready; the
                       costed evaluator is the one remaining wiring point (the NQ
                       simulator lives in sovereign/es_nq/, a separate engine).
  IntradayFXAdapter  — stub; requires purchased intraday FX data (see runbook).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalResult:
    sharpe: float
    n_trades: int
    trades: list  # list of trade dicts (for equity curves / visuals)


# ─── Forex daily (priority track) ─────────────────────────────────────────────

class ForexDailyAdapter:
    """Daily-bar forex substrate, reusing ForexBatchBacktester end-to-end."""

    track = "forex-daily"

    def __init__(self, start: str = "2015-01-01", end: str = "2024-12-31",
                 pairs: Optional[list[str]] = None):
        from sovereign.forex.batch_backtester import ForexBatchBacktester
        from sovereign.forex.pair_universe import ALL_PAIRS
        self.start, self.end = start, end
        self._pairs_req = list(pairs) if pairs else list(ALL_PAIRS)
        self.batch = ForexBatchBacktester(start=start, end=end)
        self._stop_pct = self.batch._backtester.STOP_PCT

    def preload(self) -> "ForexDailyAdapter":
        self.batch.preload(self._pairs_req)
        return self

    @property
    def pairs(self) -> list[str]:
        return sorted(self.batch._array_cache.keys())

    def price_df(self, pair: str) -> Optional[pd.DataFrame]:
        """Full OHLCV frame (Open/High/Low/Close) for feature computation."""
        return self.batch._price_cache.get(pair)

    def dataset(self, pair: str):
        """ForexArrayDataset: opens, closes, signals (the REAL macro signals), hold_days, index."""
        return self.batch._array_cache.get(pair)

    def index(self, pair: str) -> Optional[pd.Index]:
        ds = self.dataset(pair)
        return ds.index if ds is not None else None

    def eval_signals(self, pair: str, signals: np.ndarray,
                     window: Optional[tuple[str, str]] = None) -> EvalResult:
        """Run a signal array through the costed engine; return (sharpe, n, trades).

        `window` optionally restricts evaluation to a (start, end) date sub-range
        (e.g. the frozen holdout) by masking all arrays to that span.
        Reuses simulate_forex_trades_arrays + ForexBacktester._apply_costs + _compute_stats.
        """
        from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
        from sovereign.forex.forex_backtester import ForexBacktester
        ds = self.dataset(pair)
        if ds is None:
            return EvalResult(0.0, 0, [])
        opens, closes, hold, idx = ds.opens, ds.closes, ds.hold_days, ds.index
        sig = np.asarray(signals).astype(np.int8)
        if window is not None:
            mask = (idx >= pd.Timestamp(window[0])) & (idx <= pd.Timestamp(window[1]))
            opens, closes, hold, idx = opens[mask], closes[mask], hold[mask], idx[mask]
            sig = sig[mask]
        if len(idx) < 50:
            return EvalResult(0.0, 0, [])
        trades = simulate_forex_trades_arrays(
            opens=opens, closes=closes, signals=sig, hold_days=hold,
            stop_pct=self._stop_pct, index=idx,
        )
        if not trades:
            return EvalResult(0.0, 0, [])
        trades = ForexBacktester._apply_costs(trades, pair)
        st = self.batch._backtester._compute_stats(pair, trades, len(idx))
        for t in trades:
            t.setdefault("pair", pair)
        return EvalResult(float(st.sharpe), int(st.total_trades), trades)


# ─── NQ intraday (scaffolded — data exists) ───────────────────────────────────

class NQIntradayAdapter:
    """NQ futures intraday substrate. Price + features ready from cached parquet.

    The costed EVALUATOR is intentionally not wired here: NQ uses a separate
    simulator (sovereign/es_nq/backtest.py) with futures-specific costs and the
    adaptive ladder, not the forex kernel. `eval_signals` raises with a clear
    pointer so the wiring point is explicit. Everything up to the gate works.
    """

    track = "nq-intraday"
    PARQUET = {
        "1m": ROOT / "data" / "es_nq" / "nq_globex_1min.parquet",
        "5m": ROOT / "data" / "es_nq" / "nq_historical_5min.parquet",
        "1d": ROOT / "data" / "es_nq" / "nq_daily.parquet",
    }

    def __init__(self, timeframe: str = "5m"):
        self.timeframe = timeframe
        self._df: Optional[pd.DataFrame] = None

    def preload(self) -> "NQIntradayAdapter":
        path = self.PARQUET.get(self.timeframe)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"NQ parquet for timeframe {self.timeframe!r} not found at {path}. "
                f"Available: {[k for k, v in self.PARQUET.items() if v.exists()]}"
            )
        df = pd.read_parquet(path)
        # Normalise column casing to OHLCV the feature layer expects.
        rename = {c: c.capitalize() for c in df.columns if c.lower() in
                  ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=rename)
        self._df = df
        return self

    @property
    def pairs(self) -> list[str]:
        return ["NQ"]

    def price_df(self, pair: str = "NQ") -> Optional[pd.DataFrame]:
        return self._df

    def index(self, pair: str = "NQ") -> Optional[pd.Index]:
        return None if self._df is None else self._df.index

    def eval_signals(self, pair: str, signals: np.ndarray,
                     window: Optional[tuple[str, str]] = None) -> EvalResult:
        raise NotImplementedError(
            "NQ-intraday evaluation is the one remaining wiring point: route signals "
            "through the futures simulator at sovereign/es_nq/backtest.py (futures costs + "
            "adaptive ladder), not the forex kernel. Features/candidates already work — "
            "this is the completion step flagged in the plan."
        )


# ─── Intraday FX (blocked on vendor data) ─────────────────────────────────────

class IntradayFXAdapter:
    """Stub: 15yr intraday FX is not available from yfinance (caps ~60d at 15m).

    Requires a vendor (Dukascopy free / Polygon|OANDA paid) + a data-quality
    validation step. See docs/intraday_fx_acquisition.md.
    """

    track = "intraday-fx"

    def __init__(self, *_, **__):
        pass

    def preload(self):
        raise NotImplementedError(
            "intraday-fx requires purchased intraday FX data — yfinance cannot serve "
            "15yr of 1m/5m/15m forex (4H unsupported entirely). Acquire from Dukascopy "
            "(free) or Polygon/OANDA (paid), validate for drift, then implement this "
            "adapter like ForexDailyAdapter. Runbook: docs/intraday_fx_acquisition.md."
        )


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_adapter(track: str, **kwargs):
    if track == "forex-daily":
        return ForexDailyAdapter(**kwargs)
    if track == "nq-intraday":
        return NQIntradayAdapter(**kwargs)
    if track == "intraday-fx":
        return IntradayFXAdapter(**kwargs)
    raise ValueError(f"unknown track {track!r}; choose from forex-daily, nq-intraday, intraday-fx")
