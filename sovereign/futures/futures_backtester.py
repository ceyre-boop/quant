#!/usr/bin/env python3
"""Costed ES/NQ futures backtester (research).

Mirrors sovereign/forex/forex_backtester.py's discipline so futures numbers are HONEST and
comparable to the forex edge:
  - costs as a fraction of price (CME micro commission + slippage, charged per round trip),
  - equity = cumprod(1+pnl), Sharpe = mean/std · √(n/years) [empirical trades/year, NOT 252],
  - √n-aware Sharpe CI.

A signal is a function `frames -> position Series` (values in {-1,0,1}) giving the position to
hold in the TRADED instrument (default ES=F) for each day's return. Trades = days the position
is non-zero; each incurs one round-trip cost.

This is research, not an edge — verdicts come only from run_futures_hypothesis.py's gates.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for _lib in ("yfinance", "urllib3", "requests"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# CME micro contract specs.
SYMBOL_TO_MICRO = {"NQ=F": "MNQ", "ES=F": "MES"}
DOLLARS_PER_POINT = {"MNQ": 2.0, "MES": 5.0}
TICK = {"NQ=F": 0.25, "ES=F": 0.25}
SLIPPAGE_TICKS_PER_SIDE = 1.0        # 1 tick each side (conservative for index futures)
COMMISSION_PER_RT = 0.74             # ~$0.74 round-turn per micro contract


@dataclass
class FuturesBacktestResult:
    symbol: str
    signal: str
    sharpe: float
    sharpe_ci: tuple
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    years: float
    trades_per_year: float
    avg_pnl_pct: float
    pnls: list = field(default_factory=list, repr=False)


def _cost_fraction(symbol: str, price: float) -> float:
    """Round-trip transaction cost as a fraction of price (slippage + commission)."""
    if price <= 0:
        return 0.0
    tick = TICK.get(symbol, 0.25)
    slip_pts = SLIPPAGE_TICKS_PER_SIDE * tick * 2          # entry + exit
    dpp = DOLLARS_PER_POINT.get(SYMBOL_TO_MICRO.get(symbol, "MNQ"), 2.0)
    comm_pts = COMMISSION_PER_RT / dpp
    return (slip_pts + comm_pts) / price


def sharpe_ci(sr: float, n: int) -> tuple:
    """95% CI for a Sharpe estimate: SE = √((1 + 0.5·SR²)/n)."""
    if n < 2:
        return (None, None)
    se = float(np.sqrt((1 + 0.5 * sr ** 2) / n))
    return (round(float(sr) - 1.96 * se, 3), round(float(sr) + 1.96 * se, 3))


class FuturesBacktester:
    def __init__(self):
        self._cache: dict = {}

    def load(self, symbol: str, start: str, end: str):
        key = (symbol, start, end)
        if key in self._cache:
            return self._cache[key]
        import yfinance as yf
        h = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
        self._cache[key] = h
        return h

    def run_signal(self, signal_fn: Callable, start: str, end: str,
                   traded: str = "ES=F", lead: str = "NQ=F") -> Optional[FuturesBacktestResult]:
        """Backtest `signal_fn` on `traded`'s daily returns. Returns None if data is thin."""
        df_traded = self.load(traded, start, end)
        df_lead = self.load(lead, start, end)
        if df_traded is None or len(df_traded) < 30:
            return None

        frames = {traded: df_traded, lead: df_lead}
        pos = signal_fn(frames)                                  # position Series in {-1,0,1}
        ret = df_traded["Close"].astype(float).pct_change()
        common = pos.index.intersection(ret.index)
        pos = pos.reindex(common).fillna(0.0)
        ret = ret.reindex(common).fillna(0.0)

        pnls = []
        prices = df_traded["Close"].reindex(common).astype(float)
        for ts in common:
            p = float(pos.loc[ts])
            if p == 0:
                continue
            cost = _cost_fraction(traded, float(prices.loc[ts]))
            pnls.append(p * float(ret.loc[ts]) - cost)

        n = len(pnls)
        n_bars = len(common)
        years = n_bars / 252.0
        if n < 2:
            return FuturesBacktestResult(traded, getattr(signal_fn, "__name__", "signal"),
                                         0.0, (None, None), 0.0, 0.0, 0.0, n,
                                         round(years, 1), 0.0, 0.0, pnls)

        pnls_arr = np.array(pnls)
        wins = pnls_arr[pnls_arr > 0]
        losses = pnls_arr[pnls_arr <= 0]
        win_rate = len(wins) / n
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 1e-6
        profit_factor = min(gross_win / gross_loss, 20.0)

        equity = np.cumprod(1 + pnls_arr)
        rets = np.diff(np.log(equity), prepend=0)
        ann = np.sqrt(max(n, 1) / max(years, 1e-9))
        sr = float(np.mean(rets) / (np.std(rets) + 1e-9) * ann)
        roll_max = np.maximum.accumulate(equity)
        max_dd = float(((equity - roll_max) / roll_max).min())

        return FuturesBacktestResult(
            symbol=traded, signal=getattr(signal_fn, "__name__", "signal"),
            sharpe=round(sr, 3), sharpe_ci=sharpe_ci(sr, n),
            win_rate=round(win_rate, 3), profit_factor=round(profit_factor, 3),
            max_drawdown=round(max_dd, 3), total_trades=n,
            years=round(years, 1), trades_per_year=round(n / max(years, 1), 1),
            avg_pnl_pct=round(float(pnls_arr.mean()), 5), pnls=pnls,
        )


# ── Signal library ───────────────────────────────────────────────────────────

def make_nq_leads_es(threshold: float = 0.003) -> Callable:
    """Hypothesis #1: NQ leads ES. Position in ES today = sign of NQ's PRIOR-day return
    (when its magnitude clears `threshold`). Tests whether NQ's lead is tradeable in ES."""
    def nq_leads_es(frames: dict):
        import numpy as _np
        nq = frames["NQ=F"]["Close"].astype(float).pct_change()
        lead = nq.shift(1)                                       # NQ yesterday → ES today
        sig = _np.sign(lead)
        sig[lead.abs() < threshold] = 0.0
        return sig.fillna(0.0)
    nq_leads_es.__name__ = f"nq_leads_es(thr={threshold})"
    return nq_leads_es


SIGNALS = {"nq_leads_es": make_nq_leads_es}
