"""HYP-077 substrate: v015 daily portfolio series + the pre-registered reconcile guard.

The frozen HYP-077 doc requires: "the v015 decade replay used for drawdown scoring must
reproduce the canonical weighted portfolio Sharpe first" (0.6886 ± 0.01). We rebuild the
per-pair Sharpes from data/proof/backtest_trades_v015_2015_2024.csv with EXACTLY the
canonical arithmetic (forex_backtester._compute_stats: rets_i = log(1+pnl_pct_i), std
ddof=0 + 1e-9, annualized by sqrt(n_trades / (n_bars/252))), feed them through
sovereign.reporting.equity_curve.weighted_portfolio_sharpe, and SystemExit BEFORE any
artifact or ledger write if the number is off. The equity JSON's recorded
stats.portfolio_sharpe_weighted is cross-checked too.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign.reporting.equity_curve import weighted_portfolio_sharpe

ROOT = Path(__file__).resolve().parents[3]
TRADES_CSV = ROOT / "data" / "proof" / "backtest_trades_v015_2015_2024.csv"
EQUITY_JSON = ROOT / "data" / "proof" / "backtest_equity_v015_2015_2024.json"
WINDOW = ("2015-01-01", "2024-12-31")


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_date", "exit_date"])
    df["pair"] = df["pair"].str.replace("=X", "", regex=False)
    return df.sort_values(["pair", "exit_date"]).reset_index(drop=True)


def canonical_pair_sharpe(pnls: list[float], n_bars: int) -> float:
    """Mirrors forex_backtester._compute_stats exactly."""
    n = len(pnls)
    if n <= 1:
        return 0.0
    years = n_bars / 252.0
    equity = np.cumprod([1 + p for p in pnls])
    returns = np.diff(np.log(equity), prepend=0)
    ann = np.sqrt(max(n, 1) / max(years, 1e-9))
    return float((np.mean(returns) / (np.std(returns) + 1e-9)) * ann)


def reconcile_guard(trades: pd.DataFrame, bars_by_pair: dict[str, pd.Series],
                    target: float = 0.6886, tol: float = 0.01) -> float:
    """Recompute the weighted portfolio Sharpe; SystemExit if it misses the canonical number."""
    pairs = []
    for pair, grp in trades.groupby("pair"):
        closes = bars_by_pair.get(pair)
        if closes is None:
            raise SystemExit(f"RECONCILE GUARD: no spot series for {pair}")
        n_bars = int(((closes.index >= WINDOW[0]) & (closes.index <= WINDOW[1])).sum())
        pairs.append((canonical_pair_sharpe(list(grp["pnl_pct"]), n_bars), len(grp)))
    got = weighted_portfolio_sharpe(pairs)
    import json
    recorded = json.loads(EQUITY_JSON.read_text())["stats"]["portfolio_sharpe_weighted"]
    if abs(got - target) > tol:
        raise SystemExit(
            f"RECONCILE GUARD FAILED: recomputed weighted portfolio Sharpe {got} != {target}±{tol} "
            f"(equity JSON records {recorded}). The v015 replay does not reproduce the canonical "
            f"number — HALTING before any artifact/ledger write (HYP-077 reconcile_guard).")
    return got


def daily_portfolio_equity(trades: pd.DataFrame, trading_index: pd.DatetimeIndex) -> pd.Series:
    """Daily equity: each trade's pnl_pct lands on its exit day; flat days = 0 return."""
    day_ret = pd.Series(0.0, index=trading_index)
    for _, t in trades.iterrows():
        d = pd.Timestamp(t["exit_date"]).normalize()
        pos = trading_index.searchsorted(d)
        if pos >= len(trading_index):
            continue
        day = trading_index[min(pos, len(trading_index) - 1)]
        day_ret.loc[day] = (1 + day_ret.loc[day]) * (1 + float(t["pnl_pct"])) - 1
    return (1 + day_ret).cumprod()


def open_positions_on(trades: pd.DataFrame, d: date) -> dict[str, int]:
    """pair -> direction for v015 positions open on date d (entry <= d < exit)."""
    ts = pd.Timestamp(d)
    live = trades[(trades["entry_date"] <= ts) & (trades["exit_date"] > ts)]
    return {row["pair"]: int(row["direction"]) for _, row in live.iterrows()}


def fwd_max_drawdown(equity: pd.Series, t0: pd.Timestamp, h: int) -> float | None:
    """Max drawdown of the equity path within (t0, t0+h] (<=0; None if window incomplete)."""
    idx = equity.index
    pos = idx.get_indexer([t0])[0]
    if pos < 0 or pos + h >= len(idx):
        return None
    window = equity.iloc[pos:pos + h + 1]
    dd = window / window.cummax() - 1.0
    return float(dd.min())


def crowding_composite(cot: pd.DataFrame, trades: pd.DataFrame,
                       funded_pairs: list[str]) -> tuple[list[date], list[float], list[dict]]:
    """COT-only interim composite per weekly publish_date (operator-authorized deviation:
    the locked composite's rr25 term is omitted — no options data exists yet).

    aligned_p = net_pct_1y if the open v015 position is long base else (1 - net_pct_1y);
    pairs with no open v015 trade that week are excluded; composite = mean over included.
    """
    dates, values, mapping = [], [], []
    weekly = cot.pivot_table(index="publish_date", columns="pair", values="net_pct_1y")
    for d, row in weekly.sort_index().iterrows():
        open_pos = open_positions_on(trades, d)
        comps, used = [], {}
        for pair in funded_pairs:
            v = row.get(pair)
            if pair not in open_pos or v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            aligned = float(v) if open_pos[pair] == 1 else 1.0 - float(v)
            comps.append(aligned)
            used[pair] = {"direction": open_pos[pair], "net_pct_1y": float(v), "aligned": aligned}
        if comps:
            dates.append(d if isinstance(d, date) else pd.Timestamp(d).date())
            values.append(float(np.mean(comps)))
            mapping.append({"date": str(d)[:10], "pairs": used})
    return dates, values, mapping
