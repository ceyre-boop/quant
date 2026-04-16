"""
Task 1 — Add ES=F and NQ=F to the universe sweep.

Runs the same 4 strategies on S&P 500 and Nasdaq 100 futures (2020-2024),
appends results to logs/universe_backtest_results.json, and prints a
dedicated futures table.

Run: python3 scripts/add_futures.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from universe_sweep import (
    STRATEGIES,
    _rolling_hurst,
    _atr_series,
    _build_spy_weekly,
    LOGS,
    DEFAULT_STOP_MULT, DEFAULT_TP_RR, DEFAULT_ATR_PERIOD,
    DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, INITIAL_CAPITAL,
    _hurst_label,
)

FUTURES = ["ES=F", "NQ=F"]
START   = "2020-01-01"
END     = "2024-12-31"


def _load_future(ticker: str, spy_weekly_map: dict) -> dict | None:
    """Download one futures ticker and prepare arrays."""
    raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(ticker, axis=1, level=1)
    raw.columns = [str(c).lower() for c in raw.columns]

    df = raw[["open", "high", "low", "close", "volume"]].dropna()
    if len(df) < 100:
        print(f"  [warn] {ticker}: insufficient data ({len(df)} bars)")
        return None

    closes = df["close"].to_numpy(dtype=np.float64)
    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].fillna(1e6).to_numpy(dtype=np.float64)
    dates  = df.index

    hurst   = _rolling_hurst(closes)
    atr_arr = _atr_series(highs.astype(np.float32),
                          lows.astype(np.float32),
                          closes.astype(np.float32))
    atr_pct = (atr_arr / (closes.astype(np.float32) + 1e-8)).astype(np.float32)

    spy_weekly = np.zeros(len(dates), dtype=np.float32)
    for k, d in enumerate(dates):
        spy_weekly[k] = spy_weekly_map.get(str(d.date()), 0.0)

    return {
        "df":         df,
        "closes":     closes,
        "highs":      highs,
        "lows":       lows,
        "volume":     volume,
        "dates":      dates,
        "hurst":      hurst,
        "atr_pct":    atr_pct,
        "spy_weekly": spy_weekly,
    }


def _run_one(symbol: str, data: dict, strategy_name: str, sig_fn) -> dict | None:
    """Run a single strategy on a single asset. Returns summary dict or None."""
    from backtest.fast_engine import FastBacktestEngine, SweepParams

    closes = data["closes"]
    highs  = data["highs"]
    lows   = data["lows"]
    volume = data["volume"]
    hurst  = data["hurst"]
    dates  = data["dates"]
    atr_pct = data["atr_pct"]
    spy_w   = data["spy_weekly"]

    try:
        signals, confidence = sig_fn(closes, highs, lows, volume, hurst)
    except Exception as e:
        print(f"  [warn] signal error {symbol}/{strategy_name}: {e}")
        return None

    if (signals == 0).all():
        return None

    try:
        engine = FastBacktestEngine.from_signals(data["df"], signals, confidence)
        params = SweepParams(
            stop_atr_mult=DEFAULT_STOP_MULT,
            tp_rr=DEFAULT_TP_RR,
            atr_period=DEFAULT_ATR_PERIOD,
            commission_per_side=DEFAULT_COMMISSION,
            slippage_pct=DEFAULT_SLIPPAGE,
            initial_capital=INITIAL_CAPITAL,
        )
        res = engine.run_detailed(params)
    except Exception as e:
        print(f"  [warn] backtest error {symbol}/{strategy_name}: {e}")
        return None

    tc = res["trade_count"]
    if tc == 0:
        return None

    entry_idx = res["entry_idx"]
    pnl_arr   = res["pnl"]
    pnl_r     = res["pnl_r"]

    # Best regime
    regime_pnl: dict = {}
    for i in range(tc):
        ei = entry_idx[i]
        lbl = _hurst_label(float(hurst[ei])) if ei < len(hurst) else "NEUTRAL"
        regime_pnl[lbl] = regime_pnl.get(lbl, 0.0) + float(pnl_arr[i])
    best_regime = max(regime_pnl, key=regime_pnl.get) if regime_pnl else "NEUTRAL"

    # Best/worst year
    year_pnl: dict = {}
    for i in range(tc):
        ei = entry_idx[i]
        if ei < len(dates):
            yr = int(dates[ei].year)
            year_pnl.setdefault(yr, []).append(float(pnl_arr[i]))

    year_sharpe = {}
    for yr, pnls in year_pnl.items():
        arr = np.array(pnls)
        std = arr.std()
        year_sharpe[yr] = float((arr.mean() / (std + 1e-8)) * np.sqrt(252)) if std > 0 else 0.0

    best_year  = max(year_sharpe, key=year_sharpe.get) if year_sharpe else None
    worst_year = min(year_sharpe, key=year_sharpe.get) if year_sharpe else None

    n_years        = (dates[-1].year - dates[0].year + 1) if len(dates) > 1 else 1
    trades_per_yr  = tc / max(n_years, 1)

    pnl_s = pnl_arr.astype(np.float64)
    sharpe = 0.0
    if len(pnl_s) > 1 and pnl_s.std() > 0:
        sharpe = float((pnl_s.mean() / pnl_s.std()) * np.sqrt(252))

    return {
        "strategy":         strategy_name,
        "asset":            symbol,
        "sector":           "FUTURES",
        "total_pnl":        round(float(res["total_pnl"]), 2),
        "total_return_pct": round(float(res["total_return_pct"]), 3),
        "trade_count":      tc,
        "win_rate":         round(float(res["win_rate"]), 4),
        "profit_factor":    round(float(res["profit_factor"]), 4),
        "sharpe":           round(sharpe, 4),
        "max_drawdown_pct": round(float(res["max_drawdown_pct"]), 3),
        "avg_r":            round(float(np.mean(pnl_r)), 4),
        "trades_per_year":  round(trades_per_yr, 1),
        "best_regime":      best_regime,
        "best_year":        best_year,
        "worst_year":       worst_year,
    }


def main():
    t0 = time.perf_counter()

    # SPY baseline for market condition
    print("[Data] Fetching SPY weekly returns...")
    spy_raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw = spy_raw.xs("SPY", axis=1, level=1)
    spy_raw.columns = [str(c).lower() for c in spy_raw.columns]
    spy_weekly_map = _build_spy_weekly(spy_raw)

    # Download futures
    futures_data: dict = {}
    for ticker in FUTURES:
        print(f"[Data] Downloading {ticker}...")
        d = _load_future(ticker, spy_weekly_map)
        if d:
            futures_data[ticker] = d
            print(f"  {ticker}: {len(d['df'])} bars, ATR avg={d['atr_pct'].mean()*100:.2f}%")

    if not futures_data:
        print("[Error] No futures data loaded.")
        return

    # Warm JIT
    print("\n[JIT] Warming Numba cache...")
    from backtest.fast_engine import FastBacktestEngine
    sample = next(iter(futures_data.values()))
    dummy = np.zeros(len(sample["closes"]), dtype=np.int8)
    dummy[30] = 1
    eng = FastBacktestEngine.from_signals(
        sample["df"], dummy, np.ones(len(dummy), dtype=np.float32)
    )
    eng.warmup()

    # Run all combos
    futures_results = []
    print("\n[Sweep] Running 4 strategies on each futures contract...")
    for sym, data in futures_data.items():
        for strat_name, sig_fn in STRATEGIES:
            r = _run_one(sym, data, strat_name, sig_fn)
            if r:
                futures_results.append(r)

    if not futures_results:
        print("[Error] No results generated.")
        return

    # Print futures table
    print("\n" + "═" * 68)
    print(" FUTURES RESULTS — ES=F / NQ=F  (2020-2024)")
    print("═" * 68)
    header = f"{'Asset':<8} {'Strategy':<22} {'Sharpe':>7} {'WinRate':>8} {'Trades':>7} {'BestReg'}"
    print(header)
    print("-" * 68)
    for r in sorted(futures_results, key=lambda x: x["sharpe"], reverse=True):
        marker = "★" if r["sharpe"] > 1.0 else " "
        print(
            f"{marker}{r['asset']:<7} {r['strategy']:<22} "
            f"{r['sharpe']:>7.2f} {r['win_rate']:>8.1%} "
            f"{r['trade_count']:>7}  {r['best_regime']}"
        )
    print("═" * 68)

    # Best per future
    for sym in futures_data:
        best = max((r for r in futures_results if r["asset"] == sym),
                   key=lambda x: x["sharpe"], default=None)
        if best:
            print(f"FUTURES ADDED: {sym} best={best['strategy']} "
                  f"sharpe={best['sharpe']:.2f} wr={best['win_rate']:.1%}")

    # Append to universe_backtest_results.json
    out = LOGS / "universe_backtest_results.json"
    if out.exists():
        existing = json.loads(out.read_text())
        # Remove any previous futures entries to avoid duplicates
        existing["results"] = [
            r for r in existing.get("results", [])
            if r.get("sector") != "FUTURES"
        ]
        existing["results"].extend(futures_results)
        existing["results"].sort(key=lambda x: x.get("sharpe", 0), reverse=True)
        existing.setdefault("meta", {})["futures_added"] = [s for s in futures_data]
        out.write_text(json.dumps(existing, indent=2))
        print(f"\nAppended {len(futures_results)} results → {out}")
    else:
        print(f"\n[warn] {out} not found — run universe_sweep.py first")

    print(f"Total time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
