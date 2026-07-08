#!/usr/bin/env python3
"""scripts/research/exit_config_sweep.py — ExitConfig parameter sweep (READ-ONLY research).

Motivation (vault HYP-059 / REST-024): the trailing stop is a net drag (-60.8R across 411
trades), ~95% of it concentrated in 3-5 day exits; time exits are the edge engine. Strong prior
that loosening/disabling the trail and raising the hold floor wins. This sweeps the exit surface:

    stop_atr_mult  in [1.0, 1.5, 2.0, 2.5, 3.0]                       (5)
    trailing_atr_mult in [0.5, 1.0, 1.5, 2.0, 3.0, OFF(=0.0)]         (6)
    hold_limit     in [5, 10, 15, 20, 30, 45] days                    (6)   -> 180 uniform configs

over the 4 live pairs (EURUSD, GBPUSD, USDJPY, AUDUSD) on the 2015-2024 PROOF set.

Metric = the canonical prove.py number: per-pair ForexBacktester._compute_stats Sharpe (log-return
equity, annualised by n_bars/252), aggregated √n-weighted (equity_curve.weighted_portfolio_sharpe).
Baseline to beat = v015 (stop 2.0, per-pair trailing overrides, hold 60) = 0.6886 full decade.

It reuses the LIVE code paths verbatim (simulate_forex_trades -> _apply_costs -> _compute_stats);
only stop/trail/hold are overridden. It NEVER writes config, never commits, never touches live.
Signal direction is hold-independent (verified) and the VIX gate is hold-independent, so signals are
rebuilt once per (pair, hold) and gated with a cached SPY/VIX snapshot.

    python3 scripts/research/exit_config_sweep.py

Writes full results JSON to the path printed at the end.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.forex.forex_backtester import ForexBacktester            # noqa: E402
from sovereign.forex.fast_backtester import simulate_forex_trades       # noqa: E402
from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY  # noqa: E402
from sovereign.reporting.equity_curve import weighted_portfolio_sharpe  # noqa: E402

START, END = "2015-01-01", "2024-12-31"
PAIRS = list(ALL_PAIRS)                                    # EURUSD GBPUSD USDJPY AUDUSD
STOPS = [1.0, 1.5, 2.0, 2.5, 3.0]
TRAILS = [0.5, 1.0, 1.5, 2.0, 3.0, 0.0]                   # 0.0 == OFF
HOLDS = [5, 10, 15, 20, 30, 45]
BASE_STOP, BASE_HOLD = 2.0, 60
V015_TRAIL = {"GBPUSD=X": 2.0, "AUDUSD=X": 1.0, "EURUSD=X": 1.25, "USDJPY=X": 1.25}  # USDJPY -> default
TREND_YEARS = [2022, 2023]        # carry pays (documented walk-forward)
RANGE_YEARS = [2021, 2024]        # carry doesn't pay
OUT = Path("/private/tmp/claude-501/-Users-taboost-quant/f329c9e8-0ebe-4ac6-9d4e-499839e3e059/scratchpad/exit_sweep_results.json")


def trail_label(t: float) -> str:
    return "OFF" if t == 0.0 else f"{t:g}"


# ── canonical per-pair Sharpe (mirrors ForexBacktester._compute_stats exactly) ──────────────────
def pair_sharpe(pnls: np.ndarray, n_bars: int) -> float:
    n = len(pnls)
    if n <= 1:
        return 0.0
    years = n_bars / 252.0
    equity = np.cumprod(1.0 + pnls)
    returns = np.diff(np.log(equity), prepend=0.0)
    ann = np.sqrt(max(n, 1) / max(years, 1e-9))
    return float(np.mean(returns) / (np.std(returns) + 1e-9) * ann)


def portfolio(pnls_by_pair: dict, nbars_by_pair: dict) -> float:
    rows = [(pair_sharpe(p, nbars_by_pair[k]), len(p)) for k, p in pnls_by_pair.items() if len(p) > 1]
    return weighted_portfolio_sharpe(rows)


# ── setup: download once, cache signals per (pair, hold), cache VIX gate ─────────────────────────
def build_context():
    bt = ForexBacktester(start=START, end=END)
    dfs, atr, entry_years, nbars = {}, {}, {}, {}
    for p in PAIRS:
        df = bt._download_price(p)
        dfs[p] = df
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        atr[p] = bt._signals._compute_atr_pct(close, df)
        idx = pd.DatetimeIndex(df.index)
        entry_years[p] = idx.year.to_numpy()
        nbars[p] = len(df)

    # cache SPY/VIX for the gate (mirrors _apply_vix_regime_gate, downloaded once)
    spy = yf.download("SPY", start=START, end=END, progress=False)
    vix = yf.download("^VIX", start=START, end=END, progress=False)
    for d in (spy, vix):
        if hasattr(d.columns, "get_level_values"):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
    spy["sma200"] = spy["Close"].rolling(200).mean()
    spy["is_bull"] = spy["Close"] > spy["sma200"]

    def gate(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
        thr = bt.PAIR_VIX_GATES.get(pair)
        if thr is None:
            return frame
        frame = frame.copy()
        for date in frame[frame["signal"] != 0].index:
            try:
                if bool(spy["is_bull"].asof(date)) and float(vix["Close"].asof(date)) > thr:
                    frame.loc[date, "signal"] = 0.0
            except Exception:
                pass
        return frame

    # build gated signal frames per (pair, hold) for HOLDS + baseline hold 60
    frames = {}
    for p in PAIRS:
        cfg = PAIR_CONFIG[p]
        bc, qc = CB_TO_COUNTRY[cfg.base_central_bank], CB_TO_COUNTRY[cfg.quote_central_bank]
        for h in HOLDS + [BASE_HOLD]:
            sig = bt._get_pair_signals(df=dfs[p], base_country=bc, quote_country=qc, pair=p, hold_days=h)
            frames[(p, h)] = gate(sig, p) if p in bt.PAIR_VIX_GATES else sig
    return bt, dfs, atr, entry_years, nbars, frames


def sim(bt, dfs, atr, frames, pair, stop, trail, hold):
    """Byte-identical to ForexBacktester._simulate_trades with (stop, trail, hold) overridden.
    Returns the list of costed trades (pnl_pct post-cost, hold_days, direction, entry, ...)."""
    trades = simulate_forex_trades(
        dfs[pair], frames[(pair, hold)],
        stop_pct=bt.STOP_PCT, atr_series=atr[pair],
        stop_atr_mult=stop, trailing_atr_mult=trail,
        strict_mode=False, donchian_exit_days=bt.DONCHIAN_EXIT_DAYS,
        allow_pyramiding=False, max_pyramid_units=1,
        risk_pct=bt.MAX_RISK_PER_TRADE_PCT, max_risk_pct=bt.MAX_RISK_PER_TRADE_PCT,
        enable_cb_refresh=True,
    )
    return bt._apply_costs(trades, pair)


def pnls_year_slice(trades, entry_years_arr, years_set):
    """pnl_pct for trades whose entry year is in years_set (entry index inferred from entry_date)."""
    out = []
    for t in trades:
        y = pd.Timestamp(t["entry_date"]).year
        if y in years_set:
            out.append(t["pnl_pct"])
    return np.asarray(out, dtype=np.float64)


def bars_in_years(entry_years_arr, years_set):
    return int(np.isin(entry_years_arr, list(years_set)).sum())


# ── block bootstrap for the √n portfolio Sharpe ─────────────────────────────────────────────────
def block_boot(pnls_by_pair, nbars_by_pair, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    keys = [k for k in pnls_by_pair if len(pnls_by_pair[k]) > 1]
    boots = np.empty(B, dtype=np.float64)
    prepared = {}
    for k in keys:
        a = pnls_by_pair[k]
        n = len(a)
        L = max(1, int(round(np.sqrt(n))))
        prepared[k] = (a, n, L)
    for b in range(B):
        rows = []
        for k in keys:
            a, n, L = prepared[k]
            nb = int(np.ceil(n / L))
            starts = rng.integers(0, n, size=nb)
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n] % n
            rs = a[idx]
            rows.append((pair_sharpe(rs, nbars_by_pair[k]), n))
        boots[b] = weighted_portfolio_sharpe(rows)
    return boots


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns (reject_bool_array, qvalues) in original order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]          # monotone
    qvals = np.empty(m); qvals[order] = np.minimum(q, 1.0)
    # BH reject set
    thresh = alpha * np.arange(1, m + 1) / m
    passed = ranked <= thresh
    kmax = np.where(passed)[0].max() + 1 if passed.any() else 0
    reject = np.zeros(m, dtype=bool)
    if kmax > 0:
        reject[order[:kmax]] = True
    return reject, qvals


def signflip_perm(pnls_by_pair, nbars_by_pair, obs, B=1000, seed=11):
    rng = np.random.default_rng(seed)
    keys = [k for k in pnls_by_pair if len(pnls_by_pair[k]) > 1]
    ge = 0
    for _ in range(B):
        rows = []
        for k in keys:
            a = pnls_by_pair[k]
            signs = rng.choice([-1.0, 1.0], size=len(a))
            rows.append((pair_sharpe(a * signs, nbars_by_pair[k]), len(a)))
        if weighted_portfolio_sharpe(rows) >= obs:
            ge += 1
    return (ge + 1) / (B + 1)


def main():
    t0 = time.time()
    print("Building context (downloads + signal frames)…")
    bt, dfs, atr, entry_years, nbars, frames = build_context()
    print(f"  context ready in {time.time()-t0:.1f}s  pairs={PAIRS}")

    # ── v015 baseline (per-pair trailing, stop 2.0, hold 60) ────────────────────────────────────
    base_pnls = {}
    base_perpair = {}
    for p in PAIRS:
        tr = sim(bt, dfs, atr, frames, p, BASE_STOP, V015_TRAIL[p], BASE_HOLD)
        base_pnls[p] = np.asarray([t["pnl_pct"] for t in tr], dtype=np.float64)
        base_perpair[p] = {"sharpe": round(pair_sharpe(base_pnls[p], nbars[p]), 4), "n": len(tr)}
    base_sharpe = portfolio(base_pnls, nbars)
    base_trend = portfolio(
        {p: pnls_year_slice([t for t in sim(bt, dfs, atr, frames, p, BASE_STOP, V015_TRAIL[p], BASE_HOLD)], entry_years[p], set(TREND_YEARS)) for p in PAIRS},
        {p: bars_in_years(entry_years[p], set(TREND_YEARS)) for p in PAIRS})
    base_range = portfolio(
        {p: pnls_year_slice([t for t in sim(bt, dfs, atr, frames, p, BASE_STOP, V015_TRAIL[p], BASE_HOLD)], entry_years[p], set(RANGE_YEARS)) for p in PAIRS},
        {p: bars_in_years(entry_years[p], set(RANGE_YEARS)) for p in PAIRS})
    print(f"  v015 baseline full-decade √n Sharpe = {base_sharpe:.4f}  (prove.py=0.6886)")
    print(f"  v015 trending({TREND_YEARS}) = {base_trend:.3f}   ranging({RANGE_YEARS}) = {base_range:.3f}")

    # ── sweep 180 uniform configs ───────────────────────────────────────────────────────────────
    configs = list(product(STOPS, TRAILS, HOLDS))
    print(f"Sweeping {len(configs)} configs × {len(PAIRS)} pairs…")
    results = []
    # cache costed trades per (pair, stop, trail, hold) to avoid recompute in regime slices
    for i, (stop, trail, hold) in enumerate(configs):
        pnls_by_pair, trades_by_pair = {}, {}
        for p in PAIRS:
            tr = sim(bt, dfs, atr, frames, p, stop, trail, hold)
            trades_by_pair[p] = tr
            pnls_by_pair[p] = np.asarray([t["pnl_pct"] for t in tr], dtype=np.float64)
        full = portfolio(pnls_by_pair, nbars)
        # per-year
        by_year = {}
        for y in range(2015, 2025):
            yp = {p: pnls_year_slice(trades_by_pair[p], entry_years[p], {y}) for p in PAIRS}
            yb = {p: bars_in_years(entry_years[p], {y}) for p in PAIRS}
            by_year[y] = round(portfolio(yp, yb), 3)
        trend = portfolio({p: pnls_year_slice(trades_by_pair[p], entry_years[p], set(TREND_YEARS)) for p in PAIRS},
                          {p: bars_in_years(entry_years[p], set(TREND_YEARS)) for p in PAIRS})
        rng_ = portfolio({p: pnls_year_slice(trades_by_pair[p], entry_years[p], set(RANGE_YEARS)) for p in PAIRS},
                         {p: bars_in_years(entry_years[p], set(RANGE_YEARS)) for p in PAIRS})
        n_tot = int(sum(len(v) for v in pnls_by_pair.values()))
        boots = block_boot(pnls_by_pair, nbars, B=1000, seed=7)
        results.append({
            "stop": stop, "trail": trail, "trail_label": trail_label(trail), "hold": hold,
            "sharpe": round(full, 4), "n": n_tot,
            "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
            "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
            "p_vs_zero": round(float(np.mean(boots <= 0)), 4),
            "p_vs_base": round(float(np.mean(boots <= base_sharpe)), 4),
            "trend": round(trend, 3), "range": round(rng_, 3), "by_year": by_year,
            "beats_base_pt": bool(full > base_sharpe),
            "beats_both_regime": bool(trend >= base_trend and rng_ >= base_range),
        })
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(configs)} done ({time.time()-t0:.0f}s)")

    # ── FDR across 180 on p_vs_base (does this config beat v015 after multiplicity?) ────────────
    reject, qvals = bh_fdr([r["p_vs_base"] for r in results], alpha=0.05)
    for r, rej, q in zip(results, reject, qvals):
        r["fdr_beats_base"] = bool(rej)
        r["q_vs_base"] = round(float(q), 4)
    _, q0 = bh_fdr([r["p_vs_zero"] for r in results], alpha=0.05)
    for r, q in zip(results, q0):
        r["q_vs_zero"] = round(float(q), 4)

    results.sort(key=lambda r: -r["sharpe"])

    # ── gauntlet on top 5 by point Sharpe ───────────────────────────────────────────────────────
    print("Gauntlet on top 5 (permutation 2000, walk-forward, decay, both-regime)…")
    top5 = results[:5]
    gauntlet = []
    for r in top5:
        stop, trail, hold = r["stop"], r["trail"], r["hold"]
        pnls_by_pair, trades_by_pair = {}, {}
        for p in PAIRS:
            tr = sim(bt, dfs, atr, frames, p, stop, trail, hold)
            trades_by_pair[p] = tr
            pnls_by_pair[p] = np.asarray([t["pnl_pct"] for t in tr], dtype=np.float64)
        obs = portfolio(pnls_by_pair, nbars)
        perm_p = signflip_perm(pnls_by_pair, nbars, obs, B=2000, seed=13)
        # decay: IS 2015-2022 vs OOS 2023-2024
        is_years = set(range(2015, 2023)); oos_years = {2023, 2024}
        is_s = portfolio({p: pnls_year_slice(trades_by_pair[p], entry_years[p], is_years) for p in PAIRS},
                         {p: bars_in_years(entry_years[p], is_years) for p in PAIRS})
        oos_s = portfolio({p: pnls_year_slice(trades_by_pair[p], entry_years[p], oos_years) for p in PAIRS},
                          {p: bars_in_years(entry_years[p], oos_years) for p in PAIRS})
        decay = round(oos_s / is_s, 3) if is_s > 1e-9 else None
        pos_years = sum(1 for y in range(2015, 2025) if r["by_year"][y] > 0)
        gauntlet.append({
            **{k: r[k] for k in ("stop", "trail", "trail_label", "hold", "sharpe", "n",
                                 "ci_lo", "ci_hi", "trend", "range", "by_year",
                                 "beats_both_regime", "fdr_beats_base", "q_vs_base")},
            "perm_p": round(perm_p, 4),
            "is_2015_2022": round(is_s, 3), "oos_2023_2024": round(oos_s, 3),
            "decay_ratio": decay, "positive_years": pos_years,
            "pass_perm": bool(perm_p < 0.05),
            "pass_decay": bool(decay is not None and decay >= 0.50),
            "pass_both_regime": bool(r["beats_both_regime"]),
            "pass_beats_base": bool(r["sharpe"] > round(base_sharpe, 4)),
        })

    out = {
        "meta": {"start": START, "end": END, "pairs": PAIRS, "n_configs": len(configs),
                 "stops": STOPS, "trails": [trail_label(t) for t in TRAILS], "holds": HOLDS,
                 "metric": "per-pair _compute_stats Sharpe, √n-weighted (prove.py canonical)",
                 "runtime_s": round(time.time() - t0, 1)},
        "baseline_v015": {"sharpe": round(base_sharpe, 4), "prove_py": 0.6886,
                          "trend": round(base_trend, 3), "range": round(base_range, 3),
                          "per_pair": base_perpair,
                          "config": {"stop": BASE_STOP, "hold": BASE_HOLD, "trail": V015_TRAIL}},
        "results": results, "gauntlet_top5": gauntlet,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nDONE in {time.time()-t0:.0f}s → {OUT}")
    print(f"\nBaseline v015 √n Sharpe = {base_sharpe:.4f}")
    print("TOP 10 configs by full-decade √n Sharpe:")
    print(f"  {'stop':>4} {'trail':>5} {'hold':>4} | {'sharpe':>7} {'n':>4} {'[CI95]':>16} "
          f"{'trend':>6} {'range':>6} | {'>base':>5} {'both':>4} {'FDR':>4}")
    for r in results[:10]:
        print(f"  {r['stop']:>4} {r['trail_label']:>5} {r['hold']:>4} | {r['sharpe']:>7.4f} {r['n']:>4} "
              f"[{r['ci_lo']:>6.3f},{r['ci_hi']:>6.3f}] {r['trend']:>6.2f} {r['range']:>6.2f} | "
              f"{str(r['beats_base_pt']):>5} {str(r['beats_both_regime']):>4} {str(r['fdr_beats_base']):>4}")
    n_beat = sum(1 for r in results if r["beats_base_pt"])
    n_both = sum(1 for r in results if r["beats_both_regime"])
    n_fdr = sum(1 for r in results if r["fdr_beats_base"])
    print(f"\nconfigs beating v015 point: {n_beat}/180 | beating in BOTH regimes: {n_both}/180 | "
          f"FDR-significant vs base: {n_fdr}/180")
    return out


if __name__ == "__main__":
    main()
