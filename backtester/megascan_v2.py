"""Megascan v2 — exhaustive intraday search across 5 families with real compute.

Families:
  A gap reversal (minute, curated high-vol equities)
  B opening-range breakout (minute, liquid ETFs/megacaps)
  C leveraged-decay reversion (daily)
  D crypto intraday (hourly)
  E biotech binary events (daily, XBI)

Bias-free fills (gap-through stop = fill at breaching bar open, never trigger;
entry-bar spread cost). Reports RAW top-50 (uncorrected) AND FWER top-20
(Bonferroni over all distinct signal hypotheses). Permutation (sign-flip, 200x)
only on raw-Sharpe>1.5 survivors. Real 12-month holdout reserved (dirty only
here). Progress line every 60s.

Run: PYTHONPATH=~/quant python3 -m backtester.megascan_v2
"""
import itertools
import json
import multiprocessing as _mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import daily_engine as de
from . import data as _data

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/scan_results"
OUT.mkdir(parents=True, exist_ok=True)
CRYPTO = REPO / "data/cache/crypto_hourly"
_CTX = _mp.get_context("fork" if os.name == "posix" else "spawn")

HOLDOUT_START = "2025-07-17"
DIRTY_HI = HOLDOUT_START
SIZING = 0.10
COST = 0.0005
SLIP = 0.005
N_PERM = 200
SEED = 42

FAMB = "SPY QQQ IWM GLD TLT AAPL TSLA NVDA AMZN META MSFT AMD COIN MSTR".split()
FAMC = "UVXY SQQQ TQQQ SOXS SOXL LABD LABU ARKK KOLD BOIL".split()


# ---------- minute fill helpers (bias-free) ----------
def _first_idx(bars, hhmm):
    idx = np.where(bars["time"].to_numpy() >= hhmm)[0]
    return int(idx[0]) if len(idx) else None


def _intraday_trade(bars, entry_i, direction, stop_pct, exit_i):
    o = bars["open"].to_numpy(); h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy(); c = bars["close"].to_numpy()
    entry = o[entry_i]
    if entry <= 0:
        return None
    spread = (h[entry_i] - lo[entry_i]) / entry * 0.5
    exitp = c[exit_i] if exit_i > entry_i else c[-1]
    if direction == "short":
        trig = entry * (1 + stop_pct)
        fill = None
        for j in range(entry_i + 1, len(bars)):
            if j > exit_i:
                break
            if h[j] >= trig:
                fill = max(o[j], trig) if o[j] >= trig else trig
                break
        px = fill if fill is not None else exitp
        gross = (entry - px) / entry
    else:
        trig = entry * (1 - stop_pct)
        fill = None
        for j in range(entry_i + 1, len(bars)):
            if j > exit_i:
                break
            if lo[j] <= trig:
                fill = min(o[j], trig) if o[j] <= trig else trig
                break
        px = fill if fill is not None else exitp
        gross = (px - entry) / entry
    return gross - spread - SLIP


# ---------- Family A: gap reversal ----------
def _famA_ticker(args):
    ticker, gapmags, grid = args
    rows = []
    # memoise bars per gap day
    cache = {}
    for d in gapmags:
        bars = _data.get_minute_bars(ticker, d)
        if bars is None or len(bars) < 30 or d >= DIRTY_HI:
            continue
        cache[d] = bars
    for gp, direction, entry, stop, exitt in grid:
        rets, dts = [], []
        for d, bars in cache.items():
            if gapmags[d] < gp / 100.0:      # actual gap magnitude sub-filter
                continue
            ei = _first_idx(bars, entry)
            xi = _first_idx(bars, exitt)
            if ei is None or ei == 0 or xi is None or xi <= ei:
                continue
            r = _intraday_trade(bars, ei, direction, stop, xi)
            if r is not None:
                rets.append(r); dts.append(d)
        if len(rets) >= 8:
            m = de._metrics(rets, 2.0, SIZING)
            rows.append(dict(family="A_gap", scope="asset", ticker=ticker,
                             direction=direction, entry=entry, stop=stop,
                             exit=exitt, gap_pct=gp, **m,
                             _rets=json.dumps([round(x, 5) for x in rets])))
    return rows


# ---------- Family B: opening-range breakout ----------
def _famB_ticker(args):
    ticker, grid, days = args
    rows = []
    daycache = {}
    for d in days:
        b = _data.get_minute_bars(ticker, d)
        if b is not None and len(b) > 60 and d < DIRTY_HI:
            daycache[d] = b
    for orb_min, direction, stop, exitt in grid:
        rets = []
        for d, bars in daycache.items():
            t = bars["time"].to_numpy()
            o = bars["open"].to_numpy(); h = bars["high"].to_numpy()
            lo = bars["low"].to_numpy()
            # opening range = first orb_min bars from 09:30
            oidx = np.where(t >= "09:30")[0]
            if len(oidx) < orb_min + 2:
                continue
            rng = slice(oidx[0], oidx[0] + orb_min)
            hi_r, lo_r = h[rng].max(), lo[rng].min()
            start = oidx[0] + orb_min
            entry_i = None
            for j in range(start, len(bars)):
                if direction == "long" and h[j] >= hi_r:
                    entry_i = j; break
                if direction == "short" and lo[j] <= lo_r:
                    entry_i = j; break
            if entry_i is None:
                continue
            xi = _first_idx(bars, exitt)
            if xi is None or xi <= entry_i:
                xi = len(bars) - 1
            r = _intraday_trade(bars, entry_i, direction, stop, xi)
            if r is not None:
                rets.append(r)
        if len(rets) >= 8:
            m = de._metrics(rets, 2.0, SIZING)
            rows.append(dict(family="B_orb", scope="asset", ticker=ticker,
                             direction=direction, orb_min=orb_min, stop=stop,
                             exit=exitt, **m,
                             _rets=json.dumps([round(x, 5) for x in rets])))
    return rows


# ---------- Families C/D/E via daily_engine ----------
def _daily_family(args):
    fam_tag, ticker, loader, grids = args
    df = loader(ticker)
    if df is None or len(df) < 200:
        return []
    rows = []
    for fam, cfg in grids:
        bt = de.backtest_daily(df, fam, cfg, "0000", DIRTY_HI)
        rets, dts = bt["rets"], bt["dates"]
        if len(rets) >= 8:
            yrs = max((pd.to_datetime(max(dts)) - pd.to_datetime(min(dts))).days
                      / 365.25, 0.5)
            m = de._metrics(rets, yrs, SIZING)
            rows.append(dict(family=fam_tag, scope="asset", ticker=ticker,
                             sig=fam, **{k: v for k, v in cfg.items()}, **m,
                             _rets=json.dumps([round(x, 5) for x in rets])))
    return rows


def _load_daily(t):
    return de.load_daily(t)


def _load_crypto(t):
    p = CRYPTO / f"{t}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _perm_p(rets, obs_sharpe, seed):
    r = np.array(rets)
    if len(r) < 8 or r.std() == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(N_PERM):
        sr = r * rng.choice([-1, 1], size=len(r))
        sh = sr.mean() / sr.std() * np.sqrt(len(sr)) if sr.std() > 0 else 0.0
        if sh >= obs_sharpe:
            ge += 1
    return (ge + 1) / (N_PERM + 1)


def build_tasks():
    tasks = []
    # A
    gapmags = json.loads((REPO / "data/cache/famA_gapmags.json").read_text()) \
        if (REPO / "data/cache/famA_gapmags.json").exists() else {}
    gridA = list(itertools.product(
        [10, 20, 30, 50], ["short", "long"],
        ["09:35", "09:45", "10:00", "10:15", "10:30", "11:00"],
        [0.05, 0.10, 0.15, 0.20, 0.30, 0.40], ["11:00", "12:00", "14:00", "15:45"]))
    for t, mags in gapmags.items():
        tasks.append(("A", (t, mags, gridA)))
    # B
    days_all = sorted({p.stem.split("_")[-1] for p in
                       (REPO / "data/cache/minute_bars").glob("*.parquet")})
    gridB = list(itertools.product(
        [1, 3, 5, 10, 15, 30], ["long", "short"],
        [0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
        ["10:30", "11:00", "12:00", "14:00", "15:45"]))
    for t in FAMB:
        tdays = sorted({p.stem.rsplit("_", 1)[1] for p in
                        (REPO / "data/cache/minute_bars").glob(f"{t}_*.parquet")})
        if tdays:
            tasks.append(("B", (t, gridB, tdays)))
    # C leveraged decay: oversold-dip long + up-spike short (mean reversion)
    gridC = []
    for thr, hold, stop in itertools.product([0.10, 0.15, 0.20, 0.30],
                                             [1, 2, 3, 5], [0.10, 0.15, 0.20, 0.30]):
        gridC.append(("dip", dict(look=3, thr=thr, trade_dir="long",
                                  hold_days=hold, stop_pct=stop)))
        gridC.append(("gap", dict(gap_dir="up", trade_dir="short", thr=thr,
                                  hold_days=hold, stop_pct=stop)))
    for t in FAMC:
        if (REPO / f"data/cache/daily_universe/{t}.parquet").exists():
            tasks.append(("CDE", ("C_lev", t, _load_daily, gridC)))
    # D crypto hourly
    gridD = []
    for fam, params in (("gap", dict(gap_dir="down", trade_dir="short")),
                        ("gap", dict(gap_dir="up", trade_dir="short")),
                        ("dip", dict(trade_dir="long")),
                        ("breakout", dict())):
        for thr, hold, stop in itertools.product([0.03, 0.05, 0.08],
                                                 [1, 2, 4, 8], [0.03, 0.05, 0.10]):
            cfg = dict(params)
            cfg.update(hold_days=hold, stop_pct=stop)
            if fam == "gap":
                cfg["thr"] = thr
            elif fam == "dip":
                cfg.update(look=4, thr=thr)
            else:
                cfg.update(look=12, direction="long")
            gridD.append((fam, cfg))
    for t in ["BTC-USD", "ETH-USD"]:
        if (CRYPTO / f"{t}.parquet").exists():
            tasks.append(("CDE", ("D_crypto", t, _load_crypto, gridD)))
    # E biotech binary
    xbi = json.loads((REPO / "data/cache/xbi_list.json").read_text()) \
        if (REPO / "data/cache/xbi_list.json").exists() else []
    gridE = []
    for gap_dir, td, thr, hold, stop in itertools.product(
            ["up", "down"], ["short", "long"], [0.20, 0.30, 0.50],
            [1, 2, 3, 5], [0.10, 0.15, 0.20, 0.30]):
        gridE.append(("gap", dict(gap_dir=gap_dir, trade_dir=td, thr=thr,
                                  hold_days=hold, stop_pct=stop)))
    for t in xbi:
        if (REPO / f"data/cache/daily_universe/{t}.parquet").exists():
            tasks.append(("CDE", ("E_biotech", t, _load_daily, gridE)))
    return tasks


def _dispatch(task):
    kind, payload = task
    if kind == "A":
        return _famA_ticker(payload)
    if kind == "B":
        return _famB_ticker(payload)
    return _daily_family(payload)


def main():
    t0 = time.time()
    tasks = build_tasks()
    print(f"tasks: {len(tasks)} ticker-family units", flush=True)
    cores = max(1, (os.cpu_count() or 2) - 1)
    all_rows = []
    with _CTX.Pool(cores) as pool:
        last = time.time()
        for i, rows in enumerate(pool.imap_unordered(_dispatch, tasks)):
            all_rows.extend(rows)
            if time.time() - last >= 60:
                print(f"[{time.time()-t0:.0f}s] {i+1}/{len(tasks)} units, "
                      f"{len(all_rows)} configs so far", flush=True)
                last = time.time()

    df = pd.DataFrame(all_rows)
    n_tested = len(df)
    elapsed = time.time() - t0
    print(f"DONE: {n_tested} configs in {elapsed:.0f}s", flush=True)

    # RAW top 50 by sharpe (positive return, n>=~30/yr => n>=30 over dirty)
    raw = df[(df["annual"] > 0) & (df["n"] >= 30)].sort_values(
        "sharpe", ascending=False).head(50)

    # FWER on raw-Sharpe>1.5 survivors
    cand = df[(df["sharpe"] > 1.5) & (df["annual"] > 0) & (df["n"] >= 30)].copy()
    praw = []
    for _, r in cand.iterrows():
        praw.append(_perm_p(json.loads(r["_rets"]), r["sharpe"], SEED + hash(r["ticker"]) % 9973))
    cand["p_raw"] = praw
    cand["p_bonferroni"] = (cand["p_raw"] * n_tested).clip(upper=1.0)
    cand["candidate_fwer"] = cand["p_bonferroni"] < 0.05
    fwer = cand.sort_values("p_bonferroni").head(20)

    df.drop(columns=["_rets"]).to_parquet(OUT / "megascan_v2_20260717.parquet")
    cand.drop(columns=["_rets"]).to_parquet(OUT / "megascan_v2_candidates_20260717.parquet")
    summary = dict(n_tested=int(n_tested), elapsed_s=round(elapsed, 1),
                   n_raw_positive=int(len(raw)), n_sharpe_gt_1p5=int(len(cand)),
                   n_fwer_survivors=int(cand["candidate_fwer"].sum()),
                   families={k: int(v) for k, v in df["family"].value_counts().items()})
    (OUT / "megascan_v2_summary_20260717.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2))
    cols = ["family", "ticker", "annual", "sharpe", "max_dd", "n", "direction"]
    show = [c for c in cols if c in raw.columns]
    print("\n=== RAW TOP 25 (uncorrected) ===")
    print(raw[show].head(25).to_string())
    print("\n=== FWER TOP 15 (Bonferroni) ===")
    fcols = show + ["p_raw", "p_bonferroni", "candidate_fwer"]
    print(fwer[[c for c in fcols if c in fwer.columns]].head(15).to_string())


if __name__ == "__main__":
    main()
