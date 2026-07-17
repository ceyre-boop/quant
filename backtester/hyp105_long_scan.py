"""HYP-105 — long-side momentum on parabolic gappers. In-sample scan (Phase 2)
and holdout runner (Phase 4). Reuses engine.run + the cached 234 gapper events.

70/30 date split: in-sample < 2026-04-08, holdout >=. Long side, sizing 2%.
Grid: entry x exit(fixed clock / +duration / trailing) x hard_stop.
Sign-flip permutation p per config, Bonferroni over grid size.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine
from . import daily_engine as de
from ._gapper_events import load_events, build_cache

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/scan_results"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "2026-04-08"
SIZING = 0.02
SEED = 42
N_PERM = 500

ENTRIES = ["09:30", "09:31", "09:32", "09:33", "09:35", "09:40", "09:45", "10:00"]
EXITS = ["+10", "+20", "+30", "+60", "10:30", "11:00",
         "trail5", "trail10", "trail15"]
STOPS = [0.05, 0.10, 0.15, 0.20, 0.25]


def _cfg(entry, exit_spec, stop):
    c = dict(entry_time=entry, direction="long", sizing_pct=SIZING,
             stop_pct=stop, locate_required=False, slippage=0.005)
    if exit_spec.startswith("trail"):
        c["exit_time"] = "15:45"           # time cap; trailing does the work
        c["trail_pct"] = int(exit_spec[5:]) / 100.0
    else:
        c["exit_time"] = exit_spec
    return c


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


def _event_rets(ev, cfg, cache):
    res = engine.run(ev, cfg, data_cache=cache, write_audit=False)
    return [r["net_pct"] for r in res["records"] if r.get("trade_taken")]


def run_scan(hold=False):
    ev = load_events()
    cache = build_cache(ev)
    ins = ev[ev.date < SPLIT].reset_index(drop=True)
    out = ev[ev.date >= SPLIT].reset_index(drop=True)
    use = out if hold else ins
    grid = list(itertools.product(ENTRIES, EXITS, STOPS))
    rows = []
    for i, (entry, exit_spec, stop) in enumerate(grid):
        cfg = _cfg(entry, exit_spec, stop)
        rets = _event_rets(use, cfg, cache)
        if len(rets) < 8:
            continue
        m = de._metrics(rets, 1.0, SIZING)
        r = np.array(rets)
        rows.append(dict(entry=entry, exit=exit_spec, stop=stop,
                         median_ret=round(float(np.median(r)), 5),
                         annual=m["annual"], sharpe=m["sharpe"],
                         max_dd=m["max_dd"], win=m["win"], n=m["n"],
                         _rets=json.dumps([round(x, 5) for x in rets])))
    df = pd.DataFrame(rows)
    return df, len(grid)


def main():
    df, n_grid = run_scan(hold=False)
    praw = [_perm_p(json.loads(r["_rets"]), r["sharpe"], SEED + i)
            for i, r in df.iterrows()]
    df["p_raw"] = praw
    df["p_bonferroni"] = (df["p_raw"] * n_grid).clip(upper=1.0)
    df["candidate"] = df["p_bonferroni"] < 0.05
    df.drop(columns=["_rets"]).to_parquet(OUT / "hyp105_long_insample.parquet")
    df2 = df.sort_values("sharpe", ascending=False)
    print(f"in-sample configs: {len(df)}/{n_grid}  candidates(FWER<.05): "
          f"{int(df['candidate'].sum())}")
    print("\nTOP 12 by Sharpe (in-sample):")
    print(df2[["entry", "exit", "stop", "annual", "sharpe", "median_ret",
               "win", "n", "p_raw", "p_bonferroni", "candidate"]].head(12).to_string())
    best = df2.iloc[0]
    print(f"\nbest config: entry={best['entry']} exit={best['exit']} "
          f"stop={best['stop']} sharpe={best['sharpe']}")


if __name__ == "__main__":
    main()
