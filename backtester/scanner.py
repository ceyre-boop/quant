"""Parallel strategy scanner with family-wise error control.

Performance architecture: the expensive part is the per-event minute-bar fill
scan. It depends only on (entry_time, stop_pct, exit_time, direction) — NOT on
sizing. So we memoise per-event net% for each distinct (entry,stop,exit,dir)
combo ONCE (parallel across cores), then every sizing variant is a cheap
vectorised re-aggregation. A grid that multiplies sizing out stays nearly free.

Family-wise correction (the research method's soul): every evaluated config is
counted in n_tested; each gets a permutation p-value (shuffle event->day
assignment, recompute Sharpe null, 100 perms); Bonferroni and Holm-Bonferroni
corrected p-values are both reported; candidate_flag = corrected p < 0.05.
Ranking is by corrected p, never raw Sharpe.

scan(events_df, param_grid, data_cache, n_jobs, max_configs) -> DataFrame.
"""
from __future__ import annotations

import itertools
import multiprocessing as _mp
import os

import numpy as np
import pandas as pd

from . import engine as _engine
from . import holdout_guard as _hg

_CTX = _mp.get_context("fork" if os.name == "posix" else "spawn")
N_PERM = 100
SEED = 42


def _fill_key(cfg):
    return (cfg["entry_time"], cfg["stop_pct"], cfg["exit_time"],
            cfg["direction"], bool(cfg.get("locate_required")))


def _event_nets(args):
    """Per-event net% + date for one (entry,stop,exit,dir) fill combo."""
    key, events_records, cache = args
    entry_time, stop_pct, exit_time, direction, locate_required = key
    cfg = {"entry_time": entry_time, "stop_pct": stop_pct,
           "exit_time": exit_time, "direction": direction, "sizing_pct": 1.0,
           "locate_required": locate_required, "on_missing_locate": "take"}
    ev = pd.DataFrame(events_records)
    # scan() already validated the event span once; re-checking per combo
    # would log thousands of duplicate access records.
    res = _engine.run(ev, cfg, data_cache=cache, write_audit=False,
                      check_holdout=False)
    out = []
    for r in res["records"]:
        if r.get("trade_taken"):
            out.append((r["date"], r["net_pct"]))
    return key, out


def _daily_from_nets(nets, sizing):
    daily = {}
    for d, net in nets:
        daily[d] = daily.get(d, 0.0) + net * sizing
    return daily


def _sharpe(day_vals):
    if len(day_vals) < 2 or np.std(day_vals) == 0:
        return 0.0
    return float(np.mean(day_vals) / np.std(day_vals) * np.sqrt(252))


def _perm_p(nets, sizing, obs_sharpe, rng):
    """Permutation p: shuffle event->day labels, recompute Sharpe null."""
    if not nets:
        return 1.0
    dates = [d for d, _ in nets]
    vals = np.array([n for _, n in nets]) * sizing
    uniq = list(dict.fromkeys(dates))
    ge = 0
    for _ in range(N_PERM):
        perm_dates = rng.permutation(dates)
        daily = {}
        for d, x in zip(perm_dates, vals):
            daily[d] = daily.get(d, 0.0) + x
        if _sharpe(list(daily.values())) >= obs_sharpe:
            ge += 1
    return (ge + 1) / (N_PERM + 1)


def _latin_hypercube(grid, n, rng):
    keys = list(grid)
    out = []
    for _ in range(n):
        out.append({k: grid[k][rng.integers(0, len(grid[k]))] for k in keys})
    # dedupe
    seen, uniq = set(), []
    for c in out:
        t = tuple(sorted(c.items()))
        if t not in seen:
            seen.add(t)
            uniq.append(c)
    return uniq


def scan(events_df, param_grid, data_cache=None, n_jobs=None,
         max_configs=1_000_000, check_holdout=True):
    """Enumerate/sample configs, memoise fills, correct family-wise.

    A parameter scan is the single most overfitting-prone operation in the
    stack, so the holdout span is validated here once, up front, before any
    config is evaluated.
    """
    if check_holdout and len(events_df) and "date" in events_df.columns:
        d = events_df["date"].astype(str)
        _hg.validate_date_range(d.min(), d.max(), context="scanner.scan",
                                dataset="gapper_intraday")
    grid = dict(param_grid)
    grid.setdefault("exit_time", ["15:45"])
    grid.setdefault("direction", ["short"])
    grid.setdefault("locate_required", [False])
    keys = list(grid)
    total = int(np.prod([len(grid[k]) for k in keys]))
    rng = np.random.default_rng(SEED)
    if total <= max_configs:
        configs = [dict(zip(keys, combo))
                   for combo in itertools.product(*[grid[k] for k in keys])]
    else:
        configs = _latin_hypercube(grid, max_configs, rng)
    n_tested = len(configs)

    # 1) memoise per-event nets for each distinct fill combo (parallel)
    fill_keys = sorted({_fill_key(c) for c in configs})
    ev_records = events_df.to_dict("records")
    cores = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    args = [(k, ev_records, data_cache) for k in fill_keys]
    if cores == 1 or len(args) == 1:
        fills = dict(_event_nets(a) for a in args)
    else:
        with _CTX.Pool(min(cores, len(args))) as pool:
            fills = dict(pool.map(_event_nets, args))

    # 2) cheap sizing/aggregation sweep + permutation test per config
    rows = []
    for c in configs:
        nets = fills[_fill_key(c)]
        sizing = c.get("sizing_pct", 0.02)
        daily = _daily_from_nets(nets, sizing)
        dv = [max(x, -1.0) for x in daily.values()]  # a day can't lose >100%
        sharpe = _sharpe(list(daily.values()))
        peak, mdd = 1.0, 0.0
        e = 1.0
        for x in dv:
            e = max(e * (1 + x), 0.0)                 # ruin is absorbing
            peak = max(peak, e)
            mdd = max(mdd, 1 - e / peak)
            if e == 0.0:
                break
        eq = e
        p_raw = _perm_p(nets, sizing, sharpe,
                        np.random.default_rng(SEED + hash(_fill_key(c)) % 9973))
        taken = len(nets)
        wins = sum(1 for _, n in nets if n * sizing > 0)
        rows.append({**{k: c[k] for k in keys},
                     "annual_return": round(eq - 1, 5), "sharpe": round(sharpe, 4),
                     "max_dd": round(mdd, 5), "n_trades": taken,
                     "win_rate": round(wins / taken, 4) if taken else 0.0,
                     "locate_fill_rate": round(taken / max(len(ev_records), 1), 4),
                     "p_raw": round(p_raw, 5)})

    df = pd.DataFrame(rows)
    # 3) family-wise correction
    df["p_bonferroni"] = (df["p_raw"] * n_tested).clip(upper=1.0)
    order = df["p_raw"].rank(method="first").astype(int)
    m = n_tested
    holm = (df["p_raw"] * (m - order + 1)).clip(upper=1.0)
    df["p_holm"] = holm
    df["candidate_flag"] = df["p_bonferroni"] < 0.05
    df["n_tested"] = n_tested
    df = df.sort_values(["p_bonferroni", "sharpe"],
                        ascending=[True, False]).reset_index(drop=True)
    return df
