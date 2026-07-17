"""Monte Carlo — block bootstrap, multiprocessing, bias-free.

Fixes the IID-bootstrap bias (audit finding #3): daily P&L is resampled in
5-day BLOCKS so weekly autocorrelation / loss-clustering survives. Independent
day draws understated tail risk; block resampling restores it.

run_mc(daily_pnl, n_paths, challenge_config, n_cores) -> dict.
Parallel across cores-1 workers; each worker runs a numpy-vectorised bundle of
paths (all paths step day-t together) for throughput.
"""
from __future__ import annotations

import multiprocessing as _mp
import os

import numpy as np

BLOCK = 5

# fork avoids re-importing __main__ (spawn breaks from heredocs / REPLs and is
# slow); numpy-only workers are fork-safe. Fall back to spawn off-posix.
_CTX = _mp.get_context("fork" if os.name == "posix" else "spawn")


def _worker(arg):
    (pnl, n_paths, block, horizon, pass_pct, bust_pct,
     time_limit, seed) = arg
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    if n == 0:
        return (0, 0, 0, np.ones(n_paths))
    n_blocks = int(np.ceil(horizon / block))
    max_start = max(n - block, 0)

    eq = np.ones(n_paths)
    done = np.zeros(n_paths, dtype=bool)
    n_pass = n_bust = 0
    passed_time = np.zeros(n_paths, dtype=bool)
    day = 0
    for _b in range(n_blocks):
        starts = rng.integers(0, max_start + 1, size=n_paths)
        for k in range(block):
            if day >= horizon:
                break
            step = pnl[(starts + k) % n]
            eq = np.where(done, eq, eq * (1 + step))
            hit_pass = (~done) & (eq >= 1 + pass_pct)
            hit_bust = (~done) & (eq <= 1 + bust_pct)
            n_pass += int(hit_pass.sum())
            n_bust += int(hit_bust.sum())
            done |= hit_pass | hit_bust
            day += 1
        if done.all():
            break
    return (n_pass, n_bust, int((~done).sum()), eq)


def run_mc(daily_pnl, n_paths=100_000, challenge_config=None, n_cores=None,
           seed=42):
    """Block-bootstrap prop-challenge MC.

    daily_pnl: list of per-event-day P&L fractions (already sized).
    challenge_config: {account_size, pass_pct, bust_pct, time_limit_days}
        time_limit_days None -> unlimited (capped at 10y for termination).
    """
    cfg = challenge_config or {}
    pass_pct = cfg.get("pass_pct", 0.08)
    bust_pct = cfg.get("bust_pct", -0.08)
    time_limit = cfg.get("time_limit_days")
    # event-day density: challenge days are calendar; signals hit a fraction.
    # daily_pnl is the per-event-day series; we replay it directly (each drawn
    # day is an event day). horizon in event-days ~ time_limit * density; when
    # unlimited, cap at 10y of trading days.
    horizon = time_limit if time_limit else 2520

    cores = n_cores or max(1, (os.cpu_count() or 2) - 1)
    per = int(np.ceil(n_paths / cores))
    args = [(daily_pnl, per, BLOCK, horizon, pass_pct, bust_pct, time_limit,
             seed + i) for i in range(cores)]
    if cores == 1:
        results = [_worker(args[0])]
    else:
        with _CTX.Pool(cores) as pool:
            results = pool.map(_worker, args)

    total = per * cores
    n_pass = sum(r[0] for r in results)
    n_bust = sum(r[1] for r in results)
    n_time = sum(r[2] for r in results)
    finals = np.concatenate([r[3] for r in results])
    pcts = {f"p{q}": round(float(np.percentile(finals, q)), 4)
            for q in (5, 25, 50, 75, 95)}
    p_pass = n_pass / total
    return {
        "n_paths": total, "block_size": BLOCK,
        "p_pass": round(p_pass, 4), "p_bust": round(n_bust / total, 4),
        "p_time": round(n_time / total, 4),
        "final_equity_percentiles": pcts,
        "expected_paths_to_pass": (round(1 / p_pass, 2) if p_pass > 0 else None),
        "challenge": {"pass_pct": pass_pct, "bust_pct": bust_pct,
                      "time_limit_days": time_limit},
    }
