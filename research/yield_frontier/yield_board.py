"""Yield board — the single ranked table every miner writes rows into.

Rank metric: net %/day at stated capacity (pessimistic frictions).
Rows with n < 40 are recorded but excluded from ranking (posthoc min-n convention).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._lib import OUT, STAMP

MIN_N_RANK = 40


def row(universe: str, family: str, config: str, rets: np.ndarray,
        events_per_day: float, capacity_usd: float, years: dict | None = None,
        net_adjust: float = 0.0, fill_prob: float = 1.0) -> dict:
    """Build one board row from per-event fractional returns (already net of
    per-event frictions except `net_adjust`, an additional per-event drag)."""
    rets = np.asarray(rets, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)
    if n == 0:
        return {"universe": universe, "family": family, "config": config,
                "n": 0, "stamp": STAMP}
    net = rets - net_adjust
    gross_day = float(rets.mean()) * events_per_day
    net_day = float(net.mean()) * events_per_day * fill_prob
    return {
        "universe": universe, "family": family, "config": config, "n": n,
        "mean_pct": round(float(rets.mean()), 5),
        "median_pct": round(float(np.median(rets)), 5),
        "events_per_day": round(events_per_day, 3),
        "gross_pct_day": round(gross_day, 5),
        "net_pct_day": round(net_day, 5),
        "tail_p5": round(float(np.percentile(net, 5)), 5),
        "tail_p1": round(float(np.percentile(net, 1)), 5),
        "ruin_frac": round(float((net <= -0.25).mean()), 5),
        "capacity_usd": int(capacity_usd),
        "per_year": {k: round(v, 5) for k, v in (years or {}).items()},
        "stamp": STAMP,
    }


def write_session(session: str, rows: list[dict]) -> Path:
    d = OUT / session
    d.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: -(r.get("net_pct_day") or -9e9))
    (d / "board_rows.json").write_text(json.dumps(
        {"stamp": STAMP, "session": session, "rows": rows}, indent=2))
    return d / "board_rows.json"


CONTEXT_ROWS = [
    # settled families, for scale — sourced from ledger / CLAUDE.md live-state
    {"universe": "forex", "family": "carry v015", "config": "live 4-pair",
     "status": "PROVEN (OOS Sharpe 1.25, regime-fragile)", "net_pct_day": 0.0002},
    {"universe": "equities", "family": "HYP-092 CONT/EX read", "config": "card checklist",
     "status": "NOT_SIGNIFICANT sealed (p=0.594)", "net_pct_day": None},
    {"universe": "futures", "family": "ES/NQ bias engine", "config": "5-input premarket",
     "status": "KILLED (p=0.57)", "net_pct_day": None},
    {"universe": "equities", "family": "overnight QQQ", "config": "long-short",
     "status": "VALID standalone (net Sharpe 0.574) / REJECTED as diversifier",
     "net_pct_day": 0.0002},
    {"universe": "options", "family": "VRP-001 (25pt wings)", "config": "iron condor",
     "status": "DEAD by sizing arithmetic ($333k floor)", "net_pct_day": None},
    {"universe": "forex", "family": "ICT patterns", "config": "live gates",
     "status": "UNPROVEN (perm p=0.52; live 3W/24L)", "net_pct_day": None},
]
