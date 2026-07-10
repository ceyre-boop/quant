"""Sizing-policy grid + synthetic requirements frontier (TICK-022 P5).

optimize_sizing: {challenge_mult x funded_mult} grid per strategy x firm,
objective = 24-month program EV per month (run_funnel's sort key), with
constraint overlays reported alongside (P(month>=target), P(bust a funded month)).

frontier: Sharpe x daily-vol x trade-frequency grid of synthetic pools through a
firm's full funnel — the "what would any candidate strategy need to be" map.

Determinism: every cell gets its own SeedSequence spawned from (base_seed, cell
coordinates) so results are independent of execution order.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from research.prop_funnel import feeds
from research.prop_funnel.funnel import run_funnel
from research.prop_funnel.rulesets import FirmSpec

DEFAULT_MULTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FRONTIER_SHARPES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
FRONTIER_VOLS = (0.0025, 0.005, 0.01, 0.02)      # daily equity vol
FRONTIER_FREQS = (0.2, 0.5, 1.0, 2.0)            # trades per trading day


def _cell_rng(base_seed: int, *coords) -> np.random.Generator:
    ss = np.random.SeedSequence([base_seed, *[int(c * 1000) for c in coords]])
    return np.random.default_rng(ss)


def optimize_sizing(pool: feeds.TradePool, spec: FirmSpec, *, base_seed: int = 7,
                    mults: Sequence[float] = DEFAULT_MULTS,
                    n_attempts: int = 3_000, n_funded_sims: int = 3_000,
                    monthly_income_target: float = 10_000.0) -> dict:
    cells = []
    for i, cm in enumerate(mults):
        for j, fm in enumerate(mults):
            row = run_funnel(_cell_rng(base_seed, 1, i, j), pool, spec,
                             challenge_risk_mult=cm, funded_risk_mult=fm,
                             n_attempts=n_attempts, n_funded_sims=n_funded_sims,
                             monthly_income_target=monthly_income_target)
            cells.append({
                "challenge_mult": cm, "funded_mult": fm,
                "p_funded": row.get("p_funded"),
                "ev_per_month": row.get("program_ev_per_month_usd"),
                "p_month_ge_target": row.get("funded", {}).get("p_month_ge_target"),
                "p_survive_month": row.get("funded", {}).get("p_survive_month"),
            })
    ranked = sorted((c for c in cells if c["ev_per_month"] is not None),
                    key=lambda c: c["ev_per_month"], reverse=True)
    return {
        "strategy": pool.name, "firm": spec.name, "stamp": pool.stamp.value,
        "mults": list(mults), "cells": cells,
        "best": ranked[0] if ranked else None,
        "note": ("objective = program EV/month; the best cell typically sizes the CHALLENGE hot "
                 "and the FUNDED account cooler — verify constraint columns before acting"),
    }


def frontier(spec: FirmSpec, *, base_seed: int = 7,
             sharpes: Sequence[float] = FRONTIER_SHARPES,
             vols: Sequence[float] = FRONTIER_VOLS,
             freqs: Sequence[float] = FRONTIER_FREQS,
             n_attempts: int = 4_000, n_funded_sims: int = 4_000,
             monthly_income_target: float = 10_000.0) -> dict:
    cells = []
    for s in sharpes:
        for v in vols:
            for f in freqs:
                pool = feeds.synthetic_pool(sharpe_ann=s, trades_per_day=f)
                risk = feeds.synthetic_risk_pct(v, f)
                mult = risk / pool.base_risk_pct
                row = run_funnel(_cell_rng(base_seed, 2, s, v * 100, f), pool, spec,
                                 challenge_risk_mult=mult, funded_risk_mult=mult,
                                 n_attempts=n_attempts, n_funded_sims=n_funded_sims,
                                 monthly_income_target=monthly_income_target)
                p1 = row["phases"][0]
                cells.append({
                    "sharpe": s, "vol_daily": v, "trades_per_day": f,
                    "p_funded": row["p_funded"],
                    "p_pass_100": row["p_pass_100_consecutive"],
                    "p1_incomplete": p1["p_incomplete"],
                    "cal_days_to_pass_med": p1["cal_days_to_pass_med"],
                    "p_month_ge_target": row["funded"]["p_month_ge_target"],
                    "p_bust_funded_month": round(1.0 - row["funded"]["p_survive_month"], 4),
                    "ev_per_month": row["program_ev_per_month_usd"],
                })
    return {"firm": spec.name, "sharpes": list(sharpes), "vols": list(vols),
            "freqs": list(freqs), "cells": cells,
            "monthly_income_target_usd": monthly_income_target,
            "note": "SYNTHETIC requirements map — existence of any cell's strategy is NOT claimed"}
