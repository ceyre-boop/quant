"""Funnel chain (TICK-022 P4): challenge phases -> funded months -> fees -> program EV.

Modeling assumptions (all printed on the verdict table):
  - Attempts are i.i.d. (false under regime shift — the stamp/caveat travels with the row).
  - Phase outcomes chain independently: P(funded) = prod(P(phase_i)).
  - Funded months are i.i.d. via a reset-to-initial payout policy: at month end the
    trader withdraws max(equity - initial, 0) * split and equity resets to initial
    (common practice; simplification documented).
  - Subscription fees accrue on mean attempt calendar-months; INCOMPLETE attempts are
    charged up to the sim cap (conservative for no-time-limit firms).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from research.prop_funnel.feeds import TradePool
from research.prop_funnel.rulesets import DrawdownModel, FirmSpec, PhaseSpec
from research.prop_funnel.simulate import (
    OUTCOME_BUST, OUTCOME_INCOMPLETE, OUTCOME_PASS, OUTCOME_TIMEOUT, run_phase_mc,
)

CAL_PER_TRADING = 365.0 / 252.0


def run_funnel(rng: np.random.Generator, pool: TradePool, spec: FirmSpec, *,
               challenge_risk_mult: float = 1.0, funded_risk_mult: float = 1.0,
               n_attempts: int = 10_000, n_funded_sims: int = 10_000,
               monthly_income_target: float = 10_000.0,
               funded_months_horizon: int = 12,
               program_horizon_months: int = 24) -> dict:
    """Full funnel verdict for one strategy x firm x sizing policy."""
    if not pool.sufficient:
        return {
            "strategy": pool.name, "firm": spec.name, "stamp": pool.stamp.value,
            "verdict": "INSUFFICIENT_DATA",
            "note": f"pool n={pool.n} < 30 — refusing to Monte-Carlo; {pool.caveat}",
        }

    kappa_stress = spec.drawdown_model is DrawdownModel.INTRADAY_TRAILING
    risk_c = pool.base_risk_pct * challenge_risk_mult
    risk_f = pool.base_risk_pct * funded_risk_mult

    # ── challenge phases ────────────────────────────────────────────────────
    phase_rows = []
    p_funded = 1.0
    mean_attempt_tdays = 0.0
    for phase in spec.phases:
        st = run_phase_mc(rng, pool, spec, phase, risk_c, n_attempts,
                          kappa_stress=kappa_stress)
        p = st.rate(OUTCOME_PASS)
        dt = st.days_to_pass_pctiles()
        phase_rows.append({
            "phase": phase.name, "p_pass": round(p, 4),
            "p_bust": round(st.rate(OUTCOME_BUST), 4),
            "p_timeout": round(st.rate(OUTCOME_TIMEOUT), 4),
            "p_incomplete": round(st.rate(OUTCOME_INCOMPLETE), 4),
            "tdays_to_pass_med": dt["p50"], "tdays_to_pass_p10": dt["p10"],
            "tdays_to_pass_p90": dt["p90"],
            "cal_days_to_pass_med": (round(dt["p50"] * CAL_PER_TRADING, 1)
                                     if dt["p50"] is not None else None),
        })
        p_funded *= p
        mean_attempt_tdays += float(np.mean(st.event_day + 1))

    e_attempts = (1.0 / p_funded) if p_funded > 0 else math.inf
    attempt_months = mean_attempt_tdays * CAL_PER_TRADING / 30.0

    # ── fees to funded ──────────────────────────────────────────────────────
    fees = spec.fees or {}
    per_attempt_fee = float(fees.get("challenge_fee_usd") or 0.0)
    per_attempt_fee += float(fees.get("monthly_subscription_usd") or 0.0) * max(attempt_months, 1.0)
    fees_to_funded = (e_attempts * per_attempt_fee if math.isfinite(e_attempts) else math.inf)
    fees_to_funded += float(fees.get("pa_activation_usd") or 0.0)
    refund = float(fees.get("challenge_fee_usd") or 0.0) if fees.get("refunded_at_first_payout") else 0.0

    # ── funded stage: one 21-day month, i.i.d. via reset-to-initial payouts ─
    split = float((spec.funded or {}).get("profit_split", 0.8))
    month_phase = PhaseSpec("FUNDED_MONTH", float("inf"), 10 ** 9)
    month_spec_days = 21
    funded_spec = spec
    st_m = run_phase_mc(rng, pool, funded_spec, month_phase, risk_f, n_funded_sims,
                        kappa_stress=kappa_stress,
                        trades_per_day=pool.trades_per_day)
    # cap the month at 21 days: any bust with event_day < 21 counts; survivors' equity at day 20 EOD
    busted_in_month = (st_m.outcome == OUTCOME_BUST) & (st_m.event_day < month_spec_days)
    p_surv_month = 1.0 - float(np.mean(busted_in_month))
    eod20 = st_m.eod_equity[:, month_spec_days - 1]
    surv = ~busted_in_month
    payout = np.maximum(eod20 - spec.account_size, 0.0) * split
    payout_surv = payout[surv]
    p_10k_month = float(np.mean(surv & (payout >= monthly_income_target)))
    e_payout_uncond = float(np.mean(np.where(surv, payout, 0.0)))
    e_funded_months = (1.0 / (1.0 - p_surv_month)) if p_surv_month < 1.0 else math.inf

    p_surv_12 = p_surv_month ** funded_months_horizon
    p_10k_every_month_12 = p_10k_month ** funded_months_horizon

    # ── program EV per month over the horizon (renewal-cycle approximation) ─
    acq_months = e_attempts * max(attempt_months, 0.25) if math.isfinite(e_attempts) else math.inf
    funded_months = min(e_funded_months, program_horizon_months)
    cycle_months = acq_months + funded_months
    cycle_income = funded_months * e_payout_uncond - (
        fees_to_funded - (refund if e_payout_uncond > 0 else 0.0))
    ev_per_month = (cycle_income / cycle_months
                    if math.isfinite(cycle_months) and cycle_months > 0 else -per_attempt_fee)

    return {
        "strategy": pool.name, "firm": spec.name, "stamp": pool.stamp.value,
        "caveat": pool.caveat,
        "pool_n": pool.n or None, "pool_kind": pool.kind,
        "pool_sharpe_ann": (round(pool.sharpe_ann(), 3) if pool.sharpe_ann() is not None else None),
        "trades_per_day": round(pool.trades_per_day, 4),
        "challenge_risk_pct": round(risk_c, 6), "funded_risk_pct": round(risk_f, 6),
        "challenge_risk_mult": challenge_risk_mult, "funded_risk_mult": funded_risk_mult,
        "phases": phase_rows,
        "p_funded": round(p_funded, 5),
        "p_pass_100_consecutive": round(p_funded ** 100, 6),
        "e_attempts_to_funded": (round(e_attempts, 2) if math.isfinite(e_attempts) else None),
        "attempt_months_mean": round(attempt_months, 2),
        "fees_to_funded_usd": (round(fees_to_funded, 0) if math.isfinite(fees_to_funded) else None),
        "pricing_flag": (fees.get("pricing") or "UNVERIFIED_PRICING"),
        "funded": {
            "p_survive_month": round(p_surv_month, 4),
            "p_survive_12mo": round(p_surv_12, 4),
            "e_months_funded": (round(e_funded_months, 1) if math.isfinite(e_funded_months) else None),
            "e_payout_per_month_usd": round(e_payout_uncond, 0),
            "payout_month_p50_usd": (round(float(np.median(payout_surv)), 0) if payout_surv.size else None),
            "p_month_ge_target": round(p_10k_month, 4),
            "p_target_every_month_12": round(p_10k_every_month_12, 6),
            "monthly_income_target_usd": monthly_income_target,
        },
        "program_ev_per_month_usd": (round(ev_per_month, 0) if math.isfinite(ev_per_month) else None),
        "assumptions": ("iid attempts & months (reset-to-initial payouts); independence across phases; "
                        "subscription fees on mean attempt duration; INCOMPLETE charged to sim cap"),
    }
