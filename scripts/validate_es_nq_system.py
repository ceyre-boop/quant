#!/usr/bin/env python3
"""ES/NQ staged validation harness — the gauntlet.

Usage: python3 scripts/validate_es_nq_system.py --stage {1,2,3,4} [--perms N] [--seed N]

Each stage: computes its statistic, runs its pre-registered permutation null,
prints a BOTH-SIDES report, appends a ledger entry to data/agent/hypothesis_ledger.json,
writes machine results to data/research/es_nq_validation.json, and STOPS.
Stage N refuses to run unless Stage N-1 has recorded a verdict.
Stage 4 (holdout) refuses to run twice (sentinel data/research/.es_nq_holdout_done).

Pre-registered designs: data/research/es_nq_preregistration.json (incl. Amendment A1).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.es_nq import data_store                       # noqa: E402
from sovereign.es_nq.backtest import (                       # noqa: E402
    _bar_minutes, _hhmm_minutes, run_backtest, simulate_entry_at,
)
from sovereign.es_nq.config import es_nq_params              # noqa: E402
from sovereign.es_nq.daily_bias_engine import build_feature_table  # noqa: E402
from sovereign.es_nq.session_sizing import SessionLadder     # noqa: E402

VALIDATION_PATH = ROOT / "data" / "research" / "es_nq_validation.json"
LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"
HOLDOUT_SENTINEL = ROOT / "data" / "research" / ".es_nq_holdout_done"
CAL_PATH = ROOT / "data" / "es_nq" / "econ_calendar_2018_2026.json"

STAGE_IDS = {1: "ESNQ-BIAS-01", 2: "ESNQ-GATE-02", 3: "ESNQ-SIZE-03", 4: "ESNQ-HOLDOUT-04"}


# ---------------------------------------------------------------- shared I/O

def load_validation() -> dict:
    if VALIDATION_PATH.exists():
        return json.loads(VALIDATION_PATH.read_text())
    return {}


def save_validation(v: dict) -> None:
    VALIDATION_PATH.write_text(json.dumps(v, indent=1, default=str))


def append_ledger(entry: dict) -> None:
    ledger = json.loads(LEDGER_PATH.read_text())
    if any(e.get("id") == entry["id"] for e in ledger):
        raise SystemExit(f"FATAL: ledger already has {entry['id']} — stages run once. "
                         "If this is intentional, remove the entry manually with a "
                         "logged rationale.")
    ledger.append(entry)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=1))


def require_prior_stage(stage: int, v: dict) -> None:
    if stage == 1:
        return
    prior = v.get(f"stage{stage - 1}")
    if not prior or "verdict" not in prior:
        raise SystemExit(f"FATAL: stage {stage - 1} has no recorded verdict — run it first.")


def load_universe(p: dict):
    daily = data_store.load_daily()
    bars5 = data_store.load_5min()
    aux = data_store.load_aux_daily()
    cal = json.loads(CAL_PATH.read_text())
    return daily, bars5, aux, cal


def now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- nulls

def label_permutation_pvalue(calls: np.ndarray, outcomes: np.ndarray,
                             perms: int, seed: int) -> tuple[float, np.ndarray]:
    """PRIMARY Stage-1 null: permute realized outcome labels.
    Returns (p-value, null accuracy distribution)."""
    rng = np.random.RandomState(seed)
    real_acc = float((calls == outcomes).mean())
    null_acc = np.empty(perms)
    for i in range(perms):
        null_acc[i] = float((calls == rng.permutation(outcomes)).mean())
    return float((null_acc >= real_acc).mean()), null_acc


def circular_shift_pvalue(calls: np.ndarray, outcomes: np.ndarray,
                          perms: int, seed: int) -> float:
    """SECONDARY (reported-only) null: circular shifts preserve autocorrelation."""
    rng = np.random.RandomState(seed)
    real_acc = float((calls == outcomes).mean())
    n = len(outcomes)
    shift_acc = np.array([(calls == np.roll(outcomes, k)).mean()
                          for k in rng.randint(1, n, size=min(perms, 10000))])
    return float((shift_acc >= real_acc).mean())


# ---------------------------------------------------------------- stage 1

def stage1(perms: int, seed: int) -> None:
    p = es_nq_params()
    oos_start, oos_end = p["splits"]["oos"]
    is_start, is_end = p["splits"]["is_period"]
    daily, bars5, aux, cal = load_universe(p)

    ft_oos = build_feature_table(daily, aux, cal, oos_start, oos_end, p)
    ft_is = build_feature_table(daily, aux, cal, is_start, is_end, p)

    def accuracy(ft):
        called = ft[(ft["direction"] != "NEUTRAL") & ft["direction_real"].notna()]
        if len(called) == 0:
            raise SystemExit("FATAL: zero non-NEUTRAL calls — bias engine inert")
        return called, float((called["direction"] == called["direction_real"]).mean())

    called, real_acc = accuracy(ft_oos)
    is_called, is_acc = accuracy(ft_is)
    calls = called["direction"].values
    outcomes = called["direction_real"].values
    p_primary, null_acc = label_permutation_pvalue(calls, outcomes, perms, seed)
    p_shift = circular_shift_pvalue(calls, outcomes, perms, seed)

    # BOTH SIDES
    def cohort(mask, ft):
        sub = ft[mask]
        sub_called = sub[(sub["direction"] != "NEUTRAL") & sub["direction_real"].notna()]
        return {
            "n": int(len(sub)),
            "n_called": int(len(sub_called)),
            "accuracy": (float((sub_called["direction"] == sub_called["direction_real"]).mean())
                         if len(sub_called) else None),
            "mean_move_pct": float(sub["move_pct"].mean()) if len(sub) else None,
        }

    up_mask = ft_oos["direction"] == "UP"
    down_mask = ft_oos["direction"] == "DOWN"
    neutral_mask = ft_oos["direction"] == "NEUTRAL"
    report = {
        "real_accuracy_oos": real_acc, "n_calls_oos": int(len(called)),
        "n_sessions_oos": int(len(ft_oos)),
        "p_primary_label_permutation": p_primary,
        "p_secondary_circular_shift": p_shift,
        "null_acc_mean": float(null_acc.mean()), "null_acc_p95": float(np.percentile(null_acc, 95)),
        "is_reference": {"accuracy": is_acc, "n_calls": int(len(is_called)),
                         "label": "IN-SAMPLE 2018-2020 — reference only"},
        "both_sides": {
            "up_calls": cohort(up_mask, ft_oos),
            "down_calls": cohort(down_mask, ft_oos),
            "neutral_abstained": cohort(neutral_mask, ft_oos),
            "event_days": cohort(ft_oos["event_day"], ft_oos),
            "non_event_days": cohort(~ft_oos["event_day"], ft_oos),
        },
        "secondary_outcome_flat_excluded": _flat_variant(ft_oos),
        "amendment_a1_disclosure": (
            "Backtest calendar input contributes 0 direction (no historical surprise "
            "tone exists); event days apply x0.75 confidence. Live uses real tone — "
            "live confidence will differ on event days."),
    }
    passed = real_acc > p["validation"]["min_oos_accuracy"] and \
        p_primary < p["validation"]["p_threshold"]
    verdict = "VALID_EDGE" if passed else "NOT_SIGNIFICANT"

    v = load_validation()
    v["stage1"] = {"verdict": verdict, "report": report, "perms": perms, "seed": seed,
                   "run_at": datetime.now(timezone.utc).isoformat()}
    save_validation(v)
    append_ledger({
        "id": STAGE_IDS[1], "created_date": now_date(), "confirmed_date": now_date(),
        "origin": "ES/NQ Daily Intelligence Engine build (operator brief 2026-06)",
        "title": "Pre-market 5-input bias engine predicts NQ RTH direction above chance (OOS 2021-2023)",
        "description": ("Pre-registered weights {overnight .30, calendar .25 (A1-degraded), "
                        "vix .20, hurst .15, intl .10}; NEUTRAL<0.40 confidence. Outcome = "
                        "sign(rth_close - rth_open). Null = outcome-label permutation, "
                        f"{perms} draws, seed {seed}."),
        "status": verdict,
        "result": report,
    })
    _print_stage_report(1, verdict, report)


def _flat_variant(ft) -> dict:
    sub = ft[(ft["direction"] != "NEUTRAL") & ft["direction_real"].notna()
             & ~ft["flat_secondary"]]
    return {
        "accuracy": float((sub["direction"] == sub["direction_real"]).mean()) if len(sub) else None,
        "n": int(len(sub)),
        "flat_fraction": float(ft["flat_secondary"].mean()),
        "label": "secondary outcome (|move|>=0.15%) — reported only, never gates",
    }


# ---------------------------------------------------------------- stage 2

def stage2(perms: int, seed: int) -> None:
    p = es_nq_params()
    v = load_validation()
    require_prior_stage(2, v)
    if v["stage1"]["verdict"] != "VALID_EDGE":
        raise SystemExit("FATAL: Stage 1 was not VALID_EDGE — the bias engine is dead; "
                         "Stage 2 has no validated bias to confirm.")
    oos_start, oos_end = p["splits"]["oos"]
    daily, bars5, aux, cal = load_universe(p)
    ft = build_feature_table(daily, aux, cal, oos_start, oos_end, p)
    inst = p["meta"]["trade_instrument"]

    gate_sessions = run_backtest(oos_start, oos_end, "bias_gate", params=p, daily=daily,
                                 bars5_all=bars5, feature_table=ft, instrument=inst)
    base_sessions = run_backtest(oos_start, oos_end, "bias_only", params=p, daily=daily,
                                 bars5_all=bars5, feature_table=ft, instrument=inst)
    gate_trades = [t for s in gate_sessions for t in s["trades"]]
    base_trades = [t for s in base_sessions for t in s["trades"]]
    if not gate_trades:
        raise SystemExit("FATAL: structure gate produced zero trades on OOS")
    gate_r = np.array([t["r_net"] for t in gate_trades])
    base_r = np.array([t["r_net"] for t in base_trades])
    real_improvement = float(gate_r.mean() - base_r.mean())

    # PERMUTATION (entry-selection isolation): per gated session, precompute the
    # null R of entering at EVERY eligible bar (09:35–12:00) with each real
    # trade's stop distance, then sample.
    rng = np.random.RandomState(seed)
    et_dates = bars5.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
    start_min, end_min = _hhmm_minutes("09:35"), _hhmm_minutes("12:00")
    null_pools: list[np.ndarray] = []      # one pool per real trade
    for s in gate_sessions:
        if not s["trades"]:
            continue
        sb = bars5[et_dates == s["session_date"]]
        minutes = _bar_minutes(sb)
        eligible = np.where((minutes >= start_min) & (minutes < end_min))[0]
        if len(eligible) == 0:
            continue
        for t in s["trades"]:
            pool = np.array([
                simulate_entry_at(sb, int(b), s["bias"]["direction"],
                                  t["stop_points"], inst, p)["r_net"]
                for b in eligible])
            null_pools.append(pool)
    n_trades = len(null_pools)
    null_means = np.empty(perms)
    for i in range(perms):
        null_means[i] = float(np.mean([pool[rng.randint(len(pool))] for pool in null_pools]))
    # Null improvement vs the SAME bias_only baseline
    null_improvement = null_means - base_r.mean()
    p_value = float((null_improvement >= real_improvement).mean())

    # BOTH SIDES: what happened on sessions the gate did NOT fire
    fired_dates = {s["session_date"] for s in gate_sessions if s["trades"]}
    base_by_date = {s["session_date"]: s for s in base_sessions}
    cohorts = {"gate_fired": [], "gate_silent": []}
    for date, s in base_by_date.items():
        if not s["trades"]:
            continue
        key = "gate_fired" if date in fired_dates else "gate_silent"
        cohorts[key].append(s["trades"][0]["r_net"])
    report = {
        "gate_mean_r": float(gate_r.mean()), "gate_n_trades": int(len(gate_r)),
        "gate_total_r": float(gate_r.sum()),
        "bias_only_mean_r": float(base_r.mean()), "bias_only_n_trades": int(len(base_r)),
        "real_improvement_r_per_trade": real_improvement,
        "p_entry_selection_permutation": p_value,
        "null_mean_improvement": float(null_improvement.mean()),
        "both_sides": {
            "bias_only_on_gate_fired_sessions": _arr_stats(cohorts["gate_fired"]),
            "bias_only_on_gate_silent_sessions": _arr_stats(cohorts["gate_silent"]),
        },
        "legacy_cost_variant": _legacy_cost_mean_r(p, daily, bars5, ft, inst, oos_start, oos_end),
        "n_null_pools": n_trades,
    }
    passed = real_improvement > 0 and p_value < p["validation"]["p_threshold"]
    verdict = "VALID_EDGE" if passed else "NOT_SIGNIFICANT"
    v["stage2"] = {"verdict": verdict, "gate_kept": passed, "report": report,
                   "perms": perms, "seed": seed,
                   "run_at": datetime.now(timezone.utc).isoformat()}
    save_validation(v)
    append_ledger({
        "id": STAGE_IDS[2], "created_date": now_date(), "confirmed_date": now_date(),
        "origin": "ES/NQ Daily Intelligence Engine build",
        "title": "AMD structure gate (sweep+VWAP confirm) improves net R/trade over bias-only entries",
        "description": ("Gate trades vs 09:35 bias_only baseline on OOS 2021-2023. Null = "
                        "random same-direction entries from the eligible 09:35-12:00 pool "
                        "through identical exit machinery, stop distances from real trades. "
                        f"{perms} draws, seed {seed}. ICT timing alone failed p=0.52 on forex "
                        "— this tests the gate as a CONFIRMATION LAYER only."),
        "status": verdict, "result": report,
    })
    _print_stage_report(2, verdict, report)
    if not passed:
        print("\n>>> GATE DROPPED: Stage 3 will size the bias_only stream instead.\n")


def _arr_stats(xs) -> dict:
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return {"n": 0}
    return {"n": int(len(xs)), "mean_r": float(xs.mean()), "sum_r": float(xs.sum()),
            "win_rate": float((xs > 0).mean())}


def _legacy_cost_mean_r(p, daily, bars5, ft, inst, start, end) -> dict:
    p2 = copy.deepcopy(p)
    p2["costs"]["slippage_ticks_entry"] = p["costs"]["legacy_slippage_ticks_per_side"]
    p2["costs"]["slippage_ticks_stop"] = p["costs"]["legacy_slippage_ticks_per_side"]
    p2["costs"]["commission_per_side_usd"] = p["costs"]["legacy_commission_per_round_turn_usd"] / 2
    sess = run_backtest(start, end, "bias_gate", params=p2, daily=daily,
                        bars5_all=bars5, feature_table=ft, instrument=inst)
    r = np.array([t["r_net"] for s in sess for t in s["trades"]])
    return {"mean_r": float(r.mean()) if len(r) else None, "n": int(len(r)),
            "label": "harsher sandbox cost model (1 tick/side + $0.74 RT) — informational"}


# ---------------------------------------------------------------- stage 3

def stage3(perms: int, seed: int) -> None:
    p = es_nq_params()
    v = load_validation()
    require_prior_stage(3, v)
    gate_kept = bool(v["stage2"].get("gate_kept"))
    mode = "full" if gate_kept else "bias_only"
    oos_start, oos_end = p["splits"]["oos"]
    daily, bars5, aux, cal = load_universe(p)
    ft = build_feature_table(daily, aux, cal, oos_start, oos_end, p)
    inst = p["meta"]["trade_instrument"]
    account = p["sizing"]["account_base_usd"]

    # Trades held fixed; only the sizing layer differs.
    if gate_kept:
        adaptive_sessions = run_backtest(oos_start, oos_end, "full", params=p, daily=daily,
                                         bars5_all=bars5, feature_table=ft, instrument=inst)
        flat_sessions = run_backtest(oos_start, oos_end, "bias_gate", params=p, daily=daily,
                                     bars5_all=bars5, feature_table=ft, instrument=inst)
    else:
        # Gate dropped: ladder the bias_only single-trade stream (ladder degenerates
        # to probe-only — adaptive == flat by construction; report says so honestly).
        adaptive_sessions = run_backtest(oos_start, oos_end, "bias_only", params=p, daily=daily,
                                         bars5_all=bars5, feature_table=ft, instrument=inst)
        flat_sessions = adaptive_sessions

    def session_pnls(sessions):
        return np.array([s["session_usd_total"] / account for s in sessions if s["trades"]])

    pnl_a, pnl_f = session_pnls(adaptive_sessions), session_pnls(flat_sessions)
    real_diff = _ann_sharpe(pnl_a) - _ann_sharpe(pnl_f)

    # Null: within-session trade-order permutation re-laddered.
    rng = np.random.RandomState(seed)
    trade_sets = [[(t["r_net"], t["stop_points"], t["usd_net_per_contract"])
                   for t in s["trades"]] for s in adaptive_sessions if s["trades"]]
    null_diffs = np.empty(perms)
    for i in range(perms):
        adaptive_pnl, flat_pnl = [], []
        for trades in trade_sets:
            order = rng.permutation(len(trades))
            adaptive_pnl.append(_ladder_usd([trades[j] for j in order], p, inst, flat=None) / account)
            flat_pnl.append(_ladder_usd([trades[j] for j in order], p, inst, flat=0.005) / account)
        null_diffs[i] = _ann_sharpe(np.array(adaptive_pnl)) - _ann_sharpe(np.array(flat_pnl))
    p_value = float((null_diffs >= real_diff).mean())

    boot = np.empty(2000)
    n = len(pnl_a)
    for i in range(2000):
        idx = rng.randint(0, n, n)
        boot[i] = _ann_sharpe(pnl_a[idx]) - _ann_sharpe(pnl_f[idx])
    report = {
        "adaptive_sharpe": _ann_sharpe(pnl_a), "flat_sharpe": _ann_sharpe(pnl_f),
        "real_sharpe_diff": real_diff, "p_order_permutation": p_value,
        "bootstrap_ci_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "sized_stream": mode, "gate_kept": gate_kept,
        "both_sides": {
            "adaptive": _dd_stats(pnl_a, adaptive_sessions, account),
            "flat": _dd_stats(pnl_f, flat_sessions, account),
        },
        "n_traded_sessions": int(len(pnl_a)),
        "note": (None if gate_kept else
                 "gate dropped at Stage 2: bias_only stream has one trade/session — "
                 "ladder degenerates to probe-only; adaptive==flat by construction"),
    }
    passed = real_diff > 0 and p_value < p["validation"]["p_threshold"]
    verdict = "VALID_EDGE" if passed else "NOT_SIGNIFICANT"
    v["stage3"] = {"verdict": verdict, "adaptive_adopted": passed, "report": report,
                   "perms": perms, "seed": seed,
                   "run_at": datetime.now(timezone.utc).isoformat()}
    save_validation(v)
    append_ledger({
        "id": STAGE_IDS[3], "created_date": now_date(), "confirmed_date": now_date(),
        "origin": "ES/NQ Daily Intelligence Engine build",
        "title": "Adaptive 3-trade session ladder beats flat 0.5% sizing on session-level Sharpe",
        "description": ("Identical trades, sizing-only difference, OOS 2021-2023. Null = "
                        f"within-session trade-order permutation re-laddered, {perms} draws, "
                        f"seed {seed}, plus 2000-draw bootstrap CI."),
        "status": verdict, "result": report,
    })
    _print_stage_report(3, verdict, report)
    if not passed:
        print("\n>>> FLAT SIZING SHIPS: simpler is better when results are equal.\n")


def _ann_sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def _ladder_usd(trades, p, inst, flat) -> float:
    lad = SessionLadder(account_usd=p["sizing"]["account_base_usd"],
                        flat_risk_pct=flat, params=p)
    total = 0.0
    for r_net, stop_pts, usd_per_ct in trades:
        role = lad.next_role()
        if role is None:
            break
        n = lad.contracts(role, stop_pts, inst)
        usd = usd_per_ct * n
        lad.record(r_net, usd)
        total += usd
    return total


def _dd_stats(pnl: np.ndarray, sessions, account) -> dict:
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    cap = es_nq_params()["sizing"]["daily_loss_cap_pct"]
    return {
        "sharpe": _ann_sharpe(pnl), "max_dd_pct": dd,
        "worst_session_pct": float(pnl.min()) if len(pnl) else 0.0,
        "cap_hit_sessions": int(sum(1 for x in pnl if x <= -cap)),
        "mean_session_pct": float(pnl.mean()) if len(pnl) else 0.0,
    }


# ---------------------------------------------------------------- stage 4

def stage4(perms: int, seed: int) -> None:
    p = es_nq_params()
    v = load_validation()
    require_prior_stage(4, v)
    if HOLDOUT_SENTINEL.exists():
        raise SystemExit("FATAL: HOLDOUT ALREADY RUN (sentinel exists). The number you "
                         "have is the number. It does not get a second chance.")
    gate_kept = bool(v["stage2"].get("gate_kept"))
    adaptive = bool(v["stage3"].get("adaptive_adopted"))
    mode = ("full" if adaptive else "bias_gate") if gate_kept else "bias_only"
    h_start, h_end = p["splits"]["holdout"]
    daily, bars5, aux, cal = load_universe(p)
    ft = build_feature_table(daily, aux, cal, h_start, h_end, p)
    inst = p["meta"]["trade_instrument"]
    account = p["sizing"]["account_base_usd"]

    # Sentinel BEFORE results are examined — the run is spent either way.
    HOLDOUT_SENTINEL.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(), "mode": mode}))

    sessions = run_backtest(h_start, h_end, mode, params=p, daily=daily,
                            bars5_all=bars5, feature_table=ft, instrument=inst)
    called = ft[(ft["direction"] != "NEUTRAL") & ft["direction_real"].notna()]
    acc = float((called["direction"] == called["direction_real"]).mean()) if len(called) else None
    trades = [t for s in sessions for t in s["trades"]]
    r = np.array([t["r_net"] for t in trades])
    pnl = np.array([s["session_usd_total"] / account for s in sessions if s["trades"]])
    report = {
        "frozen_mode": mode, "bias_accuracy_holdout": acc, "n_calls": int(len(called)),
        "n_trades": int(len(r)), "mean_r": float(r.mean()) if len(r) else None,
        "total_r": float(r.sum()) if len(r) else None,
        "win_rate": float((r > 0).mean()) if len(r) else None,
        "sharpe": _ann_sharpe(pnl), **{"dd": _dd_stats(pnl, sessions, account)},
        "by_year": _by_year(sessions, account),
        "note": "RUN ONCE. This number is the honest answer.",
    }
    verdict = "VALID_EDGE" if (acc is not None and acc > 0.5 and _ann_sharpe(pnl) > 0) \
        else "NOT_SIGNIFICANT"
    v["stage4"] = {"verdict": verdict, "report": report,
                   "run_at": datetime.now(timezone.utc).isoformat()}
    save_validation(v)
    append_ledger({
        "id": STAGE_IDS[4], "created_date": now_date(), "confirmed_date": now_date(),
        "origin": "ES/NQ Daily Intelligence Engine build",
        "title": "ES/NQ full-system holdout 2024-2025 (run once)",
        "description": f"Configuration frozen by Stages 1-3 (mode={mode}) applied blind to 2024-2025.",
        "status": verdict, "result": report,
    })
    _print_stage_report(4, verdict, report)


def _by_year(sessions, account) -> dict:
    out = {}
    for s in sessions:
        if not s["trades"]:
            continue
        yr = s["session_date"][:4]
        out.setdefault(yr, []).append(s["session_usd_total"] / account)
    return {yr: {"n_sessions": len(xs), "sharpe": _ann_sharpe(np.array(xs)),
                 "total_pct": float(np.sum(xs))} for yr, xs in sorted(out.items())}


# ---------------------------------------------------------------- report

def _print_stage_report(stage: int, verdict: str, report: dict) -> None:
    print("\n" + "=" * 64)
    print(f"  STAGE {stage} — {STAGE_IDS[stage]} — VERDICT: {verdict}")
    print("=" * 64)
    print(json.dumps(report, indent=1, default=str))
    print("=" * 64)
    print(f"Ledger entry appended: {STAGE_IDS[stage]}. STOPPING — report to Colin "
          f"before invoking the next stage.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--perms", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    p = es_nq_params()
    perms = args.perms or p["validation"]["permutations"]
    seed = args.seed if args.seed is not None else p["validation"]["seed"]
    {1: stage1, 2: stage2, 3: stage3, 4: stage4}[args.stage](perms, seed)


if __name__ == "__main__":
    main()
