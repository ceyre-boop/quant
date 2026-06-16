#!/usr/bin/env python3
"""VRP validation — the INVERTED GAUNTLET (research only, validate-don't-deploy).

This system has NO historical option chains, so the brief's iron-condor backtest is
blocked (strategy_simulator returns DATA_INSUFFICIENT — never faked). Instead we run the
cheapest falsifying test first: does the volatility risk premium (1) EXIST, and (2) stay
ORTHOGONAL to the carry edge in crisis — or re-couple like overnight-QQQ?

  Stage 1  existence      ^VIX/^VXN vs forward realized var (BTZ), both-sides, permutation
  Stage 2  orthogonality   causal harvest return vs DBV carry / v015 carry / overnight-QQQ
  Verdict  ladder          NOT_SIGNIFICANT | REJECTED_OOS | DATA_INSUFFICIENT | PARTIAL_CONFIRMATION

Read-only w.r.t. the live system: writes data/research/vrp_validation.json,
data/research/vrp_findings.md, and appends data/agent/hypothesis_ledger.json (idempotent
by id). VALID_EDGE is unreachable here by design.

Usage:  python3 scripts/validate_vrp.py [--perms 10000] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.research.vrp import data_loader as dl          # noqa: E402
from sovereign.research.vrp import validator as val           # noqa: E402
from sovereign.research.vrp import vrp_calculator as vc        # noqa: E402
from sovereign.research.vrp.strategy_simulator import iron_condor_simulate  # noqa: E402

OUT = ROOT / "data" / "research" / "vrp_validation.json"
FINDINGS = ROOT / "data" / "research" / "vrp_findings.md"
PREREG = ROOT / "data" / "research" / "vrp_preregistration.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
CITATION = "Coval & Shumway (2001); Bakshi & Kapadia (2003); Bollerslev, Tauchen & Zhou (2009)"


def _check_prereg():
    """Tripwire: the frozen pre-registration must exist and match code constants. If this
    fails, someone retuned a gate — that invalidates the verdict. Fix the code, not the file."""
    if not PREREG.exists():
        raise SystemExit(f"FATAL: pre-registration missing ({PREREG}). Freeze it before running.")
    p = json.loads(PREREG.read_text())
    assert p["stress_threshold_vix"] == val.STRESS_VIX, "stress threshold drifted from pre-registration"
    assert p["corr_gates"]["correlated"] == val.GATE_CORRELATED, "correlated gate drifted"
    assert p["corr_gates"]["crisis_max"] == val.GATE_CRISIS, "crisis gate drifted"
    return p


def _write_findings(payload: dict):
    s1s, s1q = payload["stage1_existence"]["SPY"], payload["stage1_existence"]["QQQ"]
    s2 = payload["stage2_orthogonality"]
    cv = s2["vs_carry"]
    lines = [
        "# VRP Findings — Volatility Risk Premium (SPY/QQQ)",
        "",
        f"*Generated {payload['generated_at']} · verdict **{payload['verdict']}** · research only, validate-don't-deploy*",
        "",
        "## Pre-registered hypothesis",
        "The SPY/QQQ volatility risk premium (implied vol > realized vol) is a real systematic premium. "
        "The decisive question is whether it is an **orthogonal** second edge to the live carry edge, or "
        "correlated **return-stacking** like overnight-QQQ.",
        "",
        f"Mechanism (academic, 30+ yrs): {CITATION}. Option buyers are net hedgers/speculators who pay premium "
        "for defined outcomes; sellers earn it for warehousing volatility risk. The premium is real and large "
        "in calm regimes and collapses/inverts in vol shocks (2008/2020/2022) — that is the cost of harvesting it.",
        "",
        "## Why this is the inverted gauntlet (not the brief's 4-stage order)",
        "The system has **no historical SPY/QQQ option chains** (yfinance = current only; `data/polygon_client.py` "
        "has no options endpoints). The brief forbids synthesizing option prices, so the iron-condor backtest is "
        "genuinely blocked. Per the operation's own lesson (`sovereign_core_verdict`: run the cheapest falsifying "
        "test first), we run the free orthogonality kill-gate **before** spending money on option data. If VRP "
        "re-couples with carry in crisis, it is return-stacking — same fate as overnight-QQQ — and no option data "
        "is worth procuring.",
        "",
        "## Stage 1 — does VRP exist? (BTZ forward IV−RV gap)",
        f"- **SPY/^VIX**: mean gap {s1s.get('mean_gap_annvar')} (ann. var), {s1s.get('pct_positive')} of days positive, "
        f"t={s1s.get('t_stat')}, permutation p={s1s.get('permutation_p')} → exists={s1s.get('vrp_exists')}",
        f"- **QQQ/^VXN**: mean gap {s1q.get('mean_gap_annvar')}, {s1q.get('pct_positive')} positive, "
        f"t={s1q.get('t_stat')}, p={s1q.get('permutation_p')} → exists={s1q.get('vrp_exists')}",
        f"- **Both-sides (NN#2), SPY**: calm VIX≤30 mean {s1s.get('both_sides',{}).get('calm_VIX_le30',{}).get('mean_gap')} "
        f"vs stressed VIX>30 mean {s1s.get('both_sides',{}).get('stressed_VIX_gt30',{}).get('mean_gap')}. Both sides "
        "positive — the FORWARD gap is even larger right after spikes (implied overshoots and mean-reverts). The "
        "crisis *cost* of harvesting does not show up in this forward existence gap; it shows up in the Stage-2 "
        "daily harvest mark — which is exactly why Stage 2, not Stage 1, is the kill-gate.",
        "",
        "## Stage 2 — orthogonality kill-gate (causal harvest return vs carry)",
        f"- harvest standalone: Sharpe(rf0)={s2['harvest_profile'].get('sharpe_rf0')}, "
        f"{s2['harvest_profile'].get('pct_positive_days')} positive days (n={s2['harvest_profile'].get('n')})",
        f"- **vs DBV carry**: full ρ={cv.get('full_sample',{}).get('rho')} ({cv.get('full_sample',{}).get('band')}), "
        f"max crisis |ρ|={cv.get('max_crisis_abs_corr')}, VIX>30 ρ={cv.get('vix_gt30',{}).get('rho')} → **{cv.get('verdict')}**",
        f"- vs v015 carry (recent secondary): {s2['vs_v015_secondary'].get('verdict')}",
        f"- vs overnight-QQQ: {s2['vs_overnight_qqq'].get('verdict')}",
        "",
        "## Stage 3 — iron-condor strategy backtest",
        f"**{payload['stage3_strategy_sim']['status']}.** {payload['stage3_strategy_sim']['reason']} "
        "The pre-registered strategy + cost spec are frozen in `strategy_simulator.py`, ready for the day real "
        "chains exist. Candidate providers and required fields are listed in the JSON.",
        "",
        "## Verdict",
        f"**{payload['verdict']}** — gates: {json.dumps(payload['gates'])}",
        "",
        "## Caveats",
    ]
    lines += [f"- {c}" for c in payload["caveats"]]
    lines.append("")
    FINDINGS.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    _check_prereg()

    # ── Load (free yfinance + log reads; thin-data guards inside loader) ──
    spy = dl.load_underlying("SPY")
    qqq = dl.load_underlying("QQQ")
    vix = dl.load_vol_index("^VIX")
    vxn = dl.load_vol_index("^VXN")
    carry_dbv = dl.load_carry_proxy()
    carry_v015 = dl.load_forex_log_carry()
    overnight_qqq = dl.load_overnight_qqq(qqq)

    # ── Stage 1 — existence ──
    s1_spy = val.stage1_existence("SPY/^VIX", vix, spy["Close"], perms=args.perms, seed=args.seed)
    s1_qqq = val.stage1_existence("QQQ/^VXN", vxn, qqq["Close"], perms=args.perms, seed=args.seed)

    # ── Stage 2 — orthogonality kill-gate (SPY is the canonical VRP harvest series) ──
    harvest = vc.harvest_return_causal(vix, spy["Close"])
    s2 = val.stage2_orthogonality(harvest, carry_dbv, carry_v015, overnight_qqq, vix)

    verdict, gates = val.build_verdict(s1_spy, s1_qqq, s2)
    sim = iron_condor_simulate()
    p_spy = s1_spy.get("permutation_p")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "id": "VRP-001",
        "approach": "inverted_gauntlet",
        "citation": CITATION,
        "verdict": verdict,
        "gates": gates,
        "stage1_existence": {"SPY": s1_spy, "QQQ": s1_qqq},
        "stage2_orthogonality": s2,
        "stage3_strategy_sim": sim,
        "caveats": [
            "No historical SPY/QQQ option chains in this system — the iron-condor backtest is BLOCKED, not faked.",
            "Stage-1 BTZ gap uses forward realized variance (look-ahead) and is quarantined to the existence stat.",
            "Stage-2 harvest return is strictly causal: (IV_{t-1}/100)^2/252 - r_t^2.",
            "Carry crisis coverage comes from DBV (2006->2023-03); the v015 forex-log carry is recent/noisy secondary.",
            "Crisis/stress correlation decides orthogonality, NOT the benign full-sample average.",
            "The Stage-2 harvest proxy is a daily LINEAR variance P&L; a real short iron condor's crisis loss is "
            "CONVEX and clusters in vol spikes, so this proxy likely UNDERSTATES true crisis coupling. A "
            "TRUE_DIVERSIFIER read vs carry is encouraging but must be confirmed on real chains, not banked.",
            "Harvest standalone Sharpe is UNCOSTED and on a proxy series, not the tradeable strategy — do not "
            "treat it as a deployable number (the operation's prior over-annualized-Sharpe trap).",
        ],
        "meta": {"perms": args.perms, "seed": args.seed},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _write_findings(payload)

    # ── Ledger (idempotent by id) ──
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    led = [e for e in led if e.get("id") != "VRP-001"]
    cv = s2["vs_carry"]
    led.append({
        "id": "VRP-001",
        "name": "Volatility risk premium (SPY/QQQ) — orthogonal second edge or return-stacking?",
        "status": verdict,
        "date_tested": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "result": (f"VRP exists SPY={s1_spy.get('vrp_exists')} (mean_gap={s1_spy.get('mean_gap_annvar')}, "
                   f"p={p_spy}) QQQ={s1_qqq.get('vrp_exists')}; harvest x carry(DBV) full_rho="
                   f"{cv.get('full_sample',{}).get('rho')} maxCrisis={cv.get('max_crisis_abs_corr')} "
                   f"-> {cv.get('verdict')}. Iron-condor backtest DATA_INSUFFICIENT (no option chains)."),
        "p_value": p_spy,
        "methodology_note": ("Inverted gauntlet on free yfinance proxies (no option chains). Stage1 BTZ forward "
                             "IV-RV gap, both-sides calm/stressed, sign-flip permutation. Stage2 strictly-causal "
                             "harvest return (IV_{t-1}^2/252 - r_t^2) vs DBV carry/v015/overnight-QQQ; crisis "
                             "(2008/2020/2022)+VIX>30+rolling-60d corr. " + CITATION + ". Iron-condor sim blocked: "
                             "brief forbids synthesizing option prices."),
    })
    LEDGER.write_text(json.dumps(led, indent=2))

    # ── Print ──
    def _row(name, a):
        cr = a.get("crises", {})
        g = lambda k: str(cr.get(k, {}).get("rho"))
        print(f"  {name:26s} full={str(a.get('full_sample',{}).get('rho')):>6s}  "
              f"2008={g('GFC_2008'):>6s}  2020={g('COVID_2020'):>6s}  2022={g('RATE_SHOCK_2022'):>6s}  "
              f"VIX>30={str(a.get('vix_gt30',{}).get('rho')):>6s}  -> {a.get('verdict')}")

    print(f"\n{'='*92}\n  VRP VALIDATION — inverted gauntlet (free proxies; no option chains)\n{'='*92}")
    print(f"  STAGE 1 existence:")
    for s in (s1_spy, s1_qqq):
        print(f"    {s.get('label'):10s} mean_gap={s.get('mean_gap_annvar')}  %pos={s.get('pct_positive')}  "
              f"t={s.get('t_stat')}  p={s.get('permutation_p')}  exists={s.get('vrp_exists')}")
    print(f"  STAGE 2 orthogonality (harvest return vs comparator; crisis corr decides):")
    print(f"  {'series vs':26s} {'full':>6s}  {'2008':>6s}  {'2020':>6s}  {'2022':>6s}  {'VIX>30':>6s}")
    _row("harvest x DBV carry", cv)
    _row("harvest x overnight-QQQ", s2["vs_overnight_qqq"])
    print(f"  harvest standalone: Sharpe(rf0)={s2['harvest_profile'].get('sharpe_rf0')}  "
          f"%pos={s2['harvest_profile'].get('pct_positive_days')}")
    print(f"  STAGE 3 iron-condor sim: {sim['status']} ({sim['reason']})")
    print(f"\n  GATES: {gates}")
    print(f"  VERDICT: {verdict}")
    print(f"\n  Logged VRP-001. Saved: data/research/vrp_validation.json + vrp_findings.md\n{'='*92}\n")


if __name__ == "__main__":
    main()
