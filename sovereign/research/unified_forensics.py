"""
Unified Trade Forensics — ICT + Forex systems analyzed together.

Applies the same scrutiny framework to both systems:
  - What went right (win drivers)
  - What went wrong (failure modes)
  - What both systems share (common edges and anti-patterns)
  - Combat rules derived from evidence

Output: data/research/unified_forensics.json
         data/research/ict_forensics.json
         data/research/shared_insights.json

Run: PYTHONPATH=/path/to/quant python3 sovereign/research/unified_forensics.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "data" / "research"
OUT.mkdir(parents=True, exist_ok=True)

# ── ICT FORENSICS ─────────────────────────────────────────────────────────

def analyze_ict(trades: List[Dict]) -> Dict:
    wins   = [t for t in trades if t["pnl_r"] > 0]
    losses = [t for t in trades if t["pnl_r"] <= 0]
    components = ["kill_zone","sweep","displacement","fvg_tap","market_structure","pd_alignment"]

    # Component score delta (wins vs losses)
    comp_deltas = {}
    for c in components:
        w_avg = np.mean([t["component_scores"].get(c, 0) for t in wins])
        l_avg = np.mean([t["component_scores"].get(c, 0) for t in losses])
        comp_deltas[c] = {"win_avg": round(float(w_avg), 3),
                          "loss_avg": round(float(l_avg), 3),
                          "delta": round(float(w_avg - l_avg), 3)}

    # Grade breakdown
    by_grade = defaultdict(list)
    for t in trades:
        by_grade[t["grade"]].append(t["pnl_r"])
    grade_stats = {
        g: {"count": len(rs), "win_rate": round(sum(1 for r in rs if r > 0) / len(rs), 3),
            "avg_r": round(float(np.mean(rs)), 3)}
        for g, rs in by_grade.items()
    }

    # Session breakdown
    by_sess = defaultdict(list)
    for t in trades:
        by_sess[t["session"]].append(t["pnl_r"])
    session_stats = {
        s: {"count": len(rs), "win_rate": round(sum(1 for r in rs if r > 0) / len(rs), 3),
            "avg_r": round(float(np.mean(rs)), 3)}
        for s, rs in by_sess.items()
    }

    # Fast stop analysis (stopped within 3 bars — timing problem)
    fast_stops = [t for t in losses if t["hold_bars"] <= 3]
    fast_stop_pct = len(fast_stops) / max(len(losses), 1)

    # pd_alignment anti-edge
    pd_zero = [t for t in trades if t["component_scores"].get("pd_alignment", 0) == 0]
    pd_pos  = [t for t in trades if t["component_scores"].get("pd_alignment", 0) > 0]
    pd_zero_wr = sum(1 for t in pd_zero if t["pnl_r"] > 0) / max(len(pd_zero), 1)
    pd_pos_wr  = sum(1 for t in pd_pos  if t["pnl_r"] > 0) / max(len(pd_pos),  1)

    # Failure modes
    failure_modes = {
        "APLUS_PARADOX": {
            "description": "A+ grade (score>9) has 13% WR vs 39% for grade A. Higher scores anti-correlated with success. Over-scored setups may coincide with high-vol moments that immediately reverse.",
            "count": len([t for t in trades if t["grade"] == "A+"]),
            "win_rate": grade_stats.get("A+", {}).get("win_rate", 0),
            "avg_r": grade_stats.get("A+", {}).get("avg_r", 0),
            "evidence": "Score at TP2 wins: 8.05 | Score at STOP losses: 8.41",
        },
        "FAST_STOP_TIMING": {
            "description": "61% of losses stopped within 3 bars. Entry timing is off — price hasn't actually committed to the move yet when the system enters.",
            "count": len(fast_stops),
            "pct_of_losses": round(fast_stop_pct, 3),
            "avg_score": round(float(np.mean([t["score"] for t in fast_stops])), 2),
        },
        "NY_PM_DRAG": {
            "description": "NY_PM averaging -0.283R across all pairs. London averages +0.471R. The session edge lives in London only.",
            "session_stats": session_stats,
        },
        "PD_ALIGNMENT_ANTI_EDGE": {
            "description": "pd_alignment>0 produces 20% WR vs 35% WR with pd_alignment=0. The PD array component is hurting, not helping.",
            "pd_zero_wr": round(pd_zero_wr, 3),
            "pd_pos_wr":  round(pd_pos_wr,  3),
            "delta": round(pd_zero_wr - pd_pos_wr, 3),
        },
    }

    # Win drivers
    tp2_trades = [t for t in wins if t["outcome"] == "TP2"]
    win_drivers = {
        "LONDON_TP2": {
            "description": "14/17 TP2 hits are London session. London is the primary edge engine.",
            "count": len(tp2_trades),
            "london_pct": round(sum(1 for t in tp2_trades if t["session"] == "London") / max(len(tp2_trades), 1), 3),
            "avg_hold_bars": round(float(np.mean([t["hold_bars"] for t in tp2_trades])), 1),
        },
        "GRADE_A_NOT_APLUS": {
            "description": "Grade A (score 7-9) outperforms A+ (score 9+) by 26 percentage points in WR.",
            "grade_a_wr": grade_stats.get("A", {}).get("win_rate", 0),
            "grade_aplus_wr": grade_stats.get("A+", {}).get("win_rate", 0),
        },
    }

    # Combat rules
    combat_rules = [
        {"id": "ICT-C001", "type": "VETO", "priority": 1,
         "condition": "grade == 'A+' AND session == 'NY_PM'",
         "evidence": f"A+ NY_PM: {len([t for t in trades if t['grade']=='A+' and t['session']=='NY_PM'])} trades",
         "detail": "A+ in NY_PM is the worst combination. Neither the grade nor the session have edge here."},
        {"id": "ICT-C002", "type": "VETO", "priority": 2,
         "condition": "pd_alignment > 0 AND session == 'NY_PM'",
         "evidence": f"pd_alignment>0 WR: {pd_pos_wr*100:.0f}% vs {pd_zero_wr*100:.0f}%",
         "detail": "pd_alignment component is anti-edge. Consider zeroing its weight in the scorer."},
        {"id": "ICT-B001", "type": "BOOST", "priority": 1,
         "condition": "session == 'London' AND grade == 'A' AND score < 9.0",
         "evidence": "London A-grade: 39% WR, TP2 rate highest",
         "detail": "This is the exact setup profile that generates TP2 hits. Size at 1.0x (don't reduce)."},
        {"id": "ICT-C003", "type": "FILTER", "priority": 2,
         "condition": "Test London-only mode (disable NY_PM entirely)",
         "evidence": f"London avgR={session_stats.get('London',{}).get('avg_r',0):+.3f} vs NY_PM avgR={session_stats.get('NY_PM',{}).get('avg_r',0):+.3f}",
         "detail": "Run backtest with NY_PM disabled. Expected Sharpe improvement: +0.3-0.5 by removing -0.283R avg session."},
    ]

    return {
        "system": "ICT",
        "total_trades": len(trades),
        "win_rate": round(sum(1 for t in trades if t["pnl_r"] > 0) / len(trades), 3),
        "avg_r": round(float(np.mean([t["pnl_r"] for t in trades])), 3),
        "component_deltas": comp_deltas,
        "grade_stats": grade_stats,
        "session_stats": session_stats,
        "failure_modes": failure_modes,
        "win_drivers": win_drivers,
        "combat_rules": combat_rules,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── FOREX FORENSICS EXTENSION ─────────────────────────────────────────────

def analyze_forex_unexplained(forensics: List[Dict]) -> Dict:
    """Resolve the UNEXPLAINED category — the path to Sharpe 1.5."""
    unexplained = [r for r in forensics if r.get("failure_mode") == "UNEXPLAINED"]
    all_losses  = [r for r in forensics if r["outcome"] == "LOSS"]

    # They're macro-aligned (0.949) but flat momentum (0.005)
    # The trailing stop fires before the move materializes
    trailing_unexplained = [r for r in unexplained if r["exit_reason"] == "trailing_stop"]
    time_unexplained     = [r for r in unexplained if r["exit_reason"] == "time"]

    # Test: does requiring |momentum| > 0.5% filter them?
    mom_threshold = 0.005
    would_filter = [r for r in unexplained if abs(r["momentum_63d"]) < mom_threshold]
    would_keep   = [r for r in unexplained if abs(r["momentum_63d"]) >= mom_threshold]

    # Also check: what happens to wins if we apply the same filter?
    wins = [r for r in forensics if r["outcome"] == "WIN"]
    wins_filtered = [r for r in wins if abs(r["momentum_63d"]) < mom_threshold]

    return {
        "unexplained_count": len(unexplained),
        "unexplained_avg_r": round(float(np.mean([r["outcome_r"] for r in unexplained])), 3),
        "macro_vs_dir_avg": round(float(np.mean([r["macro_vs_direction"] for r in unexplained])), 3),
        "momentum_avg": round(float(np.mean([r["momentum_63d"] for r in unexplained])), 4),
        "exit_breakdown": {
            "trailing_stop": len(trailing_unexplained),
            "time": len(time_unexplained),
            "stop": len([r for r in unexplained if r["exit_reason"] == "stop"]),
        },
        "momentum_filter_test": {
            "threshold": mom_threshold,
            "unexplained_would_filter": len(would_filter),
            "unexplained_would_keep": len(would_keep),
            "wins_would_also_filter": len(wins_filtered),
            "net_r_saved": round(float(np.sum([r["outcome_r"] for r in would_filter])), 2),
            "win_r_lost": round(float(np.sum([r["outcome_r"] for r in wins_filtered])), 2),
            "net_gain": round(
                float(np.sum([r["outcome_r"] for r in would_filter])) -
                float(np.sum([r["outcome_r"] for r in wins_filtered])), 2
            ),
        },
        "root_cause": "Macro-aligned entries with flat momentum. Signal is directionally right but price has no energy to sustain the move. Trailing stop fires on the first counter-move before momentum confirms.",
        "combat_rule": {
            "id": "FX-C007",
            "type": "VETO",
            "condition": "|momentum_63d| < 0.5% (flat momentum at entry)",
            "detail": "Block entries where 63-day momentum magnitude is below 0.5%. Macro signal is valid but no momentum means no fuel for the move.",
        },
    }


# ── SHARED INSIGHTS ────────────────────────────────────────────────────────

def build_shared_insights(ict_result: Dict, forex_unexplained: Dict) -> Dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "core_insight": (
            "Both ICT and Forex share the same fundamental failure: "
            "ENTERING BEFORE THE MARKET COMMITS. "
            "ICT: 61% of losses stop within 3 bars — the move hasn't started. "
            "Forex: UNEXPLAINED losses are macro-aligned but momentum-flat — same cause. "
            "The edge exists. The timing is premature."
        ),
        "shared_failure": {
            "name": "PREMATURE_ENTRY",
            "ict_evidence": "61% of losses exit within 3 bars (fast stops)",
            "forex_evidence": "59 UNEXPLAINED losses: macro-aligned (0.949) but flat momentum (0.005)",
            "shared_fix": "Both systems need a momentum confirmation requirement at entry",
        },
        "shared_wins": {
            "ict": "London session A-grade entries → 14/17 TP2 hits. 7.2 bar avg hold.",
            "forex": "MACRO_ALIGNED_STRONG (rate_diff ≥ 2% + momentum confirms) → 3.08R avg",
            "pattern": "When direction + macro + momentum all agree AND the market has already started moving: that is the exact setup profile for outsized wins in BOTH systems.",
        },
        "next_experiments": [
            {
                "id": "EXP-001",
                "system": "ICT",
                "hypothesis": "London-only mode improves Sharpe by removing -0.283R NY_PM drag",
                "test": "Run ict_backtest with NY_PM disabled. Compare walk-forward stats.",
                "expected_impact": "WR: 31% → ~35% | AvgR: +0.160 → ~+0.35",
                "priority": 1,
            },
            {
                "id": "EXP-002",
                "system": "ICT",
                "hypothesis": "Score cap at 9.0 (block A+ grade) improves WR",
                "test": "Re-grade: treat all scores > 9.0 as 8.99 (grade A). Backtest.",
                "expected_impact": "Removes 26 trades at -0.375R avg, keeping 72 trades at +0.383R",
                "priority": 1,
            },
            {
                "id": "EXP-003",
                "system": "Forex",
                "hypothesis": "|momentum_63d| < 0.5% veto resolves 59 UNEXPLAINED losses",
                "test": "Add momentum floor to signal_engine._macro_signal_for_date(). Backtest.",
                "expected_impact": f"Net gain: {forex_unexplained['momentum_filter_test']['net_gain']:.2f}R across backtest",
                "priority": 2,
            },
            {
                "id": "EXP-004",
                "system": "ICT",
                "hypothesis": "pd_alignment component is anti-edge (20% WR vs 35% WR without it)",
                "test": "Zero out pd_alignment weight in ICT scorer. Backtest.",
                "expected_impact": "WR +15pp on subset with pd_alignment>0 (25 trades)",
                "priority": 2,
            },
        ],
        "shared_goal_tracker": {
            "ICT_target": "FunderPro challenge pass — need 30 paper trades, 70.5% MC pass rate confirmed",
            "Forex_target": "Sharpe 1.5 — current 1.0547, gap 0.4453",
            "joint_principle": "v004/v006 is the foundation. Add edges only when they survive out-of-sample. Never trade against macro+momentum alignment.",
        },
    }


# ── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading ICT trades...")
    ict_raw = json.loads((ROOT / "logs" / "ict_backtest_results.json").read_text())
    ict_trades = ict_raw["trades"]

    print("Analyzing ICT system...")
    ict_result = analyze_ict(ict_trades)
    (OUT / "ict_forensics.json").write_text(json.dumps(ict_result, indent=2))
    print(f"  {ict_result['total_trades']} trades | WR={ict_result['win_rate']*100:.0f}% | {len(ict_result['combat_rules'])} combat rules")

    print("Loading Forex forensics...")
    forex_forensics = json.loads((ROOT / "data" / "research" / "trade_forensics.json").read_text())

    print("Resolving UNEXPLAINED category...")
    forex_unexplained = analyze_forex_unexplained(forex_forensics)
    filter_test = forex_unexplained["momentum_filter_test"]
    print(f"  Would filter: {filter_test['unexplained_would_filter']}/{forex_unexplained['unexplained_count']} unexplained losses")
    print(f"  Net R gain: {filter_test['net_gain']:+.2f}R (saves {abs(filter_test['net_r_saved']):.2f}R, loses {filter_test['win_r_lost']:.2f}R from wins)")

    print("Building shared insights...")
    shared = build_shared_insights(ict_result, forex_unexplained)

    # Unified output
    unified = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ict": ict_result,
        "forex_unexplained": forex_unexplained,
        "shared_insights": shared,
    }
    (OUT / "unified_forensics.json").write_text(json.dumps(unified, indent=2))
    (OUT / "shared_insights.json").write_text(json.dumps(shared, indent=2))

    print(f"\nCore finding: {shared['core_insight'][:120]}...")
    print(f"\nTop experiments:")
    for exp in shared["next_experiments"]:
        print(f"  [{exp['id']}] P{exp['priority']} {exp['system']}: {exp['hypothesis'][:70]}")
    print(f"\nFiles written:")
    print(f"  {OUT}/ict_forensics.json")
    print(f"  {OUT}/unified_forensics.json")
    print(f"  {OUT}/shared_insights.json")
