"""
Universe backtest entry point.
Delegates to run_universe_sweep.main() then runs 4 correctness checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_full_universe() -> None:
    import json
    from scripts.run_universe_sweep import main as sweep_main

    sweep_main()

    # ── Load outputs ──────────────────────────────────────────────────────────
    logs = ROOT / "logs"
    with open(logs / "full_universe_results.json") as f:
        universe_results = json.load(f)
    import pandas as pd
    trade_df = pd.read_csv(logs / "full_trade_map.csv")
    with open(logs / "ai_company_analysis.json") as f:
        ai_analysis = json.load(f)

    # ── Rebuild ranked combos ─────────────────────────────────────────────────
    all_perf = []
    for sym, strats in universe_results.items():
        for strat, res in strats.items():
            if isinstance(res, dict) and "sharpe_proxy" in res:
                all_perf.append({
                    "asset": sym, "strategy": strat,
                    "sharpe": res["sharpe_proxy"],
                    "pf": res["profit_factor"],
                    "win_rate": res["win_rate"],
                    "best_params": res.get("best_param_combo", {}),
                })
    all_perf.sort(key=lambda x: x["sharpe"], reverse=True)
    top10 = all_perf[:10]
    top5  = all_perf[:5]
    bottom5 = [p for p in all_perf if p["pf"] < 1.0][-5:]

    # ── Check 1: top 5 sharpes must be distinct ───────────────────────────────
    top5_sharpes = [r["sharpe"] for r in top5]
    if len(set(top5_sharpes)) < len(top5_sharpes):
        print(f"FAIL: Check 1 — top 5 Sharpe values are NOT all distinct: {top5_sharpes}")
        sys.exit(1)
    print(f"PASS: Check 1 — top 5 Sharpe values are all distinct: {top5_sharpes}")

    # ── Check 2: at least 3 different strategies in top 10 ───────────────────
    top10_strats = set(r["strategy"] for r in top10)
    if len(top10_strats) < 3:
        print(f"FAIL: Check 2 — only {len(top10_strats)} strategies in top 10: {top10_strats}")
        sys.exit(1)
    print(f"PASS: Check 2 — {len(top10_strats)} strategies in top 10: {top10_strats}")

    # ── Check 3: AI stocks have different B&H annual returns ─────────────────
    bh_vals = [
        round(a["buy_hold_annual_pct"], 1)
        for a in ai_analysis
        if a.get("status") != "no_data"
    ]
    if len(bh_vals) >= 2 and len(set(bh_vals)) == 1:
        print(f"FAIL: Check 3 — all AI stocks have identical B&H return {bh_vals[0]:.1f}%/yr")
        sys.exit(1)
    print(f"PASS: Check 3 — AI stocks have distinct B&H annual returns: {bh_vals}")

    # ── Check 4: total trades > 10,000 ───────────────────────────────────────
    total_trades = len(trade_df)
    if total_trades <= 10_000:
        print(f"FAIL: Check 4 — only {total_trades:,} trades mapped (need > 10,000)")
        sys.exit(1)
    print(f"PASS: Check 4 — {total_trades:,} trades mapped")

    # ── All checks passed ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("ALL CHECKS PASSED")
    print()
    print("TOP 5 COMBINATIONS BY SHARPE:")
    for r in top5:
        print(f"  {r['asset']:6s} × {r['strategy']:20s}  sharpe={r['sharpe']:.3f}  "
              f"pf={r['pf']:.2f}  wr={r['win_rate']:.2%}  params={r['best_params']}")
    print()
    print("WORST 5 COMBINATIONS TO AVOID:")
    for r in bottom5:
        print(f"  {r['asset']:6s} × {r['strategy']:20s}  sharpe={r['sharpe']:.3f}  "
              f"pf={r['pf']:.2f}  wr={r['win_rate']:.2%}  params={r['best_params']}")
    print("═" * 60)


if __name__ == "__main__":
    run_full_universe()
