"""
Prop firm challenge simulator — Monte Carlo over real ICT trade history.

Loads actual R-multiples from ICT backtest windows, runs 10,000 shuffled
sequences through PropFirmRules, and reports pass/bust/timeout rates.

Run:
    python3 sovereign/propfirm/challenge_simulator.py
    python3 sovereign/propfirm/challenge_simulator.py --firm mff --account 50000
    python3 sovereign/propfirm/challenge_simulator.py --window A --n 5000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sovereign.propfirm.rules_engine import PropFirmRules

ROOT = Path(__file__).resolve().parents[2]
WINDOW_FILES = {
    "A":          ROOT / "logs" / "ict_backtest_window_A.json",
    "B":          ROOT / "logs" / "ict_backtest_window_B.json",
    "combined":   ROOT / "logs" / "ict_backtest_results.json",
    # Forensics-filtered windows (post 2026-05-18)
    "london_a":   ROOT / "logs" / "ict_backtest_london_a.json",    # deployed edge
    "london_all": ROOT / "logs" / "ict_backtest_london_all.json",  # conservative
}
OUT_FILE = ROOT / "logs" / "prop_challenge_sim.json"

# Assume 3-7 day average hold → ~8 trades/month → challenge window
MAX_CHALLENGE_DAYS = 90   # timeout if not passed in 90 days


def _load_r_multiples(window: str = "combined") -> Tuple[List[float], dict]:
    """Load actual R-multiples from backtest files."""
    path = WINDOW_FILES.get(window)
    if path is None or not path.exists():
        raise FileNotFoundError(f"No backtest file for window={window}")

    raw = json.loads(path.read_text())
    trades = raw.get("trades", [])
    if not trades:
        raise ValueError(f"No trades found in {path.name}")

    r_values = [float(t["pnl_r"]) for t in trades]
    stats = raw.get("stats", {})
    return r_values, stats


def run_single_challenge(
    r_sequence: List[float],
    rules: PropFirmRules,
    trades_per_day: float = 0.8,
) -> dict:
    """
    Run one challenge simulation through a sequence of trades.
    trades_per_day controls how many calendar days each trade uses.
    """
    rules.open_challenge()
    days_per_trade = 1.0 / trades_per_day
    day_accumulator = 0.0
    lowest_balance = rules.balance
    highest_balance = rules.balance

    for i, r in enumerate(r_sequence):
        if not rules.is_active:
            break

        # Check timeout
        if rules.trading_days >= MAX_CHALLENGE_DAYS:
            rules.outcome = "TIMEOUT"
            rules.is_active = False
            break

        rec = rules.apply_trade_pnl(r_multiple=r)

        lowest_balance  = min(lowest_balance,  rules.balance)
        highest_balance = max(highest_balance, rules.balance)

        # Simulate day boundaries
        day_accumulator += days_per_trade
        while day_accumulator >= 1.0 and rules.is_active:
            rules.update_eod()
            day_accumulator -= 1.0

            if rules.is_passed():
                rules.outcome = "PASSED"
                rules.is_active = False
                break
            if rules.is_bust():
                break

    # Flush remaining partial day
    if rules.is_active:
        if rules.trading_days < MAX_CHALLENGE_DAYS:
            rules.update_eod()
        if rules.is_passed():
            rules.outcome = "PASSED"
        elif rules.is_bust():
            rules.outcome = "BUST"
        else:
            rules.outcome = "TIMEOUT"
        rules.is_active = False

    s = rules.summary()
    s["lowest_balance"] = round(lowest_balance, 2)
    s["highest_balance"] = round(highest_balance, 2)
    return s


def run_monte_carlo(
    r_values: List[float],
    rules_factory,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Run N Monte Carlo simulations with shuffled trade order."""
    rng = random.Random(seed)
    outcomes = []

    for i in range(n_simulations):
        shuffled = r_values.copy()
        rng.shuffle(shuffled)
        rules = rules_factory()
        result = run_single_challenge(shuffled, rules)
        outcomes.append(result)

    # Aggregate
    outcome_counts = Counter(o["outcome"] for o in outcomes)
    passed  = [o for o in outcomes if o["outcome"] == "PASSED"]
    busted  = [o for o in outcomes if o["outcome"] == "BUST"]

    pass_rate   = len(passed) / n_simulations
    bust_rate   = len(busted) / n_simulations
    timeout_rate = outcome_counts.get("TIMEOUT", 0) / n_simulations

    # Days to pass distribution
    days_to_pass = [o["trading_days"] for o in passed] if passed else []

    # Blocked trades
    all_blocked_pcts = [o["blocked_pct"] for o in outcomes]
    all_reduced_pcts = [o["trades_reduced"] / max(o["trade_count"], 1) for o in outcomes]

    return {
        "n_simulations": n_simulations,
        "n_trades_in_sequence": len(r_values),
        "pass_rate": round(pass_rate, 4),
        "bust_rate": round(bust_rate, 4),
        "timeout_rate": round(timeout_rate, 4),
        "median_days_to_pass": round(float(np.median(days_to_pass)), 1) if days_to_pass else None,
        "p10_days_to_pass": round(float(np.percentile(days_to_pass, 10)), 1) if days_to_pass else None,
        "p90_days_to_pass": round(float(np.percentile(days_to_pass, 90)), 1) if days_to_pass else None,
        "avg_trades_taken": round(float(np.mean([o["trade_count"] for o in outcomes])), 1),
        "avg_blocked_pct": round(float(np.mean(all_blocked_pcts)), 4),
        "avg_reduced_pct": round(float(np.mean(all_reduced_pcts)), 4),
        "avg_return_if_passed": round(float(np.mean([o["return_pct"] for o in passed])), 2) if passed else None,
        "avg_return_if_busted": round(float(np.mean([o["return_pct"] for o in busted])), 2) if busted else None,
        "worst_drawdown_seen": round(float(np.min([o["balance"] - o["lowest_balance"]
                                                    for o in outcomes if "lowest_balance" in o])), 2),
        "outcome_counts": dict(outcome_counts),
        "sizing_warning": float(np.mean(all_blocked_pcts)) > 0.30,
    }


def print_report(mc: dict, firm: str, account_size: float) -> None:
    r = mc
    print(f"\n{'='*60}")
    print(f"PROP FIRM CHALLENGE SIMULATION — {firm.upper()} ${account_size:,.0f}")
    print(f"{'='*60}")
    print(f"Simulations:        {r['n_simulations']:,}")
    print(f"Trades in sequence: {r['n_trades_in_sequence']}")
    print()
    print(f"Pass rate:          {r['pass_rate']*100:.1f}%")
    print(f"Bust rate:          {r['bust_rate']*100:.1f}%")
    print(f"Timeout rate:       {r['timeout_rate']*100:.1f}%")
    print()
    if r['median_days_to_pass']:
        print(f"Median days to pass:  {r['median_days_to_pass']:.0f}")
        print(f"P10/P90 days:         {r['p10_days_to_pass']:.0f} / {r['p90_days_to_pass']:.0f}")
    print(f"Avg trades taken:     {r['avg_trades_taken']:.1f}")
    print()
    print(f"Trades BLOCKED:       {r['avg_blocked_pct']*100:.1f}%  {'⚠ SIZING ISSUE' if r['sizing_warning'] else '✓ OK'}")
    print(f"Trades SIZE-REDUCED:  {r['avg_reduced_pct']*100:.1f}%")
    print()
    if r['avg_return_if_passed']:
        print(f"Avg return if passed: +{r['avg_return_if_passed']:.1f}%")
    if r['avg_return_if_busted']:
        print(f"Avg return if busted:  {r['avg_return_if_busted']:.1f}%")
    print()
    verdict = (
        "🟢 READY — pass rate > 70%, buy the challenge"
        if r['pass_rate'] >= 0.70 else
        "🟡 CLOSE — simulate more paper challenges first"
        if r['pass_rate'] >= 0.60 else
        "🔴 NOT READY — pass rate < 60%, review edge before spending"
    )
    print(f"Verdict: {verdict}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--firm", choices=["lucid", "mff"], default="lucid")
    parser.add_argument("--account", type=float, default=100_000)
    parser.add_argument("--window", choices=["A", "B", "combined", "london_a", "london_all"],
                        default="london_a", help="Trade window (default: london_a = deployed edge)")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--risk", type=float, default=0.0075,
                        help="Risk per trade as fraction of balance (default 0.0075 = 0.75%%)")
    args = parser.parse_args()

    print(f"Loading ICT trade history (window={args.window})...")
    r_values, stats = _load_r_multiples(args.window)
    print(f"Loaded {len(r_values)} trades  "
          f"(WR={stats.get('win_rate',0)*100:.1f}% "
          f"avgR={stats.get('avg_r',0):.3f})")

    def rules_factory():
        if args.firm == "lucid":
            r = PropFirmRules.lucid(account_size=args.account)
        else:
            r = PropFirmRules.mff(account_size=args.account)
        r.risk_per_trade_pct = args.risk
        return r

    print(f"Running {args.n:,} Monte Carlo simulations...")
    mc = run_monte_carlo(r_values, rules_factory, n_simulations=args.n)
    mc["firm"] = args.firm
    mc["account_size"] = args.account
    mc["window"] = args.window
    mc["risk_pct"] = args.risk

    print_report(mc, args.firm, args.account)
    OUT_FILE.write_text(json.dumps(mc, indent=2))
    print(f"Full results saved: {OUT_FILE}")
