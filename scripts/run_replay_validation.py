#!/usr/bin/env python3
"""
Run replay validation over recent history and enforce ML gate checks.

Pass criteria:
- 3x-slippage stress test passes in backtest result
- all gate checks in this script evaluate True

Exit codes:
- 0: replay_passed=True
- 1: replay_passed=False (including dependency/runtime failure)

Outputs:
- Writes a structured JSON report (default: data/reports/replay_validation_latest.json)
  used by phase-11 guardrails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def evaluate_gates(snapshot: Dict[str, Any], history_count: int) -> Dict[str, Any]:
    modules = snapshot.get("modules", {})
    votes = snapshot.get("ensemble_vote", {}).get("votes", {})
    blended = snapshot.get("ensemble_vote", {}).get("blended_conf")
    runtime_mods = snapshot.get("runtime_modulators", {})
    integrations = snapshot.get("integrations", {})

    module_keys = {"softmax", "kalman", "ml_diag", "predict_now", "alpha_decay", "pegasus", "mdp", "lqr", "corr_tracker", "bs"}
    module_presence_ok = module_keys.issubset(set(modules.keys()))

    non_degenerate_ensemble = (
        history_count > 0
        and votes.get("hmm") is not None
        and (votes.get("softmax") is not None or votes.get("kmeans") is not None)
        and _is_finite_number(blended)
    )

    mdp = modules.get("mdp", {})
    pegasus = modules.get("pegasus", {})
    mdp_trades = int(mdp.get("trades", 0))
    mdp_policy = str(mdp.get("policy_source", "expert_prior"))
    mdp_gate_ok = (mdp_trades < 20 and mdp_policy == "expert_prior") or (mdp_trades >= 20 and mdp_policy == "learned")

    peg_updates = int(pegasus.get("updates", 0))
    peg_trust = float(pegasus.get("trust", 0.0))
    if peg_updates < 20:
        pegasus_gate_ok = abs(peg_trust) <= 1e-6
    elif peg_updates < 30:
        pegasus_gate_ok = 0.0 <= peg_trust <= 1.0
    else:
        pegasus_gate_ok = abs(peg_trust - 1.0) <= 1e-6

    bounded_keys = ["mdp_mult", "lqr_mult", "vol_mult", "pegasus_mult", "position_size"]
    bounded_ok = all(
        (_is_finite_number(runtime_mods.get(k)) and -10.0 < float(runtime_mods.get(k)) < 10_000.0)
        for k in bounded_keys if k in runtime_mods
    ) if runtime_mods else True

    library_ok = bool(integrations.get("alexandrian_library_loaded") or integrations.get("market_memory_loaded"))

    checks = {
        "module_presence_ok": module_presence_ok,
        "non_degenerate_ensemble": non_degenerate_ensemble,
        "mdp_gate_consistent": mdp_gate_ok,
        "pegasus_gate_consistent": pegasus_gate_ok,
        "bounded_runtime_outputs": bounded_ok,
        "library_integration_loaded": library_ok,
    }

    checks["all_pass"] = all(checks.values())
    return checks


def run_replay(symbols: List[str], days: int) -> Dict[str, Any]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)

    from sovereign.validation.backtest_engine import SovereignBacktest

    bt = SovereignBacktest(
        symbols=symbols,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        starting_equity=100000.0,
        slippage=0.001,
    )
    result = bt.run()
    snapshot = bt.orchestrator.get_latest_ml_snapshot()
    history_count = len(getattr(bt.orchestrator, "_ml_snapshot_history", []))
    gates = evaluate_gates(snapshot, history_count)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": days},
        "symbols": symbols,
        "backtest_metrics": {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "passed_3x_slippage": result.passed_3x_slippage,
        },
        "ml_snapshot": snapshot,
        "ml_snapshot_count": history_count,
        "gate_checks": gates,
        "replay_passed": bool(result.passed_3x_slippage and gates["all_pass"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 60-day replay validation with ML gates")
    parser.add_argument("--days", type=int, default=60, help="Replay window size in days")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to replay",
    )
    parser.add_argument(
        "--out",
        default="data/reports/replay_validation_latest.json",
        help="Output report path relative to repo root",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    symbols = args.symbols
    if not symbols:
        from config.loader import params
        symbols = params.get("universe", {}).get("trinity_assets", ["META", "PFE", "UNH"])

    try:
        report = run_replay(symbols=symbols, days=args.days)
    except Exception as e:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window": {"days": args.days},
            "symbols": symbols,
            "replay_passed": False,
            "error": str(e),
        }

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["replay_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
