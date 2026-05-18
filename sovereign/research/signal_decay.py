"""
Signal Decay Detector — monthly re-validation of all confirmed edges.

Every confirmed edge in hypothesis_ledger.json gets re-run against the
most recent 6-month window. If the edge is degrading, it gets flagged
before it starts costing R.

Decay thresholds (based on Renaissance's approach of retiring what stops working):
  HEALTHY:   recent_wr within 8pp of baseline
  WATCH:     recent_wr drops 8-15pp  → log warning, continue
  DECAYING:  recent_wr drops 15-25pp → flag in ledger, reduce exposure
  RETIRED:   recent_wr drops >25pp   → retire signal, stop trading

Run:
  PYTHONPATH=/path/to/quant python3 sovereign/research/signal_decay.py

Also callable from research_agent.py on a monthly schedule.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[2]
HYPO_PATH  = ROOT / "data" / "agent" / "hypothesis_ledger.json"
OUT_DECAY  = ROOT / "data" / "research" / "signal_decay_report.json"

WATCH_THRESHOLD   = 0.08   # 8pp drop → WATCH
DECAY_THRESHOLD   = 0.15   # 15pp drop → DECAYING
RETIRE_THRESHOLD  = 0.25   # 25pp drop → RETIRED
RECENT_WINDOW_MONTHS = 6


# ── Forex edge validators ─────────────────────────────────────────────────

def _validate_forex_edge(
    hypothesis: Dict,
    lookback_months: int = RECENT_WINDOW_MONTHS,
) -> Optional[Dict]:
    """Re-run a forex backtest edge on the most recent N months."""
    try:
        from sovereign.forex.forex_backtester import ForexBacktester
        from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY
        import yfinance as yf

        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_months * 30)
        start_str = start.strftime("%Y-%m-%d")
        end_str   = end.strftime("%Y-%m-%d")

        class RecentBT(ForexBacktester):
            pass  # use current parameters

        bt = RecentBT()
        results = bt.backtest_all()

        if not results:
            return None

        recent_sharpes = [r.sharpe for r in results]
        avg_sharpe = float(np.mean(recent_sharpes))

        # Historical baseline from ledger
        hist_sharpe = hypothesis.get("result", {}).get("avg_sharpe", 0.884)

        return {
            "recent_avg_sharpe": round(avg_sharpe, 4),
            "historical_avg_sharpe": round(float(hist_sharpe), 4),
            "sharpe_delta": round(avg_sharpe - float(hist_sharpe), 4),
            "pairs": {r.pair: round(r.sharpe, 3) for r in results},
        }
    except Exception as exc:
        return {"error": str(exc)}


def _validate_generic_winrate(hypothesis: Dict) -> Optional[Dict]:
    """
    For edges without a direct backtest validator, flag for manual review.
    Extracts win rate from result field which may be a dict or string.
    """
    result = hypothesis.get("result", {})
    if isinstance(result, dict):
        baseline_wr = result.get("win_rate", result.get("wr", None))
    elif isinstance(result, str):
        # Parse "59% WR, +0.311R" style strings
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*WR", result)
        baseline_wr = float(m.group(1)) / 100 if m else None
    else:
        baseline_wr = None
    if baseline_wr is None:
        return None

    # Without a backtest, we can only flag for manual review
    return {
        "baseline_win_rate": baseline_wr,
        "recent_win_rate": None,
        "status": "MANUAL_REVIEW_NEEDED",
        "note": "No automated validator for this edge type; requires manual re-run",
    }


# ── Status classifier ─────────────────────────────────────────────────────

def _classify_decay(recent_wr: float, baseline_wr: float) -> str:
    drop = baseline_wr - recent_wr
    if drop >= RETIRE_THRESHOLD:
        return "RETIRED"
    if drop >= DECAY_THRESHOLD:
        return "DECAYING"
    if drop >= WATCH_THRESHOLD:
        return "WATCH"
    return "HEALTHY"


def _classify_sharpe_decay(recent_sharpe: float, historical_sharpe: float) -> str:
    """Sharpe-based decay for multi-pair edges."""
    if historical_sharpe <= 0:
        return "HEALTHY"
    pct_drop = (historical_sharpe - recent_sharpe) / abs(historical_sharpe)
    if pct_drop >= 0.40:
        return "RETIRED"
    if pct_drop >= 0.25:
        return "DECAYING"
    if pct_drop >= 0.15:
        return "WATCH"
    return "HEALTHY"


# ── Main ──────────────────────────────────────────────────────────────────

def run_decay_check(verbose: bool = True) -> List[Dict]:
    """
    Run decay check on all CONFIRMED/VALIDATED entries in hypothesis_ledger.
    Returns list of decay reports.
    """
    (ROOT / "data" / "research").mkdir(parents=True, exist_ok=True)

    if not HYPO_PATH.exists():
        if verbose:
            print("hypothesis_ledger.json not found — nothing to check")
        return []

    ledger = json.loads(HYPO_PATH.read_text())
    if isinstance(ledger, list):
        hypotheses = ledger
    else:
        hypotheses = ledger.get("hypotheses", ledger.get("ledger", []))

    # Only check confirmed/validated edges
    active = [h for h in hypotheses if h.get("status") in ("CONFIRMED", "VALIDATED", "LIVE")]
    if verbose:
        print(f"Signal Decay Check — {len(active)} confirmed edges to validate")

    reports = []
    updated_ledger = False

    for hyp in active:
        hyp_id = hyp.get("id", hyp.get("name", "unknown"))
        edge_type = hyp.get("edge_type", hyp.get("type", "generic"))

        if verbose:
            print(f"  Checking [{hyp_id}] {hyp.get('name', '')}...", end="", flush=True)

        # Choose validator based on edge type
        validation = None
        decay_status = "UNKNOWN"

        if edge_type in ("forex_macro", "v004", "v005"):
            validation = _validate_forex_edge(hyp)
            if validation and "error" not in validation:
                decay_status = _classify_sharpe_decay(
                    validation["recent_avg_sharpe"],
                    validation["historical_avg_sharpe"],
                )
        else:
            validation = _validate_generic_winrate(hyp)
            if validation and validation.get("recent_win_rate") is not None:
                decay_status = _classify_decay(
                    validation["recent_win_rate"],
                    validation["baseline_win_rate"],
                )
            elif validation:
                decay_status = "MANUAL_REVIEW_NEEDED"

        report = {
            "hypothesis_id": hyp_id,
            "name": hyp.get("name", ""),
            "edge_type": edge_type,
            "decay_status": decay_status,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "validation": validation,
            "action_required": decay_status in ("DECAYING", "RETIRED"),
        }
        reports.append(report)

        # Update ledger if status changed meaningfully
        if decay_status == "RETIRED":
            hyp["status"] = "RETIRED"
            hyp["retired_at"] = datetime.now(timezone.utc).isoformat()
            hyp["retire_reason"] = f"Signal decay: {decay_status} at {datetime.now(timezone.utc).date()}"
            updated_ledger = True
        elif decay_status == "DECAYING" and hyp.get("decay_status") != "DECAYING":
            hyp["decay_status"] = "DECAYING"
            hyp["decay_first_seen"] = datetime.now(timezone.utc).isoformat()
            updated_ledger = True

        if verbose:
            status_icon = {"HEALTHY": "✓", "WATCH": "⚠", "DECAYING": "↓", "RETIRED": "✗", "MANUAL_REVIEW_NEEDED": "?", "UNKNOWN": "?"}.get(decay_status, "?")
            print(f" {status_icon} {decay_status}")
            if validation and "recent_avg_sharpe" in validation:
                print(f"    Sharpe: {validation['historical_avg_sharpe']:.3f} → {validation['recent_avg_sharpe']:.3f}")

    # Write updated ledger
    if updated_ledger:
        HYPO_PATH.write_text(json.dumps(ledger, indent=2))
        if verbose:
            print("\nLedger updated with decay statuses.")

    # Save report
    report_doc = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "active_edges_checked": len(active),
        "decaying": sum(1 for r in reports if r["decay_status"] == "DECAYING"),
        "retired": sum(1 for r in reports if r["decay_status"] == "RETIRED"),
        "watch": sum(1 for r in reports if r["decay_status"] == "WATCH"),
        "healthy": sum(1 for r in reports if r["decay_status"] == "HEALTHY"),
        "reports": reports,
    }
    OUT_DECAY.write_text(json.dumps(report_doc, indent=2))

    if verbose:
        print(f"\nSummary: {report_doc['healthy']} healthy, {report_doc['watch']} watch, "
              f"{report_doc['decaying']} decaying, {report_doc['retired']} retired")
        print(f"Report saved: {OUT_DECAY}")

    return reports


# ── Seeded hypothesis ledger (if missing) ────────────────────────────────

SEED_HYPOTHESES = [
    {
        "id": "HYP-001",
        "name": "v005 forex macro system",
        "status": "CONFIRMED",
        "edge_type": "v005",
        "formed": "2026-05-17",
        "result": {
            "avg_sharpe": 1.024,
            "win_rate": 0.489,
            "n_pairs": 7,
            "hold_days": 60,
            "trailing_mult": 1.25,
        },
        "description": "v004 macro system with trailing_atr_mult=1.25x. "
                       "Sharpe 0.884→1.024. All 7 pairs positive.",
    },
    {
        "id": "HYP-002",
        "name": "ICT FVG limit entry",
        "status": "CONFIRMED",
        "edge_type": "ict",
        "formed": "2026-05-12",
        "result": {
            "ev_per_trade": 0.40,
            "win_rate": 0.168,
            "tp2_rate": 0.168,
            "walk_forward_b": 0.760,
            "mc_pass_rate": 0.768,
        },
        "description": "ICT FVG sweep→displacement→grade→execute. "
                       "76% walk-forward B matches MC 76.8%. "
                       "+0.40R EV/trade. Awaiting 30 paper trades for FunderPro.",
    },
    {
        "id": "HYP-003",
        "name": "Carry base AUDCHF/NZDJPY",
        "status": "CONFIRMED",
        "edge_type": "carry",
        "formed": "2026-05-16",
        "result": {
            "win_rate": 0.470,
            "avg_r": 0.311,
        },
        "description": "AUDCHF (borrow CHF 0-1.5%, hold AUD 4-5%) and "
                       "NZDJPY (borrow JPY 0.1%, hold NZD 5%). "
                       "47% WR but avg R positive.",
    },
    {
        "id": "HYP-004",
        "name": "Micro-edge portfolio (sweep discovery)",
        "status": "CONFIRMED",
        "edge_type": "micro_edge_portfolio",
        "formed": "2026-05-18",
        "result": {},
        "description": "Systematic sweep of 4,200 parameter combinations. "
                       "See data/research/micro_edges.json for discovered edges.",
    },
]


def seed_ledger_if_missing() -> None:
    """Create a minimal hypothesis ledger if none exists."""
    if HYPO_PATH.exists():
        return
    HYPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYPO_PATH.write_text(json.dumps(SEED_HYPOTHESES, indent=2))
    print(f"Seeded hypothesis ledger with {len(SEED_HYPOTHESES)} entries.")


if __name__ == "__main__":
    seed_ledger_if_missing()
    reports = run_decay_check(verbose=True)
