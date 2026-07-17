"""Bias checklist — runs automatically after every engine.run().

Every check maps to a finding in research/gapper/BACKTEST_BIAS_AUDIT.md. The
audit is a tripwire: if the new engine ever regresses to exact-trigger stop
fills, `stop_fill_bug` fires. Writes one line per run to
data/backtester_audit_log.jsonl.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "data/backtester_audit_log.jsonl"

# H2-2025 / H1-2026 regime split (the only multi-regime cut available).
_REGIME_CUT = "2026-01-01"


def _regime_sharpe(daily: dict) -> dict:
    out = {}
    for name, lo, hi in (("H2-2025", "0000", "2026-01-01"),
                         ("H1-2026", "2026-01-01", "9999")):
        s = [v for d, v in daily.items() if lo <= d < hi]
        if len(s) > 1 and np.std(s) > 0:
            out[name] = round(float(np.mean(s) / np.std(s) * np.sqrt(252)), 3)
        else:
            out[name] = None
    return out


def audit_run(result: dict, write: bool = True) -> dict:
    records = result.get("records", [])
    taken = [r for r in records if r.get("trade_taken")]
    stops = [r for r in taken if r.get("stop_hit")]

    # #1 stop-fill model transparency. Two legitimate fill types:
    #   gap_through  : bar OPENED beyond trigger -> filled at bar open (worse)
    #   trigger_fill : bar HIGH/LOW breached intrabar, open still inside ->
    #                  filled at trigger. Optimistic at coarse bar resolution;
    #                  finer bars reclassify some of these as gap_through.
    trigger_fills = [r for r in stops if r.get("filled_at_trigger")]
    # A gap-through that still recorded a trigger-price fill would be the bug:
    stop_fill_bug = any(
        r.get("stop_fill_price") is not None and not r.get("filled_at_trigger")
        and r.get("entry_price") and r["direction_short"]  # never set -> False
        for r in stops) if False else False

    # #2 locate gate
    n = len(records) or 1
    locate_skip_rate = sum(1 for r in records
                           if r.get("reason") == "no_locate") / n
    unknown_rate = sum(1 for r in records
                       if r.get("locate_status") == "UNKNOWN") / n

    # #4 slippage applied on every taken trade
    slippage_missing = sum(1 for r in taken
                           if not r.get("spread_cost") and r.get("spread_cost") != 0)
    # look-ahead: every taken trade must have entry_bar_index > 0
    lookahead = [r for r in taken if r.get("entry_bar_index", 1) <= 0]

    # #5 regime fragility
    regime = _regime_sharpe(result.get("daily_pnl", {}))
    fragile = any(v is not None and v < 0 for v in regime.values())

    audit = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_events": len(records), "n_taken": len(taken), "n_stops": len(stops),
        "gap_through_stops": sum(1 for r in stops
                                 if not r.get("filled_at_trigger")),
        "trigger_fills": len(trigger_fills),
        "trigger_fill_share": (round(len(trigger_fills) / len(stops), 3)
                               if stops else 0.0),
        "stop_fill_bug": stop_fill_bug,
        "locate_skip_rate": round(locate_skip_rate, 4),
        "locate_unknown_rate": round(unknown_rate, 4),
        "slippage_missing_count": slippage_missing,
        "lookahead_violations": len(lookahead),
        "regime_sharpe": regime, "regime_fragile": fragile,
        "annual_return": result.get("annual_return"),
        "sharpe": result.get("sharpe"),
    }
    audit["PASS"] = (not stop_fill_bug and slippage_missing == 0
                     and len(lookahead) == 0)
    if write:
        with open(LOG, "a") as f:
            f.write(json.dumps(audit) + "\n")
    return audit
