#!/usr/bin/env python3
"""build_system_health.py — the writer for the conscience (measurement layer).

Reads each strategy's live journals/ledgers (JSON/JSONL, never imports), runs the
measurements in alta_platform/measurement.py, and writes one canonical verdict:

    data/agent/system_health_verdict.json

The conscience answers "is the process being followed, and is the edge still true
on live data?" Its headline is a per-strategy + portfolio KILL SWITCH
(TRADE / REDUCE / HALT) that fails SAFE. Read via alta_platform.health_client.get_health().

DISCIPLINE (non-negotiable):
  * Never raises. A crashed writer that leaves a stale verdict claiming freshness
    is worse than one that writes DEGRADED with a reason.
  * Every measurement carries a status; INSUFFICIENT_DATA is first-class.
  * The kill switch fails SAFE: missing/stale/thin data → HALT/REDUCE, never TRADE.
  * No hardcoded thresholds — read from config/parameters.yml (platform.health).
    New keys were logged to data/agent/param_change_log.jsonl before use.

Isolation: this script + alta_platform/ import neither ict/ nor sovereign/. It only
reads their JSON/JSONL outputs. The ict/<->sovereign wall is never touched.

Usage:
    python3 scripts/build_system_health.py            # write the verdict
    python3 scripts/build_system_health.py --dry-run  # print, write nothing
    python3 scripts/build_system_health.py --check     # exit 1 if degraded
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from alta_platform import measurement as m  # noqa: E402 (after sys.path setup)

DATA = REPO / "data"
CONFIG = REPO / "config" / "parameters.yml"
OUT = DATA / "agent" / "system_health_verdict.json"

# The ICT causal-chain setup ledger these sections need (not written yet).
ICT_CAUSAL_LEDGER = "data/agent/ict_causal_chain.jsonl (ICT setup ledger, Layer 8 — not written yet)"


def build() -> dict:
    now = datetime.now(timezone.utc)
    try:
        config = m.load_yaml(CONFIG)
    except Exception as exc:
        config = {}
        config_error = f"config unreadable: {exc}"
    else:
        config_error = None

    # --- Portfolio integrity + kill switch (the headline safety organ) ---
    portfolio = m.read_portfolio_integrity(config, DATA)
    pf_kill = portfolio["kill_switch"]

    strategies: dict[str, dict] = {}

    # --- undertow_gapper: the one strategy with a live shadow ---
    undertow = m.StrategyHealth("undertow_gapper")
    undertow.edge_health = m.edge_health_undertow(config, DATA)
    undertow.process_adherence = m.process_adherence_unavailable(ICT_CAUSAL_LEDGER)
    undertow.forecast_vs_execution = m.forecast_vs_execution_unavailable(ICT_CAUSAL_LEDGER)
    ks, why = m.combine_kill_switch(
        pf_kill, undertow.edge_health.status, undertow.edge_health.divergence_flag
    )
    undertow.kill_switch = ks
    undertow.reason = why
    strategies["undertow_gapper"] = undertow.to_dict()

    # --- ict_equities: no live edge feed yet; process/forecast need the ledger ---
    ict = m.StrategyHealth("ict_equities")
    ict.edge_health = m.EdgeHealth(
        status=m.UNAVAILABLE,
        reason=(
            "no ict_equities live edge ledger on disk; live-vs-backtest expectancy "
            "cannot be computed. Section fills once ICT logs closed-trade outcomes."
        ),
        source="(none yet)",
    )
    ict.process_adherence = m.process_adherence_unavailable(ICT_CAUSAL_LEDGER)
    ict.forecast_vs_execution = m.forecast_vs_execution_unavailable(ICT_CAUSAL_LEDGER)
    ks, why = m.combine_kill_switch(pf_kill, ict.edge_health.status, ict.edge_health.divergence_flag)
    ict.kill_switch = ks
    ict.reason = why
    strategies["ict_equities"] = ict.to_dict()

    # --- carry: FROZEN (do not wire) — reported honestly, no live edge feed ---
    carry = m.StrategyHealth("carry")
    carry.edge_health = m.EdgeHealth(
        status=m.UNAVAILABLE,
        reason="carry edge-health feed not wired (execution path frozen until 2026-07-28).",
        source="(frozen)",
    )
    carry.process_adherence = m.process_adherence_unavailable(ICT_CAUSAL_LEDGER)
    carry.forecast_vs_execution = m.forecast_vs_execution_unavailable(ICT_CAUSAL_LEDGER)
    ks, why = m.combine_kill_switch(pf_kill, carry.edge_health.status, carry.edge_health.divergence_flag)
    carry.kill_switch = ks
    carry.reason = why
    strategies["carry"] = carry.to_dict()

    # --- Top-level status ---
    if config_error or portfolio.get("data_integrity") == "FAIL":
        top = "DEGRADED"
    elif portfolio.get("drawdown_breaker_status") == m.UNAVAILABLE:
        top = "DEGRADED"
    else:
        top = "OK"

    reason_bits = [
        f"portfolio:{portfolio.get('data_integrity')}/{portfolio.get('kill_switch')}",
    ]
    reason_bits += [f"{n}:{s['kill_switch']}" for n, s in strategies.items()]
    if config_error:
        reason_bits.append(config_error)

    return {
        "generated_at": now.isoformat(),
        "status": top,
        "status_reason": "; ".join(reason_bits),
        "strategies": strategies,
        "portfolio": portfolio,
        "_meta": {
            "writer": "scripts/build_system_health.py",
            "spec": "specs/measurement_layer.md",
            "note": (
                "The conscience. Neutral alta_platform/ layer — reads JSON/JSONL outputs, "
                "imports neither ict/ nor sovereign/. Kill switch fails SAFE."
            ),
        },
    }


def write(verdict: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)
        fh.write("\n")
    tmp.replace(OUT)  # never leave a half-written verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the system health verdict (the conscience).")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    ap.add_argument("--check", action="store_true", help="exit 1 if verdict is degraded")
    args = ap.parse_args(argv)

    try:
        verdict = build()
    except Exception:  # never raise out of the writer
        verdict = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "DEGRADED",
            "status_reason": "writer crashed: " + traceback.format_exc().splitlines()[-1],
            "strategies": {},
            "portfolio": {"status": "UNAVAILABLE", "reason": "writer crashed", "kill_switch": "HALT"},
        }

    if args.dry_run:
        print(json.dumps(verdict, indent=2))
    else:
        write(verdict)
        print(f"wrote {OUT.relative_to(REPO)} — status {verdict['status']}: {verdict['status_reason']}")

    if args.check and verdict.get("status") != "OK":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
