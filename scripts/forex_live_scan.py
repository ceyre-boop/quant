#!/usr/bin/env python3
"""
Forex macro live scanner — the missing execution leg.
=====================================================

ForexSpecialist already GENERATES + logs tradeable forex candidates; nothing routed them
to OANDA (why G2 is 0/30). This wires that leg, on the OANDA *practice* account.

DEFAULT IS DRY-RUN: it runs the real gates (DailyReadiness + PropRiskManager) and logs
WOULD_PLACE — it does NOT call bridge.place_trade unless `--live` is passed. Flip to
live-practice later by adding `--live` to the launchd ProgramArguments (one line).

Design note: we do NOT route through DecisionChain.evaluate() — that method is DataFrame-
based and re-derives the trade with its stubbed (Q2-Q4) sizing, which would discard
ForexSpecialist's computed entry/stop/units/risk. Instead we apply the SAME real gates
DecisionChain uses (Q1 DailyReadiness, Q5 PropRiskManager) directly to the forex candidate.
PropRiskManager only READS account state, so it runs even in dry-run (makes the dry-run real).

Usage:
    python3 scripts/forex_live_scan.py            # DRY-RUN (default): logs WOULD_PLACE, places nothing
    python3 scripts/forex_live_scan.py --live     # live-practice: places orders on the OANDA practice account
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for lib in ("yfinance", "peewee", "urllib3", "requests", "oandapyV20"):
    logging.getLogger(lib).setLevel(logging.ERROR)

LOG = ROOT / "logs" / "forex_scan.log"
HEARTBEAT = ROOT / "logs" / ".heartbeat_forex_scan"
PROX_PATH = ROOT / "data" / "agent" / "forex_proximity.json"

_CONVICTION_ENTRY = 0.35   # CONVICTION_NEUTRAL_THRESHOLD from entry_engine.py:324


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(record: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _report_proximity(report: "ForexScanReport", verbose: bool = False) -> dict:
    """Compute proximity-to-trigger for all pairs; write PROX_PATH on every run."""
    prev: dict[str, float] = {}
    if PROX_PATH.exists():
        try:
            prev_data = json.loads(PROX_PATH.read_text())
            prev = {p["pair"]: p.get("rate_differential", 0.0)
                    for p in prev_data.get("pairs", [])}
        except Exception:
            pass

    all_signals = [c.entry_signal for c in report.tradeable] + list(report.skipped)

    rows = []
    for sig in sorted(all_signals, key=lambda s: s.macro_signal.conviction, reverse=True):
        ms = sig.macro_signal
        pct = min(100.0, (ms.conviction / _CONVICTION_ENTRY) * 100)
        trend = "OK" if ms.hurst > 0.55 else "BLOCKED"
        regime = "TRENDING" if ms.hurst > 0.55 else "RANGING"

        prev_diff = prev.get(sig.pair)
        if prev_diff is not None:
            diff_dir = "WIDENING" if abs(ms.rate_differential) > abs(prev_diff) else "NARROWING"
            diff_str = f"{diff_dir} ({prev_diff:.2f} → {ms.rate_differential:.2f})"
        else:
            diff_dir = "NEW"
            diff_str = "NEW (no previous scan)"

        est_days = None
        if pct >= 50 and diff_dir == "WIDENING":
            est_days = max(1, round((100.0 - pct) / 5))

        rows.append({
            "pair":                sig.pair,
            "pct_to_trigger":      round(pct, 1),
            "conviction":          round(ms.conviction, 3),
            "direction":           sig.direction,
            "trend":               trend,
            "regime":              regime,
            "hurst":               round(ms.hurst, 2),
            "rate_differential":   round(ms.rate_differential, 4),
            "differential_trend":  diff_dir,
            "est_days_to_trigger": est_days,
        })

        if verbose:
            flag = ">>>" if pct >= 80 and trend == "OK" else "   "
            print(f"  {flag} {sig.pair}: {pct:.0f}% to trigger | "
                  f"dir={sig.direction} trend={trend} regime={regime}")
            print(f"       rate_diff={ms.rate_differential:.3f} | "
                  f"conviction={ms.conviction:.3f} | {diff_str}")

    closest = max(rows, key=lambda r: r["pct_to_trigger"]) if rows else {}
    if closest.get("pct_to_trigger", 0) >= 80 and closest.get("trend") == "OK":
        est = closest.get("est_days_to_trigger")
        days_str = f", ~{est}d out" if est else ""
        verdict = (f"NO_TRADE_TODAY — {closest['pair']} closest at "
                   f"{closest['pct_to_trigger']:.0f}%{days_str}")
    else:
        verdict = "NO_TRADE_TODAY"

    PROX_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROX_PATH.write_text(json.dumps({
        "last_scan": datetime.now(timezone.utc).isoformat(),
        "pairs":     rows,
        "verdict":   verdict,
    }, indent=2))

    return {"rows": rows, "verdict": verdict}


def main() -> dict:
    ap = argparse.ArgumentParser(description="Forex macro live scanner")
    ap.add_argument("--live", action="store_true",
                    help="Place real orders on the OANDA PRACTICE account (default: dry-run).")
    ap.add_argument("--balance", type=float, default=100_000)
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-pair proximity-to-trigger breakdown.")
    args = ap.parse_args()
    dry_run = not args.live
    mode = "DRY-RUN" if dry_run else "LIVE-PRACTICE"
    ts = _now()

    # Heartbeat FIRST — every invocation, before any gate, so loop_health can tell
    # "launchd stopped firing me" from "ran fine, no signals".
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(ts)

    print(f"[{ts}] FOREX_LIVE_SCAN ({mode}) starting")

    # ── Gate 1: DailyReadiness ────────────────────────────────────────────
    from sovereign.oracle.daily_readiness import DailyReadiness
    readiness = DailyReadiness().assess()
    print(f"  Readiness: {readiness.status} — {readiness.reason}")
    if readiness.status == "SIT":
        _log({"timestamp": ts, "mode": mode, "verdict": "SIT", "reason": readiness.reason})
        print("  SIT — no new trades today.")
        return {"verdict": "SIT"}

    # ── Generate live forex candidates (reuse the existing scanner) ───────
    from sovereign.forex.forex_specialist import ForexSpecialist
    report = ForexSpecialist(account_balance=args.balance).run()
    tradeable = report.tradeable
    print(f"  Generated {len(tradeable)} tradeable candidate(s).")

    prox = _report_proximity(report, verbose=args.verbose)
    print(f"  Proximity: {prox['verdict']}")

    if not tradeable:
        _log({"timestamp": ts, "mode": mode, "verdict": "NO_SIGNALS",
              "reason": "No pair meets forex macro entry criteria today.",
              "readiness": readiness.status})
        print("  NO_SIGNALS — nothing qualifies today (valid output, not a failure).")
        return {"verdict": "NO_SIGNALS"}

    # ── Real risk gate + (dry-run) WOULD_PLACE / (live) place ─────────────
    from sovereign.execution.oanda_bridge import OandaBridge, to_oanda_pair
    from sovereign.risk.prop_risk_manager import PropRiskManager
    bridge = OandaBridge()                      # READ-only here unless we place
    prop = PropRiskManager(bridge)
    results = []

    from sovereign.risk import engine_adapter
    try:
        _equity = bridge.get_account_balance()
    except Exception:
        _equity = None

    for c in tradeable:
        s, p = c.entry_signal, c.position
        pair, direction = s.pair, s.direction
        # Dynamic Risk Engine is the SOLE sizing authority. Map the forex conviction onto a grade
        # base, then let the cascade govern. ForexSpecialist no longer sizes around the engine.
        _decision = engine_adapter.size(pair, direction, s.entry_price, s.stop_price,
                                        grade=engine_adapter.grade_from_risk(p.risk_pct),
                                        equity=_equity)
        risk_pct = _decision.final_risk_pct
        if readiness.status == "REDUCE":
            risk_pct = round(risk_pct * 0.5, 5)  # readiness only ever reduces further

        risk_check = prop.check_trade_allowed(pair, direction, risk_pct)
        base = {"timestamp": ts, "mode": mode, "pair": pair, "direction": direction,
                "entry": s.entry_price, "stop": s.stop_price, "tp1": s.t1,
                "risk_pct": risk_pct, "units": p.units, "score": s.score,
                "macro_conviction": s.macro_conviction}

        if not risk_check.allowed:
            rec = {**base, "verdict": "DENIED", "reason": risk_check.reason}
            print(f"    DENIED {pair} {direction}: {risk_check.reason}")
        elif dry_run:
            rec = {**base, "verdict": "WOULD_PLACE", "reason": "passed readiness + PropRiskManager"}
            print(f"    WOULD_PLACE {pair} {direction}  entry={s.entry_price:.5f} "
                  f"stop={s.stop_price:.5f} tp1={s.t1:.5f} risk={risk_pct:.2%} units={p.units:,.0f}")
        else:
            oanda_pair = to_oanda_pair(pair)
            units = bridge.compute_units(oanda_pair, s.entry_price, s.stop_price, risk_pct)
            if units == 0:
                rec = {**base, "verdict": "NO_TRADE", "reason": "computed units = 0"}
            else:
                placed = bridge.place_trade(oanda_pair, direction, units, s.stop_price, s.t1)
                rec = {**base, "verdict": "PLACED", "oanda_result": placed}
                print(f"    PLACED {pair} {direction} units={units:,.0f}")
        _log(rec)
        results.append(rec)

    print(f"[FOREX_LIVE_SCAN complete] {mode}: "
          f"{sum(1 for r in results if r['verdict'] in ('WOULD_PLACE','PLACED'))} actionable / "
          f"{len(results)} evaluated")
    return {"verdict": "COMPLETE", "results": results}


if __name__ == "__main__":
    main()
