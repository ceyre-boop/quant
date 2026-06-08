#!/usr/bin/env python3
"""
Close the Oracle loop for the futures sandbox.

The --auto / monitor path logs entries with exit=None, r_realized=None and never
backfills — so calibration + analysis are blind to your automated trades (violates
CLAUDE.md NON-NEGOTIABLE #2). This reconciler replays each open trade's bars forward
from its entry, determines the bracket outcome (T1 / STOPPED / EOD) with the SAME exit
logic as the replay, and backfills exit / exit_reason / r_realized into trade_log.jsonl.

Price-based so it runs tonight without IB. (When IB Gateway is up, real fills are more
accurate — a future enhancement can prefer bridge.fills() and fall back to this.)

Usage:
    python3.13 scripts/futures_reconcile.py                 # reconcile all open trades
    python3.13 scripts/futures_reconcile.py --dry-run       # show what would change
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.futures import bar_feed as bf            # noqa: E402
from sovereign.futures import scalp_strategy as strat   # noqa: E402

TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"


def _resolve(day_df, entry_ts: datetime, direction: str,
             entry: float, stop: float, target: float) -> tuple[float, str] | None:
    """Walk bars strictly after entry_ts; return (exit_price, reason) or None if still open."""
    after = day_df[day_df.index > entry_ts]
    if len(after) == 0:
        return None
    for j in range(len(after)):
        hi, lo = float(after["High"].iloc[j]), float(after["Low"].iloc[j])
        if direction == "LONG":
            if lo <= stop:
                return stop, "STOPPED"
            if hi >= target:
                return target, "T1_HIT"
        else:
            if hi >= stop:
                return stop, "STOPPED"
            if lo <= target:
                return target, "T1_HIT"
    # session ended without hitting either → close at last available bar
    return float(after["Close"].iloc[-1]), "EOD"


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill outcomes for futures auto trades")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TRADE_LOG.exists():
        print("No trade_log.jsonl yet — nothing to reconcile.")
        return

    records = [json.loads(l) for l in TRADE_LOG.read_text().splitlines() if l.strip()]
    bars_cache: dict[tuple[str, str], object] = {}
    closed = 0

    for rec in records:
        # only real, still-open trades with the levels we need
        if rec.get("exit") is not None or rec.get("r_realized") is not None:
            continue
        entry = rec.get("entry")
        stop = rec.get("stop")
        target = rec.get("target_1") or rec.get("target")
        direction = rec.get("direction")
        if entry is None or stop is None or target is None or direction not in ("LONG", "SHORT"):
            continue
        if not rec.get("size_contracts"):
            continue
        try:
            entry_ts = datetime.fromisoformat(rec["ts"])
        except Exception:
            continue
        inst = rec.get("instrument", "MES")
        day = entry_ts.astimezone(bf.ET).strftime("%Y-%m-%d")
        key = (inst, day)
        if key not in bars_cache:
            bars_cache[key] = bf.load_history(inst, source="yf", day=day, lookback="7d")
        day_df = bars_cache[key]
        if day_df is None or len(day_df) == 0:
            continue

        res = _resolve(day_df, entry_ts, direction, float(entry), float(stop), float(target))
        if res is None:
            continue
        exit_price, reason = res
        r = strat.compute_r(float(entry), float(stop), exit_price, direction)
        rec["exit"] = round(exit_price, 2)
        rec["exit_reason"] = reason
        rec["r_realized"] = r
        rec["reconciled_at"] = datetime.now(timezone.utc).isoformat()
        closed += 1
        if args.dry_run:
            print(f"  would close {inst} {direction} @ {entry} → {exit_price:.2f} "
                  f"({reason}, r={r:+.2f})")

    if closed and not args.dry_run:
        tmp = TRADE_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        tmp.replace(TRADE_LOG)

    print(f"\n  {'[dry-run] ' if args.dry_run else ''}Reconciled {closed} open trade(s).")
    if closed:
        print("  Next: python3 scripts/futures_analysis.py   (now sees the auto trades)")


if __name__ == "__main__":
    main()
