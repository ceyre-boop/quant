#!/usr/bin/env python3
"""
Close the Oracle loop for the futures sandbox.

The --auto / monitor path logs entries with exit=None, r_realized=None and never
backfills — so calibration + analysis are blind to your automated trades (violates
CLAUDE.md NON-NEGOTIABLE #2). This reconciler replays each open trade's bars forward
from its entry, determines the bracket outcome (T1 / STOPPED / EOD) with the SAME exit
logic as the replay, and backfills exit / exit_reason / r_realized into trade_log.jsonl.

Two modes:
  - default: price-replay (yfinance) for SIM trades only. A trade that placed a REAL IB
    bracket (its notes carry `order_id=`) is NEVER reconciled from yfinance — that would
    write a fake exit onto a real rep. Those are left OPEN until `--from-ib`.
  - `--from-ib`: connect IB Gateway, pull bridge.fills(), and close real IB trades from the
    ACTUAL paper fills (the authoritative outcome). [Needs Gateway up; verify live.]

Usage:
    python3.13 scripts/futures_reconcile.py                 # sim trades only (yfinance)
    python3.13 scripts/futures_reconcile.py --from-ib       # close real IB trades from fills
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
from sovereign.futures import reasoning as rsn          # noqa: E402

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


def _is_real_ib(rec: dict) -> bool:
    """True if this trade placed a real IB bracket (so its exit is an actual fill, not a
    yfinance replay). Keys off the order_id the monitor writes into notes."""
    if rec.get("bars_source") == "ib":
        return True
    notes = rec.get("notes") or ""
    return "order_id=" in notes and "dry-run" not in notes and "disconnected" not in notes


def _reconcile_from_ib(records: list[dict], dry_run: bool) -> int:
    """Close real IB trades from bridge.fills() (authoritative paper outcomes).
    Heuristic match: same symbol, exit-side, time after entry, unused fill. Verify live."""
    from sovereign.futures.ib_bridge import IBBridge
    bridge = IBBridge()
    bridge.connect()
    try:
        fills = bridge.fills()
    finally:
        bridge.disconnect()
    used = set()
    closed = 0
    for rec in records:
        if rec.get("exit") is not None or not _is_real_ib(rec):
            continue
        direction = rec.get("direction")
        inst = rec.get("instrument", "MES")
        exit_side = "SLD" if direction == "LONG" else "BOT"
        try:
            entry_ts = str(datetime.fromisoformat(rec["ts"]))
        except Exception:
            continue
        cand = [f for i, f in enumerate(fills) if i not in used
                and f["symbol"] == inst and f["side"] == exit_side and str(f["time"]) > entry_ts]
        if not cand:
            continue
        f = min(cand, key=lambda x: str(x["time"]))
        used.add(fills.index(f))
        exit_price = f["price"]
        entry, stop, target = float(rec["entry"]), float(rec["stop"]), float(rec.get("target_1") or rec.get("target"))
        fav = (exit_price >= target) if direction == "LONG" else (exit_price <= target)
        rec["exit"] = round(exit_price, 2)
        rec["exit_reason"] = "T1_HIT" if fav else "STOPPED_OR_MANUAL"
        rec["r_realized"] = strat.compute_r(entry, stop, exit_price, direction)
        rec["reconciled_at"] = datetime.now(timezone.utc).isoformat()
        rec["reconciled_via"] = "ib_fill"
        rec["exit_reasoning"] = rsn.exit_attribution(rec, {
            "exit_type": "TARGET" if fav else "STOP_LOSS", "exit_price": round(exit_price, 2),
            "r_realized": rec["r_realized"],
        })
        closed += 1
        if dry_run:
            print(f"  [ib] would close {inst} {direction} @ {entry} → {exit_price:.2f} (fill)")
    return closed


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill outcomes for futures auto trades")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-ib", action="store_true",
                    help="close real IB trades from bridge.fills() (Gateway must be up)")
    args = ap.parse_args()

    if not TRADE_LOG.exists():
        print("No trade_log.jsonl yet — nothing to reconcile.")
        return

    records = [json.loads(l) for l in TRADE_LOG.read_text().splitlines() if l.strip()]
    bars_cache: dict[tuple[str, str], object] = {}
    closed = 0
    skipped_ib = 0

    if args.from_ib:
        try:
            closed += _reconcile_from_ib(records, args.dry_run)
        except Exception as e:
            print(f"  {type(e).__name__}: {e} — is IB Gateway running on port 4002?")
            sys.exit(1)

    for rec in records:
        # only real, still-open trades with the levels we need
        if rec.get("exit") is not None or rec.get("r_realized") is not None:
            continue
        # NEVER reconcile a real IB rep from yfinance — leave it for --from-ib
        if _is_real_ib(rec):
            skipped_ib += 1
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
        # Learning agent: causal attribution at close (null-safe; templated from entry vs exit).
        _exit_type = {"STOPPED": "STOP_LOSS", "T1_HIT": "TARGET", "EOD": "TIME"}.get(reason, reason)
        rec["exit_reasoning"] = rsn.exit_attribution(rec, {
            "exit_type": _exit_type, "exit_price": round(exit_price, 2), "r_realized": r,
        })
        closed += 1
        if args.dry_run:
            print(f"  would close {inst} {direction} @ {entry} → {exit_price:.2f} "
                  f"({reason}, r={r:+.2f})")

    if closed and not args.dry_run:
        tmp = TRADE_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        tmp.replace(TRADE_LOG)

    print(f"\n  {'[dry-run] ' if args.dry_run else ''}Reconciled {closed} open trade(s).")
    if skipped_ib:
        print(f"  {skipped_ib} real IB trade(s) left OPEN — run `--from-ib` with Gateway up "
              f"to close them from actual fills (never from yfinance).")
    if closed:
        print("  Next: python3 scripts/futures_analysis.py   (now sees the auto trades)")


if __name__ == "__main__":
    main()
