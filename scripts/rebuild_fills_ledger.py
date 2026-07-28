#!/usr/bin/env python3
"""Rebuild data/ledger/oanda_fills.jsonl from the broker. TICK-092.

Why this exists
---------------
`data/ledger/oanda_fills.jsonl` is read by four call sites and was written by NONE:

    sovereign/oracle/pulse_check.py:511      (FILLS_PATH — outcome backfill)
    sovereign/oracle/pulse_check.py:475      (_avg_r_from_fills)
    scripts/backfill_decision_records.py:56  (decision-record recovery)
    scripts/proof_of_life.py:25              (last actual fill)

With it empty, `pulse_check._backfill_decision_outcomes()` resolved `fill = None` for every
closed trade → `stop = 0.0` → the `stop <= 0` branch skipped the trade *before* the attempt
counter incremented, so the stall alarm never fired and the outcome loop reported
`matched=0 attempted=0` — indistinguishable from healthy. Every closed trade drained silently
and the Oracle learning loop stayed empty (NON-NEGOTIABLE #2).

Design decision: reconstruct from the BROKER, do not instrument order placement. OANDA is the
source of truth, this recovers history rather than only capturing new trades, and it touches no
execution-path file — so it needs no unlock against the standing shadow/execution freeze.

Contract: audit/fills_ledger_spec.md (F3 schema + non-empty, F4 idempotence, F6 read-only).

READ-ONLY against the broker. Calls only TradesList / TradeDetails. It must never call
place_trade, close_trade or set_stop — enforced by tests/test_rebuild_fills_ledger.py.

Usage
-----
    python3 scripts/rebuild_fills_ledger.py            # merge into the ledger
    python3 scripts/rebuild_fills_ledger.py --dry-run  # report only, write nothing
    python3 scripts/rebuild_fills_ledger.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILLS_PATH = ROOT / "data" / "ledger" / "oanda_fills.jsonl"

# Field names are dictated by the four existing readers — reuse verbatim (spec F3).
# Inventing a schema here would silently break backfill_decision_records.py.
REQUIRED_FIELDS = ("trade_id", "pair", "direction", "fill_price", "stop_price", "timestamp")

log = logging.getLogger("rebuild_fills")


def _f(value: Any) -> float:
    """Best-effort float; 0.0 on anything unparseable. Never raises."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_stop(trade: dict, bridge) -> float:
    """Stop price for a trade, from the list record or a detail fetch.

    Returns 0.0 when the broker has no stop on record. A trade without a stop is OMITTED
    from the ledger rather than written with a placeholder (spec F3) — a fabricated stop
    produces a fabricated R, which is exactly what the Oracle must never learn from.
    """
    stop = _f((trade.get("stopLossOrder") or {}).get("price"))
    if stop > 0:
        return stop
    # TradesList can omit the linked orders; TradeDetails carries them.
    tid = str(trade.get("id") or trade.get("tradeID") or "")
    if not tid:
        return 0.0
    detail = bridge.get_trade(tid)
    if not detail:
        return 0.0
    return _f((detail.get("stopLossOrder") or {}).get("price"))


def _to_row(trade: dict, bridge) -> Optional[dict]:
    """Map an OANDA trade to a fills-ledger row, or None if it can't be scored."""
    tid = str(trade.get("id") or trade.get("tradeID") or "").strip()
    pair = str(trade.get("instrument") or "").strip()
    fill_price = _f(trade.get("price"))
    units = _f(trade.get("initialUnits") or trade.get("currentUnits"))
    timestamp = str(trade.get("openTime") or "").strip()

    if not tid or not pair or not timestamp or fill_price <= 0:
        log.info("skip trade %r — incomplete broker record (pair=%r ts=%r price=%s)",
                 tid, pair, timestamp, fill_price)
        return None

    stop_price = _resolve_stop(trade, bridge)
    if stop_price <= 0:
        log.info("skip %s trade %s — broker has no stop on record; omitted, not faked", pair, tid)
        return None

    row = {
        "trade_id": tid,
        "order_id": str(trade.get("id") or ""),
        "pair": pair,                                    # OANDA underscore format
        "direction": "LONG" if units >= 0 else "SHORT",
        "fill_price": fill_price,
        "stop_price": stop_price,
        "timestamp": timestamp,                          # fill time; exit side keys on openTime
        "units": units,
        "state": str(trade.get("state") or ""),
        "source": "rebuild_fills_ledger",
    }
    tp = _f((trade.get("takeProfitOrder") or {}).get("price"))
    if tp > 0:
        row["tp1_price"] = tp

    # r_realized is only meaningful once the trade has closed.
    close_px = _f(trade.get("averageClosePrice"))
    if close_px > 0:
        risk = abs(fill_price - stop_price)
        if risk > 0:
            direction = 1.0 if units >= 0 else -1.0
            row["r_realized"] = round((close_px - fill_price) / risk * direction, 3)
            row["close_price"] = close_px
            row["close_time"] = str(trade.get("closeTime") or "")
            row["realized_pl"] = _f(trade.get("realizedPL"))
    return row


def _load_existing() -> dict[str, dict]:
    """Existing ledger keyed by trade_id. Corrupt lines are skipped, not fatal."""
    if not FILLS_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    for line in FILLS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.warning("skipping corrupt ledger line")
            continue
        tid = str(row.get("trade_id", "")).strip()
        if tid:
            out[tid] = row
    return out


def rebuild(limit: int = 500, dry_run: bool = False) -> dict:
    """Merge broker state into the fills ledger. Idempotent, keyed on trade_id (spec F4)."""
    from sovereign.execution.oanda_bridge import OandaBridge

    bridge = OandaBridge()
    existing = _load_existing()
    before = len(existing)

    trades: list[dict] = []
    try:
        trades.extend(bridge.get_closed_trades(limit=limit))
    except Exception as exc:
        log.error("cannot fetch closed trades: %s", exc)
    try:
        trades.extend(bridge.get_open_trades())
    except Exception as exc:
        log.error("cannot fetch open trades: %s", exc)

    if not trades:
        # No silent mocking: an empty broker response is reported, never treated as success.
        log.error("broker returned zero trades — cannot rebuild. Check OANDA credentials "
                  "and account id; refusing to report a green rebuild on no data.")
        return {"ok": False, "reason": "no_trades_from_broker",
                "before": before, "after": before, "added": 0, "skipped_no_stop": 0}

    added = updated = skipped = 0
    for trade in trades:
        row = _to_row(trade, bridge)
        if row is None:
            skipped += 1
            continue
        tid = row["trade_id"]
        # NOTE: pairs the invariant guard treats as forbidden (AUD_NZD, USD_CAD) are written
        # faithfully. Filtering them here would hide exactly the rogue writes audit/
        # invariants_spec.md I2/I3 exist to catch.
        if tid in existing:
            if existing[tid] != row:
                existing[tid] = row
                updated += 1
        else:
            existing[tid] = row
            added += 1

    rows = sorted(existing.values(), key=lambda r: str(r.get("timestamp", "")))

    if not dry_run:
        FILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = FILLS_PATH.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(FILLS_PATH)  # atomic — never leave a half-written ledger

    return {"ok": True, "before": before, "after": len(rows), "added": added,
            "updated": updated, "skipped_no_stop": skipped,
            "examined": len(trades), "dry_run": dry_run}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=500, help="closed trades to request")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = rebuild(limit=args.limit, dry_run=args.dry_run)

    if not res.get("ok"):
        print(f"REBUILD FAILED — {res.get('reason')}")
        return 1

    print(f"fills ledger: {res['before']} → {res['after']} rows "
          f"(+{res['added']} new, {res['updated']} updated, "
          f"{res['skipped_no_stop']} skipped for no stop, "
          f"{res['examined']} examined){' [DRY RUN]' if res['dry_run'] else ''}")
    if res["skipped_no_stop"]:
        print(f"  {res['skipped_no_stop']} trades carry no broker stop and were OMITTED, "
              f"not written with a placeholder — they cannot be scored into an R.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
