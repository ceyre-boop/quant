#!/usr/bin/env python3
"""
Backfill decision records for OANDA fills that bypassed decision logging.

THE PROBLEM (CLAUDE.md NON-NEGOTIABLE #2 — close the Oracle loop):
Every OANDA fill must have a matching entry-side decision record, so that the
exit-side reconcile in sovereign/oracle/pulse_check.py::_backfill_decision_outcomes
can call update_outcome() on it when the trade closes. Some fills were placed by
paths that never called log_forex_decision() (e.g. the stray fvg_express job that
traded AUD_NZD/USD_JPY at ~03:0X UTC without logging). Those fills have no entry
record, so their outcomes can never be attributed — Oracle's sample is
survivorship-biased and the loop silently never closes on the ENTRY side.

THE FIX (this script — the durable, entry-side counterpart to pulse_check):
For every fill in data/ledger/oanda_fills.jsonl, check whether a decision record
already exists. If not, create a minimal one retroactively so the loop can close,
regardless of which path placed the trade.

MATCHING (why not "within 5s"):
Decision records carry NO trade_id (schema has none). Forex decisions are logged at
SIGNAL time, but the fill lands later — sometimes ~50 min later (e.g. trade 105:
signal 12:43 → fill 13:34). A tight timestamp-only window would wrongly flag those
legit fills as unmatched and create DUPLICATE records. So we match on
normalized-pair + same-UTC-day (mirrors decision_logger.update_outcome's Tier-2
forex fallback; forex fires <=~1 trade per pair per day), with a tight
cross-midnight proximity fallback.

IDEMPOTENT: backfilled records are written with system="FOREX" and
entry_timestamp = fill time, so on any re-run each fill matches its own backfill
record (same pair + same day) and is skipped. Existing records are never rewritten.

Matchable by the exit side: pulse_check calls
update_outcome(pair=..., entry_timestamp=<openTime>, system="FOREX"), which requires
the record to be system="FOREX", still OPEN (outcome=None), same normalized pair,
and same-date. Backfilled records satisfy all four.

Usage:
    python3 scripts/backfill_decision_records.py            # apply
    python3 scripts/backfill_decision_records.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Repo-root anchored so it runs from any cwd (paths are relative in this repo).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.intelligence.decision_logger import DecisionRecord, _norm_pair  # noqa: E402

FILLS_PATH = ROOT / "data" / "ledger" / "oanda_fills.jsonl"
LOG_DIR = ROOT / "data" / "decision_logs"

# Cross-midnight proximity fallback (seconds). Same-UTC-day is the primary key;
# this only rescues a signal/fill pair that straddles midnight.
PROXIMITY_SECONDS = 600

# An "insane stop" (risk > 50% of entry, or non-positive) marks a test/proof-of-life
# fill — same heuristic pulse_check uses to refuse fabricating an R-multiple. We still
# backfill it (spec: any unmatched fill), but tag it so Oracle can discount it.
INSANE_RISK_FRACTION = 0.5


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).strip())
    except Exception:
        try:
            return datetime.fromisoformat(str(ts).strip()[:19])
        except Exception:
            return None


def _load_fills() -> list[dict]:
    if not FILLS_PATH.exists():
        return []
    out = []
    for line in FILLS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _load_all_decisions() -> list[dict]:
    """Snapshot every decision record across all monthly files (index, not mutated)."""
    out = []
    for path in sorted(LOG_DIR.glob("decisions_*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _has_decision(fill: dict, decisions: list[dict]) -> bool:
    """True if any decision record already covers this fill.

    Key: normalized pair AND (same UTC calendar day OR within PROXIMITY_SECONDS).
    """
    fpair = _norm_pair(fill.get("pair"))
    fdt = _parse_ts(fill.get("timestamp"))
    if fdt is None:
        return False
    ftid = str(fill.get("trade_id", "")).strip()
    for rec in decisions:
        if _norm_pair(rec.get("pair")) != fpair:
            continue
        # Future-proof: if a record ever carries the trade_id, an exact id match wins.
        if ftid and str(rec.get("trade_id", "")).strip() == ftid:
            return True
        rdt = _parse_ts(rec.get("entry_timestamp"))
        if rdt is None:
            continue
        if fdt.date() == rdt.date():
            return True
        if abs((fdt - rdt).total_seconds()) <= PROXIMITY_SECONDS:
            return True
    return False


def _is_test_fill(fill: dict) -> bool:
    try:
        entry = float(fill.get("fill_price") or 0.0)
        stop = float(fill.get("stop_price") or 0.0)
    except Exception:
        return True
    if entry <= 0 or stop <= 0:
        return True
    return abs(entry - stop) > INSANE_RISK_FRACTION * entry


def _build_record(fill: dict) -> dict:
    """Minimal decision record for an unmatched fill, schema-compatible + provenance."""
    entry = fill.get("fill_price")
    rec = DecisionRecord(
        entry_timestamp=str(fill.get("timestamp")),  # fill time == openTime the exit side keys on
        system="FOREX",                               # OANDA book is the FOREX book (pulse_check line 623)
        pair=str(fill.get("pair")),                   # OANDA format; _norm_pair reconciles venues
        direction=str(fill.get("direction")),
        entry_level=float(entry) if entry is not None else None,
        stop_loss=(float(fill["stop_price"]) if fill.get("stop_price") is not None else None),
        tp1=(float(fill["tp1_price"]) if fill.get("tp1_price") is not None else None),
        tp2=None,
        signal_layers_active=[],
        grade=None,
        session=None,
        score=None,
        vix_at_entry=None,
        rate_differential_zscore=None,
        cot_percentile=None,
        library_match=None,
        commitment_score=None,
        bars_since_signal=None,
        adr_pct_used=None,
        risk_pct=None,
        risk_dollars=None,
        why_this_size="backfilled from OANDA fill — original sizing not logged",
        why_this_trade="RECONSTRUCTED from fills ledger; entry path did not call log_forex_decision()",
        outcome=None,  # OPEN — required so the exit-side update_outcome can match it
    )
    d = asdict(rec)
    # Provenance (extra top-level keys; DecisionRecord ignores them, Oracle can read them).
    d["source"] = "fills_backfill"
    d["strategy"] = "unknown"
    d["trade_id"] = fill.get("trade_id")
    d["order_id"] = fill.get("order_id")
    d["backfilled_at"] = datetime.now().astimezone().isoformat()
    d["test_fill"] = _is_test_fill(fill)
    return d


def _month_path(ts: str) -> Path:
    dt = _parse_ts(ts)
    month = dt.strftime("%Y_%m") if dt else datetime.now().strftime("%Y_%m")
    return LOG_DIR / f"decisions_{month}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill decision records for unmatched OANDA fills")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    args = ap.parse_args()

    fills = _load_fills()
    decisions = _load_all_decisions()  # snapshot BEFORE backfill → one record per unmatched fill

    matched, backfilled = [], []
    for fill in fills:
        if _has_decision(fill, decisions):
            matched.append(fill)
        else:
            backfilled.append(fill)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for fill in backfilled:
        rec = _build_record(fill)
        if not args.dry_run:
            with _month_path(fill.get("timestamp")).open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    # ── Summary ──
    print(f"{'DRY-RUN ' if args.dry_run else ''}fills-ledger -> decision-record backfill")
    print(f"  {len(fills)} fills, {len(matched)} matched, {len(backfilled)} backfilled")
    if backfilled:
        print("  backfilled (had no decision record):")
        for fill in backfilled:
            tag = " [TEST]" if _is_test_fill(fill) else ""
            print(f"    trade {str(fill.get('trade_id')):>4}  {fill.get('pair'):<8} "
                  f"{fill.get('direction'):<6} {fill.get('timestamp')}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
