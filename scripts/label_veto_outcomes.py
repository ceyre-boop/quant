#!/usr/bin/env python3
"""
scripts/label_veto_outcomes.py
==============================
Offline outcome-labeler for the ICT veto ledger.

PROBLEM
-------
The ICT veto ledger (``data/ledger/ict_veto_ledger_YYYY_MM.jsonl``) records every
signal the pipeline rejects, but 100% of records carry ``outcome=None`` — the
write path never gets a retroactive label.  Without labels the "vetoes were
correct" belief is unfalsifiable and the meta-labeling training set is empty.

This script supplies the missing label by replaying historical price.

For every rejected record it asks the falsifiable question:

    "Had we taken the trade in the setup's direction, would price have moved
     +1R in our favour within the next N hours (default 4)?"

  * favourable +1R reached first  -> FALSE_NEGATIVE  (we wrongly rejected a winner)
  * adverse -1R reached first, or
    price never reaches +1R (flat) -> TRUE_NEGATIVE   (correct to reject)

R (the risk unit, in price terms) is the record's stop distance
``|entry_level - stop|`` when present, else a 0.5% default of the reference
price.  Direction comes from the record's ``signal`` (LONG/SHORT) when present,
otherwise it is inferred from the ICT setup in ``confirmations``
(BULLISH_SWEEP -> LONG, BEARISH_SWEEP -> SHORT) or from ``veto_reason``.
Records with no derivable direction or no price coverage are emitted as
``UNLABELABLE`` so every input row is accounted for.

Price source: yfinance 15-minute bars for ``<PAIR>=X`` (the feed the system
already uses).  First-touch is evaluated on bar High/Low.

THIS IS RESEARCH DATA ONLY.  It reads the ledger and writes a *new* file
``data/ledger/ict_veto_outcomes_YYYY_MM.jsonl``.  It never mutates the source
ledger and never touches live trading logic.

Usage
-----
    python3 scripts/label_veto_outcomes.py                 # May + June 2026
    python3 scripts/label_veto_outcomes.py --month 2026_06
    python3 scripts/label_veto_outcomes.py --lookahead-hours 4 --default-r-pct 0.005
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

LEDGER_ROOT = Path("data/ledger")
DEFAULT_MONTHS = ["2026_05", "2026_06"]
DEFAULT_LOOKAHEAD_HOURS = 4
DEFAULT_R_PCT = 0.005  # 0.5% of reference price when no stop distance is available


# ── price data ──────────────────────────────────────────────────────────────── #

class PriceBook:
    """Lazily downloads and caches one 15m bar frame per pair, then slices it."""

    def __init__(self, interval: str = "15m") -> None:
        import yfinance as yf  # local import keeps --help fast and dependency-clear
        self._yf = yf
        self._interval = interval
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def _symbol(pair: str) -> str:
        return f"{pair.upper()}=X"

    def _frame(self, pair: str, start: str, end: str):
        key = pair.upper()
        if key not in self._cache:
            df = self._yf.download(
                self._symbol(pair), start=start, end=end, interval=self._interval,
                progress=False, auto_adjust=False,
            )
            if df is not None and not df.empty:
                # flatten the (field, ticker) MultiIndex yfinance returns for single symbols
                if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
            self._cache[key] = df
        return self._cache[key]

    def window(self, pair: str, start_ts: datetime, hours: int):
        """Bars strictly after start_ts, up to start_ts + hours (inclusive)."""
        # one frame spans the whole study range; pad generously for slicing
        frame = self._frame(pair, "2026-05-20", "2026-07-02")
        if frame is None or frame.empty:
            return None
        end_ts = start_ts + timedelta(hours=hours)
        mask = (frame.index > start_ts) & (frame.index <= end_ts)
        win = frame.loc[mask]
        return win if not win.empty else None

    def price_at(self, pair: str, ts: datetime):
        """Close of the most recent bar at or before ts (entry proxy)."""
        frame = self._frame(pair, "2026-05-20", "2026-07-02")
        if frame is None or frame.empty:
            return None
        prior = frame.loc[frame.index <= ts]
        if prior.empty:
            return None
        return float(prior["Close"].iloc[-1])


# ── per-record derivation ───────────────────────────────────────────────────── #

def derive_direction(rec: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Return (direction, source). source in {given, sweep, reason, none}."""
    sig = rec.get("signal")
    if sig in ("LONG", "SHORT"):
        return sig, "given"
    conf = " ".join(rec.get("confirmations") or [])
    if "BULLISH_SWEEP" in conf:
        return "LONG", "sweep"
    if "BEARISH_SWEEP" in conf:
        return "SHORT", "sweep"
    reason = (rec.get("veto_reason") or "").lower()
    if "shorting" in reason:
        return "SHORT", "reason"
    if "longing" in reason:
        return "LONG", "reason"
    return None, "none"


def derive_entry_and_r(
    rec: Dict[str, Any], book: PriceBook, ts: datetime, default_r_pct: float
) -> Tuple[Optional[float], Optional[float], str]:
    """Return (entry_ref, r_price, r_source). r_source in {stop, default_pct}."""
    entry = rec.get("entry_level")
    stop = rec.get("stop")
    if entry is None:
        entry = book.price_at(rec.get("pair", ""), ts)
    if entry is None:
        return None, None, "no_price"
    if stop is not None and entry is not None and abs(entry - stop) > 0:
        return float(entry), abs(float(entry) - float(stop)), "stop"
    return float(entry), float(entry) * default_r_pct, "default_pct"


def first_touch_label(
    win, direction: str, entry: float, r_price: float, entry_is_limit: bool
) -> Tuple[str, str, float]:
    """Walk the window bar-by-bar; return (label, first_touch, outcome_r).

    label in {FALSE_NEGATIVE, TRUE_NEGATIVE, NO_FILL}.
    first_touch in {favorable, adverse, none, no_fill}.

    When ``entry_is_limit`` the entry_level is a resting limit order: the trade
    only exists if price first trades *to* that level within the window.  A LONG
    limit sits below market and fills when Low <= entry; a SHORT limit sits above
    market and fills when High >= entry.  Measuring +/-1R excursion from a level
    price never reached would invent fills that never happened (the original bug
    this labeler was built to avoid).  Market-entry proxies (entry = price at
    signal time) fill immediately, so ``entry_is_limit`` is False for those.
    """
    up = entry + r_price       # +1R target (favourable for LONG, adverse for SHORT)
    dn = entry - r_price       # -1R target
    filled = not entry_is_limit
    for _, bar in win.iterrows():
        hi, lo = float(bar["High"]), float(bar["Low"])
        if not filled:
            if (direction == "LONG" and lo <= entry) or (direction == "SHORT" and hi >= entry):
                filled = True
            else:
                continue  # limit not yet reached — no exposure this bar
        if direction == "LONG":
            hit_fav, hit_adv = hi >= up, lo <= dn
        else:  # SHORT
            hit_fav, hit_adv = lo <= dn, hi >= up
        if hit_fav and hit_adv:
            # both inside one bar — ambiguous; treat as adverse (conservative: stop-first)
            return "TRUE_NEGATIVE", "adverse", -1.0
        if hit_fav:
            return "FALSE_NEGATIVE", "favorable", 1.0
        if hit_adv:
            return "TRUE_NEGATIVE", "adverse", -1.0
    if not filled:
        # resting limit never reached within the window -> no trade, veto was moot
        return "NO_FILL", "no_fill", 0.0
    # filled but neither target reached -> realised fractional R at window end
    last_close = float(win["Close"].iloc[-1])
    realized = (last_close - entry) / r_price
    if direction == "SHORT":
        realized = -realized
    return "TRUE_NEGATIVE", "none", round(realized, 3)


# ── main labeling pass ──────────────────────────────────────────────────────── #

def load_shard(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def parse_ts(raw: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(raw)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def label_month(
    month: str, book: PriceBook, lookahead_hours: int, default_r_pct: float
) -> Dict[str, Any]:
    src = LEDGER_ROOT / f"ict_veto_ledger_{month}.jsonl"
    if not src.exists():
        return {"month": month, "skipped": "no source shard", "records": 0}

    records = load_shard(src)
    out_path = LEDGER_ROOT / f"ict_veto_outcomes_{month}.jsonl"
    labeled_rows: List[Dict[str, Any]] = []

    counts = Counter()
    by_stage = defaultdict(Counter)
    by_dir_source = defaultdict(Counter)
    unlabelable_reasons = Counter()

    for rec in records:
        ts = parse_ts(rec.get("timestamp", ""))
        pair = rec.get("pair", "")
        stage = rec.get("veto_stage", "unknown")
        direction, dir_source = derive_direction(rec)

        row = dict(rec)  # preserve the full original record
        row["lookahead_hours"] = lookahead_hours
        row["labeled_at"] = datetime.now(timezone.utc).isoformat()
        row["direction_used"] = direction
        row["direction_source"] = dir_source

        if ts is None:
            label, reason = "UNLABELABLE", "bad_timestamp"
        elif direction is None:
            label, reason = "UNLABELABLE", "no_direction"
        else:
            entry, r_price, r_source = derive_entry_and_r(rec, book, ts, default_r_pct)
            if entry is None or r_price is None or r_price <= 0:
                label, reason = "UNLABELABLE", "no_entry_price"
            else:
                win = book.window(pair, ts, lookahead_hours)
                if win is None:
                    label, reason = "UNLABELABLE", "no_price_window"
                else:
                    entry_is_limit = rec.get("entry_level") is not None
                    label, first_touch, outcome_r = first_touch_label(
                        win, direction, entry, r_price, entry_is_limit
                    )
                    reason = ""
                    row["entry_ref"] = round(entry, 6)
                    row["entry_is_limit"] = entry_is_limit
                    row["r_price"] = round(r_price, 6)
                    row["r_source"] = r_source
                    row["first_touch"] = first_touch
                    row["bars_in_window"] = int(len(win))
                    row["label_outcome_r"] = outcome_r

        row["label"] = label
        if label == "UNLABELABLE":
            row["unlabelable_reason"] = reason
            unlabelable_reasons[reason] += 1
        elif label == "NO_FILL":
            by_stage[stage][label] += 1
            by_dir_source[dir_source][label] += 1
        else:
            by_stage[stage][label] += 1
            by_dir_source[dir_source][label] += 1
        counts[label] += 1
        labeled_rows.append(row)

    with out_path.open("w") as fh:
        for row in labeled_rows:
            fh.write(json.dumps(row, default=str) + "\n")

    labelable = counts["FALSE_NEGATIVE"] + counts["TRUE_NEGATIVE"]
    fn_rate = counts["FALSE_NEGATIVE"] / labelable if labelable else 0.0
    return {
        "month": month,
        "source": str(src),
        "output": str(out_path),
        "records": len(records),
        "false_negative": counts["FALSE_NEGATIVE"],
        "true_negative": counts["TRUE_NEGATIVE"],
        "no_fill": counts["NO_FILL"],
        "unlabelable": counts["UNLABELABLE"],
        "labelable": labelable,
        "fn_rate": round(fn_rate, 4),
        "by_stage": {s: dict(c) for s, c in by_stage.items()},
        "by_dir_source": {s: dict(c) for s, c in by_dir_source.items()},
        "unlabelable_reasons": dict(unlabelable_reasons),
    }


def print_report(results: List[Dict[str, Any]], lookahead_hours: int) -> None:
    print("\n" + "=" * 68)
    print(f"ICT VETO OUTCOME LABELING  (lookahead = {lookahead_hours}h, first-touch +/-1R)")
    print("=" * 68)
    tot = Counter()
    for r in results:
        if r.get("skipped"):
            print(f"\n[{r['month']}] skipped: {r['skipped']}")
            continue
        print(f"\n[{r['month']}]  {r['records']} records -> {r['output']}")
        print(f"  FALSE_NEGATIVE : {r['false_negative']}")
        print(f"  TRUE_NEGATIVE  : {r['true_negative']}")
        print(f"  NO_FILL        : {r['no_fill']}  (limit never reached in window — veto moot)")
        print(f"  UNLABELABLE    : {r['unlabelable']}  {r['unlabelable_reasons']}")
        if r["labelable"]:
            print(f"  >> FN rate (of labelable FN+TN): {r['fn_rate']*100:.1f}%  "
                  f"({r['false_negative']}/{r['labelable']})")
        print(f"  by veto_stage  : {r['by_stage']}")
        print(f"  by dir source  : {r['by_dir_source']}")
        tot["FALSE_NEGATIVE"] += r["false_negative"]
        tot["TRUE_NEGATIVE"] += r["true_negative"]
        tot["NO_FILL"] += r["no_fill"]
        tot["UNLABELABLE"] += r["unlabelable"]

    labelable = tot["FALSE_NEGATIVE"] + tot["TRUE_NEGATIVE"]
    print("\n" + "-" * 68)
    print("COMBINED")
    print(f"  FALSE_NEGATIVE : {tot['FALSE_NEGATIVE']}")
    print(f"  TRUE_NEGATIVE  : {tot['TRUE_NEGATIVE']}")
    print(f"  NO_FILL        : {tot['NO_FILL']}")
    print(f"  UNLABELABLE    : {tot['UNLABELABLE']}")
    if labelable:
        print(f"  >> OVERALL FN rate (of labelable): "
              f"{tot['FALSE_NEGATIVE']/labelable*100:.1f}%  "
              f"({tot['FALSE_NEGATIVE']}/{labelable})")
    print("=" * 68 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Label ICT veto-ledger outcomes (offline research).")
    ap.add_argument("--month", action="append", dest="months",
                    help="Month shard YYYY_MM (repeatable). Default: 2026_05 + 2026_06.")
    ap.add_argument("--lookahead-hours", type=int, default=DEFAULT_LOOKAHEAD_HOURS)
    ap.add_argument("--default-r-pct", type=float, default=DEFAULT_R_PCT,
                    help="R as fraction of price when no stop distance (default 0.005 = 0.5%%).")
    args = ap.parse_args(argv)

    months = args.months or DEFAULT_MONTHS
    book = PriceBook()
    results = [label_month(m, book, args.lookahead_hours, args.default_r_pct) for m in months]
    print_report(results, args.lookahead_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
