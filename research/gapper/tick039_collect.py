#!/usr/bin/env python3
"""TICK-039 data collection: bars + REAL NBBO quotes for gapper candidates.

Two disjoint sets, split by the same 70/30 date boundary the HYP-107
reconstruction used:

  MINING  (date <  cut) -> training data for the parametric spread model
  HOLDOUT (date >= cut) -> the sealed evaluation set

The parametric model is fit on MINING ONLY so the holdout stays clean for the
cost model as well as the signal. For the holdout rerun itself we do not need a
model at all: we use the DIRECTLY MEASURED spread per event. Why model what you
can measure.

Writes incrementally to JSONL so an interrupted run resumes instead of losing
everything.

Usage:
    python -m research.gapper.tick039_collect --set holdout
    python -m research.gapper.tick039_collect --set mining --limit 300
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

from execution import alpaca
from execution.config import frozen
from execution.quotes import quote_at
from execution.scan import bar_at, symbol_shape_ok

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "research" / "gapper" / "per_candidate_enriched.csv"
OUT_DIR = ROOT / "data" / "research" / "gapper" / "tick039"
UTC = timezone.utc

C107 = frozen("hyp107")
SPLIT_FRACTION = 0.70          # the 70/30 date split from the reconstruction


def split_cut() -> str:
    rows = list(csv.DictReader(open(CSV_PATH)))
    dates = sorted({r["date"] for r in rows})
    return dates[int(len(dates) * SPLIT_FRACTION)]


def load_rows(which: str) -> list[dict]:
    cut = split_cut()
    rows = list(csv.DictReader(open(CSV_PATH)))
    if which == "holdout":
        return [r for r in rows if r["date"] >= cut]
    return [r for r in rows if r["date"] < cut]


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.add(f"{r['date']}|{r['ticker']}")
    return out


def collect_one(sym: str, d: date, prev_close: float) -> dict | None:
    """Bars + real quotes at the frozen entry (09:31:00) and exit (10:31:00)."""
    bars = alpaca.minute_bars(sym, d)
    if not bars:
        return None
    b0930, b0931, b1030 = bar_at(bars, 9, 30), bar_at(bars, 9, 31), bar_at(bars, 10, 30)
    if not (b0930 and b0931 and b1030):
        return None

    og = b0930["o"] / prev_close - 1
    lv = math.log10(b0930["v"] + 1)

    entry_ts = alpaca.et_dt(d, dtime(9, 31)).astimezone(UTC)
    exit_ts = (alpaca.et_dt(d, dtime(10, 30)) + timedelta(minutes=1)).astimezone(UTC)

    eq = quote_at(sym, entry_ts)
    xq = quote_at(sym, exit_ts)

    rec = {
        "date": str(d), "ticker": sym, "prev_close": prev_close,
        "overnight_gap": round(og, 6),
        "log_vol": round(lv, 6),
        "first_min_vol": int(b0930["v"]),
        "entry_open_0931": b0931["o"],
        "exit_close_1030": b1030["c"],
        # bar-based gross, the number the original reconstruction used
        "gross_bar": round(b1030["c"] / b0931["o"] - 1, 6) if b0931["o"] else None,
        "entry_bar_range_pct": round((b0931["h"] - b0931["l"]) / b0931["o"], 6)
                               if b0931["o"] else None,
        "entry_minute_dollar_vol": round(b0931["o"] * b0931["v"], 2),
        "exit_bar_range_pct": round((b1030["h"] - b1030["l"]) / b1030["o"], 6)
                              if b1030["o"] else None,
        "exit_minute_dollar_vol": round(b1030["o"] * b1030["v"], 2),
        "has_quotes": bool(eq and xq),
    }
    if eq:
        rec.update({"entry_bid": eq.bid, "entry_ask": eq.ask,
                    "entry_mid": round(eq.mid, 6),
                    "entry_spread_pct": round(eq.spread_pct, 6),
                    "entry_bid_size": eq.bid_size, "entry_ask_size": eq.ask_size})
    if xq:
        rec.update({"exit_bid": xq.bid, "exit_ask": xq.ask,
                    "exit_mid": round(xq.mid, 6),
                    "exit_spread_pct": round(xq.spread_pct, 6)})
    if eq and xq and eq.mid > 0:
        # LONG: buy the ask, sell the bid
        rec["net_quoted"] = round(xq.bid / eq.ask - 1, 6)
        rec["gross_quoted"] = round(xq.mid / eq.mid - 1, 6)
        rec["spread_cost"] = round(rec["gross_quoted"] - rec["net_quoted"], 6)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["holdout", "mining"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args(argv)

    alpaca.load_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.set}_events.jsonl"

    rows = load_rows(args.set)
    rows = [r for r in rows if symbol_shape_ok(r["ticker"].strip().upper())]
    if args.limit:
        random.seed(args.seed)
        random.shuffle(rows)
        rows = rows[:args.limit]

    already = done_keys(out_path)
    print(f"[tick039] set={args.set} cut={split_cut()} rows={len(rows)} "
          f"already_done={len(already)}", flush=True)

    ok = fail = 0
    with open(out_path, "a") as fh:
        for i, r in enumerate(rows, 1):
            sym = r["ticker"].strip().upper()
            d = date.fromisoformat(r["date"])
            key = f"{d}|{sym}"
            if key in already:
                continue
            try:
                rec = collect_one(sym, d, float(r["prev_close"]))
            except Exception as e:                     # noqa: BLE001
                print(f"  {key}: {type(e).__name__}: {str(e)[:90]}", flush=True)
                fail += 1
                continue
            if rec is None:
                fail += 1
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ok += 1
            if ok % 25 == 0:
                print(f"  ...{ok} collected ({i}/{len(rows)} scanned)", flush=True)

    print(f"[tick039] {args.set}: collected {ok}, failed/skipped {fail} -> {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
