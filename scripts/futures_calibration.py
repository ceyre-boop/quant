#!/usr/bin/env python3
"""
Score the futures morning-oracle calls for calibration.

futures_oracle_morning.py logs a falsifiable pre-session call (bias, stated_probability,
t1_target, invalidation) with outcome_scored=None. This script replays the session's bars
and backfills outcome_hit_t1 / outcome_scored / brier_contribution, then reports the
oracle's hit rate and Brier score so its probability estimates can be held honest.

Usage:
    python3.13 scripts/futures_calibration.py --oracle --score-today
    python3.13 scripts/futures_calibration.py --date 2026-06-05 --instrument MNQ
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.futures import bar_feed as bf      # noqa: E402

ORACLE_LOG = ROOT / "data" / "futures" / "oracle_mornings.jsonl"


def _hit_t1(day_df, bias: str, t1: float, invalidation: float | None) -> int:
    """1 if price reached t1 in the bias direction before the invalidation was breached."""
    if day_df is None or len(day_df) == 0 or bias not in ("LONG", "SHORT"):
        return 0
    for j in range(len(day_df)):
        hi, lo = float(day_df["High"].iloc[j]), float(day_df["Low"].iloc[j])
        if invalidation is not None:
            if bias == "LONG" and lo <= invalidation:
                return 0
            if bias == "SHORT" and hi >= invalidation:
                return 0
        if bias == "LONG" and hi >= t1:
            return 1
        if bias == "SHORT" and lo <= t1:
            return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Score futures oracle calls for calibration")
    ap.add_argument("--oracle", action="store_true", help="(compat flag) score oracle calls")
    ap.add_argument("--score-today", action="store_true", help="only score today's calls")
    ap.add_argument("--date", default=None, help="ET day YYYY-MM-DD to score (default: all unscored)")
    ap.add_argument("--instrument", default=None, choices=[None, "MES", "MNQ"])
    args = ap.parse_args()

    if not ORACLE_LOG.exists():
        print("No oracle_mornings.jsonl yet — nothing to score.")
        return

    records = [json.loads(l) for l in ORACLE_LOG.read_text().splitlines() if l.strip()]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_date = today if args.score_today else args.date

    bars_cache: dict[tuple[str, str], object] = {}
    scored_now = 0
    for rec in records:
        if rec.get("outcome_scored") is not None:
            continue
        if target_date and rec.get("date") != target_date:
            continue
        if args.instrument and rec.get("instrument") != args.instrument:
            continue
        bias = rec.get("bias")
        kl = rec.get("key_levels") or {}
        t1 = kl.get("t1_target")
        if bias not in ("LONG", "SHORT") or t1 is None:
            continue
        inst = rec.get("instrument", "MES")
        day = rec.get("date")
        key = (inst, day)
        if key not in bars_cache:
            bars_cache[key] = bf.load_history(inst, source="yf", day=day, lookback="7d")
        day_df = bars_cache[key]
        if day_df is None or len(day_df) == 0:
            continue
        hit = _hit_t1(day_df, bias, float(t1), kl.get("invalidation"))
        prob = rec.get("stated_probability") or 0.5
        rec["outcome_hit_t1"] = hit
        rec["outcome_scored"] = hit
        rec["brier_contribution"] = round((prob - hit) ** 2, 4)
        rec["scored_at"] = datetime.now(timezone.utc).isoformat()
        scored_now += 1

    if scored_now:
        tmp = ORACLE_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        tmp.replace(ORACLE_LOG)

    graded = [r for r in records if r.get("outcome_scored") is not None]
    G, BD, RS = "\033[92m", "\033[1m", "\033[0m"
    print(f"\n{BD}ORACLE CALIBRATION{RS}  (scored {scored_now} new; {len(graded)} total)")
    if graded:
        hits = sum(r["outcome_scored"] for r in graded)
        brier = sum(r.get("brier_contribution", 0) for r in graded) / len(graded)
        print(f"  Hit rate: {hits}/{len(graded)} ({hits/len(graded):.0%})   "
              f"Brier: {brier:.3f}  (lower is better; 0.25 = coin-flip)")
    print()


if __name__ == "__main__":
    main()
