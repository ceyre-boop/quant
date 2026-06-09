#!/usr/bin/env python3
"""
Score the Big Move Oracle's pulses against what actually happened — so we KNOW if the
forecast has skill before trusting it (the self-calibration loop = the real magnum opus).

For each pulse on a day: direction_hit (did price move the predicted way by close),
level_hit (did price reach the drawn_to_level), magnitude error (|actual range − implied|).
Backfills direction_hit into the jsonl so big_move_oracle._recent_calibration can read it.
Reports Brier (lower=better; 0.25=coin-flip) so a real edge is provable, not assumed.

Usage:
    python3.13 scripts/futures_pulse_calibration.py                 # today, MES+MNQ
    python3.13 scripts/futures_pulse_calibration.py --date 2026-06-09 --source ib
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.futures import bar_feed as bf          # noqa: E402

PULSE_LOG = ROOT / "data" / "futures" / "big_move_pulse.jsonl"


def _actual(instrument: str, day: str, source: str):
    """(close, high, low) for the day's RTH session, or None."""
    df = bf.load_history(instrument, source=source, lookback="5d")
    if df is None or len(df) == 0:
        return None
    et = df.index.tz_convert(bf.ET).strftime("%Y-%m-%d")
    d = df[et == day]
    if len(d) == 0:
        return None
    return (round(float(d["Close"].iloc[-1]), 2), round(float(d["High"].max()), 2),
            round(float(d["Low"].min()), 2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Big Move Oracle pulses")
    ap.add_argument("--date", default=datetime.now(timezone.utc).astimezone(bf.ET).strftime("%Y-%m-%d"))
    ap.add_argument("--source", default="ib", choices=["ib", "yf"])
    args = ap.parse_args()

    if not PULSE_LOG.exists():
        print("No big_move_pulse.jsonl yet — run scripts/futures_pulse.py first.")
        return
    records = [json.loads(l) for l in PULSE_LOG.read_text().splitlines() if l.strip()]
    actual_cache: dict = {}
    scored = []

    for rec in records:
        if rec.get("direction_hit") is not None:
            continue
        if not str(rec.get("ts", "")).startswith(args.date) and args.date not in str(rec.get("ts", "")):
            # match by ET date of the pulse ts
            try:
                ts_et = datetime.fromisoformat(rec["ts"]).astimezone(bf.ET).strftime("%Y-%m-%d")
            except Exception:
                continue
            if ts_et != args.date:
                continue
        inst = rec.get("instrument", "MES")
        if inst not in actual_cache:
            actual_cache[inst] = _actual(inst, args.date, args.source)
        act = actual_cache[inst]
        if act is None:
            continue
        close, high, low = act
        entry = rec.get("last_price")
        d = rec.get("direction")
        if entry and d in ("LONG", "SHORT"):
            rec["direction_hit"] = int((close > entry) if d == "LONG" else (close < entry))
            lvl = rec.get("drawn_to_level")
            if lvl is not None:
                rec["level_hit"] = int(high >= lvl if lvl >= entry else low <= lvl)
            imp = rec.get("implied_move_pts")
            if imp:
                rec["magnitude_err_pts"] = round(abs((high - low) - imp), 2)
            p = rec.get("stated_probability", 0.5) or 0.5
            rec["brier"] = round((p - rec["direction_hit"]) ** 2, 4)
            rec["scored_at"] = datetime.now(timezone.utc).isoformat()
            scored.append(rec)

    if scored:
        tmp = PULSE_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r, default=str) for r in records) + "\n")
        tmp.replace(PULSE_LOG)

    G, Y, BD, RS = "\033[92m", "\033[93m", "\033[1m", "\033[0m"
    print(f"\n{BD}BIG MOVE CALIBRATION — {args.date}{RS}  (scored {len(scored)} pulses)")
    if scored:
        hits = sum(r["direction_hit"] for r in scored)
        brier = sum(r["brier"] for r in scored) / len(scored)
        lvl = [r for r in scored if r.get("level_hit") is not None]
        lvl_hits = sum(r["level_hit"] for r in lvl)
        mags = [r["magnitude_err_pts"] for r in scored if r.get("magnitude_err_pts") is not None]
        print(f"  Direction hit: {hits}/{len(scored)} ({hits/len(scored):.0%})")
        if lvl:
            print(f"  Level reached: {lvl_hits}/{len(lvl)} ({lvl_hits/len(lvl):.0%})")
        if mags:
            print(f"  Magnitude MAE: {sum(mags)/len(mags):.1f} pts (forecast range vs implied vs actual)")
        c = G if brier < 0.25 else Y
        print(f"  {c}Brier: {brier:.3f}{RS}  (0.25 = coin-flip; lower = real skill)")
        if brier >= 0.25:
            print(f"  {Y}Not beating chance yet — keep accumulating before trusting it to trade.{RS}")
    print()


if __name__ == "__main__":
    main()
