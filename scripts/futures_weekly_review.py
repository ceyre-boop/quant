#!/usr/bin/env python3
"""
Weekly review — turn a week of reps into tested hypotheses (<5min).

1. Aggregate closed trades by setup / regime / time-gate / confluence / cvd_quality.
2. Hypothesis generation: any cut with n>=10 and win-rate divergence >15pp from overall ->
   a PROPOSED hypothesis in data/futures/weekly_hypotheses.jsonl (NEVER auto-applied).
3. Gate-update recommendations from prior CONFIRMED hypotheses -> printed suggestions for
   futures_params.yml (manual sign-off only).

Nothing here changes a live param. Hypothesis -> CONFIRMED -> ship is a human decision.

Usage:  python3 scripts/futures_weekly_review.py [--days 7]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.futures import review_common as rc

TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"
HYP_LOG = ROOT / "data" / "futures" / "weekly_hypotheses.jsonl"
MIN_N = 10              # require this many trades in a cut before proposing
DIVERGE_PP = 0.15       # win-rate this far from overall = a hypothesis


CUTS = {
    "setup_type":   lambda r: r.get("setup_type") or rc.reasoning_field(r, "setup_type"),
    "regime":       lambda r: rc.reasoning_field(r, "regime"),
    "time_gate":    lambda r: rc.reasoning_field(r, "time_gate"),
    "confluence":   lambda r: rc.reasoning_field(r, "confluence_score"),
    "cvd_quality":  lambda r: rc.reasoning_field(r, "cvd_quality"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    cutoff = (datetime.now(rc.ET) - timedelta(days=args.days)).strftime("%Y-%m-%d")

    trades = [t for t in rc.load_trades(TRADE_LOG)
              if t.get("exit") is not None and (rc.session_date(t) or "") >= cutoff]
    overall = rc.winrate(trades)
    week = datetime.now(rc.ET).strftime("%Y-W%U")

    print(f"\n{'='*64}\n  FUTURES WEEKLY REVIEW — last {args.days}d (since {cutoff})\n{'='*64}")
    print(f"  closed trades: {len(trades)} | overall win rate: "
          f"{overall if overall is not None else 'n/a'}")
    if overall is None or len(trades) < MIN_N:
        print(f"  Not enough closed trades yet (need >= {MIN_N}). Keep taking reps.\n")
        return 0

    existing = {(json.loads(l).get("cut"), str(json.loads(l).get("bucket")), json.loads(l).get("week"))
                for l in HYP_LOG.read_text().splitlines() if l.strip()} if HYP_LOG.exists() else set()
    proposed = []

    for cut, fn in CUTS.items():
        print(f"\n  by {cut}:")
        for bucket, recs in sorted(rc.group_by(trades, fn).items(), key=lambda kv: str(kv[0])):
            wr = rc.winrate(recs)
            if wr is None:
                continue
            w = sum(1 for r in recs if rc.is_win(r))
            flag = ""
            if len(recs) >= MIN_N and abs(wr - overall) >= DIVERGE_PP:
                predictor = "POSITIVE" if wr > overall else "NEGATIVE"
                flag = f"  <-- {predictor} predictor ({(wr-overall)*100:+.0f}pp)"
                key = (cut, str(bucket), week)
                if key not in existing:
                    hyp = {
                        "week": week, "cut": cut, "bucket": bucket, "n": len(recs),
                        "win_rate": wr, "overall": overall, "delta_pp": round((wr - overall) * 100, 1),
                        "predictor": predictor,
                        "hypothesis": (f"{cut}={bucket} is a {predictor.lower()} predictor "
                                       f"({w}/{len(recs)}={wr:.0%} vs overall {overall:.0%}). "
                                       + (f"Consider requiring/raising it." if predictor == "POSITIVE"
                                          else f"Consider blocking/avoiding it.")),
                        "status": "PROPOSED", "created": datetime.now(rc.ET).isoformat(),
                    }
                    proposed.append(hyp)
            print(f"    {str(bucket):14s} {w:>2}/{len(recs):<3} ({wr:.0%}){flag}")

    if proposed:
        HYP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HYP_LOG.open("a") as f:
            for h in proposed:
                f.write(json.dumps(h) + "\n")
        print(f"\n  {len(proposed)} new PROPOSED hypothesis(es) -> {HYP_LOG.name} "
              f"(review + tag CONFIRMED/REJECTED manually; never auto-applied).")

    # ── recommendations from prior CONFIRMED hypotheses (manual sign-off to ship) ──
    confirmed = [json.loads(l) for l in HYP_LOG.read_text().splitlines()
                 if l.strip() and json.loads(l).get("status") == "CONFIRMED"] if HYP_LOG.exists() else []
    if confirmed:
        print("\n  GATE-UPDATE RECOMMENDATIONS (from CONFIRMED hypotheses — you decide what ships):")
        for h in confirmed:
            print(f"    - {h['cut']}={h['bucket']} ({h['predictor']}): "
                  f"recommend a futures_params.yml change reflecting this. (manual)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
