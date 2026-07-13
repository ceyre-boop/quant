#!/usr/bin/env python3
"""POST-HOC decomposition of the SEALED G2 events — cost/sizing layers only.

STAMP: POST-HOC, NON-EVIDENCE. This recomputes the exact sealed event returns of
HYP-093 and decomposes the constitutional %/day into its layers (gross edge,
slippage, borrow, locate weight, worst-case sizing). NO new strategy variants
are evaluated on the holdout — a different stop/exit is a NEW hypothesis and
must go through its own prereg on non-holdout data.
Run: python3 -m research.yield_frontier.posthoc_g2
"""
import json

import numpy as np

from ._lib import REPO
from .gauntlet_run import (apr, classify, dtime, defaultdict, et_t, gzip,
                           load_holdout_grouped, prereg, HD)


def main():
    doc = prereg("HYP-093")
    sched = doc["costs"]["borrow_apr_schedule_pessimistic"]
    grouped, _ = load_holdout_grouped()
    cands = defaultdict(list)
    with open(HD / "candidates.csv") as f:
        next(f)
        for line in f:
            d, t, pc = line.strip().split(",")
            cands[d].append(t)
    gross, borrow = [], []
    n_days = len(grouped)
    for day in sorted(grouped):
        fp = HD / f"alpaca/{day}.json.gz"
        nfp = HD / f"news/{day}.json"
        if day not in cands or not fp.exists():
            continue
        with gzip.open(fp, "rt") as f:
            payload = json.load(f)
        news = json.loads(nfp.read_text()) if nfp.exists() else {}
        for t in cands[day]:
            bars = payload["intraday"].get(t, [])
            daily = payload["daily"].get(t, [])
            if not bars or not daily:
                continue
            pc = daily[-1]["c"]
            sl = [b for b in bars if dtime(9, 30) <= et_t(b) <= dtime(10, 25)]
            if len(sl) < 8 or et_t(sl[-1]) < dtime(10, 15):
                continue
            P = sl[-1]["c"]
            if not (P >= 1.30 * pc and P >= 2.00 and sum(b["v"] for b in sl) >= 500_000):
                continue
            gain = P / pc - 1
            if gain < 0.50 or classify(news.get(t, [])) == "MERGER_ACQ":
                continue
            post = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(16, 0)]
            if not post:
                continue
            entry = post[0]["o"]
            stop_px = entry * 1.30
            exit_px = None
            for b in post[1:]:
                if b["o"] >= stop_px:
                    exit_px = b["o"]
                    break
                if b["h"] >= stop_px:
                    exit_px = stop_px
                    break
            if exit_px is None:
                exit_px = post[-1]["c"]
            gross.append((entry - exit_px) / entry)
            borrow.append(apr(gain, sched) / 365)
    g = np.array(gross)
    b = np.array(borrow)
    ev_day = len(g) / n_days
    print(f"POST-HOC (non-evidence) — HYP-093 sealed-event layer decomposition, "
          f"n={len(g)} events / {n_days} days ({ev_day:.2f}/day)")
    print(f"  gross edge/event:        mean {g.mean():+.4f}  median {np.median(g):+.4f}")
    print(f"  slippage (2x50bps):      -0.0100 per event")
    print(f"  borrow (pessimistic):    -{b.mean():.4f} per event mean")
    net = g - 0.01 - b
    print(f"  net/event:               mean {net.mean():+.4f}  median {np.median(net):+.4f}  p5 {np.percentile(net,5):+.4f}")
    print("  --- sizing layers on the SAME sealed net edge ---")
    for label, notional, locate in (
            ("prereg (worst-case 60%, locate 50%)", 0.0125, 0.50),
            ("locate 75% (mining assumption)", 0.0125, 0.75),
            ("worst-case = stop 30% (NO gap-through buffer)", 0.0250, 0.50)):
        pct_day = net.mean() * ev_day * notional * locate
        print(f"  {label:<48} -> {pct_day:+.5f}/day (floor 0.00050)")
    print("NOTE: rows 2-3 relax PREREG assumptions and are arithmetic re-weightings "
          "of sealed events, shown to locate the binding constraint. They do not "
          "change the sealed verdict. A defined-risk redesign (puts) or tighter stop "
          "is a NEW hypothesis -> HYP-096 prereg on non-holdout data.")


if __name__ == "__main__":
    main()
