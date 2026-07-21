#!/usr/bin/env python3
"""
numbers.py — "give me the numbers" engine.

Turns a CONFIRMED edge (a distribution of per-event returns + a sizing rule) into the
three tables a trader actually needs to decide:

  1. Annual return distribution (p5/p25/p50/p75/p95, mean, prob-profitable, drawdown).
  2. Dollar P&L by account size.
  3. Funded-account overlay: P(pass a challenge) vs P(blow up the eval account).

It is deliberately conservative and hard to fool:
  - Block bootstrap by default (preserves serial clustering, which inflates drawdown
    and is the honest stress; i.i.d. is available with --iid but flagged as optimistic).
  - A pessimistic disaster mixture (halt / gap-through / buy-in) is layered on by default.
  - Reports the MEDIAN year as the typical outcome, not the mean (returns are
    right-skewed; the mean oversells).
  - Never invents an edge: it only resamples the per-event returns you give it. Garbage
    in, garbage out — so feed it a SEALED, validated event set, not a mined one.

Input events JSON (either schema works):
  {"events": [{"ret_event": -0.12, "tier": "T10"}, ...]}     # W6 / gauntlet schema
  {"returns": [-0.12, 0.03, ...]}                            # bare per-event returns

Usage:
  python3 numbers.py --events path/to/events.json \
      --size 0.04 \
      --accounts 5000,10000,25000,50000,100000

  # per-tier sizing (e.g. the W6 F2+F3 recommendation):
  python3 numbers.py --events W6_inputs/hyp093_events.json \
      --size-t10 0.0799 --size-t20 0.0673 --locate 0.5 --dd-governor 0.15 \
      --label "The Undertow (HYP-093) F2+F3"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_events(path: str):
    d = json.loads(Path(path).read_text())
    if "events" in d:
        ev = d["events"]
        rets = np.array([e["ret_event"] for e in ev], dtype=float)
        tiers = np.array([0 if e.get("tier", "T10") == "T10" else 1 for e in ev])
    elif "returns" in d:
        rets = np.array(d["returns"], dtype=float)
        tiers = np.zeros(len(rets), dtype=int)
    else:
        raise SystemExit("events JSON needs an 'events' or 'returns' key")
    return rets, tiers


def one_path(rets, tiers, size_t10, size_t20, locate, dd_gov, disaster_p,
             disaster_L, wstar, rng, block, block_size):
    n = len(rets)
    if block:
        starts = rng.integers(0, max(1, n - block_size), size=(n // block_size) + 1)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    else:
        idx = rng.integers(0, n, size=n)
    pr = rets[idx].copy()
    pt = tiers[idx].copy()
    if disaster_p > 0:
        dis = rng.random(n) < disaster_p
        if dis.any():
            ws = np.where(pt == 0, wstar[0], wstar[1])
            L = rng.uniform(disaster_L[0], disaster_L[1], size=n)
            pr = np.where(dis, L * ws, pr)
    base = np.where(pt == 0, size_t10, size_t20) * locate
    if dd_gov <= 0:
        return np.cumprod(np.maximum(1.0 + base * pr, 1e-9))
    wealth, hwm = 1.0, 1.0
    curve = np.empty(n)
    for i in range(n):
        dd = (hwm - wealth) / hwm
        gov = max(0.0, 1.0 - dd / dd_gov)
        wealth *= max(1.0 + base[i] * gov * pr[i], 1e-9)
        curve[i] = wealth
        hwm = max(hwm, wealth)
    return curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--size", type=float, help="flat notional fraction per event")
    ap.add_argument("--size-t10", type=float)
    ap.add_argument("--size-t20", type=float)
    ap.add_argument("--locate", type=float, default=1.0)
    ap.add_argument("--dd-governor", type=float, default=0.0,
                    help="drawdown ceiling for a Grossman-Zhou governor; 0 disables")
    ap.add_argument("--accounts", default="5000,10000,25000,50000,100000,200000")
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--iid", action="store_true", help="i.i.d. bootstrap (optimistic)")
    ap.add_argument("--block-size", type=int, default=5)
    ap.add_argument("--disaster-p", type=float, default=0.005)
    ap.add_argument("--wstar", default="0.6269,0.7975",
                    help="per-tier worst-case for disaster scaling")
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--label", default="strategy")
    args = ap.parse_args()

    rets, tiers = load_events(args.events)
    if args.size is not None:
        s10 = s20 = args.size
    elif args.size_t10 is not None:
        s10 = args.size_t10
        s20 = args.size_t20 if args.size_t20 is not None else args.size_t10
    else:
        raise SystemExit("provide --size or --size-t10/--size-t20")

    wstar = tuple(float(x) for x in args.wstar.split(","))
    accounts = [int(x) for x in args.accounts.split(",")]
    rng = np.random.default_rng(args.seed)
    block = not args.iid

    finals, maxdds, curves = [], [], []
    for _ in range(args.paths):
        c = one_path(rets, tiers, s10, s20, args.locate, args.dd_governor,
                     args.disaster_p, (-2.0, -1.0), wstar, rng, block, args.block_size)
        finals.append(c[-1])
        hwm = np.maximum.accumulate(np.concatenate([[1.0], c]))[1:]
        maxdds.append(float(((hwm - c) / hwm).max()))
        if len(curves) < 5000:
            curves.append(c)
    finals = np.array(finals)
    maxdds = np.array(maxdds)
    ann = finals - 1.0

    pcts = [5, 25, 50, 75, 95]
    q = {p: np.percentile(ann, p) for p in pcts}
    mode = "block-bootstrap (clustering-preserving)" if block else "i.i.d. (OPTIMISTIC)"

    print("=" * 72)
    print(f"GIVE ME THE NUMBERS — {args.label}")
    print(f"{args.paths:,} paths · {mode} · disaster {args.disaster_p:.1%}/event · "
          f"size T10={s10*args.locate:.4f} T20={s20*args.locate:.4f} (post-locate)")
    print("=" * 72)
    print("ANNUAL RETURN:  p5 {:+.1%}  p25 {:+.1%}  p50 {:+.1%}  p75 {:+.1%}  p95 {:+.1%}"
          .format(*[q[p] for p in pcts]))
    print("mean {:+.1%} (skewed — trust the median) | profitable-year {:.1%} | "
          "median MaxDD {:.1%} | p95 MaxDD {:.1%}".format(
              ann.mean(), float((ann > 0).mean()),
              np.percentile(maxdds, 50), np.percentile(maxdds, 95)))
    print()
    print("DOLLARS PER YEAR BY ACCOUNT SIZE (own capital)")
    print(f"{'account':>10} | {'p5 bad':>10} | {'p25':>9} | {'p50 typical':>12} | "
          f"{'p75':>9} | {'p95 great':>11}")
    print("-" * 74)
    for a in accounts:
        r = [a * q[p] for p in pcts]
        print(f"${a:>8,} | {r[0]:>+10,.0f} | {r[1]:>+9,.0f} | {r[2]:>+12,.0f} | "
              f"{r[3]:>+9,.0f} | {r[4]:>+11,.0f}")
    print()
    print("FUNDED-ACCOUNT OVERLAY — pass target before breaching max drawdown")
    curves = np.array(curves)
    for name, tgt, dd in [("Aggressive (8% target / 10% DD)", 0.08, 0.10),
                          ("Standard   (10% target / 10% DD)", 0.10, 0.10),
                          ("Conservative (8% target / 6% DD)", 0.08, 0.06)]:
        p = b = 0
        for c in curves:
            hwm = np.maximum.accumulate(np.concatenate([[1.0], c]))[1:]
            ddc = (hwm - c) / hwm
            hit = np.argmax(c >= 1 + tgt) if (c >= 1 + tgt).any() else None
            brk = np.argmax(ddc >= dd) if (ddc >= dd).any() else None
            if hit is not None and (brk is None or hit < brk):
                p += 1
            elif brk is not None:
                b += 1
        tot = len(curves)
        print(f"  {name:<34}  PASS {p/tot:5.1%}   BLOW-UP {b/tot:5.1%}")
    print()
    print("Read the blow-up column. If it is high, the drawdown that makes this edge")
    print("compound is the same drawdown a funded account forbids — own capital only.")


if __name__ == "__main__":
    main()
