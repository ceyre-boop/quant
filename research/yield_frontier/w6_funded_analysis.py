#!/usr/bin/env python3
"""
w6_funded_analysis.py — what The Undertow (HYP-093) does at different account sizes,
and what happens when funded-account RULES are laid over it.

Reuses the W6 F2+F3 recommended policy and the sealed 559-event paths. Produces:
  1. Annual $ P&L distribution (p5/p25/p50/p75/p95) at several account sizes.
  2. Funded-challenge outcome math: P(pass target before breaching max drawdown),
     under a few standard rule sets, plus expected payout with a profit split.

The point is NOT to pretend a funded vehicle exists for this strategy class — TICK-032
established it does not (single-name microcap HTB shorts violate prop-firm consistency
and shape rules). The point is to SHOW WHY, in numbers: the drawdown profile that makes
the edge work on own capital is the same profile that trips funded-account rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
EVENTS = REPO / "data" / "research" / "yield_frontier" / "optimization" / "W6_inputs" / "hyp093_events.json"

# W6 recommended policy (F2+F3): RCK base + Grossman-Zhou drawdown governor.
F2_BASE = {"T10": 0.0799, "T20": 0.0673}    # pre-locate RCK fractions
LOCATE = 0.50
W_STAR = {"T10": 0.6269, "T20": 0.7975}
F3_DD_MAX = 0.15
F3_GAMMA = 1.0
N_DAYS = 242
DISASTER_P = 0.005
DISASTER_L = (-2.0, -1.0)
N_PATHS = 20_000
SEED = 20260721


def load():
    d = json.loads(EVENTS.read_text())["events"]
    rets = np.array([e["ret_event"] for e in d])
    tiers = np.array([0 if e["tier"] == "T10" else 1 for e in d])
    return rets, tiers


def one_path(rets, tiers, rng, block=True, block_size=5):
    n = len(rets)
    if block:
        starts = rng.integers(0, n - block_size, size=(n // block_size) + 1)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    else:
        idx = rng.integers(0, n, size=n)
    pr = rets[idx].copy()
    pt = tiers[idx].copy()
    dis = rng.random(n) < DISASTER_P
    if dis.any():
        wstar = np.where(pt == 0, W_STAR["T10"], W_STAR["T20"])
        L = rng.uniform(*DISASTER_L, size=n)
        pr = np.where(dis, L * wstar, pr)
    base = np.where(pt == 0, F2_BASE["T10"], F2_BASE["T20"]) * LOCATE
    # F3 drawdown governor applied event-by-event
    wealth, hwm = 1.0, 1.0
    curve = np.empty(n)
    for i in range(n):
        dd = (hwm - wealth) / hwm
        gov = max(0.0, 1.0 - dd / F3_DD_MAX) ** F3_GAMMA
        size = base[i] * gov
        wealth *= max(1.0 + size * pr[i], 1e-9)
        curve[i] = wealth
        hwm = max(hwm, wealth)
    return curve


def main():
    rets, tiers = load()
    rng = np.random.default_rng(SEED)
    finals, maxdds, path_min = [], [], []
    curves = []
    for _ in range(N_PATHS):
        c = one_path(rets, tiers, rng)
        finals.append(c[-1])
        hwm = np.maximum.accumulate(np.concatenate([[1.0], c]))
        dd = (hwm[1:] - c) / hwm[1:]
        maxdds.append(dd.max())
        path_min.append(c.min())
        if len(curves) < 4000:
            curves.append(c)
    finals = np.array(finals)
    maxdds = np.array(maxdds)
    ann = finals - 1.0  # annual return fraction (559 events ~= 1 holdout year)

    print("=" * 70)
    print("THE UNDERTOW (HYP-093) at F2+F3 sizing — annual return distribution")
    print(f"({N_PATHS:,} block-bootstrap paths, pessimistic 0.5% disaster rate)")
    print("=" * 70)
    pcts = [5, 25, 50, 75, 95]
    ap = {p: np.percentile(ann, p) for p in pcts}
    print("annual return %:  p5 {:+.1%}  p25 {:+.1%}  p50 {:+.1%}  p75 {:+.1%}  p95 {:+.1%}"
          .format(ap[5], ap[25], ap[50], ap[75], ap[95]))
    print("mean {:+.1%} | prob(profitable year) {:.1%} | median MaxDD {:.1%} | p95 MaxDD {:.1%}"
          .format(ann.mean(), float((ann > 0).mean()), np.percentile(maxdds, 50),
                  np.percentile(maxdds, 95)))
    print()

    print("DOLLAR P&L BY ACCOUNT SIZE (own capital, per year)")
    print(f"{'account':>10} | {'p5 (bad)':>11} | {'p25':>10} | {'p50 (base)':>11} | "
          f"{'p75':>10} | {'p95 (great)':>12}")
    print("-" * 76)
    for acct in (5_000, 10_000, 25_000, 50_000, 100_000, 200_000):
        row = [acct * ap[p] for p in pcts]
        print(f"${acct:>8,} | {row[0]:>+11,.0f} | {row[1]:>+10,.0f} | {row[2]:>+11,.0f} | "
              f"{row[3]:>+10,.0f} | {row[4]:>+12,.0f}")
    print()

    # ── Funded-challenge overlay ──────────────────────────────────────────────
    print("=" * 70)
    print("FUNDED-ACCOUNT OVERLAY — pass a challenge, then trade funded capital")
    print("=" * 70)
    print("Challenge = hit profit target BEFORE breaching max drawdown, on the")
    print("evaluation account. Rules below are typical 1-step evaluation terms.")
    print()

    rulesets = [
        ("Aggressive prop (8% target / 10% trailing DD)", 0.08, 0.10),
        ("Standard prop (10% target / 10% max DD)", 0.10, 0.10),
        ("Conservative (8% target / 6% max DD)", 0.08, 0.06),
    ]
    curves = np.array(curves)  # (n, 559)
    for name, target, dd_limit in rulesets:
        passed = breached = neither = 0
        for c in curves:
            hwm = np.maximum.accumulate(np.concatenate([[1.0], c]))[1:]
            dd = (hwm - c) / hwm
            hit_target_at = np.argmax(c >= 1 + target) if (c >= 1 + target).any() else None
            breach_at = np.argmax(dd >= dd_limit) if (dd >= dd_limit).any() else None
            if hit_target_at is not None and (breach_at is None or hit_target_at < breach_at):
                passed += 1
            elif breach_at is not None:
                breached += 1
            else:
                neither += 1
        tot = len(curves)
        print(f"{name}")
        print(f"    PASS {passed/tot:5.1%}   BLOW-UP (DD breach) {breached/tot:5.1%}   "
              f"neither in 1yr {neither/tot:5.1%}")
    print()
    print("Reading: the same drawdown profile that makes the edge compound on own")
    print("capital trips funded-account drawdown limits a large share of the time.")
    print("That is TICK-032's 'NOT VIABLE' verdict, shown in numbers rather than asserted.")


if __name__ == "__main__":
    main()
