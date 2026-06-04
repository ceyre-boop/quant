#!/usr/bin/env python3
"""Replay the combat vetoes against the real forensic trades — the Phase 0 "done when" proof.

(a) Reproduce the forensic -273.23R recoverable (primary-classification over the avoidable set),
    proving we read the data correctly.
(b) Apply the 4 PRE-TRADE force-skip vetoes (independent-OR) to every trade and report the gate's
    true historical impact — including the honest cost of winners it would also skip.

Usage:  python3 scripts/validate_combat_vetoes.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.forex.combat_vetoes import evaluate, load_config

FORENSICS = ROOT / "data" / "research" / "trade_forensics.json"
AVOIDABLE = {"MACRO_AGAINST", "LOW_MOMENTUM_ENTRY", "COUNTER_MOMENTUM", "WEAK_RATE_SIGNAL"}


def main():
    trades = json.loads(FORENSICS.read_text())
    losses = [t for t in trades if t.get("outcome") == "LOSS"]

    # ── (a) Reproduce the forensic recoverable number ────────────────────────
    total_r_lost = sum(t["outcome_r"] for t in losses)
    recoverable = sum(t["outcome_r"] for t in losses if t.get("failure_mode") in AVOIDABLE)
    print(f"\n{'='*66}\n  COMBAT-VETO REPLAY — {len(trades)} trades ({len(losses)} losses)\n{'='*66}")
    print("  (a) Forensic reproduction (primary-classification, avoidable set):")
    print(f"      total R lost to failures : {total_r_lost:+.2f}R")
    print(f"      potentially recoverable  : {recoverable:+.2f}R  "
          f"({abs(recoverable/total_r_lost)*100:.1f}% of losses)")
    print(f"      forensic target          : -273.23R  "
          f"{'✓ REPRODUCED' if abs(recoverable - -273.23) < 0.5 else '✗ MISMATCH'}")

    # ── (b) The 4 pre-trade vetoes, independent-OR, applied live-style ───────
    cfg = load_config()
    skipped_loss_r = skipped_win_r = 0.0
    n_skip_loss = n_skip_win = 0
    per_veto = defaultdict(lambda: [0, 0.0])   # rule_id -> [count, summed_outcome_r]
    for t in trades:
        hits = evaluate(t.get("real_rate_diff"), t.get("momentum_63d"),
                        t.get("atr_14d_pct"), int(t["direction"]), cfg)
        if not hits:
            continue
        r = t["outcome_r"]
        if t.get("outcome") == "LOSS":
            skipped_loss_r += r; n_skip_loss += 1
        else:
            skipped_win_r += r; n_skip_win += 1
        for h in hits:
            per_veto[h.rule_id][0] += 1
            per_veto[h.rule_id][1] += r

    recovered = -skipped_loss_r          # losses are negative R; skipping saves that
    forgone = skipped_win_r              # winners are positive R; skipping gives them up
    net = recovered - forgone
    print("\n  (b) Pre-trade force-skip vetoes (C-001/C-003/C-005/C-006, independent-OR):")
    print(f"      avoided losses (recovered): {recovered:+.2f}R across {n_skip_loss} skipped losers")
    print(f"      forgone wins   (cost)     : {forgone:+.2f}R across {n_skip_win} skipped winners")
    print(f"      NET edge impact           : {net:+.2f}R  (recovered − forgone)")
    print("      per-veto (count, net R of all setups it skips):")
    for rid in ("C-001", "C-003", "C-005", "C-006"):
        cnt, rsum = per_veto.get(rid, [0, 0.0])
        print(f"        {rid}: {cnt:4d} setups, {rsum:+.2f}R")
    print(f"\n  Reconciliation: (b)'s avoided-loss {recovered:+.1f}R vs forensic {abs(recoverable):.1f}R — "
          f"the gate ORs four pre-trade rules (swapping C-002 size-cut for C-006), so it captures\n"
          f"  overlapping losses the primary-classification missed. The net edge impact ({net:+.1f}R) is\n"
          f"  the honest number: recovered losses minus the winners the same conditions would skip.")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
