# Yield Frontier G-phase — Verdicts (HYP-093/094/095, sealed 2026-07-13)

Preregs hash-locked BEFORE holdout fetch (c5b10616 / 3e874fde / 959372e9); riders R1-R3
frozen in each; gate-zero enforced at fetch and at run; hashes verified pre/post seal.
Holdout: equities 2024-07-14→2025-06-30 (242 trading days; 9 dates denied by Polygon's
2-year lookback — partial-window clause invoked), NQ 2024-07→2026-06.

| HYP | strategy | verdict | boot p | DSR@809 | const. %/day | floor |
|---|---|---|---|---|---|---|
| 093 | parabolic gapper fade short | **VALID_BUT_BELOW_FLOOR** | 0.031 | 0.987 | +0.023% | 0.05% |
| 094 | overnight weak-close short | **NOT_SIGNIFICANT** | 0.102 | 0.932 | +0.006% | 0.03% |
| 095 | NQ high-VIX dip-day long | **VALID_BUT_BELOW_FLOOR** | 0.013 | 0.999 | +0.004% | 0.02% |

**The finding (first of its kind in this shop):** HYP-093's edge is statistically REAL on
never-touched data — 559 events, gross +3.5% mean / +6.5% median per event, still
+1.6%/+4.9% after pre-declared pessimistic slippage and borrow, significant at p=0.031
WITH the full 809-trial deflated-Sharpe penalty (DSR 0.987) and a BH family correction.
Nothing in this repo's history (ICT, geometry, positioning, adaptivity — 13 nulls) ever
cleared that bar at short horizon. The mining number (+3.8%/day) was, as expected, a
selection fantasy — the honest constitutional number is +0.023%/day.

**What kills it is monetization, not signal** (post-hoc decomposition, non-evidence):
the binding layer is R2 sizing — worst-case-beyond-stop (60%, validated by the holdout's
own p5 −32.6% gap-throughs) × 50% locate. Even relaxing BOTH to their friendly bounds
the arithmetic tops at 0.046%/day vs the 0.050% floor. And per TICK-032 (rider 4):
NO true funded vehicle exists for this strategy class (TTP bans the shape by rule;
Zimtra excludes US residents; T3/Bright = first-loss + licensing) — the funnel thesis
carries zero EV weight. Locate fees on the hottest names can exceed even the prereg's
pessimistic schedule (further-pessimism flag).

**HYP-095**: also real (DSR 0.999), exactly at min-n (40 events); constitutionally tiny
because a stopless index long demands a 10% worst-case. Unconstrained capacity.

**Honest doors this opens (NEW preregs, non-holdout data — never revisions of these):**
- HYP-096 candidate: defined-risk redesign of 093 (long puts on parabolic gappers) —
  defined risk collapses worst-case from 60% to premium-paid, transforming R2 sizing;
  the prereg must price these names' brutal option spreads (that's where it will live or die).
- HYP-097 candidate: 095 with a declared intraday stop → smaller worst-case → bigger
  constitutional size; must survive a fresh prereg (stop changes the strategy).

Verdicts sealed to ledger; sealed verdicts are final. TICK-024 cascade still gates any
real dollar (unchanged). Details: verdicts.json; decomposition: posthoc_g2.py.
