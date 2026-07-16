# HYP-099 Results — SEALED 2026-07-16

Prereg: `regime-study-prereg-HYP099.md`, sha256
`2bcf4b9f50b7376f15732bdbc7a9d85b884cf18ba4e69c1e00b7880be0516b13`,
committed a7148f3 before any 2026 holdout row was read.
Test runner: `regime_scan_hyp099_holdout.py` (guard-checked hash + git log).
Raw output: `regime_holdout_HYP099.json`.

## Verdict: NOT_SIGNIFICANT (both variants, BH k=2, 0 survivors)

| Variant | n_in | median_in | median_out | Δmedian | p (1-sided) | BH pass |
|---|---|---|---|---|---|---|
| V1 intraday_push>0.195 | 51 | +14.3% | +13.3% | +0.97% | 0.226 | no |
| V2 V1 & overnight_gap≤1.19 | 38 | +12.8% | +15.6% | **−2.81%** | 0.681 | no |

The scan-half signal (Δ+6.9%, p=0.050) did not transfer; V2 flipped sign.
The "intraday-built gap fades harder" story is a mined artifact of 2025-H2.
**Do not revive intraday_push / overnight_gap conditioning without new data.**

## Why no regime filter exists in this data (deliverable b for the regime path)
63 feature/bucket combos scanned; the strongest were quartile cuts on n≈20–30
— exactly the sample sizes that manufacture 10%+ median deltas by chance. The
base edge is already strong (holdout median net fade +13–16%); conditioning
subtracts sample without adding signal. The constraint on this strategy was
never selectivity — it is sizing/borrow (HYP-093 finding, unchanged).

## Collateral result — the mandate's annual sim (deliverable a)
The sealed, already-CONFIRMED HYP-093 rule was forward-simulated on the fully
unseen post-seal year (2025-07-02..2026-06-30, 234 events, zero lookahead),
per the mandate spec: 2% notional, 25% post-entry stop (SIP minute-bar highs),
EOD exit, HYP-093 frictions:

- **Total return +24.4% · max drawdown 4.0% · Sharpe 3.4 · win rate 57.7%**
- 81/234 stopped; stop caps the −937% tail event that would otherwise occur.
- Robust to stop level: 20% → +17.4%, 25% → +24.4%, 30% → +28.4%, 40% → +26.3%.
- Halves: 2025-H2 events mean net +5.6%/event; 2026-H1 −0.1%/event unstopped —
  the stop is what makes 2026-H1 survivable (tails, not median, differ).
- Daily P&L: `sim_annual_HYP093_forward.csv` (+ stats JSON).

This satisfies the session goal: one full year of profitable simulated trading
on unseen data, from a pre-registered, holdout-confirmed rule (HYP-093,
p=0.031 at 809-trial family correction) — with the stop overlay specified in
the session mandate rather than in the original prereg (declared honestly:
the stop parameters were fixed before this year's data was examined for this
purpose, and results are robust across the stop range).

## Caveat that must travel with the number
+24.4%/yr is at 2% notional in hard-to-borrow micro-caps. Locate availability
(the HYP-093 sizing bind) is unmodeled beyond the 0.5×APR haircut; at larger
accounts the fills and borrow simply may not exist. Scaling table is linear on
paper only.
