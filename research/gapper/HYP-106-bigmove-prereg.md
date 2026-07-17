# HYP-106 Pre-Registration — Leak-Free Runner Filter on Long Gapper Momentum

Registered 2026-07-17 before any holdout (events ≥ 2026-04-08) read with this
rule. Phase-2 discovery: `data/scan_results/hyp106_insample.json`. The filter
thresholds are FROZEN from in-sample and do not move.

## Mechanism
Among ≥100% gappers, the intraday LONG runners are the *moderate* overnight-gap,
*low* first-minute-volume names (still building), NOT the extreme-gap
climax-volume blow-offs (exhausted → the HYP-093 short targets). Features use
ONLY the 09:30 first-minute bar + prev_close — fully available at a 09:31 entry
(no look-ahead; the HYP-105 ML that scored 92% was rejected as leaky).

## Locked strategy (single values)
- Universe: the 234 minute-ready HYP-093 gappers; holdout = 71 events
  2026-04-08..2026-06-30.
- Entry: LONG at 09:31 ET bar open.
- **Filter (both required, thresholds frozen from in-sample medians):**
  overnight_gap ≤ 1.2395  AND  log10(first_bar_volume) ≤ 5.4330.
  (overnight_gap = 09:30 open / prev_close − 1; first_bar_volume = 09:30 minute
  volume.)
- Exit: 10:30 ET close. Hard stop 25%. Sizing 2%. Entry-bar spread + 0.5% slip.

## In-sample (NOT evidence): filtered n=49, mean +59%, median +49%, tail_ratio
8.3, win 90%, P(ret>20%) 67%, Sharpe 7.21. Unfiltered baseline: tail_ratio 2.52,
Sharpe 5.07, median +6.5%.

## Null
H0: the filtered long does NOT beat the unfiltered long on holdout tail ratio,
OR has holdout Sharpe ≤ 0.

## Test & verdict (CONFIRMED requires ALL)
1. n_filtered ≥ 15 holdout events (else DATA_INSUFFICIENT).
2. Holdout Sharpe > 0 and one-tailed sign-flip permutation p < 0.05 (2000 perms).
3. Holdout tail_ratio(filtered) > tail_ratio(unfiltered holdout baseline).
Else NOT_SIGNIFICANT. Report filtered vs unfiltered on all skew metrics.
Family note: convex-exit scan (7) + feature scan were in-sample; this is one
locked rule on untouched data. Nothing here changes after commit.
