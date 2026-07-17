# HYP-105 Pre-Registration — Long-Side Momentum on Parabolic Gappers

Registered 2026-07-17, before any holdout (events dated >= 2026-04-08) was read
with this rule. In-sample scan: `data/scan_results/hyp105_long_insample.parquet`
(315/360 configs; best Sharpe 5.07 in-sample; 0 survived Bonferroni×360 — the
holdout is the honest arbiter for the single locked config below).

## Locked config (single values)
- Universe: the 234 minute-bar-ready HYP-093 gappers (gain_1030≥1.0, price≥2,
  vol≥500k, M&A excluded). Holdout = the 71 events dated 2026-04-08..2026-06-30.
- Direction: LONG.
- Entry: 09:31 ET bar open.
- Exit: 10:30 ET bar close.
- Hard stop: 25% adverse (long: cover if low ≤ entry·0.75; gap-through fills at
  breaching-bar open, never at trigger).
- Sizing: 2% notional. Frictions: entry-bar spread + 0.5% slippage.

## In-sample (NOT evidence): annual +64.5%, Sharpe 5.07, win 57.1%, median +6.5%, n=163.

## Null
H0: long-side momentum on parabolic gappers produces **Sharpe ≤ 0 on holdout.**

## Test
One-tailed sign-flip permutation (2000 perms) on the 71 holdout per-event
returns; reject H0 if holdout Sharpe > 0 AND permutation p < 0.05.

## Verdict rule
- CONFIRMED: holdout Sharpe > 0 and p < 0.05 (and n ≥ 30).
- NOT_SIGNIFICANT otherwise. DATA_INSUFFICIENT if n < 30.
Family note: 360 configs scanned; 0 survived Bonferroni in-sample. This is one
locked rule on untouched data — the 360-multiplicity does not apply to it.

Nothing in this file changes after commit.
