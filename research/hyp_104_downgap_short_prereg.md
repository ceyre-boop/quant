# HYP-104 Pre-Registration — Down-Gap Continuation Short (liquid equities/ETFs)

Registered 2026-07-17, BEFORE any holdout (2025-07-17..2026-07-17) bar was read
for this rule. Emerged from the 77,016-hypothesis megascan
(`data/scan_results/megascan_20260717.parquet`). Down-gap-short was the only
coherent cluster beating the benchmark; it survives removing leveraged/vol ETFs
(so it is genuine gap-continuation, not vol-decay harvesting).

## Locked rule (single values — nothing floats)
- Universe: the 111-ticker liquid daily universe in `data/cache/daily_universe/`
  MINUS leveraged/inverse/vol ETFs {UVXY, VXX, SVXY, TQQQ, SQQQ, SPXL, SPXS,
  TNA, TZA, SOXL, SOXS, ARKK, GDXJ}.
- Signal: a session whose OPEN gaps DOWN >= 5.0% vs the prior session CLOSE.
- Entry: SHORT at the NEXT session's open (full gap day observed first — no
  look-ahead).
- Stop: 15% adverse (cover if intraday high >= entry*1.15; gap-through fills at
  that bar's open, never at trigger).
- Exit: cover at the close `hold_days = 2` sessions after entry, or stop.
- Cost: 5 bps per side. Sizing 10% notional per trade (pooled across universe).

## Dirty-window performance (2014..2025-07-17, NOT evidence)
annual +21.0%, Sharpe 2.32, max_dd 7.7%, n=1238 (~111/yr), win 54.7%.

## Benchmark to beat (all four, on the holdout)
annual > 18%, Sharpe > 2.0, max_dd < 15%, trades >= 50/yr.

## Holdout (untouched until this file is committed)
2025-07-17 .. 2026-07-17, same universe & rule.

## Null and test
H0: holdout mean per-trade return <= 0 (no continuation edge).
Test: one-tailed sign-flip permutation on the holdout per-trade returns (2000
perms) for the Sharpe null; report p. Secondary: bootstrap CI on mean.

## Verdict rule (CONFIRMED requires ALL)
1. Holdout beats the benchmark on all four metrics.
2. Permutation p < 0.05.
3. n >= 50 holdout trades (else DATA_INSUFFICIENT).
Family-correction note: 77,016 hypotheses were scanned; none survived Bonferroni
on the dirty window. This holdout test is the honest arbiter — a single locked
rule on data never used for selection, where the 77k multiplicity does not
apply. If it fails, the megascan's clear answer is "nothing beats the gapper
benchmark out of sample."

## Practical note (why this one is interesting vs the gapper fade)
Trades liquid ETFs/large caps — borrow is EASY, so the locate wall that caps the
gapper fade (TICK-032/037) does NOT bind here. That alone makes a Sharpe-2
version of this far more scalable than the +18% gapper benchmark.
