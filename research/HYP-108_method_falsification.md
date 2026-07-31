# HYP-108 — the test cannot support the claim (method falsification)

**Date:** 2026-07-30
**Status:** blocking objection to the HYP-108 methodology, filed before RQ-REST-013 is built
**Scope:** read-only. No execution-path or config change. No ledger verdict altered.

## Summary

HYP-108 reports OOS Sharpe 1.285 → 3.413, permutation p=0.0000 (10,000 trials), bootstrap
90% CI [2.97, 3.92], all 10 years positive. The entry already carries the caveat that time-exit
pnl is estimated rather than re-simulated, and that RQ-REST-013 is required before a sealed
CONFIRMED verdict.

That caveat understates the problem. **The permutation p-value does not test the claim, and a
placebo unrelated to trailing stops scores higher than the reported result.** The design cannot
distinguish "trailing stops destroy the carry edge" from "replacing losing trades with a
positive constant raises Sharpe."

## Reproduction

Baseline reproduces closely (1.353 here vs 1.285 reported; the small gap is date-span /
cost-treatment, not material to the argument). Sharpe is per-trade mean/std annualised by
sqrt(trades per year), 411 trades over 9.92 years.

| Variant | Sharpe | std |
|---|---|---|
| Baseline (unmodified) | 1.353 | 0.01296 |
| **A — HYP-108 method:** replace the 118 `trailing_stop` trades with per-pair time-exit mean × 0.70 | **3.499** | 0.01056 |
| **B — PLACEBO:** replace the 118 **worst trades regardless of exit reason** with the same value | **4.948** | 0.00964 |
| **C — PLACEBO:** replace the 118 worst **non-trailing** trades with the same value | **2.952** | 0.01188 |

A reproduces the reported 3.413. **B, which ignores exit reason entirely, scores 4.948 — higher
than the hypothesis it is supposed to falsify.** C, which explicitly excludes every trailing-stop
trade, still lifts Sharpe from 1.353 to 2.952.

The lift is a property of the substitution, not of trailing stops.

## Why p=0.0000 was guaranteed

The stated null is "trailing-stopped trades are a random subset," tested by assigning the
time-exit value to 118 randomly chosen trades. Measured null distribution over 10,000 draws:

    mean 2.066   p95 2.311   max 2.599

Two things follow.

1. **Random replacement alone lifts Sharpe from 1.353 to 2.066.** The null already contains most
   of the effect, because substituting *any* 118 trades with a positive constant raises the mean
   and cuts the variance. The test only asks whether the observed beats ~2.6.

2. **A trailing stop fires precisely when price moves against the position.** `trailing_stop` is
   therefore a label selected on bad outcomes by construction — mean −0.406%, 23.7% WR. Replacing
   a set selected on bad outcomes will beat replacing a random set with probability ~1, for any
   dataset, under any strategy. p=0.0000 is an artifact of the labelling, not evidence about exits.

Applying the same permutation test to placebo B also returns p=0.0000. A test that certifies a
placebo it was designed to exclude is not measuring the mechanism.

## Variance collapse

Substituting a per-pair mean replaces 118 of 411 trades (29% of the sample) with a **4-valued
constant**:

    within the replaced block: std 0.011641 -> 0.001449

Decomposition of the 1.353 → 3.499 lift: mean improvement alone gives 2.851, variance reduction
alone gives 1.660; the variance collapse supplies roughly 14% of the total lift. The remaining
86% is the mean shift — which is itself the artifact of substituting winners' averages for
losers' realised outcomes, not an independently measured quantity.

The bootstrap CI [2.97, 3.92] inherits this: resampling a series that already contains 118
synthetic near-constant observations produces a tight interval around an artifact.

## Contradicting prior evidence

The 2026-07-01 ExitConfig sweep evaluated 180 exit configurations across 4 pairs, 2015–2024,
by **real re-simulation** rather than substitution, and found v015 at the global peak with
0/180 both-regime survivors and 0/180 FDR-significant. "Disable trail" was refuted on Sharpe in
that sweep. HYP-108 reaches the opposite conclusion using estimated pnl. Where a re-simulated
sweep and an estimate-based substitution disagree, the re-simulation is the stronger evidence.

## What would actually test the claim

RQ-REST-013 as described (real price-path re-simulation) is necessary but not sufficient. The
re-simulation must also:

1. Hold the entry set fixed and re-run the **actual price path** to the time-exit date, so exited
   trades receive their realised outcome rather than a group average.
2. Preserve per-trade dispersion — no substitution of any summary statistic for a realised pnl.
3. Use a null built on the **re-simulated** series, not on random relabelling of the original.
4. Report placebo B above alongside the result. If B still beats A after re-simulation, the
   finding is about loser removal, not exits.
5. Walk-forward the exit rule itself, not just the trade set — the per-pair time-exit mean is
   fitted on the same decade it is evaluated on, which is in-sample leakage independent of
   everything above.

## Recommendation

Do not build RQ-REST-013 on top of the current framing, and do not let the 3.413 figure
propagate into planning documents or the v015 Sharpe target. Treat HYP-108 as **method-blocked**
rather than promising-pending-confirmation. The underlying question — do trailing stops truncate
the macro drift — remains open and worth testing; it has simply not been tested yet.

## Reproduce

    python3 - <<'PY'
    import pandas as pd, numpy as np
    d=pd.read_csv('data/proof/backtest_trades_v015_2015_2024.csv')
    d['entry_date']=pd.to_datetime(d['entry_date'])
    yrs=(d['entry_date'].max()-d['entry_date'].min()).days/365.25; tpy=len(d)/yrs
    S=lambda p: np.asarray(p).mean()/np.asarray(p).std(ddof=1)*np.sqrt(tpy)
    sub=d[d.exit_reason=='time'].groupby('pair')['pnl_pct'].mean()*0.70
    a=d.copy(); m=a.exit_reason=='trailing_stop'; a.loc[m,'pnl_pct']=a.loc[m,'pair'].map(sub)
    b=d.copy(); w=b['pnl_pct'].nsmallest(118).index; b.loc[w,'pnl_pct']=b.loc[w,'pair'].map(sub)
    print(S(d.pnl_pct), S(a.pnl_pct), S(b.pnl_pct))   # 1.353  3.499  4.948
    PY
