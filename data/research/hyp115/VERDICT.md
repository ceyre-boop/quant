# HYP-115 — the incumbent as hypothesis: equal-weight ten-ETF basket

**VERDICT: FRAGILE.** Sealed `f1b66afc75d792d1` before any pre-2014 close was read; verified before and
after; ledger `ADJUDICATED`. One run.

| | in-sample 2015–2026 | **out-of-sample 2007-06 → 2014-12** |
|---|---|---|
| basket Sharpe (daily EW) | 0.628 | **0.219**, block-bootstrap CI **[−0.35, +0.88]** → c1 FAIL |
| basket total / maxDD | +189% / −32.9% | +41.5% / **−54.4%** |
| SPY | 0.731 | 0.266, maxDD −59.6% |
| 60/40 SPY/TLT | — | **0.587**, maxDD −34.6% |
| monthly rebalance, 3 bp | — | 0.272 |
| percentile vs 10,000 random 10-ETF baskets | 84.9 | 73.8 → not lucky |
| stress maxDD basket / SPY / 60-40 | GFC −54 / −60 / −35 · COVID −33 / −36 / −20 · 2022 −22 / −26 / −28 | c3 PASS (beats SPY, not 60/40) |

Note: yfinance auto-adjusted (total-return) closes give an in-sample Sharpe of 0.63 where the
price-only `daily_universe` cache gave 0.50 for the same basket; the construction is identical.

## What it means

**It is not a lucky basket (73rd percentile OOS) and it is not a validated core.** Out of sample it is
a diversified beta basket with a 54% drawdown, a Sharpe indistinguishable from zero over seven years,
and it loses to 60/40 on every risk measure in every stress window. It beat SPY in only 4 of the last
13 calendar years. It survived every overlay on 2026-09-02 because the overlays were nothing, not
because the basket is something.

The honest description of "the incumbent" is: *ten liquid ETFs, held, which is mostly equity beta with
some bonds and gold.* Its OOS per-ETF contributions say the same — QQQ, TLT and GLD carried it;
EFA/EEM/XLF contributed nothing or less through the GFC.

## What this says about the program

The hurdle every 2026-09 test had to clear was itself unproven. That does not resurrect any of the
overlays — they were measured as deltas, and a delta against a fragile base is still a null. It does
mean the desk's only validated equity construction is *none*. The proposal that produced this study
("validated core or dodged a fitted basket") got its answer: dodged.

## Constraints honoured

One run. No weight, pool, window or rebalance rule changed after the result.
