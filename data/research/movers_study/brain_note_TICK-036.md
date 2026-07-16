---
title: TICK-036 — Top-3 Movers Are Partially Predictable
date: 2026-07-15
tags: [research, mining, movers, watchlist, icarus, unconfirmed]
ticket: TICK-036
status: MINING — hypothesis-generating, NOT evidence
window: 2024-07 → 2026-06 (492 trading days, survivorship-free)
related: ["[[HYP-099]]", "[[ICARUS]]", "[[TICK-029 Gapper Fade]]", "[[TICK-022 Prop Funnel]]"]
---

# TICK-036 — The Day's Top-3 Movers Are Partially Predictable

> **STAMP: MINING.** Look-ahead deliberately on; this is *characterization*, not evidence. Every
> rule below must clear its own pre-registration ([[HYP-099]]+) on forward-only data, with the
> deflation counter set to all 14 lens-cuts, before it is ever traded.

## One-line finding
A **fixed, unfitted** 20-name morning watchlist contains **≥1 of that day's top-3 movers on 26.8%
of days vs 2.0% for a random list — a 13× lift**, from tape features alone (n=492).
Score = `2·runner5d + runner20d + 2·(volx>5) + 2·(prior1d>15%) + y_top50`.

## The core insight: the tell is *identity*, not momentum
The day before a top-3 explosion, price is **flat at the median** — no drift warns you
(prior-1d −0.3%, prior-5d −0.2%, ≈ controls). What separates future movers from controls is **who
they are**, not what they did yesterday:

| Ex-ante feature | Top-3 | Controls | Lift |
|---|---|---|---|
| Was a top-100 mover within 5d | 28.6% | 14.9% | ~2× |
| In yesterday's top-50 | 11.9% | 1.9% | **6×** |
| In yesterday's top-3 | 2.8% | 0.1% | 28× (rare) |
| Prior-day volume ratio | 1.25 | 0.91 | mild |
| Prior-day return (median) | −0.3% | 0.0% | flat |

**Runners run.** Attention momentum and dilution history predict; price momentum does not.

## Anatomy of the arena
Median top-3 mover: **+90% on the day** (p90 +235%), trading at **$3.80** (controls ~$10) —
it's cheap stocks, always. 54% gap-led / 46% intraday grind. No day-of-week effect.

## The blue moon has a cast list
Fifteen serial tickers fill **~6% of all top-3 slots** across two years — a standing set of
dilution machines, not randomness:

`BNAI ×8 · HCWB ×7 · SGBX ×7 · PRTG ×6 · ASST ×6 · AEHL ×6 · RGC ×6 · SDOT ×6 · VRAX ×5 · MNTS ×5 · ATPC ×5 · ATNF ×5 · SOPA ×5 · BIAF ×5 · EPSM ×5`

A **serial-runner standing list is itself an edge candidate** (→ [[HYP-099]] event study).

## Regime doesn't matter
Top-1 mover magnitude is ~flat across VIX terciles (+136% to +156%). Mover violence is
**float-and-dilution-driven, not fear-driven** — decoupled from macro regime.

## What it feeds
This top-3 population **is** the parabolic-fade feedstock. Knowing tomorrow's likely suspects
pre-warms the [[ICARUS]] fade scanner — candidate quality, not a signal change. Note the standing
caveat from [[TICK-022 Prop Funnel]]: these equity edges do **not** map to the futures prop funnel;
venue/asset-class match is unresolved.

## Hygiene
Survivorship-free (delisted included). Nasdaq test symbols (ZVZZT-class) caught contaminating the
first pass, now filtered. Controls = same-day random names ≤$25, vol ≥500K, outside top-100.
**Data wall:** only the 2 on-disk full-market years (Polygon free tier); 5–10yr rerun is one
$79/mo approval away — code is window-agnostic.

## Path to tradeable (each via its own prereg)
1. **Fit + freeze** the watchlist score on this window → prereg P(hit) on **forward data only**.
2. **Serial-runner list** → event study on each name's *next* appearance.
3. **News-catalyst pass** (taxonomy ready; [[TICK-029 Gapper Fade]] says FDA/no-news dominate,
   M&A behaves differently).

Source: `data/research/movers_study/report.md` · `results.json` · pushed.
