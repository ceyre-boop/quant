# TICK-036 — Top-3 Movers Study (2024-07 → 2026-06, 492 trading days)
**STAMP: MINING — lookahead allowed; characterization, not evidence. Any rule here that gets traded must first pass its own prereg (HYP-099+) with the deflation counter (14 lens-cuts recorded).**

## The headline: yes, the top-3 are partially predictable ex-ante
A 20-name morning watchlist built from a FIXED, unfitted score (recent-runner flags + volume
surge + prior-day strength + yesterday's top-50 membership) contains **at least one of that
day's top-3 movers on 26.8% of days vs 2.0% for a random 20-name list — a 13× lift** (n=492
days). Not "more times than not" — but far from random, from tape features alone.

## Lens findings
1. **Anatomy**: median top-3 gain **+90%** (p90 +235%); median price **$3.80** (controls ~$10 —
   cheap stocks are the arena); 54% gap-led vs intraday-grind; day-of-week flat (no Monday magic).
2. **Ex-ante features (top-3 vs matched controls)**: prior-day/5-day returns ≈ FLAT at the
   median — the explosion is NOT preceded by median drift. What differs: **runner history**
   (29% were top-100 movers within 5 days vs 15% of controls; 44% within 20d vs 35%) and a
   mild prior-day volume tell (vol ratio 1.25 vs 0.91). The signal is WHO they are, not what
   they did yesterday on price.
3. **Persistence**: 2.8% of top-3 were in YESTERDAY's top-3 (28× the control rate but rare);
   **11.9% were in yesterday's top-50 (6× lift)** — attention momentum is real. 946 unique
   tickers fill 1,476 slots: most names appear once; a hard core repeats.
4. **Blue moon — the serial runners**: BNAI (8 top-3 days in 2 years), HCWB & SGBX (7),
   PRTG/ASST/AEHL/RGC/SDOT (6), VRAX/MNTS/ATPC/ATNF/SOPA/BIAF/EPSM (5). Fifteen tickers =
   ~6% of all top-3 slots. These are the "once in a blue moon, but you know the moon" names —
   a standing serial-runner list is itself an edge candidate.
5. **Regime**: top-1 magnitude barely varies by VIX tercile (median +136-156%) — mover
   violence is supply-driven (floats/dilution), not market-fear-driven.
6. **Catalyst mix**: deferred to the news pass (follow-up; taxonomy ready) — the gapper-subset
   evidence (TICK-029) says FDA/no-news dominate and M&A behaves differently.
7. **ICARUS link**: the top-3 population IS the parabolic-fade feedstock — this study's
   watchlist could pre-warm the ICARUS scanner (candidate quality, not signal change).

## Data + hygiene notes
Window = the two on-disk full-market years (Polygon free tier's 2-year wall; 5-10yr rerun is
one $79/mo approval away — code is window-agnostic). Nasdaq TEST SYMBOLS (ZVZZT-class) found
contaminating the first pass and excluded. Survivorship-free (delisted included). Controls =
same-day random names ≤$25, vol≥500K, outside the top-100.

## What would make this tradeable (next rounds, each via prereg)
(a) Fit + freeze the watchlist score on this window, prereg P(hit) on FORWARD data only;
(b) serial-runner standing list → event study on their NEXT appearance; (c) the news pass.
