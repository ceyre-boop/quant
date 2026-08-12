# TICK-036 — Top-3 Movers Study: commonalities + ex-ante predictability (MINING, Steps 1-2)
**2026-07-14 · lookahead explicitly allowed and stamped · no holdout burned (characterization only; any surviving rule goes to its own prereg later)**

## Context

Colin: look at the top 3 movers every trading day for 5 years; find commonalities; find anything that predicts one of the 3 more often than chance — or rare high-confidence "blue moon" setups; examine under many lenses. This is Steps 1-2 of THE RESEARCH METHOD by name. **Data reality:** full-market daily history on disk covers 2024-07→2026-06 (the two Polygon grouped-cache years, ~500 trading days, survivorship-free incl. delisted); Polygon free tier cannot reach further back (2-year wall, verified). Plan: run the complete study on 2 years now at $0; the 5-10-year extension is one approved purchase away (Polygon Developer $79/mo per W5) and the code will be window-agnostic so it reruns unchanged on deeper data.

## Definitions (locked in the report header, mining-grade)

- Universe/day: common-stock filter (alpha, ≤5 chars, not 5-char W/R/U), prev close ≥ $0.75, day volume ≥ 500K.
- **Top-3 movers** = top 3 by close/prev_close gain (primary) and by high/prev_close (secondary cut), per day. ~1,500 mover-days per definition per year span.

## Lenses (each = one section of the report, all computed from the grouped caches unless noted)

1. **Anatomy**: distribution of top-3 magnitudes, price bands, dollar volume, day-of-week/month, gap-at-open vs intraday grind (open/prev vs close/open), how often top-3 = our ICARUS universe (≥50% by 10:30 — link via existing intraday cache where covered).
2. **Ex-ante tape features (the predictability core)**: for every top-3 name, its OWN prior 1/3/5-day return, volume ratio vs 20d mean, range expansion, price level, prior-runner flag (top-100 mover within last 5/20 days), days-since-last-run; versus matched controls (same price/volume band, same day). Output: feature lift table — P(top-3 | feature) / base rate.
3. **Persistence/attention momentum**: P(today's top-3 ∈ yesterday's top-N) for N=3,10,50; day-2/day-3 continuation of a top-3 appearance (the "runners run" question); repeat-offender ticker census (which names appear 5+ times — the serial pumpers).
4. **Regime**: VIX tercile (on-disk daily VIX), market up/down days, cluster days (do movers cluster after index shocks?), monthly counts (are mover-days seasonal?).
5. **Catalyst (sampled)**: Alpaca historical news for top-3 name-days (chunked; keyword taxonomy from posthoc_scan verbatim) — catalyst mix of the top-3 vs the gapper population; the no-news share.
6. **"Blue moon" conditional scans**: high-precision/low-frequency rules mined explicitly — e.g., "prior-day top-3 name that closed weak + reverse-split profile," "3rd day of a sector wave" (sector proxied by news keywords), "post-halt day." Report each candidate rule with n, precision, frequency, and a MINING stamp.
7. **The practical framing**: a daily ex-ante WATCHLIST rule (top-K by mined score) scored on P(≥1 of the day's top-3 in list) vs random-K baseline — the honest version of "predict one of the 3 more times than not."

## Build

`research/movers_study/` (isolated; imports whitelist as usual): `build_panel.py` (grouped caches → per-day top-3 + universe features panel, window-agnostic), `lenses.py` (sections 1-6), `watchlist.py` (section 7), report writer → `data/research/movers_study/report.md` + `panel.parquet`. Every artifact stamped MINING; candidate-rule count appended to `mined_n.json` (feeds any future prereg's deflation). Ticket TICK-036. News pass chunked ≤10-min foreground runs. Reuses: grouped cache loaders (gapper stage1/holdout shapes), posthoc catalyst taxonomy, VIX from `data/research/modern/spot_cache`.

## Out of scope
Any prereg/holdout test (that's a later HYP-099 if a rule glows); purchases (5-10yr extension awaits Colin's word); intraday features beyond the existing gapper-cache overlap; live wiring.

## Verification
Panel row-count sanity vs known trading days (~500); determinism rerun; report sections all populated with n's; mined-rule counter updated; module tests (panel build + one lens on synthetic data); pushed with NEXT.md entry.
