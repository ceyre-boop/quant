# HYP-100 Pre-Registration — Stop Overlay on the Sealed Gapper Fade (forward data)

Registered: 2026-07-16. Purpose: the 25% post-entry stop that produced the
+24.4% forward-year sim was specified by session mandate, not by the original
HYP-093 prereg. This registers it formally against data that DOES NOT EXIST
YET (live shadow forward collection) — the cleanest possible test.

## Rule under test (frozen)
HYP-093 frozen signal (gain_1030 >= 1.00, price_1030 >= 2.00, cum_vol_1030 >=
500,000, M&A-catalyst excluded), short at 10:30 ET open, exit 15:45/EOD close,
PLUS: stop at 25% adverse from entry, evaluated on post-entry intraday high;
stop fill = entry * 1.25 * (1 + 0.005 slip).
Frictions verbatim HYP-093: slip 0.005/side, locate 0.50 * APR(gain)/252,
APR tiers {>=0.5: 2.00, >=1.0: 4.00, >=1.5: 6.00}.

## Data
Live shadow records (`data/research/yield_frontier/shadow/`), source-tagged
shadow_gapper, accumulating from 2026-07-13 forward. Events before this
prereg's commit (2026-07-13/14/15) are EXCLUDED. Evaluation window opens at
the LATER of: N >= 40 stopped-rule events, or 2027-01-16 (6 months). One
evaluation only.

## Null and verdict rule
H0: stopped-rule mean net event return <= 0.
Test: one-sided block bootstrap on daily constitutional returns (mean_block=5,
n_boot=10000, seed=42 — verbatim gauntlet_run.py machinery), p < 0.05.
Secondary (reported, not gating): stopped vs unstopped same-event comparison;
realized max drawdown of the 2%-notional curve vs the sim's 4.0%.
CONFIRMED requires: p < 0.05 AND total net return over the window > 0.
DATA_INSUFFICIENT if < 40 events by 2027-07-16 (12 months hard cap).
Nothing in this file changes after its commit; verdict seals to the ledger.
