# HYP-099 Pre-Registration — Regime-Conditional Gapper Fade ("intraday-built gap")

Registered: 2026-07-16 (before any holdout row dated >= 2026-01-01 was loaded).
Scan basis: Step-1 lookahead scan on 117 qualifying events 2025-07-02..2025-12-31
(`research/gapper/regime_scan_HYP099.json`, scan script committed alongside).
NOTHING in this file changes after the commit that introduces it. Verdict is
sealed by `regime_scan_hyp099_holdout.py` exactly as specified below.

## Mechanism hypothesis
Gappers whose move was built during the morning session (high intraday push)
rather than held from an overnight news gap are momentum-chase pumps and fade
harder after 10:30 ET. Overnight-gap-dominant events are news-anchored and
stick. (Scan evidence: intraday_push>median delta_median +6.9%, p=0.050 MW,
n=58; composite with low overnight_gap +11.3%.)

## Universe / event definition (verbatim HYP-093 frozen spec)
Rows of `data/research/gapper/per_candidate_enriched.csv` with:
gain_1030 >= 1.00, price_1030 >= 2.00, cum_vol_1030 >= 500000,
catalyst not containing any M&A term {merger, acquisition, acquire, buyout,
takeover, definitive agreement, letter of intent, strategic alternatives,
going private}. Fade return per event = -outcome_pct (short 10:30 open,
exit EOD close).

## Holdout
Events dated 2026-01-01 .. 2026-06-30 (never inspected before this commit).

## Pre-registered variants (k=2, BH alpha=0.05)
Thresholds FROZEN from scan-set medians; no re-fitting on holdout.
- V1: intraday_push > 0.195
- V2: intraday_push > 0.195 AND overnight_gap <= 1.19

## Frictions (verbatim HYP-093): net_event = fade − 2*0.005 − 0.50*APR(gain)/252
APR: 2.00 if gain>=0.5, 4.00 if gain>=1.0, 6.00 if gain>=1.5 (annualized frac,
i.e. 4.00 = 400%? NO — APR values are annual fractions as in live_shadow.py
constants: {0.5: 2.00, 1.0: 4.00, 1.5: 6.00} = 200%/400%/600% annualized
hard-to-borrow rates; per-day cost = 0.50 * APR/252).

## Null and test
H0: in-regime holdout net fade distribution is not greater than out-of-regime.
Test: one-sided Mann-Whitney (normal approximation, tie-corrected ranks as in
scan script), in-regime vs out-of-regime, per variant; BH correction across
k=2.

## Verdict rule (all conditions required for CONFIRMED)
1. n_in >= 15 in-regime holdout events (else DATA_INSUFFICIENT).
2. BH-adjusted one-sided p < 0.05.
3. In-regime median net fade >= out-of-regime median net fade + 0.03.
Else NOT_SIGNIFICANT. Economic floor reported separately: in-regime mean
%/day at constitutional sizing (0.0125 notional per event) vs 0.05%/day floor
sets VALID vs VALID_BUT_BELOW_FLOOR, exactly as HYP-093.
If two variants pass, V2 (the tighter composite) is the registered survivor.

## Pre-registered 1-year simulation (runs only if verdict CONFIRMED)
Full-year signal set 2025-07-02..2026-06-30, surviving variant only,
zero-lookahead rule application:
- Sizing 2% notional per event (stress, double constitutional); entry
  entry_open_1030; exit close_eod; stop 25% adverse (evaluated on daily high
  from Alpaca daily bars; missing high data => treated as STOPPED,
  conservative; stop fill at entry*1.25 + slip).
- Frictions as above. Track daily P&L, cumulative curve, max DD, Sharpe.
- Output research/gapper/sim_annual_HYP099.csv.

## Pre-registered fallback if NOT_SIGNIFICANT
Pivot to intraday entry-timing study as HYP-100 (new prereg, minute bars);
HYP-099 seals as-is with a written record.
