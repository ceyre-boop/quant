# HYP-103 Pre-Registration — EV-Optimal Prop-Challenge Operating Configuration

Registered 2026-07-17. Derived from a DELIBERATELY MINED 240-cell grid
(ev_scan_HYP103.json) on the dirty 234-event forward year. This prereg is the
only thing that can turn it into evidence. Live shadow data
(data/research/yield_frontier/shadow/) was not used for any optimization.

## Step-2 pattern (what the top-20 share)
All 20: sizing at grid max; account $200k (fees sublinear in size); window
unlimited >= 90d in every paired comparison; entry 10:30/10:45 only (11:00+
EV collapses); stop 20-35% plateau. Survival-adjusted EV (funded-year 10% DD
death modeled) confirms the same ordering — p(survive year) 96-99.9%.

## FROZEN OPERATING SPEC (the registered configuration)
- Entry: 10:45 ET (bar-open at/after 10:45), HYP-093 frozen signal filters
- Stop: 25% adverse from entry, post-entry high basis; fill entry*1.25+slip
- Exit: EOD close
- Sizing: 3.5% notional per event
- Frictions: slip 0.005/side, locate 0.50*APR(gain)/252, APR {2,4,6} tiers
- Vehicle terms: $200k account, +8% target, no time limit, -10% static max DD,
  fee $995 (FunderPro 2026 pricing), payout 0.80
- Chosen over the raw-EV-max cell (5% sizing) deliberately: 5% sits on the
  grid edge (EV still rising = classic mining artifact) and doubles the locate
  footprint. 3.5% is one step interior with p_bust 0.5% and EV within 0.65x.
- Dirty-data estimates for this spec (NOT evidence): annual +63.7%,
  P(pass, unlimited) 0.995, P(bust) 0.005, EV_y1 ~$103k survival-adjusted.

## Null
H0: EV_year1 of this configuration, estimated from live shadow events,
is <= $5,000.

## Evaluation (once, at N>=40 shadow events or 2027-01-16, whichever first)
1. Score each shadow event under this spec: re-price entry at 10:45 bar open
   and post-entry high from Alpaca SIP minute bars for the shadow event's
   (date, ticker) — the shadow logs 10:30 entries, so re-pricing is mechanical
   and lookahead-free (event selection already fixed by the shadow log).
2. Build the shadow empirical day-P&L distribution at 3.5% sizing; MC 100k
   paths (seed 42) -> P(pass) under (+8%, -10% DD, no limit, 2520-day cap).
3. annual_return = compounded shadow day P&L annualized by event-day density;
   EV_y1 = P(pass) * 200000 * annual_return * 0.80 - 995 / P(pass).
4. CONFIRMED if EV_y1 > $5,000 with P(pass) >= 0.90; else NOT_CONFIRMED.
   DATA_INSUFFICIENT if < 40 events by 2027-07-17 (12-month hard cap).

## Standing caveats (travel with any verdict)
Locate assumed available per HYP-093 frictions only; TICK-032 instrument wall:
EV is conditional on a funded vehicle that can actually short US micro-caps —
FunderPro's CFD instrument list has NOT been verified to include them.
Nothing in this file changes after its commit.
