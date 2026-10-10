# THS-001B — AI infrastructure + AI-health sleeve (managed RIA account)

**Status: LIVE (retro-registered).** Sealed 2026-10-10, hash `d7923706…`. Same thesis as THS-001;
this is the managed Public `3CQ09610` sleeve, where the manager rebalanced monthly. Basket sealed
as the first-day (2026-03-31) fill weights: PLTR 0.22, CRWV 0.18, NBIS 0.14, AMD 0.12, RVMD 0.10,
TEM 0.08, GEHC 0.08, SMH 0.05, REGN 0.03. Invalidation identical to THS-001.

## Scorecard (the sealed basket, held) — `output/scorecard_2026-10-09.json`
2026-03-31 → 2026-10-09 (192 d): basket **+66.5%**, SPY +20.3%, excess +46.1%, maxDD −25.3%
(came within 5 points of the 30% invalidation). Marks: 1m +27.2% / 3m +59.9% / 6m +70.1% vs SPY
+10.5 / +15.1 / +17.9. AMD +199%, NBIS +113%, RVMD +95%, SMH +57%, TEM +57%, PLTR +43%; GEHC −7%,
REGN −3%, CRWV +6%.

## Reconciliation (what the account actually did) — `output/reconciliation_3CQ09610.json`
**This is where the "140%" lives.** Ending value ÷ net contributed − 1 = **+140.7%**. That ratio
is not a return: $4,300 of the $15,800 withdrawn was gains taken out on 2026-08-20 and moved to the
margin account, which shrinks the denominator. The measured numbers:

| | account | SPY same window |
|---|---|---|
| TWR 2026-03-30 → 10-09 | **+74.5%** | +23.8% |
| MWR (Modified Dietz) | +68.3% | |
| net gain on $1,400 net contributed | $1,970 | |
| AI names sub-basket, money-weighted | +78.5% | +20.3% |
| health names sub-basket, money-weighted | +43.3% | +20.3% |

+74.5% in six months against +23.8% is the real result, and it is 3× the index, in the same
direction as the sealed-basket score (+66.5%). The excess over SPY is the thesis; the gap between
+74.5% and +66.5% is the manager's trading, which added a little this time.

Caveat recorded: the API history omitted one buy batch (~$3,815, 9 names). It was reconstructed
from broker positions minus fills at the 2026-10-01 close (rows flagged `reconstructed: true`);
modelled ending value $3,370 vs broker $3,365.

## What closes this folder
Same as THS-001. Next mark: 12 months (2027-03-31).
