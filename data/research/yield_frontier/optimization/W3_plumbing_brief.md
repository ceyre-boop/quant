# W3 — Short-Side Plumbing & Regulation (specialist agent brief, 2026-07-13)
Stamp: research input to TICK-033; modeling constants for W6; NOT evidence.

## Simulator constants card (all primary-source verified July 2026)
| Constant | Value | Basis |
|---|---|---|
| SSR state | **DETERMINISTIC boolean from our own tape**: SSR_active(D) = (min RTH trade(D-1) <= 0.9*close(D-2)) OR (min RTH trade(D) <= 0.9*close(D-1)); day-2 runners often ALREADY on SSR at open | SEC Rule 201 FAQ (upd. 2026-06-26) |
| SSR entry effect | passive-only: post above NBB, fill-on-uptick, P(fill)<1, adverse delay (filled on bounces, miss the flushes); constrains 10:30 entry on day-2+ gappers and re-entries after intraday trigger | Rule 201(b) |
| LULD bands | fixed ALL DAY by PREVIOUS close: Tier2 >$3 = 10%; $0.75-3 = 20% (40% after 15:35); <$0.75 = min($0.15,75%). No opening doubling since Amdt 18 (2019) | Nasdaq LULD FAQ |
| Halt/reopen | 15s limit state -> 5-min pause; reopen collar = band*(1+0.05k) per 5-min extension, imbalance side only, UNBOUNDED in k; pause >=15:50 -> into closing cross | Amendment 12 |
| Gap-through bound | +30% stop reachable via TWO consecutive limit-up cycles on 10% bands; ONE halt on 20% bands (prev close $0.75-3) — ties to prev-close price tier | Amdt 12 math + DERA |
| Cascades | no daily cap; DERA: Tier-2 phenomenon, post-reopen REVERSION is the norm, cascades the tail (GME ~20 halts documented) | SEC DERA white paper |
| Locate fee | per-share upfront dynamic: ~$0.01-0.05/sh typical HTB, fat tail >=$0.10-0.30; 10:30 >= premarket price; availability can be 0; **fresh locate required to re-short after a cover on HTB/threshold names** (FINRA/SEC FAQ 4.4) | Cobra/TradeZero/Guardian/FINRA |
| IBKR model | no per-share locate; borrow accrues only at settlement -> same-day round trip = $0 borrow, but availability gate at 10:30; pre-borrow $20+accrual | IBKR docs |
| Overnight borrow | 0 for cover-by-close; T+1: accrues next morning if held; past Thu = 3 nights | TradeZero |
| Same-day buy-in | P~0 regulatory/loan (nothing settles intraday); residuals: broker risk-desk (tail, unmodeled) + Rule 204(b) penalty-box = symbol flips pre-borrow-only mid-day (blocks ENTRIES on crowded names) | Rule 204, IBKR |
| T+1 (2024-05-28) | 204 close-outs a day earlier (penalty-box arrives sooner on crowded symbols); locate rule unchanged; SSR/LULD unchanged | SEC adopting release |

## The load-bearing integration flag
SSR-passive-entry and locate-availability bind HARDEST on the best events (day-2 runners,
low-float squeezers) — **adversely selected against the edge, not iid frictions**. The W6 spec
must model this correlation explicitly, and any live-candidate prereg should compute SSR
per-event from historical tape (it's deterministic — we have the data).
