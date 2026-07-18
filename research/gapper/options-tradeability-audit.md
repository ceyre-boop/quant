# Options-Chain Tradeability Audit — Confirmed Gapper Universe

**Verdict: DOOR CLOSED.** Only **5 of 234 events (2.1%)** had a tradeable ATM
call at 09:31. The call-overlay idea is not viable on the parabolic-gapper
universe. This is a fast, clean kill — it saves months of building an execution
layer for options that mostly don't exist.

## Method
234 confirmed gappers (HYP-093/107 universe, 2025-07→2026-06). For each, at the
09:31 entry: ThetaData (options-VALUE tier, ThetaTerminal :25503) nearest expiry
≥ gap date, ATM strike vs the 09:31 underlying, and the nearest two-sided call
quote in the 09:30–09:36 window. Tradeable = two-sided AND spread ≤ 20% of mid.

## Result
| Status | Count | % |
|---|---|---|
| **NO_OPTIONS** (no chain at all) | 169 | 72% |
| NO_LIVE_EXPIRY (options existed, none live — mostly delisted) | 34 | 15% |
| NO_QUOTE_0931 (chain exists, no two-sided quote at 09:31) | 18 | 8% |
| NOT_TRADEABLE (two-sided but spread > 20%) | 8 | 3% |
| **TRADEABLE** | **5** | **2.1%** |

Reconciles with the prior borrow study (169 NO_OPTIONS — exact match, no bug).

## By underlying price tier (the dominant factor)
| Tier | n | tradeable | % |
|---|---|---|---|
| < $1 | 1 | 0 | 0% |
| $1–5 | 81 | 0 | 0% |
| **$5–20** | 106 | 5 | **4.7%** |
| $20+ | 46 | 0 | 0% |

Even the best tier ($5–20) is 4.7% — nowhere near the 30% door-open threshold.
The 5 tradeable names: RGC, TMQ, GSIT, BYND, CMBM — all $5–12 underlyings,
spreads 3–16%. Sub-$5 microcaps (the bulk of the edge) have essentially no
options; $20+ names had chains but no two-sided quote at the 09:31 open.

## Why (and what it means)
Parabolic gappers are thin microcaps. Options markets don't make continuous
two-sided markets on them at the open — and 72% have no listed options at all.
The 09:31 first minute is even worse than 10:30 (which showed 6.8% two-sided;
with the spread filter at 09:31 it's 2.1%).

**The gapper edge and tradeable options do not coexist in the same names.** This
does NOT kill the options-convexity *idea* — it kills it for *this universe*. If
convex, defined-risk, no-borrow expression is the goal, it has to live on a
different universe that actually has liquid options (large-cap/ETF intraday
movers: SPY/QQQ/AAPL/TSLA/NVDA/COIN/MSTR…) — but those names don't carry the
gapper edge. That's the real trade-off to reckon with, not a data gap.

## Decision
- **Options overlay on the gapper universe: dead.** Do not build it.
- The confirmed short fade (HYP-093) stays equity-only, borrow-capped.
- The honest long direction (HYP-107) stays equity-only; convexity via options
  is off the table for these names.
- If the convexity goal persists, it's a *different research program* on a
  liquid-options universe — and the megascan already showed no standalone edge
  there (ORB/breakout dead). So the near-term money path remains: prove the
  confirmed equity edges survive live fills (the HYP-107 shadow + a real-time
  execution harness), not an options wrapper.

Data: `data/research/gapper/options_tradeability_0931.json` (per-event detail).
