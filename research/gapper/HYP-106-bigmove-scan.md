# HYP-106 — The Method in the Madness: Predicting Which Gappers Run

**A leak-free, pre-entry filter that catches the big-% intraday moves — CONFIRMED
out of sample. The signal is real. The backtest magnitude is not tradeable as-is.**

## The discovery
Among ≥100% gappers, the ones that RUN long from 09:31 are the **moderate
overnight gap + low first-minute volume** names — still building — NOT the
extreme-gap, climax-volume blow-offs, which are exhausted and fade (exactly
HYP-093's short targets). It's the front half of the same ramp-then-fade
phenomenon, and it's *separable at 09:31 using only information available then.*

Leak-free features (from the 09:30 first-minute bar + prev_close only — every
`*_1030` field is banned because it's measured at 10:30):
| feature | winners | rest | MW p |
|---|---|---|---|
| overnight_gap | +60% | +163% | ~0 |
| first-minute volume | lower | higher | ~0 |
| first-minute range | tighter | wider | 0.0006 |

Leak-free RF cross-val accuracy **81%** (base rate 34%) — a genuine lift, and
interpretable. (Contrast HYP-105's 92% ML, which was rejected as look-ahead: it
used intraday_push measured *through* the holding window.)

## Holdout verdict (locked filter, run once — prereg 9d1c3937, commit 422687d)
Filter: overnight_gap ≤ 1.2395 AND log10(first-min vol) ≤ 5.4330 (frozen from
in-sample). Entry 09:31 long, exit 10:30, 25% stop.

| Metric | Unfiltered (71) | **Filtered (22)** |
|---|---|---|
| Median / event | +7.9% | **+67.7%** |
| Mean / event | +28.5% | +84.1% |
| Tail ratio (avg win / avg loss) | 3.95 | **10.5** |
| Win rate | 56.3% | **86.4%** |
| P(ret > +20%) | 38% | **77%** |
| Sharpe | 3.63 | 4.49 |
| perm p | — | **0.0005** |

**CONFIRMED** — improves every skew metric on untouched data; passes all prereg
gates. Ledger 85. This is exactly the positive-skew profile requested: wins
consistently and massively larger than losses (10:1 tail ratio).

## Why the magnitude is NOT a live P&L (read this before believing +67%)
The signal direction is real and confirmed. The RETURN SIZE is a backtest
artifact of optimistic microcap fills:
1. **Halts.** ≥100% gappers hit LULD halts constantly. You often cannot enter at
   09:31 or exit at 10:30 at the printed price.
2. **Spreads.** Real 09:31 bid/ask on these names is 5–20%, not the entry-bar
   (high−low)/2 the engine charges. Every trade is worth far less live.
3. **Size.** These are thin. A "+67% median" evaporates the moment you need
   real notional — the print you're filling against is a few hundred shares.
4. **n=22, 2.7 months, fat-tailed** (max win +338%). Small, tail-driven sample.
5. Not independent of HYP-093/105 — same events, same phenomenon.

Honest read: **live, this is likely a positive-expectancy, high-skew edge with
returns a fraction of the backtest** — still potentially excellent (and long, so
NO borrow wall, unlike the fade), but the +67% is fantasy until proven against a
realistic halt/spread fill model and real size.

## Next (before any capital)
1. Forward shadow at the 09:31 entry with the locked filter — real prints.
2. A realistic fill model: LULD halt logic, quoted-spread costs, size caps.
3. Capacity study: at what notional does the edge survive the book?
Files: `backtester/hyp106_skew.py`, `data/scan_results/hyp106_insample.json`,
prereg + results JSON/MD.
