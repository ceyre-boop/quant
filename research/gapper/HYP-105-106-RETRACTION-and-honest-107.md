# RETRACTION of HYP-105 / HYP-106 + the honest signal underneath (HYP-107)

## What went wrong (self-caught 2026-07-17)
HYP-105 (long gapper momentum, "Sharpe 3.6") and HYP-106 (the runner filter,
"median +67%, tail 10:1") are **REFUTED — fatal look-ahead selection bias.**

The event universe was defined by `gain_1030 ≥ 100%` — the price **at 10:30**.
But the long strategy **enters at 09:31**, an hour before that condition is
knowable. The 234 events are 255 winners hand-picked from 1,475 candidates by
their 10:30 outcome; the 1,220 stocks that gapped but did NOT reach +100% by
10:30 were never in the test (only 1 had cached bars). Buying at 09:31 the
stocks we already know mooned by 10:30 is circular. The realistic-fill model
"survived" because a look-ahead +50% dwarfs any spread — it was validating a
biased number. The prior CONFIRMED verdicts are void.

## The honest reconstruction
Fetched minute bars for all 1,475 candidates (incl. the non-runners) and
re-selected the universe using ONLY 09:31-available info (overnight gap at the
open), then applied the same filter with a clean 70/30 date split.

**Blindly buying morning gappers is a loser** (select on overnight gap only):
| selection @09:31 | n | median 09:31→10:30 | win |
|---|---|---|---|
| overnight_gap ≥ 30% | 896 | −0.3% | 47% |
| overnight_gap ≥ 50% | 555 | −3.5% | 38% |
| overnight_gap ≥ 100% | 231 | −8.0% | 32% |

Bigger gap = worse long — that's just the HYP-093 fade from the other side.

**But the filter (moderate gap + low first-minute volume) still holds out of
sample** — thresholds frozen in-sample, tested on untouched holdout:
| honest holdout | n | median | mean | win | tail | perm p |
|---|---|---|---|---|---|---|
| all gap-ups | 269 | −0.2% | +5.6% | 47% | 2.4 | — |
| **filtered (HYP-107)** | 57 | **+5.4%** | +15.3% | 70% | 4.4 | **0.0005** |

The filter turns a break-even universe into a real positive-skew signal — but
the honest magnitude is **~+5% median/trade, not +50–67%.** The look-ahead
inflated it ~10×.

## HYP-107 status: real gross signal, execution UNRESOLVED
The de-biased edge is genuine on GROSS returns (holdout p=0.0005) and
mechanically sensible: among morning gappers, the moderate-gap + quiet-open
names continue into late morning while the extreme climax gaps fade. But at
~+5% gross it is fragile to the exact frictions that didn't matter at +50%:
- 09:31 fills on thin microcaps (spread 1–15%) eat a large fraction of a +5%
  edge — possibly all of it.
- LULD halts genuinely reduce the enterable set (our halt detector over-flags
  normal >10%/min open volatility as halts, so the precise net is unresolved —
  it needs real LULD data, not bar heuristics).
Net: **plausibly a small real edge, plausibly unviable after costs.** Only a
live/paper forward shadow at 09:31 with real fills resolves it. Not tradeable on
this evidence.

## Lessons locked
1. Selection criteria must use ONLY information available at entry time. Any
   filter measured after entry (here: gain_1030 at 10:30) is look-ahead even if
   the returns themselves are computed correctly.
2. A friction model that "survives" a huge edge proves nothing — re-test it
   against the honest, de-biased edge where costs actually bite.
3. The gapper LONG is not a standalone edge; the confirmed direction remains the
   SHORT fade (HYP-093), entered at 10:30 after the condition is realized.
