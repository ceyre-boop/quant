# HYP-102 — Gapper Continuation (long side): STOPPED AT STEP 1 — 2026-07-16

No prereg written; no 2026 holdout row touched. Dirty scan (2025-H2, 117
qualifying >=100% events: 89 faders / 28 continuers) per the mandate.

## Feature separation (continuers vs faders at 10:30)
Top separators by effect size (all p > 0.05 at n=28):
| Feature | Fader med | Continuer med | Effect | p (MW) |
|---|---|---|---|---|
| rvol_1030 | 36.0 | 80.3 | +123% | 0.118 |
| intraday_push | 0.209 | 0.054 | −74% | 0.176 |
| overnight_gap | 1.148 | 1.718 | +50% | 0.058 |

Directionally coherent story ("overnight-anchored, extreme-crowd gaps stick")
— but it is the SAME feature family (intraday_push / overnight_gap) that
already failed holdout transfer in HYP-099 on the fade side.

## Why no rule is preregistrable: every tradeable form LOSES in-sample
LONG at 10:30 → EOD, dirty data, frozen scan-median thresholds:
| Rule | n | LONG median | win rate |
|---|---|---|---|
| rvol>=50 | 54 | −13.5% | 30% |
| rvol>=80 | 46 | −15.6% | 30% |
| rvol>=100 | 39 | −10.4% | 33% |
| overnight_gap>=1.72 | 38 | −7.3% | 37% |
| intraday_push<=0.054 | 43 | −6.4% | 33% |
| og>=1.72 & ip<=0.054 | 28 | −5.1% | 39% |
| rvol>=80 & og>=1.72 | 13 | +5.9% | 62% |
| mins_to_high<=20 | 52 | −7.4% | 27% |

The 70/30 base rate dominates: the best honest features shift P(continue)
from 30% to only ~33–39%. The single positive cell (n=13, tenth rule tested)
is mining residue — and could not reach the mandated >=30 holdout events
anyway. Every viable-size cell still FADES on median (+5% to +16%), so any
long-switch rule would also subtract profitable shorts from the confirmed
edge. Double loss.

## Conclusion
The non-faders are not identifiable at 10:30 with the features available.
Continuation risk is why the 25% stop exists; it is not a harvestable second
edge in this dataset. Combined-signal path to 95% P(PASS) remains open only
via an INDEPENDENT signal family (different universe/catalyst, not gapper
re-slicing). Per the method: do not pre-register noise. Sealed as a Step-1
stop; revival requires new data or new features (e.g., L2/tape, borrow-pull
signals), not re-cuts of this table.
