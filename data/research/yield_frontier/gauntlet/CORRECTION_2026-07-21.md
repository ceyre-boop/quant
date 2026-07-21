# Correction Note — Gauntlet Report Decimal Error
**Filed:** 2026-07-21 · **Sealed artifact affected:** report.md (line 19) · **No revision to sealed file**

## The error

`report.md` line 19 reads:

> "The mining number (+3.8%/day) was, as expected, a selection fantasy"

The yield board column header is **net %/day**. The mined value for HYP-093 (F-EQ2_fade_short,
thr0.5|stop0.3|close|mna_excl=True, rank #4) is **+0.0380 %/day**, not +3.8 %/day.
The decimal point is missing. This is a transcription error, not a computation error.
`verdicts.json` and `yield_board.csv` are both correct.

## What the numbers actually say

| Stage | Value | Source |
|---|---|---|
| Mined (809-config sweep) | +0.0380 %/day | yield_board.md rank #4 |
| Holdout (constitutional sizing) | +0.0230 %/day | verdicts.json HYP-093 |
| Ratio (OOS retention) | **61%** | 0.023 / 0.038 |
| Haircut | **1.65×** | selection effect |

## Why this matters

As written, "3.8 → 0.023" implies a **165× collapse** — an order of magnitude worse than the
actual result and characteristic of severe data-mining overfitting.

The true finding is the opposite: after an 809-configuration sweep with a full deflated-Sharpe
penalty (DSR 0.987) and BH family correction, HYP-093 retained **61% of its mined yield
out-of-sample**. That is an unusually small selection effect. McLean-Pontiff (2016) document
~58% decay post-publication for academic anomalies (with far lighter multiple-testing loads).
This shop, running 809 configurations with DSR applied, lost 39%. The method is working.

The line was intended to flag that the mined number is inflated by selection — which is correct
in direction — but the magnitude stated is off by 100×. The instinct ("very little bias") in
the surrounding analysis is accurate; the supporting number contradicted it due to the typo.

## Standing rule

Sealed verdicts are final. `report.md` is not modified. This note is the authoritative
correction and should be read alongside any citation of the +3.8%/day figure.
