# TICK-039 — Spread model recalibration + HYP-107 re-measurement

**Date:** 2026-07-18 · **Depends on:** TICK-038 (harness, commit 36b3902)
**Verdict emitted: NONE.** HYP-107 remains `REAL_BUT_MARGINAL_EXECUTION_UNRESOLVED`.

Per operator decision, this ticket measures and reports. The verdict rule is set
afterwards, with these numbers in hand, rather than pre-committed to a threshold
chosen while the result was already expected to be favourable.

---

## 1. HEADLINE — two findings that point in opposite directions

**A. The spread model was overcharging by 11.3×.** Confirmed on independent data.

**B. It does not matter as much as it looks, because spread was never the binding
constraint.** After Article 1 sizing the constitutional yield still lands BELOW
the 0.05%/day floor — the same place HYP-093 landed.

---

## 2. The spread measurement (robust — independent of event selection)

313 real NBBO observations at the frozen 09:31 entry instant, MINING-period
events only.

| p10 | p25 | median | p75 | p90 | p99 |
|---|---|---|---|---|---|
| 0.13% | 0.28% | **0.55%** | 1.06% | 2.06% | 5.01% |

The prior model's docstring asserted "real bid/ask on these names is 1–15%".
That was never measured. At the entry leg:

| | median round-trip |
|---|---|
| Measured | 0.5510% |
| New fitted model | 0.5099% |
| **Legacy `SCENARIOS` model** | **6.2060%** |

**Overcharge factor: 11.3×.** The legacy `_half_spread` saturated its 8%
round-trip `cap` on gapper opens.

### Fitted model

`log(half_spread) = a + b·log(price) + c·log(minute_$vol) + d·log(bar_range)`

| coefficient | value |
|---|---|
| intercept | 0.324468 |
| log_price | 0.193418 |
| log_dollar_vol | **−0.383361** |
| log_bar_range | 0.368907 |

R²(log) = 0.404, residual sd(log) = 0.878. The dominant term is dollar volume
with a negative sign — more liquidity, tighter spread — which is economically
correct. Caps are **observed percentiles** (floor = p1, cap = p99 = 5.41%
round-trip), not hand-picked, so genuinely wide events are still charged.

Fit on mining events ONLY, so the holdout stays clean for the cost model as well
as the signal. Fitting the cost model on the evaluation set is the subtler
cousin of the look-ahead bug that killed HYP-105/106.

### Single-event A/B (CRE 2026-06-16)

| | net return |
|---|---|
| Real quotes (harness) | **+0.24%** |
| New measured model | −0.73% |
| Legacy model | −7.20% |

---

## 3. RECONSTRUCTION FAILED — this is NOT the sealed holdout

**The sealed holdout event list was never committed.** Only the thresholds
survived (`hyp107_shadow.py:38-39`). Rebuilding from the documented 70/30 date
split does not reproduce it:

| | published (sealed) | rebuild | |
|---|---|---|---|
| all gap-ups | 269 | 284 | +6%, plausible data drift |
| **filtered** | **57** | **98** | **+72% — not drift** |
| filtered median | +5.4% | +2.93% | |
| filtered win | 70% | 64.3% | |

My filter passes **34.5%** of gap-ups; the original passed **21.2%**. The
original applied some additional condition that cannot be recovered from the
surviving artifacts. **The 98 events are a superset, not the sealed 57.**

Consequently **nothing below is a holdout rerun** and no verdict can be sealed
from it. Reproducing the sealed set requires the original reconstruction script,
which is not in the repo.

---

## 4. Re-measurement on the reconstructed set (n=97 with quotes)

Read as: "what happens to a HYP-107-like event set when costs are measured
rather than assumed." Not as a verdict.

| | n | median | mean | win | tail | per-trade Sharpe |
|---|---|---|---|---|---|---|
| GROSS (mid-to-mid) | 97 | +3.04% | +9.19% | 63.9% | 3.19 | 0.422 |
| **NET — measured spreads** | 97 | **+2.26%** | +8.37% | 61.9% | 2.94 | 0.385 |
| NET — fitted model | 97 | +2.45% | +8.55% | 60.8% | 3.22 | 0.392 |
| **NET — legacy model** | 97 | **−1.90%** | +2.79% | 42.3% | 2.13 | 0.128 |

Median measured round-trip spread on these events: **0.72%**.

**The legacy model flips the sign of the result** — −1.90% vs +2.26% on
identical events, a 4.16pp paired swing. That comparison IS robust to event
selection, because it is paired on the same events.

---

## 5. Constitutional yield (Article 1) — the gate that actually binds

Median return per trade is not the test. HYP-093 posted a **+4.87% median** and
still sealed `VALID_BUT_BELOW_FLOOR` at 0.023%/day.

**Formula calibration.** Applied to HYP-093's known inputs (median 4.874%,
n=559, ~250 dates), this formula returns 0.068%/day where the sealed value is
0.023%/day — so it is roughly **3× too generous**. True yields are lower than
shown.

| sizing | yield/day | vs 0.05% floor |
|---|---|---|
| unsized | 2.926% | above — meaningless without sizing |
| × notional_w (0.0125) | **0.0366%** | **BELOW** |
| × notional_w × locate_w | **0.0183%** | **BELOW** |

Even on a **more permissive** event set, with a **healthy +2.26% median net**,
and using a formula that is **3× too generous**, HYP-107 lands below the floor.

### Why this matters for the verdict protocol

The briefed rule was "median net > 2% → `CONFIRMED_POSITIVE_EXPECTANCY`".
Median net came in at **+2.26%** — the rule would have fired today. The
constitutional gate says otherwise. A threshold on median return per trade does
not test the thing that binds; deploying on it would have upgraded a hypothesis
that fails Article 1.

---

## 6. What is and is not established

**Established (robust):**
- Quoted spread on these names is ~0.55% median, not 1–15%. Measured, n=313.
- The legacy model overcharges by 11.3× and inverts the sign of net results.
- Every `realistic_fills`-based gapper backtest before this commit is biased
  pessimistic. Corrected figures move UP.

**NOT established:**
- Nothing about HYP-107's verdict. The sealed holdout was not reproduced.
- Nothing about tradeability. Constitutional yield is below floor on every
  sized variant computed here.
- Nothing about funding. TICK-022 EV remains 0.0.

---

## 7. Next

1. **Recover the original reconstruction script** or re-derive the sealed 57
   with a documented, committed procedure. Until then no HYP-107 verdict is
   available at any cost model.
2. **Re-run other `realistic_fills`-dependent results** — they are all biased
   pessimistic by roughly the same mechanism.
3. **Decide the verdict rule** on constitutional yield, not median return, and
   pre-register it before the sealed set is re-run.

Artifacts: `data/research/gapper/tick039/{mining,holdout}_events.jsonl`,
`tick039_results.json`. Reproduce with `research/gapper/tick039_collect.py`
then `research/gapper/tick039_fit.py`.
