# Where a durable edge can exist for this trader

**Written before any data was read this session.** The trader: $2k, retail
brokerage, no colocation, no private data, no size. Every bucket is judged on
one question — *is it available to that trader at all*, then *what would evidence
look like*, then *what has already been tested here*.

Incumbent floor, against which everything is measured: buy-and-hold, equal
weight, ten liquid ETFs, 2015–2026 — **+131.7%, Sharpe 0.496, max DD −33.2%,
+0.029 %/day.** Anything that does not beat this on a risk-adjusted basis is dead.

---

## (i) Information / prediction

**Available?** Only public information. No order flow, no private data, no speed.
The only form available is *reading public, timestamped information more
carefully than the headline trader* — filings, earnings, macro releases, rating
changes. All of it arrives at a known instant and none of it is secret.

**Evidence would look like:** a conditional forward return after a public,
timestamped event that survives purged folds, a DSR at the honest trial count,
and a directional-permutation null (not a balanced one — that is how the NQ
track manufactured five beta "edges").

**Already tested and killed — this is the most exhausted bucket:**

| What | Verdict |
|---|---|
| Sign from macro / positioning / momentum on G10 FX — seven hypotheses | all NOT_SIGNIFICANT |
| NQ directional clustering, clean data | 0 VALID_EDGE (was beta) |
| Intraday breakout direction, 5,040 ORB configs | best Sharpe 0.21, 0 FWER |
| News-sniping direction (HYP-085) | p = 0.36 |
| Gapper checklist (HYP-092) | p = 0.59, well-powered |
| Mining this substrate (HYP-104, best of 77,016) | holdout Sharpe 0.36 |
| Post-shock direction (HYP-109) | NULL — weakly *negative*, CI excludes 0 |

**Not yet tested:** earnings-catalyst momentum (the terminal's wedge — never
run), analyst rating changes (FMP now serves them), lobbying and FDA event
classes. These are the only live threads in (i), and they inherit every null
above as a prior: direction has been null at every horizon tried.

**Honest read:** at daily resolution on liquid instruments, direction from
public information is null here. What survives in (i) is *magnitude* —
HYP-093, HYP-095, and now HYP-109(a): next-week RV is 1.36× after a shock at
p ≈ 0. That is real, but magnitude is not an edge until an instrument expresses
it — which is bucket (ii)'s problem.

---

## (ii) Execution / structure

**Available?** Not speed — a $2k trader is always last in the queue and pays the
spread. But *structural* return partitions that require no speed at all: holding
through the close, holding through a weekend, the calendar, the roll. These are
timing choices, not races.

**Evidence would look like:** returns partitioned by a structural boundary
(overnight vs intraday, day-of-week, roll date) that are stable across folds and
years, and whose *mechanism* is a known market structure rather than a story.

**Already tested:**

| What | Verdict |
|---|---|
| Passive limit-order entry (ICT) | 2% fill rate — 98% of setups expire |
| Intraday breakout timing | dead (above) |
| Long vol product after a magnitude signal (UVXY companion) | does not pay; roll decay dominates |
| **Overnight-vs-intraday partition on QQQ** | **VALID_EDGE** — 5.49 bp/day overnight vs 0.09 intraday, Sharpe 0.97. **Rejected only as a carry diversifier** (re-couples in crisis). **Never pursued on its own merits.** |

**Honest read:** this bucket holds the one standalone survivor in the whole
ledger that nobody followed up, and it is fully retail-accessible — holding
overnight costs nothing. Its known weakness: the overnight premium is a
compensation for gap risk, so it is real *and* it is where the crashes live.
The UVXY result also says something about (ii): the retail vol product is
structurally rigged against the magnitude forecast by its own roll.

---

## (iii) Risk and sizing

**Available?** Completely. Sizing and exposure are the only levers a $2k trader
controls with no counterparty and no latency. A 0/1 or continuous exposure rule
is free.

**Evidence would look like:** an improvement in risk-adjusted return **on the
delta series** (managed minus incumbent) — DSR applied to that delta, not to the
managed series alone — that is consistent across purged folds and does not
depend on one crash year.

**Already tested:**

| What | Verdict |
|---|---|
| Evidence-based worst-case sizing (HYP-097) | yield *fell* to 0.017 %/day; refuted by collar physics |
| Monte-Carlo state sizing (HYP-033), RMT Kelly (HYP-036) | rejected |
| **Post-shock abstention (HYP-109 secondary result)** | full-sample Sharpe 0.50 → 0.62, max DD halved, at −⅓ return, **8/15 folds** — interrogated in step 2 |

**Not yet tested:** continuous vol-targeting (Moreira–Muir, "C1"). Abstention is
the crude binary version of it. If the binary version halves drawdown, the
continuous version is the obvious next question, and it has never been run here.

**Honest read:** this is where the surviving finding — magnitude is
conditionable — can actually be *spent* by a retail trader, because sizing is
the instrument that pays for magnitude without needing a direction and without
a roll. Step 2 decides whether the crude version is real; if it is, the
continuous version is the natural prereg.

---

## (iv) Horizon / behaviour

**Available?** This is the only bucket where the $2k trader has an advantage
*institutions lack*: no mandate, no redemptions, no career risk, no quarter-end.
The ability to hold through a drawdown, or to do nothing for months, is a real
structural asymmetry and it costs nothing.

**Evidence would look like:** return or Sharpe as a function of holding period
or of patience — e.g. time exits vs stops, or the cost of *not* reacting — that
is stable and does not rely on a particular regime.

**Already tested:**

| What | Verdict |
|---|---|
| Time exits beat trailing stops (HYP-059/060, on v015 FX) | **CONFIRMED** — the one exit-side survivor |
| 12-month TSMOM (HYP-089/091) | too weak; Sharpe 0.28 |
| Daily adaptive re-parameterisation (HYP-090) | lost to a random placebo |
| Per-pair hold overrides (v007) | rolled back, fails walk-forward |

**Honest read:** the exit-side finding is real but it lives on the frozen v015
config (Standing Rule 1) and has not been tested on retail ETFs. The deeper
version — that patience itself is the edge — is hard to test because its null
is "buy and hold", which is already the incumbent. It may be that (iv) *is* the
incumbent, and the incumbent is the edge.

---

## Verdict on the map

| Bucket | Available to $2k? | State | Most promising unexplored thread |
|---|---|---|---|
| (i) information | public only | **exhausted for direction**; magnitude survives | earnings-catalyst momentum (untested) — but inherits every null |
| (ii) structure | timing, not speed | **one standalone survivor never followed up** | overnight partition on the ten ETFs |
| (iii) sizing | fully | crude version shows something; continuous untested | vol-targeting on the incumbent |
| (iv) behaviour | fully, uniquely | exit-side survivor on frozen config only | patience as edge — null is the incumbent itself |

**Not exhausted.** Two buckets have genuinely untested, retail-clean threads with
a prior survivor behind each: (ii) the overnight partition, and (iii) continuous
vol-targeting. Steps 2 and 3 decide which.

---

## Step 3 — regime, defined BEFORE looking

Stated here before any regime-conditioned number is computed. One definition,
one threshold, not tuned, not revisited.

**Operational definition.**
- `RV21_t` = std of SPY daily log returns over sessions t−21 … t−1
- `RV252med_t` = median of `RV21` over sessions t−252 … t−1
- **regime_t = HIGH if `RV21_t / RV252med_t > 1.0`, else LOW**

Trailing only; t is excluded everywhere. SPY is the regime instrument because
it is the broadest, and the regime is applied to all ten ETFs — a regime that
had to be defined per-instrument would already be a story, not a state.

**The ONE conditioning test.** Split the evaluation window by regime and report,
with block-bootstrap CIs:
1. the incumbent's forward 21-session return and Sharpe, HIGH vs LOW
2. the abstention delta (from step 2), HIGH vs LOW

**What would count as "regime improves something measurable":** the HIGH/LOW
difference in *incumbent Sharpe* has a CI that excludes zero. Not a difference
in return — a vol regime mechanically changes return dispersion; the question
is whether it changes the *risk-adjusted* payoff to being long.

**What would count as "a story that feels explanatory":** the CIs overlap, or
the difference is entirely a 2020 artefact (checked by reporting ex-2020).

This is declared as a trial. The step-4 prereg's DSR count is 1545 (1543 mined
+ HYP-109 + this).

---

## Step 2 — verdict on abstention (computed after the above was written)

Script: `scripts/research/diag_hyp109_abstention_regime.py` →
`data/research/hyp109/diagnostic_2026-09-02.json`. Same frozen parameters as
HYP-109; ledger untouched.

| question | answer |
|---|---|
| DSR on the **delta** (abstain − incumbent) | delta Sharpe **−0.21**, 95% CI [−0.82, +0.32], boot p(≤0) = 0.78, DSR prob **0.000** |
| ΔSharpe on the incumbent B&H | +0.120, CI **[−0.34, +0.59]** — indistinguishable from zero |
| ex-2020 | ΔSharpe **−0.044**; the whole effect is 2018 + 2020 + 2022 |
| per-year | 5/12 years positive |
| golden rule, per instrument | **5/10** instruments improve; ≈108 round-trips/yr, ≈$43/yr at 2 bp on $2k — cost is not the objection, sign inconsistency is |
| fold-level | 8/15, mean +0.09, sd 0.39, t-like +0.9 — the signature of **nothing**, not real-but-small |

**Adjudicated: the abstention overlay is noise.** It halves drawdown in three
crash years and pays for it every other year. The HYP-109 secondary result is
withdrawn as a lead. Bucket (iii)'s crude version is dead.

## Step 3 — verdict on regime

| | HIGH (47% of sessions) | LOW |
|---|---|---|
| incumbent daily Sharpe | +0.66 | +0.27 |
| diff | **+0.39, 95% CI [−0.75, +1.61] — includes 0** | |
| fwd-21d cumr | +0.604% | +0.604% (identical) |
| abstention delta Sharpe | −0.15 | −0.58 (diff CI [−0.74, +1.53]) |

**Pre-declared criterion fails: STORY_ONLY.** Regime does not improve anything
measurable. Two things it does say, descriptively: (1) the point estimate runs
*against* Moreira–Muir on this sample — risk-adjusted payoff to being long was
*higher* in high-vol months, so vol-targeting (which de-risks exactly then) has
its prior weakened, not strengthened; (2) forward 21-day return is identical by
regime to three decimals — the regime carries no directional information at all.

**Revised map.** (iii) sizing: crude version dead, continuous version's premise
points the wrong way — demoted. (ii) structure, the overnight partition, is now
the only thread with a prior standalone survivor and no contrary evidence. Trial
count for the step-4 prereg: **1545**.

---

## Step 4 — the ONE prereg: HYP-110, overnight partition on the ten ETFs

Picked by the operator from: overnight/ten ETFs · overnight/SPY-only ·
earnings-catalyst momentum · declare the map exhausted.

Sealed **`d981bf1d43170fe0`** before any open/close partition was computed
(`data/research/preregister/HYP-110.json`, ledger `PREREGISTERED`,
`scripts/research/preregister_hyp110_overnight.py --verify` OK).

- **Thesis:** the premium accrues close→open; open→close carries no positive
  drift. Own the ten ETFs overnight only.
- **Incumbent:** the identical EW buy-and-hold series (+131.7%, 0.496).
- **Frozen:** 1.0 bp round trip per instrument-day · warm 252 · L=5 · seed 42 ·
  10k draws · CPCV 6/2 · embargo 1 · **n_trials 1545** · floor 0.05 %/day.
- **All significance on the delta** — jointly-resampled ΔSharpe CI and DSR on
  the delta series. Golden rule (≥7/10 instruments) and ex-2020 are verdict
  components. Break-even cost is reported, never used.
- **Ladder:** CONFIRMED / VALID_BUT_BELOW_FLOOR / KILL_STRUCTURE / NULL / INCONCLUSIVE.
- **Prior:** NOT_SIGNIFICANT; most likely failure is the DSR hurdle at 1545
  trials or the premium being equity-only (TLT/GLD fail the golden rule).
- **Runs once:** `scripts/research/test_hyp110_overnight.py` (wiring verified
  with `--gate-only`; NOT run). Then `--verify` again.

**Result:** INCONCLUSIVE by the sealed data abort (three instruments >1% stale
opens); substantively a kill on every component — partition CI includes 0,
ΔSharpe −0.24, 3/10 instruments, 4/15 folds, break-even 0.02 bp. The premium on
the ten ETFs is not overnight. Bucket (ii)'s overnight thread is closed. See
`data/research/hyp110/VERDICT.md`.

---

## Addendum, end of day — three more preregs, the map after them

| id | what | verdict | one line |
|---|---|---|---|
| HYP-111a | post-shock intraday retrace→reclaim, 2023-06→2026-07 | VALID_BUT_BELOW_FLOOR | path fires 13%; pass driven by naive continuation *losing* −0.128%/event-day, 10/10 |
| HYP-113 | fade dose-response in shock size | FLAT | no slope; p99+ down-days carry no fade |
| HYP-112 | post-shock ATM straddle vs control, 2020–2026 | INCONCLUSIVE (abort) / substantive hard null | IV over-prices the clustering; −22% vs −13% on premium, 0/15 |

**Bucket (i) is now closed for magnitude as well as direction**: magnitude is real (HYP-109a),
conditionable, and priced by the only market that would pay for it. **What stands is one
thing**: the unconditional next-session fade after a p90+ shock — bucket (iv)/(ii), liquidity
provision by a patient holder — measured on one regime (2023-06→2026-07), all ten instruments,
not improvable by size, path or confluence. Its load-bearing test is 2020–2022 and that is a
data purchase. Trial count after today: **1550**.
