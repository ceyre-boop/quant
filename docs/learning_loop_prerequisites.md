# Learning loop — prerequisites, and what was refused

**2026-07-18 · TICK-041**

A six-part upgrade was proposed to turn the daily pipeline into a self-improving
loop. Three parts were built (in their non-adaptive form), three were refused.
This document records why, so the refusals are auditable and so the same
proposals do not silently return.

---

## The governing fact: there are almost no labels

Every proposed upgrade reads outcomes. The label inventory:

| source | count | usable? |
|---|---|---|
| ICT "closed" decision records | 3,460 | **No** — backtest replay. `decisions_2026_06.jsonl` line 1 carries `entry_timestamp: 2023-08-18` while sitting in the June file |
| Live broker-confirmed FOREX | **34** | Partially |
| …feature-complete | **well under 34** | Many are `fills_backfill` reconstructions with empty `component_scores` and `"why_this_size": "backfilled from OANDA fill — original sizing not logged"` |

**A learner cannot be trained on this.** Not 30 days of source attribution across
7 sources (2 of which are FRESH), not a threshold update from 20 events, not
XGBoost on 50 EOD summaries. The correct move is to make labels accumulate
correctly and wait — which is what was built.

## The governing precedent: adaptivity was tested and lost

HYP-090, 17,325 cells (385 configs × 15 pair subsets × 3 windows):

| arm | Sharpe |
|---|---|
| **A0 static — do nothing** | **0.9478** |
| A1_W90 adaptive | 0.1672 |
| A1_W365 adaptive | 0.2792 |
| A2_W365 adaptive (best) | 0.4343 |
| **A3 random-selection placebo, p95** | **0.9115** |

Every adaptive arm lost to picking configurations **at random**, and to doing
nothing. From the report: *"The A3 placebo envelope is the selection-noise floor:
beating A0 while not beating A3 is the in-sample-inflation signature, not an
edge."* Recorded at `research/yield_frontier/OPTIMIZATION_PROGRAM.md:12` as
**BANNED**.

---

## Built (observation only)

| # | Thing | What it does | What it deliberately does not |
|---|---|---|---|
| 1 | Outcome alarm fix (`51b6f9a`) | Counts already-matched trades separately; alarms only on genuine same-run failures | — |
| 2 | Day-boundary matching (`51b6f9a`) | ±36h asymmetric window, FIFO-oldest-open, adjacent-month lookup | Does not match a signal logged *after* its fill (that would be look-ahead) |
| 3 | Backfill schedule (`3fc749d`) | Runs the entry-side backfill daily | — |
| 4 | Drift tripwire (`3fc749d`) | Alerts when live win rate diverges from the sealed baseline; reports its own power | **Never modifies a threshold** |
| 5 | Bias scoring | Scores each daily bias against realised direction | Never gates a signal |
| 6 | Post-mortem log | Structured record per closed signal | No inference, no training |

---

## Refused

### Auto-weighting information sources by predictive track record

7 sources × 30 days, with **2 of 7 currently FRESH** and one (`reddit`) a
confirmed SILENT_NULL. That is a multiple-comparisons search over mostly-dead
inputs on a sample that cannot support it. `Discovery-Meta-Finding`: *"a second
edge needs NEW DATA, not cleverer mining."*

**Built instead:** source status is persisted alongside the day's outcome. After
enough days that *permits* an attribution study under preregistration. It does not
perform one.

### Bayesian threshold updates after every 20 events

The HYP-107 thresholds are frozen because they were preregistered before the
holdout was touched. That freeze is the only reason the result means anything.
Updating them on live data destroys the property being measured, and 20 events
cannot support the update regardless — see the power table below.

**Built instead:** a drift alert that reports divergence and never acts on it.

### VIX regime gate — third recurrence

- Not in `RISK_CONSTITUTION.md`. Verified against all five ratified numbers
  (0.75% per-trade, 2.5% carry heat, 3.5/5/6.5 ladder). The claim that the
  constitution "already contemplates this" is false.
- HYP-044's VIX gate was tested and rolled back **`REJECTED_OOS`, p=0.50,
  delta ≈ 0**, recorded in three places.
- **Root cause of the recurrence identified and fixed:** `CLAUDE.md:134` carried
  the commit-message *formatting example* `"[FOREX] Wire HYP-044 VIX gate for
  USDJPY/AUDNZD"`, which reads as an instruction. Changed to a neutral example
  with a note.

### XGBoost trained on EOD summaries

No model artifact exists (`training/xgb_model.pkl` absent; the engine silently
degrades), trainers live in `attic/`, nothing schedules training. Training on 50
EOD rows at a 3W/24L base rate would fit post-hoc narratives to a loss streak.
HYP-090 had 17,325 cells and still lost to random.

**Built instead:** the post-mortem log, which accumulates the dataset honestly. Its
header states the minimum viable n and the HYP-090 tombstone so a future session
cannot mistake it for a training set.

### Real-time Alpaca paper orders

Blocked twice, and only one is a coding problem:

1. **No order-placement code exists** — zero hits for `submit_order`,
   `TradingClient`, or any `api.alpaca.markets` endpoint.
2. **The signal cannot be computed in time.** The 15-minute SIP window applies to
   *bars*, not just quotes (`execution/alpaca.py:176`, `hyp107_shadow.py:76` both
   clamp to `now − 17min`). The 09:30 bar HYP-107 needs is unreadable until
   ~09:47. Real-time IEX is entitled but quotes AAPL at a ~10% spread on ~2% of
   volume.

A T+17min-entry variant *is* computable today — but it is a different hypothesis
with a different entry and would need its own preregistration and holdout. It is
not HYP-107.

---

## Power reference (why "20 events" is not enough)

Against the sealed 70% baseline, a 2σ alert fires only below:

| n | fires below |
|---|---|
| 20 | 49.5% |
| 50 | 57.0% |
| 100 | 60.8% |
| 200 | 63.5% |

And two distinct questions that are easy to conflate:

- an **observed** 60% first trips the alert at **n = 84**
- a **true** 60% rate is reliably caught (80% power) at **n ≈ 177**

At n=20 the instrument catches collapse, not drift.

---

## What would change these refusals

Not argument — sample size. Specifically:

1. The outcome loop accumulating **feature-complete live labels** (currently <34).
2. A preregistered attribution study, written before the data is examined.
3. For any adaptive proposal: clearing the **A3 random-selection placebo**, the
   bar that killed HYP-090. Beating a static baseline is not sufficient and never
   was.
