# COLIN v1 — The Diagnosis and the Plan
**Alta Investments | 2026-07-30**
*Not a new strategy. A correct reading of the one you already have.*

---

## THE THING YOU GOT WRONG

You have been judging a Sharpe 1.25 edge by its dollar output at 0.25% risk per
trade, and concluding the edge is weak.

Sharpe is scale-invariant. It does not change when you size up. The 3.5%/year is
not what the edge produces — it is what **you chose** for it to produce when you
set risk to a quarter of a percent.

| Risk/trade | Median year | p95 drawdown | P(losing year) | P(DD>20%) |
|---|---|---|---|---|
| 0.25% (current) | **3.6%** | 2.9% | 8.6% | 0.0% |
| 0.50% | 7.3% | 5.7% | 8.5% | 0.0% |
| 0.75% | 11.0% | 8.5% | 8.7% | 0.0% |
| **1.00%** | **14.8%** | **11.2%** | 8.8% | 0.0% |
| 1.50% | 22.4% | 16.6% | 9.5% | 1.6% |
| 2.00% | 30.0% | 21.4% | 9.5% | 6.9% |

Same edge. Same trades. Same 411-trade log. The only thing that changed is the
size. On $100K at 1% risk this edge has a median year of **$14,800** with an
11% worst-case drawdown and a 91% chance of a profitable year.

You said you can't make more than $5,000 on $100K. You can. You have been
sizing at the level appropriate for an unproven edge — which was correct while
it was unproven. It is now proven: p<0.001, survives BH, OOS Sharpe 1.25,
decay ratio 2.17 ROBUST, 10/10 positive years.

**The edge graduated. The sizing never did.**

---

## THE THING NOBODY TOLD YOU

Here is why every strategy you build "fails" — and it isn't the strategies.

**A 41-trade/year strategy cannot pass a 30-day funded evaluation. Ever. At any
skill level.**

| Window | Trades available | Return needed PER TRADE for +8% |
|---|---|---|
| 30 days | 3.4 | **2.37%** |
| 60 days | 6.8 | 1.18% |
| 90 days | 10.1 | 0.79% |
| 365 days | 41.1 | 0.19% |

Your edge delivers +0.356R per trade. To average +2.37% per trade you'd need
roughly **8% risk per trade** — four consecutive losers ends the account. It is
not a discipline problem or a strategy problem. It is arithmetic.

P(pass) on an 8%/6% eval with the real return series:

| Window | @0.5% risk | @1% risk | @2% risk |
|---|---|---|---|
| 30 days | 0.0% | 2.5% | 17.0% (8.4% blowup) |
| 90 days | 2.3% | 24.6% | 51.1% (31.7% blowup) |
| 365 days | 52.3% | 72.5% | 61.8% (38.2% blowup) |

**But with no time limit at all** — which is what Lucid and MyFundedFutures
actually offer, and which you already noted in CLAUDE.md:

| Risk | P(pass) | P(blowup) | Median time to pass |
|---|---|---|---|
| **0.25%** | **99.6%** | **0.4%** | 24 months |
| **0.50%** | **92.4%** | **7.6%** | **11 months** |
| 0.75% | 82.8% | 17.2% | 6.4 months |
| 1.00% | 75.4% | 24.6% | 4.4 months |
| 2.00% | 61.2% | 38.8% | 1.8 months |

**You already own a 92%-probability funded account.** It just takes eleven
months instead of one, and the entire skill is refusing to speed it up.

Every time you sized up to pass faster, you moved from the 92% row to the 61%
row. That is the loop you have been stuck in — and it looked like strategy
failure when it was a deadline you imposed on yourself.

---

## COLIN v1 — THE RULES

### Capital structure
- **Funded track:** carry edge, **0.50% risk/trade**, no deadline. 92% pass, ~11 months.
- **Own capital:** same edge, **1.00% risk/trade**. Median 14.8%/yr, 11% p95 DD.
- **Never** size above 1.5% on this edge. Above that P(DD>20%) becomes non-trivial
  and the compounding math inverts.

### Execution — unchanged, because it works
Entries stay exactly as v015/HYP-045 defines them. Four pairs
(EURUSD, GBPUSD, USDJPY, AUDUSD). All five gates. Two confirmations. The
incumbent exit config stands — both V3's replacement and V4's delay failed
honest testing.

### The one rule that matters
**Do not speed it up.** Every intervention you have tried this year — V3's exit
substitution, V4's trail delay, per-pair holds, the VIX gate, regime
conditioning — was an attempt to make 41 trades/year pay like 400. Each was
tested properly and each was rejected. The tests were not the problem. They were
the only thing standing between you and a blown account.

---

## WHAT TO BUILD NEXT — a specification, not a guess

You do not need a better edge. Sharpe 0.21/trade is fine. **You need more
independent bets.** Frequency is the dominant lever, and it is the one you have
never targeted directly.

At the same per-trade edge quality, going from 41 to 200 trades/year lets you cut
risk per trade to 0.25% and *still* clear an eval — because the pass condition
depends on the number of trials, not the size of each one. This is Andrew Lo's
150-trials point, already sitting in your own CLAUDE.md.

**The design target for the next research cycle:**

| Requirement | Value |
|---|---|
| Trades/year | ≥ 200 (5× current) |
| Per-trade Sharpe | ≥ 0.15 (LOWER than current — you can afford worse) |
| Correlation to carry book | < 0.3 in crisis |
| Max hold | irrelevant |

Note what that says: **you may trade a worse edge than you already have**, as
long as it fires five times more often and isn't correlated to carry. That is a
far easier research problem than "beat Sharpe 1.25," which is what you have been
attempting.

That reframe is the deliverable. Not another backtest.

---

## THE HONEST LEDGER

**What is proven:** v015 carry edge. Trailing stop costs ~63R (HYP-059).
Regime fragility is real.

**What is refuted:** V3's exit substitution (selection bias — RQ-REST-013's real
re-sim ranks pure time exits *below* the incumbent). V4's trail delay
(p=0.30 family-level; the headline was config selection).

**What is open:** RQ-REST-013 says widen the trail to 2.0×; my independent
engine says width doesn't matter and delay does. Two engines disagree. That
reconciliation is worth more than a V5.

**What was never a problem:** your strategy design. You have built a genuine,
statistically-confirmed edge — most people who trade for a decade never do. You
then measured it against a goal it is structurally incapable of meeting, on a
timeline you invented, and concluded you had failed.

You made $1,000 today on real money in your AI sector fund. You are not bad at
this. You have been reading one correct number as if it were a different number.

---

*Alta Investments — Sovereign Trading Intelligence*
*"It is not the average. It is surviving the bad days." — Tenet 3*
*You have been surviving the bad days so thoroughly that you forgot to have good ones.*
