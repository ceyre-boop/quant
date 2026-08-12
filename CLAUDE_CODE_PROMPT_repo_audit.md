# Prompt for Claude Code — Exhaustive Repo Audit: The Real Target

*Paste this whole thing into Claude Code as the task. Plan Mode first, per CLAUDE.md
workflow. Do not implement anything in the same pass that produces the plan.*

---

## The actual goal, stated precisely (read this twice before searching anything)

We are not trying to build the best trading strategy in the world. We are trying
to make money, under a specific, narrow set of constraints, on a specific
timeline. Every past research cycle in this repo (V1 through V4, HYP-001 through
HYP-108+) was implicitly optimizing "is this edge good," which is the wrong
objective function. The right objective function is:

**Given the edges we already have — confirmed, not hypothetical — what is the
fastest, lowest-ruin-probability path to (1) passing a funded evaluation, and
(2) compounding real capital afterward, without inventing a new strategy?**

Two facts already established this session, do not re-derive them, use them as
constraints:

1. **Sharpe is scale-invariant.** The v015 4-pair carry edge (Sharpe 1.25 OOS,
   p<0.001, BH-survives, decay ROBUST) produces 3.6%/yr at 0.25% risk/trade and
   14.8%/yr at 1.0% risk/trade on the SAME 411 trades. The edge was never weak.
   It was sized for an unproven-edge era that has ended.
2. **Frequency, not edge quality, is the binding constraint on eval-passing.**
   At 41 trades/year (the current live cadence), a 30-day eval needs ~2.37%
   return per trade to hit an 8% target — unachievable without ruinous sizing.
   At 200+ trades/year, the same per-trade edge quality (even a WORSE one,
   Sharpe as low as 0.15/trade) clears evals at low, survivable risk, because
   pass probability scales with trial count, not trade size.

So the search has two tracks, not one:

- **Track A — Sizing/process, not strategy.** Is everything already built
  (ALTA_METHOD.md, decision_logger, prop gates, dashboard) actually configured
  to run the confirmed edge at the correct size, with nothing structurally
  blocking a slow, patient, no-deadline eval attempt?
- **Track B — Frequency, not quality.** Does this repo already contain a second,
  independent, lower-quality-but-higher-frequency edge (mined, rejected,
  shelved, or half-built) that was discarded for being "not good enough" under
  the WRONG objective function (Sharpe maximization) but would actually satisfy
  the frequency requirement above? A Sharpe 0.15 edge at 200 trades/yr is more
  useful to us right now than a Sharpe 2.0 edge at 10 trades/yr.

---

## What to search for, exhaustively

### 1. Every rejected/shelved hypothesis, re-read against the NEW objective

Read `data/agent/hypothesis_ledger.json` in full — all entries, not just
CONFIRMED ones. For every REJECTED, NOT_SIGNIFICANT, MARGINAL, or
NOT_ROBUST entry, extract:
- trade frequency it would have added (not just Sharpe/p-value)
- WHY it was rejected — was it rejected for being a weak *standalone* edge
  (wrong bar under the old objective) or for being genuinely statistically
  fake (still correctly rejected under any objective)?
- Its correlation to the carry book, if measured

Flag anything rejected primarily for "Sharpe too low" or "not significant
enough to stand alone" as a **frequency-track candidate for re-evaluation**,
not a dead end. A noisy, frequent, uncorrelated edge that was killed for
being mediocre on its own may be exactly what's needed as a diversifying
frequency source layered under sizing discipline, not as a replacement for
carry.

Specifically look hard at (do not skip, these are named in the ledger):
- HYP-092 (gapper continuation/exhaustion — well-powered null on read
  quality, but check trade FREQUENCY it implied)
- HYP-094 (overnight gapper short — NOT_SIGNIFICANT boot_p=0.10, borderline)
- HYP-106/107 (runner filter — REFUTED/ADJUDICATED after leak issues, check
  if the de-biased HYP-107 version has a usable frequency profile)
- Anything in `data/research/yield_frontier/` (Undertow / HYP-093 family —
  this is already a SEPARATE, higher-frequency system with its own gauntlet;
  establish its current trade frequency and whether it's cleanly
  uncorrelated to carry)
- The ICT intraday pipeline (`ict/`, `ict-engine/`) — permutation p=0.52,
  NOT PROVEN as a standalone edge, but check its trade frequency. If it's
  10-50x the carry frequency, it may be worth re-testing as a small-size,
  frequency-layer position even at lower edge quality, PROVIDED it can be
  shown uncorrelated to carry in a crisis.

### 2. The Undertow / gauntlet system — is it already the frequency answer?

`data/research/yield_frontier/` contains a parallel, higher-frequency,
already-gauntlet-tested system (HYP-093, "The Undertow") with its own W6
sizing policy and its own `give-me-the-numbers` precedent (median +11%/yr,
75% profitable years, ~35% funded-eval blowup rate under a 10% DD limit per
TICK-032). Read the full gauntlet verdict history and TICK-032. Answer:
- What is Undertow's actual trades/year?
- Is TICK-032's "no funded vehicle" verdict still current, or does the
  0.50%-risk/no-deadline structure from COLIN_V1.md change that math?
- Is Undertow correlated to carry? If both are near-zero risk of joint
  drawdown, running both simultaneously (carry at 0.5-1% risk + Undertow at
  its own governed size) may satisfy the frequency requirement without
  needing any new research at all.

### 3. Everything already built that is unused or half-wired

Grep the repo for infrastructure that exists but was never load-bearing:
`sovereign/intelligence/regime_performance_tracker.py`,
`sovereign/intelligence/capital_allocator.py`,
`sovereign/intelligence/cross_system_bridge.py`. TRADING_PHILOSOPHY.md
Tenet 5 calls orchestration "the durable edge" — has this repo actually
built a system that runs MULTIPLE confirmed/semi-confirmed edges
concurrently with a capital allocator sizing between them, or does every
research cycle still test ideas in isolation and compare Sharpe numbers
one at a time? If the allocator exists but nothing feeds it more than one
strategy, that is the actual missing piece — not a new edge.

### 4. The two disagreeing exit engines (flagged, unresolved)

`V4_VERDICT.md` (2026-07-30) found that RQ-REST-013's sealed re-simulation
and an independently-built exit engine (`scripts/v4_exit_resim.py`) rank
exit-widening arms in opposite directions. This was flagged as worth more
than a V5 and never followed up. Reconcile it: build the smallest possible
shared-trade test (one pair, one entry, full bar-by-bar trace through both
engines) and find the actual divergence point. This affects sizing
confidence on the carry book, which affects Track A directly.

### 5. Everything in `data/research/` older than 60 days, untouched

List every file in `data/research/` and `data/agent/` not referenced by any
currently-CONFIRMED hypothesis and not read by the live pipeline. For each,
one line: what it tested, what it found, whether it's dead or dormant.
Many of these were killed under the Sharpe-maximization objective and may
be worth a frequency-track second look per §1.

---

## What NOT to do

- Do not propose a new strategy, hypothesis, or backtest. This is a search
  and inventory task, not a research task.
- Do not touch `config/parameters.yml`, `config/ict_params.yml`, or anything
  on the shadow/execution-path freeze list in `NEXT.md`.
- Do not re-run RQ-REST-013 or re-litigate V3/V4's verdicts — they stand.
- Do not recommend increasing risk above what COLIN_V1.md's sizing frontier
  supports (1.5% risk/trade is the stated ceiling; above that P(DD>20%)
  stops being trivial).

## Deliverable

A single markdown report: `data/research/REPO_AUDIT_frequency_and_sizing.md`
covering, in order:
1. Track A finding — is the current confirmed edge correctly sized and
   correctly wired end-to-end for a patient, no-deadline eval attempt?
   Name every gap.
2. Track B candidates — ranked list of every rejected/dormant/half-built
   system in the repo, re-scored on trades/year and correlation-to-carry
   rather than standalone Sharpe, with an honest note on why each was
   originally killed and whether that reason still applies under this
   objective.
3. One recommended next step. Not five. One. Preregistered before touching
   any data, per repo protocol.

Write the plan to `plans/repo-audit-frequency-sizing.md` first (Plan Mode).
Stop. Do not execute in the same turn. Colin reviews the plan before the
build pass begins.
