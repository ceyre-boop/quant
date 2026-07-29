# The Petrules Gate — Formal Specification
**Date:** 2026-07-21 · **Status:** SPEC (pre-code) · **Design:** `research/PETRULES_GATE_design.md`
**Spec-first rule:** No learning code runs until this document is approved and hash-locked.
**Builds on:** `research/PATTERN_FRAMEWORK.md`, `sovereign/intelligence/decision_logger.py`

---

## Purpose

Build a conviction engine that predicts the direction and magnitude of an asset's divergence
from the priced-in consensus, scores its own calibration (when it says 80%, it means 80%),
and sizes trades by conviction with a hard ruin floor. This is Edge #3+ in the Alta breadth
stack — genuinely uncorrelated with carry (rate-regime-driven) and the Undertow
(mechanical gapper fade).

The Gate produces one output: a tuple of `(direction, magnitude, conviction)` for a given
setup. The sizer turns that into a position. The calibration engine determines whether the
conviction scores mean anything. These are three separate subsystems with separate build
phases and separate pass/fail gates.

---

## Scope

**In scope:**
- Consensus baseline construction (what is priced in at any moment)
- Divergence label derivation (what actually happened vs. consensus)
- Feature set for the divergence detector (price, flow, sentiment, disclosed filings)
- Calibration protocol (reliability diagram + Brier score on holdout)
- Conviction sizer formula with hard f_max ceiling
- Pre-registration and holdout structure
- Verdict schema

**Out of scope (separate builds, after calibration is proven):**
- Live execution wiring (requires W7-style shadow period first)
- Integration with sovereign orchestrator (after conviction scores are proven calibrated)
- Options overlay (the Gate scores setups; whether to express via stock or options is a
  separate decision)
- Material non-public information of any kind — this system reads only public data

**Hard constraint:** The Gate touches public data only. Form 4, 13D/G, 13F, Congressional
disclosures, options flow from public feeds, and price/volume data. Any feature that requires
non-public information is excluded at the data-source level, not the feature level.

---

## The Three Subsystems

### Subsystem 1 — Consensus Baseline

**Definition:** For a given asset at a given point in time, the consensus baseline is the set
of expectations that are *already reflected in the current price.* A price move is only
meaningful as a divergence-from-consensus, not as an absolute direction.

**What "priced in" means operationally:**

For earnings-event setups:
```
consensus_eps_estimate   = median analyst EPS estimate as of T-1 (before print)
consensus_revenue_est    = median analyst revenue estimate as of T-1
options_implied_move     = ATM straddle price / stock price at T-1 (30-day or event-dated)
analyst_revision_trend   = net analyst estimate revisions over prior 90 days (positive/negative/flat)
consensus_price_target   = median analyst price target as of T-1
headline_sentiment_score = aggregate sentiment from published articles T-30 to T-1 (public only)
```

For non-earnings setups (macro catalyst, activist, etc.):
```
options_implied_move     = same construction
positioning_baseline     = net COT commercial positioning (futures) or options skew direction
analyst_revision_trend   = same
disclosed_flow_direction = net Form-4 cluster direction over prior 90 days (see §Features)
```

**The consensus label:** A single signed number representing what the market was pricing.
For earnings: `consensus_move_pct = options_implied_move × direction_of_revision_trend`.
For non-earnings: `consensus_move_pct = options_implied_move × sign(positioning_baseline)`.

This is not a precise prediction — it is a summary of expectation, accurate to ±1 straddle
width. That is sufficient for divergence-detection; the Gate is not trying to know the exact
priced-in number, only whether reality beat or missed it significantly.

**Data sources (public, free or accessible):**
- Analyst estimates: Nasdaq website public pages (free, T+0 for public estimate data),
  EDGAR company filings for consensus disclosures, Yahoo Finance API for historical estimates.
- Options implied move: CBOE public options chain data, or derived from Massive (formerly
  Polygon) options aggregates if Developer tier is subscribed.
- Form 4 / 13D/G / 13F: SEC EDGAR full-text search API (free, <10-second query latency).
- Congressional trades: HouseStockWatcher.com and SenateSockWatcher.com public APIs (free).
- News sentiment: free-tier NewsAPI or Finnhub news endpoint; sentiment scored via
  lightweight VADER or similar (no LLM call — too slow at historical replay scale).
- Unusual options flow / dark-pool prints: Unusual Whales free tier (500 calls/mo), or
  DIY from Massive options aggregates.

**Availability constraint:** Analyst estimates with full revision history require either
(a) a paid data subscription (Refinitiv, Bloomberg — out of budget) or (b) historical
scraping at the time estimates were published. The build phase must determine what's
available in the historical record at zero or low cost before finalizing feature set.
If high-quality estimates aren't available historically, the Gate narrows to setups where
proxy measures suffice (options-implied move is always available).

---

### Subsystem 2 — Divergence Detector

**The label:** For any setup frozen at time T (before the resolution event), the divergence
label is:

```
divergence = (actual_outcome_move - consensus_move_pct) / abs(consensus_move_pct)
```

A divergence of +1.0 means reality moved twice as far as the priced-in expectation, in the
consensus direction. A divergence of -2.0 means reality moved against consensus by twice the
implied move. The sign encodes direction vs. consensus, the magnitude encodes size.

For the Gate to "fire," it must predict:
1. `direction_vs_consensus`: {BEATS, MISSES, IN_LINE}
2. `magnitude_vs_consensus`: {SMALL (0-50% of implied), LARGE (50-150%), EXTREME (>150%)}
3. `conviction`: float [0, 1]

The Gate fires only when `direction == BEATS or MISSES` AND `magnitude == LARGE or EXTREME`
AND `conviction >= 0.65`. SMALL divergences and IN_LINE outcomes are too close to consensus
to extract alpha; the Gate stays silent on them.

**The feature set:**

Tier 1 — consensus-gap features (highest expected predictive value, per W1 mechanism logic):
```
consensus_revision_momentum  : net analyst revisions direction + velocity over 30/60/90d
options_skew_direction       : put/call skew sign relative to consensus direction
disclosed_flow_vs_consensus  : net Form-4 insider direction vs analyst consensus direction
activist_disclosure_recent   : 13D/G filed within 90d (boolean)
institutional_accumulation   : 13F net change in shares held, most recent quarter
congressional_trade_direction: net direction of disclosed congressional trades in ticker within 90d
```

Tier 2 — price/volume features (standard; lower incremental lift given Tier 1):
```
price_vs_52w_high            : current price relative to 52-week range
volume_ratio_20d             : current volume / 20-day average volume
short_interest_ratio         : days-to-cover (where available)
earnings_surprise_history    : prior 4 quarters' beat/miss direction and magnitude
```

Tier 3 — sentiment/text features (log-only in first build; validate before use in model):
```
headline_sentiment_30d       : aggregate VADER score of published news
earnings_call_tone_prior_q   : positive/negative tone of most recent prior earnings call
analyst_commentary_trend     : upgrade/downgrade ratio over 60d
```

**The model:** A gradient-boosted classifier (XGBoost, same stack as Oracle) trained to
predict `direction_vs_consensus` and `magnitude_bucket`, outputting a raw probability.
Two separate models — one for direction, one for magnitude — because the calibration
requirements differ and combining into a single score masks which component is failing.

**The replay engine (anti-lookahead requirement):**

Every historical training point must be constructed from data that was *published and
publicly accessible* at the moment of the freeze time T. This is the same discipline as
the intelligence loop. Enforced by:
- Using data timestamps, not release dates: only use data where `data.published_ts < T`
- Form 4 filings: use `filing_date` field from EDGAR, not transaction date
- 13F: use the filing date, not the period-of-report date (13Fs are filed 45 days after
  quarter end — a Q1 filing published May 15 is available at T=May 15, not March 31)
- Analyst estimates: use `estimate_revision_date`, not the period the estimate covers
- News sentiment: use article `published_at` timestamp

Any training example where the lookahead constraint cannot be verified is excluded. This is
expensive on sample size but non-negotiable — an uncaught lookahead in the conviction engine
is more dangerous than anywhere else in the system, because confidence is the thing it
manufactures.

**Sample universe:** Events with measurable outcomes and full feature coverage:
- Earnings announcements (quarterly, US-listed, ≥$500M market cap at event time)
- Activist stake disclosures (13D/G) with documented subsequent price resolution
- Executive cluster-buys (≥3 Form-4 purchases within 30d by insiders at same company)
- Macro catalyst events (FOMC decisions, CPI prints) with options-implied move measurable
  prior to release

Initial target: 5,000+ training events. Below this, the model cannot learn calibration
reliably. 10,000+ is preferred. The replay engine must generate this from historical sources;
if coverage is insufficient, the Gate narrows to a subset of event types.

---

### Subsystem 3 — Conviction Sizer

**Input:** Gate outputs `(direction, magnitude_bucket, conviction)`. Sizer translates this
to a position size.

**Formula:**

```python
def petrules_size(conviction, edge_ratio, worst_case_loss, budget, f_max=0.08):
    """
    conviction    : float [0.65, 1.0] — Gate calibrated probability
    edge_ratio    : E[gain|correct] / E[loss|wrong] — from historical analogs
    worst_case_loss : max loss fraction per position (structural, like W*)
    budget        : risk budget fraction of account (e.g., 0.0075)
    f_max         : hard ceiling — no conviction score overrides this
    """
    kelly_fraction = (conviction * edge_ratio - (1 - conviction)) / edge_ratio
    raw_size = min(kelly_fraction * 0.25, budget / worst_case_loss)  # quarter-Kelly
    return min(raw_size, f_max)
```

The parameters:
- `f_max = 0.08` (8% notional) — hard ceiling, welded before first dollar. Not a guideline.
  This is the maximum a Petrules trade can ever be, regardless of conviction = 0.99.
- `budget = 0.01` (1% risk per trade, higher than the Undertow's 0.75% because the Gate
  is discretionary and fires rarely — fewer events, higher per-event budget)
- `worst_case_loss`: derived per setup class. For earnings: max implied move × 1.5 (the
  historical extreme for the setup class). For activist: 15% (typical drawdown before
  thesis pays, per Burry precedent). Never derived in real-time — set per event class in
  `config/petrules_params.yml`.
- Quarter-Kelly (0.25×) is the ceiling for first-year live trading. After 100+ live events
  with calibration validated, revisit toward half-Kelly.

**Conviction tiers and corresponding size ranges:**

| Conviction | Tier | Max size at f_max=8% | Plain English |
|---|---|---|---|
| 0.65–0.72 | Interested | 1–2% notional | "Worth a look, small" |
| 0.72–0.82 | Confident | 2–4% notional | "Real position" |
| 0.82–0.90 | High-conviction | 4–6% notional | "Meaningful size" |
| 0.90–1.00 | Screamer | 6–8% notional | "Go big — within the floor" |

The "Screamer" tier is the legendary setup — the one that should feel like going all-in, but
isn't. 8% is 8%, not 100%. It is the largest the system will ever put on a single name.
This will feel too small on the wins and exactly right after a 20% overnight gap.

**Daily and portfolio limits (separate from per-trade f_max):**
```
daily_petrules_heat   = 0.05  (5% of account maximum across all open Petrules positions)
max_concurrent_names  = 3     (no more than 3 Petrules positions open simultaneously)
correlation_penalty   = 0.70× if two positions share sector or factor exposure > 0.6
```

---

## Calibration Protocol (the gate before the sizer runs)

This is the non-negotiable gate. The conviction sizer assumes the conviction scores mean
something. Before that assumption is allowed, the Gate must prove calibration on holdout.

**The calibration test:**

Split all training events into 70% train / 30% holdout (time-ordered, no shuffling).
Train the model on the 70%. On the holdout 30%, for every prediction with conviction in
band [c-0.05, c+0.05]:

```
calibration_error_at_c = |fraction_that_actually_resolved_as_predicted - c|
```

**Pass criterion (pre-registered, non-negotiable):**

```
CALIBRATED iff:
  - calibration_error < 0.05 at every confidence band [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
  - Brier score < 0.18 on holdout (perfect = 0.0, random = 0.25)
  - n_holdout_events >= 500 (below this, calibration statistics are unreliable)
  - Direction accuracy >= 0.62 on holdout (above random, below the overfitting threshold)

NOT_CALIBRATED otherwise — sizer does not run, Gate operates in logging-only mode
```

A reliability diagram (actual vs. predicted frequency) is generated at every calibration
check. The diagram must show the predicted diagonal — not clustering at the extremes (which
indicates overconfidence) and not a flat line (which indicates no discrimination).

**Ongoing calibration (after live deployment):**

Every 50 live events, recompute the calibration curve on the live outcomes. If
`calibration_error > 0.10` at any band, the Gate automatically enters logging-only mode
and alerts until recalibration. This is the Oracle feedback loop applied to conviction
scores: `decision_logger.log()` at call time, `decision_logger.update_outcome()` at
resolution. The same closed-loop discipline as everything else in the stack.

---

## Anti-Overfit Welds (Non-Negotiable)

The Gate is the highest-risk overfitting surface in the entire system. It must clear all of
these before any live money:

1. **Pre-registration.** Exact feature set, model architecture, calibration thresholds, and
   holdout split are locked before training begins. Filed as a JSON pre-reg in
   `data/research/preregister/PETRULES_GATE_prereq.json` with a SHA-256 hash. Any deviation
   from pre-registration is flagged as a deviation in the build report, not hidden.

2. **Single holdout use.** The 30% holdout is used exactly once — for the final calibration
   verdict. Intermediate development uses only the 70% training set with cross-validation.
   "I'll just peek at the holdout to see if I'm on the right track" is a holdout violation.

3. **Deflated Sharpe on the feature search.** The Tier 1 and Tier 2 feature sets above are
   the pre-registered features. Adding, removing, or transforming features after seeing
   training-set results requires a deviation note and a fresh holdout reservation.

4. **No sentiment features in the first live version.** Tier 3 (text/sentiment) features are
   logged but excluded from the model in the initial build. Sentiment features are the highest
   overfitting risk — they're high-dimensional and sparse. They can be added in a subsequent
   pre-registered update after the base model proves calibration.

5. **Calibration before profit.** The sizer does not run until the calibration test passes.
   No exceptions, no "let me just run it at tiny size to see if it works." An uncalibrated
   conviction engine running at any size is a wood-chipper.

---

## Pre-Registration Schema

File: `data/research/preregister/PETRULES_GATE_prereq.json`

```json
{
  "id": "PETRULES-001",
  "name": "Petrules Gate v1 — consensus-divergence conviction engine",
  "registered": "YYYY-MM-DD",
  "design_doc": "research/PETRULES_GATE_design.md",
  "spec_doc": "research/PETRULES_GATE_SPEC.md",
  "feature_set_tier1": ["consensus_revision_momentum", "options_skew_direction",
                         "disclosed_flow_vs_consensus", "activist_disclosure_recent",
                         "institutional_accumulation", "congressional_trade_direction"],
  "feature_set_tier2": ["price_vs_52w_high", "volume_ratio_20d",
                         "short_interest_ratio", "earnings_surprise_history"],
  "feature_set_excluded_tier3": ["headline_sentiment_30d", "earnings_call_tone_prior_q",
                                  "analyst_commentary_trend"],
  "model_architecture": "XGBoostClassifier, separate direction + magnitude models",
  "train_holdout_split": "0.70/0.30 time-ordered",
  "calibration_pass_criteria": {
    "calibration_error_max": 0.05,
    "brier_score_max": 0.18,
    "n_holdout_min": 500,
    "direction_accuracy_min": 0.62
  },
  "sizer_params": {
    "f_max": 0.08,
    "budget_per_trade": 0.01,
    "kelly_fraction": 0.25,
    "daily_heat_cap": 0.05,
    "max_concurrent": 3
  },
  "conviction_threshold_to_fire": 0.65,
  "universe": "US-listed equities ≥$500M market cap at event time, earnings + activist events",
  "target_training_events": 5000,
  "hash_lock": "TBD — computed at pre-registration filing"
}
```

---

## Build Phases (sequential, each phase unlocks the next)

### Phase 0 — Data Availability Audit (1–2 weeks, no code)
Before any model is built, determine what historical data actually exists and at what quality:
- EDGAR Form 4 / 13D / 13F: query a 5-year historical window, check coverage per ticker
- Analyst estimates: what is free vs. paid? What years are covered?
- Options implied move history: Massive Developer tier covers 10 years — verify
- Congressional trade history: HouseStockWatcher historical archive — check completeness

Output: a data availability report (`research/PETRULES_GATE_data_audit.md`) that either
confirms the pre-registered feature set is buildable or forces a pre-registered narrowing
of scope. Do not build the model until this is done.

### Phase 1 — Replay Engine (2–3 weeks)
Build the event extractor and lookahead-clean feature constructor. Output: a flat CSV of
`(event_id, freeze_timestamp, feature_vector, divergence_label)` for every training event.
Validate by random-sampling 50 events and manually checking that the feature values match
what was knowable at that timestamp, not after.

### Phase 2 — Model Training + Cross-Validation (1–2 weeks)
Train on 70% training set. Cross-validate to tune hyperparameters. Track calibration on
validation folds, NOT holdout. Plot reliability diagram on validation set.

### Phase 3 — Holdout Calibration Test (one run, sealed)
Apply the model to the 30% holdout. Compute all calibration metrics. File the verdict to
`data/research/preregister/verdicts_petrules.jsonl`. This run is sealed — the verdict
is permanent regardless of outcome.

If `CALIBRATED`: proceed to Phase 4.
If `NOT_CALIBRATED`: the build is parked. File a post-mortem, update the feature set or
model architecture in a new pre-registration, reset the holdout, and restart.

### Phase 4 — Logging-Only Shadow (3–6 months)
Wire the Gate into the live data stream. For every live event the Gate would trade, log:
- Feature vector as of freeze time
- Gate's prediction (direction, magnitude, conviction)
- Actual outcome after resolution

Do NOT size trades yet. Verify that the calibration curve computed on historical holdout
holds on live forward events. After 100+ live events: compute live calibration curve. If
it matches holdout within 5%: sizer is unlocked.

### Phase 5 — Live Trading at Quarter-Kelly
Only after Phase 4 calibration confirmation. Start at `kelly_fraction = 0.25`, `f_max = 0.08`.
Log every trade through `decision_logger.log()` and `decision_logger.update_outcome()`.
The Oracle feedback loop applies from day one.

---

## How This Fits the Roadmap

The Petrules Gate is the third independent edge in the Alta breadth stack:
- Edge 1: Forex carry v015 (~5% net/year after regime haircuts)
- Edge 2: The Undertow HYP-093 (~15% gross/year, F2+F3 sizing)
- **Edge 3: Petrules Gate (target 15-25% gross/year, calibration TBD)**

If Edge 3 proves calibrated and uncorrelated with Edges 1 and 2 (consensus-divergence is
not correlated with carry trades or mechanical gapper fades — the mechanisms are different),
the portfolio return approaches 30-35% with lower drawdown than any single edge. That is
the lever that moves the $10K/month calculation from $800K needed to $400K needed.

It is also the natural home for the disclosed-flow features (Form 4, 13D/G, 13F,
Congressional trades) — data sources that have no role in a mechanical gapper fade or a
carry trade, but are exactly what a consensus-divergence engine should consume.

**Timeline:** Data audit starts now. Model runs in 2–3 months. Holdout verdict in 3–4 months.
Live shadow starts 4–5 months from now. First live trade at minimum 10 months from today,
contingent on calibration passing. Not a weekend build. The right build.

---

## Cross-References

| Document | Relationship |
|---|---|
| `research/PETRULES_GATE_design.md` | The "why" — this document is the "how exactly" |
| `research/PATTERN_FRAMEWORK.md` | Predicate machinery; Petrules extends to consensus-relative labels |
| `data/research/yield_frontier/optimization/W6_SPEC.md` | Same spec discipline; same anti-overfit welds |
| `sovereign/intelligence/decision_logger.py` | Petrules trades must log entry + update_outcome at resolution |
| `sovereign/oracle/reflect_cycle.py` | Oracle reads Petrules outcomes once logged |
| `config/petrules_params.yml` | Sizing parameters (f_max, budget, worst-case by event class) — TBD |
| `data/research/preregister/PETRULES_GATE_prereq.json` | Pre-registration (to be filed at Phase 0 completion) |

---

*Alta Investments · Petrules Gate Specification v1.0*
*"Fearless to enter. Never fearless to size. That's the whole difference."*
*Spec written: 2026-07-21 · Build begins: after data audit confirms feasibility*
