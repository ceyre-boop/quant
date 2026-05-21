# TRADING PHILOSOPHY

This document is the constitution of the Sovereign trading research system.
Every component must serve one of the six tenets below or it should not exist.

---

## The Six Tenets

### 1. Statistical Utility Beats Narrative Coherence

A feature does not earn its place because it makes intuitive sense.
It earns its place by improving measurable expectancy, out-of-sample.

**The pd_alignment lesson (HYP-024, 2026-05-19):** Removing a logically
coherent feature (premium/discount zone alignment) improved win rate from
20% to 35%. The system improved by removing intelligence, not adding it.

Implication: every feature must pass formal promotion gates before deployment.
See `lab/feature_registry.py`.

### 2. Regime Appropriateness Beats Strategy Quality

The right question is not "Is this strategy good?" but
"Is this strategy appropriate right now?"

Systems are not independent predators. They thrive and decay in regime-specific
ways. The capital allocator tracks which systems earn capital in which regimes
and adjusts sizing mechanistically.

See `sovereign/intelligence/regime_performance_tracker.py`.

### 3. Systems Must Know When They Are Unreliable

A mature system can say: "I am currently unreliable."
Health metrics, structural breaks, and consecutive-loss analysis make this
measurable rather than intuitive.

When a system enters UNRELIABLE state, it freezes itself and alerts.
This is not a weakness. It is the most important safety feature in the stack.

See `sovereign/intelligence/system_health.py`.

### 4. Premature Complexity Kills More Systems Than Lack of Edge

Most trading systems die after the stage where edge is first confirmed.
The creator stacks more signals, more models, more adaptive logic — before
understanding where edge actually originates and which conditions destroy it.

**Guard: Every new component must answer a measurable question before it is
built. If the question cannot be answered with existing data, do not build the
component — collect the data first.**

### 5. Orchestration Is The Durable Edge

Individual strategy edges decay.
The ability to route capital to the right system at the right time does not.

The cross-system bridge (threat gating), capital allocator (sizing throttle),
and regime performance tracker (attribution) together form the orchestration layer.
This layer should grow in sophistication faster than individual strategies.

### 6. Research Debt Is Existential Risk

Failed experiments must be formally recorded. Features must either earn
existence (LIVE) or be buried (GRAVEYARD). Research that is not recorded
does not exist. Insights that are not logged will be rediscovered at cost.

**The feature registry enforces this. Every feature has a formal record.
The graveyard is permanent. Audit runs every 90 days.**

---

## Time-Horizon Separation Doctrine

**This is a hard constraint, not a guideline.**

| System  | Time horizon | Data source | Allowed cross-system comms |
|---------|-------------|-------------|---------------------------|
| ICT     | Intraday, 4h | Tick/bar data, session structure | Bridge: macro threat only |
| Forex   | 5–60 day holds | Daily macro signals | Bridge: macro threat only |
| Equity  | Days–weeks | Daily price + macro | Bridge: macro threat only |

**No feature sharing between systems.**

The only cross-system communication channels are:
1. **Cross-system bridge** (`sovereign/intelligence/cross_system_bridge.py`): macro threat signals that can halt or tighten all systems simultaneously.
2. **Capital allocator** (`sovereign/intelligence/capital_allocator.py`): sizing multipliers per system based on regime performance and health.

Mixing intraday ICT signals into the Forex macro engine, or using Forex regime
labels inside the ICT pipeline, is explicitly prohibited. Time horizons are
different. Regimes operate on different scales. Contamination produces
correlation artifacts that look like edge but are not.

**If you find yourself wanting to share a feature across systems, stop.
Formalize the question as a hypothesis, test it in isolation, and check whether
the cross-system IC exceeds 0.15 out-of-sample before proceeding.**

---

## Promotion Standard (feature → LIVE)

Any feature must clear all three gates before deployment:

| Gate | Requirement |
|------|------------|
| 1. IC out-of-sample | > 0.15 on held-out data |
| 2. Marginal contribution | Strictly positive in walk-forward |
| 3. Holdout degradation | No degradation on 6-month holdout window |

Shortcut: `lab/feature_registry.py::FeatureRegistry.promote()` enforces these mechanically.

---

## The Feature Graveyard

The graveyard is permanent. It is not a failure archive — it is institutional memory.

Features in the graveyard were tested with real data and found to be harmful
or neutral. Rediscovering them costs research time. The graveyard prevents that.

**Do not re-introduce a graveyarded feature without:**
1. A new hypothesis explaining why prior conditions no longer apply.
2. N ≥ 200 fresh trades on the new configuration.
3. Explicit sign-off logged in the hypothesis ledger.

Current graveyard entries: see `data/lab/feature_registry.jsonl`.

---

## What The System Should Never Do

- Deploy a feature because it "makes sense" without statistical validation.
- Share features between ICT (intraday) and Forex/Equity (multi-day).
- Re-introduce graveyarded features without new evidence.
- Increase complexity while attribution is incomplete.
- Trust a system's own assessment of its reliability without independent health metrics.
- Run a prop challenge when the capital allocator has any system frozen.
