# RISK CONSTITUTION

This document is the constitution of capital preservation for the Sovereign
trading system. Every strategy, layer, engine, and future component is bound by
it. It is the conscience layer: the rules no optimizer may reason its way past.

**STATUS: RATIFIED v1.0.0 (2026-07-07) — Colin ratified as proposed (decision made
days prior; enactment was caught missing by the Day-3 P5 preflight and executed on
his order). Amendments only per Article 5.**

Machine twin: `config/risk_constitution.yaml` · Drift test: `tests/test_risk_constitution.py`
Convention: every binding numeric appears exactly once, in its article, in bold
with a percent sign and a DRAFT marker. Unbolded numbers are rationale, not law.

---

## Article 1 — Per-Trade Exposure

No single trade may risk more than **0.75%** of account equity, measured
as the worst-case loss at the protective stop. The number comes from the 100k-sim
prop assessment. No grade, conviction score, or Kelly output raises it.

## Article 2 — The Carry Complex Is One Bet

The carry pairs `GBPUSD`, `EURUSD`, `AUDUSD`, `AUDNZD`, `USDJPY` are treated as a
single position under stress; their correlations converge exactly when it matters
most. Total simultaneous open risk across all carry positions may not exceed
**2.5%** of equity. The constitution deliberately names five pairs even
while the live universe trades four — constitutional scope outlives universe
changes.

## Article 3 — Drawdown Circuit Breakers

Drawdown is measured peak-to-trough at account level. At a drawdown of **3.5%**,
all new position sizes are halved. At **5%**, new entries halt. At **6.5%**, every
predictive-layer position is flattened. The ladder is anchored below the 8%
TRAILING prop halt (the binding external line — trailing, not static): the original
draft ladder (5 / 7 / 8.5) placed the flatten breaker above that trailing halt,
where it mathematically could never fire — a decorative emergency brake. Ratified
values keep the final breaker 1.5% below the binding line so every rung can act
before the account is dead.

## Article 4 — The Right to Abstain

When the board state is ambiguous or the classifier abstains, the predictive
layer does not enter. Doing nothing is a position. A day with no trades is a day
the system worked.

## Article 5 — No Override

No engine, optimizer, or future component may size above these caps or trade
through a breaker. There is no exceptional path. Amendments require a dated,
reasoned entry in this file and the matching change to
`config/risk_constitution.yaml` in the same commit — never a hotfix. A failing
drift test means the prose and the twin diverged: amend both together; do not
fix the test.

## Article 6 — Unproven Edges

No live capital is allocated to any edge without a confirmed, pre-registered,
out-of-sample entry in the hypothesis ledger. Paper is where hypotheses live;
money is where confirmations live. This article binds live capital only: paper,
shadow, and research runs may exercise any pre-registered hypothesis at any
evidence stage, provided their records stay source-tagged so a paper outcome can
never masquerade as live evidence.

---

## Amendments

- 2026-07-07 — v1.0.0 — **RATIFIED** by Colin as proposed (decision made days
  earlier; the enacting session was never run — the gap was caught by the Day-3 P5
  preflight and enacted on his order). Art. 1 per-trade 0.75% and Art. 2 carry heat
  2.5% confirmed; Art. 3 breaker ladder re-anchored 5/7/8.5 → **3.5/5/6.5**
  peak-to-trough, below the 8% trailing prop halt (the draft flatten breaker sat
  above the trailing line and could never fire); Art. 6 paper-vs-capital carve-out
  sentence added. Machine twin + drift test amended in the same commit.
- 2026-07-01 — v0.1.0 — Initial draft. All values DRAFT pending ratification.

## Enforcement reconciliation (pending — dated notes per the ratification order)

The three tier configs are all imported, directly or transitively, by the live
execution path (verified 2026-07-07: `risk_config.yaml` ← sovereign/risk/config/
loader ← risk_engine/base_size ← `scripts/forex_live_scan.py`; `parameters.yml` ←
config/loader ← ict/micro_risk + funderpro_executor; `ict_params.yml` ← ict/pipeline
and the live ICT scan). Under the shadow freeze none may be edited before the
window closes (~2026-07-28). Therefore:

- 2026-07-07 — **PENDING RECONCILIATION, blocked_on: shadow_close** — after the
  window closes, each tier config gains enforced clamps guaranteeing no tier value
  may exceed the ratified caps above (tier logic intact; clamp refuses over-cap
  values; mutation test proves the refusal). Until then the constitution binds by
  law and by the gates that already read it at research level; the tier files are
  reconciled the day the freeze lifts.
