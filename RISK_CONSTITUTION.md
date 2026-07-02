# RISK CONSTITUTION

This document is the constitution of capital preservation for the Sovereign
trading system. Every strategy, layer, engine, and future component is bound by
it. It is the conscience layer: the rules no optimizer may reason its way past.

**STATUS: DRAFT — every numeric value below is provisional pending ratification.**

Machine twin: `config/risk_constitution.yaml` · Drift test: `tests/test_risk_constitution.py`
Convention: every binding numeric appears exactly once, in its article, in bold
with a percent sign and a DRAFT marker. Unbolded numbers are rationale, not law.

---

## Article 1 — Per-Trade Exposure

No single trade may risk more than **0.75%** (DRAFT) of account equity, measured
as the worst-case loss at the protective stop. The number comes from the 100k-sim
prop assessment. No grade, conviction score, or Kelly output raises it.

## Article 2 — The Carry Complex Is One Bet

The carry pairs `GBPUSD`, `EURUSD`, `AUDUSD`, `AUDNZD`, `USDJPY` are treated as a
single position under stress; their correlations converge exactly when it matters
most. Total simultaneous open risk across all carry positions may not exceed
**2.5%** (DRAFT) of equity. The constitution deliberately names five pairs even
while the live universe trades four — constitutional scope outlives universe
changes.

## Article 3 — Drawdown Circuit Breakers

Drawdown is measured peak-to-trough at account level. At a drawdown of **5%**
(DRAFT), all new position sizes are halved. At **7%** (DRAFT), new entries halt.
At **8.5%** (DRAFT), every predictive-layer position is flattened. The ladder
exists to keep a buffer before the 10% prop kill line: the account must never sit
within one bad day of it.

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
money is where confirmations live.

---

## Amendments

- 2026-07-01 — v0.1.0 — Initial draft. All values DRAFT pending ratification.
