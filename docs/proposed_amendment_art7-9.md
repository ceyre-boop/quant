# PROPOSED — Risk Constitution Articles 7–9

**Status: DRAFT / UNRATIFIED. Nothing here is in force or implemented.**
Prepared 2026-07-18 under TICK-040. Ratification is Colin's alone
(`RISK_CONSTITUTION.md` Art. 5).

---

## Why this document exists

A build request asked for four risk gates: max position size, a daily-loss
circuit breaker, a consecutive-loss halt, and a VIX spike gate.

Only the first has constitutional authority. `RISK_CONSTITUTION.md` legislates
exactly five numbers (Art. 1 per-trade 0.75%, Art. 2 carry heat 2.5%, Art. 3
ladder 3.5/5/6.5). The other three appear nowhere in it, and Article 5 (`:49-56`)
states there is no override and that amendments require the prose and the YAML
twin to change in the same commit.

So `execution/risk.py` implements the ratified five and stops. The remaining
three are written here to be ratified, amended, or rejected — not quietly
invented in code. **Every number below is a PROPOSAL with its provenance shown.**

---

## The problem each would solve

### Art. 7 — Daily loss limit

**This is the urgent one, and it is not a gap — it is a live contradiction.**
Two limits are enforced *today*, on the same path, with different values:

| Value | Location |
|---|---|
| **5%** | `sovereign/risk/layers/gates.py` (via `risk_config.yaml:15`) |
| **2%** | `sovereign/risk/prop_risk_manager.py:49` `MAX_DAILY_LOSS_PCT` |

Neither derives from the constitution, because the constitution does not
legislate a daily limit at all. Whichever fires first wins by accident of call
order. Ratifying a number resolves this; leaving it unratified means the system
has an unlegislated risk control whose value depends on which module you read.

> **Proposed Art. 7.** No trading day may lose more than **2.0%** of account
> equity. On breach, entries halt for the remainder of the session; open
> positions are governed by Art. 3, not closed by this article.
>
> *Provenance of 2.0%:* the stricter of the two live values, and consistent with
> Art. 3's 3.5% halve — a daily limit above the first drawdown rung could permit
> a single day to trigger sizing reduction without ever tripping the daily gate.

### Art. 8 — Consecutive loss halt

No implementation exists anywhere, in prose or code. It is included because it
was requested, not because a gap was measured.

**There is currently no evidence for a specific streak length.** Live results are
3W/24L across 27 closed outcomes — a sample that contains long loss runs by
construction, so fitting a threshold to it would be fitting to noise.

> **Proposed Art. 8.** After **4** consecutive losing closed trades, new entries
> halt until the operator explicitly resumes.
>
> *Provenance of 4:* NONE. This is a placeholder. At an 11% observed win rate a
> 4-loss run is unremarkable and this would halt almost immediately; at a 60%
> win rate it would fire roughly once in 40 trades. **Recommend rejecting this
> article until the win rate is established on a sample that supports it.**

### Art. 9 — VIX gate

> **Recommend REJECT.**

HYP-044's VIX gate was tested and rolled back as `REJECTED_OOS` — p=0.50, delta
approximately zero (`CLAUDE.md:165`, `COMPONENT_CLASSIFICATION.md:98`,
`I_am_a_good_trader.md:171`). The rollback is recorded in three places and the
code is gone (`forex_backtester.py:147-148` reverted to 15.0 with a rollback
comment).

The reason it resurfaced in a build request is documentation drift:
`CLAUDE.md:134` still carries a worked example that reads as an instruction —

> `"[FOREX] Wire HYP-044 VIX gate for USDJPY/AUDNZD"`

— which is a *commit-message formatting example*, not a live requirement. It
should be changed to a neutral example so a refuted component stops propagating
into new specifications. That documentation fix is worth doing regardless of
what happens to this article.

Ratifying a VIX gate would re-add a component this system already tested and
rejected, on no new evidence.

---

## If ratified

1. Add Articles 7–9 to `RISK_CONSTITUTION.md` **and** `config/risk_constitution.yaml`
   in the same commit (Art. 5 requirement).
2. Extend `tests/test_risk_constitution.py` — the bold-token count assertion at
   `:34,137-149` is exact and will fail until updated.
3. Implement in `execution/risk.py` beside the existing five, with the same
   test-table treatment.
4. Reconcile the live 2%/5% daily-loss contradiction to the ratified value.
5. Log a `param_change_log.jsonl` entry per CLAUDE.md NON-NEGOTIABLE #4.

## Recommendation

Ratify **Art. 7 at 2.0%** — it resolves a live contradiction and the number is
defensible from existing enforcement rather than invented.

Defer **Art. 8** until a win rate exists that can support a streak threshold.

Reject **Art. 9**, and fix `CLAUDE.md:134` so the refuted gate stops being
re-proposed.
