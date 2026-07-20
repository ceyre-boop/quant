# Risk Framework — Ratification Document

**Date ratified:** 2026-07-20
**Branch:** sovereign-v2
**Source constants:** `execution/risk.py` lines 65–73

---

## Purpose

The five constants below have been enforced in code since sovereign-v2 was
established, but without a ratifying document to make them official policy.
This document corrects that. From 2026-07-20 onward these constants are the
**ratified framework layer** — strictly-tighter overlays sitting above the
constitutional limits in `RISK_CONSTITUTION.md`. They do not trigger an
Article 5 amendment because they impose *narrower* bounds than the constitution
already permits; no constitutional guard is weakened.

---

## Ratified Constants

### 1. `DAILY_LOSS_HALT = 0.02`

**What it enforces:** If cumulative session P&L falls to −2% of session-start
equity, the harness halts all further fills for the remainder of that trading
day.

**Rationale:** A 2% intraday loss is a meaningful signal that market conditions
have diverged from the edge's tested regime. Stopping early preserves capital
and prevents a bad day from compounding into a drawdown that triggers the
constitutional Art. 3 ladder. The 2% threshold is materially tighter than the
constitution's 5% soft-halt, which is intentional.

---

### 2. `CONSEC_LOSS_HALVE = 3`

**What it enforces:** After 3 consecutive losing trades (measured across all
history, trailing), position size is reduced to 50% of normal for subsequent
fills in the same session.

**Rationale:** Three consecutive losses is a reliable early-warning signal in
mean-reverting strategies. Halving size while still allowing participation
keeps the feedback loop alive (fills still price the edge) while reducing
exposure during a potential regime shift.

---

### 3. `CONSEC_LOSS_HALT = 5`

**What it enforces:** After 5 consecutive losing trades, all further fills are
halted for the day.

**Rationale:** Five consecutive losses at any reasonable per-trade risk implies
a cumulative hit that crosses the daily halt threshold anyway. Making it
explicit prevents any calculation artifact from allowing continued trading
through an obviously broken session. This gate and `DAILY_LOSS_HALT` are
independent; whichever fires first wins.

---

### 4. `MAX_SINGLE_POSITION = 0.10`

**What it enforces:** No single position may exceed 10% of account equity on a
notional basis.

**Rationale:** Concentration cap. Complements the constitutional Art. 1
per-trade risk limit (`PER_TRADE_CAP = 0.0075`): that cap controls loss given
a stop, this cap controls gross notional regardless of stop distance. A wide
stop on a large position could satisfy Art. 1 while still creating dangerous
single-stock concentration; this constant prevents that.

---

### 5. `DEFAULT_RISK_PER_TRADE = 0.02`

**What it enforces:** The default R-multiple target is 2% of account equity,
where R = (entry − stop) × shares. The constitutional Art. 1 limit of 0.75%
still binds in live trading; this default applies to paper/evaluation modes.

**Rationale:** 2% per trade is standard prop-firm evaluation sizing. The
tighter 0.75% live cap is preserved and takes precedence whenever the harness
runs against a live or paper account designated as live. The separation keeps
evaluation and live risk parameters explicit rather than conflated.

---

## Previously Implicit — Now Official

These constants existed in `execution/risk.py` and were enforced by
`risk.check()` on every fill, but there was no document ratifying them as
policy. The absence of a ratifying document created an ambiguity: were these
temporary implementation choices or standing policy? This document resolves
that ambiguity. They are standing policy as of 2026-07-20.

---

## Amendment Procedure

Changes to any of the five constants above require **both** of the following
before the change takes effect:

1. **An entry in `data/agent/param_change_log.jsonl`** — recording the old
   value, new value, rationale, and the ISO-8601 timestamp of the change.

2. **An update to this document** — the constant's section must be revised to
   reflect the new value and rationale, and the "Date ratified" header updated.

A code change without both steps is unauthorized and must be reverted.

---

## Known Limitation (Finding 4A)

See `tickets/backlog.md` — Finding 4A describes a condition where
`DAILY_LOSS_HALT` is effectively inert on fresh trading days because
`AccountState` is built once at session start from fills already in the log,
making `daily_pnl_frac` equal to 0.0 when evaluated. This is a known gap; its
remediation requires its own unlock ticket before implementation.
