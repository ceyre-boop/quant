# Thesis-Invalidation Exit Spec (Track T)

Spec only — zero exit-path code. The shadow window (→ ~July 28) is sacred; nothing here
is wired until the divergence audit gates go-live and an unlock is recorded in NEXT.md.

**The principle:** a predictive-layer position exits when its REASON dies, not when PnL
complains. Every predictive entry ships with machine-checkable falsification predicates
over board state; `thesis_invalidated` is a first-class exit reason alongside stop,
target, and time.

---

## 1 · Falsification-predicate DSL

Machine-checkable boolean trees over `sentiment_board_state` columns (and only board
columns — predicates must be evaluable point-in-time with zero look-ahead, which the
board's release-date keying already guarantees):

```json
{"all": [
  {"col": "cot_net_pct_1y", "op": "<",  "value": 0.80},
  {"any": [
    {"col": "rr25",  "op": "sign_change_from_entry"},
    {"col": "vix_regime", "op": "!=", "value": "SPIKE"}
  ]}
]}
```

- Leaf ops: `<`, `<=`, `>`, `>=`, `==`, `!=`, `abs_lt`, `abs_gt`,
  `sign_change_from_entry` (compares against the entry-day board value captured in the
  journal row — the journal is the entry snapshot of record).
- Composites: `all`, `any`, `not`. No custom code in predicates — pure data, auditable,
  hashable.
- Semantics: the THESIS IS ALIVE while the predicate tree evaluates TRUE. Evaluated once
  per trading day at board close. First FALSE day → `thesis_invalidated` exit signal.
- Every predicate tree is stored on the journal row at entry (W1 schema field
  `thesis.falsification_predicates`) and hashed into `board_ref` provenance — predicates
  are frozen at entry, never edited mid-trade.

## 2 · Exit-reason taxonomy (predictive layer)

`INITIAL_STOP · TARGET · TIME · thesis_invalidated` — evaluated in that priority order
(hard risk stops always outrank thesis logic; the constitution's breakers outrank
everything). `thesis_invalidated` maps to the attribution rubric's `thesis_invalidated`
class when realized R ≤ 0 and to `luck_good` scrutiny when R > 0 (won while dying).

## 3 · Ownership decision (documented, binding until amended here)

**The predictive paper loop owns predictive exits — the frozen L2 exit manager does not.**

- The L2 manager (`forex_exit_manager` + shared `decide_exit`) is parity-locked to the
  backtester and mid-audit. Its whole value is that live == backtest bar-for-bar.
  Injecting experimental thesis semantics into it would contaminate the audited machine
  and break the reconcile-to-0.6886 guarantee. It keeps owning CARRY exits, unchanged.
- The predictive paper loop (factory `paper_adapter`, Track D — built, NOT enabled)
  evaluates its own predicates daily against the board and issues its own exit
  decisions, journaled like any other decision. Two engines, two exit owners, one
  journal — the two-layer wall preserved.

## 4 · Hook points (post-July-28; documented only)

1. **Paper adapter daily step** (factory/paper_adapter.py): after board refresh —
   evaluate predicates for each open predictive position → emit
   `thesis_invalidated` close + journal row. This is the ONLY wiring needed for paper.
2. **Journal** (experience/journal): entry rows carry the predicate tree from day one
   (already in the W1 schema) so backtests of thesis-exit policies are possible before
   any live wiring.
3. **If/when a predictive stack earns real execution** (separate, post-audit,
   Colin-gated): the adapter's exit signal routes through the same broker bridge calls
   the L2 manager uses (`close_trade`) — never through `decide_exit`, which stays
   carry-only.

## 5 · Non-goals

No changes to `decide_exit` / `forex_exit_manager` / backtester exit semantics; no
predicate evaluation inside the carry engine; no PnL-triggered "thesis" exits (that's
what stops are for — conflating them is the anti-pattern this spec exists to prevent).
