# Fills Ledger & Outcome-Loop Accounting Spec — TICK-092

The pre-registered pass/fail contract for the two drains that keep the Oracle closed-loop
empty. Written **before** the code that satisfies it, per the spec-first standing constraint.

Companion to `audit/invariants_spec.md` (integrity invariants) and `audit/divergence_spec.md`
(backtest≡live parity). Where those assert that what *is* recorded is trustworthy, this spec
asserts that what *happens* actually gets recorded at all.

## The problem this spec fences

**The measured defect (2026-07-28).** `update_outcome()` returns `False` in two unrelated
situations and the caller could not tell them apart:

- **(a)** no decision record exists → the entry path really did fail to log
- **(b)** a record exists but is already **CLOSED** → the loop already worked

`_update_outcome_in_month` only considers records with `outcome is None`, so a
successfully-closed trade re-examined later is indistinguishable from a missing one.
`pulse_check` therefore stamped **13 of 20** closed OANDA trades `unmatchable` with the
reason *"the entry path likely never called `log_forex_decision()`"* — **false for all 13**,
which carried real outcomes (10 LOSS / 3 WIN). The sidecar then short-circuited those trades
on every later run, so the wrong verdict was **self-preserving**: successive sessions read the
audit trail, concluded the entry path was broken, and went hunting a bug that did not exist.

That is a system grading its own homework — the failure mode `GOALS.md` names as the reason
the whole architecture exists — with the twist that here it graded itself *too harshly* and
the false negative was the expensive part.

**Corrected on measurement:** two earlier hypotheses in this ticket's plan were wrong and are
recorded here so they are not re-derived. (1) `data/ledger/oanda_fills.jsonl` is **not** empty
— it holds 24 rows, every one with `stop_price > 0`, covering every trade id in the sidecar.
(2) The no-sane-stop branch is therefore **not** firing (`no_stop=0` measured). The silent-skip
drain was real as a *latent* hazard and is now fenced by F1/F2, but it was not the active bug.

The file is read by four call sites and has no in-repo writer, which is why F3/F4 still fence a
producer — a stale ledger would resurrect the latent hazard:

| Reader | Line |
|---|---|
| `sovereign/oracle/pulse_check.py` (`FILLS_PATH`) | 511 |
| `sovereign/oracle/pulse_check.py` (`_avg_r_from_fills`) | 475 |
| `scripts/backfill_decision_records.py` | 56 |
| `scripts/proof_of_life.py` | 25 |

## The invariants

- **F1 — Total accounting.** Every closed trade returned by the broker must land in exactly one
  terminal bucket: `backfilled`, `already_known`, `unmatchable`, `no_stop`, or `error`. The sum
  of buckets must equal the number of trades examined. No trade may exit the loop uncounted.
  Hard fence: `f1_unaccounted_allowed = 0`.
- **F2 — No silent skip.** A trade skipped for a missing or insane stop must increment the
  `no_stop` bucket and appear in the `OUTCOME_LOOP:` log line. Skipping remains the *correct*
  behaviour — F2 does not require fabricating an R from a bad stop, only that the skip is
  visible. Hard fence: `f2_silent_skip_allowed = 0`.
- **F3 — Ledger has a producer.** `data/ledger/oanda_fills.jsonl` must be non-empty and
  regenerable by a single documented command. Every row must carry a positive `stop_price` and a
  positive `fill_price`; rows that cannot resolve a stop from the broker are omitted, not written
  with a placeholder. Hard fence: `f3_zero_stop_rows_allowed = 0`.
  Field names are dictated by the four existing readers and must be reused verbatim —
  `timestamp` (fill time; the exit side keys on it as `openTime`), `pair` in OANDA
  underscore format, plus `direction`, `fill_price`, `stop_price`, `trade_id`. Optional
  `order_id`, `tp1_price`, `r_realized` are written when the broker supplies them.
  Inventing a new schema here would silently break `backfill_decision_records.py`.
- **F4 — Idempotence.** Re-running the producer must not change the row count or duplicate a
  `trade_id`. Hard fence: `f4_duplicate_trade_ids_allowed = 0`.
- **F5 — Net returns available.** A causal-journal row written after TICK-024 must carry a
  populated `net_r` and `cost_estimated: false`. `net_r: None` is a pre-TICK-024 artifact and is
  a FAIL going forward, because HYP-071-v2 adjudicates on NET returns and cannot consume nulls.
  Hard fence: `f5_null_net_r_allowed = 0`.
- **F6 — Read-only producer.** The fills-ledger producer must place no orders, amend no stops,
  and close no trades. It may call only broker read endpoints. Hard fence:
  `f6_broker_writes_allowed = 0`.
- **F7 — No false failure verdict.** A closed trade whose decision record already carries an
  outcome must be recorded as `already_closed`, never as `unmatchable`, and must not count
  toward `OUTCOME_LOOP_STALL`. Writing "the entry path never called `log_forex_decision()`"
  into the audit trail when a closed record exists is a **fabricated diagnosis** and is
  forbidden. Hard fence: `f7_false_unmatchable_allowed = 0`.

```yaml audit-spec
spec_version: 1
ticket: TICK-092
fills_path: data/ledger/oanda_fills.jsonl
matched_sidecar: data/oracle/.outcome_matched.json
causal_journal: data/agent/causal_journal.jsonl
required_fill_fields: [trade_id, pair, direction, fill_price, stop_price, timestamp]
terminal_buckets: [backfilled, already_known, already_closed, unmatchable, no_stop, error]
forbidden_broker_calls: [place_trade, close_trade, set_stop]
f1_unaccounted_allowed: 0
f2_silent_skip_allowed: 0
f3_zero_stop_rows_allowed: 0
f4_duplicate_trade_ids_allowed: 0
f5_null_net_r_allowed: 0
f6_broker_writes_allowed: 0
f7_false_unmatchable_allowed: 0
```

## Discipline

- The fence is the only source of thresholds, matching `audit/invariants_spec.md`'s contract.
- **F1 is the load-bearing one.** F3–F5 fix the present symptom; F1 is what makes the *next*
  silent drain impossible, because a new skip path cannot be added without breaking the bucket
  sum.
- The producer reconstructs from the broker rather than instrumenting order placement. This is
  deliberate: it recovers history instead of only capturing new trades, and it touches no
  execution-path file, so it needs no unlock against the standing shadow/execution freeze.
- **Expected day-1 state:** F7 failed on 13 historical sidecar entries; the backfill self-heals
  them (re-verifying a prior `unmatchable` against the record and upgrading it) rather than
  needing a one-off script, because a correction that only runs once cannot be a regression test.
  Measured after the fix: `seen=23 already_known=16 reclassified=13 unmatchable=7 no_stop=0`,
  buckets summing to 23. The 7 residual `unmatchable` (trade ids 47, 51, 72, 83, 93, 105, 231)
  are **genuine** gaps — no decision record carries those trade ids.
- **Known-honest limitation:** satisfying every invariant here does *not* imply Oracle has enough
  data to learn. The closed-loop corpus is 30 attributed FOREX outcomes across three months
  (26 LOSS / 4 WIN), out of 48 FOREX decision records. This spec fences the plumbing, not the
  statistics. Do not read a green F1–F7 as evidence the learning loop is productive — the loop
  is *closing*, and that is a different claim from the loop being *informative*.
