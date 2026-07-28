# TICK-092 — Close the Oracle loop (the loop is empty, not slow)

**Status:** BUILT 2026-07-28. Branch `sovereign-v2`.
**Date:** 2026-07-28. Supersedes the 4-session roadmap proposed in session chat.

---

## ⚠️ DIVERGENCE — the plan's own root cause was wrong

Recorded per the plan→build rule: reality diverged, so the plan is corrected rather than
quietly improvised around. **The title is wrong: the loop was not empty.**

**What the plan claimed, and what measurement showed:**

| Plan's claim | Measured reality |
|---|---|
| `data/ledger/oanda_fills.jsonl` is 0 bytes with no writer | **24 rows, every one with `stop_price > 0`**, covering every trade id in the sidecar. My earlier `wc -l` hit `data/oracle/` and `data/agent/` paths that do not exist; I never measured `data/ledger/`. |
| Silent no-stop skip drains every closed trade ("Drain 1") | **`no_stop=0`.** The branch never fires. Real as a *latent* hazard, not the active bug. |
| Entry path never called `log_forex_decision()` ("Drain 2") | **False for 13 of 20.** Those trades have decision records carrying real outcomes (10 LOSS / 3 WIN). |
| Oracle has zero closed-loop outcomes | **30 attributed FOREX outcomes** (26 LOSS / 4 WIN) across 48 FOREX decision records. The loop is closing. |

**The actual defect.** `update_outcome()` returns `False` both when no record exists and when
one exists but is already CLOSED — `_update_outcome_in_month` only inspects records with
`outcome is None`. `pulse_check` read the second case as the first and stamped 13 closed trades
`unmatchable` with the reason *"the entry path likely never called `log_forex_decision()`"*.
The sidecar then short-circuited them on every later run, making the false verdict
**self-preserving**. That fabricated diagnosis in the audit trail is what sent the original
strategy synthesis — and this plan's first two hypotheses — chasing a bug that did not exist.

The lesson is the plan's own: the system was grading its own homework, but graded itself *too
harshly*, and the false negative was the expensive part.

**Still genuinely broken:** 7 trades (ids 47, 51, 72, 83, 93, 105, 231) have no decision record
at all. That is the real residue of Drain 2, at roughly a third of the assumed size.

---

## Context

A strategy synthesis proposed a 4-session sequence: build `swap_model.py` → decade rerun →
design an exit model → open the self-play ignition gate. Verification against the repo shows
**two of its four load-bearing premises are already satisfied and two are false**, and the
corrections invert the recommendation.

### Already banked (the proposed "Session N" and "N+1")

`sovereign/forex/swap_model.py` was built, committed and **applied** today
(`bed13a2` → `697da48` → `f91fc08`). The decade rerun ran. Gate flag
`tick_024_carry_fix_landed` = **PASS**.

**The edge survived honest costs.** Portfolio Sharpe 0.6886 → **0.6452**; OOS Sharpe
1.2504 → **1.1919**, still ROBUST (decay 2.169 → 2.247). No sign flips, no verdict changes.
The "9×" was the *swap-rate magnitude* error, not the Sharpe — the Sharpe moved −4.7%.
One real caveat, already disclosed: **OOS 95% CI lower bound fell 1.001 → 0.948**, so it no
longer clears 1.0.

This answers the question `GOALS.md` names as the reason the whole system exists — *"a 9× cost
underestimate turns a real Sharpe of 0.4 into a reported Sharpe of 1.25."* It did not.
`GOALS.md`'s framing of TICK-024 is now stale and should be updated to record the measured
outcome rather than the fear.

### Premises that were false

1. **"There is no exit model."** There is: `sovereign/forex/exit_machine.py` (6-state) plus
   `sovereign/execution/forex_exit_manager.py` running `SHADOW_MODE = True` (line 49), with 31
   shadow decisions logged in `data/exec/exit_manager_shadow.jsonl` (23 AMEND_STOP / 5 HOLD /
   2 CLOSE / 1 SKIP). Building an exit model is not the gap; deciding go-live is. **Deferred** —
   it needs an execution-path unlock in `NEXT.md` and more than 31 decisions.

2. **"The causal journal is live — Oracle can learn from closed-loop data immediately."**
   It cannot. This is the real problem and the subject of this ticket.

### The ignition gate blocker is not cost

`evaluate_gate()` run live — gate **CLOSED**, 3 of 4 guards FAIL:

```
[PASS] tick_024_carry_fix_landed
[FAIL] hyp_071_net_confirmed
[FAIL] value_board_is_net          (board carries 'gross_R_caveat')
[FAIL] hyp_071_revival_confirmed   (METRIC_ARTIFACT verdict stands)
```

HYP-071 was never blocked on cost basis. It died of **metric asymmetry**: `λ·DD` only penalises
the arm carrying forecast variance, so EXIT_NOW dominance was mechanically guaranteed. The draft
prereg states gross→net is *"necessary but not sufficient."* Separately,
`Plans/TRUST_DECISION_BRIEF.md` confirms HYP-071-v2 applies BH within its own family regardless
of the retro/forward call — so the Bonferroni decision does **not** gate it.

---

## Root cause — two independent drains

Oracle's ground truth across three months of decision logs:

| Store | Volume | Usable outcomes |
|---|---|---|
| `data/decision_logs/decisions_*.jsonl` | 3,517 (3,469 ICT / **48 FOREX**) | 3,478 `EXPIRED`; ~**30 closed** (26 LOSS / 4 WIN) |
| `data/agent/ict_causal_chain.jsonl` | 255,406 rows | **0** — 100% `VETOED` (247,659) / `DISCARDED` (7,747), `outcome=None` on every row |
| `data/agent/causal_journal.jsonl` | **0 rows** | 0 |
| `data/oracle/.outcome_matched.json` | 23 OANDA closes | 3 matched, **20 `unmatchable`** |

*(The two journals are separate by design — ICT setup-level vs OANDA-close backfill. Not a path bug.)*

**Drain 1 — silent skip starves everything.** `data/ledger/oanda_fills.jsonl` is **0 bytes and has
no writer anywhere in the codebase**. It is read by four call sites
(`pulse_check.py:511`, `scripts/backfill_decision_records.py:56`, `scripts/proof_of_life.py:25`,
`pulse_check.py:475`) and produced by none. Consequence in
`pulse_check._backfill_decision_outcomes()`: `fills_by_id` is empty → `fill` is always `None` →
`stop = 0.0` → line 724 `stop <= 0` → **`continue` at line 727, which fires before
`n_attempted += 1` at line 736.** Every closed trade is skipped *before the stall counter
increments*, so `OUTCOME_LOOP_STALL` never fires and the loop reports
`matched=0 attempted=0` — indistinguishable from "nothing new to match."

This is precisely the failure mode `GOALS.md` warns about: a system grading its own homework.
It also starves `backfill_decision_records.py`, the designated recovery tool.

**Drain 2 — entry path doesn't produce matchable records.** 20 of 23 OANDA closes are marked
`unmatchable` with the system's own recorded reason: *"no matching decision record after 7d —
the entry path likely never called `log_forex_decision()`."* There is exactly **one** production
caller (`sovereign/forex/forex_specialist.py:118`), reached via
`scripts/forex_live_scan.py:157`. Whether it fails to fire, or fires with a pair/timestamp that
`_outcome_entry_match` cannot reconcile against OANDA's `openTime`, must be determined by
measurement — not assumed.

**Fixing Drain 1 alone is insufficient**; fixing Drain 2 alone leaves the silent skip in place.
Both are required for a single closed outcome to reach Oracle.

---

## Approach

**Design decision: reconstruct the fills ledger from the broker, do not instrument order
placement.** OANDA is the source of truth and `OandaBridge` is already read-capable
(`get_trade`, `get_closed_trades`, `get_open_trades`; open/closed trades carry `stopLossOrder`
per the docstring at `oanda_bridge.py:404`). This recovers *history* rather than only capturing
new trades, and touches **no execution-path file** — satisfying the standing freeze without
requiring an unlock.

### Work items

**A1 — Make the silent skip loud (spec-first).**
Write the pass/fail spec before the check, per the spec-first constraint. In
`pulse_check._backfill_decision_outcomes()`, move the no-sane-stop `continue` so it is counted
and reported (new `n_no_stop` bucket surfaced in the `OUTCOME_LOOP` log line and in
`OUTCOME_LOOP_STALL`). A closed trade must never leave the loop unaccounted for. Preserve the
existing correct behaviour: still refuse to fabricate an R from a bad stop.

**A2 — Produce `data/ledger/oanda_fills.jsonl`.**
New `scripts/rebuild_fills_ledger.py` (read-only against OANDA; idempotent; keyed on trade id).
For each closed and open trade, resolve `stop_price` from `stopLossOrder.price` via
`OandaBridge.get_trade()`, and emit the schema the four existing consumers already expect
(`trade_id`, `fill_price`, `stop_price`, plus pair/time). Reuse the existing readers' field
names verbatim — do not invent a new schema.

**A3 — Diagnose Drain 2 by measurement.**
With A2's ledger in place, run `scripts/backfill_decision_records.py` (exists; idempotent;
norm-pair + same-UTC-day matching per prior sessions) and record how many of the 20
`unmatchable` recover. Then determine empirically whether the residue is (a) `log_forex_decision`
never firing, or (b) a scan-time-vs-fill-time timestamp mismatch. Report the finding; do not
pre-commit to a fix.

**A4 — Populate `net_r`.**
`pulse_check._append_causal_journal()` writes `net_r: None` and a now-stale comment
`"cost_estimated: True  # TICK-024 swap cascade not yet applied"` (line ~651). TICK-024 *has*
landed. Wire `ratediff_financing_rate()` from `sovereign/forex/swap_model.py` so `net_r` is real.
**This is a hard prerequisite for HYP-071-v2** — that hypothesis adjudicates on NET returns, and
the journal currently cannot supply them.

**A5 — Record the four operator decisions** (below) in `NEXT.md` and the appropriate artifacts.

### Operator decisions (answered this session)

| Decision | Ruling | Action |
|---|---|---|
| Next priority | Fix the empty closed loop | This ticket |
| HYP-071-v2 metric fix | **(b) pure E[R]** — drop the λ·DD term | Update `research/HYP-071_v2_prereg.DRAFT.md` §3 to lock mechanism (b); Colin assigns + locks the hash. Must not reuse `c4f29ac3…`/`3d500bda…`/`c1fab807…` |
| BH correction scope | **Retroactive** — one family, take the hits | Separate ticket. Re-runs BH across all 14 `CONFIRMED` entries incl. HYP-045 (live v015 basis). Not started here — it can demote load-bearing edges and deserves its own session |
| TICK-024 audit (`697da48` claimed a sign-off that did not exist) | **Retroactively approve** — numbers verified | Record retroactive approval in `NEXT.md`; the applied code was diff-verified and 0.6452 reproduced exactly |

---

## Files

| File | Change |
|---|---|
| `sovereign/oracle/pulse_check.py` | A1 (account for no-stop skips, ~lines 715–791), A4 (`_append_causal_journal` net_r + stale comment, ~line 651) |
| `scripts/rebuild_fills_ledger.py` | **New** — A2, read-only OANDA → `data/ledger/oanda_fills.jsonl` |
| `audit/fills_ledger_spec.md` | **New** — A1/A2 pass-fail spec, written first |
| `research/HYP-071_v2_prereg.DRAFT.md` | §3 lock mechanism (b); leave `DRAFT_PENDING_OPERATOR_APPROVAL` until Colin hashes |
| `GOALS.md` | Replace the stale "9× → Sharpe 0.4" framing with the measured outcome |
| `NEXT.md`, `tickets/backlog.md` | TICK-092 entry + the four decisions |

**Untouched (freeze):** `sovereign/execution/forex_exit_manager.py`, `decide_exit`,
`ict/pipeline.py`, and anything importable by the live/backtest execution path. No
`config/parameters.yml` or `config/ict_params.yml` change, so no `param_change_log.jsonl` entry
is required.

---

## Verification

1. **Spec first:** `audit/fills_ledger_spec.md` exists and is hashed before A1/A2 code lands.
2. **Isolation invariant:** `python3 -m pytest tests/ -k test_pipeline_does_not_import_sovereign`
3. **Regression:** `python3 -m pytest tests/ -q`. ICT pipeline baseline is
   **4 failed / 23 passed** (`-k "ict and pipeline"`) — pre-existing, per `CLAUDE.md`. Do not
   absorb those into this ticket's result.
4. **A2 produced data:** `wc -l data/ledger/oanda_fills.jsonl` > 0, and every row carries a
   `stop_price` > 0. Re-run to confirm idempotence (line count stable).
5. **A1 no longer silent:** the `OUTCOME_LOOP:` log line reports a non-zero
   `attempted` or `no_stop` bucket where it previously read `matched=0 attempted=0`.
6. **A3 measured:** report recovered-vs-residual counts against the 20 `unmatchable`.
7. **A4 real:** at least one `causal_journal.jsonl` row with `net_r` populated and
   `cost_estimated: false`.
8. **Gate unchanged:** re-run `evaluate_gate()` — must still read **CLOSED**. This ticket does
   not open the ignition gate and must not appear to.
9. **Push before session end** (standing constraint), and log to `NEXT.md`.

**Definition of done:** at least one real closed trade reaches Oracle with a populated `net_r`,
and no closed trade can leave the backfill loop unaccounted for.

---

## Out of scope

- Flipping `SHADOW_MODE = False` — needs an execution-path unlock and more than 31 decisions.
- Running HYP-071-v2 — blocked on Colin hashing the prereg **and** on A4 supplying net returns.
- The retroactive BH re-run — its own ticket; it can demote HYP-045.
- Fixing the ICT fill-rate bottleneck (TICK-028, ~98% of setups expire unfilled). That is why
  the ICT journal is 100% vetoes, but the carry path is where the proven edge lives.

## Risk

The honest possibility this ticket must not paper over: fixing both drains may reveal the
closed-loop corpus is **~30 outcomes over three months, 26 of them losses**. If so, Oracle has no
statistical basis to learn from on any 30–60 day horizon, and the "system starts compounding on
confirmed performance" claim is unsupported regardless of how well the loop is wired. Report
that number plainly if it lands — the deliverable is a working loop *and* an honest count of what
is in it.
