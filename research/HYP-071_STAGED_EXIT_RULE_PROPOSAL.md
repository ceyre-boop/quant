# HYP-071 — Staged Exit-Rule Proposal (UNAPPLIED · FROZEN until 2026-07-28 + Colin ledger stamp)

**This document changes NOTHING.** It records exactly what the value-table result *would* imply for
the live exit machine, so that after the July-28 shadow go-date and an explicit Colin ledger approval,
the change is a one-pass, pre-reviewed edit rather than a re-derivation. Do not apply any of this
before BOTH conditions hold. See `research/HYP-071_VALIDATION_REPORT.md` for the provisional verdict
and the gross-returns caveat that could erase it.

## Preconditions before ANY of this may be applied (all must be true)

- [ ] Date ≥ 2026-07-28 AND the L2 shadow exit-machine window has passed clean and gone live.
- [ ] The table has been **recomputed on NET returns** (corrected TICK-024 financing) and the
      forward-consistent EXIT_NOW divergences **survive** net-of-carry.
- [ ] Regime-window (60d↔252d) agreement clears **0.90** on the net recompute.
- [ ] Colin has written an explicit **CONFIRMED** (or APPROVED-FOR-APPLICATION) entry to
      `data/agent/hypothesis_ledger.json` for HYP-071.
- [ ] The exit-path unlock is recorded in `NEXT.md` (extends CLAUDE.md standing constraint / NN#1).

## The proposed rule change (only if preconditions clear)

Today every evaluated cell resolves to `static_action = HOLD_AND_TRAIL`. The table proposes flipping a
**small, specific set** of cells to `EXIT_NOW`. The candidate set (decade, gross) is the 9
forward-consistent, CPCV-stable, economically-sensible divergences:

| cell | ATR tercile | excursion | hold frac | RSI-extreme | carry | current → proposed |
|---|---|---|---|---|---|---|
| 72 | high | underwater | early | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 84 | high | modest | early | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 76 | high | underwater | mid | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 88 | high | modest | mid | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 92 | high | modest | late | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 44 | mid | underwater | late | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 56 | mid | modest | late | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 8  | low | underwater | late | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |
| 20 | low | modest | late | no | aligned | HOLD_AND_TRAIL → EXIT_NOW |

**Economic summary of the rule:** *"Exit (don't hold-and-trail) when volatility is high at any hold
stage, or when a lower-vol position reaches late-hold still only underwater-to-modest."* The
**high-ATR early/mid cells (72, 84, 76, 88)** carry the strongest margins and largest samples
(n=70–154) — if only a subset is ever applied, start there.

## How it would be applied (frozen files — for reference only, DO NOT EDIT NOW)

The exit decision is made in `sovereign/forex/exit_machine.py` (and driven live by
`forex_exit_manager` / `decide_exit`) — **all frozen**. The application shape, when unlocked, is a
lookup: at each bar, compute the cell coords (already computed identically in
`sovereign/discovery/exit_value_table.py::decode_cell`/encode), and if the current cell is in the
approved EXIT_NOW set, return EXIT instead of HOLD_AND_TRAIL. Preferred implementation is a **data
file** the exit machine reads (e.g. `config/exit_value_overrides.yml`) so the rule is logged,
diff-able, and revertible without a code change — consistent with CLAUDE.md NN#4 (no live parameter
change without a logged rationale in `data/agent/param_change_log.jsonl`).

## Rollback

Because the preferred mechanism is a config override file, rollback is deleting the file (or emptying
the override set) and logging the reversal. No code revert required.
