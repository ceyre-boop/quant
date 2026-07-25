# TICK-044 — DAILY_LOSS_HALT is inert on the primary path — diagnosis + staged fix

**Status: DIAGNOSED + PATCH STAGED, NOT APPLIED.** Frozen until 2026-07-28 (shadow audit).

## Confirmed mechanism (file:line)

- `execution/risk.py:102` — `AccountState.daily_pnl_frac: float = 0.0` (default).
- `execution/risk.py:171-173` — the only read: `if state.daily_pnl_frac <= -DAILY_LOSS_HALT: ... framework_halt = True`.
- `execution/harness.py:401` — `acct = account or risk.AccountState(equity=100_000.0, peak_equity=100_000.0)`.
  `main()` (`execution/harness.py:560-563`) never passes `account=`, so the scheduled
  `--live` path (launchd, 09:25 ET Mon–Fri, per the module docstring at `harness.py:34`)
  always gets a **fresh** `AccountState` with `daily_pnl_frac=0.0`.
- **Nothing in the repo ever writes to `daily_pnl_frac` after construction** — confirmed by
  `grep -rn daily_pnl_frac` across all `*.py`: the only three hits are the field
  declaration and the two read sites in `risk.py`. `apply_risk()`
  (`execution/harness.py:133-150`) calls `risk.check()` once per candidate fill but never
  writes anything back onto `state`.

So the defect is stronger than the original backlog description ("fires only on a
mid-session restart with fills already written"): there is no code path, restart or
otherwise, that populates `daily_pnl_frac` from real fills at all. A restart wouldn't
help either, because nothing reads `fill_log.jsonl` back into `AccountState` on
construction. The -2% `DAILY_LOSS_HALT` (`RISK_FRAMEWORK.md`, ratified 2026-07-20) has
never been able to fire since it was wired.

**Related finding, same root cause, not in scope to fix here:** `AccountState.consecutive_losses`
(`risk.py:103`) has the identical problem — also defaulted, also never written anywhere in
`execution/*.py`. The backlog entry's claim that "consecutive-loss gates ... are correctly
computed from all historical fills" does not hold for this file; the hits under that name
elsewhere (`sovereign/orchestrator.py`, `sovereign/risk/*`) belong to a different system
(live forex, not this gapper/ICT harness) and don't feed `AccountState` here. The staged
patch below fixes both gates together since the fix is the same shape — see "Design", below.

## Why it can't be fixed by looking at `fill_log.jsonl` alone

`FillRecord` (`execution/harness.py:83-121`) persists `net_return` (a return fraction) but
**never** the dollar/equity-fraction size that was actually put at risk — `risk_size_mult`
is logged, but the requested `risk_fraction` and `RiskDecision.detail["effective_risk"]`
that `risk.check()` already computes (`execution/risk.py:229`) are discarded after
`apply_risk()` returns. Without that, `net_return` alone can't be turned into "fraction of
session-start equity" — a 2% return on a position sized at 0.1% of equity is not the same
daily hit as a 2% return sized at 2% of equity.

## Design

Two additive pieces, both staged in `research/TICK-044_staged_patch.diff` against
`execution/harness.py` (frozen — not applied):

1. **Persist what's needed.** Add `FillRecord.effective_risk_frac` and set it in
   `apply_risk()` from `RiskDecision.detail["effective_risk"]` (already computed, just
   previously thrown away). Purely additive field, defaults `None`, doesn't change any
   existing row's meaning.
2. **Arm the gate at two points:**
   - **Within a run:** `apply_risk()` now accumulates
     `state.daily_pnl_frac += net_return * effective_risk_frac` and updates
     `state.consecutive_losses` after every *allowed* (capital-at-risk) fill, so a losing
     streak inside a single scheduled run can trip the halt for later candidates the same
     morning — today it can't, even in principle.
   - **At session start:** `run_session()` calls the new dormant helper
     `execution/daily_pnl_store.reconstruct_daily_state(day, out_dir)` to seed
     `daily_pnl_frac`/`consecutive_losses` from whatever `fill_log.jsonl` rows already
     exist for `day` — covers a re-run of the same day or a restart mid-session.

`execution/daily_pnl_store.py` (new, **not frozen, dormant** — no existing file imports it,
so it changes nothing until the patch above wires it in) does the reconstruction: sums
`net_return * effective_risk_frac` over today's non-`SKIP_*` rows, skipping any row that
predates the patch (no `effective_risk_frac` logged yet) rather than guessing — no silent
mocking. 4/4 unit tests green (`tests/test_daily_pnl_store.py`).

## What stays a known gap after this lands

`AccountState.equity` / `peak_equity` are still hardcoded flat at 100,000 (`harness.py:401`,
unchanged by the patch) — there's still no live equity feed, so the Art. 3 drawdown ladder
(`ladder_action`, `risk.py:130`) remains unarmed for the same reason it always has been.
That's `harness.py`'s own documented comment, still accurate, and out of scope for TICK-044,
which is specifically the daily-loss overlay.

## Apply steps (2026-07-28, after the shadow-audit freeze lifts)

```bash
cd ~/quant
git status                                   # confirm nothing else pending on harness.py
git apply --check research/TICK-044_staged_patch.diff   # dry run first
git apply research/TICK-044_staged_patch.diff
python3 -m pytest tests/test_daily_pnl_store.py -q      # dormant helper, should already be green
python3 -m pytest tests/ -k "harness or risk" -q         # existing harness/risk suite, must stay green
python3 -m pytest tests/ -k "test_pipeline_does_not_import_sovereign" -q   # standing isolation gate

# Rationale + amendment procedure (RISK_FRAMEWORK.md requires this for any change
# touching DAILY_LOSS_HALT):
echo '{"date":"2026-07-28","file":"execution/harness.py","change":"TICK-044: seed and \
accumulate AccountState.daily_pnl_frac/consecutive_losses instead of leaving them at \
their 0.0/0 defaults for the life of every process, arming DAILY_LOSS_HALT for the \
first time since it was ratified","ticket":"TICK-044"}' >> data/agent/param_change_log.jsonl

git add execution/harness.py execution/daily_pnl_store.py tests/test_daily_pnl_store.py \
       data/agent/param_change_log.jsonl tickets/backlog.md NEXT.md
git commit -m "[RISK] Arm DAILY_LOSS_HALT — seed+accumulate daily_pnl_frac (TICK-044)"
git push origin sovereign-v2
```

If `git apply --check` fails (harness.py has moved since this diff was staged), re-derive
the patch against current line numbers rather than forcing it — the logic above (persist
`effective_risk_frac`, accumulate in `apply_risk`, seed in `run_session`) is what has to
land; the exact line numbers are not load-bearing.
