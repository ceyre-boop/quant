# L2 Exit Machine — GO-LIVE CHECKLIST

> The `SHADOW_MODE=False` flip is the first time the exit machine touches real positions. This is the
> pre-flight gate. Target flip date: **~2026-07-28** (after the 22-day shadow window closes). Flip only
> when every item below is **DONE/PASS**. Companion design doc: `docs/l2/EXIT_MACHINE_DESIGN.md`.

Status legend: ✅ DONE/PASS · ⏳ PENDING · ⛔ BLOCKER

---

## 1. ✅ set_stop round-trip confirmed on a market-open PRACTICE account
- **PASS — 2026-06-30**, `tests/test_oanda_set_stop.py::test_set_stop_practice_round_trip`.
- Real round-trip on practice account `101-001-35912324-001` (`OANDA_LIVE` unset): opened a 1-unit
  USD_CAD position → `bridge.set_stop` → broker confirmed `stopLossOrderTransaction.price` within 5dp →
  re-confirmed via `get_trade` → closed in `finally`. `1 passed in 8.38s`.
- Decimal-place handling verified in code (`oanda_bridge.py::_round_price`): **JPY pairs 3dp, others 5dp**.
- **Re-run before the flip** (creds must be in env; market open Mon–Fri):
  ```bash
  set -a; . ./.env; set +a; unset OANDA_LIVE
  python3 -m pytest tests/test_oanda_set_stop.py -v
  ```

## 2. ⛔→✅ run_daily idempotency guard (was the standing BLOCKER — now cleared)
- **Was ABSENT.** Live evidence pre-fix: `data/exec/exit_manager_shadow.jsonl` trade `105`, bar_date
  `2026-06-28`, `hold_count` stepped **1 → 3** across two runs on 2026-06-30 — a manual run colliding with
  the 08:30 launchd run (`scripts/com.alta.forex_exit_manager.plist`). Live, that double-steps the ATR
  trail and could double-amend a real stop.
- **DONE — guard added** in `run_daily` (`forex_exit_manager.py`): a top-level `state["processed"]` map
  `{trade_id: last_bar_date}` skips a trade whose current `bar.date` was already stepped (logs
  `action: "SKIP_DUPLICATE"`, no state advance, no broker write; guards shadow AND live). Lives entirely
  in `run_daily` + the state dict — `TradeState`, `step_trade`, `decide_exit`, and the golden fixture are
  untouched, so replay-match parity is preserved.
- Test: `tests/test_forex_exit_manager.py::test_idempotency_guard_blocks_same_day_double_step` (two
  same-`bar_date` live runs → one `set_stop`, `hold_count` stable, `SKIP_DUPLICATE`; a new `bar_date`
  steps again). Full suite **12 passed** incl. `test_replay_match_golden` + `test_module_default_is_shadow`.

## 3. ⏳ 22-day shadow window — zero divergences (audit on flip day)
- Run when the window closes. Audit `data/exec/exit_manager_shadow.jsonl` over the **post-guard window**
  (entries from the guard commit forward; the earlier known trade-105 double-step predates the fix):
  - For each `trade_id`: `hold_count` strictly increases across **distinct** `bar_date`s; no `bar_date`
    appears with two different `hold_count`s (the double-step signature — now structurally prevented).
  - Every `would_amend_stop_to` / `action` is sane (amends ratchet monotonically toward price; no CLOSE
    without a real exit `decision`); no `SKIP` from `market_data_unavailable` left unexplained.
  - State file (`data/exec/exit_manager_state.json`) parses, version 1, no orphaned trades.
- Status: **PENDING** until ~2026-07-28.

## 4. ⏳ param_change_log entry for the flip — DRAFTED, ready to paste
The flip is a logged param change (project rule #4). On flip day, append to
`data/agent/param_change_log.jsonl` (fill `ts`/`approved_by`):
```json
{"ts":"2026-07-28T<HH:MM:SS>Z","actor":"<user>","type":"exit_manager_go_live","files":["sovereign/execution/forex_exit_manager.py"],"change":"SHADOW_MODE True→False (forex_exit_manager.py:49)","rationale":"Step 5 go-live: set_stop round-trip PASS on practice; idempotency guard added + tested; 22-day shadow window audited clean (hold_count monotonic, zero divergences); replay-match parity green.","approved_by":"<approver>","reversible":true}
```

## 5. ✅ Rollback / abort plan — clean one-line revert, no state migration
- Abort = revert `SHADOW_MODE` to `True` (`forex_exit_manager.py:49`). No state migration: the state file
  tracks only `best_price`/`worst_price`/`hold_count`/`last_stop`/`processed` — broker-agnostic. Reverting
  resumes shadow logging immediately; in-flight trades keep their state.
- The `processed` guard also makes a restart-after-abort safe (the day's already-stepped trades won't
  re-step).

## 6. Flip-day runbook (~2026-07-28)
1. Re-run item 1 (set_stop round-trip) on a market-open day → PASS.
2. Run item 3 audit → zero divergences over the post-guard window.
3. Edit `forex_exit_manager.py:49` → `SHADOW_MODE = False`.
4. Append the item 4 entry to `param_change_log.jsonl` (real `ts`, `approved_by`).
5. Commit `[INFRA] L2 exit machine go-live: SHADOW_MODE False (param-change logged)`.
6. Restart the manager (`scripts/com.alta.forex_exit_manager.plist`); confirm the first live run logs
   `mode: "LIVE"` and acts only on AMEND/CLOSE decisions.
- **Abort at any point** → item 5.

---
_Created 2026-06-30. This session cleared items 1, 2, 5; drafted 4; left 3 pending the window close. No
flip performed; `SHADOW_MODE` remains `True`._
