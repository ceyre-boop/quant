# SHADOW DIVERGENCE SPEC — L2 Exit-Manager Parity Audit

This document numerically defines "parity holds" for the forex exit-manager
shadow window (2026-06-29 → ~2026-07-28) and is the pre-registered contract the
go-live decision scores against.

**Pre-registration statement.** This spec was written and committed BEFORE any
human or analyzer read the contents of `data/exec/exit_manager_shadow.jsonl`.
Everything below derives from code (`sovereign/execution/forex_exit_manager.py`,
`sovereign/forex/exit_machine.py`) and from incidents already documented in
`data/agent/param_change_log.jsonl` and the operational notes — never from the
log itself. The analyzer records this file's sha256 in every report; changing
thresholds after logs have been read requires a dated entry in §10 and a
`spec_version` bump, and the change history is the git history of this file.

---

## 1 · Purpose

The shadow window proves **parity** (the live manager makes exactly the
decisions the backtest kernel makes), not optimality. Two layers:

- **L1 — determinism**: the logged decision is reproducible from the logged
  inputs through the shared `decide_exit`. Any failure is code drift,
  nondeterminism, or state corruption. Tolerance: none (beyond log rounding).
- **L2 — input parity**: the decisions the manager made on OANDA bars match
  the decisions the backtester would make on yfinance bars for the same days.
  This measures the price-source divergence that go-live actually rides on.

## 2 · Log semantics (from code, not from logs)

Writer: `run_daily` → `_write_shadow_log` (sole appender). Record shapes:
1. **Normal step** — `run_ts` (UTC ISO, shared per pass), `mode`, `trade_id`,
   `pair`, `direction`, `bar_date`, `close` (full precision), `atr_pct`
   (round6), `signal`, `hold_count`, `hold_limit`, `best_price`/`worst_price`/
   `initial_stop` (round5), `decision`, `action`, `would_amend_stop_to`
   (round5|null), `trail_price` (round5), `reentry_signal`.
   **State fields are POST-step** (`res.new_state`): `hold_count` is pre+1,
   best/worst already folded with today's close. `initial_stop`/`hold_limit`
   are immutable.
2. **SKIP_DUPLICATE** — idempotency guard held; advances nothing.
3. **SKIP** — market data unavailable; has no `bar_date`.

Unlogged fields the analyzer must work around: `hold_today` (assumed per
machine block; cb_refresh ambiguity handled in §3), `last_stop` (reconstructed
by whole-log chaining), entry price (recovered from the first record: entry is
the best/worst member ≠ round5(close)). Days with zero open trades append
nothing — but `save_state` still runs, so the **state-file mtime is the
heartbeat**, not the log.

## 3 · L1 protocol — determinism replay

For every normal record: feed `decide_exit` the logged post-state re-based one
step (`hold_count−1`, logged best/worst — a fixed point of the max/min fold)
plus the logged bar (`close`, `atr_pct`, `signal`, `hold_today` per machine
block, donchian NaN — strict_mode is False in the live config).

- `decision`, `action`-consistency, `reentry_signal`, and post-state must match
  **exactly**; recomputed `trail_price` / `would_amend_stop_to` within
  `tol_price_abs` (derived from the log's round5/round6 quantization:
  worst-case ≈1.2e-5; threshold set 1.6× above).
- Records where the comparison sits inside `boundary_band_abs` of a decision
  threshold (|close−stop|, |close−trail|, |trail−last_stop|) are
  **BOUNDARY_INDETERMINATE**: excluded from the L1 denominator, counted, and
  warned above `boundary_share_warn`.
- cb_refresh ambiguity (`hold_today` unlogged): records with
  `signal == direction` and `hold_count ≥ 20` replay under both plausible
  `hold_today` values {60, 29}; a match under either passes with an
  `AMBIG_HOLD_TODAY` note.
- **Required L1 pass rate: 100% of non-boundary records.** Any miss is a FAIL
  and escalates URGENT — there is no "explained" L1 failure.
- Chaining checks (whole log, never window-sliced): consecutive normal records
  per trade must advance `hold_count` by exactly 1 with strictly increasing
  `bar_date`; a same-`bar_date` re-step or `hold_count` jump is a DOUBLE_STEP →
  C3 if covered by the incident register, else C5.

## 4 · L2 protocol — input-parity replay

Controlled experiment varying ONLY the price source: re-derive daily bars via
the backtest path (`yf.download(pair=X, auto_adjust=True)`, mirroring
`ForexBacktester._download_price`), recompute ATR% via the same
`ForexSignalEngine._compute_atr_pct`, take `signal`/`hold` from the log
(re-deriving signals would conflate macro drift with bar drift — out of scope),
seed each trade's entry state from its first record (§2), and step the shared
kernel over the yfinance bars.

- Date alignment: one constant shift per pair ∈ {−1, 0, +1} chosen to minimize
  median |Δclose| (OANDA D candles are 17:00-NY aligned; yfinance differs);
  a shift that changes mid-window is C2.
- Score per (trade, bar_date) from the first logged bar through the FIRST
  CLOSE decision on either side. The shadow tail after a CLOSE decision (the
  broker trade stays open in shadow) is L1-checked but never L2-scored.
- **Required: decision match rate ≥ `l2_decision_match_min` over the full
  window**, with every mismatch classified (§5) and zero left unexplained.
- Always report per-pair |Δclose| and |Δatr| percentiles — the continuous
  divergence measure — and warn when median |Δclose| > `close_delta_warn_abs`.
- yfinance unreachable → `l2_status: SKIPPED_OFFLINE` (loud), gate PENDING,
  L1 + watchdog still run; ≥ `l2_offline_consecutive_warn` consecutive skips
  escalates IMPORTANT.

## 5 · Divergence taxonomy

| Class | Meaning | Evidence required |
|-------|---------|-------------------|
| C1 | Price-source | Substitution rerun (OANDA close and/or atr into the L2 bar, that bar only) reproduces the logged decision AND the input delta is real (|Δclose|>2e-5 or |Δatr|>1e-6). Variant + deltas recorded. |
| C2 | Timing/calendar | Unalignable date, holiday, weekend candle, mid-window shift instability. |
| C3 | Documented incident | The bar_date/trade is covered by the incident register; the register entry is cited. |
| C4 | Data unavailable | SKIP records; excluded from the L2 denominator. |
| C5 | **UNEXPLAINED** | Everything else. **Allowed count: 0.** Same-day URGENT escalation. |

**Standing-incident register (seeded pre-log-read, from documented history):**
- `INC-2026-06-29`: first scheduled fire crashed (`ModuleNotFoundError`, import
  path); manual recovery run same day. Multiple `run_ts` per day expected.
- `INC-2026-06-30`: manual + launchd double-step before the idempotency guard
  landed (param_change_log 2026-06-30T22:32:54Z). Applied ONCE as a standing
  offset: shadow `hold_count` runs permanently +1 versus a clean replay from
  entry — pre-explaining both the chaining jump on that date and a one-day
  TIME-exit skew near day ~59. Not re-flagged daily.
Additions to this register after first log-read require a dated §10 entry
citing an external document (param_change_log / operational note) — the
register explains divergences from *documented operations*, never from the
log's own contents.

## 6 · Go-live gates (scored at window close, ~2026-07-28)

GO requires ALL of:
1. L1 pass rate = 100% of non-boundary records (entire window).
2. C5 count = 0 (entire window).
3. L2 decision match ≥ `l2_decision_match_min`.
4. Scored coverage ≥ `min_scored_weekdays` weekdays and ≥ `min_scored_records`
   normal records.
5. `mode: LIVE` records seen = `live_records_allowed` (zero) and
   `SHADOW_MODE is True` in code at audit time.
Anything failing 1/2/5 → NO-GO. Insufficient coverage → NOT-YET. The daily
report carries the running scorecard.

## 7 · Escalation policy

Writes to `data/agent/messages_to_colin.json` (pulse_check conventions:
prepend, cap 50, atomic replace, dedupe by type+date, `source: shadow_audit`,
text prefix `[AUDIT]`):
- URGENT 🔴 — any C5; any L1 failure; any LIVE-mode record; scheduler
  staleness (§8); analyzer crash (the crash handler itself escalates).
- IMPORTANT 🟡 — median |Δclose| above `close_delta_warn_abs`; boundary share
  above `boundary_share_warn`; ≥2 consecutive offline L2 skips; installed-plist
  hash drift (§8).
- FYI 🟢 — report written; gate scorecard state changes.

## 8 · Operational watchdog (the audit independently guards the window)

Checked every run:
- **Heartbeat**: on a weekday after 08:30 + `staleness_grace_min`, the state
  file `data/exec/exit_manager_state.json` must have today's mtime (save_state
  runs even with zero open trades) → else URGENT.
- **Loadedness**: `launchctl list` contains `com.alta.forex_exit_manager`.
- **Plist integrity**: sha256(installed plist) == sha256(`scripts/` copy) —
  detects a recurrence of the 2026-07-01 mass clobber (18 installed agents
  overwritten with JSON arrays; restored 2026-07-02, evidence in
  `audit/evidence/`).
- **Mode**: `forex_exit_manager.SHADOW_MODE is True` (import, read-only).

## 9 · Machine-readable constants

Exactly one fenced block below; the analyzer parses it and refuses to run if
the fence count ≠ 1. Reports record `spec_version`, `spec_sha256` (whole file),
and the analyzer's git commit.

```yaml audit-spec
spec_version: 1
shadow_window_start: 2026-06-29
shadow_window_expected_end: 2026-07-28
tol_price_abs: 2.0e-5
boundary_band_abs: 2.0e-5
boundary_share_warn: 0.02
l1_required_pass_rate: 1.0
hold_today_assumed: 60
hold_today_ambiguous_alternates: [60, 29]
cb_refresh_ambiguity_min_hold_count: 20
l2_decision_match_min: 0.95
c1_min_close_delta: 2.0e-5
c1_min_atr_delta: 1.0e-6
c5_allowed: 0
min_scored_weekdays: 15
min_scored_records: 30
live_records_allowed: 0
staleness_grace_min: 30
exit_manager_fire_local: "08:30"
l2_offline_consecutive_warn: 2
close_delta_warn_abs: 5.0e-4
messages_cap: 50
report_dir: audit/reports
shadow_log: data/exec/exit_manager_shadow.jsonl
state_file: data/exec/exit_manager_state.json
exit_manager_label: com.alta.forex_exit_manager
exit_manager_plist_tracked: scripts/com.alta.forex_exit_manager.plist
```

## 10 · Change control

Amendments require: a dated entry here (what changed, why, whether logs had
been read), a `spec_version` bump for semantic changes, and the same-commit
analyzer update if parsing is affected. The drift surface is one file — this
one.

- 2026-07-02 — v1 — Initial spec. Committed before any log read; analyzer does
  not exist yet.
