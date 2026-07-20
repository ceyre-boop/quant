# SYSTEM STATUS

**Generated 2026-07-19 · Autonomous agent setup · branch `sovereign-v2`**

## Autonomous operation — what runs without you

Three Claude agent plists were authored 2026-07-19 and are ready to install (see below). Once installed, the system runs three autonomous Claude sessions daily with no manual intervention: a morning agent at 07:55 ET (Mon–Fri) that pulls the information layer, generates Oracle bias, scans gapper candidates through the frozen HYP-107/HYP-093 filters, and updates the prop sim dashboard; an EOD agent at 16:00 ET (Mon–Fri) that coordinates post-harness fill reconciliation, runs `execution.eod`, updates the dashboard, and writes the Obsidian session note; and a research agent at 21:00 ET (Sun–Thu) that pulls recent mover data, runs micro-backtests, appends to `research/weekly_pattern_update.md`, and pre-registers any pattern that survives a first-pass screen. All three agents read `AGENT_DIRECTIVE.md` as their standing order, commit after every pass, and push to `sovereign-v2`. The sole action required of you: run `bash ~/quant/scripts/install_agent_plists.sh` in your terminal once, then run the test command in that file's header comment to verify the loop fires correctly.

---

**Generated 2026-07-18 · TICK-040 · branch `sovereign-v2`**

Six layers assembled and wired. Read the two tables together — **layer health and
edge status are different questions**, and a green stack does not make an
unproven edge tradeable.

---

## 1. Layers

| # | Layer | Module | State | Notes |
|---|---|---|---|---|
| 1 | Information | `execution/context.py` | **LIVE (code)** · plist PENDING INSTALL | Consolidated morning context; per-source FRESH/STALE/SILENT_NULL/UNAVAILABLE |
| 2 | Model | `execution/bias.py` | **LIVE (code)** · plist PENDING INSTALL | Bias recorded + scored. **Gates nothing** by design |
| 3 | Signal | `execution/signals.py` | **LIVE** | Ranked GO/NO-GO, NO_GO rows retained with reasons |
| 4 | Execution | `execution/harness.py` | **LIVE + LOADED** | `com.alta.execution_harness` 16:05 ET installed |
| 5 | Risk | `execution/risk.py` | **LIVE** | Ratified five enforced; 3 unlegislated gates refused |
| 6 | Feedback | `execution/eod.py` + `obsidian.py` | **LIVE (code)** · plist PENDING INSTALL | EOD note → `Trading/Ops/System-EOD-{date}.md` |

**Tests:** 1541 collect cleanly. Pre-existing failures triaged 2026-07-19: in
`test_ml_stack.py`, only `TestPCACompressor` + `TestBlackScholes` tested deleted
APIs (`pca_compressor` removed, `bs_*` refactored into `VolRegimeSignal`) — now
**skipped**. The other 12 failures there are **genuine risk-layer regressions**
against still-existing Kalman/LQR/Pegasus/TradeMDP modules (OPEN_ITEMS #7), no
longer masked. ICT session-classifier + gate-ordering drift still predate this work.

### Daily flow once installed

```
02:30  oracle cycle          FRED + daily panel          (existing)
08:00  com.alta.system_morning   L1 context -> L2 bias   [PENDING INSTALL]
09:30  market open
16:05  com.alta.execution_harness  L3 signals -> L4 fills -> L5 risk   [LOADED]
16:30  com.alta.system_eod       L6 reconcile -> Obsidian   [PENDING INSTALL]
```

Layers 3–5 run inside the 16:05 harness job because SIP quotes are 403 inside a
15-minute recency window on this account tier — fills can only be priced after
the session's quotes age past that boundary.

---

## 2. Edge status — UNCHANGED by this work

| Edge | Verdict | Reality |
|---|---|---|
| Forex carry (v015) | Real, p<0.001 | **Regime-fragile.** Forward-walk on unseen 2025-26 ≈ Sharpe 0 |
| HYP-093 gapper fade | `VALID_BUT_BELOW_FLOOR` | 0.023%/day vs 0.05% floor — under half |
| HYP-107 runner filter | `REAL_BUT_MARGINAL_EXECUTION_UNRESOLVED` | Sealed holdout could not be reproduced (TICK-039) |
| Crowd prediction (L1) | 23+ clean nulls | Zero confirmed edges |

Live results to date: **3W/24L**, −$445.68 realised on the practice account.
Prop-funnel EV measured at **0.0** (TICK-022). Paper trades toward the 30-trade
go-live gate: **1**.

**Assembling six layers changed none of the above.** The system is now able to
tell you the truth faster; it has not made the edges bigger.

---

## 3. Information health — measured, not assumed

Latest `execution/context.py` run: **3 of 7 sources FRESH (43%)** after the
sentiment plist was loaded 2026-07-19.

| Source | Status | Detail |
|---|---|---|
| `fred_macro` | FRESH | 16 series |
| `daily_panel` | FRESH | |
| `sentiment_board` | FRESH | plist installed + loaded 2026-07-19 (`plist_manifest`: OK) |
| `briefing` | STALE 37h | own plist never installed; survives via an in-process call |
| `reddit` | **SILENT_NULL** | file 0.9h old, `posts_scanned: 0` |
| `gdelt` | UNAVAILABLE | 0 rows ever ingested |
| `calendar` | UNAVAILABLE | ForexFactory 403 |

`reddit` is the instructive one: a freshness check calls it healthy. It is not.

### Scheduler drift — `python3 scripts/plist_manifest.py`

- **NOT_INSTALLED (4):** `sentiment_update`, `gdelt_retry`, `ib_shortable`, `oracle.market_briefing`
- **FAILING (1):** `gapper_shadow_scan` (exit 1 — superseded by the harness, safe to unload)
- **UNTRACKED (6):** `forex.scan`, `futures.bias`, `oracle.briefing`, `quant.pulse`,
  `clawd.ny_am_scanner`, `sovereign.papertrading` — **loaded and running with no
  plist committed in `scripts/`.** Unreviewable, and unrecoverable if this machine dies.

---

## 4. Known-broken, not fixed by this work

- ~~**Outcome loop 0/23.** NON-NEGOTIABLE #2 violated in production.~~
  **CORRECTED 2026-07-18 — this claim was wrong and I propagated it.** The alarm
  was a false positive by construction: `pulse_check.py` counted every closed
  OANDA trade as "attempted" on every 2-hourly pulse, including ones matched weeks
  earlier, while `update_outcome` correctly refused to re-close them. The count
  grew 9 → 23 over weeks *while the loop was healthy*. Fixed in `51b6f9a`
  (already-matched sidecar; alarm now fires only on genuine same-run failures).
  Two real defects were hiding behind it: day-boundary match failures (~3 of 23,
  fixed) and `backfill_decision_records.py` unscheduled since 2026-07-01 (plist
  authored, pending install).
- **OANDA 401** at session close — end-of-day position truth missing.
- **Scheduled agent sandbox** — 8 consecutive blocked runs; cannot self-heal.
- **Daily-loss contradiction** — `gates.py` 5% vs `prop_risk_manager.py:49` 2%,
  both live, neither ratified. See `docs/proposed_amendment_art7-9.md`.
- **`CLAUDE.md:134`** carries a commit-message example that reads as an
  instruction to wire the refuted HYP-044 VIX gate.

---

## 5. Next action

**~~Install the sentiment plist.~~ DONE 2026-07-19** — `com.alta.sentiment_update`
is loaded (`plist_manifest`: OK); the board is fresh and feeds the 08:00 scan.
**~~Export the 6 UNTRACKED plists into `scripts/`.~~ DONE 2026-07-19** (commit
`cb6a9bf`) — UNTRACKED count 6 → 0; live jobs are now reviewable.

Remaining, in order (see `OPEN_ITEMS.md` for the full audit):

1. **Install `com.alta.decision_backfill`** — Oracle entry-side backfill has been
   unscheduled since 2026-07-01 (OPEN_ITEMS #1). Highest value now available.
2. `com.alta.system_morning` + `com.alta.system_eod` — same install pattern; the
   six-layer loop is code-only until they load.
3. **Rule on `docs/proposed_amendment_art7-9.md`** — daily-loss cap is 5% (`gates.py`)
   vs 2% (`prop_risk_manager.py:49`), both live, neither ratified. Art. 7 at 2.0%
   recommended, Art. 8 defer, Art. 9 reject.
4. **Triage the 12 genuine `test_ml_stack` regressions** (OPEN_ITEMS #7) — risk layer.
5. **Re-run killed hypotheses under the corrected cost model.** The 11.3× spread
   overcharge (TICK-039) may have buried a real edge in the ~40 NOT_SIGNIFICANT
   verdicts. Highest expected value on the board, no new mining required.

---

*Layer status is not edge status. This file exists so the two can never be
confused on a dashboard.*
