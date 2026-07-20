# OPEN_ITEMS.md — Unfinished-work audit

**Generated 2026-07-19 · branch `sovereign-v2` · audit of last 60 commits, TODO scan, plist drift, stale data, test collection**

Method: git-log markers · `grep TODO/FIXME/NotImplemented` · `scripts/plist_manifest.py` · `find data -mtime +7` · `pytest --co` · `SYSTEM_STATUS.md`.

**Headline:** the codebase itself is disciplined — 1541 tests collect cleanly, isolation holds, research verdicts are closed. Almost every real loose end is in the **operational wiring** (plists authored-but-not-installed, feeds blocked on credentials) or **stale test scaffolding**, not in half-written logic. The three finishes below (marked ✅ FIXED) were completed in this pass.

---

## CRITICAL — blocking or corrupting the live/learning loop

1. **Oracle entry-side backfill unscheduled since 2026-07-01** — `com.alta.decision_backfill` is `NOT_INSTALLED` (plist authored in `scripts/`, never loaded). Without it, unmatched OANDA fills never get entry records → NON-NEGOTIABLE #2 (closed Oracle loop) silently degrades.
   *Fix:* `cp scripts/com.alta.decision_backfill.plist ~/Library/LaunchAgents/ && launchctl load …` then `plist_watchdog.py --rebaseline`.

2. **Daily-loss limit contradiction, both live, neither ratified** — `gates.py` enforces 5%, `prop_risk_manager.py:49` enforces 2%. A live risk cap has two different values depending on code path.
   *Fix:* rule on `docs/proposed_amendment_art7-9.md` (recommends Art.7 = 2.0%), collapse to one value, log rationale in `data/agent/param_change_log.jsonl`.

3. **OANDA 401 at session close** — end-of-day position truth is missing; `execution/eod.py` reconciliation runs blind.
   *Fix:* rotate/repair the OANDA practice token used by the session-close job; verify `alta_account_status`.

4. **Six-layer loop doesn't actually run** — `com.alta.system_morning` and `com.alta.system_eod` are `NOT_INSTALLED`; L1 context + L6 reconciliation exist only as code. The daily flow diagrammed in SYSTEM_STATUS is aspirational until installed.
   *Fix:* `bash scripts/install_agent_plists.sh` (operator), then run the header test command.

5. **Scheduled-agent sandbox — 8 consecutive blocked runs, cannot self-heal.** Autonomous agents can't acquire the permissions they need.
   *Fix:* grant the launchd agent Full Disk / Automation access, or run it outside the sandbox; this blocks all autonomous operation.

---

## HIGH — significant loose ends

6. **6 UNTRACKED plists loaded but not committed** — `forex.scan`, `futures.bias`, `oracle.briefing`, `quant.pulse`, `clawd.ny_am_scanner`, `sovereign.papertrading` run live with no plist in `scripts/`: unreviewable, unrecoverable if the machine dies.
   *Fix:* ✅ **FIXED this pass** — exported all six into `scripts/` (see Finishes).

7. **Real risk-layer test regressions hiding in `test_ml_stack.py`** — Kalman / LQR / Pegasus / TradeMDP tests fail against **modules that still exist** (`sovereign/risk/`), i.e. genuine behavior drift, not deleted APIs. Currently buried in the "40 pre-existing failures" bucket.
   *Fix:* dedicated ticket — diff each failing assertion against the current module; these are the risk layer and deserve real triage. (Spawned as a follow-up task.)

8. **Reddit sentiment feed is dead** — `data/cache/reddit_sentiment.json` shows `posts_scanned: 0` with a fresh timestamp; Reddit now returns **HTTP 403** to unauthenticated `.json` requests (verified live). `context.py` correctly flags it `SILENT_NULL`, but the 08:00 scan gets no Reddit signal.
   *Fix:* add a Reddit OAuth app (client id/secret) to `reddit_scraper.py`, or drop the source and remove it from `SOURCES`.

9. **Only 2 of 7 morning-context sources are FRESH.** `gdelt` UNAVAILABLE (0 rows ever ingested), `calendar` UNAVAILABLE (ForexFactory 403), `sentiment_board` was stale (now loaded — see #12), `reddit` SILENT_NULL (#8).
   *Fix:* per-source — GDELT off-peak backfill; ForexFactory needs a non-403 path or a paid calendar source.

10. **`com.alta.gapper_shadow_scan` FAILING (exit 1)** — superseded by the 16:05 execution harness, still loaded and erroring daily.
    *Fix:* `launchctl unload` it (operator); SYSTEM_STATUS confirms safe to remove.

11. **`gdelt_retry`, `ib_shortable`, `oracle.market_briefing` NOT_INSTALLED** — authored, never loaded; borrow snapshots and market briefing don't refresh on schedule.
    *Fix:* install per the standard cp+load+rebaseline pattern.

---

## MEDIUM — partially done, small push

12. **Dead-API ML-stack tests reported phantom failures** — `TestPCACompressor` (module `pca_compressor` deleted, only `.pyc` remains) and `TestBlackScholes` (`bs_call`/`bs_put`/`bs_digital_call` refactored into `VolRegimeSignal`) test APIs removed with the ML-archive line.
    *Fix:* ✅ **FIXED this pass** — skipped both classes with a documented reason so real regressions (#7) surface.

13. **SYSTEM_STATUS.md drifted from measured reality** — §3 still says `sentiment_board` STALE 313h / never installed and §5's "Next action: install the sentiment plist" is done (`plist_manifest` now reports `com.alta.sentiment_update` OK/loaded); §4's "40 tests all test a deleted API" is imprecise.
    *Fix:* ✅ **FIXED this pass** — reconciled §3/§4/§5 to current `plist_manifest` output.

14. **`sync_dashboard_data.py` imports `run_health_check` from `archive/`** — the 18-component health builder the dashboard needs lives in archived `agent_scheduler.py`; `health.json` API components go stale between manual syncs (`scripts/sync_dashboard_data.py:24-30`).
    *Fix:* relocate `run_health_check` to a live module (e.g. `sovereign/monitoring/`) and drop the `archive/` sys.path insert.

15. **`live_drift_report.py:72` R-multiple is a placeholder** — `compute_r_multiple` returns `None` until fills are matched to close orders by order id.
    *Fix:* join `oanda_fills.json` entries to their closing orders once exit order-ids are logged.

16. **`forex_specialist.py:128` decision-log fields are `None`** — `cot_percentile` (COT engine returns z-score, not percentile), `library_match`, `commitment_score`, `kelly_fraction` all logged as `None`; Oracle learns on partial entry context.
    *Fix:* wire the percentile conversion + the three missing fields. **Note: this is on the frozen live decision path — requires an explicit unlock in `NEXT.md` and a `param_change_log` entry before touching.**

---

## LOW — cleanup / nice-to-have

17. **`clawd_trading/swing_prediction/` is orphaned** — not imported by any live module; contains a 39 KB `TODO_COMPLETE_IMPLEMENTATION.py` and `swing_engine.py:321 # TODO: Implement actual data fetching`. Dead swing-prediction layer.
    *Fix:* move to `attic/` or delete; it's not on any live path.

18. **`data/sentiment.db.wal` was git-tracked** — a SQLite WAL file should never be committed (shows as deleted in the tree).
    *Fix:* add `*.wal` / `*.db-wal` to `.gitignore`.

19. **~716 `data/**/*.json(l)` files older than 7 days** — the vast majority are **completed research artifacts** (permutation tests, sweeps, ledgers) and are expected to be static. Genuinely-stale *feeds*: `data/calendar_snapshot.json` (Jun 30, FF 403), `data/cross_system_state.json`, `data/petroulas_log.json`.
    *Fix:* none needed for research outputs; the three feeds tie to #9.

20. **`attic/ai_trading_bridge.py`, `attic/apply_fixes.py`** carry `# TODO` stubs (`Build features`, `Run model.predict_proba`, manual reconciliation notes).
    *Fix:* attic by definition — ignore or purge.

21. **pytest emits a `service_identity` ModuleNotFoundError warning** at collection (Twisted TLS).
    *Fix:* `pip install service_identity` to silence; cosmetic.

22. **~40 `scripts/*.py` have `__main__` but no plist/AGENT_DIRECTIVE reference** — but these are almost all **intentional manual CLI/research tools** (`alta.py`, `discover.py`, `build_obsidian_graph.py`, etc.), not orphaned automation. Not a defect; listed for completeness.

---

## What was NOT flagged (and why)

- **Research "dead ends"** (HYP-089/090/091/092/099/101/102/104/105/106, megascans) — these are **closed verdicts**, not unfinished work. NOT_SIGNIFICANT is a finished result.
- **`discovery/data_adapter.py` / `vrp/data_loader.py` NotImplementedError** — intentional scaffolding stubs caught by callers (NQ path), not gaps.
- **Frozen execution path** (`forex_exit_manager`, `decide_exit`) — untouched by design per the shadow/execution freeze.
