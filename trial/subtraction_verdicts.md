# The Subtraction Trial — 2026-07-02

Every top-level module/script/config stands trial on three questions:
**(1)** what decision does it improve — name the decision; **(2)** when did its output
last change an action — cite evidence (imports, launchd schedules, logs, journal, git);
**(3)** what breaks tomorrow if deleted. Evidence sources: AST import sweep (the
Obsidian graph generator's walker), the launchd plist→script map (18 scheduled jobs),
git last-commit dates, and the new decision journal. Execution path does not move.

## Verdict classes
**ALIVE** (answers all three) · **ATTIC** (fails all three, zero live imports, zero
schedules — moved in commit `e221827`, reversible) · **AMBIGUOUS** (Colin rules) ·
**KNOWN-FAILURE** (tests only; documented, untouched).

---

## S1 · Atticked (34 files, commit `e221827` — one `git revert` restores all)

| Group | Files | Evidence |
|---|---|---|
| Equity-engine era runners | run_engine{,_v2_real_data,_production,_live}.py, run_alpaca_production.py, run_clawd.py, run_kimi.py, run_phases_9_11.py | imports=0, schedules=0, last commits Mar 30 – May 13 |
| Plug-in "brains" | xgboost_brain.py, clawd_brain.py, kimi_brain.py, ai_trading_bridge.py, example_plug_in_model.py | imports=0; the live ML path is sovereign/ml_trainer + layer1 |
| Dead dashboard plumbing | dashboard_api.py, dashboard_api_client.py, realtime_publisher.py, simple_publisher.py | imports=0; live dashboards are repo-root index.html + ict/index.html served by live_signals_server (2026-06-30 audit) |
| One-off ops scripts | execute_monday_killzone.py, setup_task_scheduler.py, verify_data_flow.py, apply_fixes.py | imports=0, schedules=0, Apr 2026 |
| Root scratch tests ×10 | test_alpaca_*, test_kimi_*, test_chi2_gate, test_expanded_universe, test_feature_match, test_key_debug, test_training | outside tests/ (pytest.ini testpaths=tests → never collected), imports=0 |
| Stale claim docs ×3 | PHASE_13_COMPLETE.md, GITHUB_FIREBASE_ENHANCEMENTS.md, JGBT_COMPLETE_100_PERCENT.md | 2026-03-13; superseded by the honest proof-engine record |
| `${HOME}` literal dir | attic/HOME_literal_dir (untracked scaffold junk from a mis-expanded script) | contained only an empty .claude skeleton |

## S2 · Ambiguous — your ruling (recommendation + evidence each)

| Item | Evidence | Recommendation |
|---|---|---|
| `execute_daily.py` | imports=0 BUT it IS `com.sovereign.papertrading` (loaded launchd job, runs daily `--mode paper`) — the un-approved paper trader from the 07-01 LaunchAgent audit | **Kill the job, then attic** — it trades the legacy equity engine nobody audits; ties to your standing keep/kill |
| `train_core.py` | imports=0, but the layer1 STEP-0 tautology fix (commit 4ba99a4) put `forward_dir_5d` computation here; layer1 retraining would call it manually | **Keep until Track-D supersedes layer1**, then re-try |
| root `config.py` | imported ×3 (legacy equity engine modules) | Keep while its importers live; re-sentence with them |
| `universe_sweep.py`, `walk_forward_validation.py` | imported ×2 each | Keep (research utilities with consumers) |
| `ict-engine/` | imported ONLY by manual `scripts/run_oanda_paper.py`; NOT the scheduled live path (that's `ict/`); CLAUDE.md NN#1 still names it the bridge | **Decide the bridge story**: either re-bless (fix NN#1 wording) or retire run_oanda_paper + attic the package |
| `firebase/` | push_to_firebase feeds a dead page (2026-06-30 dashboard audit); partial cleanup already in 3d88669 | Attic after confirming no n8n/TABOOST coupling |
| `layer2/`, `layer3/` | game-theory layer de-wired to logger (Stripped Core note); imports only within legacy orchestrator chain | Attic with the equity-engine ruling as one package |
| `dashboard/`, `ict-dashboard/` | orphaned per the 06-30 dashboard-architecture audit (live pages are root index.html + ict/index.html) | Attic both |
| `clawd_trading/`, `entry_engine/`, `imbalance_engine/`, `meta_evaluator/`, `orchestrator/`, `trading_strategies/`, `training/`, `models/`, `contracts/`, `integration/`, `examples/`, `governance/`, `templates/`, `vendor/`, `quant/`, `scratch/`, `backtest/`, `research/`, `lab/` | mixed: some feed the legacy `sovereign.orchestrator` "Stripped Core" (still importable), `lab/feature_registry` is named by TRADING_PHILOSOPHY | **Batch ruling with the equity-engine decision** — if Stripped Core is retired, ~12 of these fall together; lab/ stays (philosophy names it) |

## S3 · The 40 pre-existing test failures — verdicts (documentation only; zero test edits)

| File (n) | Module under test | Real cause | Verdict |
|---|---|---|---|
| tests/unit/test_ml_stack.py (24) | sovereign/risk/{black_scholes, ica, kalman_regime, trade_mdp, lqr, pegasus} + pca_compressor | **API drift** — tests written against renamed/removed symbols; pca_compressor module doesn't exist; sklearn/scipy ARE installed (1.8.0/1.17.1). The six modules each have 1 live import | **KEEP-AS-KNOWN-FAILURE** (2026-07-02). Rewrite-vs-current-API is its own reviewed task. **Track-D assessment: not a factory input** — the factory uses sklearn/xgboost directly; B76 math already lives in vrp_feed |
| tests/unit/test_ict_session_classifier.py (10) | ict/session_classifier | timezone/kill-zone boundary drift on this machine | KEEP-AS-KNOWN-FAILURE (2026-07-02) |
| tests/unit/test_ict_pipeline.py (4) | ict scoring/veto | behavior drift (HYP-046 veto strings, zeroed component weights) — tests predate the deployed lessons | KEEP-AS-KNOWN-FAILURE (2026-07-02); tests encode a superseded spec |
| tests/unit/test_forex_{data_fetcher(2), entry_engine(1), macro_engine(1), signal_engine(1)}.py, test_data_pipeline.py (1) | live-fetch paths | network/live-data shape dependence (fallback flags, <3 pairs returned, hold-arr from live signals) | KEEP-AS-KNOWN-FAILURE (2026-07-02); candidates for a `network` pytest marker in their own task |

Baseline stays **40** — the phase-2 suites (positioning 12, experience 13, factory 9)
add zero failures.

## S4 · Execution path
Did not move. `forex_exit_manager`, `decide_exit`, backtesters, bridges, gates: ALIVE
by definition and by schedule evidence (launchd + shadow audit reports).
