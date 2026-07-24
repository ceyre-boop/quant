# NEXT — Alta quant session log (authoritative, repo-native)

Per-session ledger: what shipped, push status, verdicts, blockers, refusals. Newest first.
The Obsidian brain (`~/Obsidian/Obsidian/00-BRAIN/NEXT.md`) is the cross-project rollup.
Standing constraints live in `CLAUDE.md` — not restated here.

### 2026-07-24 · Self-play training board BUILT — ignition GATED CLOSED

Built the AlphaZero-style self-play policy training loop from Colin's spec
(`research/SELF_PLAY_TRAINING_ARCHITECTURE.md`, 2026-07-24). **The board is built;
it is NOT ignited.** Training a trading policy is model training, gated by
RISK_CONSTITUTION Art. 6 behind a CONFIRMED ledger verdict — enforced in code, not
by convention.

**Shipped (all freeze-safe, new paths; one 4-line non-frozen edit):**
- `scripts/sovereign_train.py` — runner, `--watch`, phases 0-5, prints gate loudly.
- `sovereign/training/gate.py` — ignition gate (mirrors research_factory + autonomous.yml::live).
- `sovereign/training/value_scorer.py` — HYP-071 board READ-ONLY; `trade_score()` net-guarded.
- `sovereign/training/policy_rollout.py` — Phase 1 sim trades (DRY placeholder while gated).
- `sovereign/training/policy_updater.py` — Phase 3 sample-weight reweight (refuses refit while gated).
- `sovereign/training/director.py` — Phase 4 MECHANISM / REGIME / MAGNITUDE(±20%) checks; HUMAN-gated, never auto-approves.
- `config/training.yml` — hyperparams + ignition flags (default OFF).
- `logs/training_log.jsonl` — append-only, one entry/cycle. `data/training/` gitignored.
- `sovereign/ml_trainer.py` — added `value_weights` sample_weight passthrough (~6 lines, preserves purged split).
- `tests/test_sovereign_training_gate.py` — 12 tests, all green.

**Gate/guard proven (DRY `--watch` run pasted in session):** gate held CLOSED on all
three conditions [tick_024=FAIL, hyp_071=FAIL, value_board_is_net=FAIL]; net-return
guard FIRED in Phase 2 (board carries `gross_R_caveat`); Phase 3 wrote NO policy;
verdict `SCAFFOLD_DRY`, committed=false. Isolation test
`test_pipeline_does_not_import_sovereign` still green; `sovereign/training/` imports no `ict/`.

**Ignition is GATED CLOSED** pending BOTH: (1) TICK-024 carry-cost fix landed, and
(2) HYP-071 recomputed on NET returns with a CONFIRMED ledger stamp. The runner is
physically incapable of a real cycle until both config flags are flipped (logged
rationale per CLAUDE.md #4) AND the value board sheds its gross marker.

**July-28 ignition checklist (spec §8.3):**
- Jul 26-27: TICK-024 fix merged (net carry costs corrected) → flip `ignition.tick_024_carry_fix_landed`.
- Jul 28: HYP-071 net recompute runs (same harness, corrected costs).
- Jul 28: Colin adjudicates HYP-071 net → ledger stamp or data-ceiling. If CONFIRMED, board loses `gross_R_caveat` → flip `ignition.hyp_071_net_confirmed`.
- Jul 28: with the gate open, `sovereign_train.py` receives its reward signal.
- Jul 29 (after FOMC): first real training cycle in the FOMC-quiet afternoon.

### 2026-07-23 · Briefing synthesizer wired (AlphaZero L1 → dashboard + hypothesis batch, CONTEXT ONLY)

Colin-approved concrete fix from the two-half inventory: wire `sovereign/briefing/synthesize.py`
(the L1 directional read) to consumers. Diagnosis first, per discipline.

**`provenance.verified=false` root cause — NOT a data-integrity failure.** It is a deliberate,
hardcoded epistemic label (`scripts/morning_market_briefing.py:191` + `sovereign/briefing/__init__.py`):
"qualitative regime context + a self-scored directional call, NOT a verified data feed or validated
edge... never ingested as trading inputs." Stays false BY DESIGN until `scorecard.py` proves the call
is calibrated over a real sample. No no-lookahead guard tripped. This is correct discipline, not a bug.

**Correction to the survey framing:** the synthesizer was NOT wired to nothing — Oracle already reads
it (`reflect_cycle._load_market_briefing`, lines 445/554). The gap was Colin's two named consumers.
Also: today's output is `synthesis_source=deterministic_fallback`, bias NEUTRAL/conf 0 — the Opus call
fell back (no API key this run); wiring surfaces whatever it produces, honestly labeled.

**Shipped (append/additive, no rebuild):**
- `index.html` — NEW "🧭 L1 DIRECTIONAL READ" panel; fetches `data/oracle/market_briefings/latest.json`
  (mirrors the proven daily_digest fetch pattern), renders bias/confidence/regime/invalidation/source
  with a prominent "CONTEXT ONLY — not a trade signal" badge driven off `provenance.verified`.
- `sovereign/autonomous/hypothesis_generator.py` — NEW `_load_briefing_context()`; `run()` stamps each
  candidate + the generator-log batch + return dict with `l1_briefing_context` (role=context_only).
  Verified via dry-run: `L1 context: bias=NEUTRAL conf=0 regime=BREADTH verified=False (context-only)`,
  86 ledger entries loaded.

**Compliance:** ICT/sovereign isolation test passes; briefing is ES/NQ data-only (no OANDA/MT5, no
order_send); L2 shadow exit machine untouched (freeze/July-28 go-date respected); L1 output is context
ONLY — never gates/sizes/auto-promotes; HYP-071 NNUE value table NOT built (stays gated, unapproved);
append-only writes. Dashboard panel lives on sovereign-v2; live deploy serves from master (operator
worktree-push data+panel to master to surface on the Render/Pages dashboard).
**NOTE:** live browser render of the panel deferred — safety classifier outage mid-session blocked the
served-render check; panel verified structurally (element IDs ↔ JSON fields, proven fetch pattern).



Work order specced a from-scratch DIP build; survey said most exists. Followed diagnose-first
discipline — did NOT rebuild. **Diagnosis:** `data/harvest.db` frozen since 2026-06-29, `trades`
table empty, `_price_cache` empty. Root cause: `continuous_harvester` forks a `Pool` whose workers
call `yf.download` *inside child processes* → macOS fork-after-network deadlock. Proof: isolated
`yf.download('AAPL')` = 1.0s; harvester `--passes 1 --symbols AAPL NVDA` hung >10min, wrote nothing;
parent-prefetch + sequential compute = 65,734 trades/17s. Not a broken dep, not a rebuild.

**Shipped (commit `28640b5`, pushed to sovereign-v2):**
- NEW `scripts/dip_data_fetcher.py` — centralized PARENT-process yfinance pull that warms the
  parquet cache (reuses `continuous_harvester.load_ohlcv`; zero duplication). The one genuinely new
  file the survey identified.
- Wired harvester to prefetch in parent before the fork pool → workers only READ cached parquet.
- Fixed pre-existing bug in `training/retrain_loop.load_trades`: SELECTed 3 lagged features that are
  derived in pandas post-query and don't exist in `trades` (BinderException). Added `BASE_FEATURE_COLS`.
- NEW `scripts/dip_daily.sh` orchestrator (harvest 1 pass → retrain once, checkpoint/error files) +
  `scripts/com.alta.dip_daily.plist` (tracked-not-loaded, daily 02:30; operator promotes with launchctl).

**Verified E2E:** harvester fork pool no longer hangs — 195,688 trades (3 syms, 18s) → `retrain_loop
--once` trains XGBoost (val acc 92.3%, model saved) → threshold upgraded 0.50→0.70.

**Did NOT rebuild:** harvester compute, XGBoost trainer, sentiment, Oracle reflect, Obsidian sync —
all pre-existing and working. **Compliance:** ICT/sovereign isolation test passes; new scripts import
no sovereign; research loop only (no order_send / MT5 bridge); no frozen execution-path files touched;
training ran against accumulated harvest under CONFIRMED carry_v015/HYP-093.
**Operator TODO:** promote plist — `launchctl load ~/Library/LaunchAgents/com.alta.dip_daily.plist`
(symlink/copy from `scripts/com.alta.dip_daily.plist` first).

### 2026-07-22 · TICK-056 MT5 execution bridge — SPEC + TICKET + PLAN shipped (spec-first, STOPPED before code)
Prerequisite for Step 3 (The5%ers $100K High Stakes), forcing date FOMC 2026-07-29. Per plan→build
separation I produced the spec/ticket/plan and STOPPED before connector code — this needs a platform
decision from Colin first (below).

**Shipped (commit `31e71bc`, pushed to sovereign-v2):**
- `specs/mt5_bridge.md` (LAW) — order-routing `order_intent` JSON contract, demo-vs-live guard,
  approval flow, failure modes, file layout, test strategy. NEW-infra design: the bridge consumes a
  decoupled JSON contract and imports NOTHING from the frozen execution path.
- `tickets/backlog.md` → **TICK-056** (ready, pre_approved:false) with acceptance criteria.
- `plans/TICK-056.md` (force-added; plans/ gitignored) — build sequence, risks, the Open Decision.

**Freeze compliance:** zero edits to `forex_exit_manager` / `decide_exit` / `execution/harness.py` /
`carry_engine` / `ict/pipeline.py`. No unlock consumed. Wiring a producer to EMIT `order_intent` and
any live route are separate future unlocks.

**BLOCKER — Colin's decision needed before connector code:** `MetaTrader5` pip package is Windows-only
and is NOT importable on this Mac (Darwin — verified in-sandbox). The native bridge needs one of:
(A) Windows box/VM, (B) Wine-prefix Python + Wine MT5 on the Mac, (C) socket-EA bridge. The guard /
contract / idempotency / approval logic are identical across all three and can be built + unit-tested
now with a MockConnector; only the connector layer + live-terminal validation wait on this answer.

**Refused to shortcut:** did not scaffold an unvalidatable real connector or fake a fill (no MT5 pkg,
wrong OS) — surfaced the blocker instead. Did not proceed past the plan (not pre_approved).


Built the free-data daily scanner per `tickets/DISPATCH_PETRULES_GATE_SCANNER.md`. Rule-based
ONLY (no ML — that's a later ticket). Scores ~2,550 instruments (S&P500 + Russell2000 + 50 ETFs
+ 4 carry pairs) each morning on 4 factors and writes one JSON the dashboard already reads.

**Shipped (all standalone research — NO sovereign/ or ict/ imports; verified via grep):**
- **`config/gate_params.yml` (NEW)** — every factor weight (0.30/0.25/0.25/0.20), tier cutoff
  (4≥0.85, 3≥0.70, 2≥0.50), per-factor scoring band, sizing params, EDGAR settings, and the
  full universe (ETF list + carry pairs). No threshold is hardcoded in code.
- **`scripts/gate_universe.py`** — S&P500 (Wikipedia), Russell2000 (iShares IWM CSV), ETFs +
  carry from config → `data/agent/gate_universe.json`; weekly staleness refresh; raises (no
  fabricated universe) on source failure.
- **`scripts/gate_edgar_client.py`** — SEC EDGAR Form 4 + SC 13D/13G fetcher. Real UA
  ("Alta Research colineyre222@gmail.com"), ≥1s spacing, 429 exponential backoff, raises
  `EdgarUnavailable` (→ error JSON) rather than faking data.
- **`scripts/gate_options_screen.py`** — yfinance vol/OI flow approximation + analyst
  revision-velocity extraction. Missing data → None (scorer treats as 0.0, never fabricated).
- **`scripts/gate_scorer.py`** — pure rule-based rubric: 4 factor scorers, weighted conviction,
  tier assignment, and an INFORMATIONAL sizing block (quarter-Kelly, hard f_max=0.08) that
  always carries `calibration_status: "BACKTEST ONLY — uncalibrated, not a trade directive"`.
- **`scripts/petrules_gate_scanner.py`** — orchestrator. Writes `petrules_gate_scan.json`
  (dashboard schema), appends `gate_scan_history.jsonl` (tier 2+) and `gate_calibration.jsonl`
  (tier 3+ stubs, outcome=None, append-only). On any source failure writes the honest
  `{"error","scanned_at"}` JSON and exits non-zero. `--self-test` runs a fully offline
  fixture scan. **It writes a JSON file and nothing else — no execution/sizing wiring.**
- **`scripts/com.alta.petrules_gate.plist`** — 09:00 ET Mon–Fri, TZ=America/New_York. NOT
  installed. Colin installs with:
  `cp scripts/com.alta.petrules_gate.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.alta.petrules_gate.plist`
- **`tests/test_petrules_gate_scanner.py`** — 8 offline tests, all pass.

**Verified structurally in-sandbox (network firewalled — no live scan possible here):**
- Self-test + pytest (8/8) green: tiering (NVDA→T4, AAPL→T3), weights sum to 1.0, selling
  penalty, downgrade cap, missing-data→0.0, sizing honesty (f≤f_max, BACKTEST ONLY),
  Tier-1 noise dropped from surfaced list.
- ERROR-PATH proven: simulated `EdgarUnavailable` → honest `{"error","scanned_at"}` JSON, no crash.
- Calibration append proven: tier-3+ signal → one appended stub line, append-only, outcome=None.
- No forbidden imports (grep clean). Dashboard already wired to read the file (index.html:919).

**Needs Colin's Mac for first live run** (sandbox returns HTTP 403 for yfinance/EDGAR): the
first real `python3 scripts/petrules_gate_scanner.py` — builds the real universe, hits EDGAR +
Yahoo, writes the first live scan. Everything upstream of the network is verified.

**Push status:** committed locally in logical steps; push may fail on sandbox keychain — see
commit log. Calibration/scan/history data files left empty (fixture artifacts truncated), scan
output left as an honest "not yet run live" error placeholder until the first live scan.

### 2026-07-22 · Layer-8 ICT causal-chain journal — the update_outcome() loop CLOSED (ICT non-frozen)
Closes CLAUDE.md NON-NEGOTIABLE #2 for the ICT path: every evaluated ICT setup is now
journaled with its full causal chain, and every ICT paper close fires update_outcome().
The conscience's process-adherence + forecast-vs-execution sections moved from
"UNAVAILABLE — ledger not written yet" to reading the real ledger.

**Shipped (all ICT non-frozen; the frozen forex/carry path untouched; scoring weights
market_structure/pd_alignment stay 0.0 — HYP-024/034):**
- **`ict/causal_journal.py` (NEW)** — stdlib-only writer (NO sovereign import; reads the
  neutral regime/bias JSON as data only). Appends one record per evaluated setup to
  `data/agent/ict_causal_chain.jsonl` with the full schema (setup_id, symbol, timestamp,
  level_id/type/tf/quality, regime_state, bias_state, bias_aligned, size_multiplier,
  ict_grade, component_scores, r_r_computed, action, discard_reason, outcome, outcome_r).
  `make_setup_id()` is deterministic (symbol/dir/entry/date) so eval-site and close-site
  agree without threading state. `update_outcome()` back-fills the OPEN ENTERED record and
  FAILS LOUD (warns + returns False, fabricates nothing) when nothing matches.
- **`ict/pipeline.py`** — `evaluate()` is now a thin wrapper over `_evaluate_impl()` that
  journals WHATEVER the pipeline returns — entries AND rejects — classifying action as
  ENTERED / DISCARDED / VETOED / BELOW_MIN_RR / NO_OPPOSING_LEVEL. This is what makes
  "was it the level or the bias?" answerable. Journaling never alters the decision and
  never raises. Also stamps the applied regime `size_multiplier` onto the signal.
- **`ict/paper_trader.py`** — `_log_trade()` (the single close choke point for stop / TP2 /
  BE / session-end) now reconstructs the setup_id and calls
  `causal_journal.update_outcome(setup_id, outcome, outcome_r)`. Every paper close closes
  the loop.
- **`alta_platform/measurement.py`** — added `process_adherence_from_journal()` +
  `forecast_vs_execution_from_journal()`; `build_system_health.py` wires them for
  `ict_equities`. Real counts where data allows (by_action, n_entered, n_closed), honest
  INSUFFICIENT_DATA below `platform.health.process_adherence.min_sample` (20, logged to
  param_change_log), UNAVAILABLE at 0 records. execution_quality stays UNAVAILABLE
  (no per-fill cost in the journal — not faked from the read). undertow_gapper + carry
  keep UNAVAILABLE (the ICT ledger is ICT-specific; their own process ledgers don't exist).

**Verified (scratch run):** ENTERED/VETOED/DISCARDED records all wrote; `update_outcome`
filled a sample ENTERED line (outcome STOP, outcome_r -1.0, r_r_computed 2.0); unknown
setup_id returned False with the fail-loud warning. Synthetic records were then TRUNCATED —
the committed journal is EMPTY (no fabricated trades). Live today the ICT regime gate is
STAND_ASIDE (ict_equities stub), so real ENTERED records won't appear until that section
populates; the wiring is the deliverable.

**Tests:** `test_pipeline_does_not_import_sovereign` **PASS**; `test_alta_platform_isolation`
**2 passed**; `tests/unit/test_ict_pipeline.py` **4 failed / 17 passed** — the 4 are the
documented pre-existing failures (TestScoreAndGrade ×2, TestRiskEngineGate ×2) that fire at
the HYP046 disp-gate BEFORE the new wrapper/journal is reached. No new failures.
`ict/causal_journal.py` AST-checked: zero sovereign imports.

**update_outcome status: WIRED (not PENDING)** — the ICT paper close path exists and is
hooked. It will fire the moment a real paper trade closes; today none open (regime STAND_ASIDE).

**Push status: commit made locally; push may fail on host keychain** (sandbox has no git
creds). Explicit-path commit (never `-A`). If a stale `.git/index.lock` blocks it, on host:
`rm -f .git/index.lock` then `git push`.

---

### 2026-07-22 · The conscience (measurement layer) — BUILT + RUNNING
Second organ of the nervous system, per `specs/measurement_layer.md`. Neutral `alta_platform/`
module — imports neither `ict/` nor `sovereign/` (reads their journals/ledgers as data). Shipped:
- `alta_platform/measurement.py` — EdgeHealth/StrategyHealth models, portfolio integrity + kill
  switch, undertow edge-health, honest-UNAVAILABLE process-adherence + forecast-vs-execution.
- `alta_platform/health_client.py` — `get_health(strategy) -> HealthRead` (`.kill_switch`,
  `.edge_divergence`, `.stale`); safe-by-default HALT.
- `scripts/build_system_health.py` (writer) + `scripts/com.alta.system_health_verdict.plist`
  (30-min schedule, NOT installed).
- Config `platform.health` thresholds logged to `param_change_log.jsonl` (NN#4). NO risk caps
  duplicated — drawdown ladder read from RISK_CONSTITUTION.md Art.3.
- ICT wired to read the conscience at its Gate-0 sizing gate (Gate 0b), alongside the regime read.

LIVE VERDICT (`data/agent/system_health_verdict.json`, status DEGRADED — honest):
- **undertow_gapper: edge_health = INSUFFICIENT_DATA (n_live=3 shadow signals < n_needed=250),
  kill_switch = REDUCE.** Backtest event-mean expectancy 0.01596 (HYP-093) shown for reference;
  live expectancy NOT estimated from 3 signals. This honesty is the deliverable.
- ict_equities: edge_health UNAVAILABLE (no live edge ledger), kill_switch REDUCE.
- carry: UNAVAILABLE (execution path frozen until 2026-07-28 — not wired).
- portfolio: data_integrity OK; drawdown-breaker feed UNAVAILABLE → REDUCE (no unified
  position/P&L ledger; cannot confirm account safety → never full TRADE).

Honestly UNAVAILABLE + why: process-adherence + forecast-vs-execution across all strategies need
the ICT causal-chain setup ledger (`data/agent/ict_causal_chain.jsonl`, Layer 8) which is not
written yet — reported UNAVAILABLE with the exact source needed, no fabricated scores.

TESTS: `tests/test_alta_platform_isolation.py` green both directions (extended to import
`health_client` + `measurement`); `test_pipeline_does_not_import_sovereign` green after ICT wiring.

BLOCKER (operator hand): git commit + push could NOT run — a stale `.git/index.lock` (crashed
process from 16:10, owned by host) is unremovable from the sandbox ("Operation not permitted").
All changes are on disk, UNCOMMITTED. Colin: on the host run
`cd ~/quant && rm -f .git/index.lock`, then commit (suggested split: (1) config +
param_change_log + alta_platform measurement/health_client/__init__ + isolation test;
(2) scripts/build_system_health.py + plist + data/agent/system_health_verdict.json;
(3) ict/pipeline.py conscience gate + NEXT.md), then `git push`.

TO INSTALL the schedule (do once, on host):
`cp scripts/com.alta.system_health_verdict.plist ~/Library/LaunchAgents/ && launchctl load
~/Library/LaunchAgents/com.alta.system_health_verdict.plist`

---

## 2026-07-22 — [PLATFORM]/[ICT] Rename platform→alta_platform; wire ICT regime gate (shipped, push BLOCKED)

Three follow-on tasks on top of the regime contract: stop the stdlib shadow, wire the first
reader into a live (non-frozen) strategy, and scope the portfolio ledger.

**Shipped:**
- **[PLATFORM] `platform/` → `alta_platform/`** (commit `6f67d27`). The old name shadowed
  Python's stdlib `platform` (breaks `uuid`, therefore `pytest`); the fragile superset guard
  in `__init__.py` is **removed** — verified nothing depended on it (only imports were
  `regime_client`/`regime_contract`). Updated every import: `scripts/build_system_regime.py`,
  `scratch/proof_regime_client.py`, and the isolation test (renamed →
  `tests/test_alta_platform_isolation.py`). Legit stdlib `import platform` usages
  (`research/*/_lib.py`, `backtest/sweep.py`, `scripts/bench_throughput.py`) left untouched.
  Plist calls the script by path (no module name) — no edit needed.
- **[ICT] Regime gate at the ICT sizing boundary** (commit `cf56f64`, `ict/pipeline.py` only,
  +42 lines). Added `from alta_platform.regime_client import get_regime` and a Gate-0 read of
  `get_regime("ict_equities")` just before `risk_engine.size()`: STAND_ASIDE or stale →
  ICTVeto (skip) with a logged reason; otherwise `size *= size_multiplier`. Did **not** touch
  scoring weights or the frozen forex/carry path. `ict_equities` is an UNAVAILABLE stub today,
  so ICT correctly stands aside until that section is populated — the wiring is the deliverable.
- **Task 3 scope** — `research/position_ledger_scope.md`. Honest verdict: **partial**. Raw
  open positions exist (ICT `data/ledger/ict_paper_trades.json` `open[]`; forex reconstructable
  from `oanda_fills.json`; live NAV in `equity_curve_live.jsonl`), but **no cluster taxonomy**
  (USD_MACRO/YEN/EQUITY_SMALLCAP) and **no cluster caps** exist on disk — so a by-cluster view
  cannot be assembled without authoring those. Scoped, not built. This is what moves the
  portfolio section `UNAVAILABLE → OK` later.

**Verdicts / tests:**
- `tests/test_alta_platform_isolation.py`: **2 passed** (both directions).
- `test_pipeline_does_not_import_sovereign`: **still green** — the isolation allowlist forbids
  only sovereign/layer imports; `alta_platform` (imports neither side) is legal from `ict/`.
- `tests/unit/test_ict_pipeline.py`: **4 failed / 17 passed** — the 4 are the pre-existing,
  documented failures (TestScoreAndGrade ×2, TestRiskEngineGate ×2), and they fail at the
  disp-gate/scoring stage *before* the new regime gate is reached (reasons: `HYP046_DISP_GATE`,
  missing `_weights` keys). No new failures from this change.
- Stdlib shadow gone: `python3 -c "import platform; print(platform.system())"` → `Linux`.
- `scripts/build_system_regime.py` still writes `system_regime_state.json` with **carry =
  STAND_ASIDE** (narrowing on all 4 pairs).

**Push status: BLOCKED** — sandbox has no git credentials (`could not read Username for
github.com`). Both commits are local on `sovereign-v2` (HEAD `cf56f64`); Colin needs to
`git push`. NB: the sandbox mount blocks `unlink` inside `.git`, so a stale `index.lock` /
`HEAD.lock` may be present — remove them on the host before pushing
(`rm -f .git/index.lock .git/HEAD.lock`). The Task-2 commit was made via plumbing
(`commit-tree` + direct ref write) to work around that lock; it is a normal commit.

**Carry read-wiring still waits for the post-2026-07-28 unlock** — carry is frozen; its
one-line `size *= get_regime("carry").size_multiplier` is deferred until an explicit NEXT.md
unlock, per the execution-path freeze. Only ICT (non-frozen) was wired this session.

---

## 2026-07-22 — [PLATFORM] System regime contract: the neutral nervous system (shipped, push BLOCKED)

Built the shared connective layer from `specs/system_regime_contract.md` — a neutral
`platform/` package that unifies the per-strategy regime signals already on disk into one
canonical contract `data/agent/system_regime_state.json`, plus a tiny reader any strategy
calls. Isolation-safe by construction: communicates by data contract (reads/writes JSON),
never by cross-module import.

**Shipped:**
- `platform/__init__.py`, `platform/regime_client.py` (`get_regime(strategy) -> RegimeRead`
  with `.verdict/.favorable/.size_multiplier/.stale`; safe-by-default — stale/missing =>
  STAND_ASIDE, size 0.0), `platform/regime_contract.py` (Section model + carry/es_nq/macro
  classifiers).
- `scripts/build_system_regime.py` — the writer. Never raises. Sections from files that
  exist today: carry ← `forex_proximity.json`, es_nq ← `nqes_regime.json` (INFO, never
  sizes), macro ← `macro_snapshot.json` (INFO). Portfolio = honest UNAVAILABLE stub.
- `scripts/com.alta.system_regime.plist` — 30-min schedule, NOT installed (launchd
  unreachable). Load: `cp scripts/com.alta.system_regime.plist ~/Library/LaunchAgents/ &&
  launchctl load ~/Library/LaunchAgents/com.alta.system_regime.plist`.
- `tests/test_platform_isolation.py` — asserts `platform/` imports neither `ict` nor
  `sovereign` (static AST + runtime sys.modules), both directions.
- `config/parameters.yml` `platform.regime` block (freshness_hours + contract_max_age_hours);
  new keys + the carry 6→12h bump logged to `data/agent/param_change_log.jsonl` (NN#4).

**Verdicts / tests:**
- `tests/test_platform_isolation.py`: **2 passed**. `test_pipeline_does_not_import_sovereign`:
  **still green**. (Repo-wide collection has ~27 pre-existing errors from missing deps —
  pydantic/yfinance/sklearn — NOT from this change.)
- Live truth reproduced: **carry = STAND_ASIDE**, status OK, reason "rate differentials
  NARROWING on all 4 pairs; none widening" — verified via `get_regime("carry")` from a
  scratch script (`scratch/proof_regime_client.py`).

**Note — stdlib shadow:** `platform` is also a stdlib module. `platform/__init__.py` absorbs
the real stdlib `platform` attributes so `platform.system()` (uuid/pytest) keeps working.
Do not remove that guard.

**Follow-on (not this task):** `ict_equities` + `undertow_gapper` sections need PriceStore /
gapper inputs; portfolio needs a unified position-by-cluster ledger (none on disk today).
Strategy READ-wiring is next — carry consumption needs a post-freeze unlock recorded here
before `size *= get_regime("carry").size_multiplier` touches the frozen sizing boundary.

**BLOCKER — git push:** the sandbox has no GitHub credentials (HTTPS remote, no username)
AND stale `.git/*.lock` files (index.lock, HEAD.lock, sovereign-v2.lock) that the restricted
mount won't let me delete. Commit `20696b7` (step 1) landed on the branch; step-2+ changes are
in the WORKING TREE, uncommitted. Operator recovery on host:
`rm -f .git/index.lock .git/HEAD.lock .git/refs/heads/sovereign-v2.lock` then
`git add platform tests/test_platform_isolation.py scripts/build_system_regime.py
scripts/com.alta.system_regime.plist scratch/proof_regime_client.py config/parameters.yml
data/agent/param_change_log.jsonl data/agent/system_regime_state.json NEXT.md &&
git commit -m "[PLATFORM] regime writer + plist + proof + NEXT" && git push`.

---

## 2026-07-20 — LAUNCHD TRIAGE: THERE IS NO `claude` CLI ON THIS MACHINE; TWO CASCADE CLAIMS REFUTED

Dispatched to repoint the agent plists at the real `claude` binary, reload, verify, unload
the retired Reddit agent, and adjudicate the cascade. **The premise did not survive contact.**

**The blocker: the binary does not exist anywhere.** Not just at the path launchd names —
anywhere. Enumerated on the host: `/usr/local/bin/*` returns *nothing at all*;
`/Users/taboost/.local/bin/` holds only `mac-tune`/`dispatch`/`cc`/`route`;
`/opt/homebrew/bin` has no `claude`; npm global has only `npm`; `~/.claude/local/` empty;
no nvm/bun/volta copy. `/Applications/Claude.app` is the desktop app, not the CLI.
**The fix is an install, not a path edit.** No path was fabricated into any plist.

Also found: the installed and tracked plists have **drifted**. Installed names
`/usr/local/bin/claude` (per `logs/morning_agent.log`); the repo copies name
`/Users/taboost/.local/bin/claude`. Both are dead. Resync both when the CLI lands.

**Single-source fix to apply then (not 3 hardcodes):** the three agent plists already
`. /Users/taboost/quant/.env`. `.env` has no `CLAUDE_BIN`. Add
`CLAUDE_BIN=/verified/abs/path` there and invoke `"$CLAUDE_BIN" --print ...`. launchd's
empty PATH is a non-issue because the value arrives via the sourced file.

**Not done, handed to operator:** `~/Library/LaunchAgents` is a protected location and
cannot be mounted by a Cowork session, and `launchctl` is unavailable (the session's shell
is an isolated Linux VM — its own `/usr/local/bin/claude` is NOT this Mac and must not be
mistaken for it). So: no plist edited, nothing loaded/unloaded, nothing verified to fire.
`com.sovereign.reddit_sentiment` still loaded — retirement itself IS verified in code
(`execution/context.py:214,224,237` `Status.RETIRED`, `health_check.py:82`,
`refresh_caches.py:35`, guarded by `test_execution_context.py:115`), so the unload is
justified; it just needs the operator's hand + a `plist_watchdog --rebaseline`.

**CASCADE VERDICT — mostly INDEPENDENT.** Ran `audit/claim_check.py` first (RULE 10):
**2 REFUTED / 1 CONFIRMED / 2 UNVERIFIABLE**, exit 1. Claims file:
`audit/claims/2026-07-20-launchd-path-cascade.json`.
- **Harness "stuck on 2026-06-16" — REFUTED.** `harness.py:562` resolves live as
  `datetime.now(ET).date()`; nothing is hardcoded. It ran today: `[2026-07-20T20:05:11]
  2026-07-20: screened 14 candidate(s) / 1 filled / 3 skipped`. The `2026-06-16` lines
  (13:59–18:37) are **manual `--replay` runs** in the same log; "NO ARCHIVED UNIVERSE" is
  emitted only from the replay branch (`harness.py:405-412`) which `--live` never reaches.
  Yesterday's incident misread replay output as the scheduled job.
- **"`fill_log.jsonl` absent repo-wide" — REFUTED.** Exists, 3339 bytes, today 16:05.
  EOD Step 1's wait-for-fills *can* be satisfied.
- **`dashboard_state.json` — INDEPENDENT, and worse than stale.** `grep -rn dashboard_state
  --include=*.py` over the repo returns **zero hits**. *Nothing writes that file.*
  `sync_dashboard_data.py` writes `checklist_state.json` / `prop_challenge_state.json` /
  `g2p_state.json` / health JSONs — never `dashboard_state.json`. It exits 0 honestly.
  **`AGENT_DIRECTIVE.md:104` and `:218` are wrong** to name it as the writer (checker
  confirms the citation is unsupported). Orphan artifact; needs a decision — repoint the
  dashboard at `prop_challenge_state.json`, or delete it and correct the directive.

So installing the CLI unblocks morning/EOD/research + the empty `data/decisions/` (no
`signals_*.json` exists at all). It does **not** fix the harness (not broken) or the
dashboard (never wired). Two separate tickets.

**Found, NOT touched:**
- `execution/harness.py`, `carry_engine` — frozen, read-only, untouched.
- **`audit/claim_check.py` LOGPATH class is blind on this repo.** Both plist claims returned
  `UNVERIFIABLE: not well-formed (invalid token)` — its plist parser chokes on the XML
  comment header between DOCTYPE and `<plist>`, which most `scripts/*.plist` carry. Real bug
  in the tool RULE 10 depends on. Left alone: patching the audit tool mid-triage is exactly
  the drive-by-fix pattern it exists to prevent.
- `logs/research_agent.log` is dated **May 16** — the research agent looks dark far longer
  than today's break. Unexamined.
- Full triage note on disk (untracked, `logs/` is gitignored):
  `logs/incidents/2026-07-20-claude-binary-absent-triage.md`.

---

## 2026-07-20 — Anatomy Audit + Heartbeat Install

### Shipped (committed + pushed, final commit d12fe17)

**Commit 9bc2849 — 4 anatomy audit fixes:**
- `ict/library_bridge.py` — Added SIMILARITY_FLOOR=0.30. Was firing at sim=0.067 (noise), injecting garbage regime labels. Now abstains when below threshold.
- `sovereign/execution/oanda_bridge.py` + `scripts/oracle_session_close.py` — Fixed load_dotenv() called before ROOT defined. Oracle Session Close was 401ing on every non-~/quant CWD run.
- `scripts/launch_ny_scanner.sh` — Fixed bare python3 (system Python, no yfinance) → .venv/bin/python3. Had been silently crashing since May.
- `sovereign/data/reddit_scraper.py` — Switched from www.reddit.com (403) to old.reddit.com with raw_json=1, proper headers, exponential backoff for 429s.

**Commit 649bbe9 — Big 3 fixes:**
- `RISK_FRAMEWORK.md` — Created. Five constants ratified: DAILY_LOSS_HALT=0.02, CONSEC_LOSS_HALVE=3, CONSEC_LOSS_HALT=5, MAX_SINGLE_POSITION=0.10, DEFAULT_RISK_PER_TRADE=0.02. Amendment procedure: param_change_log.jsonl entry + doc update.
- `execution/harness.py` — Reverted unauthorized audit agent edit. Agent added _account_state_from_log() (38 lines) wiring daily halt logic, outside unlock scope (NEXT.md: "bar-fetch transport only"). Had two bugs: halt inert on fresh days, consecutive losses counted replay rows. Reverted with git checkout.
- `data/system/plist_watchdog_baseline.json` — Rebaselined. Was saturated at 8 NEWLY LOADED making drift invisible.

**Commit d12fe17 — CS229 headers + ticket renumber:**
- All 11 `sovereign/risk/` modules — Corrected false DEPRECATED headers. Audit claimed no live imports; all 11 ARE imported by sovereign/orchestrator.py (verified). Would have broken orchestrator if acted on.
- `tickets/backlog.md` — TICK-043 collision resolved. Finding 4A (daily halt inert) renumbered to TICK-044.

### New heartbeat LaunchAgents (all 5 loaded and verified)
- `alta.ny_am_scanner` — ICT NY AM scan, daily
- `alta.oracle_session_close` — Oracle reflection cycle, post-close daily
- `alta.reddit_scraper` — Reddit macro sentiment pull, daily
- `alta.system_health_check` — System health status file, daily
- `alta.plist_watchdog` — Monitors other 4 plists for unload

### Refused / not done
- No git add -A used anywhere (near-miss earlier; explicit path staging only)
- No harness.py changes accepted (execution path freeze in effect)
- Tier 2 audit agent went rogue (186+ turns, killed via stop signal) — findings treated as leads only, all verified before acting

### Open / still broken (verified, not audit noise)
- **Reddit 0 posts** — code fix committed (old.reddit.com endpoint), but system_health still shows 0. May need further debugging of the scraper run path.
- **Two hardcoded ATRs** — execute_daily.py and carry_engine._compute_atr. Pattern, not coincidence.
- **TICK-044** — -2% daily halt cannot fire on normal launchd path. P&L is flat (0.0) when gate evaluates on fresh trading days. Needs real-time P&L feed or intraday position tracking. Requires its own unlock.

### Lesson
Two audit passes today produced ~6 false claims (CS229 not imported, NY AM scanner never broken, pca_compressor deliberately removed, RISK_FRAMEWORK.md cited before it existed, stanford_cs229/ path). Treat audit findings strictly as leads — verify before acting.

### Session note
Dispatch session corrupted after tier 2 audit agent force-kill. SendUserMessage broken for remainder of session. All work committed and pushed before corruption.

---

## 2026-07-20 · EOD PASS
Fills: 0 (GO: 0, NO-GO: 0) | Session P&L: 0.0%
Challenge status: no live signal logged today — morning agent did not run; nothing to fill.
Harness delta (vs backtest): n/a (no fills)

EOD routine (16:05) completed correctly on a **zero-signal day**. `execution.eod` wrote the
honest note (`Trading/Ops/System-EOD-2026-07-20.md`: "No morning context was built / 0 GO");
brain write-back (`write_eod_summary` fills=0) + `refresh_brain_index.py` (74 lines) done.
No weakness notes invented (STANDING RULE: flat day, no ledger pattern).

**Three upstream failures surfaced (STANDING RULE 9 — logged, NOT fixed):**
`logs/incidents/2026-07-20-eod-upstream-failures.md`
1. **CRITICAL — `com.alta.morning_agent` never ran**: launchd invokes `/usr/local/bin/claude`
   which does not exist on this machine → no context/bias/signals for today. Cascades to all else.
2. **`com.alta.execution_harness` stuck on 2026-06-16**: `--live` processes the wrong date,
   screens 0, writes no `data/execution/fill_log.jsonl` (file absent repo-wide). Step 1's
   wait-for-fills can never be satisfied.
3. **`sync_dashboard_data.py` green-but-empty**: exits 0 but `dashboard_state.json` mtime still
   2026-05-31 (known dashboard data-path split).
All three are the "jobs exit 0 while producing nothing" pattern. Operator triage next session.

## 2026-07-20 · Fail-loud health infra — Reddit 403 + forex data status + green-but-empty check

**What shipped:**
- `sovereign/data/reddit_scraper.py` — was exiting 0 with 0 posts on HTTP 403 and rewriting
  the cache with an empty payload (fresh mtime kept freshness monitors GREEN). Now logs
  `REDDIT_FETCH_FAILED`, writes `data/health/reddit_status.json`, **leaves the cache untouched**
  on total failure, and exits 1 so launchctl records it. Partial failure → status `DEGRADED`.
- `scripts/forex_data_health.py` (NEW, read-only) — aggregates the existing TICK-025
  `sentinel/DEGRADED_*` flags into `data/health/forex_data_status.json`, per-pair, with
  `OK` / `DEGRADED` / `FAKE_DATA` classification.
- `scripts/health_check.py` (NEW) — the "green but empty" detector. Per-job registry of
  (output path, max age, substance predicate); verdicts `OK/EMPTY/STALE/MISSING/ERROR`.
  `--strict` exits 1 for launchd/CI use.
- `scripts/sync_dashboard_data.py` — runs both health builders last in the sync so the
  dashboard has a real health indicator, not a freshness proxy.

**Verdicts (measured, not asserted):**
- Reddit: **all 4 subreddits 403**, verified live. Exit code now 1 (was 0).
- `health_check.py` correctly flags `EMPTY reddit: file is fresh but payload is empty`.
- Forex: `GBPUSD=X` **DEGRADED** as of 12:23 UTC today (`yfinance` empty OHLCV frame).
  Classified DEGRADED not FAKE_DATA — that flag came from `macro_engine`, which drops the
  pair (returns None). No fabricated price reached sizing.
- Isolation test `test_pipeline_does_not_import_sovereign` **passes**. Suite 1542 passed /
  11 failed — all 11 pre-existing ICT session-classifier + pipeline failures, confirmed
  identical with these changes stashed. Not caused by this work; **unclaimed, still broken.**

**Blockers / refused:**
- **Reddit credentials do not exist.** No `REDDIT_*` keys in `.env`. The job cannot return
  data until a Reddit "script" OAuth app is registered and `_fetch_subreddit` is moved to
  the OAuth bearer flow. Exact steps in the module docstring. Not fabricated a workaround.
- **REFUSED the MarketDataAdapter failover for GBPUSD.** The yfinance fetches live in
  `sovereign/forex/carry_engine.py::_fetch_prices` and `macro_engine.py::_get_price_history`,
  both importable by the live/backtest execution path and under the shadow freeze. Adding a
  second price source changes what sizing sees — that is an execution-path change and needs a
  logged unlock here first. Shipped the visibility half only. **Unlock still required.**
- Note for whoever takes that unlock: `carry_engine._compute_atr` returns a hardcoded `0.001`
  when prices are None. That *is* substituted fake data reaching sizing — it is the real
  instance of the reported bug, and `forex_data_health.py` will classify it `FAKE_DATA` when
  it fires. It did not fire today.
- Left `.env.example` untouched — it carries another session's uncommitted edit.

---

## 2026-07-19 · Autonomous agent setup — AGENT_DIRECTIVE + 3 launchd plists

**What shipped:**
- `AGENT_DIRECTIVE.md` (repo root) — standing order for every autonomous Claude session.
  Covers four routines: 08:00 morning (info + bias + scan + dashboard), 09:30 signal
  scoring (GO/NO-GO via frozen HYP-107/HYP-093), 16:05 EOD (fills + reconcile + Obsidian),
  21:00 research (movers + micro-backtest + weekly_pattern_update.md). Eight standing rules
  including hash-freeze guard, holdout protection, and commit-on-every-pass.
- `scripts/com.alta.morning_agent.plist` — fires 07:55 ET Mon–Fri via launchd.
- `scripts/com.alta.eod_agent.plist` — fires 16:00 ET Mon–Fri via launchd.
- `scripts/com.alta.research_agent.plist` — fires 21:00 ET Sun–Thu via launchd.
- `scripts/install_agent_plists.sh` — one-shot installer; copies plists + loads + rebaselines.
- `research/weekly_pattern_update.md` — stub file for the research agent to append to.
- `logs/autonomous_test_2026-07-19.log` — placeholder; overwrite with live test run.
- `SYSTEM_STATUS.md` — updated with autonomous operation section.

**Push status:** PENDING (operator must run 2 commands — see below).

**OPERATOR ACTIONS REQUIRED (2):**
1. Install plists:
   ```bash
   bash ~/quant/scripts/install_agent_plists.sh
   ```
2. Run test loop:
   ```bash
   claude --print "Read ~/quant/AGENT_DIRECTIVE.md and execute the 08:00 morning routine" \
     --allowedTools "Bash,Read,Write,Edit,Glob,Grep" \
     2>&1 | tee ~/quant/logs/autonomous_test_2026-07-19.log
   git add logs/autonomous_test_2026-07-19.log && git commit -m "[AGENT] Autonomous test loop 2026-07-19" && git push
   ```

**Refusals:** sandbox cannot reach api.anthropic.com → test loop must run on Mac host.
No changes to execution path, frozen configs, or holdout data. Shadow freeze honored.

---

## 2026-07-18 · TICK-038 EXECUTION HARNESS — THE SPREAD ASSUMPTION WAS WRONG BY ~10x

Built `execution/harness.py`: one measurement instrument replacing the two forked shadows
(`live_shadow.py` HYP-093, `hyp107_shadow.py` HYP-107 — both still running, Phase 1 of a
3-phase migration; historical logs untouched as the reconciliation control group).

**THE FINDING.** First real bid/ask capture in this repo. On 59 archived gapper events the
**median quoted spread is 0.6113%**. `backtester/realistic_fills.py` assumes 1-15% and its
`_half_spread()` saturates at an 8% round-trip cap on gapper opens. Same events, same module:
model says median -3.49%, real quotes say median net **+1.02%** — a **+4.51pp** gap that is
model error, not edge. Every gapper backtest carrying that charge is biased PESSIMISTIC by
several pp/trade. Hand-checked example: TGHL 2026-07-16 09:30:59, bid 1.37 / ask 1.38 = 0.73%.
Artifacts + full caveats: `data/execution/BACKFILL_NOTE_2026-07-18.md`.

**WHAT THIS IS NOT.** The 59 events are a RANDOM ARCHIVE DRAW overlapping the mining window,
NOT the sealed n=57 holdout. Gross on this sample is +1.55% / win 55.9% vs the holdout's +5.4%
/ 70% — materially weaker, different sample, neither refutes the other. **HYP-107 stays
`REAL_BUT_MARGINAL_EXECUTION_UNRESOLVED`.** Nothing here adjudicates the edge and nothing here
touches funding (TICK-022 EV still 0.0; HYP-093 still below half its floor). The daily summary
carries NO readiness column by design.

**Corrections made to my own claims mid-build, both from measuring instead of asserting:**
- The flat-10% LULD bug is REAL but SMALL: 0.071%/trade at the 09:31 entry bar (old flagged
  2/56 events, new 0/56), ~0 at 10:30. I had drafted "nearly every opening minute" and "~2%
  per trade" — both wrong. Corrected in docstrings + `param_change_log.jsonl` (direction of the
  correction is UPWARD, logged before regenerating numbers).
- The brief specified HYP-093 as ">=100% above prior close". The sealed prereg (c5b10616) is
  `gain_min=0.50` / `qual_gain=1.30`. Used the prereg; test guards it.

**Architecture deviation from the brief, forced by measurement:** real-time is impossible.
This account 403s on SIP quotes inside a 15-minute window (probed: -13min 403, -16min 200);
real-time IEX is allowed but quotes AAPL at bid 314.75 / ask 347.97 (~10% spread, ~2% of
volume) and is unusable for cost measurement. Harness therefore does DEFERRED T+16min capture
and the plist runs 16:05 ET, not 09:25.

**Bugs found by running it:** (1) replay silently used the LIVE movers screener for past dates —
`alpaca.movers()` has no historical mode, so a replay scored today's movers against a past
session. Now sources from the archive and REFUSES to fall back. (2) exit quote was taken at the
bar OPEN; frozen specs exit at `b1030["c"]`, i.e. bar close = 10:31:00. (3) `/quotes` 400s when
symbol is passed as both path and query — surfaced only because the new client stopped swallowing
exceptions (the old `except Exception: sleep(5)` + `raise RuntimeError(url[:110])` destroyed it).

118 new tests green. Full suite 40 failed / 1368 passed — same 40 as baseline, +118 mine.
`data/execution/fills.jsonl` (Jun 30, unrelated system) untouched. Kill switch honored despite
paper-only; freeze/thaw dry-run passed, system verified 🟢 RUNNING after.

**BLOCKED ON OPERATOR:** plist is TRACKED-NOT-LOADED per convention (NEXT.md:836). Install and
`plist_watchdog.py --rebaseline` — skipping the rebaseline leaves a standing RED.
**NEXT (TICK-039):** recalibrate `SCENARIOS`/`_half_spread` against measured spreads, then re-run
the SEALED holdout with the corrected cost model before any HYP-107 viability claim.

---

## 2026-07-15 · TICK-036 TOP-3 MOVERS STUDY (MINING) — PREDICTABILITY CONFIRMED AT 13x, SERIAL RUNNERS FOUND

Colin's ask, run as Steps 1-2: 492 days (the 2 on-disk full-market years; 5yr needs the $79/mo
word), survivorship-free, test-symbols excluded after first-pass contamination catch. **A fixed
UNFITTED 20-name watchlist catches >=1 of the day's top-3 on 26.8% of days vs 2.0% random.**
Median top-3: +90% at $3.80. The ex-ante tell is IDENTITY (runner within 5d: 29% vs 15%;
y-top50: 12% vs 2%), not prior-day price drift (flat). Blue moon = 15 serial tickers filling
~6% of all slots (BNAI 8x). VIX regime: no magnitude effect. 14 lens-cuts counted into
mined_n. News pass + fitted-score prereg (HYP-099 candidates) = next rounds. Report:
data/research/movers_study/report.md.

---

## 2026-07-14 (night shift, operator asleep) · FULL DASHBOARD REBUILD SHIPPED + VERIFIED LIVE

Colin's overnight mandate executed (plan Plans/immutable-wondering-alpaca.md, 2 worktree agents
+ inline core, no swarms): **(N1) white/light-blue professional reskin** of index.html +
ict/index.html — 182 palette replacements + ~30 inline fixes + TradingView theme->light, ZERO
JS/feature changes, dark :root preserved as labeled rollback comment, all semantic colors
>=4.5:1 contrast (caveat: lightweight-charts panes stay dark — flipping needs JS, forbidden by
the do-not-break rule). **(N2) skills.html** — copy-paste Claude Code arsenal (searchable,
grouped, RESEARCH METHOD card, plugins table), linked ⚡SKILLS in the header. **(N3) Oracle
daily brief**: sovereign/oracle/daily_digest.py (haiku-tier, cost-capped, template fallback,
READ-ONLY) -> data/oracle/daily_digest.json -> "🔮 ORACLE — DAILY BRIEF + PROGRESSION" panel
under the ICARUS strip with Colin's own ladder (30 shadow days -> funded-sim day -> broker
pass/fail -> earn -> compound; G1 currently 1/30). New plist com.alta.oracle_daily_digest
LOADED (06:15 PT daily: digest + master sync — disclosed; read-only job under the night
mandate). icarus sync extended to carry the digest. **(N4)** live-verified over the wire:
light tokens serving, obrief panel present, skills.html 200, digest JSON 200, ict light.
master a10ae2f · sovereign-v2 861ffb9. First Oracle brief already generated (real LLM call):
headline + "today's one thing" = pick TICK-035 round-2 structure + W6 sizing question.
RAG/doc-index named as the next Oracle step (curated-context digest shipped instead — honest
overnight scope). Plugins documented on skills.html (chat-side; no automation built).

---

## 2026-07-14 (pre-dawn) · DASHBOARD CORRECTIONS SHIPPED + VERIFIED LIVE (~105s)

Colin's screenshot review, all three fixed on master b4d6db6 / sovereign-v2 3d7a950: (1) stale
Sharpe 1.08 -> **1.25** in all six spots (2026-06-07 re-measure; caption carries "provisional
pending TICK-024 swap re-verify"); (2) SHARPE PROGRESS chart gains the **v015 = 1.25** point
(v014 de-flagged, Current/Gap auto-update from the cur entry, title -> v015); (3) **main
Calendar tab now merges ICARUS shadow days** into day cells, weekly TOTAL column and month
total at the $100k reference basis (gold ◆ tag + legend; icarusMergeCal() folds
data/icarus_status.json into both the live-server and snapshot render paths — stays current
via the existing 16:20 auto-sync; ICARUS strip mini-calendar kept). Live-verified over the
wire: zero '1.08 OOS' strings remain, chart point + merge code + legend all serving. Out of
scope (named): prop-panel "as of 2026-06-09" is its own MC job; ICT proxy figures already carry
the honest banner.

---

## 2026-07-13 (late) · THE METHOD IS NAMED: ICARUS — DASHBOARD FLAGSHIP LIVE

HYP-093's fade is now **ICARUS** ("flew too high by 10:30; we sell the fall"). Dashboard revamp
shipped and VERIFIED LIVE on Render (~90s post-push): flagship strip above the tabs — sealed
verdict stats (p=.031, DSR .987), **shadow P&L calendar** (green/red day grid w/ per-trade
tooltips), latest closed trades, gate ladder to live. Data: `data/icarus_status.json`, generated
by `scripts/icarus_dashboard_sync.py`, WIRED into the shadow 16:20 close pass with an automatic
data-only worktree push to master (814d1e2 pattern) — results now flow to the dashboard daily,
zero hands. ICT page: hardcoded threat-1.00 placeholder RETIRED; honest banner (UNPROVEN p=.52,
live 3W/24L, gates stand) + "its FVG research seeded ICARUS →" pointer. master e917981 ·
sovereign-v2 dc183f3. Day-1 calendar cell: 2026-07-13 green, +0.046%.

---

## 2026-07-13 (night) · TICK-035 ROUND 1 EXECUTED (self, per Colin): FVG x CORRIDOR MINED OUT — HOLDOUT PRESERVED

Colin clarified the handoff was for me. P0: charter + fenced loader + causal engines (Pine
pvts() port with explicit confirmation delay; truncation-equivalence causality test — the
strongest form — green, 6/6 rails). P1: 720 cells mined 2018-01→2024-06 (3 entry families x
5 corridor conditions x 4 killzones x 6 managements x 2 FVG sizes), MNQ costs on every fill.
**Result: no prereg-worthy candidate.** Best cell +0.025%/day of NOTIONAL with negative median
event and 2/7 negative years — at/below the sealed benchmark BEFORE out-of-sample shrinkage,
against a now-1,529-trial deflation hurdle. Step-2 checkpoint honored: the 2024-26 holdout
stays VIRGIN; trials counted; bar unmoved. Structure notes (real but thin): gap-FILL fade >
retest; targets hurt; wide stop + session-flat; corridor-inside adds a sliver; NY_PM tilts F2.
Round 2 (new structure, not re-tuning): HTF FVGs / corridor-BREAK events / gap clusters /
overnight sessions — next session, operator may pick. Report:
data/research/yield_frontier/fvg_corridor_m1/ROUND1_REPORT.md.

---

## 2026-07-13 (evening) · TICK-035 RESERVED: FVG x FRACTAL-CORRIDOR NQ STUDY HANDED OFF (HYP-098 namespace)

Colin directed THE RESEARCH METHOD at the ICT/FVG + Fractal Corridors pairing "with tools laying
around." Scouting: the Pine v6 indicator EXISTS at repo root; intraday FVG detectors exist
(sovereign/forex/ict_engine.py; ict/fvg_detector.py has a documented ATR-leak the study must fix);
NO intraday FX on disk (docs/intraday_fx_acquisition.md); **ThetaData invoice read directly:
option.value.monthly $40/mo = EOD chains only, renews Jul 16 (Colin decision) — cannot power an
intraday study; terminal stays down.** Colin chose NQ-first (8.5yr 1-min on disk). Handoff prompt
(worktree law, fenced mining 2018-01→2024-06, holdout 2024-07→2026-06 w/ HYP-095 disclosure,
counted trials, both-priors prereg, benchmarks 0.023%/day + carry-1.25 context, "don't stop" =
new families at a never-lowering bar): research/yield_frontier/HANDOFF_HYP098_fvg_corridor.md.
Parallel session executes; this session stays owner of TICK-033 (W6 spec next).

---

## 2026-07-13 (afternoon) · HYP-097 GAP-THROUGH SEALED NOT_CLEARED — THE PHYSICS REFUTED THE SIZING HOPE; PLISTS LIVE; OPTIONS BRANCH WITHDRAWN

**Verification first:** Colin relayed a chat-session analysis claiming a hash-locked "HYP-096
k*r_net>=0.1776 viability rule sealed two sessions ago" + wrapper_scan.py. **Neither exists in this
repo** (no prereg file, no ledger entry, no 0.1776 anywhere, no wrapper_scan.py) — filesystem-checked
before acting. Options branch therefore closed as HYP-096 **WITHDRAWN_BY_OPERATOR** (priority
grounds + W2 physics; reopen condition = measured variance premium), NOT as a test verdict.

**HYP-097 (prereg 99e94408, BOTH priors recorded pre-lock, measurement after):** W* = 1.25 x
max(LULD collar bound, empirical max overshoot) per tier. Result: structural physics DOMINATED —
W* T10 0.627 / T20 0.798 (empirical 2yr max overshoot 0.477/0.404, n=289 stopped events).
Constitutional yield FELL to 0.0166%/day (vs 0.023% at the blanket 60%; floor 0.05%). **Colin's
~30% thesis refuted; sub-$3 names are WORSE than assumed. Sealed NOT_CLEARED.** The income gap is
physical (halt-cascade mechanics), not conservatism. Remaining honest routes: RCK distribution
sizing (W6), catalyst-split edge concentration (TICK-034, prereg on mining year + forward shadow
only), T10-only restriction (W6 grids it).

**Ops:** 3 plists LOADED (operator "yes, today, no discussion") — gapper_shadow scan/close +
market_snapshots live; today's signals AGEN/QTTB close at 16:20 ET automatically. Data budget =
shoestring-minus (no purchases; pessimistic-assumption borrow — 300% APR sensitivity barely moves
the yield: 0.0174%/day). ThetaTerminal stays down. RESEARCH METHOD codified in the charter +
feedback memory.

**Refused/held:** did not seal "NOT_VIABLE" from unverified external math; did not soften the
HYP-097 structural floor to rescue the thesis; holdout untouched by TICK-034 design.

**Push:** ✅ e3ee0ef + this entry.

---

## 2026-07-13 (Monday pre-open) · "TRADE LIVE MONDAY" → LIVE SHADOW ARMED; REAL DOLLARS REFUSED UNDER STANDING LAW

Colin: "make sure this trades live monday morning!" Real-money live REFUSED — four locks, all his
own: HYP-093 verdict VALID_BUT_BELOW_FLOOR (rider R3: no post-hoc reinterpretation); TICK-024
rider "gates any real dollar. Unchanged."; clamps not mechanically enforced until Jul-28; no
locate-capable equities broker account exists. Feedback-memory guardrail applied (sober-session
rule for live flips).

**Armed instead (Art. 6 shadow carve-out, source-tagged shadow_gapper):**
- `research/yield_frontier/live_shadow.py` — the FROZEN HYP-093 spec live-forward: --scan 10:50 ET
  (movers screener → 15-min-delayed SIP verification of the exact filters + M&A news exclusion →
  signals logged with 10:30-bar-open entry + constitutional sizing stamp + voice notify);
  --close 16:20 ET (stop-walk outcomes, same-day loop closure, shadow_daily.jsonl constitutional
  equity series). This IS the sim account proving the read — live Monday if the plists load.
- `daily_snapshots.py` — $0 forward-fill: Nasdaq halt CSV (working, 42KB captured today) + IBKR
  borrow FTP (**UNREACHABLE from this network** — ftp3 + ftp2/TLS both timeout; job fails loudly;
  if persistent → source switch to iBorrowDesk per W5).
- Plists drafted + linted, TRACKED-NOT-LOADED per operator-promote convention:
  `launchctl load ~/Library/LaunchAgents/com.alta.gapper_shadow_scan.plist` (+ _close, + market_snapshots)
  — after copying from scripts/. THREE one-liners on Colin's side.

**Path to real dollars (unchanged, owners+order):** W6 spec+simulator (Molly, this week) →
HYP-096 short-call-vertical feasibility+prereg (needs ThetaTerminal restart + Colin ack) →
CONFIRMED design → ≥20 trading days of this shadow record → TICK-024 clean → clamps enforced →
locate-capable broker account opened (Colin) → explicit go.

---

## 2026-07-13 (later) · TICK-033 W-WAVE LANDED: all five specialist briefs in, HYP-096 redirected

W1-W5 agents complete → data/research/yield_frontier/optimization/ (charter table synced).
Load-bearing: **W2 kills naive long puts** (parity re-prices the borrow into premium; IV 200-600%;
chains exist for only 20-35% of signals) → **HYP-096 = short call verticals**, gated on a ThetaData
chain-feasibility measurement (terminal currently DOWN — needs restart). W1: mechanism = attention
overpricing; edge survives BECAUSE shorting is constrained; catalyst-reliability split named the
highest-value future prereg. W3: SSR is DETERMINISTIC from our own tape (compute per-event, not a
probability); LULD collar math bounds stop gap-throughs; frictions adversely selected vs the best
events — W6 must model that correlation. W4: Risk-Constrained Kelly × per-day CVaR heat; current
sizing ~1/30-1/60 Kelly; disaster mixture mandatory in W6. W5: shoestring data stack $142/mo;
borrow-history is the market gap; **two $0 forward-fill jobs should start now** (daily IBKR FTP
borrow snapshot + halt-list scraper — unsnapshotted days are lost forever) → plists to draft,
operator promotes. ⏳ Colin: data budget (a: $142/mo / b: $860/mo) · forward-fill job promotion ·
HYP-096 short-call-vertical ack · ThetaTerminal restart.

---

## 2026-07-13 · YIELD FRONTIER G-PHASE (TICK-031): GAUNTLET RUN AND SEALED — THE SHOP'S FIRST REAL SHORT-HORIZON EDGE, AND IT'S BELOW THE CONSTITUTIONAL FLOOR

**Verdicts (preregs c5b10616/3e874fde/959372e9, riders R1-R3 enforced, hashes verified
pre/post):** HYP-093 parabolic-fade short **VALID_BUT_BELOW_FLOOR** (boot p=0.031, BH
survivor, DSR@809=0.987, 559 events on untouched 2024-25; gross +6.5% median/event, net
+4.9% after pessimistic costs; constitutional +0.023%/day vs 0.05% floor) · HYP-094
overnight weak-close short **NOT_SIGNIFICANT** (p=0.102) · HYP-095 NQ high-VIX dip long
**VALID_BUT_BELOW_FLOOR** (p=0.013, DSR 0.999, 40 events = min-n; +0.004%/day vs 0.02%).

**Meaning:** the first pre-registered short-horizon edge in shop history to survive
significance + an 809-trial deflation on unseen data — and the R2 sizing rider (worst-case
beyond the stop, validated by the holdout's own −32.6% p5 gap-throughs) correctly reduces
it below income relevance. Signal real; monetization absent AT THIS DESIGN. TICK-032
(rider 4): NO funded vehicle exists for HTB smallcap shorts (TTP bans the shape by rule,
Zimtra excludes US, T3/Bright = first-loss) — funnel thesis zero EV weight.

**Post-hoc (stamped, non-evidence):** binding constraint is sizing, not signal — even
friendly locate/worst-case bounds top at 0.046%/day. Doors opened as NEW preregs only:
HYP-096 defined-risk redesign (puts on parabolic gappers — collapses R2 worst-case;
dies or lives on option spreads), HYP-097 stop-defined 095 variant.

**Ops notes:** Polygon 2-yr lookback denied 9 holdout dates (partial-window clause invoked,
disclosed in verdicts); external kill hit one long background chunk again — foreground
<10-min chunks remain the law; holdout manifest 744 files, fetched only after gate-zero.

**Refused/held:** sealed verdicts final — no reinterpretation under the "find a loophole"
instruction (the honest loophole was already in the ladder: VALID_BUT_BELOW_FLOOR);
no new variants evaluated on holdout data; TICK-024 unchanged; execution path untouched.

**Push:** ✅ this batch to origin/sovereign-v2.

---

## 2026-07-12 (later) · YIELD FRONTIER M-PHASE (TICK-030): 809 CONFIGS MINED ACROSS 3 UNIVERSES — THE FRONTIER IS GAPPER TAILS, AND IT'S CAPACITY-BOUND

**Context:** Colin asked to "look at EVERYTHING... find a method... 2%+ daily... try it with
look back, then test the right way." Plan approved (Plans/immutable-wondering-alpaca.md,
force-added): MINING pass (dirty, stamped, counted) → G-phase gauntlet on untouched holdouts.
Universes he picked: equities gappers, NQ intraday, SPY options. Crypto declined.

**Shipped (2e9a7a9 → this):** `research/yield_frontier/` — holdout fences AT THE LOADER
(equities holdout year physically absent from disk; NQ rows >2024-06-30 truncated; chains
>2023-09-30 filtered), append-only `mined_n.json` (809 — feeds gauntlet DSR n_trials),
coarse frictions (HTB tiers/locate haircuts/halt gap-through stops/NQ ticks/options
k×half-spread), 16 families, 11 gate tests + look-ahead canary green, main-suite isolation
law green. Board: `data/research/yield_frontier/yield_board.{csv,md}` (508 ranked rows),
opens with the 2%/day arithmetic statement.

**THE FRONTIER (all MINING, not evidence):**
- **Equities gapper tails own it**: F-EQ2 parabolic-fade short (≥50% by 10:30, +30% stop,
  hold to close, M&A excluded) **+3.8%/day net, MEDIAN +5.6%/event, n=651** at 75% locate,
  p5 −30.5%, capacity ~$0.9M; F-EQ1 overnight weak-close short +3.5%/day (n=237); the
  +8.9%/day overnight-long cell is a tail/illiquidity illusion (lowest-vol tercile, cap $126k).
- **NQ over 6.5 years: ceiling +0.24%/day of notional** (high-VIX prior-day-down → long
  open-to-close, 1 neg yr of 7); ORB/day-trade families don't crack the top ten at honest costs.
- **Options: every measurable premium cell NET-NEGATIVE** (best −0.32%/day of collateral) —
  spreads eat the premium at mid±0.5×hs fills on real 2022-23 chains. Cache truth found:
  VRP-era chains are ~30-DTE monthlies only; delta/iv columns never backfilled; the modern
  spot cache is DIVIDEND-ADJUSTED (5%+ off actual — poisoned settlement until parity-spot fix).
- **2%/day sustained: not found, and the arithmetic says it cannot be** — but a confirmed
  capacity-bound 1-3%/day gapper edge would be the perfect prop-funnel vehicle (their capital,
  our %). That connection is the strategic payload.

**Traps hit + fixed:** Path.stem-on-.json.gz (also silently degraded sealed HYP-092's
robustness dedup — fixed, rerun p=0.612 vs 0.634, verdict unchanged, ledger annotated);
adjusted-spot settlement; empty chain files; OP3/OP4/OP5 unmineable on this cache (disclosed,
still counted in mined-N).

**⏳ Colin (G-phase gate):** pick ≤3 board rows for HYP-093/094/095 preregs. Recommended:
F-EQ2 thr0.5|stop0.3|close|mna_excl · F-EQ1 g30-50|locT1|volT2|short|next_close ·
F-NQ5 vixT2|prior_dn|long. Then G0 preregs → G1 holdout fetch → G2 verdicts (TICK-031).

**Refused/held:** no VRP-001-v2 execution; no live wiring; no ledger writes from M-phase
(annotation was the HYP-092 correction, backed up); mining stamped on every artifact.

**Push:** ✅ this batch to origin/sovereign-v2.

---

## 2026-07-12 · HYP-092 GAPPER-CONTINUATION READ (TICK-029): PRE-REGISTERED, RUN, ADJUDICATED — NOT_SIGNIFICANT, WELL-POWERED

**Context:** Colin's equities idea from the vault decision card (Gapper-Continuation-Decision-Card.md):
filter stocks already +30% by 10:30 ET (≥$2, ≥500K vol), read CONTINUING vs EXHAUSTED, "no look
ahead... very simple very rugged... focus on %." Built + ran the full year in one session as the
shop's first equities intraday study.

**Shipped (98ddd67 prereg → pipeline → seal):** `research/gapper_continuation/` — prereg
hash-locked 3e07c6a4 + ledger PREREGISTERED BEFORE any outcome data; Polygon grouped-daily
discovery (survivorship-free incl. delisted, buffered +20% superset per advisor review); ALL
analysis inputs from Alpaca SIP `adjustment=split` (probe-verified: delisted names serve bars —
MSW/TBH/LIXT; AAPL 4:1 split ratio exact); card checklist frozen into deterministic CONT/EX/UNC
votes (VWAP, higher-lows, up/down volume, range position, lower-highs, climax-fade, rejection
wick); read inputs strictly bars ≤10:25 ET, outcome = 10:30-bar OPEN → last RTH close (no shared
bar). 251 trading days, 11,396 candidates → 1,475 qualifying; coverage 11,395/11,396. 9/9 module
tests; deterministic rerun byte-identical; ICT isolation law green.

**VERDICT (sealed): NOT_SIGNIFICANT** — MWU one-tailed CONT>EX p=0.594 (n=558/391, unique
tickers 439/326), run-deduped robustness p=0.634. CONT median **−2.34%** vs EX **−1.81%** —
the mechanized read carries no information about the close. The real map: the filter's base
rate is a fade (ALL median −2.21%, 48% reverse >3%, 31% continue >3%); CONT mean +1.15% is
pure tail skew the read does not time (post-hoc note, not evidence); the 344 halt-excluded
names (unreadable at 10:30 by prereg rule) had descriptive median **+16.5%** — the violent
continuations were the unreadable ones. Full report: `data/research/gapper/report.md`.

**What survives:** the card's LIVE logging study still tests what the mechanization can't —
Colin's discretionary residual (catalyst, float, level, tape). If live reads separate where
the checklist didn't, the edge is the eyes, not the structure. Tail-capture mechanics
(stops + skew-riding) = NEW hypothesis, new prereg, if pursued.

**Refused/held:** no test switch after seeing tail-driven means (MWU was registered, MWU
decided); no threshold tweaks post-data (hash verified pre/post seal); no live wiring; no
short-side EV claims (borrow costs unmodeled); execution path untouched.

**Push:** ✅ this batch to origin/sovereign-v2.

**Post-hoc addendum (same day, quick scan, DESCRIPTIVE ONLY — ~30 uncorrected cuts):** catalyst
labels via Alpaca news (pre-10:30 only) + cached-bar features crossed vs continuation. The
mechanism-backed standouts: **M&A gappers are PINNED** (16.7% continue, only 31% reverse,
median −0.15% — arbs cap them; watchlist exclusion candidate); **parabolic ≥100%-by-10:30 fade
brutally** (22% continue, 66% reverse, median **−12.5%**, n=255); **no-news runners continue
more** (36.9%; recipe 30-50% extension + no-news → 40.5% cont, median +0.5%, n=185 — flips the
base-rate median positive). Chart-structure features stayed flat (consistent with the sealed
null). Files: posthoc_scan.py/.json, per_candidate_enriched.csv. Any of these → NEW prereg
(HYP-093+) on fresh data (2024-07→2025-06 is in Polygon's 2-yr window = clean holdout).

---

## 2026-07-12 · TICK-026 — CLOSED / STALE (import was never broken)

**Verdict:** stale premise, no code change. The ticket claimed `data.forex_factory_scraper`
was deleted 2026-07-02 while `data/calendar_fetcher.py:8` still imports it. It was NOT deleted:
`data/forex_factory_scraper.py` exists and is git-tracked (added in 541b47b, 2026-05-27). Verified
`python3 -c "import data.calendar_fetcher; import data.forex_factory_scraper"` → **BOTH IMPORT OK**.
`ict/daily_bias.py:102` also imports it (lazy, in-method) and resolves fine. Import chain intact —
nothing to restore or amputate.

**Note:** the backlog's exit-code-watchdog acceptance item (a repeatedly import-erroring job should
page, not whisper in launchd_err) is a real but *separate* observability concern — not reopened here,
since there is no live import error to page on.

**Refused/held:** did not reinstall/duplicate the scraper (already present); no code change made.

**Push:** see commit `fix(data): TICK-026 close stale ticket (import never broken)`.

---

## 2026-07-12 · TICK-025 — fail-loud DEGRADED sentinel for the yfinance OHLCV fallback (IN_PROGRESS)

**Context:** dispatched autonomously, diagnostic-and-fix only, explicitly scoped to *just* the
fail-loud flag — not the full backlog TICK-025 (proof_of_life/health/last_fill propagation stays
deferred). The live risk: the daily scan silently drops a pair (or stubs ATR to 0.001) when
yfinance can't return OHLCV for USDJPY=X / AUDUSD=X, so Oracle conviction rests on partial inputs
with the evidence buried in `logs/forex_scan.err`.

**Diagnosis (read before touching):** the fallback fires at two live-only points, neither imported
by `forex_backtester` (confirmed) → the 0.6886 reconcile anchor is untouched:
- `sovereign/forex/macro_engine.py::_get_price_history` — yfinance empty/exception → returns None →
  `score_pair` drops the pair (this is the Oracle-conviction path).
- `sovereign/forex/carry_engine.py::_fetch_prices` — yfinance empty/exception → ATR falls to 0.001.

**Shipped:**
- New `sovereign/forex/degraded_sentinel.py` — `flag_degraded(pair, reason, source)`: writes
  `sentinel/DEGRADED_<source>_<pair>.txt` (timestamp/pair/source/reason) + logs WARNING.
  Observability-only, exception-safe (never raises), cwd-independent path, imports nothing from ict.
- Wired both fallback points to call it. **Behavior unchanged** — both still return None; the flag is
  a pure side-effect (verified via mocked empty-frame + exception: None preserved, sentinel written).
- `sentinel/` created (`.gitkeep` tracked; `DEGRADED_*.txt` gitignored via `/sentinel/*` + negation).

**Verify:** helper + both instrumented paths unit-exercised green; `tests/unit/test_forex_macro_engine.py`
+ `test_forex_batch_backtester.py` = 18/18 relevant pass. One PRE-EXISTING failure noted, NOT mine:
`test_scan_all_pairs_returns_top3` hardcodes `ALL_PAIRS[4]` (stale 5-pair assumption; universe is the
4-pair HYP-045 set) — fails identically with my edits stashed. Flagged as a separate task, left alone
to keep commit hygiene.

**Refused/held:** did not touch signal logic, gates, sizing, params, `_apply_costs`, or any backtest
path; did not implement the broader proof_of_life/health/last_fill propagation (out of dispatched scope).

**Push:** see commit `fix(data): TICK-025 yfinance degraded sentinel`.

---

## 2026-07-11 (later) · LIVE-POSITION TRIAGE: financing ✓, no silent failure ✓, shadow agrees ✓ — AND the swap cost model is ~10× off

**Context:** dispatch flagged three checks on the live short EUR_USD (#227, opened 07-03,
−10k @ 1.14395, +$23.70). All three answered read-only (OANDA GET-only + logs), then one
approved read-only calibration script. NO live-path changes.

**Answers:**
1. **Financing: OANDA is PAYING the carry — +$1.1122 over the trade's life, every day a credit**
   (+$0.13-0.14/day, Wed triple +$0.4273; ≈ +0.42%/yr). The "$23.46 vs $23.70 gap" premise was
   wrong (unrealizedPL is exactly +23.70; financing is a separate field). The carry thesis is
   working in reality.
2. **No silent failure.** com.alta.forex.scan loaded, fired every weekday, explicit per-pair
   NO_TRADE with convictions (USDJPY .42 / GBPUSD .385 / AUDUSD .312 / EURUSD .023) through
   07-10; Saturday staleness = schedule. "5 pairs" premise stale: AUDNZD excluded by HYP-045 —
   4 pairs is by design. BUT: the scan runs **DEGRADED daily** (yfinance failures USDJPY/AUDUSD,
   synthetic AU/EU macro) visible only in forex_scan.err → TICK-025; and a calendar job dies
   repeatedly on `ModuleNotFoundError: data.forex_factory_scraper` (module atticked while still
   imported — a real silent-failure specimen) → TICK-026.
3. **Shadow exit machine: HOLD on every evaluated bar of #227, trailing stop ratcheted
   1.15739→1.14798 (+35 pips locked), zero unexplained (C5) divergences, no missed weekday.**
   Stockfish agrees with the account so far; July-28 checkpoint on track.

**THE FINDING (research/swap_calibration.py → data/research/swap_calibration.json):** the
model's swap table is an order of magnitude too small on ALL 4 pairs, with one sign flip:
EURUSD SHORT actually EARNS +0.42%/yr (model: pays −0.10%); USDJPY SHORT actually COSTS
−3.82%/yr (model: −0.35%); EURUSD LONG −2.45% vs −0.15%. At ~7d holds ≈ up to 0.07% notional
per trade mis-modeled — material vs ~0.5%/trade typical pnl; USDJPY-short-heavy periods in the
backtests are likely OVERSTATED. → **TICK-024** (gated hard: touches _apply_costs → every
backtest → the 0.6886 anchor; historical fix must derive from the rate-differential series, not
today's rates; impact study + param_change_log + Colin sign-off before any table change).

**⏳ Colin queue:** TICK-024 go/no-go (the important one) · TICK-025/026 priority · nothing
else needs you — the live position and its shadow are healthy.

**Refused/held:** no SWAP table edit, no scan/exit-manager changes, no live params; freeze intact.

---

## 2026-07-11 · HYP-090 "MODERN" (TICK-023): PRE-REGISTERED, RUN, ADJUDICATED — NOT_SIGNIFICANT SEALED

**Context:** Colin's recurring adaptive-parameters idea (3rd arrival, dispatch named it "MODERN":
daily trailing-window param sweeps + regime map). Pieces were killed before (HYP-065/066/067,
exit sweep, regime router) but the FULL daily-adaptive protocol had never run end-to-end — so it
kept coming back. Route A (Colin's pick, full surface incl. pair selection): test the maximal
version once under the real gauntlet and seal the family. Plan approved in-session
(plans/TICK-023.md). **Registered prior: NOT_ROBUST.**

**Shipped (one [RESEARCH] commit per phase, 0b0e73d→…):** `research/modern/` — prereg
hash-locked 6dd9cc85 + ledger PREREGISTERED **BEFORE any data** (gate-zero first-call, tested);
reconcile abort gate hit **0.6886 EXACT**; all inputs frozen (sha256 manifest); 64 ungated signal
builds ×2 spans + external causal VIX mask; 1,540 kernel runs (385 configs incl. #385=v015-exact
× 4 pairs; 30,788 trades; open tails recovered via flat-padding); exact daily M2M decomposition
(causal costs: spread@entry, swap daily); **config-385 parity: 411/411 canonical trades
date-identical** — the independent signal path provably equals backtest_all; A1 recent-winner /
A2 regime-kNN / A3 placebo engines with truncation-invariance PROVEN (20 sampled t × 2 windows);
block-bootstrap (L=5) + BH m=6 + DSR@5,775 + placebo envelope + per-year gauntlet. 20 tests green.
Full run 62s, seed 42, deterministic.

**VERDICT: NOT_SIGNIFICANT (the registered prior), and more decisive than predicted:**
- A0 static v015 daily-M2M Sharpe on 2016-07→2026-06: **+0.948**.
- ALL SIX adaptive runs: **+0.167 … +0.434** — every arm UNDERPERFORMS static (min one-sided
  p = 0.977; the direction is the reverse of H1).
- The killer detail: adaptive selection also loses to the **500-seed RANDOM-selection placebo**
  (p95 ≈ 0.92, i.e., even random daily config-hopping beats recent-winner chasing) —
  **the selection mechanism is actively ANTI-selective at ~13 trades per 90d window.** The map
  doesn't learn regimes; it buys noise peaks right before they mean-revert.
- Per-year non-degrade failed everywhere; regime-kNN (A2) was consistently better than pure
  recent-winner (A1) but still far below static — matching HYP-066/067's regime conclusions.
- Verdict sealed to hypothesis_ledger (backup kept), prereg hash verified pre/post seal.

**Standing instruction earned by receipt: the adaptive-parameters family on daily forex is
CLOSED.** Any future "adapt the parameters daily / regime map" idea at daily resolution
re-litigates HYP-090 and needs NEW DATA (intraday = Route B, a separate funded decision), not
new cleverness. v015's static config has now survived: 180-config sweep, GA search (10,100),
regime keying ×3, and full daily-adaptive selection over 5,775 variants.

**Refused/held:** no live wiring, no parameters.yml writes (monthly_reopt anti-pattern named in
the isolation tests), no band re-tuning, ledger sealed only through the locked criteria.

---

## 2026-07-10 · PROP-FUNNEL EV SIMULATOR (TICK-022): BUILT, PARITY-EXACT, RUN — THE "$10k/MONTH" QUESTION ANSWERED WITH NUMBERS

**Context:** Colin asked for "a strategy that passes a sim prop test 100 times and makes $10k/month
consistently by spamming the same method — find what works, why later." Plan-mode approved
(Plans/glistening-juggling-clover.md). Built the MEASUREMENT instrument instead of re-mining
settled-dead data: `research/prop_funnel/` — every strategy family through realistic prop rulesets.

**Shipped (one [RESEARCH] commit per phase, P0 03b8093 → P6):** parity harness (3 recorded MC
artifacts reproduced EXACT; two drift classes found+pinned: window_B pool, trades/yr clock),
ChallengeEngine (static/EOD-trail/intraday-trail DD, daily-loss actually enforced — PropFirmRules
stores but never enforces it; its `dd_trail_stops_at_starting` makes its "trailing" effectively
STATIC — divergence documented, not fixed), vectorized simulator (EXACT vs scalar on 6 rule
variants), feeds with evidence stamps (carry PROVEN_REGIME_FRAGILE n=110/411; ICT UNPROVEN;
**live closed outcomes n=27 = 3W/24L, WR 11% vs backtest 63.6%** — surfaced; futures ORB n=2
INSUFFICIENT), funnel chain (phases→funded→fees→program EV), sizing grids, synthetic frontier
(96 cells × 2 firms), 10 charts + verdict_table + summary_report under `data/research/prop_funnel/`.
38/38 module tests green; cross-process deterministic (found+fixed a hash()-salt seed bug).

**VERDICTS (seed 7, 10k trials; ALL pricing UNVERIFIED; iid-attempts caveat applies):**
- **P($10k every month ×12) = 0.0 on EVERY strategy × firm row.** The literal ask does not exist
  at $100k account scale with any edge this firm has. $10k/mo = 10%/mo ≈ Calmar 6.
- **"Pass 100×":** only carry_oos×FTMO comes close — P(funded)=0.996/attempt → p^100=0.68 — and
  that's the FAVORABLE-window (2023-24) pool; the honest full-decade pool gives p^100 = 1.5e-05.
- **Best PROVEN row:** carry_oos×MFF ≈ $2.1k/mo program EV; honest decade pool ≈ $630-750/mo;
  S0 scenario (carry dead forward) ≈ $260-600/mo — that residual is model-optimistic option value
  (reset-payout simplification), NOT free money.
- **Sizing optimizer's own discovery:** hot challenge / cooler funded (3.0x/2.0x) ≈ $4.2k/mo EV
  on carry_oos×MFF — a real, decision-grade lever, pending pricing verification.
- **Tension chart (the answer):** pass-90% and $10k-months contours sit on OPPOSITE sides of the
  ruin line until TRUE Sharpe ≳ 2-3. The frontier tells any future candidate what it must be.

**⏳ Colin queue:** (1) verify firm pricing (all EV rankings provisional); (2) return-scale
convention ruling (monte_carlo_prop default used, disclosed); (3) Phase R go/no-go (futures ORB
replay regen — operator-gated, never writes data/futures/); (4) Phase B new-edge hunts (HYP-089
options footprint) — the frontier now defines the bar they must clear.

**Refused/held:** no live changes, no gate loosening, no hypothesis-ledger writes (decision
analysis, not edge validation), no re-mining of settled-dead daily-bar data, futures replay not
regenerated without operator gate. rules_engine.py byte-identical.

---

## 2026-07-08 · POLITICAL-ALPHA (HYP-085 / TICK-020): BUILT, RUN, ADJUDICATED — H0 NOT REJECTED

**Pushed:** 5b5534b (P0 prereg) → 49b21c4 (P1 catalog) → 095bbdc (P2 abnormal returns) →
ffe85af (P3 positioning) → abcc120 (P4 verdict) + this entry.

**VERDICT: H0 NOT rejected — p = 0.3637** (10,000 statement-level placebo sets, seed 42,
`(n_ge+1)/(N+1)`). Trump-statement days: **11.21%** ±2σ-move rate (25/223 rows) vs placebo null
**10.30%** (σ 2.1%). The naive normal-theory two-day baseline (8.9%) would have called 11.2%
"elevated" — the pre-registered bootstrap null is exactly what stopped that false positive:
2025-26 is fat-tailed everywhere, not just around his statements. **HYP-085 verdict sealed
NOT_SIGNIFICANT** (the pre-registered prior; hash-lock 58e725ed verified before AND after the run;
status stays PREREGISTERED per convention). Colin's news-sniping thesis has its honest answer at
daily resolution: **no measured edge in trading every statement.** One thread survives for a
FUTURE, separately-preregistered look: direction-aligned pre-window skew **+1.54** (pre-announcement
returns lean toward the eventual direction) — Test 1 is descriptive BY DESIGN (spec §10 forbade
attaching significance machinery), so it is a lead, not a result.

**Process (the spec was law):** vault spec `Political-Alpha-Claude-Code-Spec.md` → prereg
hash-locked + ledgered + TICK-020 BEFORE any event data → 4 phases, one [RESEARCH] commit each.
Catalog: **168 qualifying events / 223 event×instrument rows** (honesty gate PASS, ≥30) from three
primary venues — whitehouse.gov 104 (1,327 articles scraped), Federal Register 62 (EO **+
proclamation** — disclosed: Section-232 steel/aluminum tariffs are proclamations; an EO-only query
missed the spec's own SLX mapping), Truth Social 57 (**10,081 own statuses** walked from the
trumpstruth.org mirror via cursor pagination; ET display verified two independent ways; link-share
posts excluded — his words only, disclosed tightening + a real title-concat bug found and fixed).
Phase 2: 223/223 evaluable, zero gaps; PA-0088/USDJPY hand-verified to 6 decimals. Phase 3:
**all five native ETF chains served on the Value tier** (probe surprise — XLE/SLX/XLF/KWEB/GLD),
FXE proxy for forex rows; 44 thin-smile gaps recorded, never synthesized; 13/179 manipulation-signal
flags (descriptive only). Liberation Day, steel→SLX, China→KWEB spot-checks all present.

**Isolation & safety:** new `research/political_alpha/` imports NOTHING from live namespaces
(AST wall + unit tests 11/11 green at close); zero execution-path touches; no OANDA, no launchd,
no live params. Ledger writes were append-only with .bak backups; prereg verify green post-seal.
**Suite: 40 failed / 1,243 passed / 1 skipped — the exact 07-07 baseline** (ml_stack errors =
known sklearn-missing class). **Article 6 stands; ignition locked** — a null here changes nothing
live, and even a rejection would only have been a candidate for the full gauntlet.

**Refused/handled:** did not loosen the qualifying definition at any point (the gate never needed
it — 168 events); did not fight truthsocial.com's Cloudflare (mirror + disclosure instead); did not
attach p-values to the manipulation flags or the aligned skew (spec §10); FR signing-date clock
pinned 12:00 ET and disclosed per row. Obsidian notes: `Political-Alpha-Phase{1..4}-2026-07.md`.

---

## 2026-07-07 · night — TICK-019 EXECUTED ON COLIN'S GO: THE GEOMETRY FAMILY ADJUDICATED

**Pushed:** c64acf7 (fill) → 9c7c964 (runner) → 6968ba9 (verdicts) + this entry. Zero unpushed.

**Verdicts (final, in the ledger):** **HYP-082 NOT_SIGNIFICANT** (corridor-beyond-carry: pooled residual
IC = 0.011, two-sided p = 0.598, N = 2,172 — an order of magnitude from the BH threshold 0.025) ·
**HYP-083 NOT_SIGNIFICANT** (daily-FVG continuation: median −0.0003, p = 0.741, N = 1,190; the
diversifier gate was never reached). BH m=2 per the locked manifest. **Both legs WELL-POWERED — no
UNDERPOWERED shelter; the nulls are earned.** Priors confirmed. The fractal-corridor and FVG threads
now have their honest answer at daily resolution; the prior explorations stay non-evidentiary forever
(A1). Gγ/HYP-084 untouched (dark-month clock — starts at your flip + TICK-017).

**Process integrity, end to end in one day:** specs hash-locked BEFORE features (01cacbd) → first
real-data geometry fill (19.4s, board 11,576 rows, **look-ahead 0 violations / 163,072 provenance
rows**) → runner built to spec by builder (699 lines, 82 tests; 7 ambiguities resolved WITH citations,
incl. an IEEE754 boundary catch on the cost floor and the no-flip Gβ reading) → dry-run mechanics →
seals → BH → verdicts. Gate-zero 16/16 locks verified at preflight; reconcile 0.6885; seed 42.
**Suite: 40 failed exact / 1,243 passed / 1 skipped. Watchdog GREEN (21).**

**No CONFIRMED → Article 6 stands, ignition locked.** Remaining triggers unchanged and yours:
gdelt_retry + sentiment_update loads (the positioning family's clock is still NOT running),
review_enabled flip, v2 ack, RED-1 batch, floor-clause signature.

---

## 2026-07-07 · RATIFICATION EXECUTED (Claude Code / Molly, on Colin's order) — RISK_CONSTITUTION v1.0.0

**Pushed:** 0789458 (enactment) + 565fa13 (propagation pin). The constitution is LAW: Art.1 **0.75%** ·
Art.2 **2.5%** · Art.3 breakers **3.5/5/6.5** peak-to-trough — re-anchored below the **8% TRAILING**
prop halt (the draft 8.5% flatten breaker sat ABOVE the trailing line and could never fire; the ratified
final rung sits 1.5% below it). Art.6 carve-out (final wording): *"This article binds live capital only:
paper, shadow, and research runs may exercise any pre-registered hypothesis at any evidence stage,
provided their records stay source-tagged so a paper outcome can never masquerade as live evidence."*
Prose + YAML twin + drift-test third leg amended in ONE commit (Art.5); drift tests 10/10;
param_change_log entry per NN#4. **Tier configs: all three are execution-path-imported** (risk_config ←
forex_live_scan via risk_engine/base_size · parameters ← ict/micro_risk + funderpro_executor ·
ict_params ← ict/pipeline) → **zero edits under the freeze; dated PENDING-RECONCILIATION
(blocked_on: shadow_close) note written into the constitution itself** — clamps + mutation tests land
the day the window closes. One propagation catch: the factory paper-adapter's cap stamp flipped
DRAFT-CAPS→RATIFIED-CAPS (the ratification WORKING against a DRAFT-era test pin — pin updated).
**Suite 40 failed exact / 1185 / 1 skipped. Watchdog GREEN.**

---

## 2026-07-06/07 (Claude Code / Molly) — DAY 3: THE ADJUDICATION — health truth · E0 memory integrity · VRP dead-as-specced · geometry family through the gate

**Push:** ✅ c9ad9ca → 01cacbd → 1e1b59e → dad7c47 → f1a29a0 → 78e2706 → d3ec74f (+ this entry). Zero unpushed.

**P — preflight (mandate-snapshot corrections included):** No sessions ran Jul 4-5; B2/TICK-006 was
already shipped Day-2; true baseline was 1142 (mandate's 1120 stale). **Sunday Jul-5 organic beat:
FIRED PERFECTLY** — W27 regenerated 17:11 with the full Forensics section (my Day-2 seals reported
back into it: 10 INTERIM SEALs + 1 BLOCKED counted; ratio 12:14:585; oracle line quarantine-marked),
Precedents ABSENT + citations ABSENT (dark mode held) + no PROP dupes. **review_enabled flip evidence
COMPLETE — recommendation: flip it** (your one-line logged param change). **Watchdog caught your
invariant_guard load as designed** (RED → rebaselined with reason → GREEN ×3 today, 21 jobs).
**P5 ratification: the task never started in-repo** — constitution still DRAFT v0.1.0, Day-1 YAML twin
only, zero tier-config reconciliation evidence; nothing half-landed to finish; E2 used 0.75% per your
mandate. Board tail: my one sanctioned foreground run was externally killed AGAIN (pattern now 4/4) —
**TICK-013 (your sentiment_update load) is the only honest owner of the tail.**

**H — health truth (audit/health_diagnosis_2026-07.md):** the Day-2 "dead trifecta" is **ALIVE** — all
four fired within 24h (responder 30-min cadence, generator Jul-6 03:06, factory correctly dry-run-idle,
oracle_cycle Jul-6 04:04); ict_scanner's Sunday alarm was CORRECT per its own mask. Real faults, exactly
two: **stray_tripwire WatchPaths inert since Jun-16** (reload or 15-min fallback — your batch) and
**OUTCOME_LOOP_STALL = hybrid artifact**: ~7 of "21 closed trades" are AUD_NZD/USD_CAD probe fills
pulse_check never pre-filters + a one-UTC-day signal-vs-fill skew beyond the Tier-2 window (Jul-1
matcher did NOT regress). Fixes named, NOT applied — **they ship with the RED-1 Blue change in ONE
review batch of yours** (same contamination family). Third persistence denial on sentiment_update
(consistent) — one-liner stands.

**E — evidence:**
- **E0 SHIPPED (c9ad9ca):** validator results are append-only — json stages become run lists, the
  ledger entry snapshots its prior record into runs[] before refresh, and **the account knob is now
  recorded on every run** (its absence made the 06-29 mystery). 4 tests prove two runs preserve both.
  Same treatment applied to the stage-1 VRP-001 write site.
- **E1: GDELT failed a 3rd consecutive paced attempt** (Jul-6, timeout) → built the off-peak organ:
  `scripts/gdelt_retry.py` + `com.alta.gdelt_retry.plist` (02:30 ET; done-marker stops retries; success
  rebuilds board, runs the auditor, and escalates the exact unblock sequence to you). **HYP-080 stamped
  PENDING-080-SCHEDULED; family stays 9/10; BH refused partial adjudication.** If a week of off-peak
  attempts also fails → the manifest's "family documents its handling" branch is YOUR protocol call.
- **E2: VRP-001 dead-as-specced (mechanical, sealed):** Art.-1-compliant floor = 25pt×$100/0.0075 =
  **$333,334** vs configs {100k, 50k, 10k} → the 25-pt structure cannot express one constitution-
  compliant contract in this firm. Credit-based refinement REFUSED (A1); **the 1.248 run named
  context-never-evidence in the annotation itself.** Successor **VRP-001-OPTIONS-v2 hash-locked
  (8ab13abf): 5-pt wings, 0.75%, $100k — SPECCED, UNRUN, gated on your ack.** No VRP backtest ran today.
- **E3: no CONFIRMED → no ignition command exists; Article 6 stands.**

**G — the geometry family entered THROUGH the gate:**
- **G1 (01cacbd): HYP-082/083/084 + GEOMETRY-2026-07 manifest hash-locked BEFORE any feature existed**
  (19cca02b · da070664 · 2baf6445 · 88ac7e02). Gα corridor-beyond-carry (two-sided IC, CPCV
  fold-stability, 3-pip cost floor); Gβ FVG diversifier (correlation gate IS the test; daily-bar
  UNDERPOWERED accepted); Gγ triangle→precedent-quality (outside BH, dark-month scored). All prior
  fractal/FVG material banner-marked non-evidentiary (A1). Ledger entries PREREGISTERED.
- **G2 (d3ec74f): extractors BUILT + tested (24 tests)** — trailing corridor R²/dev, REPLICATED
  look-ahead-safe daily FVG kernel (the Plan agent caught that the ict detector leaks last-bar ATR AND
  the sentiment wall is bidirectional; the parity test then caught a real inversion bug pre-merge),
  tri_state detector, 7 board columns, auditor blocks, AST wall extended. **The real-board fill + the
  locked Gα/Gβ run = TICK-019 (next session's E-track).**

**R — Numba scope (no ship, by rule):** numba NOT installed; py3.14 unsupported (needs ≤3.12 venv);
fast_backtester itself is pure numpy and imported by tests only. TICK-009 updated: dedicated research
venv + golden-set identical-output gate as acceptance; new work only, never sealed results.

**B — built (suite + watchdog after each):** TICK-015 slice 1 (f1a29a0 — shadow-log JOIN, join-never-
inference, conflicts counted, historical AMBIGUOUS untouched; slice 2 = decision_logger/oanda_bridge =
blocked_until shadow_close, freeze ruling) · TICK-007 step 1 (78e2706 — positioning_board.json export
+ guarded tail-call; NaN→null; DISPLAY-ONLY grep-verified) · TICK-018 (above). First builder wave died
on the account session limit (near-zero work lost, worktrees cleaned, re-dispatched clean).
**Suite: 40 failed EXACT / 1185 passed / 1 skipped (+41 = 24+5+8+4, zero new). Watchdog GREEN.**

**A — addenda:** A1 threaded through every spec/annotation by name · A2 scorer = TICK-017 (spec
mandatory ✓ verified design committed in plans/; build next session — Gγ's clock starts at your flip)
· A3 ratified-floor DRAFT added to DEFINITION_OF_DONE.md — **sign or strike.**

**Your queue (consolidated):** flip `experience.precedents.review_enabled` (evidence complete above) ·
load sentiment_update (TICK-013) · load gdelt_retry (TICK-016) · RED-1 review batch (Blue fix +
pulse_check probe-prefilter + match-window widening, one package) · VRP-001-OPTIONS-v2 ack (or strike)
· ratified-floor sign/strike · stray_tripwire reload · constitution ratification (still DRAFT) ·
attic/§S2 (standing) · PROP-2026-W27 formally promoted via TICK-015.

**Refused:** running v2 same-day by the same hands that saw old results · credit-refined sizing floors
(A1) · partial family BH · pulse_check/reflect_cycle unilateral fixes (RED-1 family = your review) ·
importing ict into the sentiment wall (and the look-ahead leak that import would have carried) ·
a 5th background gamble on the board tail · touching decision_logger/oanda_bridge under the freeze.

---

## 2026-07-06 · pm (Claude Code / Molly) — invariant guard now STANDING; two decisions parked

**Shipped:** `com.alta.invariant_guard` installed + loaded (Colin-authorized this session; blocked as
persistence on 07-03). In `launchctl list`, daily 09:20, last exit 0. Layer-4 detector is now standing.
TICK-004 fully closed.

**Parked for Colin (asked; away — held rather than act unattended):**
- **Close 4 USD_CAD positions** (134/144/154/165) — residue of the now-gated `test_oanda_set_stop.py`
  writer (the "rogue writer" my I2 flagged; solved by the 07-03 retargeting audit). Real broker write →
  left for the OANDA UI or an explicit go. Guard I2 keeps flagging until closed (intended reminder).
- **Next dig** — recommended the **RED-1 Blue fix** (`reflect_cycle` source/pair exclusion → I1 5→0),
  but it's pre-registered + cognition-path, so it needs his review before I start. Alts offered: L3
  regime-verify hardening (self-contained), VRP account-size re-run.

**Refused to shortcut:** did not execute a broker write unattended; did not start a gated/governance
change (RED-1 review-gated, L3 touches the training gate, VRP needs the account call) while he's away.

---

## 2026-07-06 (Claude Code / Molly) — ICT scanner RED audit: score-floor fix validated, USDJPY "gap" is intentional (no change)

**Push:** ✅ this NEXT.md entry on origin/sovereign-v2. **No trading-code change** (see below).

**Question audited:** why so few ICT paper trades after the score-floor fix (`00562bf`, 06-30)?

**Findings (July veto ledger, n=447):**
- **Score-floor fix worked.** Score-threshold vetoes are only **24/447 (5%)** — no longer the
  bottleneck. Dominant kills are **ADR-exhaustion 55% + WEEKLY_TREND_CONFLICT 31% = 85%**, both
  protective/by-design. `veto_reason`/`veto_stage` 447/447 populated; `adr_pct` 244 non-zero
  (9db295e5 fix live); `veto_detail` is **not a real schema field** (schema is reason+stage only).
- **Trades:** exactly **1** paper trade since 05-24 (`USDJPY_20260621`, Grade A 7.95, NY_Open,
  TIMEOUT 0R). Runbook's `ict_trade_ledger.jsonl` doesn't exist; real state is
  `data/ledger/ict_paper_trades.json` + `logs/ict_paper_trade_log.csv`.

**BLUE — proposed then REVERTED (RED-team caught an overclaim):**
- Suspected USDJPY missing from `PROVEN_PAIRS`/`LONDON_PAIRS` as an unintentional gap; built + tested
  the add (16 pre-existing env/data-drift failures unchanged, isolation invariant intact), **then
  reverted it.** USDJPY is **intentionally NY-AM-only** — matches `config/ict_params.yml`
  (`ny_pm_pairs`=3 pairs, `ny_am_session.pairs`=+USDJPY) and **two Colin-approved 2026-07-01
  Config-Changes-Log entries** ("USDJPY in NY-AM mode"). BLUE gate ("missing *without logged
  rationale*") **not met** → no change. `ict/orchestrator.py` unchanged from HEAD; the premature
  `param_change_log.jsonl` entry was removed (nothing retained).

**Refused to shortcut:** did not add USDJPY on the surface signal (ny-am list + line-818 include it);
verified the config section structure first, found the deliberate NY-PM vs NY-AM split, reverted.
ADR pre-session filter confirmed **draft, not wired** — left untouched (needs NN#4 logging first).

**Recommendation (Colin):** don't loosen ADR/weekly-trend gates to raise trade count — ICT edge is
unproven (p=0.52); scarcity here is the protective layer working. Full writeup:
`~/Obsidian/Obsidian/Trading/Research/ICT-Scanner-Red-Audit-2026-07-05.md`.

---

## 2026-07-03 · evening (Claude Code / Molly) — DAY 2: THE REWIRING — retargeting audit · options-leg seals · Library ascension

**Push:** ✅ 33cdcab → e9e05d3 → f243283 → 226149e all on origin/sovereign-v2 (+ this entry at close).

**P1 (preflight):** surface backfill LANDED + look-ahead GREEN (per the overnight addendum); VRP board leg
was partial — `sentiment_vrp_daily` 519 rows (2020→06-18), board vrp_signal 28% after today's rebuild;
full `update_sentiment.py` (VRP tail + all feeders + rebuild + inline audit) kicked at close — idempotent,
one re-run completes it if interrupted. ThetaTerminal healthy all day (schema probes exit 0).
**Close addendum:** that tail run was externally stopped mid-fetch (same kill pattern as the morning
vrp_feed run — long background jobs don't survive here; TICK-013's daily job is the real fix). Final
verified state: board 11,560 rows (look-ahead 0 violations from the 19:4x rebuild), vrp_signal 28.2%,
vrp data through 2026-06-18. Completed fetches are cached; **one plain `python3
scripts/update_sentiment.py` (or the loaded 07:45 job) finishes the tail + rebuild + audit.**

**Late-evening continuation ("continue"):**
- **GDELT attempt #2 (~21:00 ET): throttled 8/8 again, now with raw 30s ReadTimeouts** — sustained
  limiting, not burst timing. Family stays 9/10; TICK-014's scheduled off-peak mechanism is the path.
  If the 02:30-ET class of retry ALSO fails, the manifest's "family documents its handling" branch
  becomes a Colin-level protocol decision. Board rebuilt again on the fresher feeder rows: **11,572
  rows, look-ahead 0 violations / 71,156.**
- **Runner hardened for unblock day:** `--only HYP-080` flag (no duplicate seals; fresh seed-42 stream,
  documented; own manifest file). Sequence when GDELT yields: gdelt fill → `--only HYP-080` →
  `--adjudicate --dry-run` → `--adjudicate`.
- **TICK-006 SHIPPED (builder, worktree):** the Sunday review now reads six forensics feeds — oracle
  health (quarantine-marked), the week's ledger RESULTS Counter + terminal-verdict dedup (fails OPEN —
  can never silently suppress a proposal), acted:abstained:vetoed ratio, shadow-audit parity, lesson
  velocity, briefing macro block. 22 new tests. One test-isolation bug caught + fixed in MY OWN new
  TICK-005 test (it read the real annex — only visible after the backfill populated it; the class of
  bug the constraint exists for). **Suite: 40 failed exact / 1142 passed / 1 skipped. Watchdog GREEN.**

**E — evidence race:**
- **VRP-001-OPTIONS (TICK-002, full verdict): `NO_TRADES` at the specced $100k account** — IS 2022: 1
  trade/51 sizing-skips (net −438); OOS 2023-01→2024-06: 0 trades/78 skips. Signature f07e9f2 OK, real
  chains (1,414 cached), 15/15 VRP tests. The 06-16 sizing wall is now PROVEN on both splits: 1% × 25-pt
  wings can't floor ≥1 contract at $100k. **Yours: raise the research account (unfrozen `--account`,
  plain re-run) or re-spec wings/risk% (signed change).** Recovered context: a **2026-06-29 stage-2 run
  at a raised account logged 50t / Sharpe 1.248 IS** — it sat unread in the ledger until today's run
  overwrote it (validator REPLACES its entry — result-history loss class; old record in git history).
- **Six interim seals** (locked protocol; gate-zero 11/11; reconcile 0.6885; truncation-invariance PASS;
  seed 42; interpretations declared PRE-RESULTS in `positioning_options_legs.py` + stamped per seal):
  074 p=.161 N=24 UNDER · 075 p=.166 N=54 OK · 076 p=.825 N=48 UNDER · 078 p=.282 N=29 UNDER ·
  079 p=.209 N=7 UNDER · **077-FULL composite p=1.0 N=5 UNDER (supersedes COT-only; both remain)**.
  Coverage stamp everywhere: options history 2020+ (Value-tier depth), z-warmup eats 2020.
- **HYP-080 BLOCKED stamped** (prereg's own data_dependency): GDELT throttled **8/8 calls even at 5s
  pacing** (~19:30 ET) — the June "burst" theory is dead; it's sustained free-tier throttling. Off-peak
  retry = TICK-014. **Family: 9/10 primaries exist, all raw p ≥ .16; BH correctly REFUSED** (runner
  refuses partial adjudication; `--adjudicate` ready the moment 080 lands).
- **No CONFIRMED → no ignition command unlocked; Article 6 stands.** (Note: VRP's verdict ladder tops at
  PARTIAL_CONFIRMATION — it can never by itself satisfy Article 6.)

**R — retargeting audit (the "what decision does it feed" trial):** `audit/retargeting/R1..R7 +
RETARGETING_TABLE.md` — **16 RETARGET / 17 LEAVE / 10 ATTIC-CANDIDATE**. Corrections it forced:
the Alexandrian Library is LIVE in ICT (query every scan; `learn()` live-WRITES the canonical json —
freeze-listed forever); the "unidentified OANDA writer" was **our own `test_oanda_set_stop.py` trading
on every plain suite run** (now env-gated `OANDA_INTEGRATION=1`; 8 fills explained); **no plist-hash
watchdog existed** — built `scripts/plist_watchdog.py`, baselined (20 jobs), GREEN ×4 today.
- **R4 headline:** the fast engine idles during research (family runner replays static v015 trades — by
  prereg DESIGN; retrofit REFUSED, successor harness = TICK-012). The 1.26M bars/s claim was Numba-real;
  **Numba is dead on py3.14 → 123k/s now** (TICK-009 recovers ~10×).
- **R6 headline:** the positioning board (COT%/TFF/VRP/rr25/bf25/surprise/GDELT) reaches **no dashboard
  panel** — Colin-blindness beats machine-blindness; display-only export = TICK-007. Wiring the board
  into live readiness gates was REFUSED (Article 6 — evidence first).
- **R7:** watchdog trifecta (health.responder + 2 research nightlies) silently dead 18+ days; nothing
  watches the watchdogs → TICK-008 (diagnose-first).

**L — Library ascension (TICK-005): SHIPPED + LIVE.** 5 new `experience/` modules + 43 tests (builder
subagent, worktree). `library_annex.jsonl` holds its **first 17 lived entries** (1 review · 7
attributions · 9 seals incl. today's). Sunday review gains a guarded Precedents section + **falsifiable
citations** (`scoring_due` = attribution can later score every analogy). Canonical json byte-identical
(test-enforced). W27 flag-on dry-run (isolated dir): real precedents matched from the week's own tags
(carry/crowded_short → GOLDILOCKS_LOW_VOL '17, LOW_VOL_MELT_UP '13). **`review_enabled` ships FALSE**
(deviation, documented in parameters.yml + param_change_log: legacy test fixture would write real
citations; also keeps Sunday Jul-5's organic beat on pristine v1). **Activation = your one-line flip
post-Monday-verification.** L2b decision-time stub: default OFF, nothing imports it.

**A — `docs/REWIRING.md`** (as-is/should-be diagrams, ranked list, leave-alone list, refused list).
Top 5: ① schedule the sensory board (TICK-013, YOUR one command — machine correctly denied
persistence) ② Library slice (DONE) ③ review forensics feeds (TICK-006, ready) ④ suite-must-not-trade
(DONE) ⑤ positioning-board display export (TICK-007).

**T:** TICK-005..014 filed (renumbered live — a concurrent session claimed TICK-004; see memory note)
+ plans/TICK-005/006/007/008.md (+ 002/003 pointers; plans/ is gitignored → ticket plans force-added).

**B:** builder merged after diff review; **suite 1120 passed / 40 known-failed (EXACT set, 0 new) /
1 skipped** (the gated live-order test); watchdog GREEN after every batch.

**Attic-candidates (your ruling, nothing moved):** com.alta.cache.refresh Reddit path · dead .env keys
(Tiingo/OpenWeather/Firebase×/AV-technical; Polygon→equity batch) · cross_system_bridge.py ·
.smart-env embeddings · bench telemetry (or 5-line alert) · stray_tripwire watch-mode (inert since
Jun 7). Existing §S2 list untouched.

**Operator actions queue (consolidated):** load sentiment_update (TICK-013) · load invariant_guard
plist (afternoon session's TICK-004) · VRP account decision (above) · precedents flip after Jul-6 ·
S2 rulings · PROP-2026-W27 promote/decline (standing).

**Refused:** partial family BH · fast-engine retrofit into the locked family · exit_reason inference
(PROP-2026-W27 is the path) · board→live-gate wiring · working around the launchctl denial · stage-4
holdout touch · regenerating the historical W27 review in place · param tweaks after seeing any result.

---

## 2026-07-03 · afternoon (Claude Code / Molly) — Layer-4 audit → standing Adversarial Invariant Guard

**Context:** Colin's "four correctness layers" reframe → chose *map the layers, target the weak one*.
3-agent read-only audit: L1 Reality **STRONG**, L2 Signal **MEDIUM**, L3 Environment **MEDIUM-STRONG**,
**L4 Adversarial WEAK** — the two failures that materialized (RED-1 Oracle contamination; rogue USD_CAD
writer) were caught only by ad-hoc human audit; the red-team skills auto-fire nowhere.

**Shipped (TICK-004):**
- **`audit/invariant_guard.py`** — read-only, spec-first, self-escalating Layer-4 detector (sibling to
  `shadow_divergence`). I1 Oracle-reflection purity, I2 no rogue/sentinel OANDA writes, I3 forbidden-pair
  guard. Reimplements probe/insane-risk heuristics *independently* of the audited code; imports nothing
  from the execution path (AST-tested).
- **`audit/invariants_spec.md`** — hashed single-fence contract (mirrors `divergence_spec.md`); consolidated
  what the plan drafted as a separate `config/invariants.yml` into the one hashed spec (no 2nd source).
- **`audit/CORRECTNESS_LAYERS.md`** — the four-layer map, durable.
- **`tests/test_invariant_guard.py`** — 19/19 green, incl. the I1 exclusion test RED-1 lacked + an
  independence cross-check vs `backfill_decision_records._is_test_fill` + the no-execution-import guard.
- **`scripts/com.alta.invariant_guard.plist`** — daily 09:20, **tracked but NOT loaded** (see blockers).

**Verdicts:** guard `--run` = **FAIL** (correct) — I1=5, I2=14, I3=18. I1 caught the exact RED-1 records
(AUD_NZD + 4× USD_CAD `fills_backfill` LOSS into Oracle); I2 caught the `USD_CAD units=1 stop=1.0` sentinel
probes. 3 URGENT escalations written to `messages_to_colin.json`. Report: `audit/reports/invariants_2026-07-03.*`.

**Blockers (yours):**
- **Load the guard to make it standing** — `launchctl load` was blocked as unauthorized persistence (correct;
  operator-promotes). Run: `cp scripts/com.alta.invariant_guard.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.alta.invariant_guard.plist` (or authorize me).
- The RED-1 **fix** (source/pair exclusion in `reflect_cycle`) is still the pre-registered Blue change awaiting
  your review — the guard is its regression test, not a substitute. Until it lands, I1/I3 keep screaming (intended).

**Refused to shortcut:** did not touch the execution path or `reflect_cycle` (guard is read-only + detect-only);
did not import the audited code into the adversarial check; did not silence the day-1 escalation on a real,
confirmed contamination; did not work around the persistence denial.

**Push:** ⏳ committing guard + spec + map + tests + plist + this entry to `sovereign-v2` (report artifact
included; `messages_to_colin.json` live state NOT committed).

---

## 2026-07-02 · evening (Claude Code / Molly) — Phase 2: verdicts · memory organ · trial · factory · heartbeat

**Shipped (b20aec3..593d46d + TICK-001):**
- **V — interim seals under the locked protocol** (b20aec3): gate-zero hash verify held;
  HYP-072 raw p=0.772 (N=168) · HYP-073 p=0.723 (N=168) · HYP-077 COT-only p=0.171 (N=9
  UNDERPOWERED; reconcile guard reproduced 0.6886 first) · HYP-081 p=0.356 (N=389,
  verified-subset calendar). Sealed as dated ledger ANNOTATIONS; status stays
  PREREGISTERED. **Family BH adjudication awaits the options legs — no verdict is
  CONFIRMED, so factory ignition stays LOCKED (no command unlocked).** Priors holding.
- **T — thesis-exit spec** (62744ed, sha 0340424056db3277): predicates DSL;
  `thesis_invalidated` first-class; **ownership: the predictive paper loop owns
  predictive exits; the frozen L2 manager keeps carry** (two-layer wall).
- **W — memory organ LIVE** (0c4c6ab rubric first, then c7197c9): journal with
  abstentions (Article 4 observable), rubric v1 hash-pinned `7e646084468f7e71`,
  backfill **30 rows** / 7 attributions, `com.alta.journal_sync` 09:15 Mon-Fri +
  `com.alta.weekly_review` Sun 17:00 **both loaded**.
- **D — factory built, ignition gated** (58d69e0): hashed snapshots, CPCV validation,
  calibrated zoo + abstain wrapper, append-only registry;
  `python -m factory.train --hyp HYP-072` → **IGNITION REFUSED, Article 6 printed
  verbatim** (test + live demo). Paper adapter DRAFT-CAPS-stamped, NOT enabled.
- **S — subtraction trial** (e221827: attic 34 dead files, pure renames, reversible;
  2e4dd67: verdicts). **Ambiguous list needs your ruling** (execute_daily.py — it IS
  the loaded papertrading job; train_core.py; the ict-engine bridge story; firebase/,
  layer2/3, dashboards, ~12 legacy dirs batched with the equity-engine ruling).
  40 pre-existing failures re-diagnosed: **API/timezone/network drift — sklearn IS
  installed**; verdict keep-as-known-failure, dated; ml_stack is NOT a factory input.
  Baseline unchanged: 40 failed / 1039 passed (34 new tests, 0 new failures).
- **F — FIRM.md + DEFINITION_OF_DONE.md + THE FIRST HEARTBEAT** (593d46d): board →
  journal → attribution → `review/2026-W27.md` → **the review job itself wrote the
  ledger's first PROPOSED entry, `PROP-2026-W27-exit-reason-capture`**. The metabolized
  triple: the stray AUD_NZD fills-reconstructed loss → AMBIGUOUS (exit mechanism never
  recorded) → the machine proposes recording it. Promotion is yours. Sunday Jul 5 =
  first organic beat.
- **TICK-001 done** (pre_approved): vrp_schema_verify now probes the nearest expiration
  ≥ probe date — FXE exits 0 (probe 2022-03-11, MATCH True), SPY passes.

**Live mid-session:** your terminal restore landed — constraint-4 probe found FXE
serving, so the **full 2016→ options-surface + VRP backfill is running in the
background** (board rebuild + look-ahead audit at its tail). Next session: verify
coverage → rr25_z history → **VRP-001 first, then HYP-074/075/076/078/079 + HYP-077
full composite** under the locked protocol (TICK-002 path) → family BH still blocked on
HYP-080's GDELT backfill.

**Verdicts:** 4 interim seals (above); TICK-001 acceptance 4/4; no final ledger verdicts
(by design — the family adjudicates together).

**Blockers (yours):** ratify RISK_CONSTITUTION (DRAFT) · S2 ambiguous rulings ·
keep/kill com.sovereign.papertrading · promote/decline PROP-2026-W27 ·
cb_meetings_historical.json (fetch agent died on session limits — retry or supply dates).

**Refused to shortcut:** no improvised family adjudication on 4/10 primaries; no
exit-reason guessing (AMBIGUOUS over invention — it became the machine's own first
proposal); no fixture rows anywhere near hypothesis input; no test-greening of the 40;
zero execution-path edits; no speculative label construction (the label IS the
hypothesis).

**Push:** ✅ b20aec3..593d46d + this entry pushed to origin/sovereign-v2.

**Late addendum (post-backfill, ~00:30 Jul 3):**
- **Options surface LANDED + fused:** 1,306 real rows (bs_invert, 2020-01-03→2026-07-02);
  board rr25/bf25 **96.5%** of post-2020 rows (term_slope 28.5% — thin far-dated FX-ETF
  listings); look-ahead audit **0 violations** incl. the surface ASOF check. Coverage
  truth for all future options-leg seals: **Value-tier chain history starts ~2020 — six
  years, not the decade** (verified: 2016 chain 403s while 2024 returns 200 same-second;
  it's an entitlement depth wall, not rate limiting).
- **VRP leg NOT filled tonight** — three attempts, three distinct causes, all now handled
  in code (7337cab): depth-wall 403s → skip-and-count; terminal stall timeouts →
  skip-and-count + 10-consecutive circuit breaker + per-pair persistence + 0.1s pacing;
  final attempt found the terminal DOWN at start → graceful skip (board vrp_* stays NULL,
  never fabricated). **Needed: a stable ThetaTerminal session** — then one run of
  `python3 -c "from sovereign.sentiment import vrp_feed; vrp_feed.update()"` (+ board
  rebuild) finishes it; every chain fetched so far is parquet-cached, retries only advance.
- TICK-001 done · TICK-003 ticketed (options-leg family run, awaits TICK-002).


---

## 2026-07-02 (Claude Code / Molly)

**Shipped**
- **ThetaData gateway restored.** Old API key had been rotated (401/404) + a prior-session typo
  (`THETA_DATA_API_KEY` vs the parsed `THETADATA_API_KEY`) had it running keyless. New key stored
  as `THETADATA_API_KEY` in `~/ThetaTerminal/.env` + `~/quant/.env` (both `chmod 600`; quant
  `.env` deduped 4→1; gitignored / non-repo). The 42MB `ThetaTerminalv3-new.jar` bootstrap
  authenticated → downloaded runtime (`lib/202607021.jar`) → **serving v3 REST on `:25503`**;
  log confirms `Options: VALUE`.
- **Workflow scaffolding installed.** `CLAUDE.md` merged (added plan→build / ticket / reporting /
  efficiency layer; preserved the 5 trading NON-NEGOTIABLES, decision-logger, Oracle loop, tests,
  commit style, architecture table, live-state block; corrected the `factory/` pointer →
  `sovereign/autonomous/research_factory.py`). Created `tickets/backlog.md` (TICK-001, TICK-002)
  and this repo-root `NEXT.md`.

**Verdicts**
- **FXE options entitlement = SERVED under the Value tier ($0 ask)** — the question OPEN since
  2026-06-22 is closed. Curl-verified: SPY + FXE `list/expirations` HTTP 200 back to 2012; FXE EOD
  `2022-03-18` on `2022-03-07` returns a full `strike/right/bid/ask` chain (VRP IS window covered).
- No ledger/test verdicts today — **no models trained, execution path untouched.**

**Blockers**
- `scripts/vrp_schema_verify.py` errors HTTP 472 (median-expiration vs fixed-2022-date probe bug,
  not entitlement) → **TICK-001** (ready, pre_approved).
- VRP loader-fill state ambiguous: memory says the 4 `ThetaDataLoader` bodies were filled
  2026-06-16, runbook still lists them TODO → verify on `sovereign-v2` before Stage 2/3 (**TICK-002**).

**Refused to shortcut**
- Did **not** overwrite `CLAUDE.md` blind — merged, preserving load-bearing invariants (its own
  instruction, and the safe call).
- Did **not** auto-push the foundational `CLAUDE.md` change without review (see Push).
- Did **not** touch `forex_exit_manager` / `decide_exit` / execution path — shadow freeze intact,
  no unlock this session.

**Push:** ✅ Committed + pushed to `origin/sovereign-v2` (`[INFRA]` — CLAUDE.md workflow layer +
tickets/backlog.md + repo-root NEXT.md), Colin approved. (Terminal `.env` files are
gitignored/non-repo and carry no secrets into git.)

---

## 2026-07-03 — Red/Blue/White audit of the 2026-07-01 overnight builds (READ-ONLY, no commits)

Full report in Obsidian: `Trading/Research/Team-Audit-2026-07-01.md` + `Blue-Team-Proposed-Fix-2026-07-01.md`.

- **RED-1 CONFIRMED (SEVERE):** Oracle reflection is contaminated. `reflect_cycle._load_decision_log_summary`
  only drops OPEN/EXPIRED, so backfilled probe/stray records (attributed outcomes) pass. Live 7-day FOREX
  window = **7/7 backfilled, 5/7 forbidden pairs (USD_CAD/AUD_NZD), 0 genuine trades, fabricated 1W/6L**.
  `update_outcome()` and the summary have no pair/source filter. Root cause is the cognition read path,
  NOT the backfill (which correctly closes the loop).
- **RED-2 CLEARED:** AV `news_feed` scorer sign (`rel·sent(base) − rel·sent(quote)`, +=long-base) and
  6-char decomposition correct; `board_state` is a pure feature-store join with no live consumer yet.
- **RED-3 FALSE ALARM on named jobs** (`hypothesis.generator`, `oracle.session_close`, `cache.refresh`
  all git-tracked, repaired from `.corrupt-20260701`, none trade). **But found a still-live unlogged OANDA
  writer:** 1-unit USD_CAD LONG probes (sentinel `stop=1.0/tp=2.0`, demo acct), 8×, **most recent
  2026-07-03 01:51 UTC**. NOT fvg_express (killed), NOT execute_daily/papertrading (equities). Source
  unidentified — spun off as a task: trace + gate it (same class as fvg_express).
- **BLUE (pre-registered, NOT committed):** source-exclusion of `fills_backfill`/`test_fill` records in
  `reflect_cycle`. Awaiting Colin's review before any code change (NON-NEGOTIABLE #4).
- Shadow/exit path untouched. No unlock. No commits this session.

---

## 2026-07-03 — Blue Team fix APPLIED (RED-1), full autonomy granted

Applied the pre-registered Blue fix above. Commit `78c8c0b` on `sovereign-v2`.

- **Read-path guard added** in `reflect_cycle._load_decision_log_summary` AND
  `hypothesis_generator._load_reps`: exclude `source == "fills_backfill"` and `test_fill is True`
  before records enter the Oracle reflection input / reps population. Read path only — the backfill
  writer (which correctly closes the exit loop) is untouched.
- **Tests:** new `tests/unit/test_oracle_backfill_exclusion.py` (6 tests: backfilled USD_CAD absent,
  genuine FOREX decision present, test_fill excluded, reps guard). Related sweep 45 passed, ICT/
  sovereign isolation 61 passed — no regressions.
- **Live verification:** the Oracle's real 7-day window (2026-06-26→07-03) held **7/7 backfilled,
  forbidden pairs USD_CAD×4 / AUD_NZD×1 / USD_JPY×2, 0 genuine** → **0 after fix** (correct fallback
  to harvest-based lesson, no forbidden pairs). RED-1 was a total, active contamination, now clean.
  Detail: Obsidian `Trading/Ops/Oracle-Fix-Verification-2026-07-03.md`.
- **Push:** ✅ `origin/sovereign-v2` — remote SHA `78c8c0b` verified == local HEAD.
- Shadow/exit path untouched. No unlock. No live parameter changes.

---

## 2026-07-12 — HYP-089 TSMOM backtest → NOT_SIGNIFICANT / SEALED

Ran the locked HYP-089 pre-reg ([[HYP-089-TSMOM-Prereg-2026-07-12]]): 12-month (252d) TSMOM,
sign signal, inverse-vol scaling (10% target / 60d vol), 3× cap, 4 v015 pairs equal-weighted,
daily rebalance, 2015–2024. Zero parameter search. Fully isolated new codebase in `research/tsmom/` —
imports nothing from `sovereign/` except the read-only DEGRADED sentinel (never fired; no pair degraded).

- **Verdict: NOT_SIGNIFICANT.** Conjunction fails on the Sharpe gate alone:
  - Gross Sharpe **0.2773** — FAILS >0.3 (and economically negligible: 1.8%/yr gross, $1→$1.18, maxDD −15.2%).
  - Carry Pearson **r = −0.156** (p≈1.3e-15) — PASSES <0.7 (near-orthogonal → real diversification, but moot).
  - **6/10** positive years — PASSES ≥6, but marginal (2024 Sharpe ≈ +0.0005, effectively flat).
- Matches the Hutchinson et al. (2022) post-publication FX-momentum decay prior exactly. Sealed
  permanently alongside HYP-090; **not** re-tested with alternative lookbacks (that would be data mining).
- Carry-direction series for the correlation = sign of FRED policy-rate differential (long the
  higher-yielder), mirroring the v015 `high_yield_side` logic — documented proxy, no v015 path touched.
- **Reproducibility:** data vintage pinned to `research/tsmom/prices_cache.parquet`; re-runs are
  deterministic (verified: cache path reproduces 0.2773 exactly). Artifacts: `backtest.py`,
  `results.json`, `summary_report.md`, `equity_curve.csv`.
- **Commit/Push:** ✅ `658733f` on `origin/sovereign-v2` (`59f0b1c..658733f`). Ledger updated in Obsidian
  `Hypothesis-Status-2026-07-05.md`.
- v015 carry code, `_apply_costs`, swap table, sealed studies, shadow/exit path — all untouched. No OANDA writes.


---

## 2026-07-12 — TICK-027 (HYP-091 TSMOM) + TICK-028 (ICT 90d projection)

Two research tasks handed together ("literature summary -> HYP-089 TSMOM backtest + ICT trade-count
projection"). Both READ-ONLY; shadow/exit path untouched, no unlock.

**Concurrency reality (documented so the next session doesn't repeat it):** multiple parallel Claude
sessions share this working tree and commit to sovereign-v2. Since the scouts ran, parallel sessions
took HYP-090 (MODERN adaptive), HYP-092 (gapper), and TICK-023..026/029. My plan's HYP-090/TICK-023-24
were stale -> re-derived to **HYP-091 / TICK-027 / TICK-028**. A parallel session had also already built
+ committed an **HYP-089** TSMOM quick-look (658733f, NOT_SIGNIFICANT) in `research/tsmom/` — I did NOT
touch that dir; built the corrected study in `research/tsmom_hyp091/`. (My TICK-027/028 got swept into a
parallel session's commit 74f25f9 — benign.)

**TICK-027 / HYP-091 — TSMOM diversification of the v015 carry book -> NOT_SIGNIFICANT.**
- Phase 0 hash-locked prereg (`data/research/preregister/HYP-091_tsmom_carry_diversification.json`,
  hash c1a47738) + ledger PREREGISTERED, committed `d2caebb` BEFORE any price data. Hash-lock verified
  intact after adjudication.
- Corrected instrument vs the parallel HYP-089 quick-look (which used a carry-SIGN PROXY, NO financing,
  DAILY rebalance): monthly (Moskowitz) rebalance, correlation vs the **ACTUAL v015 returns**, and
  **correct rate-differential-derived financing** (operator decision) — NOT the Colin-gated
  SWAP_RATES_ANNUAL (TICK-024 proves it ~10x too small + one sign flip). Financing = anchored
  differential-tracking model reproducing the 2026 OANDA snapshot + trade-227 anchor, varying by the
  FRED rate-differential across 2015-2024 (captures the 2022 USDJPY carry blowout).
- VERDICT NOT_SIGNIFICANT, sealed on null leg (a): **OOS(2023-24) Sharpe = -0.349 <= 0**. Corroborated:
  permutation p=0.140 (>0.05), deflated-Sharpe prob 0.753 (<0.95). Correct financing makes it WORSE than
  price-only (it pays the real carry costs the broken model understated). Correlation vs actual v015 is
  LOW (rho -0.128 primary / -0.136 broken apples-to-apples) — TSMOM IS uncorrelated but too weak: 50/50
  equal-vol blend Sharpe 1.064 < v015 1.166 (lift -0.102). 2022 (+1.27) dominates the per-year table.
- Loader sanity: v015 per-pair-weighted Sharpe 0.7209 reproduces the 0.6886 decade headline (yfinance drift).
- Isolation 3/3; NN#1 ICT-isolation still green. Commits `d2caebb` (P0) + `34244c3` (P1-4). Ledger
  HYP-091 -> NOT_SIGNIFICANT. Research pass only; deployment OUT OF SCOPE (Art. 6).

**TICK-028 — 90-day ICT taken-trade projection -> fill rate is the bottleneck, not vetoes.** (`30b3770`)
- Read-only; dedup per-scan re-emission **13.7x** (4051 raw veto records -> 296 unique setups). Live veto
  rates recomputed: **ADR 45.3%, weekly-trend 6.8%** (memory's 55%/31% was STALE — update noted).
- Two-basis 90d projection (bootstrap-over-days, N=10000 seed 42): LOGGED/committed setups ~**94**
  (95% [72,118]) -> ABOVE the ~30 band, so signal/veto frequency is NOT the constraint. ACTUALLY FILLED
  ~**2.1** -> FAR BELOW 30: **~98% of ICT decisions are EXPIRED unfilled limit orders.** For a ~30-trade
  prop challenge the binding constraint is the FILL/EXPIRY rate, not the ADR/weekly vetoes. (Confirms the
  "grade-A signals are LIMIT orders" memory.) Shadow-freeze compliant; deterministic; guard 3/3.

**Refused to shortcut:** (1) did not reuse the parallel HYP-089 result or its research/tsmom/ dir; built a
canonically pre-registered corrected study instead. (2) did not use the known-broken SWAP_RATES_ANNUAL for
the primary financing (per operator decision) despite it being the easy path. (3) pre-registered + committed
BEFORE observing price data. (4) did not touch the execution/exit path or any live parameter.

---

## Session 2026-07-13 (EOD) — TICK-033 W6: shadow-outcome seal + friction-simulator spec (`09463d1`, pushed)

**Task 1 — sealed 07-13 shadow gapper-fade shorts** → `data/research/gapper/signals_2026-07-13.json` (new).
Closes pulled from yfinance daily bars (2026-07-13, auto_adjust=False):
- **AGEN**: entry $7.72, stop $25.10, close **$6.12** (day high 8.70). Short return **+20.73%** → **WIN**,
  stop not hit. (Earlier mark was $6.78; closed even lower — fade completed.)
- **QTTB**: entry $19.31, stop $25.10, close **$21.38** (day high 22.50). Short return **−10.72%** → **LOSS**,
  stop not hit. (Earlier mark $21.30; closed above entry — reversal against the short.)
- Net shadow day: 1W / 1L, neither stop hit. Both closes clean from yfinance (no halt/delist → no UNKNOWN).

**Task 2 — W6 friction simulator SPEC (spec-first, no code)** → `research/gapper/W6-friction-simulator-spec.md`
(new). Pre-registration law honored: spec committed before any simulator code. Models the six frictions as
**adversely selected** (correlated with signal quality via gap magnitude), per the W3 load-bearing flag —
not flat costs: locate `P(locate|gap)` gate, deterministic SSR boolean from tape, gap-conditioned borrow
regime, 10:30 market-impact slippage, LULD limit-up gap-through (corrected direction — for a short the
adverse halt is limit-UP, ties to W5 two-cycle +30% stop-run), recall-timer forced early cover. Loss model =
W4 disaster mixture (0.1–0.5%/event of −100..−200%) + GPD stop-gap (stops don't truncate). Output = per-event
friction-adjusted **distribution** (median net, 5/95, %WIN→LOSS flipped, %unavailable, annual P&L @1.25%).
Validation = run on 559 events, within 20% of W3 empirical estimates. Constants sourced from W3/W4/W5 briefs.

**Flagged, not silently resolved:** `OPTIMIZATION_PROGRAM.md` (W6 row) earmarks a *different* path
`optimization/W6_SPEC.md` for the **sizing-policy** simulator. This friction spec is the return-generating
engine that FEEDS that sizing grid — composed, not duplicate. Cross-reference written into both the spec
header and this log so the two W6 docs don't drift. Requested path (`research/gapper/...`) honored as the
explicit instruction.

**Did not:** touch execution/exit path, change any live parameter, or write simulator code (spec-first).

## 2026-07-16 — HYP-099 regime study + HYP-093 forward-year sim (autonomous session)
- **Shipped:** 5-step method run end-to-end. Step 1 scan (117 events 2025-H2, 63 combos, 24 green) → Step 3 prereg hashed 2bcf4b9f… committed a7148f3 BEFORE holdout touch → Step 4 sealed: **HYP-099 NOT_SIGNIFICANT** (V1 Δ+0.97% p=0.226; V2 sign-flipped). Ledger entry 81. Regime conditioning on the gapper fade is a mined mirage; do not revive intraday_push/overnight_gap without new data.
- **Session goal met via collateral:** sealed HYP-093 rule forward-simmed on fully unseen post-seal year (2025-07→2026-06, 234 events, zero lookahead, mandate spec 2% notional / 25% post-entry stop on SIP minute highs): **+24.4% yr, maxDD 4.0%, Sharpe 3.4, win 57.7%**, robust across 20–40% stop levels. Key mechanism: the stop caps the −937% tail; 2026-H1 unstopped mean is ~0 — the stop IS the strategy's survival.
- Files: research/gapper/regime-study-{prereg,results}-HYP099.md, regime_holdout_HYP099.json, sim_annual_HYP093_forward.{csv,py}, data/research/gapper/event_post1030_highs.json.
- **Pushed:** a7148f3, 608bbb4 → origin/sovereign-v2.
- **Refused shortcuts:** first sim run used daily highs → 60% false stop rate (−30% yr artifact); refetched post-10:30 minute-bar highs instead of shipping the conservative wrong number. Holdout never touched pre-hash.
- **Blockers/next:** locate/borrow availability still the real-world bind (linear scaling is paper-only); HYP-100 entry-timing study pre-declared as next candidate if wanted; stop overlay was mandate-specified, not in original HYP-093 prereg — a formal HYP-100 prereg of "HYP-093 + 25% stop" on NEW forward data (live shadow) would fully regularize it.

## 2026-07-16 (later) — HYP-100 prereg + borrow capacity measurement
- HYP-100 preregistered (hash a44998f6…, commit ceeb54f): 25% stop overlay sealed against FORWARD shadow data only (events ≥2026-07-16; eval at N≥40 or 2027-01-16). Closes the stop-overlay loop.
- ThetaTerminal v3 restarted (was down since 07-02 cred warn; launched with .env key via homebrew openjdk, Options: VALUE live :25503).
- **Capacity measurement (the truck-size answer): of 234 forward-year events — 169 (72%) NO_OPTIONS, 34 no live expiry, 15 no two-sided ATM quote at 10:30 → only 16 (6.8%) parity-measurable.** Of those 16: median implied borrow 46%/yr, p75 146%, p90 203% (n small, micro-cap spreads noisy). Implications: (1) long-put/HYP-096 options bypass reaches ≤7% of signals — cannot carry the strategy; (2) 93% of the universe is optionless HTB micro-cap — locate via broker is the ONLY scaling channel; (3) modeled 200–600% APR tiers are conservative vs the measured median on the measurable slice. $10k–$50k rows real; $500k theoretical, unchanged.
- Files: research/gapper/borrow_measurement.py, data/research/gapper/implied_borrow_234.json. Pushed ceeb54f + this commit.

## 2026-07-16 (evening) — Dashboard section + prop MC + TICK-037 IB locate snapshots
**TICK-037 OPENED** (note: prompt said TICK-027 but that was consumed by HYP-091 TSMOM on 07-12; max was TICK-036): daily IB shortable-list snapshot vs gapper universe.
- **Task 1:** ICARUS dashboard panel extended with HYP-093 confirmed-edge block (forward-sim stats, stop robustness, scaling+borrow caveat, dynamic HYP-100 accumulation line, MC table hook) — master 50e3add, 32cc5dd (data). Deploys via Pages deploy.yml.
- **Task 2:** research/gapper/monte_carlo_prop_hyp100.py — 100k bootstrap paths off the forward-sim daily P&L, $100k ±8%: 30d PASS 5.9%/BUST 0.0%; 60d 33.2%/0.1%; 90d **58.4%/0.1%** (2% notional). Sensitivity P(PASS 90d): 1%→9.1%, 2%→58.4%, 3%→78.6%, 4%→85.5% (BUST stays ≤4.6%). LOCATE CAVEAT stamped in every output. sovereign-v2 a243b14.
- **Task 3:** scripts/ib_shortable_snapshot.py + com.alta.ib_shortable.plist (07:00 daily, TRACKED-NOT-LOADED, operator promotes). Working endpoint: ftp2.interactivebrokers.com user `shortstock` (plain ftp.interactivebrokers.com times out; HTTPS mirrors 404). **First snapshot 2026-07-16 vs 930-ticker universe: EASY 739 (79%) / HARD 42 / UNAVAILABLE 0 / NOT_LISTED 149.** Caveat: today's list ≠ gap-day 10:30 availability — borrow evaporates on the gap; the daily series is what answers that.

## 2026-07-16 (night) — HYP-101 threshold study: STOPPED AT STEP 1 (correct negative) + sizing Pareto
- **Mandate deviation flagged:** 2022-2024 dirty data does not exist (all gapper data = 2025-07→2026-06; entitlements can't reach 2022-24 intraday). Adapted: dirty scan on 2025-H2; 2026-H1 sub-100% rows reserved as untouched holdout. Holdout never touched — no prereg was warranted.
- **Step 1 dirty scan (2bac2a2): NO candidate clears.** Cumulative thresholds: ≥75% → median +0.2%/50.0% worked; ≥60% → +1.5%/53.1%; ≥50% → +1.4%/53.8% (bar: median>0 AND ≥55%). Incremental 60-75% band technically clears (+5.2%/56.5%) but non-monotonic across bands (noise signature) and already fails the prereg null (≥ baseline +10.4%) in-sample — refused to preregister theater. **11:00 entry strictly worse** (paired Δ −3.1%). The 100% threshold is where the parabola actually breaks; relaxation dilutes the edge.
- **Fallback MC sizing sweep on confirmed edge (100k paths/level):** 1%→9.2/0.0 · 1.5%→37.2/0.0 · 2%→58.8/0.2 (reproduces prior 58.4 ✓) · 2.5%→71.0/0.7 · 3%→78.5/1.6 · 3.5%→82.9/3.0 · 4%→85.4/4.5. **No level reaches 95/5 — P(PASS) saturates ~85% because signal frequency (147 event-days/yr) binds, not risk.** Recommended operating point: **3% notional = 78.5% pass / 1.6% bust**; beyond 3%, each pass-point costs ~2x bust. 95% needs more signal (different catalyst family), not more size.
- Files: research/gapper/threshold_scan_hyp101.py+json, monte_carlo_prop_sweep.py+HYP100_sweep.json. Pushed 2bac2a2 + sweep commit.

## 2026-07-16 (late night) — HYP-102 continuation long: STOPPED AT STEP 1 (correct negative)
- Dirty scan 2025-H2 (89 faders / 28 continuers). Separating features exist (rvol +123%, overnight_gap +50%, intraday_push −74% — all p>0.05, n=28) and tell a coherent "anchored gaps stick" story — but it's the SAME feature family HYP-099 already watched fail holdout transfer.
- **Kill shot: every tradeable LONG rule loses in-sample on the dirty data itself** (8 rule shapes: medians −5% to −16%, win 27–39%). Best features shift P(continue) 30%→~39%; base rate dominates. Only positive cell = n=13 mining residue (10th rule tested; can't reach the 30-event holdout gate regardless). And every viable cell still fades on median → a long-switch rule would ALSO damage the confirmed short book. Double loss.
- **No prereg. 2026 holdout untouched.** research/gapper/HYP-102-continuation-STEP1-STOP.md sealed. Non-faders are not identifiable at 10:30 with available features — continuation risk is what the 25% stop is for. Path to 95% P(PASS) requires an INDEPENDENT signal family, not gapper re-slicing. Two consecutive correct negatives (HYP-101, HYP-102) = the gapper table is mined out at this data resolution.

## 2026-07-17 — FunderPro unlimited-time check + 3.5% + combined HYP-093/095 MC
- **FunderPro: no time limit CONFIRMED** (2-phase: +8% P1/+5% P2; 1-phase +10%; 5% daily loss incl. floating; 10% static max DD; consistency rule = hidden governor). **BUT: FunderPro is a forex/CFD shop — it almost certainly cannot trade HTB US micro-cap equities. TICK-032 verdict (no funded vehicle for HTB smallcap shorts) still stands until an instrument check proves otherwise.** Unlimited-time math is still decision-relevant for ANY equity-capable no-deadline vehicle.
- **Unlimited-time MC (100k paths):** deadline removal flips the sizing logic — bust becomes the only failure mode, so LOWER sizing wins: 3% → 99.2% pass / 0.8% bust (10% DD line; median 51 trading days to +8%, p90 124); 3.5% → 98.5%/1.5%. Colin's 3.5% recommendation is optimal for fixed 90d windows; under no-deadline, 3% dominates.
- **Combined HYP-093+095 MC (sealed rules only, zero new mining):** HYP-095 event dist regenerated exactly (40 events, mean +0.68% ✓). 90d window: adds only ~2-4pp (3.5%+NQ25%: 85.7%/2.4% — NQ leg too infrequent (8%/sessions) and too small to fill enough days). Unlimited: 3%+NQ25% → **99.4% pass / 0.6% bust**. Verdict: the structural fix is the deadline, not the second signal; HYP-095 helps at the margin.
- Files: monte_carlo_unlimited_time.json, hyp095_event_dist.json, monte_carlo_prop_HYP102.json.

## 2026-07-17 — HYP-103: EV-optimal challenge config (scan + prereg, pending shadow verdict)
- 240-cell mined grid (stop×sizing×entry, minute-slice re-pricing for 5 entry times; new cache data/research/gapper/event_minute_slices.json; anchor cell reproduces sealed sim within 1pp price-source drift). FunderPro 2026 fees confirmed: 25k/$250, 50k/$300, 100k/$550, 200k/$995, fee refunded on pass, 80% split, no time limit.
- Top-20 pattern: sizing at grid max (edge artifact — flagged), $200k always, unlimited ≥ 90d everywhere, entry 10:30/10:45 only, stop 20–35 plateau. Survival-adjusted EV (funded-year 10%-DD death MC) confirms ordering; p(survive) 96–99.9%.
- **PREREGISTERED SPEC (996a1c47…, 30dbe3f): 10:45 entry / 25% stop / 3.5% notional / $200k / unlimited.** Dirty estimates: annual +63.7%, P(pass) 0.995, P(bust) 0.005, EV_y1 ~$103k. Chose 3.5% over EV-max 5% (grid edge + locate footprint). Null: shadow EV_y1 ≤ $5k; eval at N≥40 or 2027-01-16. Ledger entry 82 (REGISTERED).
- 90d-vs-unlimited: unlimited wins every paired cell. Best fee-to-profit: $200k (fee 0.5% of account vs 1% at 25k).
- Caveats stamped: mined grid ≠ evidence; locate assumption; TICK-032 instrument wall (FunderPro CFD list unverified for US micro-caps — operator to verify before paying any fee).

## 2026-07-17 — Production backtester/ engine (foundation rebuild) + honest gapper correction
- Built `backtester/` package: data.py (Alpaca-primary minute layer, gz+parquet cache), engine.py (bias-free fills), audit.py (auto bias checklist), mc.py (block bootstrap, 2.5M paths/s, fork-parallel), scanner.py (memoized-fill + vectorized sizing sweep + permutation FWER Bonferroni/Holm). 7/7 tests green (tests/test_backtester.py). ICT/sovereign isolation test still passes.
- **Corrected the record — my −19pt bias claim was WRONG.** On 1-min bars only 5/79 stops gap through the trigger (stop-fill bias ~1-2pt, not 19). The REAL biases: (1) **transaction cost omitted** — realistic entry-bar spread drops annual +25.5%→+9.8% (full) / +18.2% (~1% spread), Sharpe 3.5→1.5-2.6; (2) **IID Monte Carlo** — block bootstrap cuts prop 90d P(PASS) 78.5%→64.7% and multiplies bust 1.6%→12.7%; unlimited pass 99.2%→88.9%.
- **Honest headline: gapper fade ≈+10-18%/yr, Sharpe ~2, ~10-13% prop bust — NOT +24.4%/3.4/~0.** EV grid: 0/240 configs survive FWER (date-shuffle conservative at ~1 event/day). Every prop/EV number from 07-16 was built on the optimistic engine; read down accordingly. HYP-100/103 preregs still valid (shadow verdicts pending) but their dirty EV estimates are now known-optimistic.
- 1-min bars for all 234 events cached to data/cache/minute_bars/*.parquet. Fetched fresh via Alpaca SIP (yfinance can't serve year-old 1m). Docs: research/gapper/HYP093_corrected_results.md, updated BACKTEST_BIAS_AUDIT.md, backtester/README.md.
- Commits: 6-part build (data/engine/mc/scanner/corrected-reruns/tests). No live/OANDA/launchd touches.

## 2026-07-17 (cont) — MEGASCAN: 77k-hypothesis search, zero beat the benchmark out of sample
- Largest scan in repo: **77,016 distinct signal hypotheses in 89s** (backtester/daily_engine.py + megascan.py; 111 liquid tickers 2014-2026, families RSI/gap/dip/breakout, per-asset+pooled). Honest FWER = distinct rules (not sizing permutations). Intraday families A/B out of Alpaca minute-quota reach → ran daily-bar families where a real 12mo holdout exists (flagged).
- **0/77k survived Bonferroni on dirty.** 6 beat raw benchmark: NVDA breakout (single-name artifact, rejected) + 5 down-gap-short variants (coherent, survived ex-leveraged → genuine not vol-decay). Prereg'd the best as **HYP-104** (hash a9843721, commit 045bb03).
- **HYP-104 NOT_CONFIRMED on holdout** (ledger 83): dirty +21.0%/Sharpe 2.32 → holdout +2.5%/Sharpe 0.36/p=0.35, n=105. Textbook overfit collapse, exactly as 0/77k Bonferroni warned.
- **Conclusion: nothing beats the gapper fade (~+10-18%/Sharpe~2 post-bias-correction) out of sample.** It remains the only edge in this repo to survive a real holdout. Next gains are execution (borrow/locate TICK-037), not new daily-resolution signals. Report: research/MEGASCAN_2026-07-17_REPORT.md.

## 2026-07-17 (cont) — MEGASCAN v2: intraday/multi-asset, nothing clears the bar
- 5 families, **7,220 configs in 159s** real compute (+~13min fetch: 11,854 Alpaca minute day-files, 2yr hourly BTC/ETH, leveraged+biotech daily). Bias-free minute fills. RAW + FWER reported separately. Added data.get_minute_range paginated bulk fetcher.
- **0 candidates clear raw Sharpe>1.5 + n≥30/yr.** Only Sharpe>1.5 hit = BOIL dip n=11/decade (noise). Per-family: **ORB DEAD** (best Sharpe 0.21, liquid ETFs/megacaps); gap-reversal on 10-50% gaps weak (confirms HYP-101 needs ≥100%); leveraged-ETF reversion + crypto-long real but sub-benchmark beta/regime artifacts; biotech nothing.
- No prereg (bar unmet — mandate's fallback branch). Report research/MEGASCAN_V2_2026-07-17_REPORT.md with top-5 raw + why-they-won't-generalize.
- **Cumulative: 84,236 hypotheses across two megascans, nothing beats the gapper fade OOS.** It stays the only holdout-surviving edge. Note: macOS fork+numpy Pool deadlocks — megascan_v2 runs serially (MEGASCAN_SERIAL=1). No live touches.

## 2026-07-17 (cont) — HYP-105 long-side gapper momentum: CONFIRMED (first new edge of session)
- Tested the MIRROR of HYP-093: ride ≥100% gappers LONG 09:31→10:30 before the fade. 234 minute-ready events, 70/30 date split. Engine extended with trailing-stop + duration exits (7/7 tests still green).
- Phase 2 in-sample: best 09:31/10:30/25%-stop Sharpe 5.07, +64% annual; 0/360 survived Bonferroni (holdout = arbiter). Prereg 334c373d, commit 4d5c387 before holdout.
- **Phase 4 HOLDOUT CONFIRMED (ledger 84): Sharpe 3.63, mean +28.5%/ev, median +7.9%, win 56.3%, perm p=0.0005, n=71.** First OOS-confirmed NEW edge this session — and it's LONG → **no borrow/locate wall (sidesteps TICK-032)**.
- Caveats (heavy): mean is fat-tail-driven (median +7.9% is the honest expectancy); 2.7mo/71-event window; 9:31 microcap slippage understated; NOT independent of HYP-093 (same events, ramp-then-fade halves). Phase-3 ML 92% CV = LOOK-AHEAD LEAK (intraday_push/overnight_gap measured through holding window) — rejected, not predictive.
- Next: forward shadow @09:31, leak-free pre-entry features, realistic first-minute fill model before sizing. Report research/gapper/HYP-105-long-momentum-scan.md.

## 2026-07-17 (cont) — HYP-106 CONFIRMED: leak-free filter catches the big-% runners
- Answered "find the method in the madness": which ≥100% gappers RUN long, predicted from PRE-09:31 info only (09:30 first-min bar + prev_close; every *_1030 feature banned as leak — that's what faked HYP-105's 92% ML).
- **Discovery (leak-free, MW p~0, RF CV 81%): runners = MODERATE overnight gap (+60% vs +163%) + LOW first-minute volume + TIGHT first-min range.** Mechanism: still-building vs exhausted climax (the exhausted ones are HYP-093 short targets). Coherent.
- **HOLDOUT CONFIRMED (ledger 85, prereg 9d1c3937 → 422687d before touch): filter (22 of 71 events) vs unfiltered — median +7.9%→+67.7%, tail_ratio 3.95→10.5, win 56%→86%, P(ret>20%) 38%→77%, Sharpe 4.49, perm p=0.0005.** Survived where HYP-104 collapsed. Positive-skew as requested (10:1). Long → no borrow wall.
- **MAGNITUDE NOT TRADEABLE AS-IS** (flagged hard): +67%/hr median assumes 09:31 microcap fills w/ only entry-bar spread — ignores LULD halts, 5-20% real spreads, size limits. Signal real; live returns a fraction. n=22, fat-tail, not independent of HYP-093/105.
- NEXT: forward shadow @09:31, realistic halt/spread fill model, capacity study before any sizing. Report research/gapper/HYP-106-bigmove-scan.md.

## 2026-07-17 (cont) — HYP-106 realistic fill model + 1/5/10yr answer
- Built backtester/realistic_fills.py: LULD halts (timestamp-gap + >10%/min band detection; entry-halted events SKIPPED as un-enterable), round-trip quoted spread (vol+illiquidity scaled, 3/8/15% scenario caps), size/impact (≤10% of entry-minute volume).
- **HYP-106 SURVIVES all frictions**: full-year base median +51.8% tail 8.1 win 89% Sharpe 8.1; pessimistic +45%; $100k size +44%; skip-entry-halted (31% skipped) still +50.8%/tail 8.9. Holdout base +61.8%, skip-halted +60.7%. Spread/halt dwarfed by +50% moves.
- **Honest ceiling**: printed-OHLC can't prove you get filled at the 09:31 print on a screaming halted thin microcap — only a LIVE FORWARD SHADOW resolves it. +50% = backtest ceiling not expected live number.
- **1yr DONE** (the 234-event year = full minute dataset). **5/10yr IMPOSSIBLE**: minute cache reaches only 2024-01-03 (Alpaca free = 2yr); no 2015-2023 intraday + no gapper universe for those years. Needs paid Polygon/ThetaData minute history + universe rebuild; pipeline (daily_engine + realistic_fills) ready to consume it. Report: research/gapper/HYP-106-realistic-fills-ADDENDUM.md.

## 2026-07-17 (cont) — RETRACTION: HYP-105/106 were LOOK-AHEAD; honest edge is ~10x smaller (HYP-107)
- **Self-caught fatal bug**: HYP-105/106 selected the universe on gain_1030>=100% (price AT 10:30) but ENTERED at 09:31 — hand-picking the stocks we already knew mooned. 234 events = 255 winners of 1475 candidates; the 1220 non-runners were never tested (1 had bars). The "+50-67% median / Sharpe 3.6 / tail 10:1" was survivorship. Realistic-fill model "survived" only because look-ahead +50% dwarfs any cost. **HYP-105/106 REFUTED** (ledger updated).
- **Honest reconstruction**: fetched all 1475 candidates' minute bars; re-selected on 09:31-only overnight gap (incl non-runners). Blind gapper long = median -0.3% (loser = HYP-093 fade mirror). Filter STILL holds OOS (70/30 split, frozen thresholds): holdout n=57 GROSS median +5.4%, win 70%, tail 4.4, perm p=0.0005 → **HYP-107 = real but ~10x smaller signal.**
- **Not tradeable on this evidence**: +5% gross vs 1-15% microcap spread + LULD halts → frictions likely eat it (halt detector over-flags open volatility; net unresolved). Needs live 09:31 shadow. The confirmed direction remains the SHORT fade (HYP-093), entered at 10:30 after the condition is real.
- Lessons: selection must use only entry-time info; a friction model that "survives" a huge edge proves nothing. Report: research/gapper/HYP-105-106-RETRACTION-and-honest-107.md.

## 2026-07-17 (cont) — HYP-107 live shadow set up + session summary
- Built research/gapper/hyp107_shadow.py (--scan 10:50 ET / --close 16:20 ET): logs hypothetical 09:31→10:30 longs on ≥30% gappers passing the frozen HYP-107 filter (overnight_gap≤0.577 AND log10 first-min vol≤5.854), source-tag shadow_hyp107, NO capital. Tracks median return + realized 09:31 spread vs backtest (median +5.4%/win .70/tail 4.4). Target 40 events. Verified: extraction matches enriched og exactly, 24% of ≥30% gappers pass filter → healthy event stream. Writes data/research/gapper/hyp107_shadow/hyp107_tracking.json.
- scripts/com.alta.hyp107_shadow.plist (10:50+16:20 ET, clock-dispatched wrapper), plutil OK, TRACKED-NOT-LOADED (operator: launchctl load). HYP-093 live_shadow.py left pristine (constitutionally sealed).
- research/session-summary-2026-07-17.md: 5 strategy families (Sovereign Carry, The Undertow, ~~Updraft~~, ~~Divining Rod~~ RETRACTED, Storm Dip), the look-ahead retraction cause, HYP-107 honest result, priority order (Undertow shadow → HYP-107 shadow → funded scaling).

## 2026-07-18 — Options-tradeability audit: DOOR CLOSED (2.1%) — call overlay dead on gapper universe
- Gate for the call-overlay strategy (convex/defined-risk/no-borrow expression of the gapper edge). ThetaData options-VALUE, ATM call at 09:31, tradeable=two-sided AND spread≤20% mid.
- **5/234 (2.1%) tradeable → DOOR CLOSED.** 169 NO_OPTIONS (72%, reconciles w/ prior borrow study), 34 no-live-expiry, 18 no-09:31-quote, 8 spread>20%. Best tier $5-20 = 4.7%; $1-5 and $20+ = 0%. The 5: RGC/TMQ/GSIT/BYND/CMBM ($5-12).
- **Key finding: the gapper edge and tradeable options don't coexist in the same names** (thin microcaps). Kills options overlay for THIS universe, not the convexity idea in general — but a liquid-options universe (SPY/AAPL/TSLA…) doesn't carry the gapper edge (megascan showed ORB/breakout dead there). Near-term money path stays: prove confirmed equity edges survive live fills (HYP-107 shadow + real-time execution harness), not an options wrapper.
- Report: research/gapper/options-tradeability-audit.md.

## 2026-07-19 — Risk-layer ml_stack test regressions FIXED (13/13)
- Audit item OPEN_ITEMS #7: 13 genuine failures in `tests/unit/test_ml_stack.py` (Kalman ×3, TradeMDP ×2, LQR ×2, Pegasus ×4, ICA ×2) against live `sovereign/risk/` modules — API drift, not deleted APIs.
- Fixes (source): Kalman `update()`→`KalmanState` (array-like via `.shape`/`__array__`, reconciles a conflicting contract in `test_cs229_ml_stack.py`), `_s` alias, `kalman_t`; TradeMDP `_discretize_state`/`_all_states`; LQR finite-horizon `_L_matrices`+`get_action`; Pegasus `TradingPolicyParams`/`Scenario`/`evaluate_policy`/`build_risk_neutral_scenarios`, optional `reinforce_update` args, running EMA baseline; ICA `max_iter`+`transform_batch`.
- Fixes (test): autouse fixture isolates the four `models/*.pkl` checkpoints (tests were non-hermetic — inherited & mutated shared state); `test_stressed_state_sizes_down` loss corrected −0.01→−1.0R (−0.01 is ~break-even on the R-multiple reward scale, legitimately no size-down).
- Suite: **33→20 failed, 1497→1510 passed** (verified). Remaining 20 are unrelated forex/ict/data-pipeline failures (separate subsystem, out of scope). Zero new failures introduced; ICT isolation test still green.
- Pushed sovereign-v2 (7829614, 8db422f, fb79f96, 45f1b46).
- **Deferred per NN#4 (live sizing, needs logged rationale):** two real TradeMDP feedback-wiring bugs found in passing (`orchestrator.py:1978`) — (a) `pnl` passed as raw price PnL not R-multiples (reward-scale mismatch); (b) `size_multiplier_used=size` passes notional size not the {0.5,0.75,1.0,1.25} multiplier, so all trades collapse to the 1.25 action. See OPEN_ITEMS #7.

## 2026-07-19 (cont) — Obsidian intelligence layer shipped (sovereign/brain)
- Built the bidirectional brain: `sovereign/brain/{obsidian_reader,obsidian_writer,_paths}.py` (+`__init__`). Reader loads long-term vault memory before agents act (`load_recent_verdicts` from auto_hypothesis_results.jsonl+factory_ledger; `load_edge_summary`/graveyard by parsing vault Hypothesis-Ledger.md — 15 confirmed / 27 killed live; plus `load_regime_context`, `load_weakness_log`, `get_morning_context`, `get_research_context`). Writer posts structured knowledge back (`write_verdict`, `write_regime_observation`, `write_weakness_note`, `write_morning_brief`, `write_eod_summary`). Stdlib-only, isolation-clean (no ict/ imports, verified), every reader degrades gracefully on a missing vault (never raises).
- Wired into AGENT_DIRECTIVE.md: morning Step 0 brain-read (regime-change lowers signal confidence) + Step 5 write-back; research Step 0 reads the graveyard so the generator can't re-propose killed families (HYP-090 adaptive-params, HYP-085 news-sniping); EOD Step 2b writes structured lessons + losing-day weakness scan + BRAIN_INDEX refresh. Oracle `reflect_cycle.py` mirrors each candidate lesson to the vault regime log (guarded try/except — a vault write can never break the reflect cycle).
- Vault artifacts: `~/Obsidian/Obsidian/BRAIN_INDEX.md` (one-shot situational-awareness file, regenerated nightly by `scripts/refresh_brain_index.py`) + `Trading Psychology/weakness_log.md` template. Vault repo commit b3b57b9 (local-only, no remote configured — expected).
- Tests: `tests/unit/test_brain_obsidian.py` 5/5 green (write-read roundtrip per writer + missing-vault degradation + ict-isolation invariant). ICT isolation suite still green.
- Pushed sovereign-v2 bbcba21. No live trading params touched (NN#4 respected); the brain is read/write-to-vault only and gates nothing.

## 2026-07-20 — Overfitting safeguards audit + hardening

Audited backtester/ + backtest/ + sovereign/discovery/ against the 5-step research
method. Method was substantially enforced already; gaps were enforcement, not concept.

**Shipped**
- `backtester/holdout_guard.py` (new) — central HOLDOUT_REGISTRY replacing four
  scattered constants (2025-07-17 / 2025-01-01 / 2024-01-01 / 2024-07-01), plus
  `validate_date_range()`. Unbounded ranges count as touching. Access needs
  explicit sanction; every access appended to `data/agent/holdout_access_log.jsonl`.
  Wired into engine.run, scanner.scan, daily_engine.backtest_daily.
  hyp104_holdout sanctions itself after its prereg-hash guard passes.
- `backtester/walk_forward.py` (new) — row-based rolling WF for the backtester
  stack, which previously had a SINGLE date split only. Test slices tile
  contiguously; reports `param_stability.refit_churn`.
- Fixed `backtest/walk_forward.py` EXPANDING bug — train_end never advanced, so
  every "expanding" window was identical (silent single-split degradation).
- `engine._check_costs()` — WARNs on zeroed slippage, unmodelled commission,
  shorts without locate gate; surfaced as `result["cost_warnings"]`.

**No parameter values changed** — registry dates mirror the constants already in
megascan/yield_frontier, so no param_change_log entry is due. Changing one later
does require one.

**Verdicts** — `tests/test_overfitting_safeguards.py` 18/18 new, test_backtester
7/7. Full suite 1533 passed / 20 failed; all 20 pre-existing ICT/forex failures,
confirmed by stashing every change and reproducing them identically (those files
don't import backtester). ICT/sovereign isolation test passing.

**Refused to shortcut** — did NOT wire `realistic_fills` (measured TICK-039 spread)
into `engine.py`. That is the top remaining gap: backtest and live harness charge
different spreads, so reconciliation compares two cost worlds. Fixing it re-prices
every gapper result and needs its own ticket + re-baseline, not a drive-by edit.
Other open gaps: commission unmodelled, no cross-script n_trials budget, the four
legacy forex-side holdout constants unmigrated, and walk_forward_backtest has no
adoption yet (first candidate: re-run a megascan survivor under rolling windows).

Full writeup: `~/Obsidian/Obsidian/System/overfitting_safeguards.md`

## 2026-07-20 — TICK-043: market data adapter layer (vendor-agnostic seam)
- **unlock: migrate backtester/engine.py and execution/harness.py to MarketDataAdapter for TICK-043** (Colin, this session). Scope of the unlock as exercised: bar-fetch *transport* only. No fill/pricing/threshold logic touched; `execution/config.py` FROZEN_HASH verified unchanged at `66907c79…0612a3`.
- **Audit**: 7 vendors live (yfinance 196 files, alpaca 73, polygon 33, thetadata 18, databento 11, fredapi 10, alpha_vantage 3). Nothing was abstracted — but three *competing* part-adapters already existed (`data/providers.py`, `sovereign/data/feeds/alpaca_feed.py` incl. its own parquet cache, `data/alpaca_client.py`). Built the seam by consolidating those rather than adding a fourth.
- **Shipped**: `sovereign/data/adapter.py` (`MarketDataAdapter`: get_bars / get_snapshot / get_top_movers / get_options_chain; `DATA_PRIMARY`+`DATA_FALLBACK` from env, primary failure logged then fallback; Alpaca/Polygon/yfinance backends wrapping the existing clients, lazily constructed so a missing key can't break an unrelated caller; central `_normalise` for the Alpaca-MultiIndex / yfinance-(Field,Symbol) / Polygon-short-key variations that have each bitten this repo). `sovereign/data/cache.py` (`DataCache.get_or_fetch`, symbol-day parquet, historical immutable / today stale until after close, corrupt→refetch, empty never cached, hit/miss counters). `tests/unit/test_data_adapter.py` **19/19**.
- **Wiring, behaviour-preserving by construction**: `backtester/data.py` — adapter added at the documented "extend here when a live source is wired" gap, i.e. *after* parquet→gz→direct-Alpaca. Every day already covered by those paths returns byte-identical bars, so the adapter can only add coverage where an EMPTY frame was returned before; it can never restate measured history. `execution/harness.py` — `_adapter_minute_bars` consulted **only when `alpaca.minute_bars` returns nothing**, and it *reapplies* `sip_ceiling(17)` rather than inheriting it (a fallback quietly serving real-time bars would manufacture look-ahead in the live shadow — same failure class as the HYP-105/106 retraction).
- **Verification**: full suite 1551 passed / 21 failed; the 21 are the pre-existing ict/forex/data-pipeline set — confirmed by stash-and-rerun on the affected subset (identical 15 failures with and without the change). **Zero new failures.** ICT isolation test green; adapter+cache import nothing from `ict/` (asserted in test).
- **12 files flagged** `# TODO: migrate to MarketDataAdapter (TICK-043)`; all 12 re-parsed clean. ThetaData / FRED / Alpha Vantage / databento deliberately left out — different auth and shape, `get_bars` is the wrong interface for them.
- **Step 6 of the request NOT executed — reported instead of performed.** The ask was to unify backtest and live fill pricing and note the resulting baseline shift. Declined on evidence: `execution/quotes.py` (captured bid/ask) and `backtester/realistic_fills.py` (model) are two independent pricing paths *by design*, and their difference is the `vs_backtest_delta` the harness exists to measure (harness.py:19-28, and the exit-bar comment at :265 explicitly warns that mispricing one side makes the delta meaningless). Unifying them drives that delta to zero by construction — it deletes the measurement rather than improving it — while re-baselining the v015 0.6886 anchor that 15 confirmed / 27 killed hypotheses are keyed to (the TICK-024 failure mode). **No baseline shift occurred and none should be recorded.** If a pricing-consistency change is genuinely wanted it needs its own ticket, a `param_change_log.jsonl` rationale (NN#4), and a re-verdict pass over the affected ledger entries — not a side effect of a transport migration.
- Doc: `~/Obsidian/Obsidian/System/data_sources.md` (vendors, rate limits — Polygon free 5/min and Alpha Vantage free 25/day are the binding ones — key names, cache rules, un-migrated surface, and the pricing-path boundary).

## 2026-07-20 — Full functionality audit (what's built but not doing anything)

Audited 37 installed LaunchAgents, 28 `sovereign/` subpackages, 30 top-level dirs against
four tests: scheduled? produces output? wired to a live decision? output non-trivial?
Commit `ac0a99e`, pushed to `origin/sovereign-v2`.

**Headline** — the scheduler layer is healthy (34/37 plists fire on time, exit 0, and every
installed plist is byte-identical to its `scripts/` copy). The losses are in wiring, not
scheduling. Biggest systemic risk found: several jobs **exit 0 while silently degrading** —
`forex.scan` falls back to the degraded sentinel on an empty yfinance frame, `cache.refresh`
reports success while Reddit 403s to 0 posts every run. Exit code is not a health signal here.

**Fixed + verified**
- **The autonomous Claude-agent stack had never executed once.** `morning_agent`, `eod_agent`,
  `research_agent` all pointed at `/usr/local/bin/claude`, which does not exist on this machine;
  `morning_agent` had been returning **exit 127 every weekday**. Repointed to
  `~/.local/bin/claude` and confirmed the binary resolves under the plists' own restricted PATH.
- **Oracle was reflecting on two-month-old forensics.** `trade_forensic_engine.py` was only ever
  a manual CLI — never scheduled — so the three files `validation_cycle._load_all_forensics()`
  reads were frozen at 05-19 while `oracle.reflect` ran nightly against them. New
  `com.alta.forensics` at 02:00 ET, deliberately ahead of reflect at 02:30 so each pass reads
  same-night data. Verified by `launchctl kickstart`: exit 0, all three outputs now 07-20.
- **`gapper_shadow_scan` was exiting 1.** One exhausted Alpaca fetch raised out of `bars_for`
  and killed the whole 50-symbol scan. Added `get_or_none` so a dead ticker degrades that symbol
  only, printing `FETCH_FAILED` to stderr rather than swallowing. Both call sites already handled
  empty results, so behaviour-preserving. Verified live: 1 signal from 7 verified movers.
- **Borrow feed restored** — installed `com.alta.ib_shortable`, which was written but never
  installed; its ftp2+ftplib path works where `daily_snapshots.py`'s ftp3+urlopen leg has failed
  5 days running. 750 EASY / 32 HARD written.
- **`stray_tripwire`** given an hourly `StartInterval`; `WatchPaths`-only meant no fire since 06-16.

**Blocker for Colin — credential, not code**
- `oracle.session_close` gets **OANDA 401 Insufficient authorization**. This is the Oracle
  *outcome* channel (NN#2): the close pass cannot read positions, so trades can close without
  `update_outcome()`. The practice token in `.env` looks rotated/expired. I did not touch it.

**Refused to shortcut** — did NOT fix the two most tempting items. `forex.scan`'s `GBPUSD=X`
yfinance degradation (fix = repoint to the OANDA candle fetcher) and the four null fields in live
ICT decision records (`vix_at_entry`, `rate_differential_zscore`, `cot_percentile`, `library_match`
— all null despite fresh COT and VIX data) are both **decision/data-path edits under the
shadow-execution freeze**. Each wants its own ticket and unlock, not a drive-by during an audit.

**Other open findings** — `data/risk/risk_decisions.jsonl` has zero readers repo-wide;
`propfirm/active_challenge.json` is 05-18 but `allocation_engine.py` reads it on the live path;
`data/execution/fills.jsonl` (06-30) vs `oanda_fills.json` (07-20) are two fill paths and one is
dead; `clawd.ny_am_scanner` has 0-byte logs since 05-30 and no heartbeat, so it cannot be
confirmed to do any work; `evening_prep`'s health check dies on a `TypeError` because live code
imports from `archive/`. `sovereign/orchestrator.py` is the sole importer of five subpackages
(`router`, `specialists`, `kimi`, `prediction`, `strategies`) and **no plist invokes it**.

**Attic candidate, deliberately not executed** — ~13 dirs of Mar–May v4/clawd-era code form one
self-referential cluster. Must move as a unit: `layer1/`, `layer2/`, `contracts/` are hard-imported
at module top level by `sovereign/specialists/base_specialist.py` and `sovereign/risk/kelly_engine.py`,
so atticking them piecemeal breaks `sovereign/orchestrator.py` at import. Wants its own ticket.

Full table + evidence: `~/Obsidian/Obsidian/System/full_functionality_audit.md`

## Reddit retirement — landed in ddec713, not its own commit

A parallel session ran `git commit` between my `git add` and my `git commit`. My four
staged files were swept into **ddec713 "[INFRA] Log 2026-07-20 anatomy audit session
in NEXT.md"**, whose message describes none of them. The code is intact and verified
(16/16 tests, health GREEN); only the attribution is wrong.

What is actually in ddec713 beyond the NEXT.md log:

- `execution/context.py` — `Status.RETIRED`; `load_reddit` returns it carrying the
  reason; retired sources leave the health denominator (`n_retired`/`retired`
  reported separately). Health reads an honest 3/6.
- `scripts/health_check.py` — RETIRED short-circuits before measurement, counts as
  neither bad nor warning. system_health RED -> GREEN; reddit was the only job that
  changed.
- `scripts/refresh_caches.py` — stops calling the scraper. It burned three retries
  per subreddit to fail, then rewrote the cache with `posts_scanned=0`, refreshing
  the mtime so staleness checks read green over an empty payload.
- `tests/unit/test_execution_context.py` — the degrade-never-fabricate contract was
  exercised *through* `load_reddit`, so retiring it would have silently dropped five
  tests' coverage. Retargeted to `load_fred` (still produces all five statuses); two
  new tests pin the retirement itself.

Evidence for retiring rather than fixing: tested live with a browser User-Agent,
`old.reddit.com` **403** and `www.reddit.com` **403**, and there is no
`REDDIT_CLIENT_ID` anywhere in the repo. The endpoint fix in 9bc2849 was correct code
against a shut door. `sovereign/data/reddit_scraper.py` is deliberately left on disk
as the starting point if an OAuth app is ever registered.

**Operator action outstanding:** `com.sovereign.reddit_sentiment` is still loaded and
will keep 403ing on schedule. Unloading it is an install action.

**Process note — third occurrence today.** Explicit-path staging protects the *content*
of a commit but not its *timing*: between `add` and `commit` another session can claim
the index. Earlier today the same race left phantom deletions of `RISK_FRAMEWORK.md`,
the watchdog baseline, and `backlog.md` staged, and produced stale `index.lock` and
`HEAD.lock` files. Sequencing `git add && git commit` as one command would have
prevented this one.

## 2026-07-20 · RESEARCH PASS (21:00 ET nightly)
Brain read clean (27 killed hyps in graveyard; confirmed edges loaded; no sealed idea re-proposed).
Queue: 5 tasks (`--max 5`) — 4 Colin-gated wiring no-ops (RQ-REST-006/007/008/015), RQ-REST-016 (CB_MEETINGS back-extend) threw its standing `EXCEPTION: s argument must not be None` (blocked on operator sign-off per FIND-REST-037-a; `code_change` cannot self-apply per NN#4). No autonomous micro-backtest ran — queue holds only operator-gated wiring.
Candidates flagged for operator: **none** — nothing cleared a first-pass permutation test because nothing was tested. Zero-candidate night, logged per STANDING RULE 5.
Movers snapshot: 50 gainers → `data/research/gapper/movers_recent.json` for tomorrow's scan (single snapshot, no lookback). ZYBT +1047% @ $8.01 leads; warrants (RNWWW/FGIWW/IVDAW/ACHR.WS) excluded by filters; non-warrant ≥40%: ZYBT/MF/ADVB.
No live parameters touched. No incident.

## 2026-07-21 · RESTORATION CAMPAIGN session 1 (TICK-049) — Phase 0 + Phase 1 loops

Executed session 1 of the multi-session restoration campaign. Diagnose-and-ledger pass;
no execution-path code touched (shadow freeze holds to 07-28). Ledger:
`plans/restoration-ledger.md` (force-added; plans/ is gitignored).

**Corrected 2 stale campaign premises before acting (its own NN#1):**
- Phase 0 ("fix claim_check LOGPATH parser FIRST") was already done in `aab90eb` this session —
  collapsed to a verification gate (self-test green, regression tests present).
- "research_agent.log dark since 05-16" — false; it fired 07-20 21:02 on schedule.

**Phase 1 findings (all 5 loops resolved or ticketed):**
- **Oracle close-loop reframed (TICK-050).** The null contract fields are explicit `None` with
  documented reasons at `forex_specialist.py:118-131`, not a silent leak. `commitment_score`
  (ICT concept, N/A to forex), `vix_at_entry` (VIX gate dead), `cot_percentile` (z-score not
  percentile) are null by design; only `library_match` and upstream `irp_z` are real gaps.
  Outcome loop is HEALTHY (73 → 56 EXPIRED / 5 closed / 12 open). `forex_specialist.py` is
  frozen-adjacent (2 live importers) → fix waits for the 07-28 unlock. Diagnose-only per ruling.
- **dashboard_state.json is a phantom (fixed + TICK-051).** No writer, no code reader; dated
  05-31. Corrected the false "sync writes it" claim in `AGENT_DIRECTIVE.md:104`. Did NOT write an
  empty file — that is the green-but-empty trap one level up. TICK-051 to delete + purge mentions.
- **dashboard "missing inputs" reframed (TICK-052).** `dashboard/index.html` fetches two absent
  files, but `deploy.yml` overwrites the served root with repo-root `index.html`, so `dashboard/`
  is the orphaned legacy view. Not a live break.

**Refused to shortcut:** did not "fix" the decision-logger nulls (frozen-path); did not write a
placeholder dashboard_state.json to make a check go green; did not touch the shadow window.

**Pre-existing failures named, not absorbed:** `ict and pipeline` = 4 failed / 23 passed on the
committed branch (campaign's "21/21" is already below that); 15 failed under broader `-k ict`.
This session added zero failures — after-counts identical to before. Isolation guard PASS.

Pushed. Next session: Phase 2 (28 DORMANT) or act on TICK-050/051/052 post-unlock. Re-verify
every open ledger row against the filesystem first — the stale-premise pattern recurs.

### 2026-07-21 · RESTORATION continued — Phase 2 (28 DORMANT) gate met
Continued the campaign as a standing command. Phase 2: all 28 DORMANT resolved with evidence
(claim_check per file) — 15 KEEP (7 mislabelled tests + 8 package __init__), 9 KEEP-DORMANT
(clawd_trading incomplete cluster, firebase ghost, strategy_selector scaffold, gauntlet diag),
4 RETIRE (TICK-053: check_account, fix_hardcoded_equity, debug_indexing, TODO_COMPLETE_IMPL).
Inventory bug found: test-root detection is top-level-only, mislabelling nested tests DORMANT
(TICK-054). Decisions in plans/restoration-ledger.md; RETIRE moves ticketed not executed (>3-files
discipline). No code touched; isolation guard still PASS. Next: Phase 3 (46 RETIRED triage).

### 2026-07-21 · RESTORATION Phase 3 (46 RETIRED triaged) — gate met
All 46 RETIRED are engineering-retired (the killed edges live in the ledger, not these files).
44 stay retired: superseded clawd/Firebase/Kimi/XGBoost generation (attic ×31), scratch one-offs
(×9), old scheduler (archive), Firebase pusher. Shortlist = 1 genuine + 1 dependent, NOT padded
to 5: lab/feature_registry.py (a GRAVEYARD registry — the antidote to HYP-044-reproposed-3x — but
partially covered by hypothesis_ledger.json/audit_live_features, so it's a consolidation proposal)
+ lab/baseline_registry.py (dependent). Nothing resurrected. Phases 0-3 done this session;
Phase 4 (spot-check 20 on-demand tools) + Phase 5 (vault Sharpe purge, relink 56 orphans) next.

### 2026-07-21 · RESTORATION Phase 4 + CLAUDE.md baseline fix
Fixed CLAUDE.md:120 "must stay 21/21" → real baseline (4 failed/23 passed, `ict and pipeline`),
the load-first re-injection source (same mechanism as HYP-044). Phase 4: 20 consequential
ON-DEMAND validation tools probed at import/load layer — 20/20 clean, no bitrot (3 initial
false-positives from probe method caught + corrected). Full live-data run deferred (infra).
Phases 0-4 done. Remaining: Phase 5 (vault Sharpe purge + relink 56 orphans), TICK-050/051/052
(post-07-28 unlock), TICK-053/054. OUTSTANDING (operator hand): obsidian_sync plist not yet
installed — 6 scheduled tasks correctly report BLOCKED until `launchctl load`.

### 2026-07-21 · Petrules Gate Phase 0 — data audit COMPLETE, verdict NARROW
Executed Phase 0 of `research/PETRULES_GATE_SPEC.md` (data availability audit, no model code).
Deliverable: `research/PETRULES_GATE_data_audit.md` + probe evidence in `research/petrules_audit/`
(commit c42a0de, pushed). Verdict: NARROW. Verified by live query: AV EARNINGS = point-in-time
pre-print consensus back to 1996 (surprise label OK) but NO revision path anywhere free →
consensus_revision_momentum DROPPED (lookahead risk); congressional_trade_direction DROPPED
(mirrors 403-dead, official = PDF parsing, ~500 PTR filings/yr); Form 4 lead vs earnings
MEASURED n=791 clusters, median 56d, 100% ≥1d pre-event (KEEP); 13F staleness 44-45d (KEEP,
demoted); 13D ~5.5k/yr (KEEP; 2025 files as "SCHEDULE 13D" new form name); ThetaData Value tier
options history cutoff pinned at 2020-01 → training window 2020+ unless STANDARD upgrade.
Biggest risk: revision-path features would have been a silent lookahead leak into a conviction
engine. Next: file PETRULES_GATE_prereq.json with the narrowed feature set, then Phase 1 replay engine.

### 2026-07-21 · RESEARCH PASS (21:00 routine, AGENT_DIRECTIVE.md)
Brain read OK (27 graveyard hyps loaded, passed to hypothesis context — no re-proposals).
Movers: 50 gainers → data/research/gapper/movers_recent.json.
Queue: 5 tasks drained (--max 5), all OK — 4 no-ops (operator-decision items RQ-REST-019b/033/005/009);
RQ-REST-017 cb_decisions audit: 1201 decisions, 71 surprises ≥25bp.
Candidates flagged for operator: NONE — no pattern cleared first-pass permutation test tonight.
Noted: recurring BLOCKED_NO_VALIDATOR on HYP-AUTO entries (07-20/21) — pre-existing, needs triage.
Note: a parallel session wrote an earlier 2026-07-21 pattern entry; both retained (append-only).

### 2026-07-22 · Petrules prereg v1.1 (merged, re-locked) + Template B scaffold
Reconciled the two competing preregs (my hash-locked v1 vs dispatch's DRAFT) with Colin's approval:
merged DRAFT's carried caveats (THINNED DIFFERENTIATOR, REGIME-LIMITED CLAIM), feature notes, legal
boundary, stricter CALIBRATED rule, and added disclosed_flow_form4_cluster → Tier 1 now 6 features.
Re-hashed as v1.1 (14a18cb4…, self-verify green, supersedes 085b83ea… recorded in-file); DRAFT
deleted. Legitimate re-lock: zero training data constructed. Template B scaffolded at repo root
(TEMPLATE_B_MINIMUM_VIABLE_INCOME.md): chain-to-payout per edge, 5 Colin-only decisions blanked,
capital→$/mo worksheet — honest reading: MVI is capital-bound (~$1,665/mo at $100k). launchctl
sanity: paper_accounts/system_regime/system_health_verdict/obsidian_sync all loaded, exit 0.

## 2026-07-22 — Always-available standalone dashboard (INFRA)
- **Shipped** `scripts/build_standalone_dashboard.py`: reads the SAME allowlist as `serve_dashboard.sh`, inlines JSON into `dashboard/dashboard_live.html`, injects a `fetch()` shim so the unmodified dashboard renders under `file://` (no server). Hard secret guards: name + content sweep abort before write; final HTML re-swept.
- **Shipped** `scripts/com.alta.dashboard_refresh.plist`: regenerates the file every 15min (`StartInterval` 900, `RunAtLoad`). NOT auto-loaded (sandbox) — Colin runs `launchctl load`.
- **Verified** rendered under `file://` in real browser: prop/carry/oracle/health/fills/gates all populate. Secret grep CLEAN.
- `dashboard/dashboard_live.html` gitignored (regenerated, data baked in). Pushed 8e4c587 on sovereign-v2.
- Untouched: execution path, `serve_dashboard.sh`, all frozen files. New files only.

## 2026-07-23 — Fix bias-panel confidence double-multiply (FIX)
- **Root cause**: `data/bias/bias_*.json` stores `confidence` as a percentage (e.g. 21.0), but `updateBias()` in `dashboard/index.html:720` did `Math.round(conf*100)` → "2100%".
- **Fix**: treat conf as already-a-percentage, guard the legacy fraction convention (`conf<=1 → ×100`), clamp 0–100. Verified in real browser: today (07-23) reads 0%; test vector 21.0→21, 0.21→21, 42→42, 150→100. Display-only.
- Regenerated `dashboard_live.html` so the fix shows in the standalone build too. Pushed 709c114 on sovereign-v2. No execution-path files touched.

## 2026-07-23 — AlphaZero/Stockfish activation (WORKSTREAM A wired LIVE, B computed) — freeze-safe
Implements `research/ALPHAZERO_STOCKFISH_REPORT.md` as far as the execution-path freeze (runs to
2026-07-28) allows. **No frozen file touched** (forex_exit_manager, decide_exit, exit_machine,
carry_engine, execution/harness.py, ict scoring, L2 shadow all untouched). Named isolation test
`test_pipeline_does_not_import_sovereign` GREEN; ICT baseline unchanged (4 failed/23 passed).

### WORKSTREAM A — briefing synthesizer (AlphaZero)
- **A1 (LIVE, code-complete)** — `scripts/morning_market_briefing.py::build()` now publishes the
  agent-facing contract `data/agent/daily_briefing.json` every run, idempotent, fail-loud (partial
  writes surface as `contract_write_errors` in the briefing). This is the "call it every morning +
  write daily_briefing.json" wiring. Verified: contract written this run (deterministic fallback).
- **A2 (BUILT + STAGED)** — new `sovereign/briefing/briefing_context.py`: a CONTINUOUS sizing
  multiplier from confidence + regime quality + optional direction-vs-carry alignment. Band
  [0.80,1.20], **fail-to-neutral (=1.0)** on deterministic fallback / missing confidence — NEVER a
  veto (strictly positive, can't zero a trade). Published each run to
  `data/agent/briefing_multiplier.json`. Consumer note: the **Petrules Gate conviction scorer does
  not exist yet** (Phase-1 groundwork only — `research/petrules/`), so the multiplier is published on
  a clean contract + a ready consumer hook rather than wired into nonexistent code. **Carry hookup is
  STAGED, unwired**: `briefing_context.staged_carry_size_multiplier()` is the one-line switch to flip
  after 2026-07-28 (editing carry_engine is frozen). 9 unit tests green (`tests/test_briefing_context.py`).
- **A3 (LIVE)** — (1) every morning's call appended to `data/agent/briefing_log.jsonl` (append-only).
  (2) `decision_logger._with_briefing()` auto-stamps the L1 call (verified=False, context_only) into
  `present_state_snapshot.l1_briefing` on EVERY forex + ICT decision at the logging choke point — so
  trade outcomes can later be matched back to the briefing that preceded them. (3) The self-improving
  loop is already turning: `scorecard.summary_line()` (n=14 dir-calls, hit 0.786) feeds back into the
  next synthesizer call. Honest n; still CALIBRATING (<30 samples).
- **BLOCKER (honest, not a code fault)**: real Opus synthesis returns HTTP 400 *"credit balance too
  low"* — key valid, model id `claude-opus-4-8` accepted, account out of credits. Until topped up the
  system runs deterministic_fallback (bias NEUTRAL, conf 0) and the multiplier correctly neutralises
  to 1.0. Wiring is LIVE and correct; it will produce real calls the moment credit is restored.

### WORKSTREAM B — HYP-071 exit value table (Stockfish) — COMPUTED + VALIDATED
- Ran `scripts/research/hyp_071_exit_value_function.py`: both prereg hashes verified, reconcile gate
  **0.6886 exact**, re-trace parity **459/459**. Table computed (108 cells, 10k continuations, L=5).
- **PROVISIONAL PASS** against the locked §7 gate — **10** CPCV-stable economically-sensible EXIT_NOW
  divergences, **9** forward-consistent across 2023-24↔2025-26. Forward agree 0.870 (n=23),
  separability 0.862 (n=29), regime-window 0.854 (n=48 — **below the 0.90 "robust" bar**). This
  **contradicts** the pre-registered NOT_SIGNIFICANT expectation.
- **Not sealed as CONFIRMED.** Dominant caveat: R is GROSS; all 9 divergences are carry-ALIGNED cells,
  and correctly-modelled carry (known ~10× mis-modelled, TICK-024) is a reason to HOLD longer → could
  ERASE the PASS. Correct next step (freeze-independent, runnable now): recompute on NET returns and
  see how many of the 9 survive. Left for Colin to adjudicate the ledger.
- Report: `research/HYP-071_VALIDATION_REPORT.md`. Results JSON:
  `data/research/HYP-071_tabular_exit_value_results.json`.

### STAGED for 2026-07-28 unlock (checklist Colin can action)
1. [ ] Confirm L2 shadow exit-machine window passed clean and went live (go-date 2026-07-28).
2. [ ] Recompute HYP-071 table on NET returns (corrected TICK-024 financing); verify the 9
       forward-consistent EXIT_NOW divergences survive AND regime-window agreement ≥ 0.90.
3. [ ] If they survive: write CONFIRMED (or APPROVED-FOR-APPLICATION) HYP-071 entry to
       `data/agent/hypothesis_ledger.json`, then apply the staged cells via a NEW
       `config/exit_value_overrides.yml` read by the exit machine (logged in param_change_log.jsonl).
       Full spec: `research/HYP-071_STAGED_EXIT_RULE_PROPOSAL.md`.
4. [ ] Flip A2 carry hookup: call `briefing_context.staged_carry_size_multiplier(carry_direction=...)`
       from the carry sizing engine (one line), record the unlock here + in param_change_log.jsonl.
5. [ ] Top up Anthropic API credit so the synthesizer produces real (non-fallback) morning calls.

## 2026-07-24 — Ollama local synthesis (3-tier chain) + phased DIP relocation (INFRA)
Removes the "credit balance too low" blocker by making the briefing synthesizer run on a FREE local
model first. Freeze-safe (no execution-path file); isolation `test_pipeline_does_not_import_sovereign`
GREEN; append-only; explicit git add.

### PART 1 — three-tier fallback in `sovereign/briefing/synthesize.py`
- Tier 1 = local **Ollama** (tried first, always): `ollama.generate(model, prompt, format="json",
  options={temperature:0.2, num_predict:1024})`. Model select via `ollama.list()`: **qwen2.5**, else
  **llama3.1**, else None → next tier. On success: `model="ollama/qwen2.5"`, `cost_usd=0.0`, logged via
  `_log_cost(..., model="ollama/qwen2.5")`. **NEVER retries** — any error → next tier immediately.
- Tier 2 = Anthropic Opus, UNCHANGED, only if Ollama returned None AND `ANTHROPIC_API_KEY` present.
- Tier 3 = None → orchestrator writes deterministic fallback (unchanged).
- `synthesis_source` reflects the tier: `ollama/qwen2.5` | `claude-opus-4-8` | `deterministic_fallback`.
- Prompt, `_parse()`, schema, downstream consumers untouched. `format="json"` on every Ollama call.
- Tests: `tests/test_synthesize_tiers.py` (4) — order + short-circuit + key gating — GREEN.

### PART 2 — phased pipeline `scripts/daily_intelligence_pipeline.py` (NEW)
- **Reconciliation (stated honestly)**: there was NO `daily_intelligence_pipeline.py`. The DIP compute
  half lives in `scripts/dip_daily.sh` (harvest→XGBoost retrain); the briefing half in
  `morning_market_briefing.build()`. This new file SEQUENCES those into explicit phases; it does not
  duplicate them.
- **Phase 1** (`--phase 1`): fetch 5 collectors → write raw JSONs to `data/briefing/`. **NO synthesis**
  (checkpoint records `synthesis_called=false`). Verified: 5/5 written, no synth.
- **Phase 2** (`--phase 2`): 2a/2b retrain delegated to dip_daily.sh (OFF by default, `--with-retrain`);
  **2c synthesis** → `morning_market_briefing.build()` writes `data/agent/daily_briefing.json`
  (Ollama-first; deterministic fallback if all tiers None; NEVER blocks the phase); **2d** hypothesis
  batch `hypothesis_generator.run()` receives the briefing as context-only.
- **DoD verified**: `python3 scripts/daily_intelligence_pipeline.py --phase 2` produces
  `daily_briefing.json` with **`synthesis_source: ollama/qwen2.5`** (bias SHORT, conf 65). A2 multiplier
  now lights up real (0.9405, effect applied) instead of neutral. Cost log records ollama @ $0.0.

### Ollama host status — INSTALLED + WORKING on this host (not a Colin to-do)
- `ollama` binary already present (v0.22.0). **This session installed** the python module
  (`python3 -m pip install ollama --break-system-packages` → ollama 0.6.2) and **pulled qwen2.5**
  (`ollama pull qwen2.5` → 4.7GB, `ollama list` confirms `qwen2.5:latest`). End-to-end verified live.
- If reproducing on a fresh host: `brew install ollama`; `ollama serve` (background);
  `ollama pull qwen2.5`; `python3 -m pip install ollama --break-system-packages`; confirm `ollama list`.
- NOTE for scheduled/launchd runs: ensure `ollama serve` is running (a login agent or `brew services
  start ollama`), else Tier 1 silently falls through to Tier 2/3 (by design, but you lose the free path).
