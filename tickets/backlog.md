# Backlog (repo-native ticket fallback)

Used when Linear MCP isn't connected in a session. Same schema either way — switching to real
Linear later is a port, not a rewrite.

Schema per ticket:
- id: TICK-XXX
- title:
- description:
- depends_on: [ticket ids]
- blocks: [ticket ids]
- acceptance_criteria: [bullet list]
- status: backlog | ready | in_progress | in_review | done
- pre_approved: true | false   # true = plan may proceed straight to build without a stop

---

## TICK-001
**title:** Fix `vrp_schema_verify.py` expiration-selection probe bug
**description:** `scripts/vrp_schema_verify.py:89` selects the *median* of the full 2012→present expiration list (an already-expired contract, ~2019) then queries a hardcoded `2022-03-07` EOD at line 92 → ThetaData returns HTTP `472 NO_DATA`. This makes the runbook's "one API touch" schema check falsely fail even though entitlement is fine (proven 2026-07-02: SPY + FXE `list/expirations` HTTP 200; FXE EOD `2022-03-18` on `2022-03-07` returns a full chain). Fix: pick an expiration `>= start_date` (nearest listed expiration on/after the probe date) instead of the median.
**depends_on:** []
**blocks:** [TICK-002]
**acceptance_criteria:**
- [x] `python3 scripts/vrp_schema_verify.py --symbol FXE` exits 0 and prints the loader-contract columns (2026-07-02: probe expiration 2022-03-11, MATCH True)
- [x] `--symbol SPY` also passes (probe 2022-03-07, MATCH True)
- [x] Expiration chosen for the EOD probe is >= the probe date (nearest listed on/after)
- [x] No change to `THETADATA_BASE_URL` handling or the local-no-auth path
**status:** done (2026-07-02, pre_approved build)
**pre_approved:** true

## TICK-002
**title:** VRP Stage 2/3 — fill `ThetaDataLoader` bodies and run IS/OOS on real FXE chains
**description:** ThetaData gateway is live on `127.0.0.1:25503` and the Options **Value** tier serves FXE option chains ($0 ask, confirmed 2026-07-02). Per `docs/vrp_activation.md` §3–6: fill the 4 `ThetaDataLoader` methods marked `# VERIFY SCHEMA AGAINST LIVE RESPONSE` (`sovereign/research/vrp/data_loader.py`), keeping the frozen `OPTION_CHAIN_COLUMNS` contract, then run `validate_vrp.py --stage 2` (IS sanity) and `--stage 3` (OOS — the number that matters). NOTE: this session's memory (`project_vrp.md`) says the bodies were filled 2026-06-16 but the runbook still lists them as TODO — **verify the live state of `data_loader.py` on `sovereign-v2` first**; the fill may already exist. Holdout (`--stage 4`) is touch-once and stays out of scope until OOS is reviewed by Colin. VRP isolation (`test_vrp_isolation.py`) and the signed pre-registration (`vrp_sign_prereg.py --check`) are non-negotiable.
**depends_on:** [TICK-001]
**blocks:** []
**acceptance_criteria:**
- [ ] `data_loader.py` loader bodies confirmed filled + `pytest tests/unit/test_vrp_options_backtest.py -q` green
- [ ] `vrp_sign_prereg.py --check` → OK before any run
- [ ] `validate_vrp.py --stage 2` completes with a plausible IS report (not NO_TRADES for the wrong reason — the 2026-06-16 NO_TRADES was a real $100k-account sizing finding; re-check account-size decision in NEXT.md before interpreting)
- [ ] OOS (`--stage 3`) result logged to `hypothesis_ledger.json` (VRP-001-OPTIONS); no param re-optimization after seeing it
- [ ] Verdict written to repo-root `NEXT.md`; Colin reviews before any deploy
**status:** backlog
**pre_approved:** false

## TICK-003
**title:** Options-leg family run — VRP-001, HYP-074/075/076/078/079 + HYP-077 full composite, then family BH
**description:** Real surface data landed 2026-07-02 (sentiment_options_surface: 1,306 rows, 2020-01-03→2026-07-02, rr25/bf25 96.5% of post-2020 board rows, 0 look-ahead violations, all bs_invert). Extend scripts/research/run_positioning_family.py with the six options-leg members under the SAME locked protocol (gate-zero hash verify; rr25_z/bf25_z = trailing-252-obs z on the weekly surface series; HYP-076 needs econ_surprise_z×crowding; HYP-080 stays blocked on the GDELT backfill — family BH cannot run until 080 has a primary p or the family documents its handling). Re-run HYP-077 with the FULL composite (COT + rr25 alignment) superseding the COT-only interim seal. VRP-001 first per the standing V4 mandate: TICK-002's stage 2/3 on real chains (verify loader bodies, prereg check, account-size note in NEXT.md before interpreting NO_TRADES). Coverage caveat to stamp on every seal: options history starts 2020-01-03 (Value-tier depth) — six years, not the decade.
**depends_on:** [TICK-002]
**blocks:** []
**acceptance_criteria:**
- [ ] rr25_z/bf25_z computed trailing-only (truncation-invariance test, same standard as cot features)
- [ ] Six new primaries sealed as dated interim annotations (or UNDERPOWERED/BLOCKED stamps where data forbids), statuses stay PREREGISTERED
- [ ] HYP-077 full-composite seal supersedes the COT-only interim (both annotations remain)
- [ ] Family BH runs ONLY when all 10 primaries exist; otherwise the blocker (HYP-080/GDELT) is stamped
- [ ] VRP-001 stage 2/3 per TICK-002; no param re-optimization after seeing OOS
**status:** ready
**pre_approved:** false

## TICK-004
**title:** Standing Adversarial Invariant Guard — close the weak (Layer-4) correctness layer
**description:** A 4-layer audit rated Adversarial WEAK: RED-1 Oracle contamination + the rogue USD_CAD writer were caught only by manual audit, no standing detector/test. Built `audit/invariant_guard.py` (read-only, spec-first, self-escalating; I1 Oracle-reflection purity, I2 no rogue/sentinel OANDA writes, I3 forbidden-pair guard), `audit/invariants_spec.md` (hashed single-fence contract), `audit/CORRECTNESS_LAYERS.md` (the map), `tests/test_invariant_guard.py` (19/19), `scripts/com.alta.invariant_guard.plist` (daily 09:20). Detect-only — does NOT fix RED-1 (that's the pre-registered Blue change); the guard is its regression test.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [x] `python3 audit/invariant_guard.py --run` = FAIL on the live contamination (I1=5 exact RED-1 records, I2 caught USD_CAD units=1 sentinels), 3 URGENT escalations, report written, exit 1
- [x] 19/19 tests green incl. I1 exclusion test, independence cross-check, no-execution-import AST guard
- [x] spec is a single hashed `yaml audit-spec` fence; guard imports nothing from `sovereign/execution/`
- [x] **operator:** load the launchd job to make it standing — DONE 2026-07-06 (Colin-authorized; `com.alta.invariant_guard` in `launchctl list`, daily 09:20)
- [ ] follow-on: land the pre-registered Blue fix in `reflect_cycle` (guard turns green when contamination stops)
**status:** done + LOADED (guard standing 2026-07-06)
**pre_approved:** false

## TICK-005
**title:** Library ascension slice — lived annex + review-time precedents + scored citations
**description:** Wire the Alexandrian Library into the memory loop (it already serves ICT live via ict/library_bridge — query every scan; the review loop is the missing theater). L1 ingest (reviews/attributions/interim seals → `data/experience/library_annex.jsonl`), L2a review-time Precedents section, L2b flagged decision-time stub (default OFF), L3 citation records with `scoring_due` so attribution can later score each analogy. `models/alexandrian_library.json` is LIVE-read AND live-written (`learn()`) — never touched; byte-equality test enforces. Full verified design: `plans/TICK-005.md`.
**depends_on:** []
**blocks:** [TICK-006]
**acceptance_criteria:** see plans/TICK-005.md §Acceptance (suite baseline holds; canonical json byte-identical; W27 dry-run produces cited Precedents section; flag-off = no section)
**status:** done (2026-07-03 — merged f243283; annex live with 17 entries; review_enabled=false pending Colin's post-Jul-6 flip)
**pre_approved:** true

## TICK-006
**title:** Review forensics feeds — oracle health, ledger results, veto ratio, audit parity (+ lesson-velocity & macro-block ports)
**description:** `experience/weekly_review.py` reads none of: oracle reflection `system_health_note` (the 07-03 one flags a data-integrity crisis the W27 review shipped blind to), the week's hypothesis-ledger RESULTS (the loop proposes but never hears verdicts), veto ledgers (acted:abstained:vetoed ratio), shadow-audit parity, `lesson_velocity.json`, morning-briefing FRED macro block. Six small guarded reads, each in try/except — the review must never die on a feed. Oracle-derived text is context-only and marked contaminated-source until TICK-011 lands (RED-1). exit_reason stays AMBIGUOUS — inference refused (PROP-2026-W27 is the sanctioned path).
**depends_on:** [TICK-005]
**blocks:** []
**acceptance_criteria:**
- [x] W27 dry-run shows: system-health line (quarantine-marked), tested-status Counter, acted:abstained:vetoed ratio, audit parity line, lesson velocity, macro block — with honest per-feed degradations
- [x] Any feed absent/corrupt → review still generates (22 tests; ledger-dedup fails OPEN)
- [x] Suite baseline holds (40 failed exact / 1142 passed / 1 skipped)
**status:** done (2026-07-03 late — builder merged)
**pre_approved:** true

## TICK-007
**title:** Dashboard parity — nightly positioning-board export + panel
**description:** The board's institutional-positioning family (cot_net_pct/_z/flush, tff_lev_net_pct, vrp_*, rr25/bf25/atm_term_slope, econ_surprise_z, gdelt_tone) reaches no dashboard panel (R6). Step 1 (SAFE-NOW): exporter → `data/agent/positioning_board.json` (latest per-pair board row + staleness marker). Step 2: "Positioning" panel in root `index.html` (master-branch worktree deploy — see project_dashboard_deploy). DISPLAY-ONLY: no live gate reads the board without a CONFIRMED verdict (Article 6 — R6's "wire the stubs" suggestion explicitly refused).
**depends_on:** []
**blocks:** []
**status:** step 1 done (2026-07-07, merged 78e2706 — exporter + update_sentiment tail-call); step 2 (panel, master worktree) own session
**pre_approved:** false

## TICK-008
**title:** Health resurrection — watchdog trifecta diagnosis + staleness deadlines + log redirects
**description:** health.responder (the watchdog itself) dead since Jun 14; oracle reflect stale Jun 28 (yfinance errors); hypothesis.generator + research.factory stale Jun 15; stray_tripwire WatchPaths inert since Jun 7; bench/evening_prep/render_keepalive log to /tmp or nowhere (R7). Diagnose root cause per job BEFORE restarting anything; per-job staleness deadlines surfaced in morning brief + dashboard; plist log-redirect batch prepared as a DIFF for Colin (plist edits = live-organ config, not applied unilaterally).
**depends_on:** []
**blocks:** []
**status:** backlog
**pre_approved:** false

## TICK-009
**title:** Re-enable Numba for the fast backtester (py3.14 incompat)
**description:** Bench history: Numba-active 728k bt/s single / 1.26M parallel (2026-06-29 18:06); Numba-inactive same day 25k/123k — ~10× research throughput behind one env fix. **Day-3 R1 scope (2026-07-06):** numba NOT INSTALLED; python 3.14.4 (numba supports ≤3.12) → fix = dedicated py≤3.12 research venv + numba install, NOT a patch; the @njit kernels live in the bench harness (fast_backtester.py itself is pure numpy, imported by tests only — import-graph safe). ACCEPTANCE now includes: golden-set verification (fast engine reproduces the current engine on a pinned input set bit-identically or within a pre-declared tolerance) BEFORE any new work uses it; NEW work only — no sealed/in-flight family result ever re-runs on a different engine. Own session; full suite before/after.
**depends_on:** []
**blocks:** [TICK-012]
**status:** backlog
**pre_approved:** false

## TICK-010
**title:** Journal context preservation — keep decision-logger reasoning fields at sync time
**description:** journal_sync keeps 3 of ~60 DecisionRecord fields; why_this_trade / signal_layers_active / component_scores / grade / commitment_score are discarded — the machine's memory forgets its own reasoning (R3). Additive pass-through + backfill where source records exist. SEQUENCED after the first unattended Sunday beat (Jul 5): do not touch the memory organ's writer before its organic-fire verification (DoD item).
**depends_on:** []
**blocks:** []
**blocked_until:** 2026-07-07
**status:** backlog
**pre_approved:** false

## TICK-011
**title:** RED-1 BLUE fix — source-exclusion in oracle reflect_cycle
**description:** Implement the pre-registered fix from the 2026-07-03 Red/Blue audit (exclude fills_backfill/test_fill + pair filter in `_load_decision_log_summary`). Oracle outputs stay quarantined from every consumer until this lands; TICK-004's invariant guard I1 turns green when it does.
**depends_on:** []
**blocks:** []
**blocked_until:** colin_review (Blue-Team-Proposed-Fix-2026-07-01.md)
**status:** backlog
**pre_approved:** false

## TICK-012
**title:** Successor research harness on the fast engine
**description:** Hypothesis testing replays static v015 trades while fast_backtester idles (R4). Design the NEXT-generation harness (signal → simulate → prereg-gated verdict, CPCV-integrated) for successor families + the discovery bench. MUST NOT touch HYP-072..081 — their hash-locked protocol is event-study by design; swapping engines mid-family is a prereg violation dressed as an optimization.
**depends_on:** [TICK-009]
**blocks:** []
**blocked_until:** family_adjudicated
**status:** backlog
**pre_approved:** false

## TICK-013
**title:** Load com.alta.sentiment_update (Colin's hand — one command)
**description:** The board's daily schedule exists as a verified plist but was never installed; the sensory organ updates only manually (3 days stale at audit; the machine's permission layer correctly refused installing a persistent job — same denial as TICK-004's guard plist). Command: `cp scripts/com.alta.sentiment_update.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.alta.sentiment_update.plist && python3 scripts/plist_watchdog.py --rebaseline "loaded sentiment_update"`.
**depends_on:** []
**blocks:** []
**status:** ready — owner: Colin
**pre_approved:** n/a (human action)

## TICK-014
**title:** Small sensor/vault efficiency batch
**description:** ~~GDELT off-peak retry mechanism~~ (BUILT 2026-07-06 as TICK-016: scripts/gdelt_retry.py + com.alta.gdelt_retry.plist — Colin loads). Remaining: GDELT resume-cursor (gdelt_feed.py:80); vault-graph post-learn regen hook (3 lines); optional bench regression alert (5 lines) pending Colin's bench ruling.
**depends_on:** []
**blocks:** []
**status:** backlog (retry portion done via TICK-016)
**pre_approved:** false

## TICK-015
**title:** close_reason capture (PROP-2026-W27 promotion) — slice 1: backfill JOIN
**description:** Day-3 B1 mandate = the promotion signal. SCOPE SPLIT per plans/TICK-015.md: slice 1 (backfill.py shadow-log JOIN, join-never-inference, historical AMBIGUOUS rows untouched) builds now; slice 2 (DecisionRecord.exit_reason + oanda_bridge close records) is `blocked_until: shadow_close` — those files are execution-path under the freeze regardless of how additive the change looks. Related pulse_check probe-prefilter + match-window widening ship in Colin's RED-1 review batch (audit/health_diagnosis_2026-07.md), not here.
**depends_on:** []
**blocks:** []
**status:** done (2026-07-07, slice 1 merged f1a29a0; slice 2 blocked_until: shadow_close)
**pre_approved:** true (Day-3 mandate B1)

## TICK-016
**title:** GDELT off-peak retry job (Colin loads)
**description:** BUILT: scripts/gdelt_retry.py (paced single nightly pass, done-marker stops retries, success rebuilds board + runs look-ahead auditor + escalates the exact BH-unblock sequence to messages_to_colin) + scripts/com.alta.gdelt_retry.plist (02:30 ET). Family stamped PENDING-080-SCHEDULED on HYP-080. Load: `cp scripts/com.alta.gdelt_retry.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.alta.gdelt_retry.plist && python3 scripts/plist_watchdog.py --rebaseline "loaded gdelt_retry"`. If a week of off-peak attempts also fails → the manifest's "family documents its handling" branch (Colin's protocol decision).
**depends_on:** []
**blocks:** []
**status:** ready — owner: Colin (load)
**pre_approved:** n/a (human action)

## TICK-017
**title:** Citation scorer (A2 — Gγ/HYP-084's measurement instrument)
**description:** Per plans/TICK-017.md (verified design): analogy_prediction_v2 via frozen SEVERITY_PREDICTION_MAP; experience/citation_scorer.py appends citation_scores.jsonl (citations.jsonl never rewritten); UNSCOREABLE first-class; one guarded forensics line. Without this the dark month produces nothing HYP-084 can adjudicate.
**depends_on:** [TICK-005]
**blocks:** []
**status:** ready (build next session if Day-3 capacity ends)
**pre_approved:** true (Day-3 mandate A2)

## TICK-018
**title:** Geometry extractors for the LOCKED HYP-082/083/084 (G2)
**description:** Per plans/TICK-018.md (verified design; specs hash-locked FIRST at 01cacbd): sovereign/sentiment/geometry_feed.py (trailing corridor R²/dev, REPLICATED look-ahead-safe daily FVG kernel — the ict detector leaks last-bar ATR and the sentiment wall is bidirectional — tri_state detector), sentiment_geometry_daily table, board ASOF join + 7 columns, look-ahead auditor block, BOARD_EXTREME_TAGS geometry keys, AST-wall coverage. Real-data update() + audit + the Gα/Gβ RUN are the orchestrating session's (or TICK-019's) — builder ships code+tests only.
**depends_on:** []
**blocks:** []
**status:** done (2026-07-07, merged d3ec74f — code+tests; the real-board update()+audit run and the Gα/Gβ locked-spec run = TICK-019)
**pre_approved:** true (Day-3 mandate G2)

## TICK-019
**title:** Geometry feature fill + Gα/Gβ run under the locked specs
**description:** Run `geometry_feed.update()` against the real board (short — parquets are local), rebuild, look-ahead auditor 0-violations gate, THEN run HYP-082/HYP-083 exactly per their hash-locked preregs (01cacbd) with the GEOMETRY-2026-07 BH (m=2) adjudication. HYP-084 waits on the dark month + TICK-017 scorer + Colin's review_enabled flip. Same seal discipline as the positioning family (dated annotations, statuses stay PREREGISTERED until family verdicts).
**depends_on:** [TICK-018]
**blocks:** []
**status:** done (2026-07-07 — Colin's "go" on plans/TICK-019.md; fill 0-violations/163,072 rows; runner 9c7c964; ADJUDICATED 6968ba9: HYP-082 NOT_SIGNIFICANT p=.598 N=2172 · HYP-083 NOT_SIGNIFICANT p=.741 N=1190)
**pre_approved:** false (evidence run — plan-first, honored: plan 3b8e900 preceded the go)

## TICK-020
**title:** Political-alpha event study (HYP-085) — build research/political_alpha/ per the vault spec
**description:** Pre-registered event study: do Trump public statements produce abnormal moves (±2σ of trailing 60d SD) in tagged forex pairs / sector ETFs, with a pre-announcement FXE rr25 positioning overlay? Governing spec (LAW, locked 2026-07-08): `~/Obsidian/Obsidian/Trading/Research/Political-Alpha-Claude-Code-Spec.md`. Prereg hash-locked BEFORE any event data: `data/research/preregister/HYP-085_political_alpha_trump_events.json` (hash 58e725ed…, ledger entry PREREGISTERED). Four phases, one [RESEARCH] commit each: P1 event catalog (FR API + whitehouse.gov + Truth Social probe chain + spec-authorized manual fallback; deterministic regex classifier; 5-trading-day per-instrument de-dup; <30 events → state shortfall, never loosen), P2 abnormal returns (T-252→T-10 mean-adjusted baseline; post = T0+T1 daily log returns), P3 rr25/put-call-volume positioning (ThetaTerminal localhost; FXE proxy for forex; probe native ETF chains, record gaps, never synthesize), P4 exactly 3 pre-registered tests (QQ+Shapiro, SD exceedance, 10k statement-level placebo bootstrap seed 42) → normality_plot.png + sd_test_results.json + summary_report.md. Isolation HARD: imports nothing from sovereign/ict/ict-engine/config/audit/scripts (AST test). No launchd, no OANDA, no live params. Null-not-rejected is a valid outcome; rejected null = candidate result only (gauntlet promotion is a separate ledgered step). Plan: plans/TICK-020.md.
**depends_on:** []
**blocks:** []
**acceptance_criteria:** spec §9 DoD verbatim — catalog ≥30 qualifying events or shortfall stated; normality + SD charts produced; bootstrap p reported honestly; only the 3 pre-registered tests run; isolation test green; all §12 artifacts exist; four [RESEARCH] phase commits pushed; NEXT.md updated
**status:** done (2026-07-08 — all 4 phases + prereg shipped 5b5534b→abcc120; VERDICT: H0 NOT rejected p=0.3637 → HYP-085 NOT_SIGNIFICANT sealed; catalog 168 events/223 rows, honesty gate PASS; isolation 11/11)
**pre_approved:** true (Colin's GREENLIGHT in the spec + approved plan 2026-07-08)

## TICK-021
**title:** Political-alpha V2 event study (HYP-087 Track A + HYP-088 Track B) — build research/political_alpha_v2/
**description:** Three-track V2 successor to the NOT_SIGNIFICANT HYP-085 event study. V1's homogeneous-event-population assumption was the wrong tool; V2 conditions on event TYPE before testing. Governing spec (LAW, locked 2026-07-09): `~/Obsidian/Obsidian/Trading/Research/Political-Alpha-V2-Claude-Code-Spec.md`. Track A (HYP-087, language taxonomy): cluster the 168 V1 Trump events by keyword ruleset on statement_text into ≤7 clusters, compute per cluster×instrument CSAR over [0,+72h] vs the 9-ETF universe (XLE/XLF/XLV/XLI/KWEB/SLX/GLD/TLT/DX-Y.NYB), within-cluster one-sample t-test with Bonferroni correction (denom = non-null clusters × 9, locked at Phase 1). Track B (HYP-088, congressional signal): pull congressional disclosures (Quiver free API / house.gov EFTS), build BUY clusters (≥3 members, same sector bucket, 30d rolling), Fisher's exact one-tailed test that BUY clusters precede sector-favorable policy actions within 90d. personal_attack cluster = negative control. Phase 0 config-commit gate: cluster_rules.json + sector_map.json + policy_events.json committed to git BEFORE any return/disclosure data pulled (pre-registration). AST isolation HARD (no imports from sovereign/ict/ict-engine/config/audit/scripts; no OANDA; no launchd). Outputs: cluster_playbook.md, cluster_sar_plots.png, congressional_signal_results.json, summary_report.md. Track C (HYP-086) is separately pre-registered — reference only, NOT built here.
**depends_on:** [TICK-020]
**blocks:** []
**acceptance_criteria:** spec §9 DoD verbatim — Phase 0 config committed before any data pull; cluster playbook lists ALL tested cluster×instrument pairs (passing or not, non-passers keep their p-values); Track B Fisher's exact p + N + underpowered flag if N<10; personal_attack negative control reported; Bonferroni denominator locked Phase 1 + applied Phase 4; only the Phase-4 tests run (no post-hoc refinement); isolation test green; honest summary report states null rejected-or-not per track + all data gaps; four [RESEARCH] phase commits + the Phase 0 config commit landed and branch pushed; NEXT.md updated.
**status:** in_progress (2026-07-09 — Claude Code / Molly)
**pre_approved:** true (Colin's GREENLIGHT in the spec header + this is the executable build order)

## TICK-022
**title:** Prop-Funnel EV Simulator — research/prop_funnel/ (Phase A of the "$10k/month" program)
**description:** Measurement instrument, NOT edge validation: simulate every strategy family through realistic prop-firm rulesets and produce (a) per-strategy×firm verdict table — P(pass P1), P(pass both), p^100, days-to-pass (med/p10/p90 + calendar), E[attempts]+fees-to-funded, P(survive 12mo funded), E[monthly payout], P(month ≥ $10k), P(≥$10k every month ×12), 24-month program EV/month (sort key); (b) synthetic requirements frontier (Sharpe {0–3} × vol {0.25–2}%/day × freq {0.2–2}/day, student-t(4)) with the pass-rate-vs-income tension chart; (c) per-phase sizing-policy optimization ({challenge_mult × funded_mult} grid, funnel-EV objective). Firms: FTMO_100K_SWING (2-phase, static 10% DD, daily 5%, no time limit), APEX_50K (intraday-trailing $2.5k, consistency 30%, κ-bracket bound), TOPSTEP_50K (EOD-trailing), parity presets LUCID/MFF/FUNDERPRO. Feeds (evidence-stamped): carry_oos n=110 PROVEN/REGIME_FRAGILE + carry_decade 411 trades + forward-Sharpe band {0, 0.69, 1.25} SCENARIO; ict_windows UNPROVEN (p=0.52); ict_live n=27 (3W/24L) LOW_N_SANITY_ONLY; futures_orb UNVALIDATED; synthetic. REUSES (unchanged): sovereign/propfirm/{rules_engine,challenge_simulator}.py, sovereign/risk/monte_carlo_prop.py, sovereign/discovery/{gate,cpcv,block_length}.py, sovereign/reporting/equity_curve.py. PARITY-FIRST GATE: reproduce logs/prop_challenge_sim.json (pool=window_B n==102, mff 100k, risk .0075, n=5000 seed 42 → pass .7444 EXACT), data/agent/ftmo_swing_mc.json (under its own trailing+60d assumptions, divergence documented — it does NOT model real FTMO), data/risk/prop_monte_carlo.json (OUT monkeypatched, seed 7) BEFORE any new engine code. Isolation: AST wall (forbidden roots ict/ict_engine/config/execution/scripts/audit; sovereign whitelist propfirm+risk.monte_carlo_prop+discovery+reporting) + no-OANDA/launchd strings. NO hypothesis-ledger writes, no live/config changes, no launchd. Outputs → data/research/prop_funnel/. Plan: plans/TICK-022.md (approved 2026-07-10; master copy Plans/glistening-juggling-clover.md).
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [x] Three parity checks green — ALL THREE EXACT (parity #1 pool pinned to window_B n==102; #3 clock pinned to recorded 53.2 trades/yr — logs/forex_backtest_results.json drift documented)
- [x] Ruleset tests green incl. hand-computed static-vs-trailing divergence + PropFirmRules equivalence (50 capped + 25 uncapped seeded sequences, incl. BUST paths)
- [x] ict_live filter n==27 asserted (3W/24L — live WR 11% vs backtest 63.6%, surfaced in report); every row stamped; carry regime caveat verbatim
- [x] Sanity anchors green (Sharpe-0 two-phase 0.18–0.40 band; Sharpe-1.5 low/hot-vol anchors; vectorized engine EXACT vs scalar on 6 rule variants — the stronger check)
- [x] Determinism: cross-process canonical results.json identical (crc32 seeds; hash() salt bug found+fixed)
- [x] All artifacts under data/research/prop_funnel/ (verdict_table.md/.csv, 10 charts incl. tension contours, summary_report.md, results.json); iid caveat on the table header
- [x] rules_engine.py byte-identical (diff empty); data/risk/prop_monte_carlo.json + data/propfirm/ + data/futures/ untouched by this module (isolation test enforces)
- [x] One [RESEARCH] commit per phase (P0 03b8093, P1 4f41853, P2 c3719e3, P3 d4e50dd, P4 7f9ca98, P5+P6 follow); NEXT.md updated; pushed
**status:** done (2026-07-10 — full 10k run seed 7, 75s. HEADLINE: P($10k every month ×12) = 0.0 on EVERY strategy×firm row at current account sizes; only carry_oos(favorable-window)×FTMO approaches "pass 100×" (p^100=0.68; honest decade pool: 1.5e-05). Phase R (futures replay regen) NOT run — operator-gated)
**pre_approved:** true (plan-mode approval 2026-07-10 — Plans/glistening-juggling-clover.md)

## TICK-023
**title:** HYP-090 "MODERN" — pre-registered adaptive walk-forward study (research/modern/)
**description:** Colin's recurring adaptive-parameters idea, tested end-to-end ONCE under the full gauntlet to seal the family. Daily adaptive parameter selection over trailing {90,180,365}cal-day windows: A1 recent-winner argmax + A2 regime-matched k-NN (k=25, expanding-standardized trailing regime vector) + A3 placebo floor (500 crc32 seeds) vs A0 static v015, on 2015→2026 daily forex. FULL surface: 385 configs (θ×hold×stop×trail×gate grid + config #385 = v015-exact incumbent w/ per-pair trailing) × 15 pair subsets = 5,775 variants × 3 windows = 17,325 cells. Architecture: precompute-then-replay (1,540 decade kernel runs via fast_backtester.simulate_forex_trades_arrays → per-day M2M series with CAUSAL costs [spread@entry, swap daily] + open-tail rows; daily loop = rolling stats, no re-backtesting). Prereg data/research/preregister/HYP-090_modern_adaptive_params.json hash-locked + ledger PREREGISTERED BEFORE any data (gate_zero enforced first-call in run_all + regression test). Gauntlet: stationary-block-bootstrap (L=5, 10k, seed 42) arm-vs-A0 one-sided p + BH m=6; DSR n_trials=5,775 primary (6 / 17,325 disclosed); A3 envelope; per-year non-degrade 2017–2025 (tol 0.05); switching-costed variant is verdict criterion 5. ABORT (never re-tune): reconcile weighted_portfolio_sharpe(2015-2024) ∉ 0.6886±0.01. Prior kills disclosed: HYP-065/066/067, 180-config exit sweep, regime router. **Registered prior: NOT_ROBUST.** Isolation whitelist sovereign.forex.{fast_backtester,forex_backtester,exit_machine,data_fetcher,signal_engine}+discovery+reporting; NO config/parameters.yml writes (never resemble monthly_reopt.py); outputs only data/research/modern/. Plan: plans/TICK-023.md (approved 2026-07-11; master Plans/glistening-juggling-clover.md).
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [x] P0: prereg hash-locked (6dd9cc85) + ledger PREREGISTERED BEFORE any data; --verify green; gate-zero regression tests
- [x] P1: reconcile EXACT 0.6886; inputs frozen + sha256 manifest; 64 ungated builds × 2 spans + external causal VIX mask (caught + fixed: hardcoded country codes broke FRED lookups — canonical CB_TO_COUNTRY codes required)
- [x] P2: 1,540 kernel runs (30,788 trades, 24 open-tails recovered via flat-padding); M2M decomposition exact ≤1e-10; config #385 parity: 411/411 canonical trades date-identical (pnl within snapshot's 6dp cost rounding)
- [x] P3: truncation-invariance (20 sampled t × 2 windows, A1+A2) + regime causality green
- [x] P4: 6 runs + costed + 1,500 placebos + block-bootstrap gauntlet; **VERDICT NOT_SIGNIFICANT sealed** (prereg hash verified pre/post seal); charts + summary + results.json; NEXT.md; pushed
- [x] 20 module tests green; write-safety enforced by isolation tests
**status:** done (2026-07-11 — ADJUDICATED **NOT_SIGNIFICANT** (the registered prior): A0 static v015 daily-M2M Sharpe **+0.948** vs adaptive arms **+0.17..+0.43** — every arm UNDERPERFORMS static (min one-sided p=0.977) AND loses to the 500-seed RANDOM-selection placebo floor (p95≈0.92): recent-winner/regime-matched selection is actively ANTI-selective on ~13-trade windows. The MODERN/adaptive-parameters family is closed with a receipt.)

## TICK-024
**title:** Recalibrate SWAP_RATES_ANNUAL — the swap cost model is ~10x too small (one sign flip)
**description:** 2026-07-11 live triage + `research/swap_calibration.py` (read-only, OANDA v3 instruments financing rates vs the model; `data/research/swap_calibration.json`): model swap magnitudes are an order of magnitude below broker reality on ALL 4 pairs (worst: USDJPY SHORT actual −3.82%/yr vs modeled −0.35%; EURUSD LONG −2.45% vs −0.15%), plus one SIGN flip (EURUSD SHORT actually EARNS +0.42%/yr vs modeled −0.10% — confirmed empirically by trade #227's +1.1122 USD of daily credits). At ~7-day mean holds this is up to ~0.07% of notional per trade of mis-modeled cost — material vs ~0.5%/trade typical pnl; direction of aggregate historical bias unknown until re-run (USDJPY shorts likely OVERSTATED in backtests). DESIGN CONSTRAINT: today's OANDA rates must NOT be pasted over 2015-2024 history — financing follows the rate differential over time; the honest fix derives swap from the rate-differential series the system already tracks (data_fetcher.get_pair_differentials) with a calibrated broker spread, validated against the current OANDA rates + trade-227 empirical anchor. GATED HARD: SWAP_RATES_ANNUAL feeds _apply_costs → every backtest → the canonical 0.6886 reconcile anchor used by every study's gate. Requires: param_change_log rationale, full re-reconcile, an explicit re-anchoring decision for RECON_TARGET/bands, and a re-read of which sealed verdicts are cost-sensitive. Colin sign-off required before ANY table change.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [ ] Historical swap model designed (rate-differential-derived + broker spread), validated vs current OANDA rates (4 pairs × 2 sides) and the trade-227 anchor
- [ ] Impact study BEFORE any live table change: canonical decade + OOS re-run with the new model, per-pair deltas reported; list of cost-sensitive sealed verdicts
- [ ] param_change_log rationale + Colin sign-off; RECON_TARGET re-anchoring decision recorded
- [ ] No execution-path change until all above land
**status:** backlog
**pre_approved:** false

## TICK-025
**title:** Surface live-scan input degradation as a first-class DEGRADED flag (fail-loud)
**description:** The daily carry scan (com.alta.forex.scan) has been running DEGRADED with the evidence visible only in logs/forex_scan.err: yfinance failures for USDJPY=X / AUDUSD=X ("possibly delisted; no price data found", "Insufficient price history"), ForexDataFetcher falling back to SYNTHETIC macro state for AU/EU. The per-pair NO_TRADE convictions therefore rest on partial/synthetic inputs — a real signal could be missed with zero surface indication ("the system is wrong when it silently succeeds"). Fix: propagate a per-pair input-quality status (OK / DEGRADED_PRICES / SYNTHETIC_MACRO) from the fetch layer into proof_of_life.json, forex_proximity.json and the health check; overall health goes YELLOW when any carry pair is degraded. Also fix the stale proof_of_life.last_fill pointer (points at the closed USD_JPY leg instead of the open book). Touches live-scan-adjacent code → plan-first; scan logic/gates themselves unchanged.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [x] Fail-loud DEGRADED flag when the yfinance OHLCV fallback fires (sentinel file + WARNING) — landed 2026-07-12
- [ ] Per-pair input-quality field in proof_of_life.json + proximity; health YELLOW on any DEGRADED carry pair
- [ ] Root-cause note for the yfinance failures (ticker format/rate-limit/API drift) with fix or documented workaround
- [ ] last_fill reflects the live open book
- [x] No change to signal logic, gates, or sizing — observability side-effect only (verified: None-return preserved, macro/carry not in backtest path)
**status:** in_progress (2026-07-12 — dispatched minimal scope done: fail-loud `sentinel/DEGRADED_yfinance_<pair>.txt` + WARNING at the yfinance OHLCV fallback in macro_engine._get_price_history and carry_engine._fetch_prices. REMAINING: input-quality propagation into proof_of_life/proximity/health + last_fill fix.)
**pre_approved:** false

## TICK-026
**title:** data.forex_factory_scraper is imported but no longer exists — restore or amputate
**description:** logs/launchd_err.log shows a scheduled job repeatedly dying on `ModuleNotFoundError: No module named 'data.forex_factory_scraper'` — a live specimen of the silent-failure class (job "runs", produces nothing, no surface alarm). `data/calendar_fetcher.py:8` imports it and `ict/daily_bias.py` references it, but the module file is GONE from the working tree (created 541b47b 2026-05-27; likely removed in the 2026-07-02 attic/subtraction pass while consumers still import it). Decide: restore from git history, or amputate the import chain (calendar_fetcher fallback path) and identify/retire the erroring job. Also add the exit-code watchdog lesson: a repeated import-error job should page, not whisper in launchd_err.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [x] Import chain verified NOT broken — no restore/amputate needed (see below)
- [ ] Next scheduled run exits 0 — n/a (premise stale; module present)
- [ ] launchd_err.log watchdog (repeat import-error should page) — separate observability concern, not this ticket
**status:** closed/stale (2026-07-12 — STALE PREMISE. `data/calendar_fetcher.py:8` and `ict/daily_bias.py:102` import `data.forex_factory_scraper`; that module EXISTS and is git-tracked (added 541b47b 2026-05-27, never removed — the "deleted 07-02" claim was wrong). Verified `python3 -c "import data.calendar_fetcher; import data.forex_factory_scraper"` → BOTH IMPORT OK. Nothing to restore or amputate.)
**pre_approved:** false
**pre_approved:** true (plan-mode approval 2026-07-11 — Plans/glistening-juggling-clover.md)


## TICK-027
**title:** HYP-091 TSMOM diversification study — research/tsmom/ (does time-series momentum add a real diversifier to the v015 carry book?)
**description:** Pre-registered backtest of 12-month time-series momentum (Moskowitz/Ooi/Pedersen) on the 4 v015 pairs (EURUSD/GBPUSD/USDJPY/AUDUSD), 2015-2024, inverse-vol sized (ex-ante vol via ewm com=60, target_vol 10%, leverage cap 2.0), monthly rebalance, net of CORRECT rate-differential-derived financing — NOT the broken Colin-gated SWAP_RATES_ANNUAL (TICK-024 proves it ~10x too small + one sign flip). Financing = anchored differential-tracking model: financing_side(t)=oanda_side_now + sign*(diff(t)-diff_now), diff from data_fetcher.get_pair_differentials, oanda_side_now from data/research/swap_calibration.json (TICK-024 output). Pre-registered null: standalone OOS (2023-24) Sharpe <= 0 OR monthly correlation with v015 carry > 0.5 -> diversification thesis fails. Deliverables: per-calendar-year Sharpe breakdown, monthly corr vs v015 + 50/50 combined-portfolio Sharpe, directional-permutation + Deflated-Sharpe + BH + holdout gauntlet -> VALID_EDGE/NOT_SIGNIFICANT. Isolated package (mirror research/prop_funnel/): reads sovereign read-only, writes only data/research/tsmom/ + hash-locked prereg + 2 ledger touches. Research pass only — deployment/live-capital OUT OF SCOPE (RISK_CONSTITUTION Art. 6). Plan: Plans/here-s-what-the-literature-sleepy-tarjan.md.
**depends_on:** []  (reads TICK-024's data/research/swap_calibration.json; does NOT require the gated live-table fix)
**blocks:** []
**acceptance_criteria:**
- [ ] Phase 0 hash-locked prereg (data/research/preregister/HYP-091_tsmom.json) + PREREGISTERED ledger entry, verify() PASS, committed BEFORE any price data observed
- [ ] Reconstructed v015 monthly carry series reproduces the decade Sharpe ~0.69 (loader sanity) before any correlation is trusted
- [ ] Per-calendar-year Sharpe table (2016-2024) + IS/OOS split reported
- [ ] Monthly corr vs v015 + combined-portfolio Sharpe + robustness cost legs (broken-model + price-only)
- [ ] Directional-permutation p (>=10k) + DSR + BH + holdout>0 -> verdict to ledger (backup first, hash_lock preserved)
- [ ] Isolation test green; full suite + ICT isolation test unaffected; no execution-path / live-param changes
**status:** backlog
**pre_approved:** true (plan-mode approval 2026-07-12 — Plans/here-s-what-the-literature-sleepy-tarjan.md)

## TICK-028
**title:** 90-day ICT taken-trade projection — research/ict_projection/ (is ICT still the right prop-challenge vehicle at current signal frequency?)
**description:** Read-only projection of how many ICT trades the system will realistically TAKE over the next 90 days at current signal/veto frequency, to sanity-check the ~30-trade prop-challenge timeline. Reads data/ledger/ict_veto_ledger_2026_{05,06,07}.jsonl (writer ict/ict_veto_ledger.py:79) + data/decision_logs/decisions_2026_{05,06,07}.jsonl. CRITICAL: dedup per-scan re-emission — raw ~95 veto records/day collapse to ~7 unique (day,pair,direction,reason) setups/day; skipping inflates the count ~13x. Recompute ADR-veto and weekly-trend-veto rates LIVE (memory's 55%/31% is stale; live ~51%/~9%, plus new HYP046_DISP_GATE/TIMING_GATE classes). Project taken count over ~62 trading days with a bootstrap-over-days CI + rate-sensitivity band; report whether it lands near 30. Committed reproducible script + short markdown report; writes only data/research/ict_projection/projection_90d.json. Shadow-freeze compliant: never import/touch ict/pipeline.py, ict/orchestrator.py, or the exit path. Plan: Plans/here-s-what-the-literature-sleepy-tarjan.md.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [ ] Dedup factor (~13x) computed and reported; projection uses unique setups, not raw records
- [ ] ADR + weekly-trend veto rates recomputed live from the ledger (not memory)
- [ ] 90-day taken-trade point estimate + bootstrap CI + rate-sensitivity band
- [ ] Deterministic: two runs produce identical projection_90d.json
- [ ] Read-only: no imports of ict/pipeline.py/orchestrator.py/exit path; nothing written outside data/research/ict_projection/
**status:** backlog
**pre_approved:** true (plan-mode approval 2026-07-12 — Plans/here-s-what-the-literature-sleepy-tarjan.md)

## TICK-029
**title:** HYP-092 Gapper-continuation read backtest — research/gapper_continuation/ (does the decision card's CONT/EX read separate post-read returns?)
**description:** One-year (2025-07-01→2026-06-30) no-look-ahead replay of the vault Gapper-Continuation-Decision-Card: survivorship-free discovery via Polygon grouped daily (buffered +20% high-vs-prev-close superset filter, day vol >=500K), verification + read + outcomes entirely from Alpaca SIP 5-min bars adjustment=split (filters at 10:30 ET: price>=1.30x prev close, >=$2, cum vol >=500K on bars start<=10:25 only); frozen mechanization of the card's checklist (VWAP, higher-lows, up/down volume, range position, lower-highs, climax-fade, rejection wick) votes CONT/EX/UNC; outcome = % from 10:30-bar OPEN to last RTH bar close. Pre-registered null per the card: MWU one-tailed CONT>EX alpha=.05, n>=30/bucket; robustness = 10-trading-day run-dedup per ticker; ticker-day non-independence disclosed. Prereg data/research/preregister/HYP-092_gapper_continuation.json hash-locked + ledger PREREGISTERED BEFORE outcome analysis. Isolated: no sovereign/ or ict/ imports; writes only data/research/gapper/. Probes done: Alpaca SIP serves delisted tickers (MSW/TBH/LIXT) + intraday adjustment=split verified (AAPL 4:1). Direct build request from Colin 2026-07-12 ("put together a simple test ... back test it ... no look ahead").
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [ ] Prereg hash-locked + ledger PREREGISTERED before outcome analysis (commit order proves)
- [ ] Stage-1 cache covers all trading days in window; stage-2 coverage % reported split by active-proxy (delisted-vs-active)
- [ ] All read/filter inputs from bars start<=10:25 ET; outcome origin = 10:30 bar open (no shared bar); guards counted (sparse/halt, IPO, split-mismatch)
- [ ] MWU primary + run-deduped robustness reported; UNC + sparse outcome distributions disclosed; verdict sealed to ledger with hash verified pre/post
- [ ] Deterministic reruns; no sovereign/ict imports; nothing written outside data/research/gapper/ + prereg; execution path untouched
**status:** done (2026-07-12 — ADJUDICATED **NOT_SIGNIFICANT** well-powered: MWU CONT>EX p=0.594 (n=558/391), robustness p=0.634; CONT median -2.34% vs EX -1.81%; filter base rate -2.21% median / 48% reverse >3%; halt-excluded set median +16.5% descriptive. Report: data/research/gapper/report.md)
**pre_approved:** true (direct operator build request in-session 2026-07-12)

## TICK-030
**title:** Yield Frontier M-phase — research/yield_frontier/ mining pass (equities gappers + NQ intraday + SPY options) → ranked yield board
**description:** Approved plan Plans/immutable-wondering-alpaca.md. MINING pass (look-back allowed, every output stamped MINING — not evidence): ~800-850 configs across 16 families — M1 equities (overnight continuation 108, parabolic-fade stop-grid 96, halt-runner re-entry 36, no-news recipe 16, catalyst-conditioned 16), M2 NQ 2018-01→2024-06 (ORB 162, first-hour 36, overnight-gap 60, time-of-day 60, VIX-regime 18, Globex 12), M3 SPY options 2022-01→2023-09 (put spreads 81, condors 54, strangles 12, VIX overlay 30, lottery 8). Holdouts fenced at loaders (equities 2024-07→2025-06 NOT on disk; NQ 2024-07→2026-06; options 2023-10→2024-06); append-only mined_n.json feeds gauntlet DSR n_trials; coarse frictions (HTB tiers, locate haircut, halt gap-through stops, NQ ticks+commission, options k×half-spread) with pessimistic headline; yield board ranks net %/day at stated capacity with tail_p5/p1 + ruin fraction mandatory. Zero network calls in M-phase. Report opens with the 2%/day arithmetic statement.
**depends_on:** []
**blocks:** [TICK-031]
**acceptance_criteria:**
- [ ] M0 scaffold + 7 test files green (isolation AST, holdout fences, frictions known-values, look-ahead canary, VRP-collision, determinism, mined-N monotonic)
- [ ] M1/M2/M3 miners run deterministic (two runs byte-identical board CSV); every row stamped MINING
- [ ] Yield board synthesized with settled-family context rows + arithmetic statement; n>=40 rank filter
- [ ] Main-suite failure count unchanged; ICT isolation law green; NEXT.md + push per session
**status:** in_progress (2026-07-12)
**pre_approved:** true (plan-mode approval 2026-07-12 — Plans/immutable-wondering-alpaca.md)

## TICK-031
**title:** Yield Frontier G-phase — preregister top 2-3 board rows (HYP-093/094/095) and gauntlet them on untouched holdouts
**description:** Blocked on TICK-030 board + Colin's candidate pick. G0 hash-locked preregs + PREREGISTERED ledger entries (n_trials = mined_n.json total; tail condition locked) BEFORE any holdout data touched; G1 sanctioned equities holdout fetch 2024-07→2025-06 (gate-zero: refuses unless prereg hash verifies; chunked --max-dates); G2 verdicts via existing machinery (>=10k permutations, DSR at honest n_trials, BH across runs, block bootstrap, per-year non-degrade, tail band) → ledger with backup + hash verified pre/post.
**depends_on:** [TICK-030]
**blocks:** []
**acceptance_criteria:**
- [x] Preregs hash-locked + ledger PREREGISTERED before holdout unfenced (gate-zero enforced; manifest postdates lock)
- [x] Verdicts sealed at n_trials=809: HYP-093 VALID_BUT_BELOW_FLOOR (p=.031, DSR .987) / HYP-094 NOT_SIGNIFICANT (p=.102) / HYP-095 VALID_BUT_BELOW_FLOOR (p=.013, DSR .999)
- [x] No VRP-001-v2 execution; execution path untouched
**status:** done (2026-07-13 — data/research/yield_frontier/gauntlet/report.md)
**pre_approved:** true (plan-mode approval 2026-07-12 — Plans/immutable-wondering-alpaca.md)

## TICK-032
**title:** Funding-vehicle reality check for single-name HTB equity shorts (rider 4, non-gating)
**description:** Colin's rider on the yield-frontier gauntlet: the TICK-022 funnel firms (FTMO/Topstep/APEX class) are forex/futures — they CANNOT trade single-name smallcap shorts. Before the "capacity-bound gapper edge -> funded accounts" thesis carries ANY EV weight: name a real funding vehicle for equities intraday shorting of HTB smallcaps (candidates to research: equities prop shops — e.g. bright/T3/SMB-style firm structures vs retail eval shops with equities support), its locate rules, borrow pass-through pricing, and payout terms. Web research + documented sources; no account opened. Does NOT gate TICK-031.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [x] 6 firms/structures evaluated with cited terms (TTP, Zimtra, T3, Bright, SMB, Funder)
- [x] Explicit verdict: **NOT VIABLE** as true funded vehicle; first-loss-only paths; US-resident wall at the one purpose-built shop
**status:** done (2026-07-12 — data/research/yield_frontier/tick032/funding_vehicle_verdict.md)
**pre_approved:** true (operator rider 2026-07-12)

## TICK-033
**title:** Execution-engineering program for the gapper-fade signal (charter: research/yield_frontier/OPTIMIZATION_PROGRAM.md)
**description:** New research field per Colin 2026-07-13: optimize the exact daily operation of the SEALED HYP-093 signal ("post these findings everywhere. set off agents... i want to use THIS to trade live but not until we know the exact precise best way to run it every day, research on possibly millions of trials"). Charter welds: signal FROZEN (re-tuning = HYP-090 tombstone, BANNED); optimization touches only the execution wrapper (instrument/stop/sizing/overlap/locate/timing); every simulated policy counted in an append-only trials ledger; any live-candidate policy needs its own hash-locked prereg + floor on non-optimized data. Workstreams: W1 mechanism lit, W2 options microstructure (→HYP-096), W3 short-side plumbing/regulation, W4 sizing under jump risk, W5 data procurement (all five dispatched as background agents 2026-07-13); W6 policy Monte-Carlo simulator (SPEC-FIRST — optimization/W6_SPEC.md before any code); W7 forward shadow replay. Live gating ladder: design CONFIRMED under own prereg → sim/paper period → TICK-024 clean → clamps enforced (Jul-28) → Colin's explicit go.
**depends_on:** [TICK-031 (done)]
**blocks:** []
**acceptance_criteria:**
- [ ] W1-W5 agent briefs saved under data/research/yield_frontier/optimization/ and synthesized
- [ ] W6 spec written and hash-noted BEFORE simulator code; trials ledger wired
- [ ] HYP-096 (defined-risk) prereg drafted from W2 numbers for Colin's ack
- [ ] No signal-parameter re-tuning anywhere (charter weld #1 enforced by review)
**status:** in_progress (2026-07-13 — W1-W5 agents dispatched)
**pre_approved:** true (direct operator request 2026-07-13)

## TICK-034
**title:** Catalyst-reliability split of the frozen fade signal (W1's load-bearing prediction)
**description:** W1 mechanism brief: the fade should concentrate in no-news/soft-news movers and weaken/flip on hard catalysts (earnings/FDA) — the overreaction-vs-lottery adjudicator and the one genuinely falsifiable descendant of the agent wave. Prereg on MINING-year events + forward shadow accumulation ONLY (2024-25 holdout is consumed — reuse forbidden). If the no-news subset concentrates the edge, constitutional yield can clear the floor at UNCHANGED sizing (HYP-097 closed the sizing route: W* T10 .627/T20 .798, yield 0.0166%/day). Null fully live.
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [ ] Prereg hash-locked (HYP-098) with catalyst taxonomy frozen before any outcome split is computed
- [ ] Mining-year adjudication + forward-shadow accumulation plan; holdout untouched
- [ ] Constitutional yield re-derivation under the surviving subset, floors unchanged
**status:** backlog
**pre_approved:** false

## TICK-035
**title:** HYP-098 FVG x Fractal-Corridor intraday study on NQ — THE RESEARCH METHOD round 1 (HANDOFF to a parallel session)
**description:** Colin's directive 2026-07-13: run THE RESEARCH METHOD on the ICT/FVG insight paired with the Fractal Corridors Pine indicator (repo root, Pine v6 — port to Python, model pivot confirmation delay), on NQ 1-min 2018-2026 (his pick; no intraday FX exists on disk; ThetaData = option.value.monthly $40/mo EOD-chains-only per invoice read 2026-07-13, renews Jul 16 — NOT usable for intraday bars, terminal stays down). Mining window 2018-01→2024-06 (fenced loader); holdout 2024-07→2026-06 (HYP-095's single daily test on the span disclosed in family accounting); benchmarks named: constitutional >0.023%/day + Sharpe vs carry 1.25 context. Route B framing — sealed daily nulls (HYP-082/083, ICT p=.52, HYP-090) not re-litigated. "Don't stop until you find something" = iterate NEW families with counted trials at a never-lowering bar; exhausting the space honestly is a valid deliverable. Full prompt: research/yield_frontier/HANDOFF_HYP098_fvg_corridor.md. HYP-098+ namespace reserved for the handoff session; it must verify numbers free before claiming (concurrency law).
**depends_on:** []
**blocks:** []
**acceptance_criteria:**
- [ ] Handoff session works in a worktree; charter written; fenced loader + AST isolation + look-ahead canary tests green before mining
- [ ] Mining stamped + trials counted; prereg hash-locked with BOTH priors before holdout touched
- [ ] Verdicts sealed with DSR at full mined-N; per-year non-degrade + tail condition + constitutional sizing per Art. 1
- [ ] Main suite baseline + ICT isolation law untouched; push per batch; NEXT.md per session
**status:** in_progress (2026-07-13 — self-executed by this session per Colin)
**pre_approved:** true (plan-mode approval 2026-07-13 — Plans/immutable-wondering-alpaca.md)

## TICK-036
**title:** Top-3 movers study — commonalities + ex-ante predictability under 7 lenses (MINING, Steps 1-2)
**description:** Colin 2026-07-14: top 3 movers/day, find commonalities + anything predictive (incl. rare blue-moon setups), many lenses, lookahead allowed. Window = the 2 on-disk full-market years (2024-07→2026-06, survivorship-free; 5-10yr extension = Polygon Developer $79/mo, operator's word). Anatomy / ex-ante tape features vs matched controls / persistence + repeat offenders / regime (VIX) / catalysts (Alpaca news, posthoc taxonomy) / blue-moon conditional scans / ex-ante watchlist scored P(>=1 of top-3) vs random. All MINING-stamped; candidate rules counted into mined_n.json; no prereg/holdout this ticket. Plan: Plans/immutable-wondering-alpaca.md.
**status:** in_progress (2026-07-15)
**pre_approved:** true (plan-mode approval 2026-07-15)

## TICK-038
**title:** Execution harness — real-quote fill measurement, unifying the two gapper shadows
**description:** Built `execution/harness.py` as a single measurement instrument replacing the forked `live_shadow.py` (HYP-093) and `hyp107_shadow.py` (HYP-107). Captures REAL SIP bid/ask at the frozen entry/exit instants (long fills at ask, short at bid), costs fills through the same `realistic_fills` module the backtests call, and emits fill_log.jsonl + daily_summary.csv. NO readiness/funding column by design. Deferred T+16min capture because this account's SIP entitlement 403s inside a 15-minute recency window (measured); real-time IEX is permitted but quotes ~10% spreads on ~2% of volume and is unusable. Also fixed the flat-10% LULD band (`backtester/luld.py`, tiered Reg NMS, doubled at open/close) — measured impact only ~7bp at the 09:31 entry bar, logged honestly. KEY FINDING: real median quoted spread on 59 archived gapper events is **0.61%**, vs the 1-15% assumed in realistic_fills, which saturates its 8% round-trip cap — every gapper backtest is biased pessimistic by several pp/trade. See data/execution/BACKFILL_NOTE_2026-07-18.md.
**depends_on:** []
**blocks:** [TICK-039 (recalibrate realistic_fills spread model against measured spreads)]
**acceptance_criteria:**
- [x] Frozen thresholds single-sourced + sha256-locked; drift fails startup
- [x] Real bid/ask capture (first in repo); gross - net == spread_cost asserted to 1bp
- [x] Same `realistic_long_return` call as the backtests, so vs_backtest_delta is meaningful
- [x] Kill switch honored despite paper-only; heartbeat before the check
- [x] 118 new tests green; full suite unchanged at 40 failed / +118 passed
- [x] plist authored + plutil OK + TRACKED-NOT-LOADED (operator installs, rebaselines watchdog)
- [ ] Operator: install plist and run `plist_watchdog.py --rebaseline`
- [ ] Re-run the SEALED HYP-107 holdout (not a random archive draw) with a recalibrated cost model
**status:** done (2026-07-18) — operator promotion pending
**pre_approved:** true (plan-mode approval 2026-07-18 — Plans/concurrent-petting-blossom.md)

## TICK-040
**title:** Six-layer system assembly — information edge through execution
**description:** Wired every existing component into one automated daily system. L1 execution/context.py (consolidated morning context, degrade-never-fabricate with FRESH/STALE/SILENT_NULL/UNAVAILABLE per source) + scripts/plist_manifest.py (authored-vs-installed reconciler). L2 execution/bias.py (recorded + scored, NEVER gating — AST tripwire enforces the ARCHITECTURE.md L1/L2 wall) + execution/obsidian.py (the vault writer Oracle-Sync-Spec specified and never built). L3 execution/signals.py (ranked GO/NO-GO, NO_GO retained with reasons). L4 harness --signals + risk routing. L5 execution/risk.py (ratified five only; daily-loss/consecutive-loss/VIX REFUSED for lack of constitutional authority → docs/proposed_amendment_art7-9.md). L6 execution/eod.py (conversion-led reconciliation → Trading/Ops/). Two plists tracked-not-loaded. SYSTEM_STATUS.md at root.
**depends_on:** [TICK-038, TICK-039]
**blocks:** []
**acceptance_criteria:**
- [x] Single consolidated morning context exists (none did — two chains never exchanged data)
- [x] Bias recorded and scored, gates nothing; wall enforced by AST test not convention
- [x] Ranked GO/NO-GO with every rejection reasoned
- [x] Fills reference signal_id and carry risk decisions
- [x] Constitutional 3.5/5/6.5 ladder implemented (was implemented NOWHERE; effective flatten was 7.5%)
- [x] EOD reconciliation written to Obsidian, conversion-led
- [x] 88 new tests; full suite unchanged at 40 pre-existing failures
- [ ] Operator: install com.alta.sentiment_update (highest value), system_morning, system_eod
- [ ] Operator: rule on proposed Articles 7-9
**status:** done (2026-07-18) — operator promotion pending
**pre_approved:** true (plan-mode approval 2026-07-18 — Plans/concurrent-petting-blossom.md)

## TICK-041
**title:** Learning-loop prerequisites — fix the label channel, refuse the learners
**description:** Six self-improvement upgrades proposed; three built in non-adaptive form, three refused with evidence. KEY CORRECTION: the "outcome loop 0/23" claim (which I made and propagated into SYSTEM_STATUS.md) was WRONG — pulse_check counted already-matched trades as attempts on every 2h pulse, so the alarm fired permanently while the loop was healthy (count grew 9→23). Fixed with an already-matched sidecar. Two real defects were hiding behind it: day-boundary match failures (~3/23, fixed with an asymmetric ±36h window + adjacent-month lookup) and backfill_decision_records.py unscheduled since 2026-07-01 (plist authored). Also built: drift tripwire (alert-only, reports both n=84 threshold-crossing and n≈177 80%-power sample sizes). REFUSED: source auto-weighting, Bayesian threshold updates, VIX gate (3rd recurrence — root cause CLAUDE.md:134 formatting example, now fixed), XGBoost on 50 EOD rows, real-time paper orders (SIP bars 15-min delayed → 09:31 signal unreadable until 09:47). GOVERNING FACT: <34 feature-complete live labels exist; 3,460 ICT records are backtest replay.
**depends_on:** [TICK-040]
**blocks:** []
**acceptance_criteria:**
- [x] Outcome alarm no longer false-positives; emits matched/attempted/already_known/unmatchable
- [x] Day-boundary matching fixed, asymmetric (no look-ahead), adjacent-month aware
- [x] backfill_decision_records.py scheduled (plist tracked-not-loaded)
- [x] Drift tripwire alert-only, power honestly reported
- [x] Refusals documented with evidence in docs/learning_loop_prerequisites.md
- [x] CLAUDE.md:134 VIX example fixed at root cause
- [x] SYSTEM_STATUS.md 0/23 claim corrected
- [ ] Operator: install decision_backfill, system_morning, system_eod plists
- [ ] Remaining: bias scoring wired into EOD; post-mortem log (deferred to next pass)
**status:** in_progress (2026-07-18)
**pre_approved:** true (plan-mode approval 2026-07-18)

## TICK-042
**title:** Sentiment pipeline: per-feed timeout, then install (11-day dark board fixed)
**description:** The sentiment job could not be scheduled as-is: it ran unbounded. One observed run survived 1h47m still holding an EXCLUSIVE DuckDB write lock (blocking every reader of sentiment.db) and never reached board_state.rebuild() — the artifact the 08:00 forex scan reads. Measured with the lock free: news 5.8s / macro 3.0s / vix 0.6s / surprise 5.3s / cot 4.0s (~19s), vs gdelt / vrp / surface unbounded (>90s each). Added FEED_TIMEOUT_S=60 per feed; abandoned feeds are logged loudly and the pipeline CONTINUES so the board always rebuilds. FeedTimeout inherits BaseException, not Exception — the first attempt failed because gdelt_feed.py:58 retries inside `except Exception` and swallowed the alarm. Result: exit 0 in 200s, board 07-06 → 07-17, look-ahead audit 0 violations / 165,385 rows. Also fixed a Layer-1 bug: sentiment_board freshness was measured from max DATA date, which is bounded by the market calendar and would read STALE on every healthy run; now measured from rebuild time with data-stall checked separately.
**depends_on:** [TICK-040]
**blocks:** []
**acceptance_criteria:**
- [x] Job completes bounded (200s) and always rebuilds the board
- [x] Degraded feeds abandoned + logged, not fatal
- [x] Look-ahead sentinel still fires (0 violations, 165,385 rows)
- [x] Board freshness measured correctly (rebuild time, not data date)
- [x] Installed, loaded, plist_watchdog rebaselined GREEN 27/27
**status:** done (2026-07-19)
**pre_approved:** true (operator instruction to install)

## TICK-043 (Finding 4A)
**title:** Daily halt gate is inert on fresh trading days — needs real-time P&L feed or intraday position tracking
**description:** `DAILY_LOSS_HALT` (2% daily loss gate, ratified in `RISK_FRAMEWORK.md`) is effectively inert at the start of each trading session. `AccountState` is constructed once at session start (`run_session`) from fills already persisted in `fill_log.jsonl`. Because no fills for the current day exist yet at session start, `daily_pnl_frac` evaluates to `0.0` and the halt gate cannot fire. The gate would only fire if the harness were restarted mid-session (with fills already written), which is not the normal scheduled path.

**Impact:** The ratified -2% daily halt never fires in the standard launchd path. Consecutive-loss gates (`CONSEC_LOSS_HALVE` / `CONSEC_LOSS_HALT`) are correctly computed from all historical fills and work as intended.

**Required fix:** Either (a) a real-time P&L feed that updates `AccountState` after each fill is recorded intra-session, or (b) intraday position tracking that recomputes `daily_pnl_frac` dynamically between fills within the same session.

**Constraint:** `DAILY_LOSS_HALT` is part of the ratified `RISK_FRAMEWORK.md`. Any implementation must be reviewed, logged in `data/agent/param_change_log.jsonl`, and this document updated per the amendment procedure.

**depends_on:** []
**blocks:** []
**status:** backlog — **requires its own unlock before implementation**
**pre_approved:** false
