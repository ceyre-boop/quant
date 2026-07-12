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
- [ ] P0: prereg hash-locked + ledger PREREGISTERED entry BEFORE any data read; --verify green; regression test enforces gate-zero-first
- [ ] P1: reconcile abort gate green (0.6886±0.01) BEFORE freeze; spot/SPY/VIX/FRED parquets + sha256 manifest; 64 ungated signal builds + external causal VIX mask → signals.npz
- [ ] P2: 1,540 kernel runs; daily M2M decomposition sums to kernel pnl_pct (1e-12); config #385 reproduces canonical backtest_all snapshot per pair (1e-9)
- [ ] P3: truncation-invariance (≥20 sampled t, A1+A2) + regime-causality tests green
- [ ] P4: 6 runs + costed variants + 1,500 placebo replays; verdict per locked criteria sealed to ledger; charts + summary_report + results.json under data/research/modern/; NEXT.md; pushed
- [ ] 12 module tests green; determinism (2 subprocesses byte-identical); no writes outside data/research/modern/ + prereg + ledger
**status:** in_progress (2026-07-11 — Claude Code / Molly; plan approved in-session)
**pre_approved:** true (plan-mode approval 2026-07-11 — Plans/glistening-juggling-clover.md)
