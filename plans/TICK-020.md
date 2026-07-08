# Political-Alpha Event Study — Build Plan (HYP-085 / TICK-020)

## Context

Colin greenlit a pre-registered event study testing whether Trump public statements produce
statistically detectable abnormal market moves, with an advance-positioning (rr25) overlay.
The hypothesis and every operational definition were locked in a chat session on 2026-07-08 and
written to the vault as the governing spec:

- **Spec (LAW):** `~/Obsidian/Obsidian/Trading/Research/Political-Alpha-Claude-Code-Spec.md`
- Companion brief: `~/Obsidian/Obsidian/Trading/Research/Political-Alpha-Hypothesis-2026-07.md`

The spec is self-contained: H0/H1, locked definitions (§2), window mechanics (§6), four build
phases (§7), hard constraints (§8), Definition of Done (§9), out-of-scope (§10), output manifest
(§12). **This plan implements the spec exactly — it does not modify any locked definition.**
The one decision the spec deferred to Colin (Phase 0 ledger registration) defaults here to the
recommended path: **register HYP-085** (approval of this plan = the confirmation; say the word to
flip to spec-as-prereg instead). Colin was asked directly and was AFK — flag at kickoff if he
answered elsewhere.

Verdict-neutral by design: **null-not-rejected is a valid, publishable outcome.** A rejected null
is a *candidate result*, not an edge (spec §10) — promotion goes through the standard gauntlet as
a separate ledgered step later.

## Verified repo facts (exploration, 2026-07-08)

- Branch `sovereign-v2`; canonical ledger `data/agent/hypothesis_ledger.json` (JSON array, max
  HYP-084 → **HYP-085 free**, verified `rg "HYP-085"` = nothing). Dashboard ledger
  (`ict-dashboard/.../hypothesis_ledger.json`) is dormant — **do not touch**.
- Next ticket = **TICK-020** (`tickets/backlog.md`, schema: id/title/description/depends_on/
  blocks/acceptance_criteria/status/pre_approved).
- **ThetaTerminal v3 is UP** on `http://127.0.0.1:25503` (probed live). Only 2 endpoints exist
  in-house: `/v3/option/list/expirations?symbol=` and
  `/v3/option/history/eod?symbol=&expiration=&start_date=&end_date=` (CSV; no greeks → invert IV
  yourself). Localhost sends **no auth header**. Depth wall ≈ 2020 → 2025→present fully covered.
  403 → retry once (sleep 5) then skip-and-count; 472 = NO_DATA; 10-consecutive-failure breaker.
- Copy-sources (COPY, never import — isolation): rr25 smile math
  `sovereign/sentiment/options_surface_feed.py:62-181` (`_strike_iv`, `_call_delta`,
  `_interp_delta_iv`, `smile_read`, `_pick_expiry`) + `sovereign/sentiment/vrp_feed.py:69-105`
  (`_phi`, `_bs76_call`, `implied_vol_atm`); ThetaData transport
  `sovereign/research/vrp/data_loader.py:120-260`; env parse
  `sovereign/sentiment/store.py:279-296`; AST isolation test
  `tests/test_sentiment_board.py:153-167`.
- rr25 = iv(25Δ call) − iv(25Δ put); per-OTM-strike Black-76 bisection IV (puts→call via parity),
  per-strike call delta, keep Δ∈[0.02,0.60], linear interp in delta space at 0.25; expiry = DTE
  nearest 30 in [20,45]; `min_strikes=5` else None; flat r=0.04. FXE symbol literal = `"FXE"`.
- yfinance: `yf.download(t, start=…, auto_adjust=True, progress=False)`; flatten MultiIndex
  (`df.columns = df.columns.get_level_values(0)`); daily → `.tz_localize(None).normalize()`;
  lazy import. Python **3.14** (no numba). scipy 1.17 + matplotlib 3.10 available (matplotlib is
  ambient/undeclared — if missing in a fresh env, stop and say `pip install matplotlib`; don't
  edit requirements.txt). **Avoid parquet/duckdb** (undeclared) — cache as CSV/JSON.
- `.gitignore`: `research/` tracked (outputs commit by default — desired); root `data/raw/`
  pattern is anchored and does NOT cover our tree (verified `git check-ignore`); `Plans/`
  ignored, `plans/` requires force-add per convention. pytest `testpaths = tests` → our tests run
  only via explicit `python3 -m pytest research/political_alpha/tests/ -q`.

## Phase 0 — Pre-registration + ticket (BEFORE any event data is read)

1. **TICK-020** in `tickets/backlog.md` (re-check max TICK at write time — concurrency rule):
   title "Political-alpha event study (HYP-085) — build research/political_alpha/ per vault spec";
   acceptance_criteria = spec §9 DoD; `pre_approved: true` (this plan + Colin's greenlight).
2. **Prereg** `data/research/preregister/HYP-085_political_alpha_trump_events.json`, mirroring
   HYP-082's exact key set/order (`id, slug, family, name, status:"PREREGISTERED", frozen_at,
   prior_materials_banner, thesis, observation_definition, primary{...}, success_criteria,
   failure_criteria, secondaries, validation_protocol{...}, prior_expectation:"NOT_SIGNIFICANT",
   verdict:null, universe, hash_method, hash_lock`). Content = spec §1–§2 + §7-Phase-4 tests +
   the pinned formulas below (they are implementation pre-declarations, locked with the hash).
   `family: "NONE (standalone; single pre-stated test — no BH family)"` → **no family manifest**.
   Bootstrap block: `{n: 10000, seed: 42, p_formula: "(n_ge + 1) / (N + 1)"}`.
   Hash: `sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(",",":")))` —
   verbatim `hash_method` string from HYP-082:48.
3. **Ledger append** to `data/agent/hypothesis_ledger.json` mirroring
   `scripts/research/preregister_positioning.py:264-304`: load array → `.bak-{UTC}` backup
   (`shutil.copy2`) → refuse duplicate id → append entry (id/name/status PREREGISTERED/
   date_registered/family/hash_lock/prereg_file/mechanism/prior_expectation/result:null/
   p_value:null/verdict:null) → atomic tmp-file replace. Verify: recompute hash == stored.
4. Copy this plan to `plans/TICK-020.md` + `git add -f` (plans/ gitignored by convention).
5. Commit: `[RESEARCH] Political-alpha Phase 0: HYP-085 pre-registration + TICK-020`.

## Module design

All new code in `research/political_alpha/` (NEW dir). **Imports nothing** from `sovereign/`,
`ict/`, `ict-engine/`, `config/`, `audit/`, `scripts/` — copied code carries provenance comments.
No `__init__.py` anywhere (phases run as `python3 research/political_alpha/<phase>.py`; script
dir on `sys.path[0]` makes `import _lib` work; tests get a 3-line `conftest.py` sys.path shim).

```
research/political_alpha/
├── _lib.py                      # env parse · jsonl IO (atomic) · fetch_daily (yf, CSV cache)
│                                # · map_t0 · log_returns · trailing_sigma60
│                                # · ThetaClient (2 endpoints, CSV cache, 403/472/breaker)
│                                # · pivot_chain (call/put volume kept SEPARATE)
│                                # · smile math verbatim · rr25_for_day
├── classification_rules.py     # POLICY/ACTION/ENTITY regex tables + SCHEDULED_MACRO_DATES
│                                # (static 2025-26 FOMC/CPI/NFP list w/ source-URL comments,
│                                #  VERIFY-AT-BUILD) + STUDY_START="2025-01-20"  — data only
├── build_event_catalog.py      # Phase 1
├── compute_abnormal_returns.py # Phase 2
├── check_positioning_signal.py # Phase 3
├── run_statistical_tests.py    # Phase 4 (only importer of matplotlib(Agg)/scipy.stats)
├── data/
│   ├── .gitignore              # "raw/\ncache/\n"  (jsonl outputs stay tracked)
│   ├── manual_events.jsonl     # curated ≤50 fallback (committed; primary source_url per row)
│   ├── trump_events.jsonl      # P1   ├── dropped_events.jsonl  # P1 de-dup log
│   ├── event_study_results.jsonl  # P2   ├── manipulation_flags.jsonl  # P3
│   ├── raw/   (uncommitted scrape/API cache)   └── cache/  (yf + theta CSV caches)
├── output/                      # normality_plot.png · sd_test_results.json · summary_report.md
└── tests/
    ├── conftest.py
    ├── test_isolation.py        # AST walk over rglob("*.py"): top-level import roots ∉
    │                            # {sovereign, ict, ict_engine, config, audit, scripts, layer1, layer2}
    └── test_lib.py              # map_t0 branches · 5-day dedup · smile_read on synthetic chain
```

Universe/asset classes: `EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, DX-Y.NYB` → `fx`;
`XLE, SLX, XLF, KWEB, GLD` → `us_etf`. `fetch_daily` start `2023-06-01` (>380 bars before
2025-01-20 → covers T-252 + 60d SD warmup).

## Pinned formulas (pre-declared; locked into the HYP-085 prereg hash)

Daily **log** returns. Hourly NOT used (spec §6 daily default — stated in the report).

- **(a) T0 mapping** — `fx`: first bar date ≥ statement's UTC date. `us_etf`: ET date if in index
  and ET time < 16:00, else next index date. Calendars from each instrument's own price index.
- **(b) Post window** — `post_return = r_T0 + r_T1` (= ln(Close_T1/Close_P0));
  `abnormal_return = post_return − 2·est_mu`; `std_abnormal_return = abnormal_return/(est_sigma·√2)`.
  Estimation = returns at positions [T0−252, T0−10] (≈243 obs; ≥200 non-NaN else
  `data_ok:false`). **big_move = |r_T0| > 2·σ60(T0) OR |r_T1| > 2·σ60(T1)**,
  σ60 = `rolling(60).std(ddof=1).shift(1)` (never includes the tested day; <60 prior → null+gap).
- **(c) pre_rr25_move = rr25(D0−1) − rr25(D0−3)`** on FXE's own calendar (us_etf mapping ⇒ both
  EODs strictly pre-statement) = rr25 drift across the T-48h→T-0 window. Missing either EOD →
  null, condition (2) unevaluated, `gap_reason` — **never widen the lookback**.
- **(d) Direction** — `expected_bull` for the option underlying: native ETF chain →
  sign(post_return) of the tagged ETF; FXE proxy (all forex rows, per spec) → USD-leg map:
  resolution USD-bearish (EURUSD/GBPUSD/AUDUSD up, or USDJPY/DXY down) ⇒ expected_bull=+1 for
  FXE. `rr25_directional = sign(pre_rr25_move) == expected_bull` (zero = NOT directional).
- **(e) Volume leg** — put/call volume ratio on the same ~30d expiry:
  `pre_pcr_move = pcr(D0−1) − pcr(D0−3)`; `volume_directional = sign(pre_pcr_move) == −expected_bull`.
  Condition (2) = `rr25_directional OR volume_directional` (legs recorded separately;
  evaluable-only). `manipulation_signal = post_big_move AND pre_directional` — descriptive count;
  the p-value comes only from the bootstrap.
- **(f) Bootstrap (Test 3)** — placebo unit = **statement** (preserves within-statement
  cross-asset correlation): one random timestamp (eligible dates, pinned 12:00 ET) per real
  evaluable statement, applied to all its instrument rows via the same map_t0. Eligible: inside
  study period; ≥60 prior returns + valid P0/T0/T1 on every tagged instrument; not within ±5
  trading days of any real event T0 on those instruments; T0/T1 ∉ SCHEDULED_MACRO_DATES.
  10,000 sets, `numpy.random.default_rng(42)`; statistic = big_move exceedance **rate** over
  evaluable rows (identical computation to observed); **p = (n_ge + 1)/(10001)**, one-sided.
- **Test 1** — pooled standardized pre-window returns (P2, P1 days) → QQ vs N(0,1) +
  `scipy.stats.shapiro`; direction-aligned skew (down-events sign-flipped).
  `normality_plot.png` = 2 panels: QQ + bootstrap-null histogram w/ observed-rate line.
  **No tests beyond these three** (spec §10 — no BH/permutation/CAR here).

## Build phases (one commit each; verification inline)

**P1 — event catalog** (`build_event_catalog.py`; riskiest phase)
Fetchers, every raw response cached to `data/raw/<source>/`, `--offline` replays cache:
1. Federal Register JSON API (no key): `/api/v1/documents.json?conditions[presidential_document_type]=executive_order&conditions[president]=donald-trump&conditions[signing_date][gte]=2025-01-20&per_page=100…` + `raw_text_url` bodies. Signing date has no clock → pin 12:00 ET → UTC, noted per row.
2. whitehouse.gov `/presidential-actions/` + `/briefings-statements/` listings (structure VERIFY-AT-BUILD; stdlib re/html.parser — bs4 undeclared, don't add). Zero-parse → loud warning, continue.
3. Truth Social probe chain (first hit wins; all cached): factba.se → trumpstruth.org mirror → truthsocial.com Mastodon API (one polite attempt; don't fight Cloudflare). All fail → proceed, gap stated in note + report.
4. `manual_events.jsonl` (spec-authorized top-50): every row needs a primary source_url (TS permalink / WH page / C-SPAN timestamped — C-SPAN enters ONLY here); scraped row beats manual duplicate (same UTC date+instrument+policy_action → drop log).
Deterministic classifier (no LLM): `qualifies = ENTITY_MAP ∧ POLICY_PATTERNS ∧ ACTION_PATTERNS`
(matched literals recorded in `notes`); entities outside the 10-instrument universe →
`qualifies:false`, never stretched onto DXY. One row per event×instrument; IDs `PA-0001…` per
statement sorted by (timestamp, source, url). Per-instrument 5-trading-day keep-first de-dup →
`dropped_events.jsonl`. Rule tables committed with P1, authored before the classifier runs on
scraped text; any post-first-run rule edit disclosed in the Phase-1 note.
**Honesty gate:** <30 qualifying events → proceed + state shortfall; never loosen.
*Verify:* isolation+unit tests green; 2025-04-02 "Liberation Day" reciprocal-tariff event present
(multi-instrument), a steel event → SLX, a China event → KWEB; all 10 tickers ≥380 rows;
drops populated where events cluster.
Commit: `[RESEARCH] Political-alpha Phase 1: Trump statement event catalog + classification`.

**P2 — abnormal returns** (`compute_abnormal_returns.py`): apply (a)/(b) per row →
`event_study_results.jsonl`. No silent mocking: short/missing data → `data_ok:false` + reason.
*Verify:* every P1 row present; `n_est_days`≈243; hand-check one known event's
std_abnormal_return. Commit: `[RESEARCH] Political-alpha Phase 2: abnormal return event-study calculator`.

**P3 — positioning** (`check_positioning_signal.py`): probe
`GET :25503/v3/option/list/expirations?symbol=FXE` first — terminal down → **exit(2), print
"start ThetaTerminal (:25503); re-run"**, write nothing. Forex rows → FXE proxy; ETF rows →
probe native chain (XLE/SLX/XLF/KWEB/GLD), unavailable → `positioning_available:false` + reason,
condition (2) unevaluated — never synthesize. → `manipulation_flags.jsonl`.
*Verify:* rr25_source distribution, NaN counts, breaker never tripped; per-symbol availability
table in the note. Commit: `[RESEARCH] Political-alpha Phase 3: pre-announcement rr25 positioning + manipulation flags`.

**P4 — tests + report** (`run_statistical_tests.py`): exactly the three tests → 3 outputs.
`sd_test_results.json`: observed_rate, placebo mean/CI, p_value, n_boot 10000, seed 42, shapiro,
aligned skew. `summary_report.md`: event count + shortfall, honest verdict (rejected / NOT
rejected — both valid), manipulation-flag count, all gaps, "hourly not used — daily per §6",
small-N power statement. Re-run isolation test.
Commit: `[RESEARCH] Political-alpha Phase 4: statistical tests + normality/SD charts + summary`.

**Optional (only if trivially available):** Quiver STOCK-Act cross-check is secondary/stretch —
skip with a note unless the free endpoint returns usable rows (30–45d lag understood).

## Stop conditions / risks

- ThetaTerminal down at P3 → stop + say what's needed (P1/P2 unaffected). No mocking, ever.
- Truth Social all-probes-fail (likeliest risk) → FR+WH+manual carry the catalog; venue-coverage
  gap stated everywhere. FR EOs alone (100+ in 2025) likely clear 30 events.
- Thin ETF chains (SLX/KWEB) → condition (2) unevaluated w/ reason, counts surfaced.
- DX-Y.NYB quirks (index: no volume, stale bars) → own-calendar mapping + data_ok gating; never
  substitute UUP.
- Small N (~30–60 rows vs ~9% two-day baseline exceedance) → only large effects detectable;
  reported honestly; null-not-rejected is pre-declared valid.

## Verification (end-to-end)

1. `python3 -m pytest research/political_alpha/tests/ -q` → green (isolation + unit).
2. `python3 -m pytest tests/ -q` unchanged (we touch no live code; ledger append is data).
3. Re-run each phase script → idempotent from caches (`--offline`).
4. Prereg hash recheck == stored `hash_lock`; ledger entry present, status PREREGISTERED.
5. Spec §9 DoD checklist walked line-by-line in the Phase-4 Obsidian note.

## Reporting & session close (spec §11)

- Obsidian per phase: `Trading/Research/Political-Alpha-Phase{1..4}-2026-07.md`.
- Repo `NEXT.md` entry: shipped, push confirmation, verdict, gaps, refusals.
- Brain rollup (`00-BRAIN/{CONTEXT,NEXT}.md`) once, after Phase 4.
- 5 commits total (P0–P4), push `sovereign-v2` at least once (standing constraint).

**Size:** ~8 new source files + tests, ~1 prereg JSON + 1 ledger append + 1 ticket entry.
Nothing under `sovereign/ ict/ ict-engine/ config/ audit/ scripts/` is touched.
