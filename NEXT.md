# NEXT — Alta quant session log (authoritative, repo-native)

Per-session ledger: what shipped, push status, verdicts, blockers, refusals. Newest first.
The Obsidian brain (`~/Obsidian/Obsidian/00-BRAIN/NEXT.md`) is the cross-project rollup.
Standing constraints live in `CLAUDE.md` — not restated here.

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
