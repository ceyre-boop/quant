# Appendix — Raw Intel Sections (workflow wf_ed5c33bf)


---

## FOMC-Readiness Inventory
As of 2026-07-24 23:43 EDT (2026-07-25 03:43Z). FOMC forcing event: 2026-07-29, statement 2:00pm ET.

### 1. MT5 bridge + FOMC-window logger (TICK-056)
- **Spec (LAW):** `/Users/taboost/quant/specs/mt5_bridge.md` — demo-only bridge, human-approval routing, live path dead code (needs BOTH `ALTA_MT5_ALLOW_LIVE=1` env AND `data/execution/mt5_LIVE_UNLOCK.json`; neither exists).
- **VM-side runbook: EXISTS** — "Operator runbook (Mac side, Option A)" in `/Users/taboost/quant/NEXT.md` (2026-07-24 entry, lines 239–247) plus spec §9. Option A chosen: Windows 11 ARM VM via UTM on Apple Silicon.
- **FOMC-window logger: BUILT** — `/Users/taboost/quant/scripts/fomc_window_logger.py` (modified Jul 24 10:46). Pure observation, places no orders. Defaults: center 2026-07-29T14:00 ET, ±15 min window, 1.0s sampling, symbols EURUSD/GBPUSD/AUDUSD/GBPJPY/USDJPY. Writes `data/execution/fomc_window_<date>.jsonl`. Shares the bridge's `MT5Connector` + demo guard. Tests: 41 (bridge) + 10 (logger) = **51 passed**.
- **What remains before it can log the FOMC window (all operator/VM-side — code side is done):**
  1. Install UTM + Windows 11 ARM VM (MetaTrader5 pip package is Windows-only; verified unimportable on this Mac).
  2. In VM: install MT5 terminal, log into The5%ers DEMO server, enable Algo Trading, add the 5 symbols to Market Watch.
  3. VM Python: `pip install MetaTrader5 pyyaml`; clone/sync repo into VM.
  4. `python mt5_bridge.py --selftest` → fill exact server name + symbol suffixes into `config/mt5.yml` (currently placeholders, no secrets).
  5. Arm before 2:00pm ET Jul 29: `python scripts/fomc_window_logger.py` (waits for window start).
  - Also outstanding (separate tickets): no producer emits `order_intent` yet; no live unlock.

### 2. System regime state — `data/agent/system_regime_state.json`
Generated 2026-07-25T03:34Z (**9 min old, fresh**). Writer: `scripts/build_system_regime.py`.
- Overall: **DEGRADED** — "carry:STALE; es_nq:OK; macro:OK; portfolio:UNAVAILABLE".
- **Carry verdict: STAND_ASIDE, size_multiplier 0.0** — but it keys on **freshness of `data/agent/forex_proximity.json`**, which is 13.9h old > 12.0h limit. This STAND_ASIDE is a staleness fallback, not a market read. **FLAG: stale input, not a signal verdict.**
- es_nq: INFO, regime `ROTATION_WARN` (NQ weak vs ES, corr 0.893), source 1.8h old.
- macro: INFO — VIX 18.58, 10y–2y 0.628, 10y–3m 0.874, HYG 79.23, DGS10 4.679 (0.4h old).
- **Portfolio block: UNAVAILABLE — "no unified position-by-cluster ledger on disk; open exposure and the drawdown breaker cannot be read."** daily_drawdown_limit 2% and max 5 concurrent positions shown from config only. **FLAG: the Art. 3 drawdown breaker is not observable from this feed going into FOMC.**

### 3. Crisis library (Alexandrian)
Two distinct modules:
- `sovereign/risk/alexandrian_library.py` — the operational 10-volume library: **63 LibraryEntry episodes**. Threat ladder (composite score → size): NORMAL 0.00–0.35 → 1.00x; ELEVATED 0.35–0.52 → 0.80x; WARNING 0.52–0.68 → 0.50x; DANGER 0.68–0.82 → 0.25x; **CRITICAL 0.82–1.00 → 0.00x**. Bull-regime floor: severity −1 top match at 0.00x is floored to 0.50x (May 2026 GOLDILOCKS 0.9338 incident).
- `sovereign/features/alexandrian_library.py` — features-layer matcher: **8 scenarios**; `SIMILARITY_HIGH = 0.85` (log + operator flag only), `SIMILARITY_LOW = 0.50`.
- **There is no literal ">90% similarity" size-cut in code.** Closest mechanisms: the CRITICAL ≥0.82 composite → 0.00x cut, and the 0.85 high-similarity flag (log-only).
- **Current reading** (`logs/scanner_state.json`, last_scan 2026-07-24T19:59Z, ~7.7h old): **sim = 0.187** (below 0.30 floor), regime UNKNOWN, threat NORMAL, threat_score 0.0, size_modifier 1.00 — no confident match.
- Health: `data/health/alexandria_status.json` 2026-07-24T22:07Z (~1.6h old): **PASS** — 1 sample, low-sim abstained correctly, 0 wrong, 0 errors.
- **Live wiring: YES on the ICT path** — `ict/library_bridge.query_library()` → `ict/orchestrator.py` (per-scan query, `G1_LIBRARY` regime veto gate, `size_modifier` passed into `ict/daily_bias.py` pair biases). **NOT wired on the carry path** — `scripts/forex_live_scan.py` has zero library references.

### 4. Risk constitution + kill switch
`RISK_CONSTITUTION.md` — RATIFIED v1.0.0, 2026-07-07. Verbatim binding numbers:
- Art. 1 per-trade: "No single trade may risk more than **0.75%** of account equity."
- Art. 2 carry complex (one bet): "Total simultaneous open risk across all carry positions may not exceed **2.5%** of equity."
- Art. 3 circuit breakers (peak-to-trough, account level): "At a drawdown of **3.5%**, all new position sizes are halved. At **5%**, new entries halt. At **6.5%**, every predictive-layer position is flattened." (Anchored below the 8% TRAILING prop halt.)
- **FLAG:** "Enforcement reconciliation PENDING, blocked_on: shadow_close (~2026-07-28)" — tier-config clamps land only when the freeze lifts, **one day before FOMC**. Combined with §2's unreadable drawdown breaker, Art. 3 currently binds by law + research-level gates, not by live clamp.
- **Kill switch: NOT ENGAGED** — `data/system/KILL_SWITCH` does not exist (`data/system/` holds only README + plist_watchdog_baseline.json). Trading path is thawed.

### 5. Paper accounts (both generated 2026-07-25T03:39Z — 4 min old, fresh)
- `data/agent/carry_paper_account.json` — Carry paper (OANDA practice): **balance $109,627.66, total_pnl −$30.45**, 157 equity points, status OK. Note in-file: not shown on dashboard (prop panel reads only prop_account_balance.json).
- `data/agent/prop_account_balance.json` — Undertow paper shadow (HYP-093): **balance $200,099.60, total_pnl +$99.60**, today 0.0, drawdown_used 0.0, days_trading 10, signals_to_date 3, last_signal_date 2026-07-14. **FLAG (labeling, not staleness):** the $200K/$10K prop framing is explicitly cosmetic — this is an own-capital strategy shadow, and returns are at constitutional sizing, not F2+F3 scaling.

### 6. Training ignition flags — `config/training.yml`
- `ignition.tick_024_carry_fix_landed: false` (BLOCKER 8.1 — carry-cost fix not landed; reward would train on fictional carry).
- `ignition.hyp_071_net_confirmed: false` (BLOCKER 8.2 — needs NET-return recompute + CONFIRMED ledger stamp).
- **Gate CLOSED — runner is SCAFFOLD/DRY only.** Additional locks: `value_function.gross_marker_key: gross_R_caveat` (marker's presence keeps gate closed regardless of flags), `placebo.enabled: true` (fail-closed), `director.auto_approve: false`.

### Stale / contradictory summary
1. `data/agent/forex_proximity.json` 13.9h old (> its own 12h limit) → carry STAND_ASIDE is a freshness artifact; fix the feed before reading it as a market verdict.
2. Portfolio drawdown breaker **UNAVAILABLE** (no position ledger) while Art. 3 clamp reconciliation is deferred to ~Jul 28 — the circuit-breaker ladder is unenforceable/unobservable at portfolio level going into the Jul 29 FOMC.
3. The tasked ">90% size-cut" does not exist verbatim; the real cut is CRITICAL ≥0.82 → 0.00x, live only on the ICT path, not carry.
4. FOMC logger critical path is entirely operator-side (Windows VM + The5%ers demo login) — zero code remaining, but nothing can be logged until the VM runbook steps 1–5 are executed.
5. Nothing among the inventoried files exceeds 48h staleness; oldest live reading is scanner_state.json at ~7.7h.

---

All data verified. Composing the final section.

## G10 Rate Landscape vs System-Recorded Differentials

**As-of 2026-07-24.** Macro backdrop: renewed Middle East (Iran) hostilities and the oil rebound have flipped G10 central banks hawkish in 2026 — ECB hiked in June (first in 3 years), RBA hiked 3× (Feb/Mar/May), BoJ hiked to a post-1995 high, and BoE hike bets are building. The Fed is the outlier still on hold.

### Central bank status

| Bank | Policy rate (now) | Last move | Next meeting | Market-priced odds for that meeting | Imminent catalysts |
|---|---|---|---|---|---|
| **Fed** | 3.50–3.75% (mid 3.625%) | Cut −25bp, eff. 11 Dec 2025 | 28–29 Jul 2026 (Wed 29, 2pm ET) | Hold expected, hawkish optionality; no SEP at this meeting | Oil pass-through into CPI/PCE |
| **BoE** | 3.75% | Cut −25bp (4.00→3.75), 18 Dec 2025 | **30 Jul 2026** + new Monetary Policy Report | Hold ~78.5–86% priced; +25bp to 4.00% ~14–21% (sources differ — exact % UNVERIFIED). ~Two 25bp **hikes** priced by Mar 2027 (as of 22 Jul) | June CPI 2.6% y/y (22 Jul, below 2.7% f/c) cooled hike bets; June MPC vote 7–2 with 2 hike dissents; Pill publicly saying rates "will need to rise" |
| **ECB** | Deposit 2.25% (refi 2.40%) | **Hike +25bp, 11 Jun 2026** (eff. 17 Jun; first hike in 3 yrs); held 23 Jul (unanimous, some governors floated a hike) | **10 Sep 2026** | **~93% priced for +25bp to 2.50%**; two hikes priced by end-2026 | Energy-shock pass-through/second-round effects; Lagarde flags upside inflation risk from oil |
| **RBA** | 4.35% | Hike +25bp (4.10→4.35), 5 May 2026 (3rd of 2026: Feb 3.60→3.85, Mar 3.85→4.10, May); unanimous hold 15–16 Jun | **10–11 Aug 2026** + quarterly SoMP | Hold ~75% / hike to 4.60% ~25%; one source cites ~36% hike odds post-jobs-surge (range UNVERIFIED — ASX rate tracker moves daily) | **Q2 CPI 29 Jul** (5 days out) is the swing input; May CPI 4.0% y/y, trimmed mean 3.6% (highest since Sep 2024); RBA/Treasury see ~5% peak mid-2026 |
| **BoJ** | 1.00% | **Hike +25bp to 1.00%, 16 Jun 2026** (highest since 1995; Ueda hospitalized, absent) | **30–31 Jul 2026** (next week) | Hold strongly expected; growth-forecast upgrade likely; OIS ~69% chance of next hike **by October**; ~71% of economists see ~1 hike per 6 months | Quarterly Outlook Report (31 Jul); Ueda health/succession optics; weak-yen inflation pressure |

### Differential cross-check (foreign − USD, vs system-recorded)

System's fed funds 3.63% **matches** the actual 3.50–3.75% midpoint (3.625%). Every foreign leg is wrong:

| Pair | System diff | Actual (24 Jul 2026) | Error | Flag |
|---|---|---|---|---|
| GBPUSD (BoE−Fed) | **−0.60** | 3.75 − 3.63 = **+0.12** | −0.72pp | **WRONG SIGN.** Implies BoE at 3.03% — never true. Actual carry is mildly GBP-positive, and market prices BoE hikes into 2027 |
| EURUSD (ECB−Fed) | **−2.08** | 2.25 − 3.63 = **−1.38** | −0.70pp | Overstates EUR carry drag by 70bp; implies ECB at 1.55% (ECB never below 2.00%). Gap narrows to −1.13 if the ~93%-priced Sep hike lands |
| AUDUSD (RBA−Fed) | **+0.13** | 4.35 − 3.63 = **+0.72** | −0.59pp | Understates AUD carry by 59bp; misses all three 2026 RBA hikes (3.60→4.35). Implies RBA 3.76% — that's roughly the pre-Feb-2026 world |
| USDJPY (Fed−BoJ) | **+3.489** | 3.63 − 1.00 = **+2.63** | +0.86pp | Overstates USDJPY carry by ~86bp; implies BoJ at 0.14%, vs actual 1.00% and a further-hike (Oct) trajectory compressing the diff toward ~2.4 |

**Diagnosis:** the four recorded differentials are mutually inconsistent with actual rates on *any* single date — a mixed-vintage stale snapshot (the EURUSD −2.08 leg matches ECB 2.25% vs EFFR ~4.33%, i.e. roughly Apr–Jun **2025**), not a live feed. All four miss the 2026 hiking cycle entirely. Direction of risk: system currently over-rewards short-JPY carry (+86bp phantom), penalizes GBP longs that actually carry positive, and under-sizes the AUD leg. Recommend re-basing the rate-differential feed before the next scan; note this is consistent with the known SWAP_RATES_ANNUAL mis-modeling already ticketed (TICK-024).

**Verification notes:** All policy rates/last moves/meeting dates confirmed against ≥2 sources incl. central-bank primary pages. UNVERIFIED items: exact BoE July hold probability (78.5% vs 86% across sources), exact RBA August hike odds (25–36% range, moves daily on ASX tracker), and Kalshi/Polymarket contract prices (not fetched directly).

Sources:
- [BoE June 2026 Monetary Policy Summary (hold 3.75%, 7–2)](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/june-2026) · [BoE Dec 2025 cut to 3.75%](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/december-2025) · [BoE July 2026 MPC page](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/july-2026) · [CNBC Dec 2025 cut](https://www.cnbc.com/2025/12/18/bank-of-england-cuts-interest-rates-to-3point75percent.html) · [HomeOwners Alliance July preview](https://hoa.org.uk/news/interest-rate-predictions-2/) · [centralbank.watch BoE odds](https://centralbank.watch/bank-of-england/) · [Traders boost BoE hike bets](https://cryptobriefing.com/bank-of-england-rate-hike-bets-2026/) · [MoneyWeek UK CPI June 2.6%](https://moneyweek.com/economy/news/live/inflation-cpi-june-2026-report) · [ONS CPI June 2026](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/june2026)
- [ECB monetary policy decision 23 Jul 2026 (hold)](https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.mp260723~29f24d99bc.en.html) · [ECB June 2026 hike decision](https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.mp260611~4d41bd5e83.en.html) · [Euronews ECB first hike in 3 years](https://www.euronews.com/business/2026/06/11/ecb-raises-interest-rates-for-the-first-time-in-three-years-as-iran-war-fuels-inflation) · [CNBC: traders see September hike](https://www.cnbc.com/2026/07/23/interest-rate-hike-iran-european-central-bank.html) · [centralbank.watch ECB odds (93% Sep)](https://centralbank.watch/european-central-bank/) · [Central Banking: ECB holds at 2.25%](https://www.centralbanking.com/central-banks/monetary-policy/monetary-policy-decisions/7976445/ecb-holds-rates-at-225-in-line-with-expectations)
- [RBA 2026 decisions](https://www.rba.gov.au/monetary-policy/int-rate-decisions/2026/) · [CommBank: RBA holds after three straight rises](https://www.commbank.com.au/articles/newsroom/2026/06/reserve-bank-june-2026-rates-decision.html) · [Selfwealth: third hike to 4.35%](https://www.selfwealth.com.au/blog/rba-hikes-to-4.35-what-the-third-rate-rise-for-2026-means-for-investors) · [RBA 2026 meeting dates media release (10–11 Aug)](https://www.rba.gov.au/media-releases/2025/mr-25-02.html) · [Kalkine: jobs surge puts hike in focus](https://kalkine.com.au/news/general-news/australian-jobs-surge-puts-another-rba-rate-rise-back-in-focus) · [ASX RBA Rate Tracker](https://www.asx.com.au/markets/trade-our-derivatives-market/futures-market/rba-rate-tracker) · [ABS CPI (Q2 release 29 Jul)](https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/latest-release)
- [BoJ June 2026 guideline change PDF (hike to 1.00%)](https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/k260616a.pdf) · [Bloomberg: BoJ historic hike, Ueda absent](https://www.bloomberg.com/news/articles/2026-06-14/boj-set-to-hike-rates-to-highest-since-1995-despite-ueda-absence) · [Bloomberg: BoJ to raise growth forecast, stand pat (July)](https://www.bloomberg.com/news/articles/2026-07-17/boj-is-said-likely-to-raise-growth-forecast-stand-pat-on-rates) · [Japan Times July meeting preview](https://www.japantimes.co.jp/business/2026/07/18/economy/boj-july-meeting-assessment/) · [Yahoo/BBG: watchers see two 2026 hikes, ~69% by Oct](https://sg.finance.yahoo.com/news/boj-watchers-see-two-rate-143515594.html) · [BoJ MPM schedule](https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm)
- [FRED DFEDTARU (3.75% upper since 11 Dec 2025)](https://fred.stlouisfed.org/series/DFEDTARU) · [FOMC minutes June 16–17 2026](https://www.federalreserve.gov/monetarypolicy/fomcminutes20260617.htm) · [CME Elite: FOMC July 28–29 preview](https://www.cmelitegroup.com/knowledge-hub/fomc-meeting-fed-decision-day/)

---

## FOMC July 2026 — What Is Priced In

**Meeting confirmed: Tue–Wed July 28–29, 2026** — statement 2:00 pm ET Wednesday July 29, Chair press conference 2:30 pm ([federalreserve.gov calendar](https://www.federalreserve.gov/newsevents/calendar.htm), [cmelitegroup.com](https://www.cmelitegroup.com/knowledge-hub/fomc-meeting-fed-decision-day/)). No SEP/dot plot at this meeting ([fedratecalc.com](https://fedratecalc.com/fomc-meeting-schedule/july-2026/)). Current target range **3.50–3.75%** (effective 3.63%), Chair **Kevin Warsh**. The live debate is **hold vs 25bp HIKE** — cut probability is ~0 ([Forbes 7/23](https://www.forbes.com/sites/simonmoore/2026/07/23/markets-see-chance-fed-hikes-next-week-at-july-meeting/)).

### 1. Market pricing (fast-moving — oil-driven repricing this week)

| Date | July hike odds (CME FedWatch) | Driver |
|---|---|---|
| Jul 15 (post-CPI) | 10.7% | Soft June CPI "killed" the July hike ([Motley Fool 7/24](https://www.fool.com/investing/2026/07/24/probability-july-fed-rate-hike-tripled-last-week/), [TechTimes 7/14](https://www.techtimes.com/articles/320507/20260714/stunning-cpi-miss-kills-july-rate-hike-hormuz-puts-september-back-table.htm)) |
| ~Jul 17 | ~15% | Pre-blackout hawk chorus ([Yahoo Finance](https://finance.yahoo.com/economy/policy/articles/fed-rate-hike-voices-swell-173156437.html)) |
| Jul 22 | 34.7% | Oil spike on US–Iran escalation ([Motley Fool](https://www.fool.com/investing/2026/07/24/probability-july-fed-rate-hike-tripled-last-week/)) |
| Jul 23 | ~33–36.5% (hold 63.5%) | Brent through $100 ([Forbes](https://www.forbes.com/sites/simonmoore/2026/07/23/markets-see-chance-fed-hikes-next-week-at-july-meeting/), [growbeansprout FedWatch](https://growbeansprout.com/tools/fedwatch)) |
| Jul 24 (latest) | **~38–46.5%** — sources conflict within the same article; exact end-of-day figure UNVERIFIED ([HNGN 7/24](https://www.hngn.com/articles/272326/20260724/fed-rate-hike-odds-surge-38-ahead-july-meeting-oil-prices-fuel-inflation-fears.htm)) | Brent hit $102 intraday Jul 23, settled $96.78 Jul 24 |

- **Polymarket**: no change 73.0%, +25bp 25.8%, all cuts <1%; $91.2M volume ([polymarket.com](https://polymarket.com/event/fed-decision-in-july-181); page timestamp ambiguous — treat exact % as approximate).
- **Rest of 2026 path**: **September hike ~78–82%** on FedWatch as of Jul 23–24, up from ~53% a week earlier ([CNBC 7/23](https://www.cnbc.com/2026/07/23/fed-interest-rate-odds-oil-jobless-claims.html), [rateprobability.com](https://rateprobability.com/fed)). Modal market scenario per Forbes: **two hikes in 2026** (Sep/Oct favored over July), i.e. year-end ~4.00–4.25%; futures path ~3.8% by Oct, ~4% at year-end (that curve snapshot dated Jul 2, pre-oil-spike — [StreetStats](https://streetstats.finance/rates/fedfunds)). FactSet economist consensus still **no hikes in 2026** — large market/economist disconnect ([PrimeRates](https://primerates.com/primerate/fed-rate-forecast-2026/)).

### 2. Fed speakers into the blackout (blackout began ~Sat Jul 18)

- **June 16–17 minutes (rel. ~Jul 8)**: "nine hawkish dots" — nine officials project ≥1 hike by end-2026; Warsh removed forward guidance, "higher-for-longer" ([federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomcminutes20260617.htm), [TechTimes](https://www.techtimes.com/articles/319827/20260707/fed-minutes-due-wednesday-nine-hawkish-dots-warshs-deliberate-silence.htm), [Yahoo](https://finance.yahoo.com/economy/policy/articles/warsh-hawkish-shock-9-fed-180221394.html))
- **Warsh (Jul 15 testimony)**: "prices are too high," zero tolerance on inflation, but no firm July signal ([Chase](https://www.chase.com/personal/investments/learning-and-insights/article/kevin-warsh-prices-are-too-high-what-to-expect-july-2026-federal-reserve-meeting), [Bloomberg 7/14](https://www.bloomberg.com/news/articles/2026-07-14/fed-rate-hike-bets-mount-before-inflation-data-warsh-testimony))
- **Hammack, Cleveland (Fri Jul 17, last pre-blackout day)**: "Inflation is too high. The labor market is right around my level of maximum employment"; estimates June core PCE **3.3%**; April dissenter ([Yahoo](https://finance.yahoo.com/economy/policy/articles/fed-rate-hike-voices-swell-173156437.html))
- **Logan, Dallas (Jul 16)**: "modestly higher interest rates" needed; April dissenter (same Yahoo source)
- **Jefferson, Vice Chair (Jul 16, Stanford)**: could be "appropriate to reconsider our current policy stance" if inflation doesn't cool soon
- **Waller (Jul 13)** and **Cook (Jul 15)**: core-inflation-trend concern despite the soft print ([Forbes](https://www.forbes.com/sites/simonmoore/2026/07/23/markets-see-chance-fed-hikes-next-week-at-july-meeting/))
- **Williams, NY (dove, ~Jul 15)**: "unquestionably high" inflation will ease; cites moderate wage growth, expected shelter disinflation

### 3. Latest data prints

- **June CPI (rel. Tue Jul 14)**: headline **−0.1% m/m, ~3.5% YoY vs 3.8% expected**; core **0.0% m/m vs +0.2% exp, 2.6% YoY**; gasoline −9.7% m/m did the work ([CNBC](https://www.cnbc.com/2026/07/14/consumer-price-index-inflation-report-june-2026.html), [CBS](https://www.cbsnews.com/news/june-2026-cpi-report/), [Conference Board](https://www.conference-board.org/research/global-economy-briefs/cpi-insights-june-2026)). Note: Forbes cites "3.7% through June" and your recorded 3.23% matches neither — headline YoY level UNVERIFIED to one decimal; the miss-vs-consensus direction is consistent everywhere.
- **June PCE: NOT yet released** — due **Thu Jul 30** (day after FOMC), core expected ~3.2% YoY ([FinancialJuice week-ahead](https://features.financialjuice.com/2026/07/24/week-ahead-economic-indicators-27th-31st-july-us/), [CMC](https://www.cmcmarkets.com/en-gb/news-and-analysis/the-week-ahead-central-bank-rate-decisions-us-pce-meta-earnings)). One source claimed a Jul 25 release — that's a Saturday, discard. Your recorded core PCE 2.89% ≈ May's 2.9% (seven-month high per [TradingEconomics](https://tradingeconomics.com/united-states/core-inflation-rate)); note Hammack's 3.3% June estimate implies she expects re-acceleration.
- **June jobs (rel. Thu Jul 2)**: NFP **+57K vs ~110–115K expected**; unemployment **4.2%** (down, but via participation −0.3pp to 61.5%); Apr/May revised **−74K** combined; leisure/hospitality −61K ([CNBC](https://www.cnbc.com/2026/07/02/jobs-report-june-2026-.html), [Yahoo](https://finance.yahoo.com/economy/article/june-jobs-report-us-payrolls-rose-by-57000-missing-expectations-190000748.html)). Weak labor + hot prices = the Fed is cornered ([Money Morning](https://moneymorning.com/2026/07/17/fed-fomc-july-28-29-2026-rates-cornered-markets)).

### 4. Concurrent events, week of Jul 27–31

- **Oil/geopolitics is the meeting's wildcard**: US striking Iran (nine consecutive nights), attacks on Saudi tankers spread to the Red Sea; Brent intraday **$102** Jul 23, settled **$96.78** Jul 24; WTI ~$83 ([CNBC 7/23](https://www.cnbc.com/2026/07/23/oil-prices-today-wti-brent-trump-iran-hormuz.html), [Bloomberg](https://www.bloomberg.com/news/articles/2026-07-19/latest-oil-market-news-and-analysis-for-july-20), [Yahoo 7/24](https://finance.yahoo.com/markets/live/stock-market-today-friday-july-24-dow-sp-500-nasdaq-081854465.html))
- **BoJ**: meets **Jul 30–31, decision Fri Jul 31** + quarterly Outlook Report; hold expected, gradual tightening bias intact ([EBC](https://www.ebc.com/forex/when-is-the-next-boj-meeting-2025-final-date-and-amp-2026-schedule), [CMC](https://www.cmcmarkets.com/en-gb/news-and-analysis/the-week-ahead-central-bank-rate-decisions-us-pce-meta-earnings))
- **ECB**: already met **Jul 22–23** (prior week) — not concurrent ([centralbank.watch calendar](https://centralbank.watch/tools/calendar/))
- **Treasury refunding**: QRA is **Wed Aug 5** (borrowing estimates ~Aug 3) — clears FOMC week ([treasury.gov](https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding/most-recent-quarterly-refunding-documents))
- **Data**: Q2 GDP advance **Thu Jul 30** (exp ~1.4% SAAR; GDPNow 2.4%, NY Fed Nowcast 1.8%); June PCE Thu Jul 30; Q2 ECI Fri Jul 31 8:30am ([FinancialJuice](https://features.financialjuice.com/2026/07/24/week-ahead-economic-indicators-27th-31st-july-us/))
- **Megacap earnings**: **MSFT, META, AMZN report Wed–Thu (Jul 29–30)** — straight into the Fed decision; Apple timing UNVERIFIED (one source implies following week). Alphabet already reported this week and its capex/negative-FCF spooked the tape; Tesla also reported Jul 23 ([CNBC week-ahead](https://www.cnbc.com/2026/07/24/stock-market-next-week-outlook-for-july-27-31-2026.html), [Investing.com](https://www.investing.com/analysis/earnings-superweek-what-to-expect-from-megacap-tech-titans-200679106), [Benzinga](https://www.benzinga.com/markets/prediction-markets/26/07/60660768/sp500-july-24-open-up-or-down-polymarket-oil-prices-alphabet-tesla-earnings-ai-spending))

### 5. Tape check vs your recorded state (Jul 24 close)

S&P 500 **7,411.98** (+0.05%, after −1.21% Jul 23, worst day since Jun 23); **VIX 18.58** (recorded 18.7 ✓); **10Y 4.71%** — 4th straight rise, highest since Jan 2025 (recorded 4.67% slightly stale); fed funds 3.63% ✓; unemployment 4.2% ✓ ([AP/king5](https://www.king5.com/article/syndication/associatedpress/how-major-us-stock-indexes-fared-friday-7242026/616-316465cf-af08-46e2-9cdb-e4994743d576), [TradingEconomics 10Y](https://tradingeconomics.com/united-states/government-bond-yield)). Recorded CPI 3.23% does not match any found June print (3.5–3.7%) — UNVERIFIED/likely stale.

**Bottom line**: Base case priced in is a **hawkish hold** at 3.50–3.75% (~55–73% depending on venue/hour), with a live ~26–46% tail of a 25bp hike, near-zero cut odds, September hike ~80% priced, and the whole distribution hostage to Hormuz/Red Sea oil headlines between now and Wednesday 2pm.

---

## Unaudited Commit Stack — Findings

Range note: `git log cf81267..b8d175f` contains only 7 commits (294c6b2 → b8d175f). The three named dashboard/training commits (0bfc414, 4a9b9b8, 3ff2d67) are **ancestors of cf81267** and fall outside the literal range; they were audited anyway per the task. All file reads were read-only; nothing was modified.

---

### 0bfc414 — [DASH] Elo bar + self-play training status panel — **SAFE**
- Adds display-only JS to `dashboard/index.html` (Elo gauge from `v015_manifest.json` recorded Sharpe; training panel reads status JSONs). New `scripts/write_training_gate_status.py` imports `gate.py`'s public `evaluate_gate()` (pure read of config + board + ledger) and writes `data/agent/training_gate_status.json`. Explicit no-glob allowlist additions to the dashboard serve/build scripts.
- No path trains, trades, or mutates config. `gate.py` logic untouched (verified by reading current file — the gate is the hardened post-GOVFLAG version).

### 4a9b9b8 — [DASH] AlphaZero confidence wiring — **SAFE**
- Display-only: renders `directional_bias · confidence%` from existing `daily_briefing.json`, greys out on `deterministic_fallback`, always shows the scorecard maturity caveat (n=14 CALIBRATING). New status-writer script calls `scorecard.report()` read-only.
- No execution-path file touched; honest-labeling direction (low-sample reads can't masquerade as proven edge).

### 3ff2d67 — [INFRA] Interactive training controls (start/kill/undo/restore) — **NEEDS-REVIEW**
**The headline question — can "start" bypass the ignition gate? NO.** Traced path: dashboard button → `POST /api/start_watch` → `action_start_watch()` → `Popen(TRAIN_CMD)` where `TRAIN_CMD = [sys.executable, "scripts/sovereign_train.py", "--watch"]` is a hard-coded argv, `shell=False`, no request data reaches it. The gate is enforced **inside the spawned process**, not the server: `sovereign_train.py:63` calls `gate_mod.evaluate_gate(CONFIG)` and threads `gate_open` into every phase. Gate is currently CLOSED four ways (`tick_024_carry_fix_landed: false`, `hyp_071_net_confirmed: false`, board still carries `gross_R_caveat`, no fresh-prereg CONFIRMED HYP-071 revival in the ledger). The server never writes `config/training.yml`, the ledger, or the board — bypass would require filesystem write access the server doesn't expose. Two further backstops even if the gate opened: `policy_updater.refit_policy` raises `NotImplementedError` on the live branch, and `committed` can literally never become `True` (`sovereign_train.py:126-128` prints "Waiting for human confirmation" but no `input()` is wired — fail-safe direction).

Why NEEDS-REVIEW anyway:
1. **CSRF-exposed mutating endpoints.** The four POST routes have no auth token and no Origin/Host validation. The `Access-Control-Allow-Origin: 127.0.0.1:8080` header only blocks response *reads* — any webpage open in Colin's browser can *send* simple cross-origin POSTs to `127.0.0.1:8787` and silently start/kill training runs or flip the snapshot pointer while the server is up. (Server is manually started only — no plist references it — which bounds exposure.)
2. **Unenforced snapshot invariant.** `snapshots.py`'s docstring claims undo/restore "cannot activate an uncommitted/rejected cycle," but `record_cycle()` makes `params_after` the current state and `restore_last_cycle()` re-applies it **regardless of `committed=False`**. Harmless today (gate-closed cycles have `params_before == params_after` by construction, and grep confirms nothing outside the training scaffold reads `data/training/current_policy_params.json`) — but it is a latent hazard for the day live refit is wired.

### cf81267 / 0dec9fa — NEXT.md logs — **SAFE** (docs only; cf81267 is the range boundary)

### 294c6b2 — [INFRA] Install-ready `com.alta.dip_daily` plist — **NEEDS-REVIEW**
- Adds `RunAtLoad` + env vars; commit takes no launchctl action (operator install only, verified not in `launchctl list`).
- The concern is what one `launchctl load` now arms: `dip_daily.sh` runs `continuous_harvester.py` then `training/retrain_loop.py --once` — **XGBoost retraining with no ledger gate anywhere on that path** (`retrain_loop.py` contains zero gate/CONFIRMED logic). The model it writes, `models/xgb_veto.json`, is the live Stage-4c `HarvestVeto` **execution gate** (`sovereign/risk/harvest_veto.py`, loaded by `sovereign/orchestrator.py:354`, blocks trades at :1426) with 60-second auto-reload and a threshold that "rises automatically every 4 hours." One human step, then continuous unattended model training feeding a trade-blocking gate. Orchestrator's live scheduling is indirect (consumers are `execute_daily.py`/`paper_trading_runner.py`, not the loaded `com.alta.forex.scan` → `forex_live_scan.py`), which softens but does not remove the finding.

### 4987d80 — [RISK] TICK-024 measure + stage — **SAFE**
- **Measured:** actual OANDA financing on all 24/24 fills vs `SWAP_RATES_ANNUAL` — ~9x median understatement (range 5.5–11.9x) across all 4 live pairs, plus the EURUSD-SHORT sign flip (model charges, broker credits); cross-checked against `swap_calibration.json`; downstream cost-sensitive figures enumerated in `research/TICK-024_cost_measurement.md`.
- **Staged vs applied:** fix exists only as `research/TICK-024_staged_patch.diff` targeting `sovereign/forex/forex_backtester.py::_apply_costs` (reuses HYP-091's `ratediff_financing`). Verified NOT applied: `git log cf81267..b8d175f -- sovereign/forex/... execution/harness.py ict/pipeline.py` is empty, and the diff header itself demands impact study + param_change_log + Colin sign-off before `git apply`.
- **Gate flag:** `config/training.yml` `ignition.tick_024_carry_fix_landed` is **still `false`** (confirmed in working tree). Only NEXT.md + research files in the commit.

### e3023d3 — [RISK] prop `daily_loss_limit_pct` 0.05 → 0.02 — **SAFE**
- Tightening-only, aligning the stale prop section to the RISK_FRAMEWORK-ratified 2% (gates section was already 0.02). Consumer chain exists (`sovereign/risk/layers/prop.py` ← `risk_engine.py` ← engine_adapter/orchestrator) but is outside the FROZEN list; the direction reduces risk. NN#4 rationale verified present in `data/agent/param_change_log.jsonl` (2026-07-24 entry). TICK-011 closure is bookkeeping with a cited verified fix.

### dd539a9 — [INFRA] DIP Phase 3 + training gate + 3 plists — **NEEDS-REVIEW**
- **Phase 3 diffusion:** append-only writes to `~/Obsidian/.../DIP-Daily-Log.md` and `gate_calibration.jsonl` (currently opens 0 rows — gate scan never ran live). No config mutation, no trading.
- **The three plists** (`dip_warmup` 06:00 / `dip_peak` 08:30 / `dip_diffuse` 16:30 ET, all `RunAtLoad`): NOT loaded by the commit — install is an explicit operator command in each header. If installed, they fire briefing collection, Ollama synthesis + hypothesis-batch generation (research, allowed), and Obsidian diffusion. `dip_peak` deliberately omits `--with-retrain`, so the scheduled DIP path never trains. Nothing trades.
- **Gate-bypass hunt hit #1 — decorative training gate.** `daily_intelligence_pipeline.py:201`: `gate_open = any(status == "CONFIRMED" for e in ledger)`. The ledger holds **14 CONFIRMED entries** (HYP-045, HYP-046 family, etc.), so this gate is permanently open in practice and is tied to no hypothesis about what `xgb_veto` actually trains on. It satisfies RISK_CONSTITUTION Art. 6's letter, not its intent (contrast with `sovereign/training/gate.py`, which pins specific hypotheses, prereg hashes, and dates). Currently moot only because the scheduled path never passes `--with-retrain`.
- **Hit #2 — undisclosed bundle.** The commit silently includes `execution/daily_pnl_store.py` (new file under `execution/`), `research/TICK-044_design_note.md`, `research/TICK-044_staged_patch.diff`, `tests/test_daily_pnl_store.py` — none mentioned in the commit message. Content is dormant (module header: "Nothing imports this module yet"; grep confirms; the staged patch targets frozen `execution/harness.py` but is NOT applied). The TICK-044 note also documents a real standing defect: `DAILY_LOSS_HALT` is permanently inert because `daily_pnl_frac` is never populated. Disclosure gap, not an active hazard.

### 415ddf1 — [AUTO] Evening data sync — **SAFE**
- Data-only churn (`data/**` JSON/parquet, logs). Includes `training_run_status.json` — evidence the control server ran a training cycle, which the gate trace above confirms was DRY. No code.

### b8d175f — [ICT] Shelve/re-scope ICT daily pipeline dispatch — **SAFE**
- Docs/tickets/NEXT.md only (0 code files). Shelves the 8-layer `ict_daily_pipeline.py` spec and re-scopes ICT as a TP/SL reference layer — consistent with the standing evidence (ICT edge NOT PROVEN, p=0.52). Marks the DIP Phase 2e skip branch as permanent intended state. Nothing to bypass.

---

### Summary verdict table

| Commit | What | Verdict | One-line reason |
|---|---|---|---|
| 0bfc414 | Elo bar + gate status panel | SAFE | Display-only; gate evaluated read-only, logic untouched |
| 4a9b9b8 | AlphaZero confidence wiring | SAFE | Display-only with honest low-sample caveats |
| 3ff2d67 | start/kill/undo/restore controls | NEEDS-REVIEW | No gate bypass (gate enforced inside spawned runner), but mutating endpoints are CSRF-open and the snapshot "no uncommitted activation" invariant is docstring-only |
| 294c6b2 | dip_daily plist install-ready | NEEDS-REVIEW | One `launchctl load` arms daily **ungated** XGBoost retrain feeding the live HarvestVeto trade-blocking gate (auto-reload, self-raising threshold) |
| 4987d80 | TICK-024 measure + stage | SAFE | Research + unapplied staged diff; ignition flag verified still `false`; no execution-path change in range |
| e3023d3 | prop daily_loss 0.05→0.02 | SAFE | Tightening-only, ratified value, NN#4 rationale logged and verified |
| 0dec9fa | NEXT.md log | SAFE | Docs only |
| dd539a9 | DIP Phase 3 + gate + 3 plists | NEEDS-REVIEW | Training gate is decorative (`any CONFIRMED` — trivially open with 14 CONFIRMED entries) + undisclosed `execution/` file and TICK-044 staged patch bundled without mention (content dormant) |
| 415ddf1 | Evening data sync | SAFE | Data-only |
| b8d175f | ICT dispatch shelved | SAFE | Docs only, aligns with p=0.52 unproven evidence |

**No RISK verdicts:** no commit in the stack can train a production policy, place a trade, or mutate live parameters without at least one explicit human step today. The two structural weaknesses worth fixing: (1) the `any(CONFIRMED)` gate in `daily_intelligence_pipeline.py:201` should pin a specific hypothesis ID + fresh-prereg requirement like `sovereign/training/gate.py` does; (2) `training_control_server.py` should validate Origin/Host or require a token on mutating POSTs, and `snapshots.record_cycle` should refuse to advance `current_policy_params.json` when `committed=False`.

Key files: `/Users/taboost/quant/scripts/training_control_server.py`, `/Users/taboost/quant/sovereign/training/gate.py`, `/Users/taboost/quant/scripts/sovereign_train.py`, `/Users/taboost/quant/sovereign/training/snapshots.py`, `/Users/taboost/quant/scripts/daily_intelligence_pipeline.py` (line 201), `/Users/taboost/quant/scripts/dip_daily.sh`, `/Users/taboost/quant/sovereign/risk/harvest_veto.py`, `/Users/taboost/quant/config/training.yml`, `/Users/taboost/quant/research/TICK-024_staged_patch.diff`, `/Users/taboost/quant/execution/daily_pnl_store.py`.