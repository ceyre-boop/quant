# FOMC Readiness Intel — 2026-07-25

FOMC statement: **Wed Jul 29, 2:00 pm ET** (press conference 2:30). No SEP at this meeting. Chair: Kevin Warsh. Operator decision point: Monday Jul 27. The machine advises; Colin decides.

---

## 1. TL;DR

- **Go/no-go changer:** the system's recorded rate differentials are wrong on **all four foreign legs** — GBPUSD carries the **wrong sign** (recorded −0.60 vs actual +0.12), USDJPY carry is overstated by **+86bp**, AUD understated by 59bp, EUR drag overstated by 70bp. Bad differentials = bad carry signals. Consistent with the already-ticketed TICK-024 mis-modeling; the fix is **staged, not applied** (`ignition.tick_024_carry_fix_landed: false`).
- **Go/no-go changer:** the Art. 3 drawdown breaker is **unobservable** (portfolio block UNAVAILABLE — no position ledger on disk) and clamp enforcement is deferred to shadow_close **~Jul 28, one day before FOMC**. The circuit-breaker ladder (3.5% halve / 5% halt / 6.5% flatten) currently binds by law, not by live code.
- **Market:** base case is a **hawkish hold** at 3.50–3.75%; July hike tail has quadrupled in ten days to **~38–46.5%** (exact end-of-day figure UNVERIFIED) on the Iran/oil spike; September hike ~78–82% priced; the whole distribution is hostage to Hormuz/Red Sea headlines through Wednesday 2 pm.
- **Repo audit:** 7 SAFE / 3 NEEDS-REVIEW / **0 RISK**. Nothing in the stack can trade or train without a human step — but **do not load `com.alta.dip_daily`** (one `launchctl load` arms daily ungated XGBoost retraining feeding the live HarvestVeto trade-blocking gate), and the training-control server's mutating POST endpoints are CSRF-open.
- **FOMC-window logger is code-complete** (51 tests passing, pure observation) but **100% blocked on operator work**: Windows VM + MT5 + The5%ers demo login (runbook steps 1–5) must be done before 2:00 pm Wednesday or nothing gets logged.

---

## 2. What the Market Prices for FOMC

**Base case: hawkish hold at 3.50–3.75%** (effective 3.63%), cut probability ~0. The live debate is hold vs **25bp hike**.

**July hike odds (CME FedWatch), oil-driven repricing:**

| Date | Hike odds | Driver |
|---|---|---|
| Jul 15 (post-CPI) | 10.7% | Soft June CPI "killed" the July hike |
| ~Jul 17 | ~15% | Pre-blackout hawk chorus |
| Jul 22 | 34.7% | Oil spike on US–Iran escalation |
| Jul 23 | ~33–36.5% (hold 63.5%) | Brent through $100 |
| Jul 24 (latest) | **~38–46.5%** — sources conflict; exact figure UNVERIFIED | Brent $102 intraday Jul 23, settled $96.78 Jul 24 |

- **Polymarket:** no change 73.0%, +25bp 25.8%, cuts <1% ($91.2M volume; page timestamp ambiguous — treat as approximate).
- **Path:** September hike ~78–82% priced (up from ~53% a week earlier). Modal market scenario: **two hikes in 2026** (Sep/Oct favored), year-end ~4.00–4.25%. FactSet economist consensus still **no hikes in 2026** — a large market/economist disconnect.
- **Fed posture into blackout:** June minutes showed **nine officials projecting ≥1 hike by end-2026**; Warsh removed forward guidance ("higher-for-longer"). Hammack ("inflation is too high," estimates June core PCE 3.3%), Logan ("modestly higher rates"), Jefferson (open to reconsidering stance), Waller/Cook (core-trend concern) vs Williams (dove).
- **Data cross-currents:** June CPI soft (headline −0.1% m/m, core 0.0% m/m vs +0.2% exp; YoY headline reported 3.5–3.7% across sources — level UNVERIFIED to one decimal) but June jobs weak (+57K vs ~110–115K exp, Apr/May −74K revisions). The Fed is cornered: weak labor, hot prices, oil rising.
- **Event congestion around the decision:** BoJ decides **Fri Jul 31** (hold expected); RBA's swing input **Q2 CPI lands Jul 29** (same day); June PCE and Q2 GDP advance **Thu Jul 30**; Q2 ECI Fri Jul 31; MSFT/META/AMZN earnings Wed–Thu straight into the decision. Treasury QRA clears FOMC week (Aug 5).
- **Tape check vs recorded state (Jul 24 close):** VIX 18.58 (recorded 18.7, OK); 10Y 4.71% — highest since Jan 2025 (recorded 4.679 slightly stale); fed funds 3.63% correct; unemployment 4.2% correct. **Recorded CPI 3.23% matches no found June print (3.5–3.7%) — UNVERIFIED/likely stale.**

---

## 3. Rate Landscape vs System-Recorded Differentials

**THE HEADLINE FINDING: every foreign leg in the recorded differentials is wrong.** The Fed leg (3.63%) is correct; nothing else is. G10 context: 2026 has been a hiking cycle the system's feed never saw — ECB hiked in June (first in 3 years), RBA hiked three times (Feb/Mar/May), BoJ hiked to 1.00% (highest since 1995), BoE hike bets building. The Fed is the outlier on hold.

| Pair | System diff | Actual (Jul 24) | Error | Flag |
|---|---|---|---|---|
| GBPUSD (BoE−Fed) | −0.60 | 3.75 − 3.63 = **+0.12** | −0.72pp | **WRONG SIGN.** Implies BoE at 3.03% — never true. Actual carry mildly GBP-positive; market prices BoE hikes into 2027 |
| EURUSD (ECB−Fed) | −2.08 | 2.25 − 3.63 = **−1.38** | −0.70pp | Overstates EUR drag by 70bp; implies ECB at 1.55% (never below 2.00%). Narrows to −1.13 if the ~93%-priced Sep ECB hike lands |
| AUDUSD (RBA−Fed) | +0.13 | 4.35 − 3.63 = **+0.72** | −0.59pp | Understates AUD carry by 59bp; misses all three 2026 RBA hikes |
| USDJPY (Fed−BoJ) | +3.489 | 3.63 − 1.00 = **+2.63** | +0.86pp | Overstates USDJPY carry by ~86bp; implies BoJ at 0.14% vs actual 1.00% with an Oct hike ~69% priced |

**Diagnosis:** the four recorded values are mutually inconsistent on any single date — a mixed-vintage stale snapshot (the EURUSD leg matches roughly **Apr–Jun 2025**), not a live feed. **Direction of risk: the system over-rewards short-JPY carry (+86bp phantom), penalizes GBP longs that actually carry positive, and under-sizes AUD.** This is the carry engine's core input and it is measuring a world 12–15 months gone. Consistent with the TICK-024 swap mis-modeling (~9x median financing understatement measured on 24/24 fills, plus EURUSD-SHORT sign flip); the staged patch remains unapplied pending impact study + sign-off. **The differential feed must be re-based before any scan output is trusted.**

**Meeting-week sensitivity:** a 25bp Fed hike Wednesday moves every differential 25bp against foreign legs; RBA Q2 CPI (Jul 29) and BoJ (Jul 31) can move the AUD and JPY legs the same week.

UNVERIFIED items in this section: exact BoE July hold probability (78.5% vs 86% across sources), exact RBA August hike odds (25–36%, moves daily).

---

## 4. System Readiness

| Component | Status | Detail |
|---|---|---|
| Rate-differential feed | **RED** | All 4 foreign legs wrong, one wrong sign (Section 3). TICK-024 fix staged, not applied |
| Portfolio drawdown breaker | **RED** | UNAVAILABLE — no unified position ledger; Art. 3 ladder (3.5%/5%/6.5%) unobservable at portfolio level; clamp reconciliation blocked on shadow_close ~Jul 28 |
| Carry verdict / regime state | **YELLOW** | Overall DEGRADED. STAND_ASIDE (size 0.0) is a **staleness fallback**, not a market read — `forex_proximity.json` 13.9h old vs 12h limit. Fix the feed before reading the verdict |
| FOMC-window logger (TICK-056) | **YELLOW** | Code done: 51 tests passing, demo-only, pure observation, ±15 min window at 2:00 pm ET Jul 29. Blocked entirely on operator VM runbook (UTM + Win11 ARM + MT5 + The5%ers demo + selftest + arm). Live path is dead code — neither unlock artifact exists |
| Crisis library (Alexandrian) | **GREEN with caveats** | 63 episodes; current sim 0.187 (below floor), threat NORMAL, size 1.00x; health PASS. Caveats: live only on the **ICT path**, NOT wired into `forex_live_scan.py` (carry); no literal ">90% similarity" cut exists — real mechanism is CRITICAL ≥0.82 composite → 0.00x |
| Risk constitution / kill switch | **GREEN (law) / YELLOW (enforcement)** | RATIFIED v1.0.0: 0.75% per trade, 2.5% carry complex, 3.5/5/6.5% breakers. Kill switch NOT engaged, path thawed, switch available. But see RED rows — the law is currently not machine-enforced at portfolio level |
| Paper accounts | **GREEN** | Both fresh (4 min). Carry paper $109,627.66 (−$30.45); Undertow shadow $200,099.60 (+$99.60, 10 days, 3 signals). Labeling caveat: $200K/$10K prop framing is cosmetic — own-capital shadow at constitutional sizing |
| Training ignition | **GREEN (fail-closed)** | Gate CLOSED four ways; runner SCAFFOLD/DRY only; placebo fail-closed; auto-approve off. This is the intended state |
| Data freshness overall | **GREEN** | Nothing inventoried exceeds 48h; oldest live reading is scanner_state.json ~7.7h |

---

## 5. Repo Risk Findings (Commit Audit)

**Verdict: 7 SAFE, 3 NEEDS-REVIEW, 0 RISK.** No commit in the stack can train a production policy, place a trade, or mutate live parameters without at least one explicit human step. The headline question — can the dashboard "start" button bypass the ignition gate — is **NO** (gate enforced inside the spawned runner, hard-coded argv, no request data reaches it, plus two backstops: `refit_policy` raises NotImplementedError and `committed` can never become True).

The three NEEDS-REVIEW items, in priority order:

1. **`294c6b2` — dip_daily plist is one `launchctl load` from arming ungated retraining of a live gate.** `retrain_loop.py` has zero ledger-gate logic and writes `models/xgb_veto.json` — the live Stage-4c HarvestVeto **execution gate** with 60-second auto-reload and a self-raising threshold. Verified not loaded today. **Recommendation: do not install before FOMC; ticket a real gate first.**
2. **`3ff2d67` — CSRF-open mutating endpoints.** The four POST routes on the training-control server (127.0.0.1:8787) have no auth token and no Origin/Host validation — any webpage in Colin's browser can silently start/kill runs or flip the snapshot pointer while the server is up (manually started only, which bounds exposure). Secondary: the snapshot "cannot activate uncommitted cycle" invariant is docstring-only — `restore_last_cycle()` re-applies regardless of `committed=False`. Harmless today, latent hazard for live refit.
3. **`dd539a9` — decorative training gate + undisclosed bundle.** `daily_intelligence_pipeline.py:201` gates on `any(CONFIRMED)` — trivially open with 14 CONFIRMED ledger entries, tied to no hypothesis about what xgb_veto trains on; satisfies Art. 6's letter, not intent (moot only because the scheduled path never passes `--with-retrain`). The commit also silently bundles a new `execution/daily_pnl_store.py` plus a staged patch targeting frozen `execution/harness.py` (verified NOT applied, nothing imports it). The TICK-044 note documents a real standing defect: **DAILY_LOSS_HALT is permanently inert** because `daily_pnl_frac` is never populated.

Also verified safe and relevant: `4987d80` measured the ~9x financing understatement and staged the TICK-024 fix without applying it; `e3023d3` tightened prop `daily_loss_limit_pct` 0.05→0.02 with the NN#4 rationale logged; `b8d175f` shelved the ICT daily pipeline consistent with ICT's unproven edge (p=0.52).

Two structural fixes worth doing regardless of the FOMC decision: pin the DIP gate to a specific hypothesis ID + fresh prereg (like `sovereign/training/gate.py` does), and add token/Origin validation to the control server + make `record_cycle` refuse to advance params when `committed=False`.

---

## 6. The Decision Colin Faces Monday

The July 29 FOMC was slated as the first live test. Here is the honest state: the carry engine's core input (rate differentials) is measuring mid-2025, the constitutional circuit breakers are unobservable at portfolio level until at least Jul 28, the carry verdict on disk is a staleness artifact, and the event itself carries a ~38–46% hike tail whipsawed by oil headlines. Three real options:

**Option A — No-go; make FOMC an observation event (lowest risk, still productive).**
Execute the VM runbook (steps 1–5) and arm `fomc_window_logger.py` before 2:00 pm Wednesday; fix the `forex_proximity.json` feed; re-base the differential feed against actual policy rates; leave the TICK-024 patch staged pending its required impact study. You get real FOMC-window microstructure data on all 5 pairs, a corrected input stack, and zero exposure. Cost: the live-test milestone slips one cycle. Note the proven carry edge only pays in rate-trending regimes — a corrected feed showing GBP carry-positive and JPY carry 86bp thinner may itself change what the system would want to hold.

**Option B — Conditional go, only if the stack is fixed by Tuesday.**
Minimum bar to trade this event within the constitution: (1) differentials re-based and verified against Section 3's actuals; (2) proximity feed fresh so STAND_ASIDE/GO is a real verdict; (3) a readable position ledger so the Art. 3 ladder is enforceable, or an explicit manual-breaker protocol Colin executes himself. That is a heavy two-day lift landing the day the shadow freeze closes — rushing execution-path changes into an event window is exactly what the freeze exists to prevent.

**Option C — Go as-is.**
Listed for completeness: it means knowingly sizing off a wrong-signed GBP differential and a +86bp phantom JPY carry, with no machine-enforced drawdown breaker, into a binary event. Nothing in the evidence supports this.

**Independent housekeeping decisions, whichever option is chosen:**
- Do **not** `launchctl load` `com.alta.dip_daily` (finding #1) before the retrain path gets a real gate.
- Decide whether the shadow_close/clamp reconciliation lands Jul 28 as scheduled (one day pre-FOMC) or is deferred until after the event — changing enforcement plumbing 24h before the test it's meant to protect cuts both ways.
- CSRF token on the training-control server and the DIP gate pin can wait past FOMC but should be ticketed now.

*All figures sourced from the four intel sections dated 2026-07-24/25. UNVERIFIED items are marked inline: exact Jul 24 FedWatch close (~38–46.5%), Polymarket timestamp, June CPI YoY to one decimal, recorded CPI 3.23% provenance, BoE July hold % (78.5–86%), RBA August hike odds (25–36%), Apple earnings timing.*
