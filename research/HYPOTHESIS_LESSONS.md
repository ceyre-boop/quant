# What every hypothesis taught — HYP-001 → HYP-116

Compiled 2026-09-03 from `data/agent/hypothesis_ledger.json`, the older ledger at
`ict-dashboard/data/agent/hypothesis_ledger.json` (HYP-001–043), `data/research/preregister/*.json`,
the `data/research/hyp1xx/VERDICT.md` docs, `NEXT.md` and `tickets/backlog.md`. Verdicts are quoted
from the ledger. Companion to `research/EDGE_LEDGER.md` (status) — this file is the *lessons*.

## The ten things 116 hypotheses collapse into

1. **Carry off the rate level is the only forex edge that ever survived everything** (HYP-001/003/045/046, FDR in HYP-063). Every attempt to *improve* it — gates, regimes, velocity, acceleration, adaptive params, exits — was null or negative (HYP-027…067, 090, H1, H2, V007).
2. **Direction from public information is null at every horizon and resolution tried** — macro, positioning, momentum, news, breakouts, post-shock (HYP-006–013, 085, 089–092, 104, 109, 111, 114, ESNQ, SENTIMENT-*).
3. **Magnitude is predictable and already priced** (HYP-109a real; HYP-112 the option market over-prices it; UVXY roll eats it).
4. **Small samples "confirm" and scale falsifies** (HYP-002→010, 020, 051, 055). Anything under ~100 independent events is a story.
5. **Threshold sweeps and scans manufacture edges** (HYP-044 p=0.50 OOS, HYP-098, HYP-104 best-of-77k → 0.36, HYP-090 loses to a placebo). Scan size *is* the multiplicity.
6. **Look-ahead hides in universe definitions, not just features** (HYP-105/106 refuted; HYP-057 tautological router; HYP-064 caught a live tautology before training).
7. **The most obvious setups are the worst ones** (HYP-023 A+ paradox, HYP-047 score inversion, HYP-113 p99+ down-days carry no fade).
8. **Exits and sessions matter more than entries** (HYP-022 NY_PM anti-edge, HYP-050 Tue/Thu veto, HYP-059/060 time exits beat trailing stops) — but the exit surface is already at its optimum on v015 (180-config sweep, HYP-066/067/108 blocked).
9. **A benchmark is a hypothesis too** (HYP-115: the EW-10 basket every 2026-09 overlay was measured against is itself FRAGILE — OOS Sharpe 0.22, −54% GFC).
10. **The process catches its own errors** — the sign bug in HYP-111/111a/113 was found by HYP-114; the HYP-087/088 and HYP-096 "phantom" claims were caught by checking the filesystem. Verify every claim against disk; a chat message is not a lock.

## Table 1 — HYP-001 … HYP-116

| id | tested | verdict | lesson |
|---|---|---|---|
| HYP-001 | Carry base positive EV | CONFIRMED | Rate-differential carry has real positive expectancy (59% WR, +0.311R) — the seed of the whole forex book. |
| HYP-002 | Quarter-end rebalancing flow | CONFIRMED (n=15) | A 60% WR on 15 trades is not an edge; killed at scale by HYP-010. Small-sample confirmations are the desk's most expensive habit. |
| HYP-003 | v004 macro system is the best forex edge | CONFIRMED | Sharpe 0.801 with 8/8 pairs positive — breadth across pairs, not one pair's luck, is what makes a macro signal credible. |
| HYP-004 | ICT FVG limit entry positive EV | CONFIRMED | +0.40R/trade replicated in two independent windows; replication-in-windows became the house standard. |
| HYP-005 | ICT walk-forward B validates edge | CONFIRMED | Walk-forward WR 76.0% vs Monte-Carlo 76.8% — matching the MC null within 1pp means the "edge" is mostly the setup's base rate. |
| HYP-006 | CPI surprise fade as a standalone signal | REJECTED | 43% WR / −0.320R (n=63): a macro surprise is not a tradeable direction — already repriced. |
| HYP-007 | Post-CB drift entered before price confirmation | REJECTED | 36% WR / −0.517R (n=14): anticipating CB drift without price confirmation is guessing. |
| HYP-008 | Rate divergence at 20-day hold | REJECTED | 44% WR, −0.186R over n=312: a well-powered null — divergence needs a conditioning regime, not a fixed hold. |
| HYP-009 | March JPY seasonality in isolation | REJECTED | 26% WR — policy dominates seasonality. |
| HYP-010 | Quarter-end fade at scale | REJECTED | 39% WR on n=366 — HYP-002 was noise. Scaling n is the cheapest falsification. |
| HYP-011 | State allocator outperforms v004 | REJECTED | Sharpe −0.12 vs 0.801: regime-state machinery destroyed the simple signal it routed. |
| HYP-012 | Confirmation protocol outperforms v004 | REJECTED | Sharpe −0.17: waiting for confirmation costs more than the false starts it avoids. |
| HYP-013 | Rate divergence at 60-day hold | REJECTED | 60d Sharpe 0.300 vs 20d 0.658 — a longer hold buys magnitude at a worse Sharpe. |
| HYP-014 | Calendar edges as v004 size boosts | REJECTED | +0.0014 Sharpe; 30/863 trades touched a window — an overlay firing 3.5% of the time cannot move a portfolio. |
| HYP-015 | ICT walk-forward stability vs live paper | QUEUED | Never executed. |
| HYP-016 | FunderPro challenge attempt | QUEUED | Never executed; resurfaced as HYP-103. |
| HYP-017 | Stop width is invariant to Sharpe | REJECTED (closed) | Across 4,200 combos hold period dominates, stop width barely matters — tune the horizon, not the stop. |
| HYP-018 | Library bull-regime CRITICAL false positive fixed | CONFIRMED | A −1-severity analogue drove sizing to 0.00×; a 0.50× floor restored trading — analogue matchers need regime sanity floors. |
| HYP-019 | Pure carry direction (rate sign) standalone | REJECTED | 2/8 pairs positive; EURUSD −5.6 — carry is an income overlay, not a direction. |
| HYP-020 | March JPY seasonal persists pre-2020 | REJECTED | n=6, avg R≈0; a seasonal with six occurrences is a story. |
| HYP-021 | Library CRITICAL match driven by equity dislocation | REJECTED | The matched pattern's features were bullish equity; the true CRITICAL fingerprint is consecutive down weeks + sector dispersion + realized vol. |
| HYP-022 | NY_PM session is an anti-edge | CONFIRMED | −0.283R vs London +0.471R → NY_PM vetoed; session is a first-order ICT filter. |
| HYP-023 | A+ grade paradox | CONFIRMED | A+ WR 13% (−0.375R) vs A 39% (+0.383R) — the most obvious setups are the ones already positioned for. |
| HYP-024 | pd_alignment component is an anti-edge | CONFIRMED | Component actively negative (−15pp WR) — validate scoring components individually. |
| HYP-025 | Unexplained losses resolve on a momentum threshold | REJECTED | Distributions identical; the driver is an unmeasured factor. |
| HYP-026 | London-only ICT + Grade A is the TP2 setup | CONFIRMED | n=39, WR 41%, +0.840R — one session × one grade turned ICT from noise into a target profile. |
| HYP-027 | USDJPY regime gate (bull + VIX>15) | RETEST_BLOCKED | No costs/OOS originally; REGIME-ROUTER-SCREEN later found the gate inert (n=0) on honest data — a lift that doesn't reproduce was a backtester artifact. |
| HYP-028 | US10Y–EURUSD divergence as size modifier | RETEST_BLOCKED | IC 0.086 (below gate) but conditional cell 70% (n=125) — too weak to stand alone, legitimate as a multiplier. |
| HYP-029 | XGBoost vs neural net for R prediction | REJECTED | NN wins by 0.006 correlation, loses on Sharpe — model-class upgrades buy nothing when features bind. |
| HYP-030 | Three-confirmation gate for commodity pairs | RETEST_BLOCKED | USDCAD term-structure gate genuine; AUDUSD copper filter redundant with VIX and hurts — stacked gates proxying the same regime subtract. |
| HYP-031 | USDCAD re-entry with term-structure gate | RETEST_BLOCKED | Hurts average-pair Sharpe, raises portfolio Sharpe via decorrelation — never judge a pair standalone. |
| HYP-032 | Drift-to-noise (SNR) entry gate | REJECTED | Top-quartile SNR slightly underperforms — daily FX drift estimates are too noisy to gate on. |
| HYP-033 | Monte-Carlo path-convergence size modulator | REJECTED | ρ≈−0.02, structurally capped — a simulated statistic with no dispersion carries no information. |
| HYP-034 | (ledger slot; no distinct test recorded) | — | — |
| HYP-035 | RMT covariance cleaning, per-trade penalty | RETEST_BLOCKED | No improvement, but 30% of apparent correlation is noise — right theory, wrong application layer. |
| HYP-036 | RMT at the portfolio-Kelly level | REJECTED | Holdout Δ −0.103 — with 4–5 assets there isn't enough matrix to clean. |
| HYP-037 | fvg_tap anti-edge is timing-conditional | RETEST_BLOCKED | Flips positive inside UTC-03xx — an anti-edge already neutralised by an existing gate. |
| HYP-038 | displacement=0 as a standalone veto | REJECTED_GLOBAL / PARTIAL_LONDON | Works in one session only — a session interaction, not a universal veto. |
| HYP-039 | UTC 15xx as a second London-close window | REJECTED | +0.090R — you cannot manufacture frequency by relaxing the time gate. |
| HYP-040 | Monday gap fill rate | QUEUED | Never run. |
| HYP-041 | Time-of-month flow bias, days 1–3 | QUEUED | Never run. |
| HYP-042 | Carry-unwind velocity (COT z) as VIX-spike detector | QUEUED | Never run; the COT thesis was nulled as SENTIMENT-COT. |
| HYP-043 | Macro signal freshness gate | REJECTED — architecture void | All 648 trades were 1–2d fresh; the variance you want to exploit must exist in your architecture first. |
| HYP-044 | VIX gate threshold sweep 15→13 | CONFIRMED (v014) → REJECTED_OOS p=0.50 | The canonical sweep trap: "monotonic improvement across thresholds" was pure in-sample selection. |
| HYP-045 | AUDNZD exclusion → 4-pair book | CONFIRMED / LIVE | OOS Sharpe 1.08–1.25, p=0.003 — removing the worst pair was worth more than any gate. This is v015. |
| HYP-046 | Displacement gate ≥1.5 (London) | CONFIRMED | Keeps 81% of trades, lifts +0.146R — the right (non-destructive) gate shape. |
| HYP-046a/b/c | EUR/GBP hold 5/7/10d, ex-AUDNZD | CONFIRMED | Identical OOS at every horizon — insensitivity to the hold parameter says the edge is drift, not a tuned exit. |
| HYP-047 | Score inversion inside high-displacement trades | CONFIRMED → DEPLOYED (perm p=0.004) | Monotonic decay from score 7–8 (+1.21R) to 10–11 (−1.0R) — ultra-high scores are over-obvious. |
| HYP-048 | Score-cap sweep + frequency/quality tradeoff | Study; escalated | Quality gates have a *time* cost (17–27 months vs 7 months to target) that must be priced. |
| HYP-049 | Short natural-duration trades underperform | CONFIRMED (p<0.0001) | Fast closures are entries before institutional commitment — forensic signal, not a tradeable rule (the backtester contradicts the rule-as-written). |
| HYP-050 | Tue + Thu veto (London ICT) | CONFIRMED (perm p=5e-05) | Replicated in two windows; a cheap veto beat an expensive one. |
| HYP-051 | Day-of-month bias | CONFIRMED (candidate) | Days 8–15 +1.54R vs 1–7 −0.18R at n=20 — monitor-only. |
| HYP-052 | Rate-differential trend gate (cross-pair) | CONFIRMED in ledger, rejected in note (1/4 OOS years) | A cross-pair aggregate washes out pair-level information. |
| HYP-052b | Rate-differential volatility gate | REJECTED | Opposite of prediction at every threshold — rate vol tracks regime, not trade outcomes. |
| HYP-052c | Pair-level rate-trend gate | MARGINAL, FAILS_PERMUTATION (p=0.135) | OOS>IS and calls years correctly by hand — still a narrative if it fails permutation. |
| HYP-053 | 90d rate-spread velocity size gate | REJECTED | Negative even with full look-ahead — if look-ahead can't make it work, the signal isn't there. |
| HYP-054 | Rate-level gate |real diff|>1% | MARGINAL, FAILS_PERMUTATION (p=0.171) | Good OOS behaviour does not substitute for a significance test. |
| HYP-055 | USDJPY+USDCAD macro-contrarian filter | PARTIAL | One pair carrying a two-pair claim; 2024 hurt. |
| HYP-056 | Counter-momentum entry vs 63d trend | CONFIRMED | Entering against recent trend times the carry entry better. |
| HYP-057 | Sovereign Core mixture-of-experts (master-ml-archive) | NO_EDGE (p=0.164) | Router OOS accuracy 0.998 was a tautology — near-perfect accuracy is a leak alarm. |
| HYP-058 | Rate-spread velocity as portfolio throttle | NOT_CONFIRMED | Premise falsified by its own anchor; signed velocity cut maxDD ~60% at flat Sharpe — a defense overlay, not a Sharpe enhancer. |
| HYP-059 | Regime fragility lives in the trailing-stop exit | CONFIRMED (later refuted as a live lever) | Trailing −49R vs time exits +142R — the diagnosis was real; the 180-config sweep showed v015 already optimal, so the lever wasn't. |
| HYP-060 | Velocity overlay and trailing drag orthogonal? | CONFIRMED | A sizing overlay can never substitute for fixing an exit rule. |
| HYP-061 | CB-blackout veto 3–14d pre-decision | REJECTED in ledger despite Δ+0.13, p=0.005 | Pre-decision chop degrades displacement setups; the first run used a contaminated cb_decisions.json — data hygiene lesson. |
| HYP-062 | Is the 4-pair edge cost-robust? | CONFIRMED | Breakeven at ~12.5× modelled spread — not cost-fragile; cost fragility tracks regime fragility. |
| HYP-063 | Do deployed edges survive FDR? | CONFIRMED | 12 of 26 survive BH at 5% incl. the live book; only HYP-059 under Bonferroni. Multiple-testing correction separates the book from the folklore. |
| HYP-064 | XGBoost directional-bias agreement gate on carry | NOT_SIGNIFICANT (Phase 1) | The prereg caught a live tautology in `layer1/bias_engine.py` before any model was trained. |
| HYP-065 | Carry edge conditional on Fed-cycle position | REJECTED | Edge-ON in 2/3 cycles, p=0.09 — not switchable by policy phase. |
| HYP-066 | Regime-conditioned exit params (VIX×ATR) | NOT_SIGNIFICANT | Regime-keying degraded OOS 0.90→0.77; it found structure that didn't exist. |
| HYP-067 | GA-evolved regime/age exit policy | NOT_ROBUST | OOS −0.40 — more search over the same exit surface generalises worse. |
| HYP-068 | — | no record found | |
| HYP-069 | — | no record found | |
| HYP-070 | — | no record found | |
| HYP-071 | Tabular exit value function (NNUE-style) | METRIC_ARTIFACT | EXIT_NOW dominance was tautological (locked value has zero forecast variance). A recompute later "passed" without a prereg → HYP-071-GOVFLAG. |
| HYP-072–081 | Positioning-board family (COT extremes, risk-reversals, butterflies, GDELT tone, event-conditioned crowding) | PREREGISTERED, unresolved | Hash-locked before the board-state data existed; deliberately dark pending the family BH protocol. Ten sealed, none run. |
| HYP-082 | Log-corridor deviation beyond carry | NOT_SIGNIFICANT | IC 0.011, p=0.60, N=2,172 — well-powered; fractal corridors add nothing to carry daily. |
| HYP-083 | Daily FVG event book as carry diversifier | NOT_SIGNIFICANT | p=0.74; all pre-lock geometry exploration declared non-evidentiary. |
| HYP-084 | Triangle tags improve precedent retrieval | PREREGISTERED, dark | Left untouched on purpose. |
| HYP-085 | Political alpha: Trump statements → abnormal moves | NOT_SIGNIFICANT (p=0.36) → GRAVEYARD | 11.2% vs 10.3% placebo; averaging a heterogeneous event population is the wrong tool. |
| HYP-086 | Political-Alpha V2 Track C | name only, never built | |
| HYP-087 | V2 Track A: statement clusters × instruments | no verdict sealed | Built by a parallel session with a commit as the only lock; "ledger updated" claim had zero trace — governance lesson. |
| HYP-088 | V2 Track B: congressional BUY clusters → policy | no verdict sealed | Same gap. |
| HYP-089 | 12-month TSMOM quick-look | NOT_SIGNIFICANT → GRAVEYARD | Sharpe 0.28 < 0.30 gate; done properly as HYP-091. |
| HYP-090 | Daily adaptive parameter selection vs static v015 | NOT_SIGNIFICANT (p=0.977) | 5,775 variants lose to the static incumbent AND to a random placebo — no parameter-level regime structure at daily resolution. |
| HYP-091 | TSMOM as carry diversifier, correct financing | NOT_SIGNIFICANT | OOS −0.35; uncorrelated is necessary, not sufficient. Reusable: the financing model. |
| HYP-092 | Gapper "decision card" mechanised | NOT_SIGNIFICANT (p=0.594) | Well-powered null on the checklist; the collateral map (all gappers fade −2.2% after 10:30) became HYP-093. |
| HYP-093 | Parabolic gapper fade, +30% stop | VALID_BUT_BELOW_FLOOR (p=0.031, DSR 0.987) | Real; the stop *is* the strategy (caps a −937% tail); yield below floor; needs HTB borrow → fails the golden rule. |
| HYP-094 | Overnight short of weak-closing gappers | NOT_SIGNIFICANT (p=0.10) | The fade is intraday exhaustion, not a persistent short. |
| HYP-095 | NQ high-VIX prior-down-day long | VALID_BUT_BELOW_FLOOR (p=0.013, n=40) | Fires on 8% of sessions — real and useless. |
| HYP-096 | Options wrapper for the gapper fade | WITHDRAWN | Parity re-prices borrow into premium; and a chat session claimed a rule that didn't exist on disk. |
| HYP-097 | Evidence-based worst-case sizing | NOT_CLEARED | LULD collar physics dominated the empirical overshoot; yield fell. Refuted by exchange structure, not statistics. |
| HYP-098 | FVG × fractal corridor on NQ 5-min (720 cells) | mined out, holdout untouched | Best cell +0.025%/day with full look-ahead selection — don't burn holdout on what fails in dirty data. |
| HYP-099 | Regime-conditional gapper fade | NOT_SIGNIFICANT | Scan showed p=0.05; both prereg variants failed holdout, one sign-flipped. Mined mirage. |
| HYP-100 | HYP-093 + 25% stop, forward-only seal | PREREGISTERED, evaluates N≥40 / 2027-01 | Capacity study: 93% of the gapper universe is optionless HTB micro-cap. |
| HYP-101 | Relax gapper threshold below 100% | stopped at step 1 | 100% is where the parabola breaks; relaxation dilutes; 11:00 entry strictly worse. |
| HYP-102 | Continuation long on non-faders | stopped at step 1 | Separating features exist but every long rule loses in-sample — non-faders aren't identifiable at 10:30. Gapper table mined out. |
| HYP-103 | EV-optimal prop-challenge config | REGISTERED, PENDING | Derived from a grid, labelled not-evidence; chose off the boundary of the grid on purpose. |
| HYP-104 | Down-gap continuation short (megascan survivor of 77,016) | NOT_CONFIRMED | Dirty Sharpe 2.32 → holdout 0.36, p=0.35. The best of 77k is still overfit. |
| HYP-105 | Long momentum on parabolic gappers | REFUTED_LOOKAHEAD | Universe defined by gain at 10:30, entered at 09:31 — a clean holdout cannot save a contaminated universe. |
| HYP-106 | Leak-free runner filter on the long | REFUTED_LOOKAHEAD | Leak-free features on a look-ahead sample are still look-ahead. |
| HYP-107 | De-biased runner filter, honest universe | REAL_BUT_MARGINAL, execution unresolved | ~10× smaller than the look-ahead version (n=57, median +5.4%, p=0.0005); spread model was 11× too pessimistic (TICK-039); sealed holdout list never committed. |
| HYP-108 | Seykota-Lipschutz per-pair time exits | RESEARCH, blocked from CONFIRMED | Claimed +166% Sharpe from *estimated*, not re-simulated, P&L — a substitution is a hypothesis, not a result. |
| HYP-109 | Post-shock abstention (flat 5 sessions) | NULL — KILL_DIRECTIONAL | Shocks predict volatility (1.36×, p≈0), not sign. Abstention overlay later shown to be noise on the delta. |
| HYP-110 | Overnight-only holding, ten ETFs | INCONCLUSIVE (abort) / substantive kill | Break-even 0.02 bp; OVERNIGHT-QQQ does not generalise to a basket. |
| HYP-111 | Post-shock intraday retrace→reclaim + fade, 2020–26 | path INCONCLUSIVE (3.6% fires); fade passed (corrected +0.087%) | The path is too rare; the fade was 2020 and 2025. Fade figures overstated 0.06% by a cost-sign bug, disclosed. |
| HYP-111a | Same, 2023-06→2026-07 pilot | VALID_BUT_BELOW_FLOOR | "The pass is the incumbent's failure, not the structure's success." |
| HYP-112 | Post-shock ATM straddle vs control | INCONCLUSIVE (abort) / hard null | Shock straddle −22% vs −13% on premium; IV rises 28%, realized 16% — you cannot buy the magnitude. |
| HYP-113 | Fade dose-response in shock size | FLAT | p99+ down-days carry no fade — "fade harder when bigger" is the trade that gets run over. |
| HYP-114 | Deploy the fade: unseen 2016–19, 20 ETFs, exit | FAIL / FAIL / NO_DIFFERENCE | −0.027% on unseen years (4/10), +0.001% on 20 ETFs — a regime, not an edge. Found the sign bug in 111/111a/113. |
| HYP-115 | The incumbent EW-10 basket itself | FRAGILE | OOS 2007–14 Sharpe 0.22 CI [−0.35, +0.88], −54% GFC, loses to 60/40 everywhere; not lucky, just beta. |
| HYP-116 | Shock-deferred contributions vs DCA | POLICY_FAILS | Ratio 0.993; loses in 88% of 5-year windows. Shock signal closed at every horizon. |

## Table 2 — named (non-numbered) entries

| id | tested | verdict | lesson |
|---|---|---|---|
| H1-RATE-ACCEL | Rate-differential acceleration | NOT_SIGNIFICANT (p=0.44) | Carry works off the level, not the derivative. |
| H2-VOL-REGIME | MED vol beats tails | NOT_SIGNIFICANT, LOW POWER | The pre-registered LOW bin was empty — flagged, not re-binned. |
| MOMENTUM-CARRY-ORTHO | 63d momentum agreement on carry | NOT_SIGNIFICANT (p=0.85) | Ran backwards — AGAINST-momentum did better. |
| OVERNIGHT-QQQ | Overnight vs intraday QQQ | VALID_EDGE (p=0.005) | 5.5 bp/day overnight, Sharpe 0.97 — real on QQQ; did not generalise (HYP-110). |
| OVERNIGHT-CARRY-DIVERSIFICATION | Does it diversify carry in crisis? | REJECTED | ρ rises to 0.42–0.57 in crisis — only crisis correlation decides. |
| V007-HOLD-VALIDATION | Per-pair hold overrides | NOT_SIGNIFICANT_ROLLED_BACK | Passes OOS and permutation, fails walk-forward — regime-concentrated. |
| GBPUSD-GRADE-EXPECTANCY | A vs B grade | NOT_SIGNIFICANT (p=0.78) | Don't build a grade-based size multiplier. |
| ESNQ-BIAS-01 | 5-input pre-market NQ bias | NOT_SIGNIFICANT | 51.4% vs 51.6% null — below its own permutation null. |
| VRP-001 | Volatility risk premium | DATA_INSUFFICIENT | Premium exists and is a true diversifier vs carry; the strategy was unrunnable without chains. |
| VRP-001-OPTIONS | Iron condor on real SPY chains | NO_TRADES | A fully specified rule that never fires is its own falsification. |
| REGIME-ROUTER-SCREEN | USDJPY gate cross-pair | NOT_SUPPORTED | The HYP-027 gate is inert in current data. |
| SENTIMENT-ECON-SURPRISE | Release surprise → FX | NOT_SIGNIFICANT | 517 days collapse to 164 events once de-overlapped; surprises are priced. |
| SENTIMENT-COT-POSITIONING | COT extremes → contrarian FX | NOT_SIGNIFICANT (p=0.15) | Positioning is +0.18 correlated with carry — not even a diversifier. |
| HYP-071-GOVFLAG | Un-prereg'd recompute revives a sealed verdict? | governance_flag | A recompute that contradicts a sealed verdict needs its own lock. |
| BENCH-THROUGHPUT | Backtest throughput | MEASURED | 12.4k/s single-core; the 148k claim wasn't beaten (numba inactive on 3.14). Measure the number you quote. |

Gaps: HYP-068/069/070 have no record anywhere; HYP-086/087/088 were built (partly) but never registered or sealed — an open governance gap.
