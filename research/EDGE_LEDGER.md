# EDGE LEDGER — what Alta has actually proven, and where the proof lives

**Read this before claiming an edge exists or doesn't.** One page, kept current, every claim
points at a sealed pre-registration (hash), a one-shot test script, and a result file in this repo.
The authoritative machine record is `data/agent/hypothesis_ledger.json`; this page is the human
index of it. Rules: a claim gets on this page only with a sealed prereg and a single run; a null is
recorded with the same care as a pass; corrections are appended, never overwritten.

_Last updated 2026-09-03 (HYP-109 → HYP-116, eleven preregs)._

**Lessons, one per hypothesis, HYP-001 → HYP-116:** `research/HYPOTHESIS_LESSONS.md`.

## Status line

| what | status | evidence |
|---|---|---|
| **Forex v015 carry portfolio (live)** | **CONFIRMED, LIVE** — OOS Sharpe 1.25, permutation p<0.001; regime-fragile (pays in rate-trending years) | `CLAUDE.md` "Current live state"; ledger HYP-045 |
| Time exits beat trailing stops (v015 FX) | CONFIRMED on the frozen config | ledger HYP-059/060 |
| Post-shock **magnitude** (next-week RV 1.36×) | REAL, not monetisable — options over-price it | HYP-109(a), HYP-112 |
| Post-shock **direction / fade** | **NOT AN EDGE** — one regime (2020–2026, +0.087%/event-day), absent 2016–2019 and on 20 other ETFs | HYP-109, HYP-111, **HYP-114** |
| **The incumbent EW-10 basket itself** | **FRAGILE** — OOS 2007–14 Sharpe 0.22 CI [−0.35, +0.88], maxDD −54%, loses to 60/40 everywhere; not lucky (73rd pct) | **HYP-115** `f1b66afc` |
| Shock-deferred contributions vs DCA (the fade as a policy) | **POLICY_FAILS** — ratio 0.993, wins 12% of 5-yr windows | **HYP-116** `b76837ac` |
| Everything else tried 2026-09-02 | null | table below |

**There is currently no proven retail-clean equity/ETF edge in this repo.** The one that looked like
one (the fade) failed its out-of-regime and out-of-universe test the day after it was found.

**Retracted, explicitly, at the operator's request (2026-09-03):** the statement *"after a sharp move
down, the crowd's continuation is the wrong side — fade it, and fade it harder when the drop was bigger;
10/10 instruments"* is **false** on the desk's own data. "Harder when bigger" was killed by HYP-113 (p99+
down-days carry no fade); "10/10" was a cost-sign error (8/10 corrected); the fade itself is absent on
2016–2019 and on 20 other ETFs (HYP-114). It is the best-looking *candidate* the desk has tested, and it
is not an edge. Do not describe it as one.

## The 2026-09-02 program — nine sealed tests

| id | hypothesis | hash | verdict | one-line read | files |
|---|---|---|---|---|---|
| HYP-109 | flat 5 sessions after a p90+ shock (ten ETFs, 2015–26) | `a7f32774` | NULL — KILL_DIRECTIONAL | RV 1.36× after shock (real); 5-day drift −0.24%/wk; abstention = noise on the delta | `data/research/hyp109/` |
| regime test | SPY RV21/median252, pre-declared | — | STORY_ONLY | incumbent Sharpe by regime CI [−0.75, +1.61] | `research/TAXONOMY_2026-09-02_where_edges_can_exist.md` |
| HYP-110 | overnight-only holding, ten ETFs | `d981bf1d` | INCONCLUSIVE (abort) / substantive kill | premium is not overnight on this set; break-even 0.02 bp | `data/research/hyp110/` |
| HYP-111a | intraday retrace→reclaim after shock, 2023-06→2026-07 | `41b7b6a1` | VALID_BUT_BELOW_FLOOR | fires 12.8%; pass driven by naive continuation losing | `data/research/hyp111/HYP-111a_VERDICT.md` |
| HYP-112 | post-shock ATM straddle vs control, 2020–26 (ThetaData) | `f2d55244` | INCONCLUSIVE (abort) / hard null | −22% vs −13% on premium; IV rises 28%, RV 16% | `data/research/hyp112/` |
| HYP-113 | fade dose-response in shock size | `f3cf34d6` | FLAT | p99+ down-days carry no fade | `data/research/hyp111/HYP-113_VERDICT.md` |
| HYP-111 | scoped path + fade, 2020–26 (Alpaca SIP) | `d3d52582` | path INCONCLUSIVE (3.6% fires); fade HOLDS (corrected +0.087%, CI [+0.004, +0.174]) | one regime | `data/research/hyp111/HYP-111_VERDICT.md` |
| **HYP-114** | **deploy the fade: unseen 2016–19, 20 new ETFs, exit** | `0efc2fdc` | **FAIL / FAIL / NO_DIFFERENCE** | −0.027% on unseen years (4/10); +0.001% on 20 ETFs (11/20); account CAGR +1.3% | `data/research/hyp114/VERDICT.md` |

| HYP-115 | the incumbent EW-10 basket as hypothesis: OOS 2007–14, lucky-basket, stress | `f1b66afc` | **FRAGILE** | diversified beta; OOS Sharpe 0.22, −54% GFC; 60/40 beats it on every risk measure | `data/research/hyp115/` |
| HYP-116 | shock-deferred contributions vs DCA, 2007–2026 | `b76837ac` | **POLICY_FAILS** | ratio 0.993; DEFERRED wins 12% of rolling 5-yr windows; closes the shock signal at every horizon | `data/research/hyp116/` |

Correction (2026-09-03): HYP-111/111a/113 fade figures were overstated by 0.06%/event-day (cost-sign
error, found by HYP-114). Corrected numbers are in `data/research/hyp114/VERDICT.md`; sealed verdicts
are annotated, not rewritten. Multiplicity after this program: **1557 trials** (`mined_n._total`
1543 + 14 declared claims). Any new prereg starts at 1558.

## Where the map says an edge can still exist (for a trader with no speed, size, or information)

`research/TAXONOMY_2026-09-02_where_edges_can_exist.md`. After HYP-114: bucket (i) information is
closed for direction and magnitude; (ii) structure closed for overnight; (iii) sizing closed (abstention,
vol-targeting prior weakened); (iv) behaviour — the incumbent (buy-and-hold) is the null and has not
been beaten. Untested and retail-clean: earnings-catalyst momentum on single names (needs a PIT
event build, `sovereign/pit/`). The desk's real edge remains the FX carry book and the process on
this page.

## Standing infrastructure that makes the next claim cheap

- Alpaca SIP serves 1-min bars from 2016-01-04 on the existing key (`research/hyp111/alpaca_1m.py`,
  cache `data/cache/alpaca_1m_rth/`, 8,000+ sessions on disk). Do not buy ThetaData stock tiers for this.
- ThetaData OPTION.VALUE (active) serves EOD chains + 1-min option quotes from 2020-01 via
  `research/hyp111/theta_v2.py` (v2 API on :25510; terminal needs Homebrew JDK). 2,876 chains cached.
- Date-block bootstrap (`research/hyp111/date_bootstrap.py`) — use it; pooled-event resampling
  overstated n by ~2.7× on shock dates.
- Prereg/seal/verify/adjudicate helpers: `research/hyp111/prereg.py`.
- Forward log of the fade rule, observation only: `scripts/research/fade_forward_log.py` (kept running
  as a live null check; `com.alta.fade_forward_log.plist` tracked-not-loaded).
