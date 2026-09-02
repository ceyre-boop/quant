# Brief — find one real edge

**Written 2026-09-02 as priming. Read this before touching anything.**

Goal: one edge that clears the CONFIRMED bar. Not a survey, not a scan, not a
shortlist. One.

---

## 0. Read these first, in this order

1. `archive/AGENT_DIRECTIVE.md` lines 324-390 — **the ten standing rules**. They
   are not advice. Rule 10 exists because two autonomous audit passes on
   2026-07-20 produced six false claims that would have deleted 11 live modules.
2. This file.
3. `sovereign/pit/README.md` — the point-in-time contract. Every read is
   `as_of(t)`; leakage is structurally blocked, not discouraged.

---

## 1. There are THREE research engines and two of them are decorative

| Engine | State | Use it? |
|---|---|---|
| **A. Research Factory** `sovereign/autonomous/research_factory.py` | launchd job has not fired since **2026-08-24**. 91 QUEUED rows waiting. `live: false` in `config/autonomous.yml`. Its validator registry has **2 keyword entries**, so ~90 of the 91 would emit `BLOCKED_NO_VALIDATOR` even if it ran. | **No** |
| **B. Nightly Claude agent** `com.alta.research_agent` | Runs, but its queue has had **0 QUEUED since 2026-07-09** and STANDING RULE 4 forbids it refilling its own queue. It correctly no-ops. Also: the installed plist inlines `ANTHROPIC_API_KEY` from `.env`, which burned the API quota and blacked it out for 8 nights (08-19 → 08-30). `scripts/run_agent.sh` fixes exactly this; the plist was never repointed at it. | **No** |
| **C. Manual / gauntlet track** `scripts/discover.py`, `research/yield_frontier/gauntlet_run.py`, `scripts/research/*` | **The only engine that has ever produced a CONFIRMED verdict.** Human-driven, prereg-first. | **Yes. Drive this one.** |

Do not spend the session repairing A or B. They are recorded as broken in
§7 so an operator can fix them; they are not the path to an edge today.

---

## 2. What it takes to be CONFIRMED

Authoritative ledger is **`data/agent/hypothesis_ledger.json`** (88 entries).
`data/hypotheses_ledger.jsonl` (7 rows) is a dashboard summary — not authoritative.

A candidate must clear **all** of:

1. Pre-registered **before any result exists** — hash-locked, `verdict: null`
2. Permutation **p < 0.05 at n ≥ 10,000** shuffles (the 500-perm discovery gate is screening only)
3. Walk-forward present; in regime mode, **positive costed Sharpe in every OOS year**
4. Both-sides test
5. **DSR prob ≥ 0.95** with an honest `n_trials`
6. **BH survivor** at alpha 0.05 across the family frozen at prereg
7. Bonferroni-corrected p < 0.05 if it came from `backtester/scanner.py`
8. Positive holdout Sharpe on an **untouched** window
9. Tail: event p5 ≥ −0.60, and no single event worse than −0.60
10. **Economic floor ≥ 0.05%/day over the full calendar** (0.0 on no-event days)
11. n ≥ 40 signal sessions; pooled N ≥ 50 de-overlapped
12. Human approval (`approve_edge.py`). Nothing auto-deploys.

**The floor is what kills things, not significance.** HYP-093 hit boot_p=0.0312
and DSR=0.987 on 559 events and still landed `VALID_BUT_BELOW_FLOOR` because it
made +0.023%/day against a 0.05%/day floor. Design for the floor from the start
or do not start.

### The multiplicity debt is stale and it matters

`data/research/yield_frontier/mined_n.json` `_total` is now **1543**. The
HYP-093/094/095 preregs were frozen at **809**. `record_mined()` is append-only
and refuses to shrink. **A new yield-frontier candidate must declare n_trials ≈
1543+, not 809** — that is a materially higher DSR hurdle than the one HYP-093
cleared. You cannot spend 809 twice.

---

## 3. What is already dead — 25 in the graveyard

Do not re-propose. Revival requires **NEW DATA**, not a new lookback.

Killed: VRP-001 · ESNQ-BIAS-01 · GBPUSD-GRADE-EXPECTANCY · H1-RATE-ACCEL ·
H2-VOL-REGIME · HYP-029 · HYP-032 · HYP-033 · HYP-036 · HYP-038 · HYP-039 ·
HYP-043 · HYP-044 · HYP-052b · HYP-053 · HYP-061 · HYP-064 · HYP-065 · HYP-066 ·
HYP-071 (METRIC_ARTIFACT) · HYP-085 · HYP-089 · HYP-090 · HYP-091 · HYP-092 ·
MOMENTUM-CARRY-ORTHO · SENTIMENT-COT-POSITIONING · SENTIMENT-ECON-SURPRISE ·
OVERNIGHT-CARRY-DIVERSIFICATION

Specifically closed and worth naming because they are tempting:

- **Generic ORB / intraday breakout continuation.** MEGASCAN v2, 5,040 configs,
  6 opening-range lengths, 14 liquid names: best Sharpe **0.210**, median
  **−4.81**, **1 positive in 5,040**, **0 FWER survivors**. On NQ over 6.5 years:
  *"ORB/day-trade families don't crack the top ten at honest costs."*
- **The gapper table is mined out at this resolution.** HYP-101 and HYP-102 both
  stopped at step 1 as correct negatives. HYP-102: *"the non-faders are not
  identifiable at 10:30 with the features available."*
- **HYP-104** was the best of a **77,016-hypothesis megascan**: dirty Sharpe 2.32
  → **holdout 0.36, p=0.35**. Zero survived Bonferroni. That is what mining this
  substrate produces.
- **HYP-105/106 REFUTED_LOOKAHEAD** — universe selected on the 10:30 outcome,
  entered 09:31. The look-ahead was worth +50%/event, which is why the realistic
  fill model "survived": it was validating a biased number.

---

## 4. HYP-107 — correcting the record, twice

I previously called this "a free hit — spread was measured 10× cheaper than the
objection assumed, nobody re-ran it." **That was wrong in one direction and the
correction to it was wrong in the other.** Both errors are recorded here so
neither gets repeated.

**What HYP-107 is:** among ≥30% morning gap-ups, buy 09:31 exit 10:30, filtered
to `overnight_gap ≤ 0.577 AND log10(first-minute volume) ≤ 5.854` (frozen
in-sample, `research/gapper/hyp107_shadow.py:38-39`). Holdout n=57, **gross**
median +5.4%, mean +15.3%, win 70%, tail 4.4, permutation **p=0.0005**, against a
blind-gap-buying baseline that loses (gap≥100%: n=231, median −8.0%).

**Error 1 — "nobody re-ran it" is half wrong, and the reason is fatal.**
TICK-039 *did* attempt the re-run and **could not**: the sealed holdout event
list was never committed. Only the thresholds survived. Rebuilding from the
documented 70/30 split gives **98 filtered events against the original 57** — a
+72% superset, meaning the original applied an additional condition that is now
unrecoverable. The doc is explicit: *"nothing below is a holdout rerun and no
verdict can be sealed from it."* **No HYP-107 verdict is available at any cost
model until the sealed set is recovered or re-derived by a committed procedure.**

**Error 2 — the "live spreads are 6× worse" reading is FALSE, and I fixed the
cause today.** `hyp107_tracking.json` reports `median_realized_spread: 0.0378`,
which was read as a 3.78% bid-ask spread contradicting the 0.55% NBBO
measurement. It is not a spread. `hyp107_shadow.py:162` computes
`(high − low) / entry` — the **09:31 bar's high-low range**. A 3-4% one-minute
range on a microcap gapper and a 0.55% bid-ask spread are both unremarkable and
entirely consistent. The field is now `first_bar_range_pct`, with the old key
kept as a deprecated alias for series continuity.

**So the cost objection does stand corrected**: `TICK-039` measured **313 real
NBBO observations at the frozen 09:31 instant — median 0.55%**, p90 2.06%, vs a
legacy model charging 6.206% round-trip. Overcharge factor **11.3×**, and it
flips the sign (−1.90% → +2.26% on identical events).

**But the real blocker was never the spread.** Even at +2.26% net median,
constitutional yield is **0.0366%/day** against the **0.05%/day floor** — and the
yield formula in use is calibrated ~3× too generous. It lands below the floor
even being flattered.

**Live shadow, as of 2026-09-01:** n=31 of a 40 target, median return **+6.14%**,
win rate **83.9%**, tail 4.03 — running *ahead* of the +5.4% / 0.70 backtest.

**Precise re-adjudication requirement**, in order:
1. Recover the original sealed 57, or re-derive them by a committed, documented
   procedure. Until then no verdict exists.
2. Pre-register the verdict rule **on constitutional yield, not median return**.
3. Let the shadow reach n=40.

---

## 5. Data actually on disk — more than the last brief claimed

| Asset | Size | Window | Note |
|---|---|---|---|
| `data/cache/minute_bars/` | **12,284+ files**, 268 MB | 2024-01-03 → **2026-09-01** | Alpaca **SIP**, adjustment=all, premarket from 04:00 ET. Refreshed today. **23 symbols with ~504 continuous days**; ~600 microcaps are single event-days |
| `data/research/gapper/cache/grouped/` | 261 files, 54 MB | 2025-07-01 → 2026-06-30 | **Polygon full-market grouped daily — 12,474 tickers/day.** A whole-market panel for one year |
| `data/cache/daily_universe/` | 158 symbols | **2014-01-02** → 2026-07-16 | Heavily biotech-weighted |
| `data/research/gapper/candidates.csv` | 11,396 events | 2025-07-02 → 2026-06-30 | 2,922 distinct tickers |
| `filings` (fundamentals.db) | **6,944** | event_date 2015 → 2026-08-28 | `knowable_at` is the EDGAR **acceptance instant**. **87% is Form 4; item 2.02 is only ~270** |
| `data/harvest.db::trades` | 3,467,956 rows | — | megascan substrate |
| `data/sentiment.db` | 104 MB | — | 9 feeders, **FX-pair-keyed, not equity-ticker-keyed**. news=181 rows, gdelt=**0** |

**The binding constraint is universe at minute resolution, not history.** Daily
resolution has a genuine wide panel (12,474 × 261). Minute resolution has 23
continuous names. Alpaca free caps history at ~2 years, so 2022-2024 intraday
does not exist and **no 5/10-year intraday claim is validatable here.**

---

## 6. Credentials — verified live today, 14/17

Run `.venv313/bin/python scripts/probe_new_keys.py` to re-verify. It never prints
key material.

**Two documented gaps are now closed:**
- **FMP** `/stable/` — forward analyst estimates (epsAvg to 2030), analyst
  **rating changes** with dates, est-vs-actual including forward rows. The v3 API
  is retired; everything must go through `/stable/`, and `analyst-estimates`
  needs a `period` arg or it 400s.
- **Finnhub** — earnings surprise history, and an earnings **calendar carrying
  the BMO/AMC hour flag** (1,017 events per 10-day window).

`sovereign/fundamentals/providers/{fmp,finnhub}.py` still raise
`NotImplementedError` with stale comments claiming these need Premium. They do
not. `fund_estimate_snapshot` is empty and its `FactSpec` is already correct at
`sovereign/pit/spec.py` — it is the drop-in target.

**Two genuinely new timestamped event classes, zero existing code:** Senate LDA
(56,238 filings/2026, `dt_posted`) and openFDA (20.7M records).

**Needs operator action:** BEA (valid but not activated), Census (invalid key).

---

## 7. Broken, recorded so it can be fixed — not today's job

1. `com.alta.research.factory` has not executed since 2026-08-24 (`pended
   nondemand spawn`). 91 candidates queued behind it. Nothing alerts.
2. `com.alta.research_agent.plist` inlines the `ANTHROPIC_API_KEY` auth leak that
   `scripts/run_agent.sh` exists to fix. Repoint it at the wrapper before
   reloading or the quota dies again.
3. Validator registry has 2 keyword entries; the generator emits categories that
   match neither. Structural block on the whole autonomous path.
4. `factory_report.json` reports `queue_depth: 0` when it is **91** — the
   dashboard is lying about the one number that would have surfaced (1) and (3).
5. `mined_n` cited as 809 in preregs; honest count is **1543**.
6. RQ-006 needs 20 trades in 30 days from an ICT path producing ~2 fills/90d.
   Unfixable as scoped; retire or re-scope it.
7. `data/agent/ict_causal_chain.jsonl` is 167 MB and uncommitted in the tree.

---

## 8. Where to actually look

Ranked. The first two are mine; the rest came from the source audit.

**(1) Finish HYP-107 properly.** It is the only thing on the board with a real
holdout, a real mechanism, a live shadow running ahead of backtest, and a
*known, nameable* blocker. Recover the sealed 57 first. Everything else is a new
hypothesis paying a 1543-trial DSR tax; this one already paid its.

**(2) Analyst downgrade into a weak tape — the forced-seller cohort.**
Mechanism: a downgrade is a coordination signal that releases mandate-constrained
holders, so the reaction should be asymmetric versus matched upgrades, and larger
when the constraint binds. Cohort: FMP `/stable/grades` tier crossings (not
reiterations), split on prior 20-day return, **with the upgrade cohort as the
built-in control** — the asymmetry is the test, which kills the "it's just beta"
objection. Evidence: new `grades` table + `FactSpec` × `daily_universe` (158
symbols to 2014). Honest n: **1,500-4,000 rating changes, 600-1,200 tier
crossings** — the largest of anything available. Daily resolution only, because
grades carry no intraday hour. **Validate the `date` field against known
downgrades before spending a week on it** — if any rows are backfilled or
date-shifted, a next-open rule silently becomes a same-day rule.

**(3) Catalyst-conditioned gapper fade.** Not a new edge — a mechanism test on
the one confirmed one. Split the 11,396 candidates by whether a filed catalyst
(8-K within 24h, or same-morning news) exists. Hypothesis: gaps with no
identifiable catalyst are promotion/squeeze and fade hardest. Costs nothing;
every input is on disk. **Audit the news cache's fetch coverage first** — it was
built *for* the gapper study, so absence may be an artifact of what was
requested, not of the world.

**Bet against, and why:** macro-release reaction (honest n≈60 at the single most
arbitraged instant of the month, and it will not survive scanning horizons ×
instruments); BMO-vs-AMC drift (heavily confounded with market cap — small caps
report BMO far more often, so any result is a size effect in costume, and it
needs a within-size-decile design from the start, not as a robustness check).

**On the cost model — checked today, so you do not have to.** `TICK-039` §7
listed the merge into `_half_spread` as "next", which reads as unfinished. **It
did land and it is active.** `backtester/realistic_fills.py:115` —
`_half_spread(..., measured=True)` defaults to the measured model, line 122
dispatches to `_half_spread_measured`, and all three call sites (154, 168, 173)
take the default. `_half_spread_legacy` survives only for deliberately
reproducing a pre-TICK-039 number.

So: **any result you compute now already uses the measured spread.** What is
biased pessimistic ~11× is every *previously recorded* gapper figure computed
before that commit. Corrected figures move **up**. Do not re-apply the
correction to a number that already has it.
