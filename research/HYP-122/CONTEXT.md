# HYP-122 — Dufour & Engle (2000): does inter-trade duration still condition price impact, and is any of it tradable at retail?

**Layer 2: Stage-Specific Hypothesis Definition**

*Created: 2026-09-29 · Stage: SCOPED (paper read in full; prereg DRAFTED, not sealed)*

---

## Status

- The paper has been **read in full** (47 pp., all tables, footnotes and figures). See `output/extraction.md`.
- The data needed exists on the current Alpaca key at $0 (SIP tick trades + NBBO quotes, 2016 → today; verified 2026-09-29).
- **Not yet done:** no prereg sealed, no hash lock, no run. Nothing here is a claim that an edge exists.
- **Prior expectation, stated before any data is touched:**
  - Stage A (replication): **LIKELY TO REPLICATE in sign, WEAKER in size.** The mechanism is a classic microstructure result, but the paper's own robustness check (Table VI) cut it to 8/18 stocks once size and spread were included.
  - Stage B (tradability): **NOT_TRADABLE.** The impact is realised within the next few quote revisions, which is under the latency of any retail path. What remains is a magnitude and liquidity-state signal, which is exactly what HYP-119 and HYP-120 already returned as MAGNITUDE_ONLY.

---

## The claim, extracted (see `output/extraction.md` §1–4)

After a signed trade x_t (+1 buyer-initiated, −1 seller-initiated, 0 at the mid), the mid-quote change r_t following it is modelled as
`r_t = Σ a_i r_{t−i} + Σ_{i=0..5} [γ_i + δ_i ln T_{t−i}] x_{t−i} + …`,
where T is the inter-trade duration. The paper finds **δ < 0** (shorter duration → bigger impact) for 13/18 NYSE stocks in Nov 1990–Jan 1991, faster price convergence in fast markets (about 4 min vs 23 min for FNM), and stronger trade-sign autocorrelation when durations are short.

Three things the paper does **not** provide, which is why the tradability question is separate: a decision rule, a cost model, and any out-of-sample period.

## Why bother, given the prior

1. **It is cheap.** The data is free, the model is OLS with HC errors, and the ACD only matters for calendar-time impulse responses. One session of pull plus one of compute.
2. **A clean null is worth recording.** `HYPOTHESIS_LESSONS.md` currently lacks a direct test of the informed-trading-intensity story on modern tape. Two prior microstructure hypotheses died as MAGNITUDE_ONLY, and this one names the mechanism they may share.
3. **The tick infrastructure is reusable.** The signing and duration machinery is the same one any intraday execution question in the gapper program (HYP-093/107) needs. Building infrastructure is unrestricted; ignition is not (`RISK_CONSTITUTION.md` Art. 6).

What it is **not** for: the live book. FX carry runs 4–14 trades a year through OANDA, and there is no consolidated FX tape to sign trades against. Nothing in this folder touches `forex_*`, `decide_exit`, or `config/`.

---

## Stage A — Replication on modern tape (mechanism validation)

**Question:** On 2016–2025 US equities, is the duration coefficient on signed-trade impact still negative, and does it survive size and spread controls?

**Universe (rule, not a list):** the 20 NYSE-listed common stocks with the highest 60-day median dollar volume as of the first day of each sample window, excluding ETFs and anything under $5. Recorded in `output/universe_<window>.json` before any regression is run.

**Windows (three regimes, 20 trading days each):**
- W1: 2016-06-01 → 2016-06-29 (earliest SIP coverage on this key).
- W2: 2020-03-02 → 2020-03-27 (stress).
- W3: 2024-06-03 → 2024-06-28 (calm, sub-penny, odd-lot-heavy).

**Construction, pre-specified:**
- Regular-hours prints only (09:30:00–16:00:00 ET). Drop the opening print and any trade before the first NBBO of the day.
- Conditions: keep regular (` `, `@`) and intermarket sweeps (`F`). Odd lots (`I`) are **excluded** in the primary run and **included** in one sensitivity run. Anything else (late, out of sequence, derivatively priced) is dropped.
- Same-timestamp, same-side prints across venues are merged into one event, which is the modern analogue of the paper's same-venue merge.
- Sign: mid-quote rule against the prevailing NBBO at the trade timestamp with **no lag**. Trades at the mid are 0, as in the paper. Sensitivity: tick-test fallback.
- Duration: milliseconds since the previous event, `ln(1 + T_ms)`. The paper's `+1 s` on seconds is replaced by `+1 ms` on milliseconds because modern durations are sub-second.
- Diurnal adjustment: piecewise-linear spline with the paper's nodes (09:30, 10, 11, 12, 13, 14, 15, 15:30, 16).
- Return: 100·Δln(NBBO mid) following the event, as in the paper.
- Model: the paper's eq. (9), 5 lags, OLS, White HC standard errors. Then eq. (11) with `ln(size/100)` and spread added.

**Primary statistic:** per stock and per window, the sum Σδ_i in the return equation and its Wald p-value.

**Pass (replication):** Σδ < 0 with p < 0.05 for **≥ 12 of 20** stocks in **every** window, *and* the same in the eq. (11) specification (size and spread included) for **≥ 10 of 20**. Cross-stock p-values go through BH at q = 0.05 (`sovereign/discovery/gate.py`, the existing gauntlet).
**Fail:** anything less. Record as `MECHANISM_NOT_REPLICATED` with the counts.

**Trials:** 1 claim (A1) plus 1 robustness variant (A2, eq. 11). The stock-level tests are the unit of replication, not separate trials.

## Stage B — Tradability kill-test (runs only if A passes)

**Question:** After a *fast-market* signed event, does the mid-quote move enough, over a retail-feasible horizon and after retail latency, to clear costs?

- **Fast** = the event's diurnally-adjusted duration is in the bottom decile of that stock-window. The paper's own contrast (its fastest vs slowest day) is the template.
- **Entry:** at NBBO **+1.0 s after** the event (retail latency), crossing the spread in the direction of the trade sign.
- **Exit:** at NBBO mid after horizon h ∈ {10 s, 60 s, 300 s}, again crossing the spread.
- **Cost:** full quoted spread at entry and exit (from the same quote stream) plus $0.0000278/share SEC and TAF where applicable. No slippage model beyond the quoted spread, so this is **generous** to the strategy.
- **Statistic:** mean net return per event with a block-bootstrap CI (blocks = trading days), and a sign-permutation placebo.
- **Pass:** net > 0 with the CI excluding 0 and permutation p < 0.01 at **any** horizon, in **all three windows**.
- **Fail:** `NOT_TRADABLE`. The mechanism can be real and still unreachable at retail latency, which is the expected outcome.

**Trials:** 3 horizons = 3 trials (B1–B3). Stage B is not run if A fails, so those trials are not spent.

## Stage C — noted only, not scoped, not counted

A repo-relevant follow-on if A passes: **duration-gated entries for the gapper fade** (HYP-093 universe; HYP-107 shadow signals under `data/research/gapper/`). The paper's reading is that short durations mark informed flow. Fading a gapper *while* durations are short means fading informed traders, and waiting for durations to lengthen might mean fading after they are done. This is a different hypothesis with its own prereg, and it is listed so it isn't re-derived. It is not part of HYP-122's trial count.

---

## Multiplicity

`EDGE_LEDGER.md`: any new prereg starts at trial **1649**. HYP-122 books A1–A2 on sealing and B1–B3 only if A passes. It totals 2 to 5 trials. DSR at the running count applies to any Stage B pass.

## Data and build (unrestricted; ignition is not)

- Pull: `data.alpaca.markets/v2/stocks/{sym}/trades` and `/quotes`, `feed=sip`, paged at 10,000 rows. Quotes are the bottleneck: liquid names run into the hundreds of thousands of NBBO updates per day. Twenty names for 60 days is roughly one overnight pull at the 200 req/min limit. Store under `data/research/hyp122/ticks/<window>/<sym>.parquet` (gitignored, like all pools). Only the derived per-event table (sign, duration, Δmid, size, spread) is small enough to keep.
- Code home: `research/HYP-122/scripts/` for fetch, sign, and estimate. pandas/numpy/statsmodels plus the gate functions above. Nothing from `ict/`, and nothing importable by the execution path.
- Existing pieces to reuse: `permutation_test` in `_config/gate_functions.md`; BH, `deflated_sharpe_ratio` and the bootstrap p-values in `sovereign/discovery/gate.py`; Alpaca key loading from `.env` as the other `scripts/` do. No tick fetcher exists in the repo yet; the block bootstrap by trading day is new.
- Cost: $0 for data, one session for the pull, one for the estimation and writeup.

## Process to close the folder

1. Seal the prereg: write `output/prereg.json` from the Stage A and B specs above, hash-lock it, log the hash in both ledgers. **Operator seals**, as in the repo's research method.
2. Pull the three windows. Record the universe files before any regression.
3. Run Stage A. Write `output/stage_a.json` and `output/stage_a.md`, with counts per window and per specification.
4. If A passes, run Stage B and write `output/stage_b.*`. If A fails, stop.
5. Verdict in `output/verdict.md`, a ledger entry with the verdict, and a `HYPOTHESIS_LESSONS.md` line. `EDGE_LEDGER.md` changes only on a Stage B pass.

## Success criteria for this stage (SCOPED)

- [x] Paper obtained and read in full, with the hash recorded
- [x] `output/extraction.md`: model, data, every headline number and its location, robustness, what the paper does not claim
- [x] Data availability verified on the existing key
- [x] Stages, universe rule, construction, pass/fail and trial counts written
- [x] Ledger entries (`IN_RESEARCH / scoped`) in both ledger files
- [ ] Prereg sealed (operator)
- [ ] Stage A run

## Inputs

**Layer 3:** `_config/trading_philosophy.md` (Tenet 1: a claim needs a falsifiable rule), `_config/gate_functions.md` (BH, block bootstrap, permutation), `_config/risk_constitution.md` (Art. 6), `shared/hypothesis_ledger_schema.md`

**Layer 4 / cross-reference:** `research/HYP-121/` (how this lead was found), `data/research/hyp119/result.json` (MAGNITUDE_ONLY on liquidity cascades), `data/research/hyp120/result.json`, `research/EDGE_LEDGER.md`, `research/HYPOTHESIS_LESSONS.md`
