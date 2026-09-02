# HYP-111 — post-shock intraday retrace-then-continuation

## Context

HYP-109 confirmed post-shock **magnitude** (next-week RV 1.36×, p≈0) and killed post-shock
**direction** at a 5-session horizon on daily bars. HYP-110 killed the overnight partition.
This tests whether a *path* — retrace against the shock, then continuation with it — exists
inside session t+1, at 1-minute resolution, which daily bars cannot see. Different
resolution, not a re-run. If null, the post-shock thread is closed at every resolution and
that is the finding.

Every definition below is frozen in this file **before any minute bar is read**. Nothing
moves after the hash.

## 0. Data gate — first use of ThetaData, and it is not a given

**State found (read-only, 2026-09-02):**
- ThetaTerminal v3 (pid 9252, up 17 days) is **hung**: listening on 25510/25520/10000/11000
  — *not* 25503, which every repo client uses — and every HTTP request times out with 0
  bytes. No log line since 2026-08-07. The `ThetaData` MCP SSE server failed to connect for
  the same reason.
- Its login bundle is **`STOCK.FREE, OPTION.VALUE, INDEX.FREE`**. The repo's own probe
  (`scripts/thetadata_probe.py:7`) recorded that on this tier "even stock history returns
  403". No `/v3/stock` call exists anywhere in the repo. The user's premise — 1-min stock
  bars 2020–2026 — is **unverified** and the probe is the step that verifies it.
- Fallbacks do not reach 2020: Alpaca SIP minute cache walls at 2024-01-03 (free = 2 yr;
  only 5 of the ten ETFs cached), Polygon free = 2 yr. There is no silent fallback.

**Step 0a — restart the terminal.** `kill 9252`, relaunch
`cd ~/ThetaTerminal && java -jar ThetaTerminalv3.jar --creds-file creds.txt` (nohup, log to
`theta-live.log`), wait for `CONNECTED` in the log, confirm 25503 answers
`/v3/option/list/expirations?symbol=SPY` (the known-good endpoint). The HYP-098 handoff's
"do not restart it" was scope guidance for that session, not a standing rule; it was
restarted 2026-08-07 already.

**Step 0b — `scripts/research/probe_thetadata_stock_1m.py`** (new; the `--probe` the user
asked for, following `research/petrules_audit/probe_sources.py::probe_thetadata` idiom).
Read-only against the terminal, never prints credentials. Calls
`GET /v3/stock/history/ohlc?symbol=SPY&interval=1m&start_date=…&end_date=…` for five
dates — 2020-03-16, 2021-06-15, 2022-06-13, 2024-01-03, 2026-08-03 — then one date
(2022-06-13) for all ten ETFs. Records per call: HTTP status (200 / 403 entitlement / 472
no-data), row count, first/last bar time, columns. Writes
`data/research/hyp111/probe.json`. **These probe dates are declared and excluded from the
event sample.**

**Decision rule, sealed now:**
- 200 with ≥370 RTH bars on all five SPY dates and all ten symbols → ThetaData is the
  source; proceed to seal.
- 403 on the 2020–2023 dates but 200 on 2024+ → entitlement depth wall. **STOP.** Report
  the exact tier needed (Stock STANDARD or above) with price from the ThetaData pricing
  page; upgrading is spending money and is Colin's call. Do not substitute Alpaca/Polygon
  for a 2-year sample and call it the same hypothesis.
- 403 everywhere → same STOP, same report.

## 1. Power — stated before running

Counted from the daily cache with the frozen HYP-109 shock rule, 2020-01-01 → 2026-07-16
(read-only, no outcomes looked at):

- **1,798 instrument-events on 670 unique dates** (2.7 instruments per shock date — shocks
  cluster cross-sectionally; pooled-event inference would overstate n by ~2.7×).
- Per year (dates): 2020 **94** · 2021 79 · 2022 **143** · 2023 64 · 2024 111 · 2025 100 ·
  2026 (to Jul) 79. 2020 is 22% of dates but will be a larger share of *triggered, wide*
  trades — the ex-2020 component is essential, not decorative.
- Trigger rate unknown a priori; assume 30–50% → ~550–900 trades on ~250–350 independent
  dates (ex-2020 ~200–280).
- Per-trade sd for a 1R bracket sized at 0.382×shock-day range ≈ 0.6–0.9% of notional.
  With ~300 independent dates, SE(mean) ≈ 0.04%/trade → **80% power at ≈0.11%/trade**.
  The constitutional floor (0.05%/day) is *below* detectability: a real edge at the floor
  reads NULL/INCONCLUSIVE here. That is a limitation of a 6.5-year sample and is accepted;
  it will be stated in the verdict. A CONFIRMED needs ≈0.1%+ per trade net, i.e. a strong
  effect.
- The sample **starts at a vol-regime break**: the first 94 dates are the largest vol
  event in the window. Any aggregate result that dies ex-2020 is a 2020 result.

## 2. Frozen definitions

**Instruments.** SPY QQQ IWM DIA TLT GLD EFA EEM XLF XLE (golden rule: ≥7/10).

**Shock (identical to HYP-109).** Daily close-to-close log return from
`data/cache/daily_universe/<SYM>.parquet`; shock at t iff |r_t| ≥ p90 of trailing 252
|r| (t excluded, per instrument). Direction **s = sign(r_t)**. Events with t+1 in
2020-01-02 … 2026-07-16 (daily window) and t ≥ 2019-12-31.

**Levels — from the shock day's daily bar only, fixed before session t+1 opens.**
- `range_t = high_t − low_t`, `C = close_t`
- **retrace level** `L = C − s·0.382·range_t`
- **target** `T = C + s·0.382·range_t` (symmetric ⇒ a 1R bracket; null expectation is
  exactly −cost)
- **stop** `= L`

**Path / entry, session t+1, 1-min RTH bars 09:30–16:00 ET.**
1. *Retrace*: first bar τ1 whose extreme against s reaches L (up-shock: `low ≤ L`;
   down-shock: `high ≥ L`). An open already beyond L satisfies this at 09:30.
2. *Reclaim*: first bar τ2 > τ1 with `close` back on the shock side of C (up-shock:
   `close > C`). τ2 must be ≤ 14:30 ET or no trade.
3. *Entry*: open of bar τ2+1, direction s. One trade per instrument-event.
4. *Exit*, first of: stop bar (`low ≤ L`, filled at L); target bar (`high ≥ T`, filled at
   T); both in one bar → **stop** (conservative, frozen); else **time exit at the 15:55
   bar close**.
5. No entry ⇒ return 0 for that instrument-event (flat is a real outcome of the strategy).

**Return units.** `s·(exit − entry)/entry` in % of notional; also R = that ÷ `(C−L)/C`.
Verdict statistics in % of notional; R reported.

**Costs.** **3.0 bp round trip all-in** (2 bp spread + 1 bp stop/market slip), charged on
every executed trade, both series. Break-even cost reported, never a pass condition.

**Incumbent — named and justified.** *Naive post-shock continuation*: on every
instrument-event, long s from the 09:30 bar open to the 15:55 bar close, same costs. This
is what the momentum trader does *without* the path; the delta isolates the value of the
retrace-then-reclaim structure specifically. Buy-and-hold is not the comparator — a
day-trader is flat overnight. Cash (0) is printed as the absolute reference alongside.

**Delta series.** For each event date d: `Δ_d = mean over instruments shocked on d of
(structure_net − naive_net)`. One number per date → cross-sectional averaging is the
first line of defence against same-date correlation. Annualised with
`sqrt(event_dates_per_year)`; DSR `n_obs` = number of event dates.

## 3. Confluence — enumerated and frozen now, five conditions, all knowable at τ2

| # | condition (binary) | source |
|---|---|---|
| C1 | shock-day volume ≥ 1.5× median of the trailing 20 sessions' volume | daily |
| C2 | session t+1 open on the shock side of C (gap agrees) | 09:30 bar |
| C3 | at τ2 the close is on the shock side of session t+1's running VWAP (cumulative, minute bars) | 1-min |
| C4 | market proxy agrees: proxy's 09:30→τ2 return has sign s (proxy = SPY; for SPY itself, QQQ) | 1-min |
| C5 | strong close: `close_t` in the shock-side 25% of `range_t` | daily |

Count 0–5, buckets **≤1 / 2 / ≥3**. **Monotonicity claim, pass iff both:**
(i) `mean(≥3) ≥ mean(2) ≥ mean(≤1)` on net structure return, executed trades; and
(ii) OLS slope of net return on count (0..5) has a date-block-bootstrap 95% CI that excludes
zero from above. Anything else ⇒ **"confluence is a story"** — sealed as its own verdict
line, no weight on the primary. Per-condition means printed, descriptive only. Any bucket
with < 30 trades ⇒ confluence INCONCLUSIVE.

## 4. Statistics and verdict components

- **Bootstrap — new, fixes the HYP-110 flaw.** Stationary block bootstrap over the ordered
  **event dates** (L = 5, 10,000 draws, seed 42); every instrument on a resampled date is
  carried together. Skeleton: `research/modern/_lib.py::_stationary_block_indices`,
  generalised from paired arrays to a date→rows grouping. New module
  `research/hyp111/date_bootstrap.py`, with a unit test proving same-date rows always
  co-resample.
- **CPCV** `sovereign/discovery/cpcv.py::combinatorial_purged_splits` over event dates,
  6/2 → 15 folds, embargo 1 date.
- **DSR** `sovereign/discovery/gate.py::deflated_sharpe_ratio` on Sharpe(Δ), **n_trials =
  1546** (`mined_n._total` 1543 + HYP-109 + regime test + HYP-110). The confluence slope
  is a declared second claim; it is judged by its CI and its multiplicity (3 buckets, 1
  slope) is recorded, not DSR'd.

| component | statistic | pass |
|---|---|---|
| (p) path frequency | share of events that trigger; per instrument | descriptive; abort if pooled triggered n < 100 |
| (b1) delta | Sharpe(Δ), date-block CI | CI excludes 0 from above |
| (b2) delta | DSR prob @1546 | ≥ 0.95 |
| (c) folds | Sharpe(Δ) on 15 purged folds | ≥12/15 > 0 (null ≤7, inconclusive 8–11) |
| (g) golden rule | per-instrument Sharpe(structure − naive) | ≥7/10 > 0 (null ≤5, inconclusive 6) |
| (x) ex-2020 | Sharpe(Δ) with 2020 dates removed | > 0 |
| (d) economics | mean structure_net per event-day vs 0.0005 | reported; gates CONFIRMED vs BELOW_FLOOR |
| (m) confluence | §3 | separate line: MONOTONIC / STORY / INCONCLUSIVE |

**Ladder.** Data abort (any instrument with < 80% of event sessions having ≥ 370 RTH
bars, or triggered n < 100, or fold error) ⇒ **INCONCLUSIVE**. Else **NULL** if b1 fails
or c ≤ 7 or g ≤ 5. Else if b1 ∧ b2 ∧ c ≥ 12 ∧ g ≥ 7 ∧ x: **CONFIRMED** if d ≥ floor,
**VALID_BUT_BELOW_FLOOR** otherwise. Else **INCONCLUSIVE**. One run. No re-run. Reported
as what it is. If NULL: *post-shock closed at daily and intraday resolution* — written as
a finding in the taxonomy doc.

**Prior expectation (Colin, 2026-09-02, before any minute bar):** **CONFIRMED** —
retrace-then-reclaim after a shock reliably continues; beats naive continuation on ≥7/10
ETFs and ex-2020.
**Most likely failure mode (Colin):** **data gate fails** — the STOCK.FREE tier 403s on
2020–2023 1-min bars and the hypothesis is untestable without a paid tier.
**Claude's prior, recorded alongside (both priors sealed, HYP-098 precedent):**
NOT_SIGNIFICANT; most likely failure is (x)/(g) — aggregate carried by 2020 and the
equity-index names, absent in TLT/GLD.

## 5. Files

New, all isolated under `research/hyp111/` (imports only `research.modern._lib`,
`sovereign.discovery.*`, stdlib/numpy/pandas — AST-guarded like
`research/yield_frontier/tests/test_gates.py`; nothing from the execution path):
- `research/hyp111/theta_stock.py` — `get_1m(sym, date)` → cached parquet
  `data/cache/theta_1m/{SYM}_{DATE}.parquet`, same frame contract as
  `backtester/data.py::_BAR_COLS` (`time` "HH:MM" ET string, ohlcv), RTH only; 403/472
  handled as in `research/political_alpha/_lib.py::ThetaClient` (472 = cached emptiness,
  403 = raise, no retry storm), 0.1 s pacing, chunked `--max-days` foreground fetch.
- `research/hyp111/engine.py` — pure functions: `levels(daily_row, s)`,
  `simulate(bars, s, C, L, T)` → trade record, `naive(bars, s)`, `confluence(...)`.
  Look-ahead canary test: no bar after τ2 influences the entry decision.
- `research/hyp111/date_bootstrap.py` — §4.
- `research/hyp111/tests/` — engine on synthetic bars (retrace/reclaim/stop-and-target-
  same-bar/time-exit), bootstrap co-resampling, AST import guard. Not collected by the
  main suite.
- `scripts/research/probe_thetadata_stock_1m.py` — §0b.
- `scripts/research/preregister_hyp111_postshock_intraday.py` — exact
  `preregister_hyp110_overnight.py` idiom; embeds §2–§4 verbatim plus the probe result
  (`data_source`, entitlement depth) and Colin's prior; `--write` / `--verify`; hash
  sha256 over doc minus `hash_lock`; ledger `PREREGISTERED`, backup first.
- `scripts/research/test_hyp111_postshock_intraday.py` — gate zero, `--gate-only`,
  `--fetch-only` (populates the cache in chunks, computes nothing), then the run;
  writes `data/research/hyp111/{result.json,VERDICT.md}`, ledger `ADJUDICATED` once.
- `research/HYP-111_SCOPE.md` — this spec, human-readable, committed before the probe.

**Order of operations (sealing discipline):** commit SCOPE → restart terminal → probe →
(STOP or) seal prereg with probe result embedded → `--verify` → `--fetch-only` (minute
bars enter the cache but nothing is computed; ~1,800 symbol-days ≈ 3–5 min at 0.1 s
pacing) → `--gate-only` → **wait for "run it"** → run once → `--verify` → VERDICT +
taxonomy update + NEXT.md → push.

## 6. Verification

```bash
.venv313/bin/python scripts/research/probe_thetadata_stock_1m.py          # data gate; writes probe.json
.venv313/bin/python -m pytest research/hyp111/tests -q                     # engine / bootstrap / import guard
.venv313/bin/python scripts/research/preregister_hyp111_postshock_intraday.py --write && ... --verify
.venv313/bin/python scripts/research/test_hyp111_postshock_intraday.py --fetch-only --max-days 400   # repeat to fill
.venv313/bin/python scripts/research/test_hyp111_postshock_intraday.py --gate-only
# on instruction only:
.venv313/bin/python scripts/research/test_hyp111_postshock_intraday.py
.venv313/bin/python scripts/research/preregister_hyp111_postshock_intraday.py --verify
python3 -m pytest tests/ -k test_pipeline_does_not_import_sovereign -q    # before push
```
