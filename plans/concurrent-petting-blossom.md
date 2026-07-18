# Execution Harness — plan

**Ticket:** TICK-038 (proposed; max in `tickets/backlog.md` is TICK-036, TICK-037 exists only in `NEXT.md:1124`)
**Status:** PLAN — awaiting approval
**Date:** 2026-07-18

---

## Context

Two gapper edges are adjudicated but not tradeable, and both fail on the *same* unanswered question: what does a fill actually cost?

- **HYP-107** — `REAL_BUT_MARGINAL_EXECUTION_UNRESOLVED`. Holdout n=57, gross median +5.4%, p=0.0005. Its own doc: *"plausibly a small real edge, plausibly unviable after costs. Not tradeable on this evidence."*
- **HYP-093** — `VALID_BUT_BELOW_FLOOR`. p=0.031, DSR 0.987, 559 events — but constitutional yield **+0.00023%/day against a 0.0005 floor**, under half.

Neither is a confirmed edge. This harness is an **adjudication instrument**, not a pre-flight check before funding.

Three facts, established by probe during planning, shape the design:

1. **Nothing in the repo measures spread.** `hyp107_shadow.py:148` computes `(h-l)/entry` — the first-minute bar **range** — and logs it as `realized_spread_pct`. Off by an order of magnitude, biased pessimistic.
2. **Real spreads are ~10× better than assumed.** Live SIP quote for TGHL (the symbol that crashed the shadow) at 09:30:59 on 2026-07-16: **bid 1.37 / ask 1.38 = 0.73% spread**. The retraction doc assumed 1–15%.
3. **Backtest and shadow disagree about fills.** `backtester/realistic_fills.py` is what backtests use; *neither* shadow calls it. `vs_backtest_delta` today would compare two models, not model-vs-reality.

**Outcome:** one harness replacing two forked shadows, costing fills through the same module the backtests use, emitting measurements and no verdict.

---

## Decisions (operator-confirmed)

1. **Unify.** One engine, both legs. Deprecate `research/yield_frontier/live_shadow.py` (HYP-093) and `research/gapper/hyp107_shadow.py` (HYP-107). Historical logs preserved.
2. **Reuse `realistic_fills.py` and fix its LULD bug.** Makes `vs_backtest_delta` measure reality-vs-model. Re-baselines prior backtests.
3. **Measure only.** No readiness column, no funding flag, ever.

---

## Architecture change from the brief: deferred capture, not real-time

**The brief's premise does not hold on this subscription.** Probed empirically:

| Endpoint | Result |
|---|---|
| SIP historical quotes | **200** — full bid/ask/size/exchange |
| SIP quote at −16 min | **200** |
| SIP quote at −13 min | **403** `subscription does not permit querying recent SIP data` |
| SIP latest / snapshot | **403** |
| IEX realtime | 200 — but AAPL `bid 314.75 / ask 347.97` = 10% spread. **Unusable.** |

The recency boundary is **exactly 15 minutes**. A real-time harness is impossible; real-time IEX would produce confidently wrong spreads from a ~2%-volume venue.

**This costs nothing that matters.** A measurement instrument needs accuracy, not latency. The harness decides signals live off delayed bars (as both shadows already do), then captures the true quote at the decision timestamp on a **T+16min deferred pass**. Recorded fills are exact, not approximated.

**Consequence — the fastest result is a backfill, not the forward run.** All 57 HYP-107 holdout events have retrievable historical quotes *today*. `--backfill` answers the execution question at n=57 this weekend, against the same preregistered holdout, adding no multiplicity. The forward harness remains the honest out-of-sample test.

---

## Files

```
execution/config.py     frozen params + sha256 lock
execution/alpaca.py     ONE client (replaces 2 forked copies)
execution/quotes.py     NEW — real SIP bid/ask
backtester/luld.py      NEW — tiered LULD (see boundary note)
execution/halts.py      halt status; imports backtester/luld
execution/borrow.py     IB locate reader
execution/scan.py       screener (ported)
execution/harness.py    engine: --live | --backfill | --replay DATE
scripts/com.alta.execution_harness.plist   TRACKED-NOT-LOADED
```

### Import boundaries — verified, not assumed
- `execution/` **already imports** `sovereign/` (`funderpro_executor.py:282`). So `kill_switch`, `timestamps`, `_common` are fair game.
- `backtester/` imports **neither** `execution/` nor `sovereign/` — it is a clean leaf. **Do not make `backtester/` depend on `execution/`.** Tiered LULD goes in `backtester/luld.py`; `execution/halts.py` imports *it*.
- `execution/` also imports `integration.production_engine` and `meta_evaluator` (May-era stack). Do not extend that coupling; new modules import only stdlib, numpy, `sovereign.utils.*`, `sovereign.autonomous._common`, `backtester.luld`.

### `execution/config.py`
Frozen values **byte-identical** to `hyp107_shadow.py:38-39`: `og_max=0.577`, `logvol_max=5.854`. Plus HYP-093 `gain_floor=1.00`, screener params, entry/exit times.
`verify_frozen_hash()` runs **before any network I/O**; mismatch is a hard startup failure. Threshold drift becomes impossible without a `param_change_log.jsonl` entry.

### `execution/alpaca.py`
Consolidates `keys()/get()/bars_for()/et_t()` duplicated across both shadows.

**Fix the error swallow.** `live_shadow.py:52-71` does `except Exception: sleep(5)` then `raise RuntimeError(url[:110])` — destroying the real error. This is why the crash was misdiagnosed. Replacement: 429→sleep 10; **403→raise immediately** (entitlement signal, must not be absorbed); 5xx/timeout→backoff 2/4/8/16; anything else→re-raise. On exhaustion `raise AlpacaError(url, status, body[:500], attempts) from last_exc`.

### `execution/quotes.py` — the core new capability
`Quote` dataclass (bid/ask/sizes/exchanges/conditions/tape), `quote_at(symbol, ts_utc)` for deferred + backfill capture, `latest_quotes()` reserved for a future entitled plan.

**Fill convention, applied everywhere:** LONG entry at **ask**, exit at **bid**. SHORT entry at **bid**, exit at **ask**. `spread_cost` = entry half-spread + exit half-spread as fraction of entry mid. `gross_return` mid-to-mid; `net_return` fill-to-fill; **`gross − net ≈ spread_cost` asserted to 1bp in tests.**

Gates: crossed/locked → last clean quote within 10s else `SKIP_NO_QUOTE`; `spread_pct > 0.25` → fill but flag `wide_quote` (**do not drop — dropping biases upward**); no quote at timestamp → `SKIP_NO_QUOTE`. Raw quote persisted so runs are re-derivable offline.

### `backtester/luld.py` — the bug fix
`realistic_fills.py:39-41` applies a **flat 10%** band:
```python
jump = np.abs(o / prev_c - 1) > LULD_BAND      # 0.10
intrabar = np.abs(c / o - 1) > LULD_BAND
```
On a stock that gapped 100%, this flags nearly every opening minute as a halt and applies `HALT_RESUME_SLIP=0.02` spuriously. **Backtests were penalised ~2%/trade.** Only `gap_before` (missing RTH minute) is genuine.

Correct tiered rule (Reg NMS LULD): ≥$3.00 → 10% (Tier 2; 5% Tier 1); $0.75–$3.00 → 20%; <$0.75 → lesser of 75% or $0.15/px. **Doubled 09:30–09:45 and 15:35–16:00.** Every gapper here is Tier 2 at the open → **20% band, not 10%**.

`realistic_fills._halt_flags` delegates to `backtester.luld.halt_flags`. Keep `LULD_BAND` defined-but-deprecated so external importers don't break.

**Validation ground truth exists but is thin: 5 days** (`data/research/yield_frontier/halt_snapshots/nasdaq_2026-07-{14,15,16,17,18}.csv.gz`). Report precision/recall old-vs-new **with N=5 stated plainly** — do not present a confident number off five sessions.

### `execution/borrow.py`
Reads `data/research/gapper/ib_locate_YYYY-MM-DD.json` (via `scripts/ib_shortable_snapshot.py`, `com.alta.ib_shortable.plist` daily 07:00 ET).

**Only one snapshot exists (2026-07-16).** Policy: no snapshot for the day → `SKIP_NO_BORROW / no_locate_snapshot`. **Never fall back to a stale file.** `HARD` also skips (`tier_HARD`) — a locate you might not get would flatter the short leg.

⚠️ **Second borrow path is broken.** `daily_snapshots.py:30-33`: *"IBKR FTP unreachable from this network 2026-07-13 (ftp3 and ftp2+TLS both timed out)"*; `borrow_snapshots/` is empty. The working path is `ib_shortable_snapshot.py`. Do not wire the dead one.

### `execution/harness.py`
Startup order is load-bearing:
```
load_env() → verify_frozen_hash() → heartbeat → skip_if_frozen() → market-open check → run
```
Kill switch is gate #1 **even though paper-only** (precedent `es_nq_paper_runner.py:54`), and comes **after** the heartbeat so a freeze reads as FROZEN, not DOWN. Frozen → `sys.exit(0)`.

| ET | action |
|---|---|
| 09:25 | scan universe, snapshot prev closes |
| 09:30:59 | HYP-107 filter → record LONG intent |
| 10:29:59 | HYP-093 filter → borrow → record SHORT intent |
| 10:47 | **deferred capture**: true quotes for 09:30:59 entry + 10:30 exit |
| 15:45 | record SHORT exit intent |
| 16:02 | deferred capture for 10:29:59 + 15:45; write summary; exit 0 |

Monotonic-clock scheduler re-deriving time-to-next-event on each wake (laptop sleep otherwise silently drifts a 6-hour session). Past-instant on wake → `SKIP_MISSED_WINDOW`, never a late fill.

Idempotent on `(date, symbol, signal_type)` — restart re-reads today's rows and skips recorded decisions.

`backtest_expected_return` calls **the same** `realistic_long_return(..., scenario="base")` the backtests call, same bars, same day. That identity is what makes the delta meaningful.

### Outputs
`data/execution/fill_log.jsonl` via `_common.append_jsonl()` (raises deliberately — let it). Timestamps via `canonical_timestamp()` (UTC). Required 10 fields exactly as briefed; appended: `reason`, `wide_quote`, `luld_band_used`, `scenario`, `frozen_hash`, `quote_raw`, `capture_lag_s`.
**`date` is the ET session date** — the one ET leak into persistence; document it in the module docstring.

`data/execution/daily_summary.csv`:
```
date,n_signals,n_filled,n_skipped,median_net_return,median_spread_cost,vs_backtest_delta
```
No readiness column. Zero-fill days write `""`, not `0.0`. Same-date rerun rewrites in place.

⚠️ **`data/execution/fills.jsonl` (Jun 30) is untouched** — different file. Comment in the header so nobody tidies it.

### Plist
09:25 ET Mon–Fri (launchd fires local time = `America/New_York`; **write ET directly, DST free**). Absolute `/opt/homebrew/bin/python3`, `WorkingDirectory` = repo root, logs to `logs/harness.log|.err`.
**TRACKED-NOT-LOADED** — author, `plutil -lint`, commit. Operator installs and runs `plist_watchdog.py --rebaseline`. Skipping the rebaseline leaves a standing RED resembling an intrusion (`NEXT.md:836`).

---

## Tests

Copy the pre-registration lock pattern from `tests/unit/test_es_nq_isolation.py:40-47` — literal values, message *"revert the config, don't fix the test."*

- `test_execution_frozen_params.py` — `og_max==0.577`, `logvol_max==5.854` as literals; hash matches.
- `test_luld_bands.py` — table-driven: ($5,11:00)→0.10; ($5,09:35)→0.20; ($2,11:00)→0.20; ($2,09:35)→0.40; ($0.50,11:00)→0.30; tier1($5,11:00)→0.05; boundaries at $3.00/$0.75 and 09:45:00/15:35:00.
- `test_luld_regression.py` — a 12% opening minute is **not** a halt under the new rule and **was** under the old; a 3-minute tape gap **is** under both. The executable statement of the fix.
- `test_quotes.py` — side conventions, crossed/locked/stale, `gross − net ≈ spread_cost` to 1bp.
- `test_filters.py` — pure filters straddling each threshold.
- `test_alpaca_errors.py` — mocked 500/429/403/timeout; `AlpacaError` carries status/body/attempts/`__cause__`; 403 raises without retry-looping.
- `test_summary_csv.py` — exact header; empty medians `""`; rerun rewrites.
- `test_harness_replay.py` (integration) — full replay on a fixture day.

---

## Param change log (CLAUDE.md NON-NEGOTIABLE #4)

Mandatory entry in the same commit as the LULD fix. **State the direction:** the old bug made backtests look *worse* than reality, so re-baselining moves numbers **up**. Log it *before* generating new numbers — an upward correction is exactly where motivated reasoning enters.

```json
{"ts":"<canonical_timestamp()>","component":"backtester/realistic_fills.py::_halt_flags",
 "param":"LULD_BAND","old":"flat 0.10","new":"tiered 5/10/20%, doubled 09:30-09:45 & 15:35-16:00",
 "rationale":"Flat 10% flagged ~<X>% of microcap opening minutes as halts, applying HALT_RESUME_SLIP=0.02 spuriously. Validated vs archived nasdaqtrader halts: N=5 sessions only.",
 "affects":["all realistic_fills-based backtests prior to <commit>"],
 "action_required":"re-run affected backtests; prior net figures are stale and biased LOW"}
```

---

## Migration

**Phase 1 (this commit)** — harness ships; both shadows keep running. `hyp107_shadow.py` imports thresholds from `execution.config` (values unchanged, single-sourced) and gains a `DEPRECATED:` banner.
**Phase 2 (≥10 overlapping sessions)** — symbol-set agreement must be **100%**; return differences are the finding. Any symbol disagreement is a porting bug and blocks Phase 3.
**Phase 3** — unload shadow plists, move scripts to `attic/`, **leave all historical logs in place** (they are the control group).

---

## Verification (Saturday — no market hours)

1. `python -c "from execution.config import verify_frozen_hash; verify_frozen_hash()"` → silent.
2. `pytest tests/unit/test_execution_*.py tests/unit/test_luld*.py tests/unit/test_quotes.py tests/unit/test_filters.py tests/unit/test_alpaca_errors.py tests/unit/test_summary_csv.py -q`
3. `python -m execution.halts --validate --since 2026-07-14` → precision/recall old-vs-new, **N=5 stated**.
4. **`python -m execution.harness --replay 2026-07-16 --out /tmp/exec_replay`** — full path incl. quotes, halts, borrow, CSV. Acceptance gate. `data/execution/` untouched.
5. Diff replay symbol selection vs `hyp107_shadow` archived log for 2026-07-16 — **must match exactly**.
6. **`python -m execution.harness --backfill --events research/gapper/<holdout list>`** — the 57 HYP-107 events with real quotes. The result that actually resolves the open verdict.
7. `plutil -lint scripts/com.alta.execution_harness.plist`; `launchctl list | grep execution_harness` → **empty**.
8. `git status` → `data/execution/fills.jsonl` unmodified.
9. Freeze → run → assert exit 0, heartbeat written, zero fills → thaw.
10. `pytest tests/ -k "not sklearn" -q` → no new failures vs the 40 known.

Steps 4 and 6 are the ones that de-risk this.

---

## Risks

- **Sample size.** Medians over <20 fills carry no weight. No readiness column exists precisely so none is read in.
- **Short leg under-samples.** One locate snapshot → mostly `SKIP_NO_BORROW` until history accrues. The two legs are not equally sampled; the summary must not be read as if they were.
- **Halt validation is N=5.** State it; don't present confident precision off five days.
- **Reference price approximated** as 5×1-min bar closes vs true 5-min print average. Fine for detection, imprecise at band edges. Documented, not papered over.
- **Backfill uses consumed holdout.** Re-measuring *execution cost* on the same 57 events is not a new signal test and adds no multiplicity — but it must be labelled execution-measurement, never a fresh verdict on the edge.
- **Deferred capture assumes quote history is stable** at T+16min. Verified for 2026-07-16; spot-check one more day during implementation.

---

## Out of scope

No new hypothesis research. No live orders. No funding-readiness verdict. No changes to `ict/`. No `launchctl load`.
