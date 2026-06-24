# Intraday FX Data Acquisition — runbook

The `intraday-fx` discovery track is **blocked on data**. yfinance (the system's free
source) cannot serve 15 years of intraday forex: 1m ≈ 7 days, 5m/15m ≈ 60 days, 1H ≈ 2.75
years, **4H not supported at all**. Genuine intraday/ICT/session discovery on FX therefore
requires a vendor feed. NQ already has 8.5yr of 1-min data on disk — use the `nq-intraday`
track for intraday discovery today; this runbook is the path to add FX.

## Options

| Vendor | Cost | Coverage | Notes |
|--------|------|----------|-------|
| **Dukascopy** | Free | Tick → resampled, ~2003+ | Best free intraday FX; download via `dukascopy-node` or their historical exporter. Some weekend/holiday gaps. |
| **Polygon.io** | Paid (~$30–200/mo) | Minute bars, FX + more | Clean API, well-documented; aggregates endpoint. |
| **OANDA v20** | Free w/ practice acct | Candles via API, several years | Same broker the live system trades; consistent with execution. Rate-limited for bulk history. |

## Steps to activate

1. **Acquire** 1m (and/or 5m/15m) bars for the pairs in `pair_universe.ALL_PAIRS`
   (EURUSD, GBPUSD, USDJPY, AUDUSD) over the longest window the vendor offers.
2. **Store** as partitioned parquet, e.g. `data/fx_intraday/<PAIR>_<TF>.parquet`, UTC index,
   columns `Open/High/Low/Close[/Volume]` — mirror the NQ parquet shape so `compute_features`
   works unchanged.
3. **Validate for drift/quality** (the system has documented yfinance drift — don't trust blind):
   - spot-check OHLC against OANDA candles on a few dates,
   - assert monotonic UTC index, no duplicate timestamps, gaps only on weekends/holidays,
   - confirm bar counts ≈ expected sessions.
4. **Implement** `IntradayFXAdapter` in `sovereign/discovery/data_adapter.py` like
   `ForexDailyAdapter`, but: load from the parquet, and provide `eval_signals` via a forex
   intraday backtest (reuse `fast_backtester.simulate_forex_trades_arrays` with an intraday
   `hold_days`/bar convention and an intraday cost model — spreads are the dominant cost intraday).
5. **Run** `python3 scripts/discover.py --track intraday-fx`. The gate, candidates, features, and
   visuals are already track-agnostic and will work once the adapter returns real data.

## Why it's gated, not skipped
The discovery engine, the methodology gate, candidate generation, and the visual suite are all
**track-agnostic** — they only need an adapter that returns OHLCV + a costed `eval_signals`. The
only missing piece is the FX intraday data and its costed evaluator. Everything else is built.
