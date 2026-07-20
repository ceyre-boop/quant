# TICK-043 — Market data adapter layer

Status: PLAN — awaiting approval. Not implemented.

## Step 1 audit (DONE, read-only)

Seven vendors in tree. Nothing is abstracted behind one interface; there are
already **three competing part-adapters**, which is the real finding.

| Vendor | .py files | Where | How |
|---|---|---|---|
| yfinance | 196 | `data/providers.py`, `universe_sweep.py`, `research/**`, most backtests | direct `import yfinance` everywhere |
| alpaca | 73 | `sovereign/data/feeds/alpaca_feed.py`, `data/alpaca_client.py`, `train_core.py`, `execute_daily.py`, `walk_forward_validation.py` | two separate clients |
| polygon | 33 | `data/polygon_client.py` (REST + WS), gapper research | direct REST wrapper |
| thetadata | 18 | VRP / options-tradeability audit | direct terminal HTTP :25503 |
| databento | 11 | ES/NQ sandbox | direct |
| fredapi | 10 | `sovereign/data/feeds/macro_feed.py`, briefing | direct |
| alpha_vantage | 3 | `sovereign/sentiment/` news | direct |

Existing partial abstractions (must be consolidated, not duplicated):
- `data/providers.py` — `DataProvider`, yfinance primary + polygon fallback, has `SYMBOL_MAP`. **Closest thing to the requested adapter already.**
- `sovereign/data/feeds/alpaca_feed.py` — `AlpacaFeed.get_bars()` **already has a parquet cache** (`_cache_path`/`_load_cache`/`_save_cache`) + `_validate_raw_bars`. Step 3's cache substantially exists here.
- `data/alpaca_client.py` — `AlpacaDataClient`, third overlapping bar-fetcher.

Config already has the env switches (`config.py:60-76`): `USE_POLYGON` / `USE_ALPACA` / `USE_YFINANCE` — no `DATA_PRIMARY`/`DATA_FALLBACK` yet.

## Step 2-3 plan (additive, no execution path)

- `sovereign/data/adapter.py` — `MarketDataAdapter` with `get_bars` / `get_snapshot` /
  `get_top_movers` / `get_options_chain`; `DATA_PRIMARY` + `DATA_FALLBACK` from env;
  primary failure logged then fallback, transparently. Backends wrap the *existing*
  clients (`AlpacaFeed`, `PolygonRestClient`, yfinance) — no new vendor code.
- `sovereign/data/cache.py` — `DataCache.get_or_fetch`, `data/cache/{symbol}/{date}.parquet`,
  historical immutable / today invalidates after close, hit-vs-API counters logged.
  Lift the proven logic out of `alpaca_feed.py` rather than writing a second cache.
- Isolation: `sovereign/` only — no `ict/` import, keeps NN#1 clean.
- Tests: `tests/unit/test_data_adapter.py` (fallback fires on primary raise, schema
  columns, cache hit path, today-vs-historical invalidation).

## Step 4 — BLOCKED, needs your call

`# TODO: migrate to MarketDataAdapter` comment-flagging everywhere is safe.
The three "most critical" callers named are **not**:

- gapper scanner → `research/gapper/*` — OK, research-only. Safe to migrate.
- backtester + execution harness (`execution/harness.py`, `scripts/forex_live_scan.py`)
  → these are the **shadow/execution-path freeze** targets. CLAUDE.md standing
  constraint: nothing importable by the live/backtest path changes without an
  explicit unlock recorded in `NEXT.md`. `grep unlock NEXT.md` → no unlock present.

Swapping the data source under the backtester also silently re-baselines v015
(0.6886 reconcile). That is exactly the failure mode TICK-024 documented.

**Proposed:** ship Steps 1-3 + 5 + the TODO flags + gapper migration. Leave
backtester/harness on TODO until you record an unlock, and migrate them as a
separate ticket with a reconcile check (v015 must still print 0.6886 post-swap).

## Step 5

`~/Obsidian/Obsidian/System/data_sources.md` — vendor roles, rate limits
(Alpha Vantage free 25/day and Polygon free 5/min are the live constraints),
key names, cache rules, un-migrated surface.
