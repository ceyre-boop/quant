# backtester/ — production bias-free backtest engine

Universal, bias-free, parallel. Built 2026-07-17 to kill the biases in
`research/gapper/BACKTEST_BIAS_AUDIT.md`. Not gapper-specific — feed it any
`events_df` + minute bars.

## Modules
| module | does |
|---|---|
| `data.py` | `get_minute_bars/get_daily_bars`; Alpaca-primary (yfinance 1m only serves 7 days), gz+parquet cache, never refetch |
| `engine.py` | `run(events_df, cfg, data_cache)` — bias-free fills (see below) |
| `audit.py` | auto after every run: stop-fill transparency, locate skip %, look-ahead, per-regime Sharpe → `data/backtester_audit_log.jsonl` |
| `mc.py` | `run_mc(daily_pnl, n_paths, challenge_cfg)` — 5-day block bootstrap, fork-parallel, ~2.5M paths/s |
| `scanner.py` | `scan(events_df, param_grid, data_cache)` — memoised fills + vectorised sizing sweep + permutation FWER (Bonferroni + Holm) |

## Fill model (the point)
- Entry: first bar ≥ `entry_time`, fill at bar OPEN (requires a prior bar — else refused, no look-ahead).
- Stop: first bar breaching trigger; fill at bar OPEN if it gapped through, else at trigger. **Never blindly at trigger on a gap-through.**
- Exit: `exit_time` close (or last bar).
- Slippage: entry-bar (high−low)/price·0.5 one-way (the whole-day range in the original mandate is pathological for parabolic gappers — flagged in results).
- Locate: `locate_required` gates against `data/research/gapper/ib_locate_*.json`; no snapshot for a date → `UNKNOWN`, take-and-tag (configurable).

## Run
```
PYTHONPATH=~/quant python3 -m backtester.run_corrected_gapper   # HYP-093 + EV grid, honest
PYTHONPATH=~/quant pytest -q tests/test_backtester.py
```

## Headline correction
Biased harness: +24.4% / Sharpe 3.4 / ~0 bust. Bias-free (1-min bars, realistic
spread, block bootstrap): **~+10–18% / Sharpe ~2 / ~10–13% prop bust.** The
stop-fill bias was small (~5/79 stops gap through); the real biases were omitted
transaction cost and IID Monte Carlo. See `research/gapper/HYP093_corrected_results.md`.
