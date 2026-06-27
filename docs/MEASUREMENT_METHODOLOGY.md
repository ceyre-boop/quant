# Measurement Methodology — permanent rules

Rules that protect every measurement from known failure modes. Added to as failures are caught.

---

## RULE 1 — Gate-warmup / window-dependence (added 2026-06-27)

**Any v015 measurement on a window shorter than the gates' longest lookback is INVALID.**

v015's per-pair VIX regime gates (`PAIR_VIX_GATES`, `sovereign/forex/forex_backtester.py`: EURUSD 18 /
GBPUSD 18 / AUDUSD 20 / USDJPY 15) classify regime from a series that needs ~200 trading days of
warmup (SPY 200-day SMA + VIX history). A backtest started inside that warmup window runs with the
**gates effectively OFF** — it silently measures an *ungated* strategy.

**How this bit us:** `build_v015_dossier.py` ran `prove.py --start 2025-01-01`, a ~18-month window.
The gates never warmed up, so the "fresh 2025-26 Sharpe = −0.085 collapse" was measured on **ungated
v015** (USDJPY 15 trades, the bad VIX>15 ones included). Re-measured full-history (gates active,
USDJPY 6 trades), fresh 2025-26 = **+0.038** (flat), not −0.085. The "collapse to negative" was an
artifact; the real finding is a milder decay to flat. (Same `backtest_all` code: 15 trades
short-window vs 6 full-history — the difference is *only* the window.)

**The rule, operationally:**
- The reference standard for v015 behavior is **full-history, gate-active** measurement
  (`ForexBacktester(start="2015-01-01", end=...).backtest_all()`), which reconciles OOS 2023-24 ≈ 1.25.
- **Short-window measurements are NEVER canonical.** To measure any sub-period (e.g. a fresh slice),
  run the FULL history and *extract* the sub-period from it — never start the backtest at the
  sub-period boundary.
- Sanity check before trusting any window's number: confirm OOS 2023-24 reproduces ~1.25 from the
  same ledger. If it doesn't, the ledger is mis-configured — stop.

**Corollary — `run_pair_with_trades` vs `backtest_all`:** generate the canonical v015 ledger via
`backtest_all` (writes `logs/forex_backtest_trades.json`, the file `prove.py` reads). `run_pair_with_trades`
applies per-pair overrides differently and is not the canonical headline path.
