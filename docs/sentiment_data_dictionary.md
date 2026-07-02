# Sentiment Board — Data Dictionary

Every column of `sentiment_board_state` (DuckDB `data/sentiment.db`, one row per
(trading-day, pair)): source, provenance/release timing, look-ahead rule, units, coverage.
The board spine is VIX trading days × the 4 live pairs (EURUSD, GBPUSD, USDJPY, AUDUSD).
AUDNZD positioning lives in the SOURCE tables (constitutional carry-complex scope,
HYP-072/073/077) — it is not a board pair.

Look-ahead discipline: every value on row `date` was PUBLIC on/before that date's close.
Enforced three ways: structural keying in each feeder, `pytest tests/test_sentiment_board.py`,
and `scripts/audit_look_ahead.py` (runs at the end of every `update_sentiment` pass; DB-level
`COUNT(*)==0` checks + an empirical board-leak check). Current audit: **0 violations**.

| Column | Source → table | Release/provenance rule | Units / range |
|---|---|---|---|
| `news_score` | Alpha Vantage NEWS_SENTIMENT → `sentiment_news_daily` | article publish time; plain same-date join | directional score [−1, 1] |
| `gdelt_tone` / `gdelt_tone_5d` / `gdelt_volume` | GDELT DOC API → `sentiment_gdelt_daily` | GDELT day bucket; same-date join. NULL until the off-peak backfill fills history | tone z-ish score; article count |
| `econ_surprise_z` | ALFRED first prints → `sentiment_surprise_release/_daily` | **publish-date keyed** (`realtime_start`); release-innovation, USD-signed (UNRATE inverted); test `publish>ref` 0/750 | z-score, daily EWMA |
| `cot_net_pct` | CFTC legacy COT (6dca-aqww) → `sentiment_cot_weekly` | measured Tue, dated to **Fri publish** (+3d); `release_ts` = Fri 15:30 ET; ASOF join on publish_date | trailing-3y percentile [0,1] |
| `cot_net_oi` | 〃 | 〃 | net_spec / OI, unbounded ≈ [−1,1] |
| `cot_net_pct_1y` | 〃 (positioning layer, 2026-07-02) | 〃 | trailing-52w percentile [0,1] |
| `cot_net_z` | 〃 | 〃 | trailing-52w z of net_spec |
| `cot_flush_1w` | 〃 | 〃 | WoW Δnet / trailing-1y Δstd (≈z) |
| `tff_lev_net_pct` | CFTC TFF futures-only (gpe5-46if) → `sentiment_cot_tff_weekly` | 〃 (identical keying); leveraged-funds category; history 2006-06→ | trailing-3y percentile [0,1] |
| `vrp_signal` / `vrp_pct` / `vrp_iv_atm` / `vrp_rv_trailing` | ThetaData FX-ETF chains + yfinance spot → `sentiment_vrp_daily` | `iv_obs_date == rv_last_date == date` (EOD close); weekly Fri obs, ASOF-filled | annualized vol units; pct [0,1] |
| `rr25` | ThetaData FX-ETF chains → `sentiment_options_surface` | `iv_obs_date == date`; weekly Fri obs, ASOF-filled; **FIXTURE-stamped rows excluded from fusion** | iv(25Δc) − iv(25Δp), vol pts |
| `bf25` | 〃 | 〃 | 0.5(iv25c+iv25p) − atm_iv_1m, vol pts |
| `atm_term_slope` | 〃 | 〃 | atm_iv_1m − atm_iv_3m (>0 = inverted) |
| `macro_curve` / `macro_spread` / `macro_inflation` | FRED T10Y2Y / BAMLH0A0HYM2 / T10YIE → `sentiment_macro_daily` | FRED daily series, ASOF join (publish-lag safe) | %; HY OAS only 2023-06-30→ |
| `vix_level` / `vix_momentum` / `vix_regime` | yfinance ^VIX → `sentiment_vix_daily` | same-day close (the date spine itself) | level; 5d Δ; LOW/NORMAL/HIGH/SPIKE |

## Futures ↔ pair mapping (positioning layer)

| Pair | CFTC code | CME future | Note |
|---|---|---|---|
| EURUSD | 099741 | 6E (EUR/USD) | |
| GBPUSD | 096742 | 6B (GBP/USD) | |
| USDJPY | 097741 | 6J (JPY/USD) | **INVERTED** vs the pair: 6J-long = USDJPY-down. The feeder stays in currency space; direction rules invert at TEST time (HYP-072 prereg). |
| AUDUSD | 232741 | 6A (AUD/USD) | |
| AUDNZD | 232741 − 112741 | 6A − 6N legs | Cross = leg difference: `net_spec = AUD_net − NZD_net`, `net_oi = AUD_net_oi − NZD_net_oi`; `open_interest` stored 0 (meaningless for a difference). Source tables only, not a board pair. |

## Known, documented biases (accepted; not violations)

1. **COT holiday weeks (~3%/56 of 1,923 publish-Fridays):** the real CFTC release slips to
   Monday when Friday is a holiday; the fixed +3d rule dates those rows ≤1 business day early.
   Counted every audit run (`holiday_week_publish_bias` telemetry). Any hypothesis on COT extremes
   must carry this in its limitations (the prereg protocol references this note).
2. **econ_surprise_z is a release-innovation, not a consensus surprise** (no free consensus
   history exists) — labeled honestly, weaker proxy (ledger SENTIMENT-ECON-SURPRISE).
3. **FIXTURE rows** (`iv_source LIKE 'FIXTURE%'` in `sentiment_options_surface`): synthetic
   chains priced from a known smile for tests/wiring — excluded from board fusion by the
   rebuild SQL and from any hypothesis input by protocol.

## Coverage snapshot (2026-07-02 rebuild)

Board 11,560 rows (2,890 VIX days × 4 pairs). COT legacy features 100% (8,163 source rows,
1986→); TFF 100% of post-2006 spine (5,227 source rows); rr25/bf25/term_slope **0% — real
ThetaData backfill pending** (terminal not running; entitlement for FXE/FXB/FXY/FXA chains
unverified); vrp_* NULL for the same reason; gdelt NULL pending off-peak backfill; news recent
window only.
