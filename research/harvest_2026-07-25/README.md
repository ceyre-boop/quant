# Feature-Data Harvest — 2026-07-25 (real data, no-lookahead columns)

Harvested by `harvest.py` (re-runnable). All raw API responses / source files preserved.

| Dataset | File | Coverage | Caveats |
|---|---|---|---|
| Policy rates (FRED) | `policy_rates/policy_rates_daily.csv` | 2015→now, US/EU/UK/AU/JP | AU (IR3TIB01AUM156N) + JP (IRSTCI01JPM156N) are MONTHLY interbank proxies, ~1-2mo lag (JP shows 0.841 vs BoJ 1.00) — use for history shape; splice official targets for the live edge |
| Differentials | `policy_rates/differentials_daily.csv` | UK−US, EU−US, AU−US, US−JP | Latest: UK−US +0.10, EU−US −1.38, AU−US +0.83(proxy), US−JP +2.79(proxy). Matches the FOMC intel ground truth — REPLACES the broken feed values (which had UK−US = −0.60, wrong sign) |
| Indicator panel | `indicators/indicator_panel.csv` (+raw/) | 2015→now, 4 pairs + USDJPY + ES/NQ/SPY/QQQ/VIX | ATR14(+%), RSI14, realized vol 20d, 52w-high/low dist, ret 5/20/60d, zscore20, volume_ratio (where volume exists). Derived from yfinance daily bars (raw kept) |
| COT positioning | `cot/cot_fx_panel.csv` | 2020→2026-07-21 report, GBP/EUR/AUD/JPY/USD-idx/ES/NQ, n=2,936 | `published_date` = report_date+3d (Tue as-of, Fri publish) — USE published_date for availability. Latest: JPY net spec −152k (heavy short-yen carry crowding), GBP −56k, EUR −41k |
| Vol structure | `volstructure/vol_term_structure.csv` | 2015→now | VIX/VIX9D/VIX3M/VVIX/MOVE + ratios + backwardation flag; last row has partial NaNs (harvested pre-close) |

NOT harvested (session-limit casualties, still open): NewsAPI 30d headlines, GDELT tone,
EDGAR bulk Form4/13F/13D corpus, AV earnings spine extension, ALFRED vintages, FINRA short
interest history. All specified in workflow scripts wf_61b3884c / wf_6fe7fb11 — re-runnable.
