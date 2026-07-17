# Megascan v2 — exhaustive intraday/multi-asset search (2026-07-17)

**Result: nothing clears the candidate bar (raw Sharpe >1.5, n≥30/yr, positive).
Zero pre-registrations. The gapper fade remains the only real edge in the repo.**

## Scale & compute
- **7,220 distinct strategy configs** across 5 families, 159s dense compute
  (after ~13 min of Alpaca/yfinance data fetch: 11,854 minute day-files + 2yr
  hourly crypto + leveraged/biotech daily). A first parallel attempt deadlocked
  on macOS fork+numpy — re-run serially, which itself is the requested real
  compute-time floor.
- Bias-free fills (gap-through stop = breach-bar open, entry-bar spread).
- Reported RAW (uncorrected) and FWER (Bonferroni × 7,220) separately.

## Per-family verdict (dirty window, best config each)
| Family | configs | best Sharpe | best annual | read |
|---|---|---|---|---|
| A gap reversal (minute, 9 hi-vol gappers) | 1,104 | 1.09 | +2.4% | weak; n=17. 10–50% gaps don't fade like the ≥100% microcaps (confirms HYP-101) |
| B opening-range breakout (minute, 14 liquid) | 5,040 | **0.21** | +0.7% | **DEAD** — ORB has no edge net of cost on liquid ETFs/megacaps; most configs negative |
| C leveraged-ETF decay (daily) | 784 | 0.98* | +8.2% | weakly real but tiny return / high DD (SOXL/ARKK/UVXY mean reversion) |
| D crypto intraday (hourly BTC/ETH) | 132 | 0.85 | +7.2% | ETH long — basically "ETH rose over 2yr", a bull-bias artifact, not an intraday edge |
| E biotech binary (daily, 43 XBI) | 160 | 0.84 | +1.4% | nothing; tiny n |

\* The only config with Sharpe>1.5 was **BOIL dip-buy, n=11 over a decade**
(~1 trade/yr) — filtered out by the n≥30 rule as noise.

## RAW top-5 candidates + why they won't generalize
1. **ARKK oversold-dip long** (Sharpe 0.98, +1.8%, n=44): buying ARKK after a
   3-day drop. Return is negligible and the "edge" is beta — ARKK is a
   high-beta basket in a rising tape. No standalone alpha.
2. **SOXL dip long** (Sharpe 0.92, +8.2%, n=331, DD 18%): leveraged-semis decay
   reversion. Real-ish but the 18% drawdown eats the 8% return; Calmar <0.5.
3. **ETH-USD long** (Sharpe 0.85, +7.2%, n=1374): a directional long bias over a
   2-year window where ETH rose. Regime-dependent, not a signal.
4. **UVXY dip long** (Sharpe 0.73, +5.9%, n=78): catching vol-ETF bounces —
   crowded, squeeze-prone, and the decay makes long-side timing brutal live.
5. **RGTI gap-short** (Sharpe 1.09, +2.4%, n=17): quantum-meme gap fade, 17
   events. Under-powered; a coin-flip dressed as a Sharpe.

Every one is either beta in disguise, a bull-regime artifact, or under-powered.
None survives even raw scrutiny, let alone the family correction (0 configs with
Bonferroni p<0.05).

## Conclusion
Two megascans (77,016 daily + 7,220 intraday/multi-asset = 84,236 hypotheses)
have now failed to find anything beating the gapper fade out of sample. Opening-
range breakout is dead; leveraged-ETF reversion and crypto long-bias are real
but sub-benchmark and mostly beta; the gap edge genuinely needs the ≥100%
microcap extreme (as HYP-101 already showed). **The gapper fade — honestly
~+10–18% / Sharpe ~2 after the bias-correction rebuild — stays the only edge in
this repo to survive a real holdout.** Next gains are in its execution
(borrow/locate, TICK-037), not in a new signal.

Data: `data/scan_results/megascan_v2_20260717.parquet` (all 7,220 rows).
Reusable: `backtester/megascan_v2.py`, `data.get_minute_range` bulk fetcher.
