# v015 Re-measurement — fresh period under LIVE gates (full-history methodology)

> The dossier's fresh -0.085 came from a short-window run where the VIX gates never warmed up (ungated v015). This re-measures fresh 2025-26 from the full-history gate-active ledger (which reconciles OOS 2023-24). Sharpe = √n-weighted costed portfolio (risk_adjusted_pnl_pct).

## Headline

- **Gated v015 fresh 2025-26 portfolio Sharpe: 0.038**
- Ungated (dossier backup) fresh portfolio Sharpe: -0.076 (documented headline -0.085)
- Sanity: full-history OOS 2023-24 gated Sharpe 1.168 (≈ documented 1.25)

## Fresh 2025-26 by pair — gated vs ungated

| pair | gated n | gated Sharpe | ungated n | ungated Sharpe |
|---|---|---|---|---|
| EURUSD | 13 | -0.608 | 14 | -0.462 |
| GBPUSD | 14 | 0.452 | 16 | 1.016 |
| USDJPY | 6 | -0.937 | 16 | -1.017 |
| AUDUSD | 12 | 0.953 | 13 | 0.156 |

**EURUSD (load-bearing — its VIX>18 gate didn't fire this window, so gated≈ungated): fresh Sharpe -0.608.** If EURUSD is still negative under proper gating, the collapse is real regardless of USDJPY's gate effect.

## Year-by-year — METHOD MISMATCH, do NOT read as a correction

| Year | Documented (rolling walk-forward, OOS) | Per-year in-sample backtest Sharpe |
|---|---|---|
| 2021 | -0.13 | +1.82 |
| 2022 | +0.51 | +0.85 |
| 2023 | +1.26 | +1.53 |
| 2024 | -0.09 | +0.75 |
| 2025-26 | -0.085 | +0.04 |

**These two columns are NOT comparable.** The documented sequence is a *rolling walk-forward*
(refit on prior data, predict each year OOS) — conservative, the honest fragility measure. My right
column is a *per-calendar-year IN-SAMPLE* portfolio Sharpe of the single static full-period backtest,
and single-year buckets inflate under √n annualisation (that is why 2021 reads +1.82). So the right
column does NOT "correct" the walk-forward — it measures a different thing in-sample. **Do not
conclude the fragility was an artifact from this table.** The only method-matched, valid comparisons
are the fresh-period gated-vs-ungated above (same per-period portfolio Sharpe) and the OOS-2023-24
sanity (1.168 ≈ documented 1.25, which validates the method on a 2-year window).

## Read

The dossier's -0.085 vs the gated fresh Sharpe above shows how much the missing live gates moved the number. If gated fresh is still ~0 or negative (esp. EURUSD), the collapse is REAL and HYP-065 (regime conditionality) remains a live question on a clean ledger. If gated fresh recovers to clearly positive, the 'collapse' was largely a short-window gating artifact and the question changes.

