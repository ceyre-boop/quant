# Backtest Throughput — Measured Leaderboard

> 2026-09-06T16:35:30Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 151 backtests/sec  
**Measured parallel ceiling:** 189 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 382,568

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 151 | 13,608 |
| 90bar | 90 | nojit_fallback_12core | 12 | 189 | 17,044 |
| 90bar | 90 | pure_python_forex | 1 | 280 | 25,173 |
| daily | 2,175 | nojit_fallback_1core | 1 | 31 | 67,045 |
| daily | 2,175 | nojit_fallback_12core | 12 | 31 | 67,868 |
| daily | 2,175 | pure_python_forex | 1 | 2 | 3,204 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 0 | 38,831 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 0 | 67,294 |
| 5min | 166,941 | pure_python_forex | 1 | 0 | 2,003 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 0 | 22,669 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 0 | 382,568 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 46,379 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
