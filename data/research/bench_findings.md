# Backtest Throughput — Measured Leaderboard

> 2026-08-23T14:28:03Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 3,148 backtests/sec  
**Measured parallel ceiling:** 11,414 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 1,027,298

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 3,148 | 283,279 |
| 90bar | 90 | nojit_fallback_12core | 12 | 11,414 | 1,027,298 |
| 90bar | 90 | pure_python_forex | 1 | 1,903 | 171,254 |
| daily | 2,175 | nojit_fallback_1core | 1 | 12 | 26,436 |
| daily | 2,175 | nojit_fallback_12core | 12 | 149 | 324,813 |
| daily | 2,175 | pure_python_forex | 1 | 91 | 197,829 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 5 | 806,766 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 1 | 233,475 |
| 5min | 166,941 | pure_python_forex | 1 | 0 | 49,142 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 0 | 91,992 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 0 | 735,254 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 12,416 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
