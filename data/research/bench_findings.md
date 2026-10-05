# Backtest Throughput — Measured Leaderboard

> 2026-10-04T10:55:44Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 7,477 backtests/sec  
**Measured parallel ceiling:** 1,616 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 1,870,335

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 7,477 | 672,970 |
| 90bar | 90 | nojit_fallback_12core | 12 | 1,616 | 145,481 |
| 90bar | 90 | pure_python_forex | 1 | 1,924 | 173,201 |
| daily | 2,175 | nojit_fallback_1core | 1 | 4 | 7,958 |
| daily | 2,175 | nojit_fallback_12core | 12 | 98 | 212,174 |
| daily | 2,175 | pure_python_forex | 1 | 19 | 41,443 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 0 | 48,491 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 9 | 1,479,886 |
| 5min | 166,941 | pure_python_forex | 1 | 0 | 48,379 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 0 | 53,643 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 1 | 1,870,335 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 341,428 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
