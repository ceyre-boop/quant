# Backtest Throughput — Measured Leaderboard

> 2026-08-09T07:01:00Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 22,413 backtests/sec  
**Measured parallel ceiling:** 73,792 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 8,509,115

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 22,413 | 2,017,180 |
| 90bar | 90 | nojit_fallback_12core | 12 | 73,792 | 6,641,314 |
| 90bar | 90 | pure_python_forex | 1 | 5,087 | 457,827 |
| daily | 2,175 | nojit_fallback_1core | 1 | 968 | 2,105,247 |
| daily | 2,175 | nojit_fallback_12core | 12 | 3,912 | 8,509,115 |
| daily | 2,175 | pure_python_forex | 1 | 181 | 393,892 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 13 | 2,105,583 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 46 | 7,681,949 |
| 5min | 166,941 | pure_python_forex | 1 | 2 | 399,814 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 2,370,766 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 3 | 8,488,370 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 534,672 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
