# Backtest Throughput — Measured Leaderboard

> 2026-08-30T12:02:01Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 12,410 backtests/sec  
**Measured parallel ceiling:** 24,491 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 3,782,434

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 12,410 | 1,116,929 |
| 90bar | 90 | nojit_fallback_12core | 12 | 24,491 | 2,204,181 |
| 90bar | 90 | pure_python_forex | 1 | 443 | 39,857 |
| daily | 2,175 | nojit_fallback_1core | 1 | 331 | 719,155 |
| daily | 2,175 | nojit_fallback_12core | 12 | 1,404 | 3,054,857 |
| daily | 2,175 | pure_python_forex | 1 | 34 | 73,510 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 6 | 1,048,156 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 23 | 3,782,434 |
| 5min | 166,941 | pure_python_forex | 1 | 1 | 127,551 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 0 | 256,039 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 1 | 1,827,863 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 174,515 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
