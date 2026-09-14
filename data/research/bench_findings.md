# Backtest Throughput — Measured Leaderboard

> 2026-09-13T07:25:34Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 11,842 backtests/sec  
**Measured parallel ceiling:** 39,266 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 5,999,482

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 11,842 | 1,065,826 |
| 90bar | 90 | nojit_fallback_12core | 12 | 39,266 | 3,533,949 |
| 90bar | 90 | pure_python_forex | 1 | 4,740 | 426,622 |
| daily | 2,175 | nojit_fallback_1core | 1 | 831 | 1,807,953 |
| daily | 2,175 | nojit_fallback_12core | 12 | 2,758 | 5,999,482 |
| daily | 2,175 | pure_python_forex | 1 | 130 | 283,420 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 10 | 1,693,755 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 33 | 5,452,363 |
| 5min | 166,941 | pure_python_forex | 1 | 2 | 278,465 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 1,922,409 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 2 | 5,399,865 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 498,975 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
