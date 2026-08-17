# Backtest Throughput — Measured Leaderboard

> 2026-08-16T07:00:46Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 21,822 backtests/sec  
**Measured parallel ceiling:** 99,893 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 15,334,294

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 21,822 | 1,964,012 |
| 90bar | 90 | nojit_fallback_12core | 12 | 99,893 | 8,990,366 |
| 90bar | 90 | pure_python_forex | 1 | 6,199 | 557,934 |
| daily | 2,175 | nojit_fallback_1core | 1 | 914 | 1,988,142 |
| daily | 2,175 | nojit_fallback_12core | 12 | 6,412 | 13,946,400 |
| daily | 2,175 | pure_python_forex | 1 | 223 | 484,560 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 12 | 2,032,841 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 87 | 14,486,255 |
| 5min | 166,941 | pure_python_forex | 1 | 3 | 500,789 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 2,257,314 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 5 | 15,334,294 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 570,969 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
