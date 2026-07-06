# Backtest Throughput — Measured Leaderboard

> 2026-07-05T08:13:24Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 21,585 backtests/sec  
**Measured parallel ceiling:** 13,955 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 5,484,788

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 21,585 | 1,942,673 |
| 90bar | 90 | nojit_fallback_12core | 12 | 13,955 | 1,255,980 |
| 90bar | 90 | pure_python_forex | 1 | 2,124 | 191,199 |
| daily | 2,175 | nojit_fallback_1core | 1 | 592 | 1,287,967 |
| daily | 2,175 | nojit_fallback_12core | 12 | 2,522 | 5,484,788 |
| daily | 2,175 | pure_python_forex | 1 | 140 | 303,860 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 13 | 2,127,386 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 12 | 2,078,356 |
| 5min | 166,941 | pure_python_forex | 1 | 1 | 151,236 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 2,015,898 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 1 | 3,628,692 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 158,619 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
