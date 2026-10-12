# Backtest Throughput — Measured Leaderboard

> 2026-10-11T08:49:51Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 15,431 backtests/sec  
**Measured parallel ceiling:** 9,531 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 3,283,475

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 15,431 | 1,388,828 |
| 90bar | 90 | nojit_fallback_12core | 12 | 9,531 | 857,761 |
| 90bar | 90 | pure_python_forex | 1 | 5,388 | 484,928 |
| daily | 2,175 | nojit_fallback_1core | 1 | 676 | 1,470,236 |
| daily | 2,175 | nojit_fallback_12core | 12 | 430 | 935,674 |
| daily | 2,175 | pure_python_forex | 1 | 26 | 55,408 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 2 | 329,688 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 20 | 3,283,475 |
| 5min | 166,941 | pure_python_forex | 1 | 2 | 278,692 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 0 | 270,387 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 0 | 1,489,686 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 104,029 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
