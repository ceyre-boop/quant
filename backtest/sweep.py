"""
Parallel Parameter Sweep — M4 Pro (14 cores)

Distributes thousands of SweepParams combinations across all CPU cores,
sharing pre-loaded DataArrays with each worker once (not once per task).

Usage:
    from backtest.sweep import ParameterSweep, quick_sweep
    from backtest.fast_engine import FastBacktestEngine

    engine = FastBacktestEngine.from_dataframe(df)
    sweep  = ParameterSweep(engine)

    results = sweep.run_grid({
        "stop_atr_mult": [1.5, 2.0, 2.5, 3.0],
        "tp_rr":         [1.5, 2.0, 2.5, 3.0, 4.0],
        "atr_period":    [10, 14, 20],
    })
    # → sorted DataFrame, best profit_factor first

    # Or the one-liner for quick experiments:
    results = quick_sweep(df, stop_atr_mults=[1.5,2,2.5], tp_rrs=[2,2.5,3])
"""

from __future__ import annotations

import os
import time
import multiprocessing as mp
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.fast_engine import (
    FastBacktestEngine,
    DataArrays,
    SweepParams,
    FastResult,
)


# ── Worker machinery ─────────────────────────────────────────────────────────
# Must be module-level for multiprocessing 'spawn' (macOS default).

_WORKER_ARRAYS: Optional[DataArrays] = None


def _worker_init(arrays: DataArrays) -> None:
    """Initialiser: runs once per worker process, stores shared data."""
    global _WORKER_ARRAYS
    _WORKER_ARRAYS = arrays


def _worker_run(params_tuple: tuple) -> dict:
    """Task: deserialise params, run one backtest, return dict."""
    params = SweepParams.from_tuple(params_tuple)
    engine = FastBacktestEngine(_WORKER_ARRAYS)
    result = engine.run_single(params)
    return result.to_dict()


# ── Sweep orchestrator ───────────────────────────────────────────────────────

class ParameterSweep:
    """Runs a grid or list of SweepParams across all CPU cores.

    Args:
        engine: Pre-built FastBacktestEngine (data already loaded).
        n_cores: Worker processes. Defaults to all physical cores.
    """

    def __init__(self, engine: FastBacktestEngine, n_cores: Optional[int] = None):
        self.engine = engine
        self.n_cores = n_cores or _physical_cores()

    # ── Grid sweep ────────────────────────────────────────────────────────

    def run_grid(
        self,
        param_grid: Dict[str, list],
        min_trades: int = 0,
        sort_by: str = "profit_factor",
    ) -> pd.DataFrame:
        """Cartesian product sweep.

        Args:
            param_grid: Dict of param_name → list of values to sweep.
                        Any SweepParams field can be included.
            min_trades: Filter out results with fewer trades.
            sort_by: Column to sort output by (descending).

        Returns:
            DataFrame sorted by sort_by, columns = all SweepParams fields +
            [total_pnl, total_return_pct, trades, win_rate, profit_factor, max_dd_pct].
        """
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        combos = list(product(*value_lists))

        param_tuples = []
        default = SweepParams()
        default_dict = default.__dict__.copy()

        for combo in combos:
            d = default_dict.copy()
            for k, v in zip(keys, combo):
                d[k] = v
            param_tuples.append(SweepParams(**d).as_tuple())

        return self._dispatch(param_tuples, min_trades, sort_by)

    # ── List sweep (arbitrary params) ────────────────────────────────────

    def run_list(
        self,
        params_list: List[SweepParams],
        min_trades: int = 0,
        sort_by: str = "profit_factor",
    ) -> pd.DataFrame:
        """Run an arbitrary list of SweepParams (e.g. from random search or Bayesian opt)."""
        param_tuples = [p.as_tuple() for p in params_list]
        return self._dispatch(param_tuples, min_trades, sort_by)

    # ── Random search ────────────────────────────────────────────────────

    def run_random(
        self,
        n: int,
        stop_atr_range: tuple = (1.0, 4.0),
        tp_rr_range: tuple = (1.0, 5.0),
        atr_period_range: tuple = (8, 30),
        seed: Optional[int] = None,
        min_trades: int = 0,
        sort_by: str = "profit_factor",
    ) -> pd.DataFrame:
        """Latin-hypercube-style random parameter search.

        Args:
            n: Number of random combinations to evaluate.
        """
        rng = np.random.default_rng(seed)
        params_list = []
        for _ in range(n):
            params_list.append(SweepParams(
                stop_atr_mult=float(rng.uniform(*stop_atr_range)),
                tp_rr=float(rng.uniform(*tp_rr_range)),
                atr_period=int(rng.integers(*atr_period_range)),
            ))
        return self.run_list(params_list, min_trades, sort_by)

    # ── Internal dispatch ─────────────────────────────────────────────────

    def _dispatch(
        self,
        param_tuples: list,
        min_trades: int,
        sort_by: str,
    ) -> pd.DataFrame:
        n = len(param_tuples)
        arrays = self.engine.arrays

        # Warm JIT on main process (writes to on-disk cache for workers)
        self.engine.warmup()

        chunksize = max(1, n // (self.n_cores * 8))

        t0 = time.perf_counter()
        # 'fork' on macOS inherits the already-compiled Numba JIT from the parent
        # process — workers skip re-import (~200 ms each) and start in <10 ms.
        # Safe here because we are pure numpy/numba with no GUI or ObjC threads.
        # Fall back to 'spawn' on Windows.
        import platform
        ctx_method = "fork" if platform.system() != "Windows" else "spawn"
        ctx = mp.get_context(ctx_method)
        with ctx.Pool(
            processes=self.n_cores,
            initializer=_worker_init,
            initargs=(arrays,),
        ) as pool:
            raw = pool.map(_worker_run, param_tuples, chunksize=chunksize)
        elapsed = time.perf_counter() - t0

        rate = n / elapsed if elapsed > 0 else 0
        print(
            f"[Sweep] {n:,} backtests across {self.n_cores} cores "
            f"in {elapsed*1000:.0f} ms  →  {rate:,.0f}/sec"
        )

        df = pd.DataFrame(raw)
        if min_trades:
            df = df[df["trades"] >= min_trades]
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        return df.reset_index(drop=True)


# ── Convenience one-liner ────────────────────────────────────────────────────

def quick_sweep(
    df: pd.DataFrame,
    stop_atr_mults: Optional[List[float]] = None,
    tp_rrs: Optional[List[float]] = None,
    atr_periods: Optional[List[int]] = None,
    signal_fn=None,
    n_cores: Optional[int] = None,
    min_trades: int = 5,
    sort_by: str = "profit_factor",
) -> pd.DataFrame:
    """One-liner: build engine from DataFrame and run a grid sweep.

    Sensible defaults if you omit ranges.

    Example:
        results = quick_sweep(df,
                              stop_atr_mults=[1.5, 2.0, 2.5],
                              tp_rrs=[2.0, 2.5, 3.0])
        print(results.head(10))
    """
    stop_atr_mults = stop_atr_mults or [1.5, 2.0, 2.5, 3.0]
    tp_rrs         = tp_rrs         or [1.5, 2.0, 2.5, 3.0, 4.0]
    atr_periods    = atr_periods    or [10, 14, 20]

    engine = FastBacktestEngine.from_dataframe(df, signal_fn=signal_fn)
    sweep  = ParameterSweep(engine, n_cores=n_cores)

    return sweep.run_grid(
        {
            "stop_atr_mult": stop_atr_mults,
            "tp_rr":         tp_rrs,
            "atr_period":    atr_periods,
        },
        min_trades=min_trades,
        sort_by=sort_by,
    )


# ── Helper ───────────────────────────────────────────────────────────────────

def _physical_cores() -> int:
    """Return physical core count (not hyperthreads). M4 Pro → 14."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.physicalcpu"], text=True
        ).strip()
        return int(out)
    except Exception:
        return os.cpu_count() or 4
