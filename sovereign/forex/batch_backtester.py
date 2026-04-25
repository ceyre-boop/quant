"""
Forex batch backtester.

Preloads compact signal/price arrays for all pairs and drives the fast
simulation kernel in serial or parallel mode.  A synthetic no-network mode
generates random data so the hot path can be benchmarked in isolation.

Typical usage
-------------
::

    from sovereign.forex.batch_backtester import ForexBatchBacktester

    bt = ForexBatchBacktester()

    # Preloaded serial — needs network for price + macro data
    datasets = bt.preload(start='2015-01-01', end='2024-12-31')
    results  = bt.run_serial(datasets)

    # Preloaded parallel
    results  = bt.run_parallel(datasets, workers=4)

    # Synthetic no-network benchmark
    stats = bt.run_synthetic_benchmark(n_pairs=11, n_bars=2500, n_iterations=400)
    print(stats)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.forex.signal_engine import build_signal_arrays
from sovereign.forex.fast_backtester import (
    ForexArrayDataset,
    ForexFastBacktester,
    HOLD_DAYS,
    STOP_PCT,
    simulate_kernel,
)

logger = logging.getLogger(__name__)


class ForexBatchBacktester:
    """
    Batch layer that sits in front of the fast simulation kernel.

    Preload fetches price data and builds signal arrays via
    :func:`~sovereign.forex.signal_engine.build_signal_arrays` — no
    :class:`~pandas.DataFrame` signal frame is constructed in the hot path.
    """

    def __init__(self, start: str = "2015-01-01", end: str = "2024-12-31"):
        self.start = start
        self.end = end
        self._fast_bt = ForexFastBacktester()

    # ------------------------------------------------------------------
    # Preload
    # ------------------------------------------------------------------

    def preload(self, pairs: Optional[List[str]] = None) -> List[ForexArrayDataset]:
        """
        Download prices and build signal arrays for *pairs*.

        Returns a list of :class:`~sovereign.forex.fast_backtester.ForexArrayDataset`
        ready for the simulation kernel.  Pairs with insufficient data are
        silently skipped.
        """
        from sovereign.forex.data_fetcher import ForexDataFetcher

        if pairs is None:
            pairs = ALL_PAIRS

        fetcher = ForexDataFetcher()
        datasets: List[ForexArrayDataset] = []

        for pair in pairs:
            cfg = PAIR_CONFIG.get(pair)
            if not cfg:
                logger.warning("Unknown pair: %s", pair)
                continue

            df = self._download(pair)
            if df is None or len(df) < 252:
                logger.warning("Insufficient data for %s — skipping", pair)
                continue

            close = (
                df["Close"].values
                if "Close" in df.columns
                else df.iloc[:, 0].values
            ).astype(np.float64)
            opens = (
                df["Open"].values
                if "Open" in df.columns
                else close
            ).astype(np.float64)

            base_country = CB_TO_COUNTRY[cfg.base_central_bank]
            quote_country = CB_TO_COUNTRY[cfg.quote_central_bank]

            try:
                signal_arr, _ = build_signal_arrays(
                    pair, df, base_country, quote_country, fetcher
                )
            except Exception as exc:
                logger.warning("Signal build failed for %s: %s", pair, exc)
                continue

            datasets.append(
                ForexArrayDataset(
                    pair=pair,
                    close=close,
                    opens=opens,
                    signal=signal_arr,
                    n_bars=len(close),
                )
            )

        return datasets

    # ------------------------------------------------------------------
    # Serial run
    # ------------------------------------------------------------------

    def run_serial(self, datasets: List[ForexArrayDataset]) -> list:
        """Run the fast kernel on each dataset sequentially."""
        results = []
        for ds in datasets:
            try:
                r = self._fast_bt.run(ds)
                if r is not None:
                    results.append(r)
            except Exception as exc:
                logger.warning("Fast backtest failed for %s: %s", ds.pair, exc)
        return results

    # ------------------------------------------------------------------
    # Parallel run
    # ------------------------------------------------------------------

    def run_parallel(
        self, datasets: List[ForexArrayDataset], workers: int = 4
    ) -> list:
        """
        Run the fast kernel across *workers* processes.

        Falls back to serial execution when *workers* ≤ 1.
        """
        if workers <= 1:
            return self.run_serial(datasets)

        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one_dataset, ds): ds for ds in datasets}
            for fut in as_completed(futures):
                ds = futures[fut]
                try:
                    r = fut.result()
                    if r is not None:
                        results.append(r)
                except Exception as exc:
                    logger.warning(
                        "Parallel backtest failed for %s: %s", ds.pair, exc
                    )
        return results

    # ------------------------------------------------------------------
    # Synthetic no-network benchmark
    # ------------------------------------------------------------------

    def run_synthetic_benchmark(
        self,
        n_pairs: int = 11,
        n_bars: int = 2500,
        n_iterations: int = 400,
        workers: int = 1,
    ) -> Dict:
        """
        Benchmark the simulation kernel on synthetic random data.

        No network calls are made.  Each iteration generates a fresh random
        price series and sparse monthly signal array, then runs the kernel.

        Returns a dict with timing and throughput metrics.
        """
        rng = np.random.default_rng(42)
        total_runs = n_pairs * n_iterations
        t0 = time.perf_counter()

        if workers <= 1:
            _run_synthetic_batch(rng, n_pairs, n_bars, n_iterations)
        else:
            _run_synthetic_batch_parallel(n_pairs, n_bars, n_iterations, workers)

        elapsed = time.perf_counter() - t0
        runs_per_sec = total_runs / elapsed if elapsed > 0 else float("inf")

        return {
            "n_pairs": n_pairs,
            "n_bars": n_bars,
            "n_iterations": n_iterations,
            "total_runs": total_runs,
            "elapsed_s": round(elapsed, 3),
            "runs_per_sec": round(runs_per_sec, 0),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download(self, pair: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(
                pair,
                start=self.start,
                end=self.end,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna()
        except Exception as exc:
            logger.warning("Download failed for %s: %s", pair, exc)
            return None


# ---------------------------------------------------------------------------
# Module-level helpers (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _run_one_dataset(dataset: ForexArrayDataset):
    """Top-level wrapper — must be module-level so it can be pickled."""
    bt = ForexFastBacktester()
    return bt.run(dataset)


def _make_synthetic_dataset(rng: np.random.Generator, n_bars: int) -> ForexArrayDataset:
    """Build one synthetic dataset with random log-normal prices and sparse signals."""
    log_returns = rng.normal(0.0, 0.01, n_bars)
    close = np.exp(np.cumsum(log_returns)) * 1.2
    opens = close * rng.uniform(0.999, 1.001, n_bars)

    signal = np.zeros(n_bars, dtype=np.float64)
    # Approximately monthly signals — one every ~21 bars
    choices = np.array([-1.0, 0.0, 0.0, 1.0])
    for j in range(0, n_bars, 21):
        signal[j] = choices[rng.integers(0, 4)]

    return ForexArrayDataset(
        pair="SYNTHETIC",
        close=close,
        opens=opens,
        signal=signal,
        n_bars=n_bars,
    )


def _run_synthetic_batch(
    rng: np.random.Generator,
    n_pairs: int,
    n_bars: int,
    n_iterations: int,
) -> None:
    """Serial synthetic run — called inside run_synthetic_benchmark."""
    for _ in range(n_pairs * n_iterations):
        ds = _make_synthetic_dataset(rng, n_bars)
        simulate_kernel(ds.close, ds.opens, ds.signal, HOLD_DAYS, STOP_PCT)


def _run_synthetic_worker(args) -> None:
    """Worker function for parallel synthetic benchmark (must be picklable)."""
    n_pairs, n_bars, n_iterations, seed = args
    rng = np.random.default_rng(seed)
    _run_synthetic_batch(rng, n_pairs, n_bars, n_iterations)


def _run_synthetic_batch_parallel(
    n_pairs: int,
    n_bars: int,
    n_iterations: int,
    workers: int,
) -> None:
    """Distribute synthetic iterations across *workers* processes."""
    # Split pairs evenly across workers
    pairs_per_worker = max(1, n_pairs // workers)
    remainder = n_pairs % workers
    tasks = []
    seed = 0
    for w in range(workers):
        w_pairs = pairs_per_worker + (1 if w < remainder else 0)
        if w_pairs > 0:
            tasks.append((w_pairs, n_bars, n_iterations, seed + w))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_run_synthetic_worker, tasks))
