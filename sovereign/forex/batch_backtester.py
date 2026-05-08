"""
Batch-oriented forex backtest utilities.

This is the bridge between the shared signal engine and a machine that can
actually take advantage of an M4: preload once, warm once, then run many
simulations without reconstructing everything each time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sovereign.forex.fast_backtester import (
    simulate_forex_trades_arrays,
    warmup_forex_kernel,
    ForexArrayDataset as LegacyForexArrayDataset,
    ForexFastBacktester,
    HOLD_DAYS,
    STOP_PCT,
)
from sovereign.forex.forex_backtester import ForexBacktester, ForexBacktestResult
from sovereign.forex.pair_universe import ALL_PAIRS, CB_TO_COUNTRY, PAIR_CONFIG
from sovereign.forex.signal_engine import build_signal_arrays


@dataclass
class ForexBatchBenchmark:
    pair_count: int
    iterations: int
    total_runs: int
    elapsed_sec: float

    @property
    def runs_per_sec(self) -> float:
        return self.total_runs / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


@dataclass
class ForexArrayDataset:
    opens: np.ndarray
    closes: np.ndarray
    signals: np.ndarray
    hold_days: np.ndarray
    index: pd.Index


def _make_synthetic_dataset(rng: np.random.Generator, n_bars: int = 2500, pair: str = "SYN00=X") -> LegacyForexArrayDataset:
    trend = 0.00015
    noise = rng.normal(0.0, 0.0008, size=n_bars)
    rets = trend + noise
    close = 1.20 * np.cumprod(1.0 + rets)
    opens = close * (1.0 - rng.normal(0.0, 0.0001, size=n_bars))
    signal = rng.choice(np.array([-1.0, 0.0, 1.0]), size=n_bars, p=[0.1, 0.8, 0.1])
    return LegacyForexArrayDataset(
        pair=pair,
        close=close.astype(np.float64),
        opens=opens.astype(np.float64),
        signal=signal.astype(np.float64),
        n_bars=n_bars,
    )


class ForexBatchBacktester:
    def __init__(self, start: str = '2015-01-01', end: str = '2024-12-31'):
        self._backtester = ForexBacktester(start=start, end=end)
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._array_cache: Dict[str, ForexArrayDataset] = {}

    def preload(self, pairs: Optional[list[str]] = None) -> None:
        pairs = pairs or list(ALL_PAIRS)
        for pair in pairs:
            if pair in self._price_cache:
                continue
            cfg = PAIR_CONFIG.get(pair)
            if not cfg:
                continue
            df = self._backtester._download_price(pair)
            if df is None or len(df) < 252:
                continue
            base_country = CB_TO_COUNTRY[cfg.base_central_bank]
            quote_country = CB_TO_COUNTRY[cfg.quote_central_bank]
            close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
            signals, hold_days = build_signal_arrays(
                pair=pair,
                prices=df,
                base_country=base_country,
                quote_country=quote_country,
                fetcher=self._backtester._fetcher,
                cb_trigger=self._backtester._cb,
                config=self._backtester._signals.config,
                start=self._backtester.start,
                end=self._backtester.end,
            )
            self._price_cache[pair] = df
            self._array_cache[pair] = ForexArrayDataset(
                opens=df['Open'].to_numpy(dtype=np.float64) if 'Open' in df.columns else df.iloc[:, 0].to_numpy(dtype=np.float64),
                closes=close.to_numpy(dtype=np.float64),
                signals=signals,
                hold_days=hold_days,
                index=df.index,
            )

    def backtest_pair(self, pair: str) -> Optional[ForexBacktestResult]:
        if pair not in self._price_cache:
            self.preload([pair])
        dataset = self._array_cache.get(pair)
        if dataset is None:
            return None
        trades = simulate_forex_trades_arrays(
            opens=dataset.opens,
            closes=dataset.closes,
            signals=dataset.signals,
            hold_days=dataset.hold_days,
            stop_pct=self._backtester.STOP_PCT,
            index=dataset.index,
        )
        if not trades:
            return None
        return self._backtester._compute_stats(pair, trades, len(dataset.index))

    def benchmark(self, iterations: int = 200) -> ForexBatchBenchmark:
        self.preload()
        warmup_forex_kernel()

        pairs = sorted(self._array_cache.keys())
        t0 = time.perf_counter()
        for _ in range(iterations):
            for pair in pairs:
                self.backtest_pair(pair)
        elapsed = time.perf_counter() - t0
        return ForexBatchBenchmark(
            pair_count=len(pairs),
            iterations=iterations,
            total_runs=len(pairs) * iterations,
            elapsed_sec=elapsed,
        )

    def benchmark_parallel(self, iterations: int = 200, workers: Optional[int] = None) -> ForexBatchBenchmark:
        self.preload()
        warmup_forex_kernel()

        pairs = sorted(self._array_cache.keys())
        workers = workers or min(len(pairs), os.cpu_count() or 1)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(iterations):
                list(pool.map(self.backtest_pair, pairs))
        elapsed = time.perf_counter() - t0
        return ForexBatchBenchmark(
            pair_count=len(pairs),
            iterations=iterations,
            total_runs=len(pairs) * iterations,
            elapsed_sec=elapsed,
        )

    def benchmark_synthetic(
        self,
        pair_count: int = 11,
        bars: int = 2500,
        iterations: int = 500,
        workers: Optional[int] = None,
    ) -> ForexBatchBenchmark:
        self._array_cache = self._build_synthetic_cache(pair_count=pair_count, bars=bars)
        warmup_forex_kernel()

        pairs = sorted(self._array_cache.keys())
        workers = workers or min(len(pairs), os.cpu_count() or 1)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(iterations):
                list(pool.map(self.backtest_pair, pairs))
        elapsed = time.perf_counter() - t0
        return ForexBatchBenchmark(
            pair_count=len(pairs),
            iterations=iterations,
            total_runs=len(pairs) * iterations,
            elapsed_sec=elapsed,
        )

    def _build_synthetic_cache(self, pair_count: int, bars: int) -> Dict[str, ForexArrayDataset]:
        cache: Dict[str, ForexArrayDataset] = {}
        base_index = pd.bdate_range('2014-01-01', periods=bars)
        for i in range(pair_count):
            pair = f'SYN{i:02d}=X'
            trend = 1.0 + (i % 3) * 0.0002
            noise = np.sin(np.linspace(0, 18, bars)) * (0.3 + i * 0.01)
            closes = np.cumprod(1.0 + 0.0001 * trend + 0.0005 * noise) * (100 + i)
            opens = closes * (1.0 - 0.0002)
            signals = np.zeros(bars, dtype=np.int8)
            signals[::180] = 1 if i % 2 == 0 else -1
            hold_days = np.full(bars, 45 + (i % 4) * 5, dtype=np.int32)
            cache[pair] = ForexArrayDataset(
                opens=opens.astype(np.float64),
                closes=closes.astype(np.float64),
                signals=signals,
                hold_days=hold_days,
                index=base_index,
            )
        return cache

    # Compatibility wrappers used by unit tests
    def run_synthetic_benchmark(self, n_pairs: int = 11, n_bars: int = 2500, n_iterations: int = 500) -> dict:
        bm = self.benchmark_synthetic(pair_count=n_pairs, bars=n_bars, iterations=n_iterations)
        return {
            "n_pairs": bm.pair_count,
            "n_bars": n_bars,
            "n_iterations": n_iterations,
            "total_runs": bm.total_runs,
            "elapsed_s": bm.elapsed_sec,
            "runs_per_sec": bm.runs_per_sec,
        }

    def run_serial(self, datasets: list[LegacyForexArrayDataset]):
        bt = ForexFastBacktester(hold_days=HOLD_DAYS, stop_pct=STOP_PCT)
        out = []
        for ds in datasets:
            result = bt.run(ds)
            if result is not None:
                out.append(result)
        return out
