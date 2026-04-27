#!/usr/bin/env python3
"""
Forex scan runner.

Usage:
    python scripts/run_forex_scan.py                      # full scan, $10k account
    python scripts/run_forex_scan.py --balance 50000      # custom account size
    python scripts/run_forex_scan.py --pair EURUSD=X      # single pair
    python scripts/run_forex_scan.py --backtest           # run backtester
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s %(name)s: %(message)s',
)

from sovereign.forex import ForexSpecialist  # noqa: E402


def _backtest():
    print("\nFOREX BACKTEST — 2015–2024")
    print('=' * 60)
    from sovereign.forex.forex_backtester import ForexBacktester as BT
    bt = BT()
    results = bt.backtest_all()
    if not results:
        raise SystemExit('Backtest produced no results.')

    best = max(results, key=lambda r: r.sharpe)
    print(f"\nBest Sharpe: {best.pair}  sharpe={best.sharpe:.2f}  "
          f"win={best.win_rate:.1%}  pf={best.profit_factor:.2f}")


def _benchmark(iterations: int):
    from sovereign.forex.batch_backtester import ForexBatchBacktester
    batch = ForexBatchBacktester()
    bench = batch.benchmark(iterations=iterations)
    print("\nFOREX BACKTEST BENCHMARK")
    print('=' * 60)
    print(f"Pairs loaded: {bench.pair_count}")
    print(f"Iterations:   {bench.iterations}")
    print(f"Total runs:   {bench.total_runs}")
    print(f"Elapsed:      {bench.elapsed_sec:.3f}s")
    print(f"Runs/sec:     {bench.runs_per_sec:,.0f}")


def _benchmark_parallel(iterations: int, workers: Optional[int]):
    from sovereign.forex.batch_backtester import ForexBatchBacktester
    batch = ForexBatchBacktester()
    bench = batch.benchmark_parallel(iterations=iterations, workers=workers)
    print("\nFOREX BACKTEST BENCHMARK (PARALLEL)")
    print('=' * 60)
    print(f"Pairs loaded: {bench.pair_count}")
    print(f"Iterations:   {bench.iterations}")
    print(f"Total runs:   {bench.total_runs}")
    print(f"Elapsed:      {bench.elapsed_sec:.3f}s")
    print(f"Runs/sec:     {bench.runs_per_sec:,.0f}")


def _benchmark_synthetic(iterations: int, pair_count: int, bars: int, workers: Optional[int]):
    from sovereign.forex.batch_backtester import ForexBatchBacktester
    batch = ForexBatchBacktester()
    bench = batch.benchmark_synthetic(
        pair_count=pair_count,
        bars=bars,
        iterations=iterations,
        workers=workers,
    )
    print("\nFOREX SYNTHETIC BENCHMARK")
    print('=' * 60)
    print(f"Pairs loaded: {bench.pair_count}")
    print(f"Bars/pair:    {bars}")
    print(f"Iterations:   {bench.iterations}")
    print(f"Total runs:   {bench.total_runs}")
    print(f"Elapsed:      {bench.elapsed_sec:.3f}s")
    print(f"Runs/sec:     {bench.runs_per_sec:,.0f}")


def main():
    parser = argparse.ArgumentParser(description='Forex scan')
    parser.add_argument('--balance', type=float, default=10_000,
                        help='Account balance in USD (default 10000)')
    parser.add_argument('--pair', type=str, default=None,
                        help='Single pair to evaluate (e.g. EURUSD=X)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtester instead of live scan')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark the preloaded forex backtest path')
    parser.add_argument('--parallel-benchmark', action='store_true',
                        help='Benchmark the preloaded forex backtest path in parallel')
    parser.add_argument('--synthetic-benchmark', action='store_true',
                        help='Benchmark the array-native forex path without network/data download')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Benchmark iterations (default 200)')
    parser.add_argument('--pair-count', type=int, default=11,
                        help='Synthetic pair count for benchmark mode')
    parser.add_argument('--bars', type=int, default=2500,
                        help='Synthetic bars per pair for benchmark mode')
    parser.add_argument('--workers', type=int, default=None,
                        help='Worker count for parallel benchmark')
    args = parser.parse_args()

    if args.backtest:
        _backtest()
        return
    if args.benchmark:
        _benchmark(args.iterations)
        return
    if args.parallel_benchmark:
        _benchmark_parallel(args.iterations, args.workers)
        return
    if args.synthetic_benchmark:
        _benchmark_synthetic(args.iterations, args.pair_count, args.bars, args.workers)
        return

    specialist = ForexSpecialist(account_balance=args.balance)

    if args.pair:
        candidate = specialist.evaluate_pair(args.pair)
        if candidate:
            print(candidate.summary())
        else:
            print(f"No tradeable setup for {args.pair} right now.")
    else:
        report = specialist.run()
        report.print()


if __name__ == '__main__':
    main()
