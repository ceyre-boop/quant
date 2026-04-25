#!/usr/bin/env python3
"""
Forex scan runner.

Usage:
    python scripts/run_forex_scan.py                      # full scan, $10k account
    python scripts/run_forex_scan.py --balance 50000      # custom account size
    python scripts/run_forex_scan.py --pair EURUSD=X      # single pair
    python scripts/run_forex_scan.py --backtest           # run backtester

    # Benchmark modes (no live data needed for synthetic):
    python scripts/run_forex_scan.py --benchmark                  # preloaded serial
    python scripts/run_forex_scan.py --benchmark --workers 4      # preloaded parallel
    python scripts/run_forex_scan.py --synthetic-benchmark        # no-network hot path
    python scripts/run_forex_scan.py --synthetic-benchmark --pair-count 11 --bars 2500 --iterations 400 --workers 1
"""
import argparse
import logging
import sys
from pathlib import Path

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


def _benchmark_preloaded(workers: int):
    """Preloaded benchmark — fetches real prices and macro data once, then runs the kernel."""
    from sovereign.forex.batch_backtester import ForexBatchBacktester

    print("\nFOREX PRELOADED BENCHMARK")
    print('=' * 60)
    bt = ForexBatchBacktester()

    print("Preloading datasets (requires network)…")
    datasets = bt.preload()
    if not datasets:
        raise SystemExit("Preload returned no datasets — check network / data.")

    print(f"Loaded {len(datasets)} pair(s).  Running kernel…")
    if workers > 1:
        results = bt.run_parallel(datasets, workers=workers)
        mode = f"parallel ({workers} workers)"
    else:
        results = bt.run_serial(datasets)
        mode = "serial"

    print(f"\nCompleted {len(results)} backtest(s) [{mode}]")
    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        print(
            f"  {r.pair:12s}  win={r.win_rate:.1%}  pf={r.profit_factor:.2f}"
            f"  sharpe={r.sharpe:.2f}  dd={r.max_drawdown:.1%}"
        )


def _benchmark_synthetic(pair_count: int, bars: int, iterations: int, workers: int):
    """Synthetic no-network benchmark — measures the hot path in isolation."""
    from sovereign.forex.batch_backtester import ForexBatchBacktester

    print("\nFOREX SYNTHETIC BENCHMARK (no network)")
    print('=' * 60)
    bt = ForexBatchBacktester()
    stats = bt.run_synthetic_benchmark(
        n_pairs=pair_count,
        n_bars=bars,
        n_iterations=iterations,
        workers=workers,
    )
    print(f"  {stats['n_pairs']} pairs")
    print(f"  {stats['n_bars']} bars per pair")
    print(f"  {stats['n_iterations']} iterations")
    print(f"  {stats['total_runs']} total runs in {stats['elapsed_s']:.3f}s")
    print(f"  about {stats['runs_per_sec']:,.0f} runs/sec")


def main():
    parser = argparse.ArgumentParser(description='Forex scan')
    parser.add_argument('--balance', type=float, default=10_000,
                        help='Account balance in USD (default 10000)')
    parser.add_argument('--pair', type=str, default=None,
                        help='Single pair to evaluate (e.g. EURUSD=X)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run pandas backtester instead of live scan')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run preloaded benchmark (serial unless --workers > 1)')
    parser.add_argument('--synthetic-benchmark', action='store_true',
                        help='Run synthetic no-network benchmark')
    parser.add_argument('--pair-count', type=int, default=11,
                        help='Number of synthetic pairs (default 11)')
    parser.add_argument('--bars', type=int, default=2500,
                        help='Bars per synthetic pair (default 2500)')
    parser.add_argument('--iterations', type=int, default=400,
                        help='Iterations per pair in synthetic benchmark (default 400)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Worker processes for parallel modes (default 1 = serial)')
    args = parser.parse_args()

    if args.backtest:
        _backtest()
        return

    if args.benchmark:
        _benchmark_preloaded(workers=args.workers)
        return

    if args.synthetic_benchmark:
        _benchmark_synthetic(
            pair_count=args.pair_count,
            bars=args.bars,
            iterations=args.iterations,
            workers=args.workers,
        )
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
