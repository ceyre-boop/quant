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


def main():
    parser = argparse.ArgumentParser(description='Forex scan')
    parser.add_argument('--balance', type=float, default=10_000,
                        help='Account balance in USD (default 10000)')
    parser.add_argument('--pair', type=str, default=None,
                        help='Single pair to evaluate (e.g. EURUSD=X)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtester instead of live scan')
    args = parser.parse_args()

    if args.backtest:
        _backtest()
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
