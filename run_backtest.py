"""run_backtest.py — Canonical entry point for backtesting.

Delegates to backtest/backtest_runner.py and scripts/full_backtest.py.
To run the statistical backtest:
    python run_backtest.py [--symbol SPY] [--start 2023-01-01] [--end 2023-12-31]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Clawd Trading backtest")
    parser.add_argument("--full", action="store_true", help="Run full statistical backtest (scripts/full_backtest.py)")
    args, remaining = parser.parse_known_args()

    if args.full:
        script = ROOT / "scripts" / "full_backtest.py"
    else:
        script = ROOT / "scripts" / "full_backtest.py"

    result = subprocess.run([sys.executable, str(script)] + remaining)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
