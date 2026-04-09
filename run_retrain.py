"""run_retrain.py — Canonical entry point for model retraining.

Delegates to clawd_trading/meta_evaluator/refit_scheduler.py.
Usage:
    python run_retrain.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
_SCHEDULER = ROOT / "clawd_trading" / "meta_evaluator" / "refit_scheduler.py"


def main() -> None:
    result = subprocess.run([sys.executable, str(_SCHEDULER)] + sys.argv[1:])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
