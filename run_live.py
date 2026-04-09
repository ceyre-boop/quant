"""run_live.py — Canonical entry point for live / paper trading.

Delegates to scripts/legacy/run_engine_production.py (the production 3-layer engine).
Usage:
    python run_live.py [--paper]   # default: paper trading on
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
_LEGACY = ROOT / "scripts" / "legacy" / "run_engine_production.py"


def main() -> None:
    result = subprocess.run([sys.executable, str(_LEGACY)] + sys.argv[1:])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
