#!/usr/bin/env python3
"""
End-of-session review in one command — so the nightly rep loop is a single habit.

Runs, in order:
  1. futures_reconcile.py   — close the loop (backfill exits). `--ib` → close real IB
     trades from actual fills; default closes only sim trades (never yfinance over a real rep).
  2. futures_calibration.py — score today's oracle morning call.
  3. futures_analysis.py    — the session/edge review.

Usage:
    python3.13 scripts/futures_session_close.py          # sim reconcile + score + analysis
    python3.13 scripts/futures_session_close.py --ib     # reconcile real IB fills (Gateway up)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(label: str, argv: list[str]) -> int:
    print(f"\n\033[1m── {label} {'─' * (54 - len(label))}\033[0m")
    return subprocess.run([PY, *argv], cwd=ROOT).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command end-of-session review")
    ap.add_argument("--ib", action="store_true", help="reconcile real IB trades from fills (Gateway up)")
    args = ap.parse_args()

    recon = ["scripts/futures_reconcile.py"] + (["--from-ib"] if args.ib else [])
    steps = [
        ("RECONCILE (close the loop)", recon),
        ("CALIBRATION (score oracle call)", ["scripts/futures_calibration.py", "--oracle", "--score-today"]),
        ("ANALYSIS (session review)", ["scripts/futures_analysis.py"]),
    ]
    rc = 0
    for label, argv in steps:
        if not (ROOT / argv[0]).exists():
            print(f"  [skip] {argv[0]} not found")
            continue
        rc |= _run(label, argv)
    print(f"\n\033[1mSession close complete.\033[0m (exit {rc})")
    sys.exit(rc)


if __name__ == "__main__":
    main()
