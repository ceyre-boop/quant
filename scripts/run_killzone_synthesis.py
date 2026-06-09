#!/usr/bin/env python3
"""
Killzone synthesis trigger — fires a Sonnet 4.6 intraday Oracle read at the current killzone.

Launched by com.alta.oracle.killzone.plist at the three institutional order-flow windows
(London 03:00 ET, NY-AM 09:30 ET, NY-PM 14:00 ET). Resolves which killzone we're in from the
ET clock, then runs futures_oracle_morning.py --killzone <KZ> for MES and MNQ. The Oracle
"thinks when the market is moving," not on a daily clock.

Usage:
  python3 scripts/run_killzone_synthesis.py              # auto-detect killzone from ET time
  python3 scripts/run_killzone_synthesis.py --killzone NY_AM   # force one (manual/backfill)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.futures_oracle_morning import KILLZONES  # {LONDON:"03:00", NY_AM:"09:30", NY_PM:"14:00"}

ET = ZoneInfo("America/New_York")
INSTRUMENTS = ["MES", "MNQ"]
MATCH_WINDOW_MIN = 45   # how close to a killzone open we must be to auto-fire


def _nearest_killzone(now_et: datetime) -> str | None:
    """The killzone whose open time is within MATCH_WINDOW_MIN of `now` (ET), else None."""
    now_min = now_et.hour * 60 + now_et.minute
    best, best_d = None, 10 ** 9
    for name, hhmm in KILLZONES.items():
        h, m = map(int, hhmm.split(":"))
        d = abs((h * 60 + m) - now_min)
        if d < best_d:
            best, best_d = name, d
    return best if best_d <= MATCH_WINDOW_MIN else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--killzone", choices=list(KILLZONES), default=None)
    args = ap.parse_args()

    now_et = datetime.now(ET)
    kz = args.killzone or _nearest_killzone(now_et)
    if kz is None:
        print(f"[killzone] {now_et:%H:%M} ET is not within {MATCH_WINDOW_MIN}m of a killzone "
              f"({KILLZONES}); nothing to synthesize.")
        return 0

    print(f"[killzone] {now_et:%Y-%m-%d %H:%M} ET -> {kz} synthesis (Sonnet) for {INSTRUMENTS}")
    rc = 0
    for inst in INSTRUMENTS:
        try:
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "futures_oracle_morning.py"),
                 "--killzone", kz, "--instrument", inst],
                cwd=str(ROOT), timeout=180,
            )
            if r.returncode != 0:
                rc = r.returncode
        except Exception as e:  # noqa: BLE001 — one instrument failing must not block the other
            print(f"  {inst}: FAILED ({type(e).__name__}: {e})")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
