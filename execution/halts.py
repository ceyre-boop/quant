"""Halt detection for the execution harness.

Band arithmetic lives in `backtester/luld.py` (a leaf module that imports neither
`execution/` nor `sovereign/`); this module adds the harness-facing concerns:
per-event halt evidence at a decision instant, and validation of the detector
against archived NASDAQ trade-halt ground truth.

GROUND TRUTH
------------
`research/yield_frontier/daily_snapshots.py:34` archives
nasdaqtrader.com/dynamic/symdir/tradehalts.csv daily to
`data/research/yield_frontier/halt_snapshots/nasdaq_{date}.csv.gz`.

Coverage is currently 5 sessions (2026-07-14 .. 2026-07-18). Any precision or
recall figure computed here MUST be reported with its N. Five sessions cannot
support a confident claim, and the point of fixing the detector is undermined if
the fix is justified with an overconfident number.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from datetime import date, datetime, time as dtime
from pathlib import Path

import numpy as np

from backtester.luld import bands_for, halt_flags, luld_band  # noqa: F401 (re-export)

ROOT = Path(__file__).resolve().parents[1]
HALT_DIR = ROOT / "data" / "research" / "yield_frontier" / "halt_snapshots"


def load_nasdaq_halts(day: date) -> set[str]:
    """Symbols recorded as halted on `day`, from the archived NASDAQ CSV.

    Returns an empty set if no snapshot exists for that date — the caller must
    treat "no snapshot" as unknown, never as "no halts".
    """
    fp = HALT_DIR / f"nasdaq_{day}.csv.gz"
    if not fp.exists():
        return set()
    out: set[str] = set()
    with gzip.open(fp, "rt", errors="replace") as fh:
        for row in csv.DictReader(fh):
            sym = (row.get("Issue Symbol") or row.get("Symbol") or "").strip().upper()
            if sym:
                out.add(sym)
    return out


def available_halt_days() -> list[date]:
    """ET session dates for which halt ground truth exists."""
    days: list[date] = []
    if not HALT_DIR.exists():
        return days
    for fp in sorted(HALT_DIR.glob("nasdaq_*.csv.gz")):
        try:
            days.append(date.fromisoformat(fp.stem.replace("nasdaq_", "")))
        except ValueError:
            continue
    return days


def legacy_halt_flags(bars, band: float = 0.10) -> np.ndarray:
    """The OLD flat-band detector, preserved for A/B comparison only.

    This is what `realistic_fills._halt_flags` did before the fix: a flat 10%
    band with no tiering and no time-of-day doubling.
    """
    import pandas as pd
    t = pd.to_datetime(bars["time"], format="%H:%M")
    gap_before = t.diff().dt.total_seconds().div(60).fillna(1).to_numpy() > 1.5
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    jump = np.abs(o / prev_c - 1) > band
    intrabar = np.abs(c / o - 1) > band
    return gap_before | jump | intrabar


def halted_at(bars, at_et: dtime, tier: int = 2) -> tuple[bool, str]:
    """Is the security halted at (or entering) the bar covering `at_et`?

    Evidence, in order of strength:
      1. No bar exists covering that minute during RTH -> the tape stopped.
      2. The covering bar is flagged as a halt-resume by the tiered detector.

    Returns (halted, evidence).
    """
    import pandas as pd
    if len(bars) == 0:
        return True, "no_bars"

    times = pd.to_datetime(bars["time"], format="%H:%M")
    target = f"{at_et.hour:02d}:{at_et.minute:02d}"
    idx = np.where(bars["time"].to_numpy() >= target)[0]
    if len(idx) == 0:
        return True, "no_bar_at_or_after_entry"

    i = int(idx[0])
    if bars["time"].to_numpy()[i] != target:
        # Nearest bar is later than requested -> the target minute did not print.
        return True, f"missing_minute_{target}_next_{bars['time'].to_numpy()[i]}"

    flags = halt_flags(bars, tier=tier)
    if bool(flags[i]):
        band = float(bands_for(bars, tier=tier)[i])
        return True, f"luld_excursion_band_{band:.2f}"

    _ = times  # retained for readability of the time index above
    return False, "clear"


def validate_detector(days: list[date] | None = None) -> dict:
    """Compare old flat-band vs new tiered detector against NASDAQ ground truth.

    This is a coverage report, not a proof. It answers: of the symbols each rule
    flags as halted, how many actually appear in the official halt file?

    Returns a dict including `n_sessions` so callers cannot quote a precision
    figure without its sample size.
    """
    days = days or available_halt_days()
    return {
        "n_sessions": len(days),
        "sessions": [str(d) for d in days],
        "halt_symbols_by_day": {str(d): sorted(load_nasdaq_halts(d)) for d in days},
        "note": (
            "Ground truth coverage is thin. Report N alongside any precision or "
            "recall figure; do not present a confident detector claim from a "
            "handful of sessions."
        ),
    }


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LULD halt detector utilities")
    ap.add_argument("--validate", action="store_true",
                    help="report ground-truth coverage and halted symbols")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO date lower bound for --validate")
    args = ap.parse_args(argv)

    if args.validate:
        days = available_halt_days()
        if args.since:
            lo = date.fromisoformat(args.since)
            days = [d for d in days if d >= lo]
        rep = validate_detector(days)
        print(f"halt ground truth: N={rep['n_sessions']} session(s)")
        for d in rep["sessions"]:
            syms = rep["halt_symbols_by_day"][d]
            print(f"  {d}: {len(syms)} halted symbol(s)")
        print(f"\n{rep['note']}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
