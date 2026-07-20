#!/usr/bin/env python3
"""
Per-pair forex data-source health → data/health/forex_data_status.json
======================================================================

READ-ONLY. Aggregates the DEGRADED sentinels that carry_engine/macro_engine
already drop into ``sentinel/`` (TICK-025) into one per-pair status file the
dashboard and health_check.py can read.

Why this is a reader and not a patch to the fetchers: ``carry_engine`` and
``macro_engine`` are importable by the live/backtest execution path and are
under the shadow freeze (CLAUDE.md standing constraints). Adding a
MarketDataAdapter failover inside them changes what price series sizing sees,
which is an execution-path change and needs a logged unlock in NEXT.md. This
script gives the visibility half of that fix with zero execution-path risk;
the failover half stays blocked. See NEXT.md.

A pair is:
  OK        — no sentinel, or sentinel older than --stale-hours (recovered)
  DEGRADED  — fresh sentinel; macro_engine drops the pair (no fake data)
  FAKE_DATA — fresh sentinel from carry_engine's ATR fallback (0.001 stub is
              a real substituted value and does reach sizing)

RUN:
  python3 scripts/forex_data_health.py
  python3 scripts/forex_data_health.py --stale-hours 12 --quiet
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SENTINEL_DIR = ROOT / "sentinel"
OUT_PATH = ROOT / "data" / "health" / "forex_data_status.json"

# The ATR stub in carry_engine._compute_atr is a substituted value, not a skip.
_FAKE_DATA_MARKER = "ATR falls back to"


def _parse_sentinel(path: Path) -> dict:
    """Parse a DEGRADED_{source}_{PAIR}.txt key=value file."""
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            fields[k.strip()] = v.strip()
    return fields


def build_status(stale_hours: float = 24.0) -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=stale_hours)

    pairs: dict[str, dict] = {}
    for path in sorted(SENTINEL_DIR.glob("DEGRADED_*.txt")) if SENTINEL_DIR.exists() else []:
        f = _parse_sentinel(path)
        pair = f.get("pair") or path.stem.split("_")[-1]
        reason = f.get("reason", "")
        ts_raw = f.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            ts = None

        if ts is None or ts < cutoff:
            # Stale flag — the pair has fetched cleanly since. Report recovered,
            # keep the last-failure timestamp for context.
            status = "OK"
        elif _FAKE_DATA_MARKER in reason:
            status = "FAKE_DATA"
        else:
            status = "DEGRADED"

        pairs[pair] = {
            "status":       status,
            "source":       f.get("source", "unknown"),
            "reason":       reason,
            "last_failure": ts_raw or None,
            "age_hours":    round((now - ts).total_seconds() / 3600, 2) if ts else None,
            "sentinel":     str(path.relative_to(ROOT)),
        }

    bad = [p for p, d in pairs.items() if d["status"] != "OK"]
    fake = [p for p, d in pairs.items() if d["status"] == "FAKE_DATA"]
    overall = "RED" if fake else "YELLOW" if bad else "GREEN"

    return {
        "ts":              now.isoformat(),
        "overall":         overall,
        "stale_hours":     stale_hours,
        "degraded_pairs":  sorted(bad),
        "fake_data_pairs": sorted(fake),
        "pairs":           pairs,
        "note": ("FAKE_DATA means carry_engine substituted a 0.001 ATR stub that reaches "
                 "sizing. Fetcher failover is blocked by the execution-path freeze — see "
                 "the module docstring and NEXT.md."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-pair forex data-source health")
    ap.add_argument("--stale-hours", type=float, default=24.0,
                    help="Sentinels older than this are treated as recovered (default 24).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    status = build_status(args.stale_hours)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(status, indent=2))

    if not args.quiet:
        print(f"forex data health: {status['overall']} "
              f"({len(status['pairs'])} pair(s) flagged)")
        for pair, d in sorted(status["pairs"].items()):
            if d["status"] != "OK":
                print(f"  {d['status']:<9} {pair}: {d['reason']}")
        if status["fake_data_pairs"]:
            print(f"  FOREX_FETCH_FAILED: {', '.join(status['fake_data_pairs'])} "
                  f"— substituted ATR reaching sizing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
