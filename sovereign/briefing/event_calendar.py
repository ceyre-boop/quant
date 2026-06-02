#!/usr/bin/env python3
"""C5 — Event calendar.

Tracks high-impact scheduled macro events with size-down annotations, so the briefing can
say "ISM today — reduce size into the print" / "FOMC in N days — event risk building".

SOURCES & HONESTY:
  - FOMC meetings: a maintained SEED of known 2026 meeting dates (flagged `source:"seed"` —
    verify against the Fed's published calendar; exact dates are published a year ahead).
  - CPI / PCE / NFP / ISM: ESTIMATED from each release's typical monthly cadence (flagged
    `source:"cadence_estimate"`). Exact release dates need a real econ-calendar feed; these
    are approximate windows for sizing awareness, NOT verified timestamps.

Writes data/briefing/event_calendar.json.

Usage:  python3 -m sovereign.briefing.event_calendar [--horizon 21]
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "briefing" / "event_calendar.json"

# Maintained seed — 2026 FOMC meeting end-dates (decision day). Flagged; verify vs Fed calendar.
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth `weekday` (Mon=0..Sun=6) of a month; n=1 → first."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _first_business_day(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() >= 5:  # Sat/Sun
        d += timedelta(days=1)
    return d


def _last_business_day(year: int, month: int) -> date:
    d = date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _monthly_estimates(ref: date, months: int = 2) -> list[dict]:
    """Approximate next CPI/PCE/NFP/ISM dates from typical cadence (flagged estimates)."""
    out = []
    for i in range(months):
        m = ref.month + i
        y = ref.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        out.append({"event": "ISM Manufacturing PMI", "date": _first_business_day(y, m).isoformat(),
                    "source": "cadence_estimate", "action": "reduce size into the 10:00 ET print"})
        out.append({"event": "Nonfarm Payrolls", "date": _nth_weekday(y, m, 4, 1).isoformat(),
                    "source": "cadence_estimate", "action": "high-impact 8:30 ET — widen stops / size down"})
        out.append({"event": "CPI", "date": date(y, m, 12).isoformat(),
                    "source": "cadence_estimate", "action": "inflation print 8:30 ET — reduce size into window"})
        out.append({"event": "PCE (Fed's preferred inflation)", "date": _last_business_day(y, m).isoformat(),
                    "source": "cadence_estimate", "action": "8:30 ET — reduce size into window"})
    return out


def upcoming(horizon_days: int = 21) -> list[dict]:
    today = date.today()
    horizon = today + timedelta(days=horizon_days)
    events = [{"event": "FOMC meeting", "date": d, "source": "seed",
               "action": "reduce size / widen stops into the window"} for d in FOMC_2026]
    events += _monthly_estimates(today, months=2)

    out = []
    for e in events:
        try:
            d = date.fromisoformat(e["date"])
        except Exception:
            continue
        if today <= d <= horizon:
            days_until = (d - today).days
            out.append({**e, "days_until": days_until,
                        "note": ("TODAY — " + e["action"]) if days_until == 0
                                else f"in {days_until}d — {e['action']}"})
    out.sort(key=lambda x: x["date"])
    return out


def build(horizon_days: int = 21) -> dict:
    evs = upcoming(horizon_days)
    payload = {
        "as_of": _now(),
        "horizon_days": horizon_days,
        "events": evs,
        "provenance_note": ("FOMC dates are a maintained seed (verify vs Fed calendar); "
                            "CPI/PCE/NFP/ISM are cadence ESTIMATES, not verified release dates."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=21)
    args = ap.parse_args()
    p = build(args.horizon)
    print(f"Upcoming high-impact events (next {args.horizon}d):")
    for e in p["events"]:
        print(f"  {e['date']} ({e['source']}): {e['event']} — {e['note']}")
    print(f"  Saved: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
