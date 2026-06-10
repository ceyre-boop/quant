#!/usr/bin/env python3
"""Build the static high-impact economic event calendar for the ES/NQ system.

Output: data/es_nq/econ_calendar_2018_2026.json
  {"YYYY-MM-DD": {"events": ["FOMC"|"CPI"|"NFP", ...]}, "_meta": {...}}

Sources (Amendment A1, data/research/es_nq_preregistration.json):
  FOMC — hand-tabulated decision days from the Fed's published meeting calendars
         (second day of each scheduled meeting; 2020 includes the two emergency
         actions of Mar 3 and Mar 15 which replaced the cancelled Mar 18 meeting).
  CPI  — actual historical release dates from the FRED release-dates API
         (release_id=10, Consumer Price Index). Needs FRED_API_KEY in .env.
  NFP  — actual historical release dates from FRED (release_id=50, Employment
         Situation). No first-Friday approximation needed.

Fails loud on any missing key or short response — never silently degrades.
Usage: python3 scripts/build_econ_calendar_es_nq.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "es_nq" / "econ_calendar_2018_2026.json"
START, END = "2018-01-01", "2026-12-31"

# Fed's published meeting calendars — decision (statement) days, public record.
FOMC_DECISION_DAYS = [
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (Mar 3 + Mar 15 were emergency actions; scheduled Mar 18 was cancelled)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (published schedule)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

FRED_RELEASES = {"CPI": 10, "NFP": 50}  # Consumer Price Index, Employment Situation


def _fred_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("FRED_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("FATAL: FRED_API_KEY not set (env or .env) — cannot build calendar")
    return key


def fred_release_dates(release_id: int, key: str) -> list[str]:
    """All release dates for a FRED release in [START, END], paginated, fail-loud."""
    dates: list[str] = []
    offset = 0
    while True:
        qs = urllib.parse.urlencode({
            "release_id": release_id, "api_key": key, "file_type": "json",
            "realtime_start": START, "realtime_end": END,
            "include_release_dates_with_no_data": "true",
            "limit": 1000, "offset": offset,
        })
        url = f"https://api.stlouisfed.org/fred/release/dates?{qs}"
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.loads(r.read())
        batch = [d["date"] for d in payload.get("release_dates", [])]
        dates.extend(d for d in batch if START <= d <= END)
        if len(payload.get("release_dates", [])) < 1000:
            break
        offset += 1000
    if not dates:
        raise SystemExit(f"FATAL: FRED returned zero release dates for release_id={release_id}")
    return sorted(set(dates))


def main() -> None:
    key = _fred_key()
    calendar: dict[str, dict] = defaultdict(lambda: {"events": []})

    for d in FOMC_DECISION_DAYS:
        calendar[d]["events"].append("FOMC")
    for name, rid in FRED_RELEASES.items():
        rel = fred_release_dates(rid, key)
        print(f"{name}: {len(rel)} release dates from FRED (release_id={rid})")
        for d in rel:
            calendar[d]["events"].append(name)

    per_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d, v in calendar.items():
        for ev in v["events"]:
            per_year[d[:4]][ev] += 1
    for yr in sorted(per_year):
        counts = dict(per_year[yr])
        print(f"  {yr}: {counts}")
        # Sanity gates — fail loud if a source is obviously broken (full years only).
        if yr < "2026":
            if not 10 <= counts.get("CPI", 0) <= 14:
                raise SystemExit(f"FATAL: {yr} CPI count {counts.get('CPI', 0)} outside [10, 14]")
            if not 10 <= counts.get("NFP", 0) <= 14:
                raise SystemExit(f"FATAL: {yr} NFP count {counts.get('NFP', 0)} outside [10, 14]")
            expected_fomc = 9 if yr == "2020" else 8
            if counts.get("FOMC", 0) != expected_fomc:
                raise SystemExit(f"FATAL: {yr} FOMC count {counts.get('FOMC', 0)} != {expected_fomc}")

    out = dict(sorted(calendar.items()))
    out["_meta"] = {
        "built_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "range": [START, END],
        "sources": {
            "FOMC": "hand-tabulated Fed meeting calendars (decision days; 2020 incl. Mar 3 + Mar 15 emergency actions)",
            "CPI": "FRED release-dates API, release_id=10",
            "NFP": "FRED release-dates API, release_id=50 (Employment Situation)",
        },
        "amendment": "A1 — see data/research/es_nq_preregistration.json",
        "per_year_counts": {yr: dict(v) for yr, v in sorted(per_year.items())},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=1))
    n_days = len(out) - 1
    print(f"Wrote {OUT_PATH} ({n_days} event days)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
