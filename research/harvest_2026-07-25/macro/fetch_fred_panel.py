#!/usr/bin/env python3
"""FRED macro panel harvester — harvest_2026-07-25.

Fetches FULL history 2000-present for 12 series, saves raw JSON per series,
builds one merged wide CSV (fred_panel.csv), and pulls ALFRED real-time
vintages (latest 24 months of realtime window) for CPIAUCSL and PAYEMS so
first-print vs revised values are preserved (no-lookahead training discipline).

No fabrication: any fetch failure is logged to failures.log and the series is
omitted from the panel rather than filled.
"""

import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
VINTAGE_DIR = HERE / "vintages"
FAILURES = HERE / "failures.log"

SERIES = [
    "DFF", "DGS2", "DGS10", "T10Y2Y",
    "CPIAUCSL", "CPILFESL", "PCEPILFE",
    "UNRATE", "PAYEMS", "VIXCLS",
    "DTWEXBGS",  # broad dollar index (starts 2006-01 — full available history)
    "DCOILWTICO",  # WTI
]
VINTAGE_SERIES = ["CPIAUCSL", "PAYEMS"]

OBS_START = "2000-01-01"
BASE = "https://api.stlouisfed.org/fred/series/observations"


def load_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    env_path = Path("/Users/taboost/quant/.env")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("FRED_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("FRED_API_KEY not found in env or /Users/taboost/quant/.env")


def log_failure(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with FAILURES.open("a") as f:
        f.write(f"{ts} {msg}\n")
    print(f"FAILURE: {msg}", file=sys.stderr)


def fetch_all(params: dict, api_key: str) -> dict:
    """Fetch all observations, following FRED's 100k-row pagination."""
    merged = None
    offset = 0
    while True:
        p = dict(params)
        p.update({"api_key": api_key, "file_type": "json",
                  "limit": 100000, "offset": offset})
        r = requests.get(BASE, params=p, timeout=60)
        r.raise_for_status()
        data = r.json()
        if merged is None:
            merged = data
        else:
            merged["observations"].extend(data["observations"])
        got = offset + len(data["observations"])
        if got >= data.get("count", got):
            break
        offset = got
    merged.pop("limit", None)
    merged.pop("offset", None)
    return merged


def main() -> int:
    api_key = load_api_key()
    VINTAGE_DIR.mkdir(parents=True, exist_ok=True)
    if FAILURES.exists():
        FAILURES.unlink()

    today = date.today()
    # latest 24 months of real-time window
    rt_start = date(today.year - 2, today.month, min(today.day, 28)).isoformat()

    panel: dict[str, dict[str, str]] = {}  # series -> {date: value}
    stats: list[tuple[str, int, str]] = []

    for sid in SERIES:
        try:
            data = fetch_all({"series_id": sid, "observation_start": OBS_START}, api_key)
            obs = data.get("observations", [])
            (HERE / f"{sid}.json").write_text(json.dumps(data, indent=1))
            valid = {o["date"]: o["value"] for o in obs if o["value"] != "."}
            panel[sid] = valid
            last = max(valid) if valid else "NONE"
            stats.append((sid, len(obs), last))
            print(f"{sid:11s} rows={len(obs):6d} (non-missing={len(valid):6d}) last_obs={last}")
        except Exception as e:  # noqa: BLE001
            log_failure(f"series={sid} error={type(e).__name__}: {e}")
        time.sleep(0.6)  # stay under 120 req/min

    # ALFRED vintages: realtime window = last 24 months, all revisions preserved
    for sid in VINTAGE_SERIES:
        try:
            data = fetch_all({
                "series_id": sid,
                "observation_start": OBS_START,
                "realtime_start": rt_start,
                "realtime_end": today.isoformat(),
            }, api_key)
            obs = data.get("observations", [])
            out = VINTAGE_DIR / f"{sid}_vintages_{rt_start}_to_{today.isoformat()}.json"
            out.write_text(json.dumps(data, indent=1))
            n_rev = sum(1 for o in obs if o["realtime_start"] != rt_start)
            print(f"VINTAGE {sid:9s} rows={len(obs):6d} realtime={rt_start}..{today.isoformat()} "
                  f"(rows entering after window start: {n_rev}) -> {out.name}")
        except Exception as e:  # noqa: BLE001
            log_failure(f"vintage series={sid} error={type(e).__name__}: {e}")
        time.sleep(0.6)

    if not panel:
        log_failure("no series fetched — panel not written")
        return 1

    # merged wide CSV: union of dates, one column per series, blank = missing
    all_dates = sorted(set().union(*[set(v) for v in panel.values()]))
    cols = [s for s in SERIES if s in panel]
    with (HERE / "fred_panel.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + cols)
        for d in all_dates:
            w.writerow([d] + [panel[c].get(d, "") for c in cols])
    print(f"\nfred_panel.csv: {len(all_dates)} rows x {len(cols)} series "
          f"({all_dates[0]} .. {all_dates[-1]})")

    print("\n=== SUMMARY (series, raw rows incl. '.', last observation) ===")
    for sid, n, last in stats:
        print(f"{sid:11s} {n:6d} {last}")
    missing = [s for s in SERIES if s not in panel]
    if missing:
        print(f"MISSING SERIES (see failures.log): {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
