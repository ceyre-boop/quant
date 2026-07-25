#!/usr/bin/env python3
"""Policy-rate vintage harvester (research/harvest_2026-07-25/policy_rates/).

Fetches FULL HISTORY (2015-01-01 -> present) of central-bank policy rates from
the FRED API, saves the raw JSON per series (real observation dates preserved),
and builds:
  - policy_rates_daily.csv        date,US,EU,UK,AU,JP  (forward-filled daily)
  - policy_rate_differentials.csv date,UK_US,EU_US,AU_US,US_JP
  - policy_rates_observations.csv long-format REAL observations (no fill):
                                  obs_date,country,series_id,value

Rules: no fabrication. If a series cannot be fetched or has gone stale, that is
reported, not papered over. Forward-fill is the only transformation, and the
observations file preserves every real print with its real date.

FRED_API_KEY is parsed manually from ~/quant/.env (no dotenv dependency).
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/taboost/quant")
START = "2015-01-01"

# candidate series per country, in preference order (first with usable data wins)
CANDIDATES = {
    "US": ["DFF", "FEDFUNDS"],
    "EU": ["ECBDFR"],
    "UK": ["BOERUKM", "IUDBEDR", "IRSTCB01GBM156N"],
    "AU": ["IRSTCI01AUM156N", "IR3TIB01AUM156N"],
    "JP": ["IRSTCI01JPM156N"],
}


def load_key() -> str:
    env = REPO / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("FRED_API_KEY"):
            _, _, val = line.partition("=")
            return val.strip().strip('"').strip("'")
    sys.exit("FRED_API_KEY not found in .env")


def fetch_series(series_id: str, key: str):
    """Fetch full history; save raw JSON; return list[(iso_date, float)] or None."""
    params = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": START,
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read()
    except Exception as e:  # noqa: BLE001 - report, don't fabricate
        print(f"  FETCH FAILED {series_id}: {e}")
        return None
    (HERE / f"raw_{series_id}.json").write_bytes(body)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        print(f"  BAD JSON {series_id}: {e}")
        return None
    obs = data.get("observations")
    if not obs:
        print(f"  NO OBSERVATIONS {series_id}: {str(data)[:200]}")
        return None
    out = []
    for o in obs:
        v = o.get("value", ".")
        if v in (".", "", None):
            continue  # FRED missing marker — skip, never invent
        try:
            out.append((o["date"], float(v)))
        except ValueError:
            continue
    print(f"  {series_id}: {len(out)} real obs, {out[0][0]} -> {out[-1][0]}" if out else f"  {series_id}: 0 usable obs")
    return out or None


def main() -> None:
    key = load_key()
    chosen: dict[str, tuple[str, list]] = {}
    fetched: dict[str, list | None] = {}
    failures: list[str] = []

    for country, ids in CANDIDATES.items():
        print(f"[{country}]")
        for sid in ids:
            if sid not in fetched:
                fetched[sid] = fetch_series(sid, key)
                time.sleep(0.6)  # be polite to FRED
            if fetched[sid] and country not in chosen:
                chosen[country] = (sid, fetched[sid])
        if country not in chosen:
            failures.append(country)
            print(f"  !! NO USABLE SERIES for {country} — will be empty, NOT fabricated")

    if not chosen:
        sys.exit("Nothing fetched — aborting.")

    # long-format real observations (the vintage truth file)
    obs_path = HERE / "policy_rates_observations.csv"
    with obs_path.open("w") as f:
        f.write("obs_date,country,series_id,value\n")
        for country, (sid, series) in sorted(chosen.items()):
            for d, v in series:
                f.write(f"{d},{country},{sid},{v}\n")

    # daily forward-filled panel
    end = max(date.fromisoformat(s[-1][0]) for _, s in chosen.values())
    start = date.fromisoformat(START)
    lookup = {c: dict(s) for c, (_, s) in chosen.items()}
    cols = ["US", "EU", "UK", "AU", "JP"]
    last: dict[str, float | None] = {c: None for c in cols}

    daily_path = HERE / "policy_rates_daily.csv"
    diff_path = HERE / "policy_rate_differentials.csv"
    with daily_path.open("w") as fd_, diff_path.open("w") as fx:
        fd_.write("date," + ",".join(cols) + "\n")
        fx.write("date,UK_US,EU_US,AU_US,US_JP\n")
        d = start
        while d <= end:
            iso = d.isoformat()
            for c in cols:
                if c in lookup and iso in lookup[c]:
                    last[c] = lookup[c][iso]
            fd_.write(iso + "," + ",".join("" if last[c] is None else f"{last[c]:.4f}" for c in cols) + "\n")
            if all(last[c] is not None for c in ("US", "EU", "UK", "AU", "JP")):
                fx.write(
                    f"{iso},{last['UK'] - last['US']:.4f},{last['EU'] - last['US']:.4f},"
                    f"{last['AU'] - last['US']:.4f},{last['US'] - last['JP']:.4f}\n"
                )
            d += timedelta(days=1)

    # report
    print("\n=== SERIES CHOSEN ===")
    for country, (sid, series) in sorted(chosen.items()):
        print(f"  {country}: {sid}  last real obs {series[-1][0]} = {series[-1][1]}")
    if failures:
        print(f"  FAILED COUNTRIES: {failures}")
    print(f"\nFiles: {daily_path}\n       {diff_path}\n       {obs_path}")


if __name__ == "__main__":
    main()
