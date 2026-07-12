#!/usr/bin/env python3
"""HYP-092 stage 1 — survivorship-free gapper discovery via Polygon grouped daily.

Fetches one grouped-daily snapshot per trading day (free tier: 5 req/min, so this
runs long — cache is resumable, rerun skips fetched dates), then scans the cache
for candidate ticker-days per the prereg's BUFFERED superset filter:

    prev grouped close exists AND >= $0.75
    day high >= 1.20 * prev close        (buffer under the 1.30 stage-2 filter)
    day volume >= 500,000                (superset of by-10:30 volume)
    ticker alphabetic, len<=5, not 5-letter ending W/R/U

Writes: cache/grouped/{date}.json.gz (NOT committed), candidates.csv (committed).
Prereg: data/research/preregister/HYP-092_gapper_continuation.json (hash-locked).
"""
import gzip
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/research/gapper"
CACHE = OUT / "cache/grouped"
WINDOW_START = date(2025, 7, 1)
WINDOW_END = date(2026, 6, 30)
BUFFER_GAIN = 1.20
MIN_PREV_CLOSE = 0.75
MIN_DAY_VOL = 500_000


def load_env_key(name: str) -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"missing {name} in .env")


def weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def fetch_day(day: date, key: str) -> dict:
    url = (f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
           f"{day.isoformat()}?adjusted=true&apiKey={key}")
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.loads(r.read())
            if d.get("status") in ("OK", "DELAYED"):
                slim = {row["T"]: [row.get("o"), row.get("h"), row.get("l"),
                                   row.get("c"), row.get("v")]
                        for row in d.get("results", [])}
                return {"date": day.isoformat(), "n": len(slim), "bars": slim}
            raise RuntimeError(f"status={d.get('status')} {d.get('error','')[:80]}")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20)
                continue
            raise
        except Exception:
            time.sleep(15)
    raise RuntimeError(f"gave up on {day}")


def run_fetch() -> None:
    key = load_env_key("POLYGON_API_KEY")
    CACHE.mkdir(parents=True, exist_ok=True)
    days = list(weekdays(WINDOW_START, WINDOW_END))
    todo = [d for d in days if not (CACHE / f"{d.isoformat()}.json.gz").exists()]
    print(f"[stage1] {len(days)} weekdays in window, {len(todo)} to fetch", flush=True)
    for i, d in enumerate(todo):
        payload = fetch_day(d, key)
        with gzip.open(CACHE / f"{d.isoformat()}.json.gz", "wt") as f:
            json.dump(payload, f)
        if i % 10 == 0:
            print(f"[stage1] {i+1}/{len(todo)} fetched ({d}, {payload['n']} tickers)",
                  flush=True)
        time.sleep(12.5)  # 5 req/min free tier
    print("[stage1] fetch complete", flush=True)


def ticker_ok(t: str) -> bool:
    if not t.isalpha() or len(t) > 5:
        return False
    if len(t) == 5 and t[-1] in "WRU":
        return False
    return True


def run_scan() -> None:
    files = sorted(CACHE.glob("*.json.gz"))
    prev: dict | None = None
    rows = []
    trading_days = []
    for fp in files:
        with gzip.open(fp, "rt") as f:
            day = json.load(f)
        if day["n"] == 0:  # holiday
            continue
        trading_days.append(day["date"])
        if prev is not None:
            for t, (o, h, l, c, v) in day["bars"].items():
                if not ticker_ok(t) or t not in prev["bars"]:
                    continue  # no prev close -> IPO-day exclusion (counted implicitly)
                pc = prev["bars"][t][3]
                if pc is None or pc < MIN_PREV_CLOSE or h is None or v is None:
                    continue
                if h >= BUFFER_GAIN * pc and v >= MIN_DAY_VOL:
                    rows.append((day["date"], t, round(pc, 4), round(h, 4),
                                 round(h / pc - 1, 4), int(v)))
        prev = day
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "candidates.csv", "w") as f:
        f.write("date,ticker,prev_close_polygon,day_high,stage1_gain,day_volume\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    # active-proxy set: tickers seen in the final 10 trading days of the window
    active = set()
    for d in trading_days[-10:]:
        with gzip.open(CACHE / f"{d}.json.gz", "rt") as f:
            active |= set(json.load(f)["bars"].keys())
    (OUT / "active_proxy.json").write_text(json.dumps(sorted(active)))
    print(f"[stage1] {len(rows)} candidate ticker-days over {len(trading_days)} "
          f"trading days; active-proxy set {len(active)} tickers", flush=True)


if __name__ == "__main__":
    if "--scan-only" not in sys.argv:
        run_fetch()
    run_scan()
