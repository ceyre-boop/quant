#!/usr/bin/env python3
"""G1 — the ONLY sanctioned holdout fetch. Refuses to run unless HYP-093/094/095
prereg hashes verify and their ledger entries are PREREGISTERED (gate-zero).

Fetches 2024-07-01→2025-06-30: Polygon grouped daily → candidate scan →
Alpaca SIP stage-2 (intraday+daily) → Alpaca news (for the M&A exclusion).
Everything lands under data/research/yield_frontier/holdout_equities/ — never
in the mining caches. Chunked: --max-dates N per session law.
Run: python3 -m research.yield_frontier.g1_fetch_equities_holdout [--max-dates N]
"""
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ._lib import REPO, canonical_hash, sha256_file

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
HD = REPO / "data/research/yield_frontier/holdout_equities"
START, END = date(2024, 7, 1), date(2025, 6, 30)
BUFFER_GAIN, MIN_PREV_CLOSE, MIN_DAY_VOL = 1.20, 0.75, 500_000


def gate_zero():
    ledger = json.loads((REPO / "data/agent/hypothesis_ledger.json").read_text())
    for hyp in ("HYP-093", "HYP-094", "HYP-095"):
        fp = REPO / f"data/research/preregister/{hyp}_yield_frontier.json"
        doc = json.loads(fp.read_text())
        lock = doc.pop("hash_lock")
        if canonical_hash(doc) != lock:
            raise SystemExit(f"GATE-ZERO FAIL: {hyp} hash broken")
        entry = next((e for e in ledger if e.get("id") == hyp), None)
        if entry is None or entry.get("status") != "PREREGISTERED":
            raise SystemExit(f"GATE-ZERO FAIL: {hyp} ledger status "
                             f"{entry.get('status') if entry else 'MISSING'}")
    print("[G1] gate-zero OK — all three preregs locked + PREREGISTERED")


def load_env(name):
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"missing {name}")


def weekdays():
    d = START
    while d <= END:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def fetch_grouped(max_dates=None):
    key = load_env("POLYGON_API_KEY")
    gdir = HD / "grouped"
    gdir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in weekdays() if not (gdir / f"{d.isoformat()}.json.gz").exists()]
    if max_dates:
        todo = todo[:max_dates]
    print(f"[G1] grouped: {len(todo)} dates this chunk", flush=True)
    for i, d in enumerate(todo):
        url = (f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
               f"{d.isoformat()}?adjusted=true&apiKey={key}")
        j = None
        for _ in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    j = json.loads(r.read())
                if j.get("status") in ("OK", "DELAYED"):
                    break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(20)
                    continue
                if e.code == 403:   # free tier: 2-year lookback wall (dates age in daily)
                    j = {"results": [], "denied_403": True}
                    break
                raise
            except Exception:
                time.sleep(15)
        if j is None:
            raise RuntimeError(f"gave up on {d}")
        if j.get("denied_403"):
            with gzip.open(gdir / f"{d.isoformat()}.json.gz", "wt") as f:
                json.dump({"date": d.isoformat(), "n": 0, "denied_403": True}, f)
            print(f"[G1] {d} DENIED (outside 2y lookback) — partial-window clause",
                  flush=True)
            continue
        slim = {row["T"]: [row.get("o"), row.get("h"), row.get("l"),
                           row.get("c"), row.get("v")] for row in j.get("results", [])}
        with gzip.open(gdir / f"{d.isoformat()}.json.gz", "wt") as f:
            json.dump({"date": d.isoformat(), "n": len(slim), "bars": slim}, f)
        if i % 10 == 0:
            print(f"[G1] {i+1}/{len(todo)} ({d})", flush=True)
        time.sleep(12.5)
    remaining = sum(1 for d in weekdays()
                    if not (gdir / f"{d.isoformat()}.json.gz").exists())
    print(f"[G1] grouped chunk done; {remaining} dates remain", flush=True)
    return remaining


def ticker_ok(t):
    return t.isalpha() and len(t) <= 5 and not (len(t) == 5 and t[-1] in "WRU")


def scan_candidates():
    files = sorted((HD / "grouped").glob("*.json.gz"))
    prev, rows = None, []
    for fp in files:
        with gzip.open(fp, "rt") as f:
            day = json.load(f)
        if day["n"] == 0:
            continue
        if prev is not None:
            for t, (o, h, l, c, v) in day["bars"].items():
                if not ticker_ok(t) or t not in prev["bars"]:
                    continue
                pc = prev["bars"][t][3]
                if pc is None or pc < MIN_PREV_CLOSE or h is None or v is None:
                    continue
                if h >= BUFFER_GAIN * pc and v >= MIN_DAY_VOL:
                    rows.append((day["date"], t, round(pc, 4)))
        prev = day
    with open(HD / "candidates.csv", "w") as f:
        f.write("date,ticker,prev_close_polygon\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"[G1] {len(rows)} holdout candidates", flush=True)


def api_get(url, kid, sec):
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
    for _ in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10)
                continue
            raise
        except Exception:
            time.sleep(5)
    raise RuntimeError(url[:120])


def fetch_bars(symbols, s_utc, e_utc, tf, kid, sec):
    out, token = defaultdict(list), None
    while True:
        params = {"symbols": ",".join(symbols), "timeframe": tf,
                  "start": s_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": e_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "feed": "sip", "adjustment": "split", "limit": "10000"}
        if token:
            params["page_token"] = token
        d = api_get("https://data.alpaca.markets/v2/stocks/bars?" +
                    urllib.parse.urlencode(params), kid, sec)
        for s, bars in (d.get("bars") or {}).items():
            out[s].extend(bars)
        token = d.get("next_page_token")
        if not token:
            return out
        time.sleep(0.25)


def fetch_stage2_and_news():
    kid, sec = load_env("ALPACA_API_KEY"), load_env("ALPACA_SECRET_KEY")
    cands = defaultdict(list)
    with open(HD / "candidates.csv") as f:
        next(f)
        for line in f:
            d, t, pc = line.strip().split(",")
            cands[d].append(t)
    adir, ndir = HD / "alpaca", HD / "news"
    adir.mkdir(exist_ok=True), ndir.mkdir(exist_ok=True)
    days = sorted(cands)
    for i, day in enumerate(days):
        if not (adir / f"{day}.json.gz").exists():
            d = date.fromisoformat(day)
            symbols = sorted(cands[day])
            intraday, daily = {}, {}
            for j in range(0, len(symbols), 50):
                ch = symbols[j:j + 50]
                intraday.update(fetch_bars(
                    ch, datetime.combine(d, dtime(9, 30), tzinfo=ET).astimezone(UTC),
                    datetime.combine(d, dtime(16, 10), tzinfo=ET).astimezone(UTC),
                    "5Min", kid, sec))
                daily.update(fetch_bars(
                    ch, datetime.combine(d - timedelta(days=14), dtime(0, 0), tzinfo=ET).astimezone(UTC),
                    datetime.combine(d - timedelta(days=1), dtime(23, 59), tzinfo=ET).astimezone(UTC),
                    "1Day", kid, sec))
                time.sleep(0.3)
            with gzip.open(adir / f"{day}.json.gz", "wt") as f:
                json.dump({"date": day, "intraday": intraday, "daily": daily}, f)
        if not (ndir / f"{day}.json").exists():
            d = date.fromisoformat(day)
            s_utc = datetime.combine(d - timedelta(days=1), dtime(16, 0), tzinfo=ET).astimezone(UTC)
            e_utc = datetime.combine(d, dtime(10, 30), tzinfo=ET).astimezone(UTC)
            per = defaultdict(list)
            symbols = sorted(cands[day])
            for j in range(0, len(symbols), 40):
                ch = symbols[j:j + 40]
                token = None
                while True:
                    params = {"symbols": ",".join(ch),
                              "start": s_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "end": e_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": "50"}
                    if token:
                        params["page_token"] = token
                    r = api_get("https://data.alpaca.markets/v1beta1/news?" +
                                urllib.parse.urlencode(params), kid, sec)
                    for a in r.get("news", []):
                        for s in a.get("symbols", []):
                            per[s].append(a.get("headline", ""))
                    token = r.get("next_page_token")
                    if not token:
                        break
                    time.sleep(0.2)
                time.sleep(0.2)
            (ndir / f"{day}.json").write_text(json.dumps(per))
        if i % 25 == 0:
            print(f"[G1] stage2+news {i+1}/{len(days)}", flush=True)
    manifest = {str(p.relative_to(HD)): sha256_file(p)
                for p in sorted(HD.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (HD / "manifest.json").write_text(json.dumps(
        {"generated": datetime.now(UTC).isoformat(), "files": manifest}, indent=2))
    print(f"[G1] complete; manifest {len(manifest)} files", flush=True)


if __name__ == "__main__":
    gate_zero()
    n = None
    if "--max-dates" in sys.argv:
        n = int(sys.argv[sys.argv.index("--max-dates") + 1])
    if "--stage2-only" not in sys.argv:
        remaining = fetch_grouped(n)
        if remaining:
            sys.exit(0)
    scan_candidates()
    fetch_stage2_and_news()
