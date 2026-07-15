#!/usr/bin/env python3
"""ICARUS T+1 coverage reconciliation — did the movers-list scan miss anything?

Runs each morning for YESTERDAY (Polygon free tier serves the full-market
grouped snapshot next-day): rebuilds the full-universe candidate set exactly
like the sealed backtest's discovery, applies the at-10:30 signal test incl.
the M&A news exclusion, and compares with what the live shadow actually took.
Output: shadow/reconcile_{date}.json + a coverage line the dashboard shows.
"""
import json
import os
import sys
import urllib.request
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.yield_frontier.live_shadow import (  # noqa: E402
    BUCKETS_MNA, bars_for, et_t, keys, OUT as SHADOW, REPO)


def polygon_key():
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("POLYGON_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no polygon key")


def grouped(day):
    url = (f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
           f"{day}?adjusted=true&apiKey={polygon_key()}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return {x["T"]: x for x in json.loads(r.read()).get("results", [])}


def prev_trading_day(d):
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def main(target=None):
    day = date.fromisoformat(target) if target else prev_trading_day(date.today())
    prev = prev_trading_day(day)
    try:
        g, p = grouped(day.isoformat()), grouped(prev.isoformat())
    except Exception as e:
        print(f"[reconcile] {day}: grouped unavailable ({e}) — skipping")
        return
    if not g or not p:
        print(f"[reconcile] {day}: empty snapshot — market holiday?")
        return
    kid, sec = keys()
    full = [t for t, row in g.items()
            if t.isalpha() and len(t) <= 5 and not (len(t) == 5 and t[-1] in "WRU")
            and p.get(t, {}).get("c") and p[t]["c"] >= 0.75
            and row.get("h") and row.get("v")
            and row["h"] >= 1.5 * p[t]["c"] and row["v"] >= 500_000]
    sig_fp = SHADOW / f"signals_{day}.json"
    taken = set()
    if sig_fp.exists():
        taken = {s["ticker"] for s in json.loads(sig_fp.read_text())["signals"]}
    qualified, mna_excl, missed = [], [], []
    for t in sorted(full):
        daily = bars_for(t, prev, kid, sec, "1Day", 14)
        bars = bars_for(t, day, kid, sec)
        if not daily or not bars:
            continue
        pc = daily[-1]["c"]
        sl = [b for b in bars if dtime(9, 30) <= et_t(b) <= dtime(10, 25)]
        if len(sl) < 8 or et_t(sl[-1]) < dtime(10, 15) or pc <= 0:
            continue
        P = sl[-1]["c"]
        if not (P / pc - 1 >= 0.50 and P >= 2.00
                and sum(b["v"] for b in sl) >= 500_000):
            continue
        import urllib.parse
        from zoneinfo import ZoneInfo
        ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
        s_ = datetime.combine(prev, dtime(16, 0), tzinfo=ET).astimezone(UTC)
        e_ = datetime.combine(day, dtime(10, 30), tzinfo=ET).astimezone(UTC)
        q = urllib.parse.urlencode({"symbols": t,
                                    "start": s_.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "end": e_.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "limit": "50"})
        req = urllib.request.Request(
            f"https://data.alpaca.markets/v1beta1/news?{q}",
            headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
        blob = " ".join(a.get("headline", "").lower() for a in
                        json.loads(urllib.request.urlopen(req, timeout=45).read())
                        .get("news", []))
        if any(k in blob for k in BUCKETS_MNA):
            mna_excl.append(t)
            continue
        qualified.append(t)
        if t not in taken:
            missed.append(t)
    doc = {"date": str(day), "full_universe_candidates": len(full),
           "qualified_at_1030": qualified, "taken": sorted(taken),
           "mna_excluded": mna_excl, "MISSED": missed,
           "coverage": "FULL" if not missed else f"GAP: {missed}"}
    SHADOW.mkdir(parents=True, exist_ok=True)
    (SHADOW / f"reconcile_{day}.json").write_text(json.dumps(doc, indent=2))
    print(f"[reconcile] {day}: {len(full)} candidates -> qualified "
          f"{qualified} | taken {sorted(taken)} | M&A-excl {mna_excl} | "
          f"MISSED {missed or 'NONE'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
