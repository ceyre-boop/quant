#!/usr/bin/env python3
"""HYP-092 stage 2 — at-10:30 filters, frozen CONT/EX/UNC read, outcomes.

All inputs from ONE source+basis: Alpaca v2 bars, feed=sip, adjustment=split
(both 1Day and 5Min — probe-verified). Read/filter inputs use ONLY bars whose
America/New_York start time is in [09:30, 10:25] (complete by 10:30). Outcome
origin is the OPEN of the first bar starting in [10:30, 11:00) — disjoint from
read inputs. Outcome end is the close of the last bar starting before 16:00 ET.

Implements data/research/preregister/HYP-092_gapper_continuation.json verbatim.
No thresholds live here that are not in the prereg.
"""
import gzip
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/research/gapper"
CACHE = OUT / "cache/alpaca"
BASE = "https://data.alpaca.markets/v2/stocks/bars"

# prereg-locked constants
GAIN_MIN = 1.30
PRICE_MIN = 2.00
VOL_MIN = 500_000
SLICE_START, SLICE_END = dtime(9, 30), dtime(10, 25)
LAST_BAR_MIN = dtime(10, 15)
MIN_SLICE_BARS = 8
OR_END = dtime(9, 55)
LH_START = dtime(10, 0)
ENTRY_LO, ENTRY_HI = dtime(10, 30), dtime(11, 0)
RTH_END = dtime(16, 0)
CLIMAX_LATEST = dtime(10, 10)


def load_keys():
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"]


def api_get(url: str, kid: str, sec: str) -> dict:
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
    for attempt in range(6):
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
    raise RuntimeError(f"gave up: {url[:120]}")


def fetch_bars(symbols, start_utc, end_utc, timeframe, kid, sec):
    """Multi-symbol bar fetch, follows pagination. Returns {sym: [bars]}."""
    out = defaultdict(list)
    token = None
    while True:
        params = {
            "symbols": ",".join(symbols), "timeframe": timeframe,
            "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feed": "sip", "adjustment": "split", "limit": "10000",
        }
        if token:
            params["page_token"] = token
        d = api_get(BASE + "?" + urllib.parse.urlencode(params), kid, sec)
        for s, bars in (d.get("bars") or {}).items():
            out[s].extend(bars)
        token = d.get("next_page_token")
        if not token:
            return out
        time.sleep(0.25)


def day_payload(day: str, symbols: list, kid: str, sec: str) -> dict:
    """Fetch (cached) intraday 5Min RTH bars + prior-14-calendar-day daily bars."""
    fp = CACHE / f"{day}.json.gz"
    if fp.exists():
        with gzip.open(fp, "rt") as f:
            return json.load(f)
    d = date.fromisoformat(day)
    rth_open = datetime.combine(d, dtime(9, 30), tzinfo=ET).astimezone(UTC)
    rth_close = datetime.combine(d, dtime(16, 10), tzinfo=ET).astimezone(UTC)
    prev_start = datetime.combine(d - timedelta(days=14), dtime(0, 0), tzinfo=ET).astimezone(UTC)
    prev_end = datetime.combine(d - timedelta(days=1), dtime(23, 59), tzinfo=ET).astimezone(UTC)
    intraday, daily = {}, {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        intraday.update(fetch_bars(chunk, rth_open, rth_close, "5Min", kid, sec))
        daily.update(fetch_bars(chunk, prev_start, prev_end, "1Day", kid, sec))
        time.sleep(0.3)
    payload = {"date": day, "intraday": intraday, "daily": daily}
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(fp, "wt") as f:
        json.dump(payload, f)
    return payload


def et_start(bar) -> dtime:
    ts = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
    return ts.astimezone(ET).time()


def classify(slice_bars):
    """Frozen prereg read rules. Returns (read, votes dict)."""
    P = slice_bars[-1]["c"]
    H = max(b["h"] for b in slice_bars)
    L = min(b["l"] for b in slice_bars)
    vsum = sum(b["v"] for b in slice_bars)
    vwap = sum(b["vw"] * b["v"] for b in slice_bars) / vsum if vsum > 0 else None
    or_bars = [b for b in slice_bars if et_start(b) <= OR_END]
    lh_bars = [b for b in slice_bars if et_start(b) >= LH_START]
    up_v = sum(b["v"] for b in slice_bars if b["c"] > b["o"])
    dn_v = sum(b["v"] for b in slice_bars if b["c"] < b["o"])

    c1 = vwap is not None and P >= vwap
    c2 = bool(or_bars and lh_bars) and min(b["l"] for b in lh_bars) >= min(b["l"] for b in or_bars)
    c3 = up_v > dn_v
    c4 = P >= L + 0.5 * (H - L)

    e1 = vwap is not None and P < vwap
    e2 = bool(or_bars and lh_bars) and max(b["h"] for b in lh_bars) < max(b["h"] for b in or_bars)
    maxv = max(b["v"] for b in slice_bars)
    argmax_bar = next(b for b in slice_bars if b["v"] == maxv)
    last3 = slice_bars[-3:]
    e3 = et_start(argmax_bar) <= CLIMAX_LATEST and (sum(b["v"] for b in last3) / 3) < 0.5 * maxv
    hbar = next(b for b in slice_bars if b["h"] == H)
    rng = hbar["h"] - hbar["l"]
    e4 = rng > 0 and (hbar["h"] - max(hbar["o"], hbar["c"])) / rng > 0.6

    cont, ex = sum([c1, c2, c3, c4]), sum([e1, e2, e3, e4])
    if cont >= 3 and ex <= 1:
        read = "CONT"
    elif ex >= 3 and cont <= 1:
        read = "EX"
    else:
        read = "UNC"
    votes = {"C1": c1, "C2": c2, "C3": c3, "C4": c4,
             "E1": e1, "E2": e2, "E3": e3, "E4": e4}
    return read, votes, P, H, L, vwap, vsum


def process_candidate(tkr, bars, daily_bars, counters):
    if not daily_bars:
        counters["no_prev_close"] += 1
        return None
    prev_close = daily_bars[-1]["c"]
    if prev_close <= 0:
        counters["no_prev_close"] += 1
        return None
    slice_bars = [b for b in bars if SLICE_START <= et_start(b) <= SLICE_END]
    if len(slice_bars) < MIN_SLICE_BARS or (
            not slice_bars or et_start(slice_bars[-1]) < LAST_BAR_MIN):
        counters["unreadable_sparse"] += 1
        # descriptive outcome of excluded set where computable (disclosure)
        post = [b for b in bars if ENTRY_LO <= et_start(b) < RTH_END]
        if slice_bars and post:
            counters.setdefault("sparse_outcomes", []).append(
                round(post[-1]["c"] / slice_bars[-1]["c"] - 1, 4))
        return None
    read, votes, P, H, L, vwap, cumvol = classify(slice_bars)
    passes = (P >= GAIN_MIN * prev_close) and (P >= PRICE_MIN) and (cumvol >= VOL_MIN)
    if not passes:
        counters["failed_1030_filters"] += 1
        return None
    entry_bars = [b for b in bars if ENTRY_LO <= et_start(b) < ENTRY_HI]
    if not entry_bars:
        counters["unreadable_at_entry"] += 1
        return None
    entry = entry_bars[0]["o"]
    rth_bars = [b for b in bars if et_start(b) < RTH_END]
    outcome_end = rth_bars[-1]["c"]
    return {
        "ticker": tkr, "prev_close": round(prev_close, 4),
        "price_1030": round(P, 4), "gain_1030": round(P / prev_close - 1, 4),
        "cum_vol_1030": int(cumvol), "vwap_1030": round(vwap, 4),
        "read": read, **{k: int(v) for k, v in votes.items()},
        "entry_open_1030": round(entry, 4), "close_eod": round(outcome_end, 4),
        "outcome_pct": round(outcome_end / entry - 1, 4),
        "n_slice_bars": len(slice_bars),
    }


def main():
    kid, sec = load_keys()
    cands = defaultdict(list)
    with open(OUT / "candidates.csv") as f:
        next(f)
        for line in f:
            d, t, pc, hi, gain, vol = line.strip().split(",")
            cands[d].append((t, float(gain)))
    active = set(json.loads((OUT / "active_proxy.json").read_text()))
    counters = defaultdict(int)
    rows = []
    days = sorted(cands)
    for i, day in enumerate(days):
        symbols = sorted(t for t, _ in cands[day])
        gains = dict(cands[day])
        payload = day_payload(day, symbols, kid, sec)
        for tkr in symbols:
            bars = payload["intraday"].get(tkr, [])
            if not bars:
                counters["no_intraday_data"] += 1
                counters["no_intraday_delisted" if tkr not in active
                         else "no_intraday_active"] += 1
                continue
            row = process_candidate(tkr, bars, payload["daily"].get(tkr, []), counters)
            if row:
                row["date"] = day
                row["active_proxy"] = int(tkr in active)
                row["boundary_leak_zone"] = int(0.20 <= gains[tkr] < 0.35)
                rows.append(row)
        if i % 25 == 0:
            print(f"[stage2] {i+1}/{len(days)} days, {len(rows)} qualifying so far",
                  flush=True)
    rows.sort(key=lambda r: (r["date"], r["ticker"]))
    cols = ["date", "ticker", "prev_close", "price_1030", "gain_1030",
            "cum_vol_1030", "vwap_1030", "read", "C1", "C2", "C3", "C4",
            "E1", "E2", "E3", "E4", "entry_open_1030", "close_eod",
            "outcome_pct", "n_slice_bars", "active_proxy", "boundary_leak_zone"]
    with open(OUT / "per_candidate.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    sparse = counters.pop("sparse_outcomes", [])
    guards = dict(counters)
    guards["sparse_excluded_outcome_median"] = (
        sorted(sparse)[len(sparse) // 2] if sparse else None)
    guards["sparse_excluded_outcome_n"] = len(sparse)
    guards["qualifying_rows"] = len(rows)
    (OUT / "stage2_guards.json").write_text(json.dumps(guards, indent=2))
    print(f"[stage2] done: {len(rows)} qualifying rows; guards: {guards}", flush=True)


if __name__ == "__main__":
    main()
