#!/usr/bin/env python3
"""Options-chain tradeability audit on the confirmed gapper universe.

Gate for the call-overlay strategy: at 09:31 (the entry), does a tradeable ATM
CALL exist? Tradeable = two-sided quote AND spread <= 20% of mid. Broken down by
underlying price tier (the likely dominant factor). ThetaData options-VALUE tier
via ThetaTerminal :25503 (proven to serve historical intraday option quotes).

>=30% tradeable -> door open. <30% -> door closed (per-tier reported too).
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:25503/v3"
OUT = REPO / "data/research/gapper/options_tradeability_0931.json"
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]
SPREAD_CAP = 0.20   # tradeable if (ask-bid)/mid <= 20%


def get(path, **params):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                txt = r.read().decode()
            if txt.startswith("No data"):
                return []
            return list(csv.DictReader(txt.splitlines()))
        except Exception:
            time.sleep(1 + attempt)
    return None


def underlying_0931(ticker, d):
    from backtester import data as _d
    b = _d.get_minute_bars(ticker, d)
    if b is None or not len(b):
        return None
    b = b.loc[:, ~b.columns.duplicated()]
    t = b["time"].to_numpy()
    w = np.where(t >= "09:31")[0]
    return float(b.iloc[w[0]]["open"]) if len(w) else None


def call_quote_0931(sym, exp, strike, d):
    """Nearest two-sided ATM call quote to 09:31 from 1-min quotes 09:30-09:36."""
    rows = get("option/history/quote", symbol=sym, expiration=exp,
               strike=strike, right="C", start_date=d, end_date=d, interval="1m")
    if not rows:
        return None
    best = None
    for row in rows:
        ts = row.get("timestamp", "")
        hhmm = ts[11:16] if len(ts) >= 16 else ""
        if not ("09:30" <= hhmm <= "09:36"):
            continue
        try:
            b, a = float(row["bid"]), float(row["ask"])
        except (ValueError, KeyError):
            continue
        if b <= 0 or a <= 0:
            continue
        rec = {"bid": b, "ask": a, "hhmm": hhmm,
               "oi": _num(row.get("open_interest")),
               "vol": _num(row.get("volume"))}
        # prefer the bar closest to 09:31
        if best is None or abs(_mins(hhmm) - 571) < abs(_mins(best["hhmm"]) - 571):
            best = rec
    return best


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mins(hhmm):
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def tier(price):
    if price is None:
        return "unknown"
    if price < 1:
        return "<$1"
    if price < 5:
        return "$1-5"
    if price < 20:
        return "$5-20"
    return "$20+"


def main():
    csv_path = REPO / "data/research/gapper/per_candidate_enriched.csv"
    events = []
    for r in csv.DictReader(open(csv_path)):
        try:
            g, p, v = (float(r["gain_1030"]), float(r["price_1030"]),
                       float(r["cum_vol_1030"]))
        except (ValueError, KeyError):
            continue
        if g < 1.0 or p < 2.0 or v < 500_000:
            continue
        if any(m in (r.get("catalyst") or "").lower() for m in MNA):
            continue
        events.append(r)
    print(f"events: {len(events)}")

    results = []
    for i, r in enumerate(events):
        d, sym = r["date"], r["ticker"]
        S = underlying_0931(sym, d)
        rec = {"date": d, "ticker": sym, "underlying_0931": S,
               "price_tier": tier(S), "status": None, "spread_pct": None,
               "oi": None, "volume": None}
        exps = get("option/list/expirations", symbol=sym)
        if not exps:
            rec["status"] = "NO_OPTIONS"
            results.append(rec)
            continue
        future = sorted(e["expiration"] for e in exps if e["expiration"] >= d)
        if not future or S is None:
            rec["status"] = "NO_LIVE_EXPIRY" if not future else "NO_UNDERLYING"
            results.append(rec)
            continue
        exp = future[0]
        strikes = get("option/list/strikes", symbol=sym, expiration=exp)
        if not strikes:
            rec["status"] = "NO_STRIKES"
            results.append(rec)
            continue
        K = min((float(s["strike"]) for s in strikes), key=lambda k: abs(k - S))
        rec["strike"], rec["expiry"] = K, exp
        q = call_quote_0931(sym, exp, int(K) if K == int(K) else K, d)
        if q is None:
            rec["status"] = "NO_QUOTE_0931"
            results.append(rec)
            continue
        m = (q["bid"] + q["ask"]) / 2
        spr = (q["ask"] - q["bid"]) / m if m > 0 else 9.99
        rec.update({"spread_pct": round(spr, 4), "oi": q["oi"],
                    "volume": q["vol"],
                    "status": "TRADEABLE" if spr <= SPREAD_CAP else "NOT_TRADEABLE"})
        results.append(rec)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(events)}")

    n = len(results)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    tradeable = sum(1 for r in results if r["status"] == "TRADEABLE")
    # per-tier
    tiers = {}
    for tk in ("<$1", "$1-5", "$5-20", "$20+", "unknown"):
        sub = [r for r in results if r["price_tier"] == tk]
        if sub:
            tr = sum(1 for r in sub if r["status"] == "TRADEABLE")
            tiers[tk] = {"n": len(sub), "tradeable": tr,
                         "pct_tradeable": round(tr / len(sub), 3)}
    summary = {"n_events": n, "status_counts": counts,
               "pct_tradeable_overall": round(tradeable / n, 3) if n else 0,
               "by_price_tier": tiers, "spread_cap": SPREAD_CAP,
               "door": "OPEN" if (n and tradeable / n >= 0.30) else "CLOSED"}
    OUT.write_text(json.dumps({"summary": summary, "events": results}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
