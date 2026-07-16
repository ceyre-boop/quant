#!/usr/bin/env python3
"""Measure ACTUAL borrow conditions on the 234 HYP-093 forward-year events via
options put-call parity (ThetaData VALUE tier, ThetaTerminal :25503).

Per event: nearest expiry after event date, ATM strike vs 10:30 entry, CALL &
PUT quotes at 10:30 ET, mid prices -> implied annualized borrow fee:
    b = -ln( (C_mid - P_mid + K*exp(-r*T)) / S ) / T,   r = 0.04
High b = expensive/hard borrow priced into options. NO_OPTIONS = ticker has no
usable chain (the capacity ceiling in its bluntest form).
Output: data/research/gapper/implied_borrow_234.json
"""
import csv
import json
import math
import time
import urllib.request
import urllib.parse
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
OUT = REPO / "data/research/gapper/implied_borrow_234.json"
BASE = "http://127.0.0.1:25503/v3"
R = 0.04
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]


def get(path, **params):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                txt = r.read().decode()
            if txt.startswith("No data"):
                return []
            rows = list(csv.DictReader(txt.splitlines()))
            return rows
        except Exception:
            time.sleep(1 + attempt)
    return None


def mid(row):
    b, a = float(row["bid"]), float(row["ask"])
    if b <= 0 or a <= 0:
        return None
    return (a + b) / 2


def quote_1030(sym, exp, strike, right, d):
    rows = get("option/history/quote", symbol=sym, expiration=exp,
               strike=strike, right=right, start_date=d, end_date=d,
               interval="30m")
    if not rows:
        return None
    for row in rows:
        if row["timestamp"].endswith("10:30:00.000"):
            return mid(row)
    return None


def main():
    events = []
    for r in csv.DictReader(open(CSV)):
        try:
            g, p, v = (float(r["gain_1030"]), float(r["price_1030"]),
                       float(r["cum_vol_1030"]))
        except ValueError:
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
        S = float(r["entry_open_1030"])
        rec = {"date": d, "ticker": sym, "entry": S, "status": None,
               "implied_borrow": None, "expiry": None, "strike": None}
        exps = get("option/list/expirations", symbol=sym)
        if not exps:
            rec["status"] = "NO_OPTIONS"
            results.append(rec)
            continue
        ev_date = date.fromisoformat(d)
        future = sorted(e["expiration"] for e in exps
                        if e["expiration"] > d)
        if not future:
            rec["status"] = "NO_LIVE_EXPIRY"
            results.append(rec)
            continue
        exp = future[0]
        T = max((date.fromisoformat(exp) - ev_date).days, 1) / 365
        strikes = get("option/list/strikes", symbol=sym, expiration=exp)
        if not strikes:
            rec["status"] = "NO_STRIKES"
            results.append(rec)
            continue
        K = min((float(s["strike"]) for s in strikes),
                key=lambda k: abs(k - S))
        kfmt = int(K) if K == int(K) else K
        C = quote_1030(sym, exp, kfmt, "C", d)
        P = quote_1030(sym, exp, kfmt, "P", d)
        if C is None or P is None:
            rec["status"] = "NO_QUOTE_1030"
            results.append(rec)
            continue
        try:
            x = (C - P + K * math.exp(-R * T)) / S
            if x <= 0:
                rec["status"] = "PARITY_DEGENERATE"
            else:
                rec["implied_borrow"] = round(-math.log(x) / T, 4)
                rec["status"] = "OK"
                rec["expiry"], rec["strike"] = exp, K
                rec["call_mid"], rec["put_mid"], rec["T_years"] = C, P, round(T, 4)
        except (ValueError, ZeroDivisionError):
            rec["status"] = "PARITY_DEGENERATE"
        results.append(rec)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(events)} done")

    counts = {}
    for rec in results:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    ok = sorted(r["implied_borrow"] for r in results if r["status"] == "OK")
    summary = {
        "n_events": len(results), "status_counts": counts,
        "implied_borrow_pctiles": (
            {"p10": ok[len(ok)//10], "p25": ok[len(ok)//4],
             "p50": ok[len(ok)//2], "p75": ok[3*len(ok)//4],
             "p90": ok[9*len(ok)//10], "n": len(ok)} if len(ok) >= 10 else None),
    }
    OUT.write_text(json.dumps({"summary": summary, "events": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
