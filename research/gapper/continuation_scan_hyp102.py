#!/usr/bin/env python3
"""HYP-102 Step 1 — DIRTY feature scan: do the ~30% non-faders (continuers)
look different from faders at 10:30? 2025-H2 ONLY — 2026 rows never read.

CONTINUER: close_eod > entry_open_1030. FADER: close_eod < entry.
Features at 10:30 (no lookahead within the event):
  gain_1030, rvol_1030 (vol vs trailing avg, precomputed), vwap_dist,
  intraday_push, mins_to_high (proxy for "still making highs into 10:30"),
  spread proxy (10:25-10:30 bar (high-low)/close via Alpaca SIP), catalyst,
  spy_am. Sector unavailable in dataset — catalyst class used instead (noted).
Candidate bar: |median diff| >= 20% of fader median AND directionally intuitive.
"""
import csv
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
SPREAD_CACHE = REPO / "data/research/gapper/h2_1030bar_spread.json"
OUT = Path(__file__).with_name("continuation_scan_HYP102.json")
CUTOFF = "2026-01-01"
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]
ET = ZoneInfo("America/New_York")


def mw_p(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 8 or n2 < 8:
        return None
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    r1 = sum(ranks[k] for k, (v, g) in enumerate(allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu, sigma = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / sigma
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def env_keys():
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def fetch_spread(events):
    if SPREAD_CACHE.exists():
        return json.loads(SPREAD_CACHE.read_text())
    env = env_keys()
    H = {"APCA-API-KEY-ID": env["ALPACA_API_KEY"],
         "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"]}
    out = {}
    for e in events:
        d, t = e["date"], e["ticker"]
        start = datetime.fromisoformat(f"{d}T10:25:00").replace(tzinfo=ET) \
            .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = datetime.fromisoformat(f"{d}T10:30:00").replace(tzinfo=ET) \
            .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        q = urllib.parse.urlencode({"symbols": t, "timeframe": "5Min",
                                    "start": start, "end": end,
                                    "adjustment": "all", "feed": "sip",
                                    "limit": 10})
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    f"https://data.alpaca.markets/v2/stocks/bars?{q}",
                    headers=H), timeout=30) as resp:
                bars = (json.loads(resp.read()).get("bars") or {}).get(t) or []
            if bars:
                b = bars[-1]
                out[f"{d}|{t}"] = (b["h"] - b["l"]) / b["c"] if b["c"] else None
        except Exception:
            pass
        time.sleep(0.3)
    SPREAD_CACHE.write_text(json.dumps(out))
    return out


def main():
    evs = []
    for r in csv.DictReader(open(CSV)):
        if r["date"] >= CUTOFF:
            continue
        try:
            g, p, v = (float(r["gain_1030"]), float(r["price_1030"]),
                       float(r["cum_vol_1030"]))
            entry, close = float(r["entry_open_1030"]), float(r["close_eod"])
        except (ValueError, KeyError):
            continue
        if g < 1.0 or p < 2.0 or v < 500_000:
            continue
        if any(m in (r.get("catalyst") or "").lower() for m in MNA):
            continue
        r["_long_ret"] = (close - entry) / entry
        evs.append(r)
    faders = [e for e in evs if e["_long_ret"] < 0]
    cont = [e for e in evs if e["_long_ret"] > 0]
    print(f"H2-2025: {len(evs)} events -> faders {len(faders)}, "
          f"continuers {len(cont)} "
          f"(cont median long ret {statistics.median([e['_long_ret'] for e in cont]):+.4f})")

    spread = fetch_spread(evs)

    def fval(e, k):
        if k == "spread_1030bar":
            return spread.get(f"{e['date']}|{e['ticker']}")
        try:
            return float(e[k])
        except (ValueError, KeyError, TypeError):
            return None

    feats = ["gain_1030", "rvol_1030", "vwap_dist", "intraday_push",
             "mins_to_high", "overnight_gap", "dollar_vol_m", "spy_am",
             "spread_1030bar"]
    rows = []
    for k in feats:
        fa = [fval(e, k) for e in faders]
        co = [fval(e, k) for e in cont]
        fa = [x for x in fa if x is not None]
        co = [x for x in co if x is not None]
        if len(fa) < 8 or len(co) < 8:
            continue
        mf, mc = statistics.median(fa), statistics.median(co)
        eff = (mc - mf) / abs(mf) if mf != 0 else None
        rows.append({"feature": k, "median_fader": round(mf, 4),
                     "median_continuer": round(mc, 4),
                     "effect_rel_to_fader_median": (round(eff, 3)
                                                    if eff is not None else None),
                     "p_mw": round(mw_p(co, fa), 5),
                     "clears_effect_bar": (eff is not None and abs(eff) >= 0.20)})
    # catalyst distribution (categorical, informational)
    def cat_mix(group):
        c = {}
        for e in group:
            k = (e.get("catalyst") or "NONE").upper() or "NONE"
            c[k] = c.get(k, 0) + 1
        n = len(group)
        return {k: round(v / n, 3) for k, v in sorted(c.items(), key=lambda x: -x[1])}
    rows.sort(key=lambda r: -(abs(r["effect_rel_to_fader_median"] or 0)))
    out = {"step": "1-dirty-scan", "hyp": "HYP-102",
           "scan_window": ["2025-07-02", "2025-12-31"],
           "n_events": len(evs), "n_faders": len(faders),
           "n_continuers": len(cont),
           "continuer_median_long_ret": round(
               statistics.median([e["_long_ret"] for e in cont]), 4),
           "features": rows,
           "catalyst_mix_faders": cat_mix(faders),
           "catalyst_mix_continuers": cat_mix(cont),
           "note": "sector unavailable in dataset; catalyst class reported instead"}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
