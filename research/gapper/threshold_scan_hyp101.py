#!/usr/bin/env python3
"""HYP-101 Step 1 — DIRTY threshold + timing scan (2025-H2 ONLY; hypothesis
generation). 2026 sub-100% outcomes are the untouched holdout — never read here.

Bands (incremental, price>=2, vol>=500k, M&A excluded):
  75-100%, 60-100%, 50-100% gain by 10:30, vs the >=100% confirmed reference.
Quality bar (mandate): median fade positive (stock fades) AND >=55% of events
fade (directional consistency).
Timing variant: 11:00 entry on 2025-H2 >=100% events (paired vs 10:30 entry),
using Alpaca SIP minute bars (fetched into data/research/gapper/h2_1100_bars.json).
"""
import csv
import json
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
BARS = REPO / "data/research/gapper/h2_1100_bars.json"
OUT = Path(__file__).with_name("threshold_scan_HYP101.json")
CUTOFF = "2026-01-01"                     # >= this date is HOLDOUT: never read
SLIP, LOCATE_W = 0.005, 0.50
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]
ET = ZoneInfo("America/New_York")


def apr(g):
    return 6.00 if g >= 1.5 else 4.00 if g >= 1.0 else 2.00


def net(fade, g):
    return fade - 2 * SLIP - LOCATE_W * apr(g) / 252


def load_h2():
    evs = []
    for r in csv.DictReader(open(CSV)):
        if r["date"] >= CUTOFF:
            continue
        try:
            g, p, v = (float(r["gain_1030"]), float(r["price_1030"]),
                       float(r["cum_vol_1030"]))
            fade = -float(r["outcome_pct"])
        except (ValueError, KeyError):
            continue
        if p < 2.0 or v < 500_000 or g < 0.5:
            continue
        if any(m in (r.get("catalyst") or "").lower() for m in MNA):
            continue
        r["_g"], r["_net"] = g, net(fade, g)
        evs.append(r)
    return evs


def env_keys():
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def fetch_1100_bars(events):
    """Fetch 11:00 open + 11:00-16:00 bars for >=100% H2 events (once, cached)."""
    if BARS.exists():
        return json.loads(BARS.read_text())
    env = env_keys()
    H = {"APCA-API-KEY-ID": env["ALPACA_API_KEY"],
         "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"]}
    out = {}
    for e in events:
        d, t = e["date"], e["ticker"]
        start = datetime.fromisoformat(f"{d}T11:00:00").replace(tzinfo=ET) \
            .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = datetime.fromisoformat(f"{d}T16:00:00").replace(tzinfo=ET) \
            .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        q = urllib.parse.urlencode({"symbols": t, "timeframe": "1Min",
                                    "start": start, "end": end,
                                    "adjustment": "all", "feed": "sip",
                                    "limit": 10000})
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    f"https://data.alpaca.markets/v2/stocks/bars?{q}",
                    headers=H), timeout=30) as resp:
                bars = (json.loads(resp.read()).get("bars") or {}).get(t) or []
            if bars:
                out[f"{d}|{t}"] = {"open_1100": bars[0]["o"],
                                   "high_post_1100": max(b["h"] for b in bars)}
        except Exception:
            pass
        time.sleep(0.3)
    BARS.write_text(json.dumps(out))
    return out


def band_stats(evs, lo, hi, label):
    sel = [e for e in evs if lo <= e["_g"] < hi]
    nets = [e["_net"] for e in sel]
    if not nets:
        return None
    return {"band": label, "n_half_year": len(sel),
            "annualized_events": len(sel) * 2,
            "median_net_fade": round(statistics.median(nets), 4),
            "pct_fade_worked": round(sum(1 for x in nets if x > 0) / len(nets), 3),
            "mean_net_fade": round(statistics.mean(nets), 4),
            "worst_event": round(min(nets), 4),
            "clears_bar": (statistics.median(nets) > 0 and
                           sum(1 for x in nets if x > 0) / len(nets) >= 0.55)}


def main():
    evs = load_h2()
    ref = [e for e in evs if e["_g"] >= 1.0]
    ref_med = statistics.median([e["_net"] for e in ref])
    print(f"H2-2025 events (gain>=0.5): {len(evs)}; >=100% ref n={len(ref)} "
          f"median net {ref_med:+.4f} (reconcile vs +0.1036 gross-era ~ +0.10)")

    bands = [band_stats(evs, 1.00, 99, ">=100% (reference)"),
             band_stats(evs, 0.75, 1.00, "75-100%"),
             band_stats(evs, 0.60, 1.00, "60-100% (cumulative)"),
             band_stats(evs, 0.60, 0.75, "60-75% (incremental)"),
             band_stats(evs, 0.50, 1.00, "50-100% (cumulative)"),
             band_stats(evs, 0.50, 0.60, "50-60% (incremental)")]

    # Timing variant: 11:00 entry, paired, >=100% H2 events
    bars = fetch_1100_bars(ref)
    paired = []
    for e in ref:
        b = bars.get(f"{e['date']}|{e['ticker']}")
        if not b:
            continue
        entry11 = b["open_1100"]
        close = float(e["close_eod"])
        fade11 = (entry11 - close) / entry11
        paired.append({"net_1030": e["_net"],
                       "net_1100": net(fade11, e["_g"])})
    t30 = [p["net_1030"] for p in paired]
    t11 = [p["net_1100"] for p in paired]
    timing = {"n_paired": len(paired),
              "median_net_1030": round(statistics.median(t30), 4) if t30 else None,
              "median_net_1100": round(statistics.median(t11), 4) if t11 else None,
              "pct_worked_1030": round(sum(1 for x in t30 if x > 0) / len(t30), 3) if t30 else None,
              "pct_worked_1100": round(sum(1 for x in t11 if x > 0) / len(t11), 3) if t11 else None,
              "median_paired_delta_1100_minus_1030": (
                  round(statistics.median([p["net_1100"] - p["net_1030"]
                                           for p in paired]), 4) if paired else None)}

    out = {"step": "1-dirty-scan", "hyp": "HYP-101",
           "scan_window": ["2025-07-02", "2025-12-31"],
           "reference_median_net": round(ref_med, 4),
           "bands": [b for b in bands if b], "timing_variant_1100": timing}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
