#!/usr/bin/env python3
"""HYP-103 Step 1 — EV scan over strategy x challenge grid. DIRTY DATA ONLY
(234-event forward year). Live shadow is the untouched holdout.

Grid: stop {15..40%} x sizing {1..5%} x entry {10:30..11:30} = 240 strategy
cells; per cell, per-event nets rebuilt from minute-slice cache; MC 10k paths
for 90d (+-8%) and unlimited (+8% target / -10% static DD, FunderPro shape).
EV_y1 = P(PASS) x account x annual_ret x 0.80 - fee / P(PASS); EV_3yr = 3x.
(FunderPro refunds the fee on pass; refund-adjusted EV also reported.)

WARNING stamped in output: 240-cell grid = deliberate mining. Nothing here is
evidence; the pre-registered shadow test is the only evidence.
"""
import csv
import json
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
SLICES = REPO / "data/research/gapper/event_minute_slices.json"
OUT = Path(__file__).with_name("ev_scan_HYP103.json")

SLIP, LOCATE_W = 0.005, 0.50
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]
STOPS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SIZINGS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
ENTRIES = ["10:30", "10:45", "11:00", "11:15", "11:30"]
ACCOUNTS = {25_000: 250, 50_000: 300, 100_000: 550, 200_000: 995}  # FunderPro 2026
PAYOUT = 0.80
N_PATHS = 10_000
SEED = 42


def apr(g):
    return 6.00 if g >= 1.5 else 4.00 if g >= 1.0 else 2.00


def load_events():
    slices = json.loads(SLICES.read_text())
    evs = []
    for r in csv.DictReader(open(CSV)):
        try:
            g, p, v = (float(r["gain_1030"]), float(r["price_1030"]),
                       float(r["cum_vol_1030"]))
        except (ValueError, KeyError):
            continue
        if g < 1.0 or p < 2.0 or v < 500_000:
            continue
        if any(m in (r.get("catalyst") or "").lower() for m in MNA):
            continue
        s = slices.get(f"{r['date']}|{r['ticker']}")
        evs.append({"date": r["date"], "g": g, "slice": s})
    return evs


def cell_day_rets(evs, entry, stop, sizing):
    """Daily P&L at (entry time, stop, sizing); missing slice => stopped."""
    daily = {}
    for e in evs:
        s = (e["slice"] or {}).get(entry) if e["slice"] else None
        if s is None:
            gross = -(stop + SLIP)                       # conservative
        else:
            ep, hi = s["entry_open"], s["post_high"]
            close = e["slice"]["close_last"]
            if ep <= 0:
                continue
            if hi >= ep * (1 + stop):
                gross = -(stop + SLIP)
            else:
                gross = (ep - close) / ep
        net = gross - 2 * SLIP - LOCATE_W * apr(e["g"]) / 252
        daily[e["date"]] = daily.get(e["date"], 0.0) + net * sizing
    return list(daily.values())


def mc(day_rets, rng, window, bust):
    p_event = len(day_rets) / 252
    npass = nbust = 0
    cap = window or 2520
    for _ in range(N_PATHS):
        eq, d = 1.0, 0
        while d < cap:
            d += 1
            if rng.random() < p_event:
                eq *= (1 + rng.choice(day_rets))
            if eq >= 1.08:
                npass += 1
                break
            if eq <= 1 + bust:
                nbust += 1
                break
    return npass / N_PATHS, nbust / N_PATHS


def main():
    evs = load_events()
    n_sliced = sum(1 for e in evs if e["slice"])
    print(f"events {len(evs)}, with minute slices {n_sliced}")
    rows = []
    for entry in ENTRIES:
        for stop in STOPS:
            for sizing in SIZINGS:
                rng = random.Random(SEED)
                dr = cell_day_rets(evs, entry, stop, sizing)
                eq = 1.0
                for x in dr:
                    eq *= (1 + x)
                annual = eq - 1
                p90, b90 = mc(dr, rng, 90, -0.08)
                pun, bun = mc(dr, rng, None, -0.10)
                for size, fee in ACCOUNTS.items():
                    profit = size * annual * PAYOUT
                    for wname, pp, pb in (("90d", p90, b90),
                                          ("unlimited", pun, bun)):
                        if pp <= 0:
                            continue
                        ev1 = pp * profit - fee / pp
                        ev1_refund = pp * profit - fee * (1 - pp) / pp
                        rows.append({
                            "entry": entry, "stop_pct": stop * 100,
                            "sizing_pct": sizing * 100, "account": size,
                            "window": wname, "annual_return": round(annual, 4),
                            "p_pass": round(pp, 4), "p_bust": round(pb, 4),
                            "ev_y1": round(ev1), "ev_3yr": round(3 * ev1),
                            "ev_y1_fee_refunded": round(ev1_refund),
                        })
    rows.sort(key=lambda r: -r["ev_y1"])
    top20 = rows[:20]
    # Pareto: best p_pass per p_bust band (90d, 100k account for comparability)
    pareto = {}
    for r in rows:
        if r["account"] != 100_000:
            continue
        band = round(min(r["p_bust"], 0.10) * 100)
        if band not in pareto or r["p_pass"] > pareto[band]["p_pass"]:
            pareto[band] = r
    out = {
        "WARNING": ("240-cell mined grid on dirty data. NOT evidence. The "
                    "pre-registered live-shadow test (HYP-103 prereg) is the "
                    "only evidence. Locate assumed available on every signal; "
                    "all EV conditional on an equity-capable funded vehicle "
                    "(TICK-032 wall)."),
        "fees": {str(k): v for k, v in ACCOUNTS.items()},
        "payout": PAYOUT, "n_paths": N_PATHS, "seed": SEED,
        "n_events": len(evs), "n_with_slices": n_sliced,
        "top20_by_ev_y1": top20,
        "pareto_90d_100k": [pareto[k] for k in sorted(pareto)],
    }
    OUT.write_text(json.dumps(out, indent=1))
    for r in top20[:10]:
        print(r)


if __name__ == "__main__":
    main()
