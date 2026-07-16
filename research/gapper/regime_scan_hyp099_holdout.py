#!/usr/bin/env python3
"""HYP-099 Step 4 — FROZEN holdout test. Refuses to run unless the prereg is
committed to git. Everything here is verbatim from regime-study-prereg-HYP099.md."""
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
PREREG = Path(__file__).parent / "regime-study-prereg-HYP099.md"
PREREG_SHA = "2bcf4b9f50b7376f15732bdbc7a9d85b884cf18ba4e69c1e00b7880be0516b13"

GAIN_MIN, PRICE_MIN, VOL_MIN = 1.00, 2.00, 500_000
SLIP, LOCATE_W, NOTIONAL = 0.005, 0.50, 0.0125
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]


def guard():
    sha = subprocess.run(["shasum", "-a", "256", str(PREREG)],
                         capture_output=True, text=True).stdout.split()[0]
    assert sha == PREREG_SHA, f"prereg hash mismatch: {sha}"
    log = subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "--",
                          str(PREREG)], capture_output=True, text=True).stdout
    assert log.strip(), "prereg not committed — holdout run refused"


def apr(gain):
    if gain >= 1.5:
        return 6.00
    if gain >= 1.0:
        return 4.00
    return 2.00


def mannwhitney_one_sided(a, b):
    """P(one-sided) that a > b, normal approx with tie-averaged ranks."""
    n1, n2 = len(a), len(b)
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
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / sigma
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))   # one-sided, a>b


def load(lo, hi):
    out = []
    for r in csv.DictReader(open(CSV)):
        if not (lo <= r["date"] < hi):
            continue
        try:
            gain, price, vol = (float(r["gain_1030"]), float(r["price_1030"]),
                                float(r["cum_vol_1030"]))
            fade = -float(r["outcome_pct"])
        except (ValueError, KeyError):
            continue
        if gain < GAIN_MIN or price < PRICE_MIN or vol < VOL_MIN:
            continue
        if any(m in (r.get("catalyst") or "").lower() for m in MNA):
            continue
        r["_net"] = fade - 2 * SLIP - LOCATE_W * apr(gain) / 252
        out.append(r)
    return out


def main():
    guard()
    ev = load("2026-01-01", "2026-07-01")
    print(f"holdout qualifying events: {len(ev)}")
    days = len({e["date"] for e in ev})

    variants = {
        "V1": lambda e: float(e["intraday_push"]) > 0.195,
        "V2": lambda e: (float(e["intraday_push"]) > 0.195
                         and float(e["overnight_gap"]) <= 1.19),
    }
    res = {}
    for name, pred in variants.items():
        ins = [e["_net"] for e in ev if pred(e)]
        outs = [e["_net"] for e in ev if not pred(e)]
        p = mannwhitney_one_sided(ins, outs) if len(ins) >= 15 else None
        res[name] = {
            "n_in": len(ins), "n_out": len(outs),
            "median_in": round(statistics.median(ins), 4) if ins else None,
            "median_out": round(statistics.median(outs), 4) if outs else None,
            "mean_in": round(statistics.mean(ins), 4) if ins else None,
            "p_one_sided": round(p, 5) if p is not None else None,
        }
    # BH across k=2
    ps = sorted([(v["p_one_sided"], k) for k, v in res.items()
                 if v["p_one_sided"] is not None])
    m = len(ps)
    passed_bh = set()
    for i, (p, k) in enumerate(ps, 1):
        if p <= 0.05 * i / m:
            passed_bh.add(k)
        else:
            break
    for k, v in res.items():
        v["bh_pass"] = k in passed_bh
        v["delta_median"] = (round(v["median_in"] - v["median_out"], 4)
                             if v["median_in"] is not None else None)
        v["verdict_conditions"] = {
            "n_in_ge_15": v["n_in"] >= 15,
            "bh_p_lt_05": k in passed_bh,
            "delta_ge_3pct": (v["delta_median"] or -1) >= 0.03,
        }
        v["confirmed"] = all(v["verdict_conditions"].values())
        # economic floor: in-regime mean %/day at constitutional sizing
        in_ev = [e for e in ev if variants[k](e)]
        daily = {}
        for e in in_ev:
            daily[e["date"]] = daily.get(e["date"], 0) + e["_net"] * NOTIONAL
        total_days = len({e["date"] for e in ev})  # trading days with any event
        v["mean_pct_day_constitutional"] = (round(sum(daily.values()) /
                                            max(total_days, 1), 6))

    survivor = ("V2" if res["V2"]["confirmed"] else
                "V1" if res["V1"]["confirmed"] else None)
    out = {"hyp": "HYP-099", "holdout": ["2026-01-01", "2026-06-30"],
           "n_events": len(ev), "n_event_days": days,
           "variants": res, "survivor": survivor,
           "verdict": "CONFIRMED" if survivor else "NOT_SIGNIFICANT"}
    Path(__file__).with_name("regime_holdout_HYP099.json").write_text(
        json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
