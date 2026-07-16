#!/usr/bin/env python3
"""HYP-099 Step 1 — LOOKAHEAD SCAN (hypothesis generation ONLY).

Scans regime features on the SCAN half (events dated < 2026-01-01) of
per_candidate_enriched.csv. Rows dated 2026+ are NEVER loaded here — that is
the untouched holdout reserved for the pre-registered Step 4 test.

This is deliberate data mining: any lookahead is allowed, nothing found here
is evidence. Output: research/gapper/regime_scan_HYP099.json (the green list).
"""
import csv
import json
import statistics
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
OUT = Path(__file__).parent / "regime_scan_HYP099.json"
SCAN_CUTOFF = "2026-01-01"          # rows >= this date are HOLDOUT, never read
# HYP-093 frozen qualifying filters (verbatim from sealed spec)
GAIN_MIN, PRICE_MIN, VOL_MIN = 1.00, 2.00, 500_000
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private", "M&A", "MNA"]


def mannwhitney_u(a, b):
    """Two-sided Mann-Whitney via normal approximation (no scipy dependency)."""
    import math
    n1, n2 = len(a), len(b)
    if n1 < 8 or n2 < 8:
        return None
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    # average ranks with ties
    ranks = {}
    i = 0
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
    if sigma == 0:
        return None
    z = (u1 - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


def load_scan_events():
    events = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["date"] >= SCAN_CUTOFF:
                continue  # holdout — do not touch
            try:
                gain = float(r["gain_1030"])
                price = float(r["price_1030"])
                vol = float(r["cum_vol_1030"])
                out = float(r["outcome_pct"])
            except (ValueError, KeyError):
                continue
            if gain < GAIN_MIN or price < PRICE_MIN or vol < VOL_MIN:
                continue
            cat = (r.get("catalyst") or "").lower()
            if any(m.lower() in cat for m in MNA):
                continue
            r["_fade"] = -out           # short return: positive = fade profit
            events.append(r)
    return events


def fnum(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return None


def main():
    ev = load_scan_events()
    print(f"scan events (qualifying, pre-2026): {len(ev)}")
    fades = [e["_fade"] for e in ev]
    print(f"base: median fade {statistics.median(fades):+.4f}  mean {statistics.mean(fades):+.4f}")

    results = []

    # --- categorical features ---
    def cat_split(name, keyfn):
        groups = {}
        for e in ev:
            groups.setdefault(keyfn(e), []).append(e["_fade"])
        base_med = statistics.median(fades)
        for g, vals in groups.items():
            if len(vals) < 15:
                continue
            rest = [e["_fade"] for e in ev if keyfn(e) != g]
            p = mannwhitney_u(vals, rest)
            results.append({
                "feature": name, "bucket": str(g), "n": len(vals),
                "median_fade": round(statistics.median(vals), 4),
                "median_rest": round(statistics.median(rest), 4),
                "delta_median": round(statistics.median(vals) - statistics.median(rest), 4),
                "mean_fade": round(statistics.mean(vals), 4),
                "p_mw": round(p, 5) if p is not None else None,
            })

    cat_split("dow", lambda e: e.get("dow"))
    cat_split("catalyst", lambda e: (e.get("catalyst") or "NONE").upper() or "NONE")
    cat_split("gap_bucket", lambda e: (
        "100-130%" if fnum(e, "gain_1030") < 1.3 else
        "130-200%" if fnum(e, "gain_1030") < 2.0 else
        "200-500%" if fnum(e, "gain_1030") < 5.0 else "500%+"))
    cat_split("active_proxy", lambda e: e.get("active_proxy"))
    cat_split("read", lambda e: e.get("read"))

    # --- numeric features: median split + quartile extremes ---
    numerics = ["overnight_gap", "intraday_push", "rvol_1030", "dollar_vol_m",
                "mins_to_high", "vwap_dist", "spy_am", "prev_close",
                "price_1030", "cum_vol_1030", "gain_1030"]
    for k in numerics:
        vals = [(fnum(e, k), e["_fade"]) for e in ev if fnum(e, k) is not None]
        if len(vals) < 40:
            continue
        vs = sorted(v for v, _ in vals)
        med = vs[len(vs) // 2]
        q1, q3 = vs[len(vs) // 4], vs[3 * len(vs) // 4]
        for label, lo_pred in [
            (f"{k}<=med({med:.3g})", lambda v, t=med: v <= t),
            (f"{k}>med({med:.3g})", lambda v, t=med: v > t),
            (f"{k}<=q1({q1:.3g})", lambda v, t=q1: v <= t),
            (f"{k}>=q3({q3:.3g})", lambda v, t=q3: v >= t),
        ]:
            ins = [f for v, f in vals if lo_pred(v)]
            outs = [f for v, f in vals if not lo_pred(v)]
            if len(ins) < 15 or len(outs) < 15:
                continue
            p = mannwhitney_u(ins, outs)
            results.append({
                "feature": k, "bucket": label, "n": len(ins),
                "median_fade": round(statistics.median(ins), 4),
                "median_rest": round(statistics.median(outs), 4),
                "delta_median": round(statistics.median(ins) - statistics.median(outs), 4),
                "mean_fade": round(statistics.mean(ins), 4),
                "p_mw": round(p, 5) if p is not None else None,
            })

    # --- 2-feature composites from the top singles ---
    singles = sorted([r for r in results if r["delta_median"] >= 0.02],
                     key=lambda r: -r["delta_median"])[:8]
    # rebuild predicates for composites
    def pred_of(r):
        feat, buck = r["feature"], r["bucket"]
        if feat in ("dow", "catalyst", "gap_bucket", "active_proxy", "read"):
            if feat == "dow":
                return lambda e: str(e.get("dow")) == buck
            if feat == "catalyst":
                return lambda e: ((e.get("catalyst") or "NONE").upper() or "NONE") == buck
            if feat == "gap_bucket":
                def gb(e):
                    g = fnum(e, "gain_1030")
                    lbl = ("100-130%" if g < 1.3 else "130-200%" if g < 2.0
                           else "200-500%" if g < 5.0 else "500%+")
                    return lbl == buck
                return gb
            return lambda e: str(e.get(feat)) == buck
        # numeric bucket label encodes op+threshold
        import re
        m = re.match(r".*(<=|>=|>|<)\w*\(([-\d.e+]+)\)", buck)
        op, thr = m.group(1), float(m.group(2))
        def np_(e):
            v = fnum(e, feat)
            if v is None:
                return False
            return {"<=": v <= thr, ">": v > thr, ">=": v >= thr, "<": v < thr}[op]
        return np_

    for a, b in combinations(singles, 2):
        if a["feature"] == b["feature"]:
            continue
        pa, pb = pred_of(a), pred_of(b)
        ins = [e["_fade"] for e in ev if pa(e) and pb(e)]
        outs = [e["_fade"] for e in ev if not (pa(e) and pb(e))]
        if len(ins) < 20 or len(outs) < 20:
            continue
        p = mannwhitney_u(ins, outs)
        results.append({
            "feature": f"COMPOSITE[{a['feature']} & {b['feature']}]",
            "bucket": f"{a['bucket']} & {b['bucket']}", "n": len(ins),
            "median_fade": round(statistics.median(ins), 4),
            "median_rest": round(statistics.median(outs), 4),
            "delta_median": round(statistics.median(ins) - statistics.median(outs), 4),
            "mean_fade": round(statistics.mean(ins), 4),
            "p_mw": round(p, 5) if p is not None else None,
        })

    results.sort(key=lambda r: -(r["delta_median"] or 0))
    green = [r for r in results if r["delta_median"] >= 0.03]
    OUT.write_text(json.dumps({
        "step": "1-lookahead-scan", "hyp": "HYP-099",
        "scan_window": ["2025-07-02", "2025-12-31"],
        "n_scan_events": len(ev),
        "base_median_fade": round(statistics.median(fades), 4),
        "n_combos_tested": len(results),
        "n_green_delta_ge_3pct": len(green),
        "results": results,
    }, indent=1))
    print(f"combos tested: {len(results)}  green (delta>=3%): {len(green)}")
    for r in results[:15]:
        print(f"  {r['delta_median']:+.4f}  n={r['n']:4d}  p={r['p_mw']}  {r['feature']} :: {r['bucket']}")


if __name__ == "__main__":
    main()
