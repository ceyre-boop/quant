#!/usr/bin/env python3
"""TICK-036 — Top-3 movers study (MINING, Steps 1-2 of THE RESEARCH METHOD).

STAMP: MINING — lookahead allowed and used; characterization, not evidence.
Single chronological pass over both on-disk full-market years
(2024-07→2026-06); window-agnostic (reruns unchanged on deeper data).
Run: python3 -m research.movers_study.study
"""
import gzip
import json
import random
from collections import defaultdict, deque, Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/research/movers_study"
DIRS = [REPO / "data/research/yield_frontier/holdout_equities/grouped",
        REPO / "data/research/gapper/cache/grouped"]
STAMP = "MINING — lookahead allowed; characterization only; NOT evidence"
MIN_VOL, MIN_PC = 500_000, 0.75
rng = random.Random(42)


import re
TEST_SYM = re.compile(r"^Z[A-Z]ZZT$")   # ZVZZT/ZWZZT/ZXZZT... Nasdaq test symbols

def ok(t):
    return (t.isalpha() and len(t) <= 5 and not (len(t) == 5 and t[-1] in "WRU")
            and not TEST_SYM.match(t))


def load_days():
    files = {}
    for d in DIRS:
        for fp in sorted(d.glob("*.json.gz")):
            files.setdefault(fp.name.split(".")[0], fp)
    for day in sorted(files):
        with gzip.open(files[day], "rt") as f:
            doc = json.load(f)
        if doc.get("n", 0) > 0 and not doc.get("denied_403"):
            yield day, doc["bars"]


def vix_series():
    try:
        v = pd.read_parquet(REPO / "data/research/modern/spot_cache/VIX.parquet")
        v.index = pd.to_datetime(v.index).strftime("%Y-%m-%d")
        return v["Close"].to_dict()
    except Exception:
        return {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vix = vix_series()
    hist = defaultdict(lambda: deque(maxlen=21))   # ticker -> (close, volume)
    last_run = {}                                   # ticker -> day index of last top-100
    top3_hist = deque(maxlen=3)                     # yesterday's sets
    topN_hist = deque(maxlen=1)
    rows, controls, day_rows = [], [], []
    offenders = Counter()
    watch_hits = rand_hits = watch_days = 0
    di = -1
    for day, bars in load_days():
        di += 1
        univ = {}
        for t, b in bars.items():
            if not ok(t) or b[3] is None or b[4] is None or b[0] is None:
                continue
            h_ = hist.get(t)
            pc = h_[-1][0] if h_ and len(h_) else None
            if pc and pc >= MIN_PC and b[4] >= MIN_VOL:
                univ[t] = (b, pc)
        gains = {t: v[0][3] / v[1] - 1 for t, v in univ.items()}
        if len(gains) < 100:
            for t, b in bars.items():
                if b[3] and b[4]:
                    hist[t].append((b[3], b[4]))
            continue
        srt = sorted(gains, key=gains.get, reverse=True)
        top3, top100 = srt[:3], set(srt[:100])
        hi3 = sorted(univ, key=lambda t: univ[t][0][1] / univ[t][1] if univ[t][0][1] else 0,
                     reverse=True)[:3]

        def feats(t):
            b, pc = univ[t]
            h_ = list(hist[t])
            closes = [x[0] for x in h_]
            vols = [x[1] for x in h_]
            r1 = closes[-1] / closes[-2] - 1 if len(closes) >= 2 else np.nan
            r5 = closes[-1] / closes[-6] - 1 if len(closes) >= 6 else np.nan
            vr = vols[-1] / np.mean(vols) if len(vols) >= 10 and np.mean(vols) > 0 else np.nan
            lr = last_run.get(t)
            return {"prior_1d": r1, "prior_5d": r5, "vol_ratio_prev": vr,
                    "price": pc, "runner_5d": int(lr is not None and di - lr <= 5),
                    "runner_20d": int(lr is not None and di - lr <= 20),
                    "in_y_top3": int(any(t in s for s in list(top3_hist)[-1:])),
                    "in_y_top50": int(bool(topN_hist) and t in topN_hist[-1])}

        for rank, t in enumerate(top3, 1):
            f = feats(t)
            rows.append({"date": day, "ticker": t, "rank": rank,
                         "gain": round(gains[t], 4),
                         "gap_open": round(univ[t][0][0] / univ[t][1] - 1, 4),
                         "intraday": round(univ[t][0][3] / univ[t][0][0] - 1, 4)
                         if univ[t][0][0] else np.nan,
                         "dollar_vol": univ[t][0][3] * univ[t][0][4], **f})
            offenders[t] += 1
        pool = [t for t in univ if t not in top100 and univ[t][1] <= 25]
        for t in rng.sample(pool, min(3, len(pool))):
            controls.append(feats(t))
        # ex-ante watchlist (FIXED a-priori score, not fitted): runner_20d + vol
        # surge + prior-day strength, K=20
        scored = []
        for t in univ:
            f = feats(t)
            sc = (2 * f["runner_5d"] + f["runner_20d"]
                  + (2 if (f["vol_ratio_prev"] or 0) > 5 else 0)
                  + (2 if (f["prior_1d"] or 0) > 0.15 else 0)
                  + f["in_y_top50"])
            if sc > 0:
                scored.append((sc, t))
        watch = {t for _, t in sorted(scored, reverse=True)[:20]}
        if len(univ) > 100:
            watch_days += 1
            if any(t in watch for t in top3):
                watch_hits += 1
            rnd = set(rng.sample(list(univ), min(20, len(univ))))
            if any(t in rnd for t in top3):
                rand_hits += 1
        day_rows.append({"date": day, "n_univ": len(univ),
                         "top1_gain": round(gains[srt[0]], 3),
                         "vix": vix.get(day), "dow": pd.Timestamp(day).dayofweek})
        for t in top100:
            last_run[t] = di
        top3_hist.append(set(top3))
        topN_hist.append(set(srt[:50]))
        for t, b in bars.items():
            if b[3] and b[4]:
                hist[t].append((b[3], b[4]))

    df = pd.DataFrame(rows)
    dc = pd.DataFrame(controls)
    dd = pd.DataFrame(day_rows)
    df.to_parquet(OUT / "panel.parquet")
    res = {
        "stamp": STAMP, "days": len(dd), "top3_rows": len(df),
        "anatomy": {
            "gain_p50": round(float(df.gain.median()), 3),
            "gain_p90": round(float(df.gain.quantile(.9)), 3),
            "price_p50": round(float(df.price.median()), 2),
            "gap_led_share": round(float((df.gap_open > df.intraday).mean()), 3),
            "dow_counts": df.merge(dd[["date", "dow"]], on="date").dow.value_counts().sort_index().tolist(),
        },
        "exante_lifts_top3_vs_controls": {
            c: {"top3": round(float(df[c].mean()), 3),
                "controls": round(float(dc[c].mean()), 3)}
            for c in ("runner_5d", "runner_20d", "in_y_top3", "in_y_top50")
        } | {
            c: {"top3_p50": round(float(df[c].median()), 3),
                "controls_p50": round(float(dc[c].median()), 3)}
            for c in ("prior_1d", "prior_5d", "vol_ratio_prev", "price")
        },
        "persistence": {
            "P_top3_was_y_top3": round(float(df.in_y_top3.mean()), 3),
            "P_top3_was_y_top50": round(float(df.in_y_top50.mean()), 3),
            "repeat_offenders_5plus": {t: c for t, c in offenders.most_common(15) if c >= 5},
            "unique_tickers": int(df.ticker.nunique()),
        },
        "regime": {
            "top1_gain_by_vix_tercile": None if dd.vix.isna().all() else
            {str(k): round(float(v), 3) for k, v in
             dd.assign(vt=pd.qcut(dd.vix, 3, labels=["low", "mid", "high"]))
             .groupby("vt", observed=True).top1_gain.median().items()},
        },
        "watchlist_K20": {
            "P_hit_at_least_1_of_top3": round(watch_hits / watch_days, 3),
            "random_K20_baseline": round(rand_hits / watch_days, 3),
            "days": watch_days,
            "score": "FIXED a-priori: 2*runner5d + runner20d + 2*(volx>5) + 2*(prior1d>15%) + y_top50",
        },
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str)[:2400])


if __name__ == "__main__":
    main()
