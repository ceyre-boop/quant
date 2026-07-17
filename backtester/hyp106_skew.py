"""HYP-106 — catching the big-% moves: leak-free pre-entry filter + convex exit.

Only the 09:30 first-minute bar is closed at a 09:31 entry, so features use ONLY
that bar + prev_close + catalyst. Label uses the post-entry return (outcome).
Positive-skew objective: tail_ratio = avg_win / |avg_loss| and P(ret > +20%),
not win rate.

Phase 2 (in-sample) here. Holdout runner is hyp106_holdout via the same helpers.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine
from ._gapper_events import load_events, build_cache

REPO = Path(__file__).resolve().parents[1]
ENR = REPO / "data/research/gapper/per_candidate_enriched.csv"
OUT = REPO / "data/scan_results"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "2026-04-08"
SIZING = 0.02


# ---------- leak-free features at 09:31 ----------
def leakfree_features(ev, cache):
    enr = {(r["date"], r["ticker"]): r for r in csv.DictReader(open(ENR))}
    feats = {}
    for _, e in ev.iterrows():
        d, t = e["date"], e["ticker"]
        bars = cache.get((t, d))
        if bars is None or len(bars) < 3:
            continue
        i0 = bars.index[bars["time"] == "09:30"]
        if not len(i0):
            continue
        b0 = bars.loc[i0[0]]
        o, h, l, c, v = (float(b0["open"]), float(b0["high"]), float(b0["low"]),
                         float(b0["close"]), float(b0["volume"]))
        rng = (h - l) / o if o else 0.0
        row = enr.get((d, t), {})
        try:
            prev_close = float(row.get("prev_close", 0)) or o
            og = o / prev_close - 1 if prev_close else 0.0
        except ValueError:
            og = 0.0
        cat = (row.get("catalyst") or "NONE").upper()
        try:
            dow = float(row.get("dow", 0))
        except ValueError:
            dow = 0.0
        feats[(d, t)] = dict(
            overnight_gap=og,
            first_bar_ret=(c / o - 1) if o else 0.0,
            first_bar_range=rng,
            first_bar_pos=((c - l) / (h - l)) if h > l else 0.5,
            first_bar_vol=v,
            log_vol=np.log10(v + 1),
            prev_close=prev_close,
            dow=dow,
            catalyst=cat,
        )
    return feats


# ---------- skew metrics ----------
def skew_metrics(rets):
    r = np.array(rets)
    wins = r[r > 0]; losses = r[r < 0]
    aw = wins.mean() if len(wins) else 0.0
    al = -losses.mean() if len(losses) else 1e-9
    sharpe = (r.mean() / r.std() * np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else 0.0
    return dict(n=len(r), mean=round(float(r.mean()), 4),
                median=round(float(np.median(r)), 4),
                tail_ratio=round(float(aw / al), 3) if al else 0.0,
                p_win=round(float((r > 0).mean()), 3),
                p_ret_gt20=round(float((r > 0.20).mean()), 3),
                max_win=round(float(r.max()), 3),
                sharpe=round(float(sharpe), 3))


def long_rets(ev, cfg, cache, keys=None):
    res = engine.run(ev, cfg, data_cache=cache, write_audit=False)
    out = []
    for r in res["records"]:
        if r.get("trade_taken") and (keys is None or (r["date"], r["ticker"]) in keys):
            out.append(((r["date"], r["ticker"]), r["net_pct"]))
    return out


EXITS = {
    "to_1030": dict(exit_time="10:30"),
    "to_1100": dict(exit_time="11:00"),
    "to_close": dict(exit_time="15:45"),
    "trail10": dict(exit_time="15:45", trail_pct=0.10),
    "trail15": dict(exit_time="15:45", trail_pct=0.15),
    "trail20": dict(exit_time="15:45", trail_pct=0.20),
    "trail25": dict(exit_time="15:45", trail_pct=0.25),
}


def main():
    ev = load_events()
    cache = build_cache(ev)
    ins = ev[ev.date < SPLIT].reset_index(drop=True)
    feats = leakfree_features(ins, cache)

    # 1) convex-exit scan (skew engineering), 25% hard stop, entry 09:31
    print("=== CONVEX EXIT SCAN (in-sample, long 09:31, 25% stop) ===")
    exit_rows = []
    base_cfg = None
    for name, ex in EXITS.items():
        cfg = dict(entry_time="09:31", direction="long", stop_pct=0.25,
                   sizing_pct=SIZING, locate_required=False, slippage=0.005, **ex)
        rets = [v for _, v in long_rets(ins, cfg, cache)]
        m = skew_metrics(rets)
        exit_rows.append(dict(exit=name, **m))
    edf = pd.DataFrame(exit_rows).sort_values("tail_ratio", ascending=False)
    print(edf.to_string())
    best_exit = edf.iloc[0]["exit"]

    # 2) leak-free feature -> big-winner (top tercile of to_close return)
    cfg_lbl = dict(entry_time="09:31", direction="long", stop_pct=0.25,
                   sizing_pct=SIZING, locate_required=False, slippage=0.005,
                   **EXITS[best_exit])
    rr = dict(long_rets(ins, cfg_lbl, cache))
    keys = [k for k in feats if k in rr]
    y_ret = np.array([rr[k] for k in keys])
    thr = np.quantile(y_ret, 2 / 3)
    y = (y_ret >= thr).astype(int)
    Fnum = ["overnight_gap", "first_bar_ret", "first_bar_range", "first_bar_pos",
            "log_vol", "prev_close"]
    X = np.array([[feats[k][f] for f in Fnum] for k in keys])
    print(f"\n=== LEAK-FREE FEATURE SCAN (label=top-tercile winner, thr={thr:.3f}) ===")
    from scipy.stats import mannwhitneyu
    lifts = []
    for j, f in enumerate(Fnum):
        hi = X[y == 1, j]; loww = X[y == 0, j]
        try:
            p = mannwhitneyu(hi, loww, alternative="two-sided").pvalue
        except ValueError:
            p = 1.0
        lifts.append((f, round(float(np.median(hi)), 4),
                      round(float(np.median(loww)), 4), round(float(p), 4)))
    for f, mh, ml, p in sorted(lifts, key=lambda x: x[3]):
        print(f"  {f:16s} med_winner={mh:>10}  med_rest={ml:>10}  MW_p={p}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    lr = cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=5).mean()
    rf = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=42)
    rfcv = cross_val_score(rf, X, y, cv=5).mean()
    rf.fit(X, y)
    print(f"\nLEAK-FREE CV acc: logreg {lr:.3f} | rf {rfcv:.3f} "
          f"(base rate {y.mean():.3f})")
    print("rf importances:",
          {f: round(float(i), 3) for f, i in zip(Fnum, rf.feature_importances_)})

    out = dict(best_exit=best_exit, exit_scan=edf.to_dict("records"),
               label_threshold=round(float(thr), 4),
               feature_mw=lifts, cv_logreg=round(float(lr), 3),
               cv_rf=round(float(rfcv), 3), base_rate=round(float(y.mean()), 3),
               rf_importances={f: round(float(i), 3)
                               for f, i in zip(Fnum, rf.feature_importances_)})
    (OUT / "hyp106_insample.json").write_text(json.dumps(out, indent=2))
    print("\nwrote hyp106_insample.json")


if __name__ == "__main__":
    main()
