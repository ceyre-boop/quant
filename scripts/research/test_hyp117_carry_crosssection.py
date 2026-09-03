#!/usr/bin/env python3
"""HYP-117 — THE holdout run. Once. Unlocks the data guard AFTER gate zero, trains the fitted models once
on 2006-2026, scores 1990-2005, applies the sealed ladder, writes the ledger."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402

HYP = "HYP-117"; OUT = ROOT / "data" / "research" / "hyp117"


def main(argv) -> int:
    doc = prereg.gate_zero(HYP, "start")
    if "--gate-only" in argv:
        print("gate-only OK"); return 0
    os.environ["CARRY_HOLDOUT_UNLOCK"] = "1"                       # the only place this is ever set
    from research.carry_model.panel import build, zscore_cs, FEATS_CALC, FEATS_ALL, FEATS_INTER, FEATS_BASE  # noqa: E402
    from research.carry_model.evaluate import summarize, perm_null_topk, topk_returns, monthly_ic  # noqa: E402
    from sklearn.linear_model import Ridge  # noqa: E402
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
    OUT.mkdir(parents=True, exist_ok=True)
    h0, h1 = doc["data"]["holdout"]
    P, F, R = build(unlock_holdout=True); P = zscore_cs(P, sorted(set(FEATS_CALC))).reset_index(drop=True)
    factor = F["factor"]
    dev = P[P["date"] > pd.Timestamp("2005-12-31")]; ho = P[(P["date"] >= h0) & (P["date"] <= h1)]
    print(f"\n{HYP}  holdout {ho['date'].min().date()} → {ho['date'].max().date()}  {len(ho)} pair-months, {ho['date'].nunique()} months, {ho['pair'].nunique()} pairs")
    print(f"carry factor on holdout: ann {factor.loc[h0:h1].mean()*12*100:+.2f}%   train (dev) months {dev['date'].nunique()}\n")
    z = lambda fs: [f + "_z" for f in fs]
    def fit_apply(feats, mk):
        m = mk().fit(dev[z(feats)].fillna(0).values, dev["target"].values)
        return pd.Series(m.predict(ho[z(feats)].fillna(0).values), index=ho.index)
    mono = [1 if f in ("carry", "mom12", "value") else (-1 if f == "rvol3" else 0) for f in FEATS_ALL]
    scores = {
        "M1 baseline": ho[z(FEATS_BASE)].fillna(0).sum(axis=1),
        "M2 ridge": fit_apply(FEATS_ALL, lambda: Ridge(alpha=10.0)),
        "M3 ridge+interactions": fit_apply(FEATS_INTER, lambda: Ridge(alpha=10.0)),
        "M4 boosting": fit_apply(FEATS_ALL, lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, monotonic_cst=mono, random_state=42)),
        "M5 SHOT": fit_apply(FEATS_CALC, lambda: Ridge(alpha=10.0)) * ho["rvol3"].fillna(ho["rvol3"].median()),
        "M5b SHOT-pure": np.sign(ho[z(FEATS_BASE)].fillna(0).sum(axis=1)) * ho["rvol3"].fillna(ho["rvol3"].median()),
    }
    res = {}; pvals = {}
    for name, s in scores.items():
        r = summarize(name, ho, s, factor); r["perm_p_top5"] = perm_null_topk(ho, s, 5, draws=2000); pvals[name] = r["perm_p_top5"]; res[name] = r
    # BH across the six
    names = sorted(pvals, key=pvals.get); m = len(names); bh = {}
    for i, nm in enumerate(names, 1): bh[nm] = pvals[nm] * m / i
    for nm in names: res[nm]["perm_p_bh"] = min(1.0, bh[nm])
    verdicts = {}
    for nm, r in res.items():
        ic_pos = r["ic_ci"][0] > 0; port = r["sharpe_diff_top5_vs_factor"][0] > 0; perm = r["perm_p_bh"] < 0.05
        verdicts[nm] = "MODEL_HOLDS" if (ic_pos and port and perm) else ("IC_ONLY" if ic_pos else "NULL")
        print(f"{nm}\n  IC {r['ic_mean']:+.3f} CI [{r['ic_ci'][0]:+.3f}, {r['ic_ci'][1]:+.3f}]  years>0 {r['ic_years_pos']}   top-5 ann {r['top5']['ann']*100:+.2f}% Sh {r['top5']['sharpe']:.2f} maxDD {r['top5']['max_dd']*100:.1f}%   "
              f"factor Sh {r['carry_factor_same_window']['sharpe']:.2f}  ΔSh CI [{r['sharpe_diff_top5_vs_factor'][0]:+.2f}, {r['sharpe_diff_top5_vs_factor'][1]:+.2f}]   perm p {r['perm_p_top5']:.3f} (BH {r['perm_p_bh']:.3f})   top-1 ann {r['top1']['ann']*100:+.2f}%  → {verdicts[nm]}")
        print(f"  IC by year: {r['ic_by_year']}")
    # crashes, descriptive: top-5 of M1 and factor in 1992-07..1993-03 and 1998-06..1998-12
    r5 = topk_returns(ho, scores["M1 baseline"], 5)
    for lab, a, b in (("1992 ERM", "1992-07-01", "1993-03-31"), ("1998 LTCM/JPY", "1998-06-01", "1998-12-31")):
        seg = r5.loc[a:b]; fseg = factor.loc[a:b]
        print(f"  {lab}: M1 top-5 {' '.join(f'{v*100:+.1f}' for v in seg.values)} | factor {' '.join(f'{v*100:+.1f}' for v in fseg.values)}")
    overall = "MODEL_HOLDS" if "MODEL_HOLDS" in verdicts.values() else ("IC_ONLY" if "IC_ONLY" in verdicts.values() else "NULL")
    orc = ho.groupby("date")["target"].apply(lambda t: t.abs().max())
    print(f"\noracle on holdout: ann {orc.mean()*12*100:+.1f}%\n=== VERDICT: {overall}   per model {verdicts} ===\n")
    out = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(), "holdout": [str(ho['date'].min().date()), str(ho['date'].max().date())],
           "months": int(ho["date"].nunique()), "factor_ann": float(factor.loc[h0:h1].mean() * 12), "oracle_ann": float(orc.mean() * 12), "models": res, "verdicts": verdicts, "verdict": overall}
    (OUT / "result.json").write_text(json.dumps(out, indent=2, default=float))
    prereg.adjudicate(HYP, overall, json.dumps(verdicts), {"result_file": "data/research/hyp117/result.json"})
    prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
