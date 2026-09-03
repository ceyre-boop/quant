#!/usr/bin/env python3
"""Development run — 2006-2026 only (holdout 1990-2005 sealed by the data guard). Walk-forward, no
lookback: trained models score month t using fits on months < t (min 120, refit yearly). Outputs
data/research/carry_model/dev_results.json. This is Step 4 of MODEL_DESIGN.md §7 on the dev window;
it is NOT the sealed test. Every model here counts toward the multiplicity declared in HYP-117."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.carry_model.panel import build, zscore_cs, FEATS_CALC  # noqa: E402
from research.carry_model.models import MODELS  # noqa: E402
from research.carry_model.evaluate import summarize, perm_null_topk, topk_returns, monthly_ic  # noqa: E402

OUT = ROOT / "data" / "research" / "carry_model"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    P, F, R = build()
    P = zscore_cs(P, sorted(set(FEATS_CALC)))
    P = P.reset_index(drop=True)
    factor = F["factor"]
    print(f"dev panel {P['date'].min().date()} → {P['date'].max().date()}  {len(P)} pair-months  {P['pair'].nunique()} pairs")
    print(f"carry factor on dev window: ann {factor.loc[P['date'].min():].mean()*12*100:+.2f}%\n")
    results = {}
    for name, fn in MODELS.items():
        s = fn(P)
        mask = s.notna()
        if mask.sum() == 0: continue
        res = summarize(name, P[mask], s[mask], factor)
        res["perm_p_top5"] = perm_null_topk(P[mask], s[mask], 5, draws=1000)
        results[name] = res
        print(f"{name}\n  window {res['window'][0]}→{res['window'][1]} ({res['months']} m)   IC {res['ic_mean']:+.3f} CI [{res['ic_ci'][0]:+.3f}, {res['ic_ci'][1]:+.3f}]  years>0 {res['ic_years_pos']}")
        print(f"  top-5: ann {res['top5']['ann']*100:+.2f}%  Sharpe {res['top5']['sharpe']:.2f}  maxDD {res['top5']['max_dd']*100:.1f}%  hit {res['top5']['hit']:.2f}   perm p {res['perm_p_top5']:.3f}")
        print(f"  top-1: ann {res['top1']['ann']*100:+.2f}%  Sharpe {res['top1']['sharpe']:.2f}  maxDD {res['top1']['max_dd']*100:.1f}%")
        print(f"  carry factor same window: ann {res['carry_factor_same_window']['ann']*100:+.2f}% Sharpe {res['carry_factor_same_window']['sharpe']:.2f}   ΔSharpe(top5−factor) CI [{res['sharpe_diff_top5_vs_factor'][0]:+.2f}, {res['sharpe_diff_top5_vs_factor'][1]:+.2f}]")
        print(f"  IC by year: {res['ic_by_year']}\n")
    # oracle on the same panel, for scale
    orc = P.groupby("date")["target"].apply(lambda t: t.abs().max())
    print(f"oracle (best signed pair each month) on this retail-adjusted panel: ann {orc.mean()*12*100:+.1f}%")
    results["_oracle_ann"] = float(orc.mean() * 12); results["_n_models"] = len(MODELS)
    (OUT / "dev_results.json").write_text(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
