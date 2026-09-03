#!/usr/bin/env python3
"""HYP-118 — THE run. Once."""
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
from research.hyp111.date_bootstrap import stationary_block_indices  # noqa: E402

HYP = "HYP-118"; OUT = ROOT / "data" / "research" / "hyp118"


def sharpe(x): s = np.std(x, ddof=1); return float(np.mean(x) / s * np.sqrt(12)) if s > 0 else 0.0
def max_dd(x): eq = np.cumprod(1 + np.asarray(x)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def main(argv) -> int:
    doc = prereg.gate_zero(HYP, "start")
    if "--gate-only" in argv: print("gate-only OK"); return 0
    os.environ["EM_HOLDOUT_UNLOCK"] = "1"
    from research.carry_model.em import build_em  # noqa: E402
    from research.carry_model.m6 import run, carry_factor, score  # noqa: E402
    from research.carry_model.evaluate import monthly_ic, perm_null_topk  # noqa: E402
    OUT.mkdir(parents=True, exist_ok=True)
    P, vix = build_em(); P = P.reset_index(drop=True)
    print(f"\n{HYP}  EM universe {P['date'].min().date()} → {P['date'].max().date()}  {P['date'].nunique()} months  {P['pair'].nunique()} positions\n")
    m6 = run(P, 5); m6b = run(P, 5, dd_boost=True); plain = carry_factor(P, 5)
    idx = m6["managed"].index.intersection(plain.index); a, b, raw, bb = m6["managed"].reindex(idx).values, plain.reindex(idx).values, m6["raw"].reindex(idx).values, m6b["managed"].reindex(idx).values
    ic = monthly_ic(P, m6["score"]); rng = np.random.default_rng(42)
    icb = np.array([ic.values[stationary_block_indices(rng, len(ic), 6)].mean() for _ in range(5000)]); ic_lo, ic_hi = np.percentile(icb, [2.5, 97.5])
    dif = np.array([sharpe(a[ix]) - sharpe(b[ix]) for ix in (stationary_block_indices(rng, len(a), 6) for _ in range(5000))]); dlo, dhi = np.percentile(dif, [2.5, 97.5])
    perm_p = perm_null_topk(P, m6["score"], 5, draws=2000)
    c1, c2, c3, c4 = ic_lo > 0, dlo > 0, max_dd(a) >= (2 / 3) * max_dd(b), perm_p < 0.05
    def line(nm, x): print(f"  {nm:34s} ann {np.mean(x)*12*100:+6.2f}%  Sharpe {sharpe(x):5.2f}  maxDD {max_dd(x)*100:6.1f}%  hit {(x>0).mean():.2f}  worst {x.min()*100:+.1f}%")
    line("plain EM carry (top-5, EW)", b); line("M6 raw (rank + risk parity)", raw); line("M6 managed (vol target + VIX brake)", a); line("M6b managed + DD boost", bb)
    print(f"\n(c1) IC {ic.mean():+.3f} CI [{ic_lo:+.3f}, {ic_hi:+.3f}] years>0 {int((ic.groupby(ic.index.year).mean()>0).sum())}/{ic.index.year.nunique()}  {'PASS' if c1 else 'FAIL'}")
    print(f"(c2) ΔSharpe M6−plain {sharpe(a)-sharpe(b):+.2f} CI [{dlo:+.2f}, {dhi:+.2f}]  {'PASS' if c2 else 'FAIL'}")
    print(f"(c3) maxDD M6 {max_dd(a)*100:.1f}% vs plain {max_dd(b)*100:.1f}% (need ≤ {max_dd(b)*2/3*100:.1f}%)  {'PASS' if c3 else 'FAIL'}")
    print(f"(c4) perm p {perm_p:.3f}  {'PASS' if c4 else 'FAIL'}")
    yrs = pd.Series(a, index=idx).groupby(idx.year).sum(); pyrs = pd.Series(b, index=idx).groupby(idx.year).sum()
    print("\nby year (%): M6 managed / plain"); print("  " + "  ".join(f"{y}:{v*100:+.0f}/{pyrs[y]*100:+.0f}" for y, v in yrs.items()))
    print("mean exposure", round(float(m6["exposure"].mean()), 2), " months at 0.5× VIX brake", int((m6["exposure"] < 0.6).sum()))
    per = P.assign(s=m6["score"].values).groupby("a").apply(lambda g: float((np.sign(g["s"]) * g["target"]).mean() * 12)); print("per EM currency signed ann (M6 sign):", (per * 100).round(2).to_dict())
    verdict = "ADVANCED_HOLDS" if (c1 and c2 and c3 and c4) else ("RISK_ONLY" if (c1 and c3 and c4) else ("IC_ONLY" if (c1 and c4) else "NULL"))
    print(f"\n=== VERDICT: {verdict} ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(), "months": len(idx),
           "plain": {"ann": float(np.mean(b) * 12), "sharpe": sharpe(b), "max_dd": max_dd(b)}, "m6_raw": {"ann": float(np.mean(raw) * 12), "sharpe": sharpe(raw), "max_dd": max_dd(raw)},
           "m6": {"ann": float(np.mean(a) * 12), "sharpe": sharpe(a), "max_dd": max_dd(a)}, "m6b": {"ann": float(np.mean(bb) * 12), "sharpe": sharpe(bb), "max_dd": max_dd(bb)},
           "ic": {"mean": float(ic.mean()), "ci": [float(ic_lo), float(ic_hi)], "by_year": {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()}},
           "dsharpe_ci": [float(dlo), float(dhi)], "perm_p": perm_p, "claims": {"c1": c1, "c2": c2, "c3": c3, "c4": c4},
           "by_year": {int(y): {"m6": float(v), "plain": float(pyrs[y])} for y, v in yrs.items()}, "verdict": verdict}
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame({"m6": a, "m6b": bb, "raw": raw, "plain": b}, index=idx).to_parquet(OUT / "monthly.parquet")
    prereg.adjudicate(HYP, verdict, verdict, {"oos_sharpe": sharpe(a), "p_value": perm_p, "result_file": "data/research/hyp118/result.json"})
    prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
