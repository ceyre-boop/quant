#!/usr/bin/env python3
"""HYP-113 — THE test. Runs once. Ladder exactly as sealed."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.hyp111.engine import daily  # noqa: E402
from research.hyp111.date_bootstrap import date_block_bootstrap, ci95  # noqa: E402

HYP = "HYP-113"
OUT = ROOT / "data" / "research" / "hyp111"


def main(argv) -> int:
    doc = prereg.gate_zero(HYP, "start")
    if "--gate-only" in argv:
        print("gate-only OK"); return 0
    P = doc["frozen_parameters"]; edges = P["bins"]
    ev = pd.read_parquet(OUT / "hyp111a_trades.parquet")[["sym", "t", "t1", "s", "naive_net"]]
    ev["fade"] = -ev["naive_net"]
    # size measure: trailing-252 percentile rank of |r_t|
    xs = []
    for sym, g in ev.groupby("sym"):
        d = daily(sym); a = d["r"].abs()
        for _, row in g.iterrows():
            i = d.index.get_loc(pd.Timestamp(row["t"]))
            prior = a.iloc[i - 252:i].values
            xs.append((row["sym"], row["t"], float((prior < a.iloc[i]).mean())))
    x = pd.DataFrame(xs, columns=["sym", "t", "x"])
    ev = ev.merge(x, on=["sym", "t"])
    ev["bin"] = pd.cut(ev["x"], bins=edges, labels=["B1", "B2", "B3"], right=False, include_lowest=True)
    ev.loc[ev["x"] >= edges[-1], "bin"] = "B3"
    rng = np.random.default_rng(P["seed"])
    print(f"\n{HYP} — {doc['name']}\n{len(ev)} events; x range [{ev['x'].min():.3f}, {ev['x'].max():.3f}]\n")

    def claim(sub: pd.DataFrame, label: str, min_n: int):
        means = sub.groupby("bin", observed=False)["fade"].agg(["mean", "count"])
        print(f"[{label}] n={len(sub)}"); print((means.assign(mean=means["mean"] * 100)).round(3).to_string())
        if (means["count"] < min_n).any():
            print(f"  -> INCONCLUSIVE (bin < {min_n})"); return "INCONCLUSIVE", {"bins": means.to_dict()}
        mono = bool(means.loc["B1", "mean"] <= means.loc["B2", "mean"] <= means.loc["B3", "mean"])
        xx, yy, dd = sub["x"].values, sub["fade"].values, sub["t1"].values
        slope = lambda ix: float(np.polyfit(xx[ix], yy[ix], 1)[0]) if xx[ix].std() > 0 else 0.0
        sb = date_block_bootstrap(dd, slope, L=P["block_L"], draws=P["draws"], rng=rng)
        lo, hi = ci95(sb); s0 = slope(np.arange(len(sub)))
        # per-bin CIs (descriptive)
        cis = {}
        for b in ("B1", "B2", "B3"):
            m = (sub["bin"] == b).values
            bb = date_block_bootstrap(dd[m], lambda ix, m=m: yy[m][ix].mean(), L=P["block_L"], draws=2000, rng=rng)
            cis[b] = [float(v * 100) for v in ci95(bb)]
        print(f"  slope {s0*100:+.3f}% per unit rank  CI [{lo*100:+.3f}, {hi*100:+.3f}]  monotone {mono}  bin CIs(%) {cis}")
        v = "DOSE_RESPONSE" if (mono and lo > 0) else "FLAT"
        print(f"  -> {v}"); return v, {"bins": means.to_dict(), "slope": s0, "slope_ci": [lo, hi], "monotone": mono, "bin_ci_pct": cis}

    v1, r1 = claim(ev, "primary: all shocks", 40)
    v2, r2 = claim(ev[ev["s"] < 0], "secondary: down-shocks", 25)
    _, r3 = claim(ev[ev["s"] > 0], "descriptive: up-shocks", 25)
    per_inst = ev.groupby(["sym", "bin"], observed=False)["fade"].mean().unstack().round(4)
    print("\nper-instrument bin means (%):"); print((per_inst * 100).round(3).to_string())
    print(f"\n=== VERDICT primary: {v1}   secondary (down): {v2} ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(), "n": len(ev),
           "primary": {"verdict": v1, **r1}, "secondary_down": {"verdict": v2, **r2}, "up_descriptive": r3,
           "per_instrument": per_inst.to_dict()}
    (OUT / "hyp113_result.json").write_text(json.dumps(res, indent=2, default=str))
    ev.to_parquet(OUT / "hyp113_events.parquet", index=False)
    prereg.adjudicate(HYP, v1, f"primary {v1}; down-only {v2}", {"result_file": "data/research/hyp111/hyp113_result.json"})
    prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
