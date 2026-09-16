#!/usr/bin/env python3
"""HYP-120 — THE run. Once. --forecast-only computes and caches forecasts (no statistic)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.hyp111.date_bootstrap import date_block_bootstrap, ci95  # noqa: E402

HYP = "HYP-120"; OUT = ROOT / "data" / "research" / "hyp120"


def qlike(sig, rv):
    s2 = np.maximum(sig, 1e-6) ** 2; return np.log(s2) + (rv ** 2) / s2


def main(argv) -> int:
    doc = prereg.gate_zero(HYP, "start")
    if "--gate-only" in argv: print("gate-only OK"); return 0
    from research.tfm.panel import daily_returns, daily_windows, cascade_windows  # noqa: E402
    from research.tfm.forecast import forecast  # noqa: E402
    OUT.mkdir(parents=True, exist_ok=True)
    fd, fc = OUT / "daily_forecasts.parquet", OUT / "cascade_forecasts.parquet"
    if not fd.exists():
        W = daily_windows(daily_returns()); print(f"daily windows {len(W)}", flush=True)
        pt, sg = forecast([w["ctx"] for w in W], 5)
        D = pd.DataFrame({k: [w[k] for w in W] for k in ("series", "date", "r_next", "rv_next", "rv21", "ewma")})
        D["p1"] = pt[:, 0]; D["sig_tfm"] = np.sqrt((sg ** 2).mean(axis=1)); D.to_parquet(fd, index=False)
    if not fc.exists():
        E = pd.read_parquet(ROOT / "data" / "research" / "hyp119" / "events.parquet"); E = E[E["universe"] == "liquid"]
        C = cascade_windows(E); print(f"cascade windows {len(C)}", flush=True)
        pt, sg = forecast([c["ctx"] for c in C], 30)
        Cd = pd.DataFrame({k: [c[k] for c in C] for k in ("kind", "sym", "date", "m", "z", "range")}); Cd["sig30"] = np.sqrt((sg ** 2).sum(axis=1)); Cd.to_parquet(fc, index=False)
    if "--forecast-only" in argv: print("forecasts cached, nothing computed"); return 0
    D = pd.read_parquet(fd); C = pd.read_parquet(fc); rng = np.random.default_rng(42)
    print(f"\n{HYP}  daily {len(D)} windows / {D['series'].nunique()} series / {D['date'].nunique()} dates   cascade {len(C)} minutes ({(C['kind']=='state').sum()} state)\n")
    # c1 magnitude
    q_t, q_rv, q_ew = qlike(D["sig_tfm"].values, D["rv_next"].values), qlike(D["rv21"].values, D["rv_next"].values), qlike(D["ewma"].values, D["rv_next"].values)
    d1, d2 = q_t - q_rv, q_t - q_ew; dates = D["date"].values
    b1 = date_block_bootstrap(dates, lambda ix: float(d1[ix].mean()), draws=3000, rng=rng); b2 = date_block_bootstrap(dates, lambda ix: float(d2[ix].mean()), draws=3000, rng=rng)
    c1 = ci95(b1)[1] < 0 and ci95(b2)[1] < 0
    sp = {k: spearmanr(D[k], D["rv_next"]).correlation for k in ("sig_tfm", "rv21", "ewma")}
    print(f"(c1) ΔQLIKE vs RV21 {d1.mean():+.4f} CI {np.round(ci95(b1),4)}   vs EWMA {d2.mean():+.4f} CI {np.round(ci95(b2),4)}   {'PASS' if c1 else 'FAIL'}")
    print(f"     Spearman(σ̂, RV): TimesFM {sp['sig_tfm']:.3f}  RV21 {sp['rv21']:.3f}  EWMA {sp['ewma']:.3f}   mean σ̂/RV: {float((D['sig_tfm']/D['rv_next'].clip(lower=1e-6)).median()):.2f}")
    # c2 direction
    hit = (np.sign(D["p1"].values) == np.sign(D["r_next"].values)).astype(float); nz = D["r_next"].values != 0
    bh = date_block_bootstrap(dates[nz], lambda ix: float(hit[nz][ix].mean()), draws=3000, rng=rng); hlo, hhi = ci95(bh)
    ics = D.groupby("series").apply(lambda g: spearmanr(g["p1"], g["r_next"]).correlation); ic_b = ics.values
    ic_lo, ic_hi = np.percentile([rng.choice(ic_b, len(ic_b)).mean() for _ in range(3000)], [2.5, 97.5])
    cs = D.groupby("date").apply(lambda g: spearmanr(g["p1"], g["r_next"]).correlation if len(g) > 5 else np.nan).dropna()
    c2 = hlo > 0.5 and ic_lo > 0
    print(f"(c2) hit {hit[nz].mean():.4f} CI [{hlo:.4f}, {hhi:.4f}]   per-series Spearman mean {ic_b.mean():+.4f} CI [{ic_lo:+.4f}, {ic_hi:+.4f}]   cross-sectional daily IC {cs.mean():+.4f}   {'PASS' if c2 else 'FAIL'}")
    print("     per series:", {k: round(float(v), 3) for k, v in ics.items()})
    # c3 cascade
    Cz = C.copy(); Cz["zfill"] = Cz["z"]
    # non-state minutes have no z in the events file; rank them by the model only vs state-only comparison AND pooled with z = -inf proxy declared: use state-only for the paired comparison
    S = C[C["kind"] == "state"]; s_t = spearmanr(S["sig30"], S["range"]).correlation; s_z = spearmanr(S["z"], S["range"]).correlation
    dts = S["date"].values; sig, zz, rr = S["sig30"].values, S["z"].values, S["range"].values
    bd = date_block_bootstrap(dts, lambda ix: spearmanr(sig[ix], rr[ix]).correlation - spearmanr(zz[ix], rr[ix]).correlation, draws=2000, rng=rng); dlo, dhi = ci95(bd)
    ratio_model = float(C[C["kind"] == "state"]["sig30"].median() / C[C["kind"] == "nonstate"]["sig30"].median()); ratio_real = float(C[C["kind"] == "state"]["range"].median() / C[C["kind"] == "nonstate"]["range"].median())
    pooled_sp = spearmanr(C["sig30"], C["range"]).correlation
    c3 = dlo > 0
    print(f"(c3) within state events: Spearman(TimesFM σ30, range) {s_t:+.3f} vs Spearman(z, range) {s_z:+.3f}   diff CI [{dlo:+.3f}, {dhi:+.3f}]   {'PASS' if c3 else 'FAIL'}")
    print(f"     pooled state∪nonstate Spearman(σ30, range) {pooled_sp:+.3f};  model's state/nonstate σ ratio {ratio_model:.2f} vs realized range ratio {ratio_real:.2f}")
    verdict = "+".join([v for v, ok in (("MAGNITUDE", c1), ("DIRECTION", c2), ("CASCADE", c3)) if ok]) or "NULL"
    print(f"\n=== VERDICT: {verdict} ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(), "n_daily": len(D), "n_cascade": len(C),
           "c1": {"dq_rv21": float(d1.mean()), "ci_rv21": ci95(b1), "dq_ewma": float(d2.mean()), "ci_ewma": ci95(b2), "spearman": {k: float(v) for k, v in sp.items()}, "pass": bool(c1)},
           "c2": {"hit": float(hit[nz].mean()), "hit_ci": [hlo, hhi], "ic_mean": float(ic_b.mean()), "ic_ci": [float(ic_lo), float(ic_hi)], "cs_ic": float(cs.mean()), "per_series": {k: float(v) for k, v in ics.items()}, "pass": bool(c2)},
           "c3": {"sp_tfm": float(s_t), "sp_z": float(s_z), "diff_ci": [dlo, dhi], "pooled_sp": float(pooled_sp), "ratio_model": ratio_model, "ratio_real": ratio_real, "pass": bool(c3)}, "verdict": verdict}
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=float))
    prereg.adjudicate(HYP, verdict, verdict, {"result_file": "data/research/hyp120/result.json"}); prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
