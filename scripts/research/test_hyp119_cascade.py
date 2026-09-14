#!/usr/bin/env python3
"""HYP-119 — THE run. Once. --gate-only checks wiring; --build-only builds and caches the event panel
(computes no statistic)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.hyp111.date_bootstrap import date_block_bootstrap, ci95  # noqa: E402
from research.formal.mechanics import max_loss_before_halt  # noqa: E402
from backtester.realistic_fills import _half_spread_measured, HALT_RESUME_SLIP  # noqa: E402
from backtester.luld import halt_flags  # noqa: E402

HYP = "HYP-119"; OUT = ROOT / "data" / "research" / "hyp119"


def precompute(e: pd.DataFrame, slip: float) -> dict:
    """Per-event quantities computed ONCE (the slow part); a position's net in either direction is then closed-form:
    net(d) = d·(exit_d − entry_d)/entry_d with entry_d = o·(1+s·d), exit_d = X·(1 − c·d), c = spread s or halt slip."""
    o = np.empty(len(e)); s = np.empty(len(e)); X = np.empty(len(e)); c = np.empty(len(e)); lo = np.empty(len(e)); hi = np.empty(len(e)); halted = np.zeros(len(e), bool)
    for i, ev in enumerate(e.to_dict("records")):
        f = pd.DataFrame(ev["bars_fwd"]); o[i] = ev["entry_open"]
        s[i] = _half_spread_measured(o[i], max(ev["spread_proxy"], 1e-4), max(ev["minute_dollar_vol"], 1e3))
        hf = halt_flags(f) if len(f) else np.zeros(0, bool); h = np.flatnonzero(hf)
        if len(h) and h[0] > 0:
            X[i] = float(f["open"].iloc[h[0]]); c[i] = slip; path = f.iloc[: h[0]]; halted[i] = True
        else:
            X[i] = float(f["close"].iloc[-1]); c[i] = s[i]; path = f
        lo[i] = float(path["low"].min()); hi[i] = float(path["high"].max())
    return {"o": o, "s": s, "X": X, "c": c, "lo": lo, "hi": hi, "halted": halted}


def nets(pc: dict, d: np.ndarray) -> np.ndarray:
    entry = pc["o"] * (1 + pc["s"] * d); exit_px = pc["X"] * (1 - pc["c"] * d)
    return d * (exit_px - entry) / entry


def adverse(pc: dict, d: np.ndarray) -> np.ndarray:
    entry = pc["o"] * (1 + pc["s"] * d)
    return np.where(d > 0, (entry - pc["lo"]) / entry, (pc["hi"] - entry) / entry)


def main(argv) -> int:
    doc = prereg.gate_zero(HYP, "start")
    if "--gate-only" in argv: print("gate-only OK"); return 0
    from research.cascade.panel import build  # noqa: E402
    OUT.mkdir(parents=True, exist_ok=True); P = doc["frozen_parameters"]
    cache = OUT / "events.parquet"
    if cache.exists() and "--rebuild" not in argv:
        E = pd.read_parquet(cache); meta = json.loads((OUT / "meta.json").read_text())
    else:
        E, m = build(); ns = m["nonstate"]
        E.to_parquet(cache, index=False); ns.to_parquet(OUT / "nonstate.parquet", index=False)
        meta = {"n_liquid_files": m["n_liquid_files"], "n_micro_files": m["n_micro_files"]}; (OUT / "meta.json").write_text(json.dumps(meta))
    NS = pd.read_parquet(OUT / "nonstate.parquet")
    print(f"\n{HYP}  events: {E.groupby('universe').size().to_dict()}  files liquid {meta['n_liquid_files']} micro {meta['n_micro_files']}")
    if "--build-only" in argv: print("build-only: panel cached, nothing computed"); return 0
    rng = np.random.default_rng(P["seed"]); res = {"universes": {}}; verdicts = {}
    for uni in ("liquid", "microcap"):
        e = E[E["universe"] == uni].copy()
        if uni == "microcap": e = e[e["direction"] > 0]                       # long-side only (declared)
        if len(e) < P["min_events_per_universe"]:
            print(f"{uni}: {len(e)} events < {P['min_events_per_universe']} → INCONCLUSIVE"); verdicts[uni] = "INCONCLUSIVE"; res["universes"][uni] = {"n": int(len(e))}; continue
        # (a) magnitude
        fr = e["bars_fwd"].apply(lambda d: float(np.log(max(d["high"]) / min(d["low"]))))
        e = e.assign(fwd_range=fr.values)
        ns_med = NS[NS["universe"] == uni].dropna(); ratio = float(np.median(fr) / np.median(ns_med["range"]))
        b = date_block_bootstrap(e["date"].values, lambda ix: float(np.median(fr.values[ix]) / np.median(ns_med["range"])), L=P["block_L"], draws=P["draws"], rng=rng)
        alo, ahi = ci95(b); a_pass = alo > 1
        # (b) continuation
        pc = precompute(e, P["halt_slip"]); d0 = e["direction"].values.astype(float)
        net = nets(pc, d0); adv = adverse(pc, d0)
        bb = date_block_bootstrap(e["date"].values, lambda ix: float(net[ix].mean()), L=P["block_L"], draws=P["draws"], rng=rng); blo, bhi = ci95(bb)
        plac = np.array([nets(pc, rng.choice([-1.0, 1.0], len(e))).mean() for _ in range(P["placebo_draws"])])
        p_plac = float((plac >= net.mean()).mean())
        signs = rng.choice([-1, 1], size=(2000, len(net))); perm = (signs * net).mean(axis=1); p_perm = float((perm >= net.mean()).mean())
        b_pass = blo > 0 and p_plac < 0.05 and p_perm < 0.05
        # (c) bound
        bounds = np.array([max_loss_before_halt(r["entry_open"], dtime(*map(int, r["time"].split(":"))), 2, r["prior_closes"]).max_adverse_frac for r in e.to_dict("records")])
        breach = int((adv > bounds + 1e-9).sum()); c_pass = breach == 0
        print(f"\n{uni}: n={len(e)}  (a) range ratio {ratio:.2f} CI [{alo:.2f}, {ahi:.2f}] {'PASS' if a_pass else 'FAIL'}   (b) net {net.mean()*100:+.3f}%/trade CI [{blo*100:+.3f}, {bhi*100:+.3f}] placebo p {p_plac:.3f} perm p {p_perm:.3f} {'PASS' if b_pass else 'FAIL'}   "
              f"(c) worst adverse {adv.max()*100:.1f}% vs proven bound median {np.median(bounds)*100:.1f}%, breaches {breach} {'PASS' if c_pass else 'FAIL'}")
        print(f"   hit {(net>0).mean():.2f}  by direction: up {net[e['direction']>0].mean()*100:+.3f}% (n={int((e['direction']>0).sum())})  down {net[e['direction']<0].mean()*100:+.3f}% (n={int((e['direction']<0).sum())})   halts hit {int(pc['halted'].sum())}")
        verdicts[uni] = "CASCADE_TRADEABLE" if (a_pass and b_pass and c_pass) else ("MAGNITUDE_ONLY" if a_pass else "NULL")
        res["universes"][uni] = {"n": int(len(e)), "a": {"ratio": ratio, "ci": [alo, ahi], "pass": a_pass}, "b": {"net": float(net.mean()), "ci": [blo, bhi], "p_placebo": p_plac, "p_perm": p_perm, "pass": b_pass},
                                 "c": {"worst_adverse": float(adv.max()), "bound_median": float(np.median(bounds)), "breaches": breach, "pass": c_pass}, "verdict": verdicts[uni]}
    overall = "CASCADE_TRADEABLE" if "CASCADE_TRADEABLE" in verdicts.values() else ("MAGNITUDE_ONLY" if "MAGNITUDE_ONLY" in verdicts.values() else ("INCONCLUSIVE" if all(v == "INCONCLUSIVE" for v in verdicts.values()) else "NULL"))
    print(f"\n=== VERDICT: {overall}   per universe {verdicts} ===\n")
    res.update({"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(), "verdict": overall})
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=float))
    prereg.adjudicate(HYP, overall, json.dumps(verdicts), {"result_file": "data/research/hyp119/result.json"}); prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
