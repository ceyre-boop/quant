#!/usr/bin/env python3
"""HYP-111 — THE scoped test (Alpaca SIP, 2020-2026). Runs once. --fetch-only fills the minute cache (computes nothing);
--gate-only checks wiring. Verdict ladder exactly as sealed (41b7b6a15a9f2f1e)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg, alpaca_1m as th  # noqa: E402
from research.hyp111.engine import daily, events, simulate, naive, confluence, sharpe_dates, trade_dict  # noqa: E402
from research.hyp111.date_bootstrap import date_block_bootstrap, ci95  # noqa: E402
from sovereign.discovery.cpcv import combinatorial_purged_splits  # noqa: E402
from sovereign.discovery.gate import deflated_sharpe_ratio  # noqa: E402

HYP = "HYP-111"
OUT = ROOT / "data" / "research" / "hyp111"


def build_events(doc):
    w = doc["data"]["window_t1"]; excl = set(doc["data"]["probe_dates_excluded"])
    ev = []
    for sym in doc["instrument_set"]:
        for e in events(daily(sym), w[0], w[1], excl):
            e["sym"] = sym; ev.append(e)
    return ev


def main(argv) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--gate-only", action="store_true"); ap.add_argument("--max-days", type=int, default=100000)
    a = ap.parse_args(argv)
    doc = prereg.gate_zero(HYP, "start")
    if a.gate_only:
        print("gate-only OK"); return 0
    P = doc["frozen_parameters"]
    ev = build_events(doc)
    need = sorted({(e["sym"], e["t1"]) for e in ev} | {(("QQQ" if e["sym"] == "SPY" else "SPY"), e["t1"]) for e in ev})
    if a.fetch_only:
        done = 0
        for sym, d in need:
            if (th.CACHE / f"{sym}_{d}.parquet").exists():
                continue
            th.stock_1m(sym, d); done += 1
            if done % 50 == 0: print(f"  fetched {done}")
            if done >= a.max_days: break
        missing = sum(1 for s, d in need if not (th.CACHE / f"{s}_{d}.parquet").exists())
        print(f"fetched {done}; still missing {missing} of {len(need)}"); return 0

    missing = [f"{s}_{d}" for s, d in need if not (th.CACHE / f"{s}_{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sessions not cached — run --fetch-only first")
    bp = P["round_trip_bp"] / 1e4
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{HYP} — {doc['name']}\nevents: {len(ev)} instrument-events, {len({e['t1'] for e in ev})} dates\n")

    rows, quality = [], {}
    for e in ev:
        b = th.stock_1m(e["sym"], e["t1"])
        quality.setdefault(e["sym"], []).append(len(b) >= 370)
        px = th.stock_1m("QQQ" if e["sym"] == "SPY" else "SPY", e["t1"])
        tr = simulate(b, e["s"], e["C"], e["L"], e["T"])
        nv = naive(b, e["s"])
        cf = confluence(b, px, tr, e["s"], e["C"], e["c1_volume"], e["c5_strong_close"])
        rows.append({**e, **trade_dict(tr), "structure_net": tr.ret_gross - (bp if tr.triggered else 0.0),
                     "naive_net": nv - bp, **{f"conf_{k}": v for k, v in cf.items()}})
    df = pd.DataFrame(rows)
    df["delta"] = df["structure_net"] - df["naive_net"]
    df.to_parquet(OUT / "hyp111_trades.parquet", index=False)

    q = {s: float(np.mean(v)) for s, v in quality.items()}
    trig = df[df["triggered"]]
    data_abort = any(v < 0.8 for v in q.values()) or len(trig) < 100
    print("bars>=370 share per instrument:", {k: round(v, 3) for k, v in q.items()})
    print(f"(p) triggered {len(trig)}/{len(df)} = {len(trig)/len(df):.1%}; by exit: {trig['exit_kind'].value_counts().to_dict()}")
    print("    per instrument:", df.groupby("sym")["triggered"].mean().round(2).to_dict())

    # date-level delta series
    by_date = df.groupby("t1").agg(delta=("delta", "mean"), structure=("structure_net", "mean"), naive=("naive_net", "mean")).sort_index()
    dates = by_date.index.values
    years = pd.to_datetime(by_date.index).year.values
    dpy = len(dates) / ((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25)
    d = by_date["delta"].values
    sh = sharpe_dates(d, dpy)
    rng = np.random.default_rng(P["seed"])
    boot = date_block_bootstrap(dates, lambda ix: sharpe_dates(d[ix], dpy), L=P["block_L"], draws=P["draws"], rng=rng)
    lo, hi = ci95(boot); b1 = lo > 0
    dsr_sr, dsr_p = deflated_sharpe_ratio(sh, n_trials=P["n_trials"], n_obs=len(d)); b2 = dsr_p >= 0.95
    print(f"\nincumbent (naive continuation): mean {by_date['naive'].mean()*100:+.4f}%/event-day   structure: {by_date['structure'].mean()*100:+.4f}%/event-day   cash: 0")
    print(f"(b) delta Sharpe {sh:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  {'PASS' if b1 else 'FAIL'}   DSR@{P['n_trials']} prob {dsr_p:.3f} {'PASS' if b2 else 'FAIL'}")

    folds = []
    try:
        for tr_, te in combinatorial_purged_splits(dates, dates, n_groups=6, test_groups=2, embargo_frac=1 / len(dates)):
            folds.append(sharpe_dates(d[te], dpy))
    except Exception as ex:
        print("fold error", ex)
    folds = np.asarray(folds); c_pos = int((folds > 0).sum())
    print(f"(c) folds positive {c_pos}/15  mean {folds.mean() if len(folds) else float('nan'):+.3f}")
    g = df.groupby("sym").apply(lambda x: sharpe_dates(x.groupby("t1")["delta"].mean().values, dpy))
    g_pos = int((g > 0).sum())
    print(f"(g) per-instrument delta Sharpe: {g.round(2).to_dict()}  positive {g_pos}/10")
    m = years != 2020
    x = sharpe_dates(d[m], dpy); x_pass = x > 0
    print(f"(x) ex-2020 delta Sharpe {x:+.3f} {'PASS' if x_pass else 'FAIL'}")
    dd = float(by_date["structure"].mean()); print(f"(d) structure_net {dd*100:+.4f}%/event-day vs floor 0.05%")
    per_year = {int(y): float(sharpe_dates(d[years == y], dpy)) for y in sorted(set(years))}
    print("per-year delta Sharpe:", {k: round(v, 2) for k, v in per_year.items()})
    # break-even cost: structure vs naive both pay cost only when executed; delta mean as function of bp
    trig_share = float(df["triggered"].mean())
    be = None
    if trig_share < 1:
        # delta(bp) = delta_gross - bp*(trig_share - 1) per event; solve delta mean = 0
        dg = float((df["structure_net"] + np.where(df["triggered"], bp, 0) - (df["naive_net"] + bp)).mean())
        be = dg / (1 - trig_share) * 1e4 if (1 - trig_share) else None
    print(f"[descriptive] break-even round-trip cost ≈ {be:.1f} bp" if be is not None else "[descriptive] break-even n/a")

    # confluence
    conf_line, conf = "INCONCLUSIVE", {}
    if len(trig) and "conf_count" in trig:
        t = trig.dropna(subset=["conf_count"]).copy(); t["conf_count"] = t["conf_count"].astype(int)
        t["bucket"] = np.where(t["conf_count"] <= 1, "<=1", np.where(t["conf_count"] == 2, "2", ">=3"))
        means = t.groupby("bucket")["structure_net"].agg(["mean", "count"])
        conf = {"buckets": means.to_dict(), "per_condition": {k: float(t[t[f"conf_{k}"]]["structure_net"].mean()) for k in ("c1", "c2", "c3", "c4", "c5") if f"conf_{k}" in t}}
        print("(m) confluence buckets:", means.round(4).to_dict())
        if (means["count"] >= 30).all() and set(means.index) == {"<=1", "2", ">=3"}:
            mono = means.loc[">=3", "mean"] >= means.loc["2", "mean"] >= means.loc["<=1", "mean"]
            cc, yy = t["conf_count"].values.astype(float), t["structure_net"].values
            def slope(ix):
                xx, y2 = cc[ix], yy[ix]
                return float(np.polyfit(xx, y2, 1)[0]) if xx.std() > 0 else 0.0
            sb = date_block_bootstrap(t["t1"].values, slope, L=P["block_L"], draws=P["draws"], rng=rng)
            slo, shi = ci95(sb); conf.update({"slope": slope(np.arange(len(t))), "slope_ci": [slo, shi], "monotone": bool(mono)})
            conf_line = "MONOTONIC" if (mono and slo > 0) else "STORY"
            print(f"    slope {conf['slope']*100:+.4f}%/condition CI [{slo*100:+.4f}, {shi*100:+.4f}]  monotone {mono}  -> {conf_line}")
    print(f"(m) confluence: {conf_line}")

    # ── SECONDARY CLAIM: the unconditional fade ──────────────────────────────
    print("\n── SECONDARY: next-session fade (−naive_net), all events ──")
    df["fade"] = -df["naive_net"]
    fb = df.groupby("t1")["fade"].mean().sort_index(); fd = fb.values; fdates = fb.index.values; fy = pd.to_datetime(fb.index).year.values
    fboot = date_block_bootstrap(fdates, lambda ix: fd[ix].mean(), L=P["block_L"], draws=P["draws"], rng=rng)
    flo, fhi = ci95(fboot)
    fg = df.groupby("sym")["fade"].mean(); fg_pos = int((fg > 0).sum())
    f20 = fd[fy == 2020]; f20boot = date_block_bootstrap(fdates[fy == 2020], lambda ix: f20[ix].mean(), L=P["block_L"], draws=P["draws"], rng=rng) if len(f20) > 5 else np.array([0.0])
    f20lo, f20hi = ci95(f20boot); fex = float(fd[fy != 2020].mean())
    per_inst_n = df.groupby("sym").size()
    print(f"  mean fade {fd.mean()*100:+.4f}%/event-day  CI [{flo*100:+.4f}, {fhi*100:+.4f}]   instruments>0 {fg_pos}/10   ex-2020 {fex*100:+.4f}%   2020-only {f20.mean()*100:+.4f}% CI [{f20lo*100:+.4f}, {f20hi*100:+.4f}] (n={len(f20)} dates)")
    fyear = {int(y): float(fd[fy == y].mean()) for y in sorted(set(fy))}
    print("  per-year fade %/event-day:", {k: round(v * 100, 3) for k, v in fyear.items()})
    fw = float((df["fade"] > 0).mean()); faw = float(df.loc[df["fade"] > 0, "fade"].mean()); fal = float(-df.loc[df["fade"] <= 0, "fade"].mean())
    print(f"  expectancy: n={len(df)} W={fw:.3f} avgWin={faw*100:.3f}% avgLoss={fal*100:.3f}% E={(fw*faw-(1-fw)*fal)*100:+.4f}%/trade")
    for sgn, nm in ((-1, "down-shocks"), (1, "up-shocks")):
        sub = df[df["s"] == sgn]; print(f"  {nm}: n={len(sub)} mean {sub['fade'].mean()*100:+.4f}%  W={(sub['fade']>0).mean():.3f}")
    m03 = df[df["t1"].str.startswith("2020-03")]; print(f"  2020-03 alone: n={len(m03)} mean {m03['fade'].mean()*100:+.3f}%  worst {m03['fade'].min()*100:+.2f}%")
    print("  worst 5 trades:", df.nsmallest(5, "fade")[["sym", "t1", "fade"]].assign(fade=lambda x: (x["fade"] * 100).round(2)).to_dict("records"))
    if (per_inst_n < 30).any():
        fade_verdict = "FADE_INCONCLUSIVE"
    elif flo > 0 and fg_pos >= 7 and not (f20hi < 0) and fex > 0:
        fade_verdict = "FADE_HOLDS"
    else:
        fade_verdict = "FADE_FAILS"
    print(f"  -> {fade_verdict}")

    if data_abort or len(folds) != 15:
        verdict = "INCONCLUSIVE"
    elif (not b1) or c_pos <= 7 or g_pos <= 5:
        verdict = "NULL"
    elif b1 and b2 and c_pos >= 12 and g_pos >= 7 and x_pass:
        verdict = "CONFIRMED" if dd >= P["floor_per_day"] else "VALID_BUT_BELOW_FLOOR"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n=== VERDICT primary: {verdict}   confluence: {conf_line}   secondary fade: {fade_verdict} ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(),
           "n_events": len(df), "n_dates": len(dates), "n_triggered": len(trig), "quality": q,
           "incumbent_per_event_day": float(by_date["naive"].mean()), "structure_per_event_day": dd,
           "b": {"delta_sharpe": sh, "ci": [lo, hi], "dsr_prob": float(dsr_p)}, "c": {"folds": folds.tolist(), "n_pos": c_pos},
           "g": {"per_instrument": g.to_dict(), "n_pos": g_pos}, "x": {"ex2025": x}, "per_year": per_year,
           "breakeven_bp": be, "confluence": conf, "confluence_verdict": conf_line, "verdict": verdict,
           "fade": {"verdict": fade_verdict, "mean": float(fd.mean()), "ci": [flo, fhi], "instruments_pos": fg_pos, "per_instrument": fg.to_dict(), "ex2020": fex, "y2020": {"mean": float(f20.mean()), "ci": [f20lo, f20hi], "n": int(len(f20))}, "per_year": fyear, "W": fw, "avg_win": faw, "avg_loss": fal}}
    (OUT / "hyp111_result.json").write_text(json.dumps(res, indent=2, default=float))
    prereg.adjudicate(HYP, verdict, f"{verdict}; confluence {conf_line}; fade {fade_verdict}", {"oos_sharpe": sh, "p_value": float((boot <= 0).mean()),
                                                                             "result_file": "data/research/hyp111/hyp111_result.json"})
    prereg.verify(HYP)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
