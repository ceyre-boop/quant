#!/usr/bin/env python3
"""HYP-112 — THE test. Runs once. --fetch-only fills the option-chain cache in chunks (computes
nothing); --gate-only checks wiring. Ladder exactly as sealed (f2d552448a236cc7)."""
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
from research.hyp111 import prereg, theta_v2 as th  # noqa: E402
from research.hyp111.engine import daily, pick_expiration, straddle, sharpe_dates  # noqa: E402
from research.hyp111.date_bootstrap import date_block_bootstrap, ci95  # noqa: E402
from sovereign.discovery.cpcv import combinatorial_purged_splits  # noqa: E402
from sovereign.discovery.gate import deflated_sharpe_ratio  # noqa: E402

HYP = "HYP-112"
OUT = ROOT / "data" / "research" / "hyp112"


def ymd(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def build_legs(doc):
    """One record per (event, leg) with the chain request needed. leg = 'event' | 'control'."""
    P = doc["frozen_parameters"]; w = doc["data"]["window_t"]
    legs = []
    for sym in doc["instrument_set"]:
        df = daily(sym); idx = df.index; exps = th.expirations(sym)
        for i in np.flatnonzero(df["shock"].values):
            t = idx[i]
            if not (pd.Timestamp(w[0]) <= t <= pd.Timestamp(w[1])) or i + P["hold_sessions"] >= len(idx):
                continue
            ci = i - P["control_offset_sessions"]
            if ci < 0 or ci + P["hold_sessions"] >= len(idx) or bool(df["shock"].iloc[ci]):
                continue
            for leg, j in (("event", i), ("control", ci)):
                d0, d1 = idx[j], idx[j + P["hold_sessions"]]
                exp = pick_expiration(exps, d0.strftime("%Y-%m-%d"), P["min_dte_days"])
                if exp is None:
                    continue
                legs.append({"sym": sym, "t": t.strftime("%Y-%m-%d"), "leg": leg, "d0": ymd(d0), "d1": ymd(d1),
                             "exp": exp, "spot": float(df["close"].iloc[j]),
                             "realized": float(abs(np.log(df["close"].iloc[j + P["hold_sessions"]] / df["close"].iloc[j])))})
    return legs


def main(argv) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--gate-only", action="store_true"); ap.add_argument("--max-calls", type=int, default=100000)
    a = ap.parse_args(argv)
    doc = prereg.gate_zero(HYP, "start")
    if a.gate_only:
        print("gate-only OK"); return 0
    P = doc["frozen_parameters"]
    legs = build_legs(doc)
    keys = sorted({(l["sym"], l["exp"], l["d0"], l["d1"]) for l in legs})
    cached = lambda k: (th.CACHE_OPT / f"{k[0]}_{k[1]}_{k[2]}_{k[3]}.parquet").exists()
    if a.fetch_only:
        from concurrent.futures import ThreadPoolExecutor
        todo = [k for k in keys if not cached(k)][: a.max_calls]
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            for _ in ex.map(lambda k: th.option_bulk_eod(*k), todo):
                done += 1
                if done % 100 == 0: print(f"  fetched {done}", flush=True)
        print(f"fetched {done}; still missing {sum(not cached(k) for k in keys)} of {len(keys)}"); return 0
    if any(not cached(k) for k in keys):
        raise SystemExit("chains not cached — run --fetch-only first")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{HYP} — {doc['name']}\nlegs: {len(legs)} ({len(legs)//2} events with a control)\n")

    recs = []
    for l in legs:
        ch = th.option_bulk_eod(l["sym"], l["exp"], l["d0"], l["d1"])
        r = straddle(ch, l["d0"], l["d1"], l["spot"], P["commission_per_contract"]) if not ch.empty else None
        recs.append({**l, **(r or {}), "priced": r is not None})
    df = pd.DataFrame(recs)
    df.to_parquet(OUT / "hyp112_legs.parquet", index=False)
    ev = df[df["leg"] == "event"].set_index(["sym", "t"]); co = df[df["leg"] == "control"].set_index(["sym", "t"])
    both = ev.join(co, lsuffix="_e", rsuffix="_c", how="inner")
    both = both[both["priced_e"] & both["priced_c"]].reset_index()
    print(f"priced on both legs: {len(both)} of {len(ev)} events; per instrument {both.groupby('sym').size().to_dict()}")
    per_inst_n = both.groupby("sym").size()
    data_abort = len(both) < 100 or (per_inst_n.reindex(doc["instrument_set"]).fillna(0) < 30).any()

    both["delta"] = both["ret_on_premium_e"] - both["ret_on_premium_c"]
    by_date = both.groupby("t").agg(delta=("delta", "mean"), ev=("ret_on_premium_e", "mean"), co=("ret_on_premium_c", "mean"),
                                    ev_spot=("ret_on_spot_e", "mean")).sort_index()
    dates = by_date.index.values; years = pd.to_datetime(by_date.index).year.values
    dpy = len(dates) / ((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25)
    d = by_date["delta"].values
    sh = sharpe_dates(d, dpy)
    rng = np.random.default_rng(P["seed"])
    boot = date_block_bootstrap(dates, lambda ix: sharpe_dates(d[ix], dpy), L=P["block_L"], draws=P["draws"], rng=rng)
    lo, hi = ci95(boot); b1 = lo > 0
    dsr_sr, dsr_p = deflated_sharpe_ratio(sh, n_trials=P["n_trials"], n_obs=len(d)); b2 = dsr_p >= 0.95
    print(f"incumbent (control straddle): mean {by_date['co'].mean()*100:+.2f}% on premium   shock straddle: {by_date['ev'].mean()*100:+.2f}% on premium   ({by_date['ev_spot'].mean()*100:+.3f}% on spot/event)")
    print(f"(a, descriptive) implied move: shock {both['implied_move_e'].mean()*100:.2f}% vs control {both['implied_move_c'].mean()*100:.2f}%;  realized |5d|: shock {both['realized_e'].mean()*100:.2f}% vs control {both['realized_c'].mean()*100:.2f}%")
    print(f"    realized/implied: shock {(both['realized_e']/both['implied_move_e']).median():.2f}  control {(both['realized_c']/both['implied_move_c']).median():.2f}")
    print(f"(b) delta Sharpe {sh:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  {'PASS' if b1 else 'FAIL'}   DSR@{P['n_trials']} prob {dsr_p:.3f} {'PASS' if b2 else 'FAIL'}")
    folds = []
    try:
        for tr_, te in combinatorial_purged_splits(dates, dates, n_groups=6, test_groups=2, embargo_frac=P["embargo_dates"] / len(dates)):
            folds.append(sharpe_dates(d[te], dpy))
    except Exception as ex:
        print("fold error", ex)
    folds = np.asarray(folds); c_pos = int((folds > 0).sum())
    print(f"(c) folds positive {c_pos}/15  mean {folds.mean() if len(folds) else float('nan'):+.3f}")
    g = both.groupby("sym")["delta"].mean(); g_pos = int((g > 0).sum())
    print(f"(g) per-instrument mean delta on premium: {(g*100).round(1).to_dict()}  positive {g_pos}/10")
    m = years != 2020; x = sharpe_dates(d[m], dpy); x_pass = x > 0
    print(f"(x) ex-2020 delta Sharpe {x:+.3f} {'PASS' if x_pass else 'FAIL'}")
    dd = float(by_date["ev_spot"].mean()); print(f"(d) shock straddle on spot {dd*100:+.3f}%/event vs floor 0.25%")
    per_year = {int(y): {"delta_sharpe": float(sharpe_dates(d[years == y], dpy)), "shock_prem": float(by_date['ev'].values[years == y].mean()), "control_prem": float(by_date['co'].values[years == y].mean())} for y in sorted(set(years))}
    print("per-year:", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in per_year.items()})
    dte = (pd.to_datetime(both["exp_e"]) - pd.to_datetime(both["d0_e"])).dt.days
    print(f"[descriptive] DTE at entry median {dte.median():.0f}; premium/spot median {(both['premium_in_e']/both['spot_e']).median()*100:.2f}%; lost to unquoted {1-len(both)/len(ev):.1%}")

    if data_abort or len(folds) != 15:
        verdict = "INCONCLUSIVE"
    elif (not b1) or c_pos <= 7 or g_pos <= 5:
        verdict = "NULL"
    elif b1 and b2 and c_pos >= 12 and g_pos >= 7 and x_pass:
        verdict = "CONFIRMED" if dd >= P["floor_per_event_on_spot"] else "VALID_BUT_BELOW_FLOOR"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n=== VERDICT: {verdict} ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(),
           "n_events_priced": len(both), "n_dates": len(dates), "per_instrument_n": per_inst_n.to_dict(),
           "shock_prem": float(by_date["ev"].mean()), "control_prem": float(by_date["co"].mean()), "shock_on_spot": dd,
           "implied": {"shock": float(both['implied_move_e'].mean()), "control": float(both['implied_move_c'].mean())},
           "realized": {"shock": float(both['realized_e'].mean()), "control": float(both['realized_c'].mean())},
           "b": {"delta_sharpe": sh, "ci": [lo, hi], "dsr_prob": float(dsr_p)}, "c": {"folds": folds.tolist(), "n_pos": c_pos},
           "g": {"per_instrument": g.to_dict(), "n_pos": g_pos}, "x": {"ex2020": x}, "per_year": per_year, "verdict": verdict}
    (OUT / "hyp112_result.json").write_text(json.dumps(res, indent=2, default=float))
    prereg.adjudicate(HYP, verdict, verdict, {"oos_sharpe": sh, "p_value": float((boot <= 0).mean()), "result_file": "data/research/hyp112/hyp112_result.json"})
    prereg.verify(HYP)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
