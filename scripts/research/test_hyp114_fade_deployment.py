#!/usr/bin/env python3
"""HYP-114 — deploying the unconditional post-shock fade. THE test. Runs once.
--fetch-only fills the Alpaca minute cache in chunks; --gate-only checks wiring.
Three sealed claims (sizing/denominator on unseen years, universe, exit) + the account simulation."""
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
from research.hyp111 import prereg, alpaca_1m as al  # noqa: E402
from research.hyp111.engine import daily, events, naive, INSTRUMENTS  # noqa: E402
from research.hyp111.date_bootstrap import date_block_bootstrap, ci95  # noqa: E402

HYP = "HYP-114"
OUT = ROOT / "data" / "research" / "hyp114"


def all_events(doc):
    P = doc["frozen_parameters"]
    ev = []
    for sym in INSTRUMENTS:                                   # core ten: tracked daily cache, shocks from 2016-01
        for e in events(daily(sym), P["core_window_t1"][0], P["core_window_t1"][1]):
            ev.append({**e, "sym": sym, "set": "core"})
    for sym in doc["wider_universe"]:                         # wider: Alpaca daily from 2016-01-04 → shocks from 2017-01
        d = daily(sym, window=("2016-01-04", "2026-07-16"), frame=al.daily_bars(sym))
        for e in events(d, P["wider_window_t1"][0], P["wider_window_t1"][1]):
            ev.append({**e, "sym": sym, "set": "wider"})
    return ev


def main(argv) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--gate-only", action="store_true"); ap.add_argument("--max-days", type=int, default=100000)
    a = ap.parse_args(argv)
    doc = prereg.gate_zero(HYP, "start")
    if a.gate_only:
        print("gate-only OK"); return 0
    P = doc["frozen_parameters"]
    ev = all_events(doc)
    need = sorted({(e["sym"], e["t1"]) for e in ev})
    if a.fetch_only:
        from concurrent.futures import ThreadPoolExecutor
        todo = [k for k in need if not (al.CACHE / f"{k[0]}_{k[1]}.parquet").exists()][: a.max_days]
        n = 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            for _ in ex.map(lambda k: al.stock_1m(*k), todo):
                n += 1
                if n % 200 == 0: print(f"  fetched {n}", flush=True)
        print(f"fetched {n}; still missing {sum(not (al.CACHE / f'{s}_{d}.parquet').exists() for s, d in need)} of {len(need)}")
        return 0
    if any(not (al.CACHE / f"{s}_{d}.parquet").exists() for s, d in need):
        raise SystemExit("sessions not cached — run --fetch-only first")
    OUT.mkdir(parents=True, exist_ok=True)
    bp, f_pos, floor = P["round_trip_bp"] / 1e4, P["fraction_per_instrument"], P["floor_per_deployed_day"]
    rng = np.random.default_rng(P["seed"])
    print(f"\n{HYP} — {doc['name']}\nevents: core {sum(e['set']=='core' for e in ev)}  wider {sum(e['set']=='wider' for e in ev)}\n")

    rows = []
    for e in ev:
        b = al.stock_1m(e["sym"], e["t1"])
        if len(b) < 370:
            continue
        rows.append({**e, "fade": -naive(b, e["s"]) - bp, "fade_early": -naive(b, e["s"], P["early_exit_et"]) - bp})
    df = pd.DataFrame(rows); df["year"] = pd.to_datetime(df["t1"]).dt.year
    df.to_parquet(OUT / "hyp114_events.parquet", index=False)
    complete = len(df) / len(ev)
    print(f"complete sessions {complete:.1%}")

    def mean_ci(sub, col="fade", draws=P["draws"]):
        by = sub.groupby("t1")[col].mean().sort_index(); v = by.values
        b = date_block_bootstrap(by.index.values, lambda ix: v[ix].mean(), L=P["block_L"], draws=draws, rng=rng)
        return float(v.mean()), ci95(b), len(by)

    def account(sub):
        """10% of equity per shocked instrument, compounding daily, calendar series over the sub-window."""
        cal = pd.bdate_range(sub["t1"].min(), sub["t1"].max())
        pnl = sub.groupby("t1")["fade"].sum() * f_pos            # sum over shocked instruments × fraction
        k = sub.groupby("t1").size()
        r = pd.Series(0.0, index=cal); r.loc[pd.to_datetime(pnl.index)] = pnl.values
        dep = pd.Series(0.0, index=cal); dep.loc[pd.to_datetime(k.index)] = (k * f_pos).clip(upper=1.0).values
        eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
        yrs = len(cal) / 252
        return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "max_dd": dd, "sharpe_calendar": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
                "mean_deployed_frac": float(dep.mean()), "event_dates_per_year": float(len(pnl) / yrs), "total": float(eq.iloc[-1] - 1)}

    # ── CLAIM 1: denominator + unseen years, core ten ─────────────────────────
    core = df[df["set"] == "core"]
    m_all, ci_all, nd_all = mean_ci(core)
    unseen = core[core["year"] <= 2019]
    m_un, ci_un, nd_un = mean_ci(unseen)
    g_un = unseen.groupby("sym")["fade"].mean(); g_un_pos = int((g_un > 0).sum())
    acc_core = account(core)
    print("── CLAIM 1: core ten, 2016-2026 ──")
    print(f"  deployed-capital yield (mean fade/instrument-day): {m_all*100:+.4f}%  CI [{ci_all[0]*100:+.4f}, {ci_all[1]*100:+.4f}]  ({nd_all} dates)  floor {floor*100:.2f}%")
    print(f"  UNSEEN 2016-2019: {m_un*100:+.4f}%  CI [{ci_un[0]*100:+.4f}, {ci_un[1]*100:+.4f}]  ({nd_un} dates)  instruments>0 {g_un_pos}/10")
    print(f"  per-year: {core.groupby('year')['fade'].mean().mul(100).round(3).to_dict()}")
    print(f"  account (10%/instrument, compounding): CAGR {acc_core['cagr']*100:+.2f}%  maxDD {acc_core['max_dd']*100:.1f}%  calendar Sharpe {acc_core['sharpe_calendar']:.2f}  "
          f"mean deployed {acc_core['mean_deployed_frac']*100:.1f}%  event-dates/yr {acc_core['event_dates_per_year']:.0f}")
    c1 = "PASS" if (ci_all[0] > 0 and m_all >= floor and ci_un[0] > 0 and g_un_pos >= 7 and acc_core["max_dd"] > -P["max_dd_abort"]) else "FAIL"
    if ci_all[0] > 0 and ci_un[0] > 0 and g_un_pos >= 7 and m_all < floor: c1 = "VALID_BUT_BELOW_FLOOR"
    print(f"  -> {c1}")

    # ── CLAIM 2: wider universe ───────────────────────────────────────────────
    wide = df[df["set"] == "wider"]
    m_w, ci_w, nd_w = mean_ci(wide)
    g_w = wide.groupby("sym")["fade"].mean(); g_w_pos = int((g_w > 0).sum())
    m_w20 = float(wide[wide["year"] != 2020].groupby("t1")["fade"].mean().mean())
    acc_all = account(df[df["t1"] >= P["wider_window_t1"][0]])
    acc_core_same = account(core[core["t1"] >= P["wider_window_t1"][0]])
    print("── CLAIM 2: wider 20, 2017-2026 ──")
    print(f"  mean fade {m_w*100:+.4f}%  CI [{ci_w[0]*100:+.4f}, {ci_w[1]*100:+.4f}]  ({nd_w} dates)  instruments>0 {g_w_pos}/20  ex-2020 {m_w20*100:+.4f}%")
    print(f"  per instrument: {(g_w*100).round(3).to_dict()}")
    print(f"  30-ETF account 2017+: CAGR {acc_all['cagr']*100:+.2f}% maxDD {acc_all['max_dd']*100:.1f}% deployed {acc_all['mean_deployed_frac']*100:.1f}% dates/yr {acc_all['event_dates_per_year']:.0f}"
          f"   vs 10-ETF same window: CAGR {acc_core_same['cagr']*100:+.2f}% maxDD {acc_core_same['max_dd']*100:.1f}% deployed {acc_core_same['mean_deployed_frac']*100:.1f}% dates/yr {acc_core_same['event_dates_per_year']:.0f}")
    c2 = "PASS" if (ci_w[0] > 0 and g_w_pos >= 14 and m_w20 > 0) else "FAIL"
    print(f"  -> {c2}")

    # ── CLAIM 3: exit ─────────────────────────────────────────────────────────
    core = core.assign(dexit=core["fade_early"] - core["fade"])
    m_x, ci_x, _ = mean_ci(core, "dexit")
    print(f"── CLAIM 3: exit {P['early_exit_et']} vs 15:55, core ten ──")
    print(f"  early {core['fade_early'].mean()*100:+.4f}%  late {core['fade'].mean()*100:+.4f}%  delta {m_x*100:+.4f}%  CI [{ci_x[0]*100:+.4f}, {ci_x[1]*100:+.4f}]")
    c3 = "EARLY_EXIT_BETTER" if ci_x[0] > 0 else ("LATE_EXIT_BETTER" if ci_x[1] < 0 else "NO_DIFFERENCE")
    print(f"  -> {c3}")

    verdict = "INCONCLUSIVE" if complete < 0.95 else c1
    print(f"\n=== VERDICT: claim1 {c1} · claim2 {c2} · claim3 {c3}   (ledger: {verdict}) ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(), "complete": complete,
           "claim1": {"verdict": c1, "yield": m_all, "ci": ci_all, "unseen": {"mean": m_un, "ci": ci_un, "dates": nd_un, "instruments_pos": g_un_pos, "per_instrument": g_un.to_dict()},
                      "per_year": core.groupby("year")["fade"].mean().to_dict(), "account": acc_core},
           "claim2": {"verdict": c2, "mean": m_w, "ci": ci_w, "instruments_pos": g_w_pos, "per_instrument": g_w.to_dict(), "ex2020": m_w20,
                      "account_30": acc_all, "account_10_same_window": acc_core_same},
           "claim3": {"verdict": c3, "early": float(core["fade_early"].mean()), "late": float(core["fade"].mean()), "delta": m_x, "ci": ci_x},
           "verdict": verdict}
    (OUT / "hyp114_result.json").write_text(json.dumps(res, indent=2, default=float))
    prereg.adjudicate(HYP, verdict, f"claim1 {c1}; claim2 {c2}; claim3 {c3}", {"result_file": "data/research/hyp114/hyp114_result.json"})
    prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
