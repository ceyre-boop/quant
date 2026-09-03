#!/usr/bin/env python3
"""HYP-116 — THE test. Runs once. Ladder exactly as sealed (b76837ace4234dfb)."""
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
from research.hyp111.date_bootstrap import stationary_block_indices, ci95  # noqa: E402
from research.incumbent.data import closes, CORE  # noqa: E402

HYP = "HYP-116"
OUT = ROOT / "data" / "research" / "hyp116"


def simulate(basket_r: np.ndarray, shock: np.ndarray, month_start: np.ndarray, cap: int):
    """Returns (wealth_dca, wealth_deferred, n_deploy, cash_days). Contributions: 1 unit at each month start.
    DCA invests at that close. DEFERRED accrues cash (0%) and invests the balance at the close of the session
    AFTER a shock day; if 'cap' sessions pass with cash uninvested, invests at the next month start."""
    n = len(basket_r); g = np.exp(basket_r)
    w_dca = 0.0; w_def = 0.0; cash = 0.0; pending = False; since = 0; n_dep = 0; cash_days = 0
    for i in range(n):
        w_dca *= g[i]; w_def *= g[i]                       # grow invested wealth through session i
        if month_start[i]:
            w_dca += 1.0; cash += 1.0
            if cash > 0 and since >= cap:                  # cap: invest at this month start
                w_def += cash; cash = 0.0; since = 0; n_dep += 1
        if pending:                                        # shock was yesterday → invest at today's close
            if cash > 0:
                w_def += cash; cash = 0.0; n_dep += 1
            since = 0; pending = False
        if shock[i]:
            pending = True
        if cash > 0:
            since += 1; cash_days += 1
    return w_dca, w_def + cash, n_dep, cash_days


def main(argv) -> int:
    doc = prereg.gate_zero(HYP, "start")
    if "--gate-only" in argv:
        print("gate-only OK"); return 0
    P = doc["frozen_parameters"]; a, b = doc["window"]
    rng = np.random.default_rng(P["seed"]); OUT.mkdir(parents=True, exist_ok=True)
    px = closes(); lr = np.log(px).diff()
    spy = lr["SPY"].dropna(); thr = spy.abs().shift(1).rolling(P["trailing"]).quantile(P["percentile"])
    shock_s = ((spy <= -thr) & thr.notna())
    core = lr[CORE].dropna(); basket = core.mean(axis=1)
    sixforty = (0.6 * core["SPY"] + 0.4 * core["TLT"])
    idx = basket.index[(basket.index >= a) & (basket.index <= b)]
    br, sf = basket.reindex(idx).values, sixforty.reindex(idx).values
    sh = shock_s.reindex(idx).fillna(False).values
    months = idx.to_period("M"); ms = np.r_[True, months[1:] != months[:-1]]
    print(f"\n{HYP} — {doc['name']}\n{idx[0].date()}→{idx[-1].date()}  {len(idx)} sessions  {ms.sum()} contributions  {int(sh.sum())} SPY down-shocks\n")

    d, f, nd, cd = simulate(br, sh, ms, P["cap_sessions"])
    ratio = f / d
    print(f"EW-10 vehicle: DCA terminal {d:.1f}  DEFERRED {f:.1f}  ratio {ratio:.4f}   deployments {nd}  avg cash-days per contribution {cd/ms.sum():.1f}")
    d2, f2, nd2, _ = simulate(sf, sh, ms, P["cap_sessions"])
    print(f"60/40 vehicle: DCA {d2:.1f}  DEFERRED {f2:.1f}  ratio {f2/d2:.4f}")

    # bootstrap: resample joint daily (return, shock) blocks; month starts regenerated on the resampled length
    boots = np.empty(P["draws"])
    for k in range(P["draws"]):
        ix = stationary_block_indices(rng, len(idx), P["block_L"])
        dd, ff, _, _ = simulate(br[ix], sh[ix], ms, P["cap_sessions"])
        boots[k] = ff / dd
    lo, hi = ci95(boots)
    # rolling 5-year windows, monthly start
    starts = np.flatnonzero(ms); wins = []
    for s0 in starts:
        e0 = s0 + 252 * 5
        if e0 > len(idx): break
        dd, ff, _, _ = simulate(br[s0:e0], sh[s0:e0], ms[s0:e0], P["cap_sessions"]); wins.append(ff > dd)
    share = float(np.mean(wins))
    print(f"ratio 95% CI [{lo:.4f}, {hi:.4f}]   rolling 5y windows DEFERRED > DCA: {share:.2f} ({len(wins)} windows)")
    if lo > 1 and share >= 0.6: verdict = "POLICY_HOLDS"
    elif hi < 1 or share <= 0.4: verdict = "POLICY_FAILS"
    else: verdict = "NO_DIFFERENCE"
    print(f"\n=== VERDICT: {verdict} ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(), "n_sessions": len(idx),
           "contributions": int(ms.sum()), "shocks": int(sh.sum()), "dca": d, "deferred": f, "ratio": ratio, "ci": [lo, hi],
           "deployments": nd, "cash_days_per_contribution": cd / ms.sum(), "rolling_share": share, "n_windows": len(wins),
           "sixforty": {"dca": d2, "deferred": f2, "ratio": f2 / d2}, "verdict": verdict}
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=float))
    prereg.adjudicate(HYP, verdict, verdict, {"result_file": "data/research/hyp116/result.json"})
    prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
