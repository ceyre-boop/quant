#!/usr/bin/env python3
"""
DIAGNOSTIC — not a prereg, not a re-run of HYP-109, writes nothing to the ledger.

Session 2026-09-02, steps 2 and 3 of "map where edges can exist":

  STEP 2  interrogate the HYP-109 abstention result properly
          - DSR / bootstrap on the DELTA series (abstain_net − incumbent), not on
            the abstain series (the recorded flaw in the HYP-109 spec)
          - confirm the comparator is the incumbent (EW buy-and-hold, ten ETFs)
          - golden rule: per-instrument ΔSharpe, turnover a $2k account must carry
          - fold-level: is 8/15 the signature of real-but-small, or of nothing?
          - ex-2020 (is it one crash?)

  STEP 3  regime as HYPOTHESIS — definition frozen in
          research/TAXONOMY_2026-09-02_where_edges_can_exist.md BEFORE this ran:
            RV21_t   = std(SPY r, t−21..t−1)
            RV252med = median(RV21, t−252..t−1)
            HIGH iff RV21/RV252med > 1.0
          ONE test: incumbent Sharpe HIGH vs LOW (CI must exclude 0), plus the
          abstention delta HIGH vs LOW. Reported ex-2020 too. Counted as a trial.

Every frozen HYP-109 parameter is reused unchanged (p90, k=5, 252, 2bp, L=5,
seed 42, 10k draws, CPCV 6/2). Nothing is swept. HYP-109's verdict is not
touched — its ledger entry stays ADJUDICATED NULL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from test_hyp109_postshock import (  # noqa: E402  reuse, do not re-derive
    gate_zero, load, shocks, flat_mask, sharpe, max_dd, PREREG,
)
from sovereign.discovery.cpcv import combinatorial_purged_splits  # noqa: E402
from sovereign.discovery.gate import deflated_sharpe_ratio        # noqa: E402

OUT = ROOT / "data" / "research" / "hyp109" / "diagnostic_2026-09-02.json"


def boot_idx(n: int, L: int, draws: int, rng) -> np.ndarray:
    """Stationary block bootstrap index matrix (draws × n), Politis–Romano, p=1/L."""
    out = np.empty((draws, n), dtype=int)
    for d in range(draws):
        pos = 0
        while pos < n:
            start = rng.integers(n)
            length = rng.geometric(1.0 / L)
            seg = (start + np.arange(length)) % n
            take = min(length, n - pos)
            out[d, pos:pos + take] = seg[:take]
            pos += take
    return out


def ci(vals: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    doc = gate_zero("diagnostic")            # hash still holds; status must be ADJUDICATED
    P, win, warm = doc["frozen_parameters"], doc["data"]["window"], doc["data"]["warmup_sessions"]
    pct, k, bp = P["percentile"], P["k"], P["round_trip_bp"] / 1e4
    L, seed, draws = P["block_L"], P["seed"], P["draws"]
    n_trials = 1544                            # HYP-109's declared count; this is a re-statistic, not a new trial
    rng = np.random.default_rng(seed)
    res: dict = {"frozen": P, "n_trials_for_delta_dsr": n_trials}

    # ── rebuild the HYP-109 series exactly ──────────────────────────────────
    rets, flats = {}, {}
    for sym in doc["instrument_set"]:
        r = load(sym, win)
        rets[sym] = r
        flats[sym] = flat_mask(shocks(r, pct, warm), k)
    R = pd.DataFrame(rets).dropna()
    F = pd.DataFrame(flats).reindex(R.index).fillna(False)
    R, F = R.iloc[warm:], F.iloc[warm:]
    bh = R.mean(axis=1)                                     # THE INCUMBENT: EW buy-and-hold
    exposed = (~F).astype(float)
    episodes = (F.astype(int).diff().abs() == 1)
    cost = episodes.sum(axis=1) * (bp / 2) / R.shape[1]
    ab_net = (R * exposed).mean(axis=1) - cost
    delta = ab_net - bh                                     # THE DELTA SERIES
    n = len(delta)
    years = delta.index.year

    print(f"\n=== STEP 2 — interrogating HYP-109 abstention on the DELTA ===")
    print(f"comparator = EW buy-and-hold of {list(R.columns)} (the incumbent)")
    print(f"incumbent: Sharpe {sharpe(bh.values):.3f}  {bh.mean()*100:+.4f}%/day  maxDD {max_dd(bh.values)*100:.1f}%")
    print(f"abstain  : Sharpe {sharpe(ab_net.values):.3f}  {ab_net.mean()*100:+.4f}%/day  maxDD {max_dd(ab_net.values)*100:.1f}%")

    # 2a. delta series: Sharpe, block-bootstrap CI, bootstrap p, DSR on the delta
    I = boot_idx(n, L, draws, rng)
    d = delta.values
    sh_delta = sharpe(d)
    boot_sh = np.array([sharpe(d[ix]) for ix in I])
    lo, hi = ci(boot_sh)
    p_delta = float((boot_sh <= 0).mean())
    dsr_sr, dsr_prob = deflated_sharpe_ratio(sh_delta, n_trials=n_trials, n_obs=n)
    # ΔSharpe (abstain − bh) bootstrap: resample the same dates for both
    boot_dsh = np.array([sharpe(ab_net.values[ix]) - sharpe(bh.values[ix]) for ix in I])
    dlo, dhi = ci(boot_dsh)
    print(f"\n[2a] DELTA series (abstain_net − incumbent)")
    print(f"  mean {delta.mean()*100:+.4f}%/day   Sharpe {sh_delta:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   boot p(≤0) {p_delta:.4f}")
    print(f"  DSR on the delta @ n_trials={n_trials}: deflated SR {dsr_sr:+.3f}   prob {dsr_prob:.3f}   {'PASS' if dsr_prob >= 0.95 else 'FAIL'} (≥0.95)")
    print(f"  ΔSharpe(abstain − incumbent) = {sharpe(ab_net.values)-sharpe(bh.values):+.3f}   95% CI [{dlo:+.3f}, {dhi:+.3f}]")
    res["delta"] = {"mean_per_day": float(delta.mean()), "sharpe": sh_delta, "ci": [lo, hi],
                    "boot_p_le_0": p_delta, "dsr_deflated": float(dsr_sr), "dsr_prob": float(dsr_prob),
                    "d_sharpe": float(sharpe(ab_net.values) - sharpe(bh.values)), "d_sharpe_ci": [dlo, dhi]}

    # 2b. ex-2020
    m = years != 2020
    d20 = sharpe(ab_net.values[m]) - sharpe(bh.values[m])
    print(f"\n[2b] ex-2020: ΔSharpe {d20:+.3f}   delta Sharpe {sharpe(d[m]):+.3f}   "
          f"delta {delta[m].mean()*100:+.4f}%/day   incumbent maxDD {max_dd(bh.values[m])*100:.1f}% vs abstain {max_dd(ab_net.values[m])*100:.1f}%")
    res["ex_2020"] = {"d_sharpe": float(d20), "delta_sharpe": sharpe(d[m]), "delta_per_day": float(delta[m].mean())}

    # 2c. per-year
    print(f"\n[2c] per-year ΔSharpe (abstain − incumbent)")
    per_year = {}
    for y in sorted(set(years)):
        my = years == y
        dy = sharpe(ab_net.values[my]) - sharpe(bh.values[my])
        per_year[int(y)] = float(dy)
        print(f"  {y}  ΔSharpe {dy:+.3f}   delta {delta[my].mean()*100:+.4f}%/day   inc {bh[my].mean()*100:+.4f}%/day")
    res["per_year_d_sharpe"] = per_year
    print(f"  years positive: {sum(v > 0 for v in per_year.values())}/{len(per_year)}")

    # 2d. golden rule — per instrument
    print(f"\n[2d] golden rule — per instrument (abstain_net vs hold, same instrument)")
    per_inst = {}
    for s in R.columns:
        r = R[s].values
        e = exposed[s].values
        c = episodes[s].values.astype(float) * (bp / 2)
        a = r * e - c
        ds = sharpe(a) - sharpe(r)
        legs = int(episodes[s].sum())
        per_inst[s] = {"d_sharpe": float(ds), "d_per_day": float(a.mean() - r.mean()),
                       "hold_sharpe": sharpe(r), "abstain_sharpe": sharpe(a),
                       "round_trips_per_year": legs / 2 / (n / 252)}
        print(f"  {s:5s} hold {sharpe(r):+.3f}  abstain {sharpe(a):+.3f}  Δ {ds:+.3f}   "
              f"{(a.mean()-r.mean())*100:+.4f}%/day   {legs/2/(n/252):.1f} round-trips/yr")
    npos = sum(v["d_sharpe"] > 0 for v in per_inst.values())
    print(f"  instruments with ΔSharpe>0: {npos}/{len(per_inst)}")
    rt_total = sum(v["round_trips_per_year"] for v in per_inst.values())
    print(f"  a $2k account holding all ten (≈$200 each, fractional shares) executes ≈{rt_total:.0f} "
          f"round-trips/yr; at 2bp that is ≈${2000*rt_total*bp:.2f}/yr — cost is not the constraint")
    res["per_instrument"] = per_inst
    res["golden_rule"] = {"instruments_positive": npos, "round_trips_per_year_total": rt_total}

    # 2e. fold-level: real-but-small vs nothing
    dates = R.index.values
    exit_ = np.append(dates[k:], [dates[-1]] * k)
    folds = []
    for tr, te in combinatorial_purged_splits(dates, exit_, n_groups=6, test_groups=2, embargo_frac=k / n):
        folds.append(sharpe(ab_net.values[te]) - sharpe(bh.values[te]))
    folds = np.asarray(folds)
    t_like = folds.mean() / (folds.std(ddof=1) / np.sqrt(len(folds)))
    print(f"\n[2e] fold-level: {int((folds>0).sum())}/15 positive   mean ΔSharpe {folds.mean():+.3f}   "
          f"sd {folds.std(ddof=1):.3f}   min {folds.min():+.3f}   max {folds.max():+.3f}   t-like {t_like:+.2f}")
    print(f"  (folds overlap — t-like is descriptive, not a p-value)")
    res["folds"] = {"values": folds.tolist(), "n_pos": int((folds > 0).sum()), "mean": float(folds.mean()),
                    "sd": float(folds.std(ddof=1)), "t_like": float(t_like)}

    # ── STEP 3 — regime, exactly as pre-declared ─────────────────────────────
    print(f"\n=== STEP 3 — regime as hypothesis (definition frozen before this ran) ===")
    spy = load("SPY", win)
    rv21 = spy.rolling(21).std(ddof=1).shift(1)              # uses t−21..t−1
    rv252med = rv21.shift(1).rolling(252).median()           # median of RV21 over t−252..t−1
    ratio = (rv21 / rv252med).reindex(bh.index)
    regime = (ratio > 1.0)
    valid = ratio.notna().values
    hi_m = regime.values & valid
    lo_m = (~regime.values) & valid
    print(f"regime: HIGH iff SPY RV21 / median(RV21, 252) > 1.0   "
          f"HIGH {hi_m.sum()} sessions ({hi_m.mean()*100:.1f}%)  LOW {lo_m.sum()}  undefined {(~valid).sum()}")

    def regime_split(series: np.ndarray, hm, lm):
        return sharpe(series[hm]), sharpe(series[lm])

    # (1) incumbent Sharpe HIGH vs LOW, CI by joint block bootstrap of (bh, regime)
    b = bh.values
    sh_hi, sh_lo = regime_split(b, hi_m, lo_m)
    reg = regime.values
    boot_diff = []
    for ix in I:
        rr, bb, vv = reg[ix], b[ix], valid[ix]
        h, l = rr & vv, (~rr) & vv
        if h.sum() > 20 and l.sum() > 20:
            boot_diff.append(sharpe(bb[h]) - sharpe(bb[l]))
    boot_diff = np.asarray(boot_diff)
    rlo, rhi = ci(boot_diff)
    # forward 21-session return by regime (declared)
    fwd21 = bh.rolling(21).sum().shift(-21)
    f_hi, f_lo = float(fwd21[hi_m].mean()), float(fwd21[lo_m].mean())
    print(f"\n[3.1] incumbent by regime")
    print(f"  daily Sharpe  HIGH {sh_hi:+.3f}  LOW {sh_lo:+.3f}  diff {sh_hi-sh_lo:+.3f}   95% CI [{rlo:+.3f}, {rhi:+.3f}]   "
          f"{'EXCLUDES 0' if not (rlo <= 0 <= rhi) else 'includes 0'}")
    print(f"  mean %/day    HIGH {b[hi_m].mean()*100:+.4f}  LOW {b[lo_m].mean()*100:+.4f}")
    print(f"  fwd-21d cumr  HIGH {f_hi*100:+.3f}%  LOW {f_lo*100:+.3f}%")
    # ex-2020
    hm2, lm2 = hi_m & m, lo_m & m
    print(f"  ex-2020: Sharpe HIGH {sharpe(b[hm2]):+.3f}  LOW {sharpe(b[lm2]):+.3f}  diff {sharpe(b[hm2])-sharpe(b[lm2]):+.3f}")
    res["regime"] = {"definition": "SPY RV21/median(RV21,252) > 1.0", "n_high": int(hi_m.sum()), "n_low": int(lo_m.sum()),
                     "incumbent_sharpe_high": sh_hi, "incumbent_sharpe_low": sh_lo,
                     "diff": sh_hi - sh_lo, "diff_ci": [rlo, rhi],
                     "fwd21_high": f_hi, "fwd21_low": f_lo,
                     "ex2020_diff": sharpe(b[hm2]) - sharpe(b[lm2])}

    # (2) abstention delta HIGH vs LOW
    dh, dl = regime_split(d, hi_m, lo_m)
    boot_dd = []
    for ix in I:
        rr, dd, vv = reg[ix], d[ix], valid[ix]
        h, l = rr & vv, (~rr) & vv
        if h.sum() > 20 and l.sum() > 20:
            boot_dd.append(sharpe(dd[h]) - sharpe(dd[l]))
    ddlo, ddhi = ci(np.asarray(boot_dd))
    print(f"\n[3.2] abstention delta by regime")
    print(f"  delta Sharpe  HIGH {dh:+.3f}  LOW {dl:+.3f}  diff {dh-dl:+.3f}   95% CI [{ddlo:+.3f}, {ddhi:+.3f}]")
    print(f"  delta %/day   HIGH {d[hi_m].mean()*100:+.4f}  LOW {d[lo_m].mean()*100:+.4f}")
    print(f"  share of shock-flat sessions falling in HIGH: {float(F.values.any(axis=1)[hi_m].mean()/max(F.values.any(axis=1)[valid].mean(),1e-9)):.2f}× base rate")
    res["regime"]["delta_sharpe_high"] = dh
    res["regime"]["delta_sharpe_low"] = dl
    res["regime"]["delta_diff_ci"] = [ddlo, ddhi]

    # verdict on step 3 as pre-declared
    improves = not (rlo <= 0 <= rhi)
    res["regime"]["verdict"] = "IMPROVES_MEASURABLE" if improves else "STORY_ONLY"
    print(f"\n[3.v] regime verdict (pre-declared criterion: incumbent Sharpe HIGH−LOW CI excludes 0): "
          f"{res['regime']['verdict']}")

    OUT.write_text(json.dumps(res, indent=2, default=float))
    gate_zero("end")
    print(f"\nwritten {OUT.relative_to(ROOT)}  — ledger untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
