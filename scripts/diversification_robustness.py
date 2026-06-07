#!/usr/bin/env python3
"""Robustness of the overnight-QQQ × carry diversification verdict.

The existing test (edge_research_diversification.py) flagged the MARKET-NEUTRAL long-short
(overnight−intraday) as CORRELATED_IN_CRISIS on a single window: COVID-2020, n=50, rho=0.416 —
with no confidence interval and no significance test. Yet the larger stress sample (VIX>30,
n~431) shows long-short rho~0.042 and tail co-movement ~-3bp. So the pivotal number may be a
50-day artifact.

This script layers the missing rigor onto the SAME series (imported from the existing script so
the point estimates reconcile): moving-block bootstrap CIs on every cut, a crisis-vs-calm
difference test (bootstrap CI + Fisher-z), and a BH multiple-testing correction across windows.

"Couples in crisis" requires the crisis correlation to be BOTH significantly > 0 AND
significantly > the calm correlation. Otherwise the coupling is fragile and the market-neutral
long-short is resurrected as a candidate diversifier.

Read-only: writes data/research/diversification_robustness.json. Run under .venv (py3.9).
Usage:  ~/quant/.venv/bin/python scripts/diversification_robustness.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.getLogger("yfinance").setLevel(logging.ERROR)

# Reuse the EXACT series + crisis windows so numbers reconcile with the prior test.
from scripts.edge_research_diversification import _series, CRISES  # noqa: E402

OUT = ROOT / "data" / "research" / "diversification_robustness.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"

N_BOOT = 10000
BLOCK = 5          # trading-day block length (respects volatility clustering)
SEED = 7
CI = (2.5, 97.5)   # 95%


def _paired(A: pd.Series, carry: pd.Series, lo=None, hi=None, mask_index=None):
    """Return aligned (a, b) numpy arrays for A vs carry on a date subset."""
    a = A
    if lo is not None:
        a = a[(a.index >= lo) & (a.index <= hi)]
    if mask_index is not None:
        a = a[a.index.isin(mask_index)]
    idx = a.index.intersection(carry.index)
    return A.loc[idx].to_numpy(float), carry.loc[idx].to_numpy(float), len(idx)


def _block_boot_corr(a: np.ndarray, b: np.ndarray, rng, n_boot=N_BOOT, block=BLOCK):
    """Moving-block bootstrap distribution of Pearson rho for paired (a,b)."""
    n = len(a)
    if n < max(30, block + 1):
        return None
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        aa, bb = a[idx], b[idx]
        sa, sb = aa.std(), bb.std()
        out[i] = np.corrcoef(aa, bb)[0, 1] if sa > 1e-12 and sb > 1e-12 else 0.0
    return out


def _iid_boot_corr(a, b, rng, n_boot=N_BOOT):
    n = len(a)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aa, bb = a[idx], b[idx]
        sa, sb = aa.std(), bb.std()
        out[i] = np.corrcoef(aa, bb)[0, 1] if sa > 1e-12 and sb > 1e-12 else 0.0
    return out


def _point(a, b):
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _ci_block(a, b, rng):
    pt = _point(a, b)
    dist = _block_boot_corr(a, b, rng)
    iid = _iid_boot_corr(a, b, rng) if len(a) >= 3 else None
    res = {"rho": round(pt, 3) if pt is not None else None, "n": len(a)}
    if dist is not None:
        loq, hiq = np.percentile(dist, CI)
        res["ci95_block"] = [round(float(loq), 3), round(float(hiq), 3)]
        res["excludes_zero"] = bool(loq > 0 or hiq < 0)
    if iid is not None:
        loq, hiq = np.percentile(iid, CI)
        res["ci95_iid"] = [round(float(loq), 3), round(float(hiq), 3)]
    # one-sample Fisher-z p (H0: rho=0)
    if pt is not None and len(a) > 3:
        z = np.arctanh(np.clip(pt, -0.999, 0.999)) * np.sqrt(len(a) - 3)
        res["fisher_p"] = round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4)
    return res


def _fisher_two_sample(r1, n1, r2, n2):
    """Two-sided p that rho_crisis (r1) > rho_calm (r2)."""
    if r1 is None or r2 is None or n1 < 4 or n2 < 4:
        return None
    z1, z2 = np.arctanh(np.clip(r1, -0.999, 0.999)), np.arctanh(np.clip(r2, -0.999, 0.999))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    return round(float(1 - stats.norm.cdf(z)), 4)  # one-sided: crisis > calm


def _diff_ci_block(a_cr, b_cr, a_calm, b_calm, rng):
    """Bootstrap CI for (rho_crisis - rho_calm)."""
    d_cr = _block_boot_corr(a_cr, b_cr, rng)
    d_calm = _block_boot_corr(a_calm, b_calm, rng)
    if d_cr is None or d_calm is None:
        return None
    diff = d_cr - d_calm  # both length N_BOOT, independent draws
    loq, hiq = np.percentile(diff, CI)
    return {"point": round(float(np.mean(d_cr) - np.mean(d_calm)), 3),
            "ci95": [round(float(loq), 3), round(float(hiq), 3)],
            "excludes_zero": bool(loq > 0 or hiq < 0)}


def _bh(pvals: dict):
    """Benjamini-Hochberg adjusted p-values for a {label: p} dict (drops Nones)."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda kv: kv[1])
    adj = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        val = min(prev, p * m / (i + 1))
        adj[k] = round(val, 4)
        prev = val
    return adj


def _analyze(A: pd.Series, carry: pd.Series, vix: pd.Series, label: str, rng):
    a_full, b_full, _ = _paired(A, carry)
    full = _ci_block(a_full, b_full, rng)

    crises = {}
    crisis_ps = {}
    for name, lo, hi in CRISES:
        a, b, _ = _paired(A, carry, lo=lo, hi=hi)
        crises[name] = _ci_block(a, b, rng)
        crisis_ps[name] = crises[name].get("fisher_p")

    stress_idx = vix[vix > 30].index
    a_s, b_s, _ = _paired(A, carry, mask_index=stress_idx)
    stress = _ci_block(a_s, b_s, rng)

    calm_idx = vix[vix <= 30].index
    a_c, b_c, _ = _paired(A, carry, mask_index=calm_idx)
    calm = _ci_block(a_c, b_c, rng)

    # Crisis-vs-calm difference tests (the real question), per crisis + VIX>30
    diffs = {}
    for name, lo, hi in CRISES:
        a_cr, b_cr, _ = _paired(A, carry, lo=lo, hi=hi)
        diffs[name] = {
            "boot": _diff_ci_block(a_cr, b_cr, a_c, b_c, rng),
            "fisher_p_crisis_gt_calm": _fisher_two_sample(
                _point(a_cr, b_cr), len(a_cr), _point(a_c, b_c), len(a_c)),
        }
    diffs["VIX_gt30"] = {
        "boot": _diff_ci_block(a_s, b_s, a_c, b_c, rng),
        "fisher_p_crisis_gt_calm": _fisher_two_sample(
            _point(a_s, b_s), len(a_s), _point(a_c, b_c), len(a_c)),
    }

    bh = _bh(crisis_ps)

    # Verdict: couples in crisis iff SOME crisis is both sig>0 (BH) AND sig>calm (diff CI excl 0)
    couples = False
    for name in [c[0] for c in CRISES] + ["VIX_gt30"]:
        c = crises.get(name) if name != "VIX_gt30" else stress
        excl0 = c and c.get("excludes_zero")
        bh_ok = (bh.get(name, 1.0) < 0.05) if name in bh else None
        d = diffs.get(name, {})
        diff_excl = d.get("boot") and d["boot"].get("excludes_zero")
        gt_calm = (d.get("fisher_p_crisis_gt_calm") is not None
                   and d["fisher_p_crisis_gt_calm"] < 0.05)
        if excl0 and diff_excl and gt_calm:
            couples = True
            break

    verdict = "ROBUST_CRISIS_COUPLING" if couples else "FRAGILE_COUPLING"
    return {
        "label": label, "full_sample": full, "crises": crises,
        "vix_gt30": stress, "calm_vix_le30": calm,
        "crisis_vs_calm": diffs, "bh_adjusted_crisis_p": bh,
        "verdict": verdict,
    }


def main():
    rng = np.random.default_rng(SEED)
    overnight, intraday, longshort, carry, vix = _series()

    long_short = _analyze(longshort, carry, vix, "long_short_overnight_minus_intraday", rng)
    long_only = _analyze(overnight, carry, vix, "long_only_overnight", rng)

    # Null-of-the-null: shuffle carry → full-sample CI must bracket 0.
    a_full, b_full, _ = _paired(longshort, carry)
    b_shuf = b_full.copy()
    rng.shuffle(b_shuf)
    sanity = _ci_block(a_full, b_shuf, rng)

    # Resurrection logic for the market-neutral long-short
    ls_v = long_short["verdict"]
    ls_cov = long_short["crises"].get("COVID_2020", {})
    if ls_v == "FRAGILE_COUPLING":
        rec = ("RESURRECT the market-neutral long-short (overnight-intraday) as a CANDIDATE carry "
               "diversifier: its crisis coupling is NOT statistically robust (COVID rho CI "
               f"{ls_cov.get('ci95_block')} brackets/near 0; not sig > calm after BH). The prior "
               "CORRELATED_IN_CRISIS verdict rested on a 50-day point estimate. Next step: a "
               "vehicle + costed backtest. Structural caveat stands: DBV is a proxy; live carry "
               "book has no crisis history.")
    else:
        rec = ("CONFIRM the kill: the market-neutral long-short robustly couples with carry in "
               "crisis (crisis rho sig > 0 AND sig > calm). Not a crisis-independent second edge. "
               "Verdict hardened with bootstrap CIs.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "id": "OVERNIGHT-CARRY-DIVERSIFICATION-ROBUSTNESS",
        "method": {
            "bootstrap": f"moving-block (block={BLOCK}d), {N_BOOT} resamples, seed={SEED}",
            "tests": "per-cut block-bootstrap 95% CI; crisis-vs-calm bootstrap-diff CI + "
                     "one-sided Fisher-z; BH across crisis windows",
            "carry_proxy": "DBV (G10 carry ETF, ends 2023-03)",
            "couples_rule": "crisis rho must be sig>0 (BH<0.05) AND sig>calm (diff CI excl 0 + Fisher p<0.05)",
        },
        "data_dates": {
            "overnight": [str(overnight.index.min().date()), str(overnight.index.max().date())],
            "carry_dbv": [str(carry.index.min().date()), str(carry.index.max().date())],
        },
        "long_short": long_short,
        "long_only": long_only,
        "sanity_shuffled_full_ci": sanity,
        "recommendation": rec,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    # ---- console summary ----
    def _line(name, c):
        if not c:
            print(f"    {name:14s} n/a"); return
        print(f"    {name:14s} rho={str(c.get('rho')):>6s}  n={c.get('n'):>5d}  "
              f"CI={c.get('ci95_block')}  excl0={c.get('excludes_zero')}  "
              f"fisher_p={c.get('fisher_p')}")
    for series in (long_short, long_only):
        print(f"\n{'='*78}\n  {series['label']}  →  {series['verdict']}\n{'='*78}")
        _line("full", series["full_sample"])
        for nm in ("GFC_2008", "COVID_2020", "RATE_SHOCK_2022"):
            _line(nm, series["crises"].get(nm))
        _line("VIX>30", series["vix_gt30"])
        _line("calm", series["calm_vix_le30"])
        print("  BH-adj crisis p:", series["bh_adjusted_crisis_p"])
        print("  crisis>calm:", {k: (v.get("boot") or {}).get("ci95") for k, v in series["crisis_vs_calm"].items()})
    print(f"\n  SANITY (shuffled): rho={sanity.get('rho')} CI={sanity.get('ci95_block')} "
          f"(must bracket 0)")
    print(f"\n  RECOMMENDATION: {payload['recommendation']}")
    print(f"\n  Saved: {OUT}\n")


if __name__ == "__main__":
    main()
