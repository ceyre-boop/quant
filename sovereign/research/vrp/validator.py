"""VRP validator — pure stage logic + verdict ladder (no I/O, no network).

The INVERTED GAUNTLET. Because the iron-condor backtest is data-blocked, we run the
cheapest falsifying test first:

  Stage 1  existence       does IV systematically exceed forward RV? (BTZ, both-sides)
  Stage 2  orthogonality   does the CAUSAL harvest return stay uncorrelated with carry in
                           crisis, or re-couple like overnight-QQQ?  <-- KILL-GATE

Stage 2 thresholds and structure mirror scripts/edge_research_diversification.py so the
verdict is comparable to the prior overnight-QQQ diversification finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from clawd_trading.meta_evaluator.metrics_calculator import calculate_sharpe
from sovereign.research.vrp import vrp_calculator as vc

CRISES = [("GFC_2008", "2008-06-01", "2009-03-31"),
          ("COVID_2020", "2020-02-20", "2020-04-30"),
          ("RATE_SHOCK_2022", "2022-01-01", "2022-12-31")]

STRESS_VIX = 30.0
SPLITS = [("IS_2006_2020", "2006-01-01", "2020-12-31"),
          ("OOS_2021_2023", "2021-01-01", "2023-12-31"),
          ("HOLDOUT_2024_2025", "2024-01-01", "2025-12-31")]

# Correlation gates (scale-invariant) — same bands as the overnight-QQQ diversification test.
GATE_CORRELATED = 0.35
GATE_DIVERSIFIER_FULL = 0.25
GATE_CRISIS = 0.35


# ── helpers (mirroring scripts/edge_research_*.py conventions) ──────────────────
def _tstat(x) -> float | None:
    x = np.asarray(x, float)
    sd = float(np.std(x, ddof=1))
    return round(float(np.mean(x) / (sd / np.sqrt(len(x)))), 3) if sd > 0 and len(x) > 1 else None


def _signflip_p(x, perms: int, seed: int):
    """One-sample permutation — is mean(x) > 0? Sign-flip null, Knuth (count+1)/(perms+1)."""
    x = np.asarray(x, float)
    observed = float(np.mean(x))
    rng = np.random.default_rng(seed)
    null = np.empty(perms)
    for i in range(perms):
        null[i] = float(np.mean(x * rng.choice((-1.0, 1.0), size=len(x))))
    p_high = float((np.sum(null >= observed) + 1) / (perms + 1))
    return observed, p_high


def _corr(a: pd.Series, b: pd.Series):
    idx = a.index.intersection(b.index)
    if len(idx) < 30:
        return None, len(idx)
    return round(float(np.corrcoef(a.loc[idx], b.loc[idx])[0, 1]), 3), len(idx)


def _band(rho) -> str:
    if rho is None:
        return "N/A"
    r = abs(rho)
    return "LOW" if r < 0.25 else "MODERATE" if r < 0.45 else "HIGH"


# ── Stage 1 — existence ─────────────────────────────────────────────────────────
def stage1_existence(label, vol_index, close, window: int = 21, perms: int = 10000, seed: int = 7) -> dict:
    gap = vc.btz_vrp_gap(vol_index, close, window=window)
    if len(gap) < 250:
        return {"label": label, "vrp_exists": False, "note": f"insufficient gap series (n={len(gap)})"}
    g = gap.to_numpy(float)
    mean_gap, p = _signflip_p(g, perms, seed)
    calm, stressed = vc.regime_split(gap, vol_index, STRESS_VIX)
    by_split = {}
    for name, lo, hi in SPLITS:
        sub = gap[(gap.index >= lo) & (gap.index <= hi)]
        by_split[name] = {
            "n": int(len(sub)),
            "mean_gap": round(float(sub.mean()), 5) if len(sub) else None,
            "pct_positive": round(float((sub.to_numpy(float) > 0).mean()), 3) if len(sub) else None,
        }
    return {
        "label": label,
        "n": int(len(gap)),
        "window": f"{gap.index.min().date()}..{gap.index.max().date()}",
        "mean_gap_annvar": round(mean_gap, 5),
        "median_gap_annvar": round(float(np.median(g)), 5),
        "pct_positive": round(float((g > 0).mean()), 3),
        "t_stat": _tstat(g),
        "permutation_p": p,
        "both_sides": {
            "calm_VIX_le30": {"n": int(len(calm)),
                              "mean_gap": round(float(calm.mean()), 5) if len(calm) else None},
            "stressed_VIX_gt30": {"n": int(len(stressed)),
                                  "mean_gap": round(float(stressed.mean()), 5) if len(stressed) else None},
        },
        "by_split": by_split,
        "vrp_exists": bool(mean_gap > 0 and p < 0.05),
    }


# ── Stage 2 — orthogonality kill-gate ───────────────────────────────────────────
def _analyze(A: pd.Series, B: pd.Series, vix: pd.Series, label: str) -> dict:
    """Crisis-conditional correlation of harvest return A vs comparator B. Clone of the
    overnight-QQQ diversification analyzer (full + per-crisis + VIX>30 + tail + rolling)."""
    full_rho, full_n = _corr(A, B)
    crises = {}
    for name, lo, hi in CRISES:
        sub = A[(A.index >= lo) & (A.index <= hi)]
        rho, n = _corr(sub, B)
        crises[name] = {"rho": rho, "n": n}
    stress_days = vix[vix > STRESS_VIX].index
    stress_rho, stress_n = _corr(A[A.index.isin(stress_days)], B)

    idx = A.index.intersection(B.index)
    tail = {}
    if len(idx) >= 50:
        a, b = A.loc[idx], B.loc[idx]
        tail = {
            "A_mean_on_B_worst_decile": round(float(a[b <= b.quantile(0.10)].mean()), 8),
            "B_mean_on_A_worst_decile": round(float(b[a <= a.quantile(0.10)].mean()), 8),
        }
    rolling = {}
    if len(idx) >= 90:
        a, b = A.loc[idx], B.loc[idx]
        roll = a.rolling(60).corr(b).dropna()
        if len(roll):
            vix_al = vix.reindex(roll.index).ffill()
            rs, rc = roll[vix_al > STRESS_VIX], roll[vix_al <= STRESS_VIX]
            rolling = {
                "overall_mean": round(float(roll.mean()), 3),
                "mean_when_VIX_gt30": round(float(rs.mean()), 3) if len(rs) else None,
                "mean_when_VIX_le30": round(float(rc.mean()), 3) if len(rc) else None,
            }

    crisis_rhos = [c["rho"] for c in crises.values() if c["rho"] is not None]
    if stress_rho is not None:
        crisis_rhos.append(stress_rho)
    max_crisis = max((abs(r) for r in crisis_rhos), default=0.0)
    spikes = (rolling.get("mean_when_VIX_gt30") is not None and rolling.get("mean_when_VIX_le30") is not None
              and rolling["mean_when_VIX_gt30"] - rolling["mean_when_VIX_le30"] > 0.15)

    if full_rho is None:
        verdict = "NO_DATA"
    elif abs(full_rho) >= GATE_CORRELATED:
        verdict = "CORRELATED"
    elif max_crisis >= GATE_CRISIS or spikes:
        verdict = "CORRELATED_IN_CRISIS"
    elif abs(full_rho) < GATE_DIVERSIFIER_FULL and max_crisis < GATE_CRISIS:
        verdict = "TRUE_DIVERSIFIER"
    else:
        verdict = "CORRELATED_IN_CRISIS"   # moderate full-sample, borderline
    return {"label": label, "full_sample": {"rho": full_rho, "n": full_n, "band": _band(full_rho)},
            "crises": crises, "vix_gt30": {"rho": stress_rho, "n": stress_n},
            "max_crisis_abs_corr": round(max_crisis, 3), "tail": tail, "rolling": rolling, "verdict": verdict}


def stage2_orthogonality(harvest, carry_dbv, carry_v015, overnight_qqq, vix) -> dict:
    hv = harvest.dropna().to_numpy(float)
    profile = {
        "n": int(len(hv)),
        "mean_daily_var_units": round(float(np.mean(hv)), 10) if len(hv) else None,
        "sharpe_rf0": calculate_sharpe(list(hv), risk_free_rate=0.0) if len(hv) > 1 else None,
        "pct_positive_days": round(float((hv > 0).mean()), 3) if len(hv) else None,
        "note": "harvest is a short-variance P&L in variance units; Sharpe/correlation are scale-invariant.",
    }
    vs_carry = _analyze(harvest, carry_dbv, vix, "vrp_harvest_vs_DBV_carry") if len(carry_dbv) else {"verdict": "NO_DATA"}
    vs_v015 = (_analyze(harvest, carry_v015, vix, "vrp_harvest_vs_v015_carry")
               if len(carry_v015) >= 30 else {"verdict": "NO_DATA", "note": "too few overlapping forex-log dates"})
    vs_overnight = _analyze(harvest, overnight_qqq, vix, "vrp_harvest_vs_overnight_qqq") if len(overnight_qqq) else {"verdict": "NO_DATA"}
    return {"harvest_profile": profile, "vs_carry": vs_carry,
            "vs_v015_secondary": vs_v015, "vs_overnight_qqq": vs_overnight}


# ── Verdict ladder ──────────────────────────────────────────────────────────────
def build_verdict(stage1_spy: dict, stage1_qqq: dict, stage2: dict):
    vrp_exists = bool(stage1_spy.get("vrp_exists") or stage1_qqq.get("vrp_exists"))
    carry_verdict = stage2.get("vs_carry", {}).get("verdict")
    recouples = carry_verdict in ("CORRELATED", "CORRELATED_IN_CRISIS")
    orthogonal = carry_verdict == "TRUE_DIVERSIFIER"
    gates = {
        "vrp_exists": vrp_exists,
        "carry_relationship": carry_verdict,
        "recouples_with_carry": recouples,
        "orthogonal_to_carry": orthogonal,
    }
    if not vrp_exists:
        verdict = "NOT_SIGNIFICANT"
    elif recouples:
        verdict = "REJECTED_OOS"          # return-stacking, not a diversifier — STOP, no paid data
    elif orthogonal:
        verdict = "DATA_INSUFFICIENT"     # survives cheap gate; iron-condor sim needs real chains
    else:
        verdict = "PARTIAL_CONFIRMATION"  # carry relationship indeterminate (NO_DATA / borderline)
    return verdict, gates
