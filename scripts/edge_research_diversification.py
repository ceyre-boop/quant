#!/usr/bin/env python3
"""Edge test — does overnight-QQQ DIVERSIFY the carry edge? (the strategic decider)

A second edge only matters if it's uncorrelated with the first WHEN IT COUNTS (crisis/tail), not
just on average. Tests long-only overnight AND market-neutral long-short (overnight−intraday) vs a
carry proxy (DBV, the G10 carry-factor ETF, 2008→2023-03 — covers 2008/2020/2022). VIX>30 stress
subset. v015 actual carry returns as a recent-window (2023-24) secondary check.

The crisis/tail/stress correlations decide — NOT the benign full-sample average (the trap is low
average corr that spikes in crisis). Read-only: writes data/research json + appends ledger.

Usage:  python3 scripts/edge_research_diversification.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.getLogger("yfinance").setLevel(logging.ERROR)

OUT = ROOT / "data" / "research" / "edge_research_diversification.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
TRADES = ROOT / "logs" / "forex_backtest_trades.json"
CITATION = "Lou, Polk & Skouras 'A Tug of War' (2019); overnight/intraday decomposition + carry-factor (DBV) diversification"

CRISES = [("GFC_2008", "2008-06-01", "2009-03-31"),
          ("COVID_2020", "2020-02-20", "2020-04-30"),
          ("RATE_SHOCK_2022", "2022-01-01", "2022-12-31")]


def _series():
    import yfinance as yf
    q = yf.Ticker("QQQ").history(period="max", interval="1d", auto_adjust=True)
    q.index = q.index.tz_localize(None)
    overnight = (q["Open"] / q["Close"].shift(1) - 1).dropna()
    intraday = (q["Close"] / q["Open"] - 1).reindex(overnight.index)
    longshort = (overnight - intraday).dropna()
    dbv = yf.Ticker("DBV").history(period="max", interval="1d", auto_adjust=True)
    dbv.index = dbv.index.tz_localize(None)
    carry = dbv["Close"].pct_change().dropna()
    vix = yf.Ticker("^VIX").history(period="max", interval="1d", auto_adjust=True)
    vix.index = vix.index.tz_localize(None)
    return overnight, intraday, longshort, carry, vix["Close"]


def _corr(a: pd.Series, b: pd.Series):
    idx = a.index.intersection(b.index)
    if len(idx) < 30:
        return None, len(idx)
    return round(float(np.corrcoef(a.loc[idx], b.loc[idx])[0, 1]), 3), len(idx)


def _band(rho):
    if rho is None:
        return "N/A"
    r = abs(rho)
    return "LOW" if r < 0.25 else "MODERATE" if r < 0.45 else "HIGH"


def _analyze(A: pd.Series, B_carry: pd.Series, vix: pd.Series, label: str):
    full_rho, full_n = _corr(A, B_carry)
    crises = {}
    for name, lo, hi in CRISES:
        sub = A[(A.index >= lo) & (A.index <= hi)]
        crises[name] = dict(zip(("rho", "n"), _corr(sub, B_carry)))
    # VIX>30 stress subset
    stress_days = vix[vix > 30].index
    A_stress = A[A.index.isin(stress_days)]
    stress_rho, stress_n = _corr(A_stress, B_carry)
    # Tail dependence (on common window)
    idx = A.index.intersection(B_carry.index)
    a, b = A.loc[idx], B_carry.loc[idx]
    carry_worst = b <= b.quantile(0.10)
    a_worst = a <= a.quantile(0.10)
    tail = {
        "A_mean_on_carry_worst_decile_bp": round(float(a[carry_worst].mean()) * 1e4, 2),
        "carry_mean_on_A_worst_decile_bp": round(float(b[a_worst].mean()) * 1e4, 2),
    }
    # Rolling 60d corr + stress conditional
    roll = a.rolling(60).corr(b).dropna()
    vix_al = vix.reindex(roll.index).ffill()
    roll_stress = roll[vix_al > 30]
    roll_calm = roll[vix_al <= 30]
    rolling = {
        "overall_mean": round(float(roll.mean()), 3),
        "min": round(float(roll.min()), 3), "max": round(float(roll.max()), 3),
        "mean_when_VIX_gt30": round(float(roll_stress.mean()), 3) if len(roll_stress) else None,
        "mean_when_VIX_le30": round(float(roll_calm.mean()), 3) if len(roll_calm) else None,
    }

    # Verdict
    crisis_rhos = [c["rho"] for c in crises.values() if c["rho"] is not None] + (
        [stress_rho] if stress_rho is not None else [])
    max_crisis = max((abs(r) for r in crisis_rhos), default=0.0)
    tail_couples = tail["A_mean_on_carry_worst_decile_bp"] < -5.0   # A loses >5bp avg when carry crashes
    spikes = (rolling["mean_when_VIX_gt30"] is not None and rolling["mean_when_VIX_le30"] is not None
              and rolling["mean_when_VIX_gt30"] - rolling["mean_when_VIX_le30"] > 0.15)
    if full_rho is None:
        verdict = "NO_DATA"
    elif abs(full_rho) >= 0.35:
        verdict = "CORRELATED"
    elif max_crisis >= 0.35 or tail_couples or spikes:
        verdict = "CORRELATED_IN_CRISIS"
    elif abs(full_rho) < 0.25 and max_crisis < 0.35:
        verdict = "TRUE_DIVERSIFIER"
    else:
        verdict = "CORRELATED_IN_CRISIS"   # moderate full-sample, borderline
    return {
        "label": label, "full_sample": {"rho": full_rho, "n": full_n, "band": _band(full_rho)},
        "crises": crises, "vix_gt30": {"rho": stress_rho, "n": stress_n},
        "max_crisis_abs_corr": round(max_crisis, 3),
        "tail": tail, "rolling": rolling, "verdict": verdict,
    }


def _v015_secondary(overnight: pd.Series):
    try:
        raw = json.loads(TRADES.read_text())
        by_date = {}
        for lst in raw.values():
            for t in lst:
                d = pd.Timestamp(str(t["entry_date"])[:10])
                by_date[d] = by_date.get(d, 0.0) + float(t["pnl_pct"])
        carry = pd.Series(by_date).sort_index()
        idx = carry.index.intersection(overnight.index)
        if len(idx) < 20:
            return {"n": len(idx), "rho": None, "note": "too few overlapping dates"}
        rho = float(np.corrcoef(carry.loc[idx], overnight.loc[idx])[0, 1])
        return {"n": len(idx), "rho": round(rho, 3),
                "window": f"{idx.min().date()}..{idx.max().date()}",
                "note": "v015 ACTUAL carry pnl vs QQQ-overnight on trade dates — short/noisy, recent window"}
    except Exception as e:
        return {"rho": None, "note": f"err: {e}"}


def main():
    overnight, intraday, longshort, carry, vix = _series()
    long_only = _analyze(overnight, carry, vix, "long_only_overnight")
    long_short = _analyze(longshort, carry, vix, "long_short_overnight_minus_intraday")
    secondary = _v015_secondary(overnight)

    # Recommendation
    lo_v, ls_v = long_only["verdict"], long_short["verdict"]
    if ls_v == "TRUE_DIVERSIFIER" and lo_v != "TRUE_DIVERSIFIER":
        rec = ("Deploy the MARKET-NEUTRAL long-short (overnight−intraday): it diversifies carry in crisis "
               "while long-only is crisis-correlated (long equity beta) — the trap. This changes what to trade.")
    elif lo_v == "TRUE_DIVERSIFIER":
        rec = "Long-only overnight diversifies carry even in crisis — rare; deployable as-is (still needs a vehicle)."
    elif ls_v == "TRUE_DIVERSIFIER":
        rec = "Long-short overnight−intraday is the diversifier; long-only also acceptable. Prefer long-short."
    else:
        rec = ("NEITHER version diversifies carry in crisis — overnight-QQQ is correlated return-stacking, "
               "not true diversification. Do NOT treat it as a crisis-independent second edge.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "id": "OVERNIGHT-CARRY-DIVERSIFICATION", "citation": CITATION,
        "carry_proxy": "DBV (Invesco G10 carry-factor ETF, 2008→2023-03)",
        "caveats": ["DBV is the standard tradeable carry proxy, NOT our exact real-rate-differential "
                    "strategy.", "DBV liquidated 2023-03 → recent window covered only by the v015 "
                    "secondary (short/noisy).", "Crisis/tail/stress corr decide, not the full-sample average."],
        "long_only": long_only, "long_short": long_short,
        "v015_actual_secondary": secondary, "recommendation": rec,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    led = [e for e in led if e.get("id") != "OVERNIGHT-CARRY-DIVERSIFICATION"]
    led.append({
        "id": "OVERNIGHT-CARRY-DIVERSIFICATION",
        "name": "Does overnight-QQQ diversify the carry edge (crisis-conditional)?",
        "status": f"long_only={lo_v}; long_short={ls_v}",
        "date_tested": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "result": f"long-only full rho={long_only['full_sample']['rho']} maxCrisis={long_only['max_crisis_abs_corr']}; "
                  f"long-short full rho={long_short['full_sample']['rho']} maxCrisis={long_short['max_crisis_abs_corr']}. {rec}",
        "methodology_note": ("QQQ overnight & overnight−intraday vs DBV carry proxy; full + crisis "
                             "(2008/2020/2022) + VIX>30 + tail-decile + rolling-60d correlations. " + CITATION
                             + ". DBV≠our exact strategy; crisis corr decides."),
    })
    LEDGER.write_text(json.dumps(led, indent=2))

    # Print
    def _row(name, a):
        cr = a["crises"]
        print(f"  {name:30s} full={str(a['full_sample']['rho']):>6s}  "
              f"2008={str(cr['GFC_2008']['rho']):>6s}  2020={str(cr['COVID_2020']['rho']):>6s}  "
              f"2022={str(cr['RATE_SHOCK_2022']['rho']):>6s}  VIX>30={str(a['vix_gt30']['rho']):>6s}  → {a['verdict']}")
    print(f"\n{'='*92}\n  OVERNIGHT-QQQ × CARRY DIVERSIFICATION  (carry proxy: DBV; crisis corr decides)\n{'='*92}")
    print(f"  {'series vs carry':30s} {'full':>6s}  {'2008':>6s}  {'2020':>6s}  {'2022':>6s}  {'VIX>30':>6s}")
    _row("LONG-ONLY overnight", long_only)
    _row("LONG-SHORT (on−intra)", long_short)
    print(f"\n  Tail (mean on carry's worst-decile days):")
    print(f"    long-only A : {long_only['tail']['A_mean_on_carry_worst_decile_bp']:+.1f}bp   "
          f"long-short A: {long_short['tail']['A_mean_on_carry_worst_decile_bp']:+.1f}bp")
    print(f"  Rolling-60d corr in stress vs calm (long-only): "
          f"VIX>30={long_only['rolling']['mean_when_VIX_gt30']} vs VIX≤30={long_only['rolling']['mean_when_VIX_le30']}")
    print(f"  Rolling-60d corr in stress vs calm (long-short): "
          f"VIX>30={long_short['rolling']['mean_when_VIX_gt30']} vs VIX≤30={long_short['rolling']['mean_when_VIX_le30']}")
    print(f"  v015 actual-carry secondary (2023-24, noisy): rho={secondary.get('rho')} (n={secondary.get('n')})")
    print(f"\n  VERDICTS:  long-only={lo_v}   long-short={ls_v}")
    print(f"  RECOMMENDATION: {rec}")
    print(f"\n  Logged. Saved: data/research/edge_research_diversification.json\n{'='*92}\n")


if __name__ == "__main__":
    main()
