#!/usr/bin/env python3
"""C2 — NQ/ES lead-lag regime engine (the crown jewel of the briefing).

The NQ/ES spread is a regime CLASSIFIER, not an edge:
  - CONCENTRATION : NQ leads / pulls away from ES → narrow, tech/AI-driven, FRAGILE.
  - BREADTH       : ES & NQ lockstep (or ES leads) → broad agreement, more DURABLE.
  - ROTATION_WARN : NQ and ES diverging (one up, one down/flat) → rotation, regime shifting.

Enriches the Phase-1 classifier with rolling spreads (1d/5d/20d) and an intraday 5m
cross-correlation at lags -3..+3 to detect which instrument moves FIRST.

Output is a regime INPUT (informs how much to trust a directional move / how to size), NOT
a trade signal. ES/NQ are CME futures — DATA + classification only, no execution path.

Writes BOTH:
  - data/research/nqes_regime.json  (legacy key read by reflect_cycle + oracle_session_open)
  - data/briefing/lead_lag_regime.json  (richer record)

Usage:  python3 -m sovereign.briefing.lead_lag [--lookback 20]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for _lib in ("yfinance", "urllib3", "requests"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

LEGACY_OUT = ROOT / "data" / "research" / "nqes_regime.json"
RICH_OUT = ROOT / "data" / "briefing" / "lead_lag_regime.json"

# Documented heuristic thresholds (not optimized — a regime classifier, not a fitted edge):
SPREAD_CONCENTRATION = 0.0015   # avg daily NQ-ES return spread beyond this = a real tilt
LEADLAG_STRONG = 0.30           # |lagged cross-corr| above this = a real lead-lag relationship


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _daily_returns(sym: str, n: int):
    import yfinance as yf
    h = yf.Ticker(sym).history(period=f"{n + 40}d", interval="1d", auto_adjust=True)
    if h is None or len(h) < n + 2:
        return None, None
    close = h["Close"].astype(float)
    return close.pct_change().dropna(), float(close.iloc[-1])


def _intraday_returns(sym: str, days: int = 5):
    import yfinance as yf
    h = yf.Ticker(sym).history(period=f"{days}d", interval="5m", auto_adjust=True)
    if h is None or len(h) < 10:
        return None
    return h["Close"].astype(float).pct_change().dropna()


def _xcorr_lags(nq: np.ndarray, es: np.ndarray, max_lag: int = 3) -> dict:
    """Cross-correlation at lags -max_lag..+max_lag.
    lag k>0 pairs nq[t] with es[t-k] → ES leads NQ; k<0 → NQ leads ES."""
    n = min(len(nq), len(es))
    nq, es = nq[-n:], es[-n:]
    by_lag = {}
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            a, b = nq, es
        elif k > 0:
            a, b = nq[k:], es[:-k]
        else:
            kk = -k
            a, b = nq[:-kk], es[kk:]
        m = min(len(a), len(b))
        if m > 2:
            by_lag[k] = round(float(np.corrcoef(a[-m:], b[-m:])[0, 1]), 3)
    best = max((k for k in by_lag if k != 0), key=lambda k: abs(by_lag[k]), default=0)
    best_corr = by_lag.get(best, 0.0)
    if best > 0 and abs(best_corr) >= LEADLAG_STRONG:
        leader = "ES"
    elif best < 0 and abs(best_corr) >= LEADLAG_STRONG:
        leader = "NQ"
    else:
        leader = "NONE"
    return {"by_lag": {str(k): v for k, v in by_lag.items()},
            "best_lag": best, "best_corr": best_corr, "leader": leader}


def _write(res: dict) -> None:
    for p in (LEGACY_OUT, RICH_OUT):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))


def classify(lookback: int = 20) -> dict:
    nq_r, nq_px = _daily_returns("NQ=F", lookback)
    es_r, es_px = _daily_returns("ES=F", lookback)
    if nq_r is None or es_r is None:
        res = {"as_of": _now(), "regime": "NO_DATA", "read": "ES=F/NQ=F fetch failed",
               "note": "regime input, not an edge"}
        _write(res)
        return res

    common = nq_r.index.intersection(es_r.index)
    nq = nq_r.loc[common].to_numpy()[-lookback:]
    es = es_r.loc[common].to_numpy()[-lookback:]

    def _spread(w):
        return round(float(np.mean(nq[-w:] - es[-w:])), 5) if len(nq) >= w else None

    spread_1d = round(float(nq[-1] - es[-1]), 5)
    spread_5d = _spread(5)
    spread_20d = _spread(min(20, lookback))
    avg_spread = next((s for s in (spread_20d, spread_5d, spread_1d) if s is not None), 0.0)
    corr = float(np.corrcoef(nq, es)[0, 1]) if len(nq) > 2 else 0.0
    cum_spread = round(float(np.sum(nq - es)), 4)

    nq_i = _intraday_returns("NQ=F")
    es_i = _intraday_returns("ES=F")
    if nq_i is not None and es_i is not None:
        ci = nq_i.index.intersection(es_i.index)
        ll = _xcorr_lags(nq_i.loc[ci].to_numpy(), es_i.loc[ci].to_numpy())
    else:
        ll = {"by_lag": {}, "best_lag": 0, "best_corr": 0.0, "leader": "NONE"}

    divergence = bool(np.sign(nq[-5:].sum()) != np.sign(es[-5:].sum())) if len(nq) >= 5 else False

    if divergence and abs(avg_spread) >= SPREAD_CONCENTRATION:
        regime = "ROTATION_WARN"
        read = ("NQ and ES diverging (one up, one down/flat) — rotation underway, regime "
                "shifting. Tech-led longs are suspect.")
    elif avg_spread > SPREAD_CONCENTRATION and cum_spread > 0:
        regime = "CONCENTRATION"
        read = ("NQ leading / pulling away from ES — narrow, tech/AI-driven, FRAGILE. Trust "
                "directional moves less; watch for reversal when AI names cool.")
    elif avg_spread < -SPREAD_CONCENTRATION:
        regime = "ROTATION_WARN"
        read = ("NQ weak vs ES — rotation OUT of tech. A breadth/rotation warning; tech-led "
                "longs are suspect.")
    else:
        regime = "BREADTH"
        read = ("ES & NQ moving together — broad agreement, more DURABLE directional signal. "
                "Trust the direction more.")

    res = {
        "as_of": _now(),
        "lookback_days": lookback,
        "nq_last": round(nq_px, 1), "es_last": round(es_px, 1),
        "nq_es_return_spread_avg": avg_spread,
        "spread_1d": spread_1d, "spread_5d": spread_5d, "spread_20d": spread_20d,
        "nq_es_cumulative_spread": cum_spread,
        "contemporaneous_corr": round(corr, 3),
        "lead_lag": ll,
        "divergence": divergence,
        "regime": regime,
        "read": read,
        "note": ("Regime INPUT, not an edge. ES/NQ research only — no execution path "
                 "(CME futures; OANDA is forex-only)."),
    }
    _write(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=20)
    args = ap.parse_args()
    r = classify(args.lookback)
    print(f"NQ/ES regime: {r['regime']}")
    if r["regime"] != "NO_DATA":
        print(f"  NQ {r['nq_last']} / ES {r['es_last']} | spread_avg {r['nq_es_return_spread_avg']:+.4f} "
              f"| corr {r['contemporaneous_corr']} | leader {r['lead_lag']['leader']} "
              f"(lag {r['lead_lag']['best_lag']}, corr {r['lead_lag']['best_corr']})")
        print(f"  → {r['read']}")
    print(f"  Saved: {LEGACY_OUT.relative_to(ROOT)} + {RICH_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
