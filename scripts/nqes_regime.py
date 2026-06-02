#!/usr/bin/env python3
"""
NQ/ES lead-lag regime classifier (ES/NQ workstream, Phase 1 — research).
========================================================================

The NQ/ES spread is a regime classifier, not an edge. Per the morning-briefing logic:
  - CONCENTRATION       : NQ leads / pulls away from ES → narrow, tech/AI-driven, FRAGILE.
  - BREADTH             : ES & NQ lockstep (or ES leads) → broad agreement, more DURABLE.
  - ROTATION_DIVERGENCE : NQ weak while ES holds → rotation warning (the May-27 pattern).

Output is a regime INPUT (informs how much to trust a directional move / how to size), NOT a
trade signal. ES/NQ are CME futures — this is DATA + classification only; no execution path here
(OANDA is forex-only; live ES/NQ needs a separate futures broker).

Writes data/research/nqes_regime.json. Reads ES=F / NQ=F via yfinance (daily).

Usage:  python3 scripts/nqes_regime.py [--lookback 20]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

OUT = ROOT / "data" / "research" / "nqes_regime.json"

# Thresholds (documented, not optimized — a regime heuristic):
SPREAD_CONCENTRATION = 0.0015   # NQ - ES daily-return spread (avg over lookback) this far + → NQ leading
LEADLAG_STRONG = 0.30           # |lag-1 cross-correlation| above this = a real lead-lag relationship


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _returns(sym: str, n: int):
    import yfinance as yf
    h = yf.Ticker(sym).history(period=f"{n + 40}d", interval="1d", auto_adjust=True)
    if len(h) < n + 2:
        return None, None
    close = h["Close"].astype(float)
    rets = close.pct_change().dropna()
    return rets, float(close.iloc[-1])


def _lead_lag(nq: np.ndarray, es: np.ndarray) -> dict:
    """Cross-correlation at lag ±1: does ES today predict NQ tomorrow (ES leads) or vice versa?"""
    n = min(len(nq), len(es))
    nq, es = nq[-n:], es[-n:]
    if n < 5:
        return {"es_leads_nq": 0.0, "nq_leads_es": 0.0}
    # corr(ES[t-1], NQ[t]) → ES leads NQ ; corr(NQ[t-1], ES[t]) → NQ leads ES
    es_leads = float(np.corrcoef(es[:-1], nq[1:])[0, 1]) if n > 2 else 0.0
    nq_leads = float(np.corrcoef(nq[:-1], es[1:])[0, 1]) if n > 2 else 0.0
    return {"es_leads_nq": round(es_leads, 3), "nq_leads_es": round(nq_leads, 3)}


def classify(lookback: int = 20) -> dict:
    nq_r, nq_px = _returns("NQ=F", lookback)
    es_r, es_px = _returns("ES=F", lookback)
    if nq_r is None or es_r is None:
        return {"as_of": _now(), "regime": "NO_DATA", "reason": "ES=F/NQ=F fetch failed"}

    # Align on common dates
    common = nq_r.index.intersection(es_r.index)[-lookback:]
    nq = nq_r.loc[common].to_numpy()
    es = es_r.loc[common].to_numpy()

    spread = float(np.mean(nq - es))              # +ve → NQ outperforming ES (tech leading)
    corr = float(np.corrcoef(nq, es)[0, 1]) if len(nq) > 2 else 0.0
    ll = _lead_lag(nq, es)
    cum_spread = float(np.sum(nq - es))           # cumulative NQ-ES over the window

    # Classify
    if spread > SPREAD_CONCENTRATION and cum_spread > 0:
        regime = "CONCENTRATION"
        read = "NQ leading / pulling away from ES — narrow, tech/AI-driven, FRAGILE. Trust directional moves less; watch for reversal when AI names cool."
    elif spread < -SPREAD_CONCENTRATION:
        regime = "ROTATION_DIVERGENCE"
        read = "NQ weak vs ES — rotation OUT of tech. A breadth/rotation warning; tech-led longs are suspect."
    else:
        regime = "BREADTH"
        read = "ES & NQ moving together — broad agreement, more DURABLE directional signal. Trust the direction more."

    return {
        "as_of": _now(),
        "lookback_days": lookback,
        "nq_last": round(nq_px, 1), "es_last": round(es_px, 1),
        "nq_es_return_spread_avg": round(spread, 5),
        "nq_es_cumulative_spread": round(cum_spread, 4),
        "contemporaneous_corr": round(corr, 3),
        "lead_lag": ll,
        "regime": regime,
        "read": read,
        "note": "Regime INPUT, not an edge. ES/NQ research only — no execution path (CME futures; OANDA is forex-only).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=20)
    args = ap.parse_args()
    result = classify(args.lookback)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"NQ/ES regime: {result['regime']}")
    if result["regime"] != "NO_DATA":
        print(f"  NQ {result['nq_last']} / ES {result['es_last']} | spread_avg {result['nq_es_return_spread_avg']:+.4f} "
              f"| corr {result['contemporaneous_corr']} | lead-lag {result['lead_lag']}")
        print(f"  → {result['read']}")
    print(f"  Saved: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
