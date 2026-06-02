#!/usr/bin/env python3
"""C3 — Volume profile (honest approximation).

Builds a volume-at-price profile from yfinance 5m bars (last ~5 days): the histogram of
volume traded at each price level, the Point of Control (POC), the Value Area (70% of
volume), and high-volume nodes (support/resistance magnets).

HONEST LIMITATION (baked into the output): this is a VOLUME PROFILE — it shows WHERE volume
traded, NOT the buy/sell aggression split (order-flow delta) that proprietary indicators like
AlgoAlpha show. True delta needs a CME order-flow feed (Databento/CQG) — a paid data tier,
flagged for later, never faked here.

Writes data/briefing/volume_profile.json.

Usage:  python3 -m sovereign.briefing.volume_profile
"""
from __future__ import annotations

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

OUT = ROOT / "data" / "briefing" / "volume_profile.json"
_LIMITATION = ("Volume profile = WHERE volume traded (volume-at-price). NOT order-flow delta "
               "(buy/sell aggression). True delta needs a CME order-flow feed (Databento/CQG).")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build(symbol: str = "NQ=F", days: int = 5, bins: int = 40) -> dict:
    import yfinance as yf
    h = yf.Ticker(symbol).history(period=f"{days}d", interval="5m", auto_adjust=True)
    if h is None or len(h) < 20:
        return {"symbol": symbol, "poc": None, "error": "insufficient intraday data"}

    tp = ((h["High"] + h["Low"] + h["Close"]) / 3).to_numpy(float)
    vol = h["Volume"].to_numpy(float)
    lo, hi = float(tp.min()), float(tp.max())
    if hi <= lo:
        return {"symbol": symbol, "poc": None, "error": "degenerate price range"}

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    hist = np.zeros(bins)
    for i, v in zip(idx, vol):
        hist[i] += v
    centers = (edges[:-1] + edges[1:]) / 2
    total = float(hist.sum())
    if total <= 0:
        return {"symbol": symbol, "poc": None, "error": "no volume"}

    poc_i = int(np.argmax(hist))
    poc = float(centers[poc_i])

    # Value area: highest-volume bins until 70% of total volume is covered.
    order = sorted(range(bins), key=lambda i: hist[i], reverse=True)
    cum, sel = 0.0, []
    for i in order:
        sel.append(i)
        cum += hist[i]
        if cum >= 0.70 * total:
            break
    va_prices = [centers[i] for i in sel]

    return {
        "symbol": symbol,
        "as_of": _now(),
        "days": days, "bins": bins,
        "poc": round(poc, 2),
        "value_area_low": round(float(min(va_prices)), 2),
        "value_area_high": round(float(max(va_prices)), 2),
        "high_volume_nodes": [round(float(centers[i]), 2) for i in order[:5]],
        "price_range": [round(lo, 2), round(hi, 2)],
        "limitation": _LIMITATION,
    }


def build_all(symbols=("ES=F", "NQ=F")) -> dict:
    out = {"as_of": _now(), "instruments": {}}
    for s in symbols:
        try:
            out["instruments"][s] = build(s)
        except Exception as e:
            out["instruments"][s] = {"symbol": s, "poc": None, "error": str(e)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = build_all()
    for sym, d in r["instruments"].items():
        if d.get("error"):
            print(f"{sym}: ERROR {d['error']}")
        else:
            print(f"{sym}: POC {d['poc']} | VA [{d['value_area_low']}, {d['value_area_high']}] "
                  f"| nodes {d['high_volume_nodes']}")
    print(f"  Saved: {OUT.relative_to(ROOT)}")
