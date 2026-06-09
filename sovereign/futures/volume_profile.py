"""Volume profile for the futures sandbox (Increment 4) — POC / Value Area / nodes.

Sandbox-local: no forex/ICT/intelligence imports. Re-implements the proven algorithm from
sovereign/briefing/volume_profile.py (histogram bins → POC=argmax → top bins to value_area_pct)
but operates on a bar DataFrame the replay already holds (no fetch).

HONEST LIMITATION (same as the briefing module): this is volume-at-price — WHERE volume traded,
NOT the buy/sell aggression split (order-flow delta). And on yfinance, futures volume itself is
unreliable, so treat profiles built from yfinance as telemetry, not truth. Real volume = IB.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sovereign.futures.config import futures_params


def compute_profile(df, bins: Optional[int] = None,
                    value_area_pct: Optional[float] = None) -> Optional[dict]:
    """Volume-at-price profile from an OHLCV DataFrame. Returns
    {poc, vah, val, nodes, price_range} or None if data/volume is insufficient."""
    if df is None or len(df) < 5:
        return None
    p = futures_params()["volume_profile"]
    bins = bins or p["bins"]
    value_area_pct = value_area_pct or p["value_area_pct"]

    tp = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    lo, hi = float(np.nanmin(tp)), float(np.nanmax(tp))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    hist = np.zeros(bins)
    for i, v in zip(idx, vol):
        if np.isfinite(v):
            hist[i] += v
    centers = (edges[:-1] + edges[1:]) / 2
    total = float(hist.sum())
    if total <= 0:
        return None

    poc_i = int(np.argmax(hist))
    order = sorted(range(bins), key=lambda i: hist[i], reverse=True)
    cum, sel = 0.0, []
    for i in order:
        sel.append(i)
        cum += hist[i]
        if cum >= value_area_pct * total:
            break
    va_prices = [centers[i] for i in sel]
    return {
        "poc": round(float(centers[poc_i]), 2),
        "vah": round(float(max(va_prices)), 2),
        "val": round(float(min(va_prices)), 2),
        "nodes": [round(float(centers[i]), 2) for i in order[:5]],
        "price_range": [round(lo, 2), round(hi, 2)],
    }


def confluence_score(level: float, profile: Optional[dict], tol_price: float) -> int:
    """How many of {POC, VAH, VAL} the entry `level` sits within `tol_price` of (0–3).
    The 1% scalper's edge: an entry that lands on a real volume level, not just an indicator."""
    if not profile or level is None:
        return 0
    score = 0
    for key in ("poc", "vah", "val"):
        lv = profile.get(key)
        if lv is not None and abs(level - lv) <= tol_price:
            score += 1
    return score
