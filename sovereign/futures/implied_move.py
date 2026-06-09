"""Options/VIX-implied expected daily move — the MAGNITUDE ingredient of the Big Move Oracle.

The market itself prices how big today is expected to be: VIX is annualized implied vol (%),
so the 1-day expected move ≈ price · (VIX/100) / √252. This is the single genuinely-new input
the Big Move Oracle adds — it tells us the *size* of the move to expect before direction.

Sandbox-local: no forex/ICT/intelligence imports. Null-safe (returns None VIX if unavailable).
"""
from __future__ import annotations

from math import sqrt
from typing import Optional

TRADING_DAYS = 252


def get_vix() -> Optional[float]:
    """Latest VIX close. yfinance ^VIX primary; None if unavailable (never raises)."""
    try:
        import yfinance as yf
        d = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if d is None or d.empty:
            return None
        return round(float(d["Close"].iloc[-1].item()), 2)
    except Exception:
        return None


def expected_daily_move(price: float, vix: Optional[float] = None) -> dict:
    """Expected 1-day move in points and % from VIX. {vix, expected_move_pts, expected_move_pct,
    one_sigma_upper, one_sigma_lower}. If VIX is missing, pts/pct are None (fail-soft)."""
    if vix is None:
        vix = get_vix()
    if vix is None or price is None or price <= 0:
        return {"vix": vix, "expected_move_pts": None, "expected_move_pct": None,
                "one_sigma_upper": None, "one_sigma_lower": None}
    move_pct = (vix / 100.0) / sqrt(TRADING_DAYS)        # 1-sigma daily, as a fraction
    move_pts = price * move_pct
    return {
        "vix": vix,
        "expected_move_pts": round(move_pts, 2),
        "expected_move_pct": round(move_pct * 100, 3),
        "one_sigma_upper": round(price + move_pts, 2),
        "one_sigma_lower": round(price - move_pts, 2),
    }
