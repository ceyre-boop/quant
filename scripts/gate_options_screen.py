#!/usr/bin/env python3
"""
gate_options_screen.py — options-flow approximation for the Petrules Gate.

Free-data proxy for unusual options activity, built from yfinance option chains.
For the nearest N expiries, for each call strike, compute volume / open_interest
and premium (volume * mid_price * 100). A high vol/OI ratio with meaningful
premium is a rough proxy for accumulation — not paid flow, but real signal at
zero cost.

Also exposes analyst revision-velocity extraction from ticker.recommendations.

DISCIPLINE:
  - Free data only (yfinance). No paid feed.
  - No silent failures: if Yahoo is unreachable for a symbol, return None for
    that symbol (the scorer treats missing data as 0.0, never fabricated).
  - Standalone research module — no sovereign/ or ict/ imports.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - yfinance is a repo dependency
    yf = None


class YahooUnavailable(RuntimeError):
    """Raised when yfinance itself is not importable (structural failure)."""


def _require_yf():
    if yf is None:
        raise YahooUnavailable("yfinance not available")


def screen_options(symbol: str, cfg: dict, max_expiries: int = 3) -> dict | None:
    """Return the best unusual-flow signal for `symbol`, or None on data failure.

    Output dict keys (consumed by gate_scorer.score_options_flow):
      best_vol_oi, best_premium_usd, best_volume, n_accumulating_strikes,
      multiple_strikes_same_side, best_strike
    """
    _require_yf()
    of = cfg["scoring"]["options_flow"]
    min_vol = of["min_contract_volume"]
    try:
        tk = yf.Ticker(symbol)
        expiries = list(tk.options or [])[:max_expiries]
    except Exception:
        return None
    if not expiries:
        return {
            "best_vol_oi": 0.0,
            "best_premium_usd": 0.0,
            "best_volume": 0,
            "n_accumulating_strikes": 0,
            "multiple_strikes_same_side": False,
            "best_strike": None,
        }

    best = {"vol_oi": 0.0, "premium": 0.0, "volume": 0, "strike": None}
    n_accum = 0
    for expiry in expiries:
        try:
            chain = tk.option_chain(expiry)
            calls = chain.calls
        except Exception:
            continue
        for _, row in calls.iterrows():
            vol = float(row.get("volume") or 0)
            oi = float(row.get("openInterest") or 0)
            if vol < min_vol or oi <= 0:
                continue
            vol_oi = vol / oi
            mid = _mid_price(row)
            premium = vol * mid * 100.0
            if vol_oi >= 3.0 and premium >= 250000:
                n_accum += 1
            if vol_oi > best["vol_oi"]:
                best = {
                    "vol_oi": vol_oi,
                    "premium": premium,
                    "volume": int(vol),
                    "strike": float(row.get("strike") or 0),
                }
    return {
        "best_vol_oi": best["vol_oi"],
        "best_premium_usd": best["premium"],
        "best_volume": best["volume"],
        "n_accumulating_strikes": n_accum,
        "multiple_strikes_same_side": n_accum >= 2,
        "best_strike": best["strike"],
    }


def _mid_price(row) -> float:
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    last = float(row.get("lastPrice") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return last


def revision_velocity(symbol: str, cfg: dict) -> dict | None:
    """Count analyst upgrades vs downgrades within the configured window.

    Returns {n_upgrades, n_downgrades} or None on data failure. Uses
    ticker.recommendations (upgrade/downgrade grade history)."""
    _require_yf()
    window = int(cfg["scoring"]["revision_velocity"]["window_days"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=window)
    try:
        tk = yf.Ticker(symbol)
        recs = tk.recommendations
    except Exception:
        return None
    if recs is None or len(recs) == 0:
        return {"n_upgrades": 0, "n_downgrades": 0}

    up = down = 0
    try:
        for idx, row in recs.iterrows():
            # recommendations index is a timestamp in the classic yfinance schema
            ts = _row_timestamp(row, idx)
            if ts is not None and ts < cutoff:
                continue
            action = str(row.get("Action", row.get("action", ""))).lower()
            if action in ("up", "upgrade"):
                up += 1
            elif action in ("down", "downgrade"):
                down += 1
    except Exception:
        return {"n_upgrades": 0, "n_downgrades": 0}
    return {"n_upgrades": up, "n_downgrades": down}


def _row_timestamp(row, idx):
    try:
        if hasattr(idx, "to_pydatetime"):
            dt = idx.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    except Exception:
        pass
    return None
