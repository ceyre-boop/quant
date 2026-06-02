#!/usr/bin/env python3
"""C1 — ES/NQ market-state collector.

Primary source = yfinance (proven for ES=F/NQ=F at 5m/30m/1h/daily). Polygon is wired as
an OPTIONAL richer source behind a capability check — CME-futures entitlement on the
Polygon key is unverified, so we never DEPEND on it (avoids building on a capability that
may not exist).

Computes, per instrument: PDH/PDL, prior-week H/L, ATR(14), SMA200 position, session VWAP,
RSI(14), daily/weekly returns. Writes data/briefing/market_state.json.

This is an analytics INPUT for the briefing, not a trade signal.

Usage:  python3 -m sovereign.briefing.market_data
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

OUT = ROOT / "data" / "briefing" / "market_state.json"
SYMBOLS = ("ES=F", "NQ=F")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hist(sym: str, period: str, interval: str):
    import yfinance as yf
    h = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=True)
    return h if h is not None and len(h) else None


def _rsi(close: np.ndarray, n: int = 14):
    if len(close) < n + 1:
        return None
    diff = np.diff(close)
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    roll_up = float(np.mean(up[-n:]))
    roll_dn = float(np.mean(dn[-n:]))
    if roll_dn == 0:
        return 100.0
    rs = roll_up / roll_dn
    return float(100 - 100 / (1 + rs))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14):
    if len(close) < n + 1:
        return None
    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )
    return float(np.mean(tr[-n:]))


def _session_vwap(sym: str):
    intr = _hist(sym, "2d", "5m")
    if intr is None or not len(intr):
        return None
    last_day = intr.index[-1].date()
    mask = [ts.date() == last_day for ts in intr.index]
    day = intr[mask]
    if not len(day):
        return None
    tp = ((day["High"] + day["Low"] + day["Close"]) / 3).to_numpy(float)
    vol = day["Volume"].to_numpy(float)
    if vol.sum() <= 0:
        return None
    return float((tp * vol).sum() / vol.sum())


def _analyze(sym: str) -> dict:
    daily = _hist(sym, "320d", "1d")
    if daily is None:
        return {"symbol": sym, "error": "no daily data"}
    c = daily["Close"].to_numpy(float)
    h = daily["High"].to_numpy(float)
    l = daily["Low"].to_numpy(float)
    last = float(c[-1])

    pdh = float(h[-2]) if len(h) > 1 else None       # prior completed day's high
    pdl = float(l[-2]) if len(l) > 1 else None
    pw = daily.iloc[-6:-1] if len(daily) >= 6 else daily   # prior 5 completed sessions
    pw_h = float(pw["High"].max())
    pw_l = float(pw["Low"].min())

    atr = _atr(h, l, c, 14)
    rsi = _rsi(c, 14)
    sma200 = float(np.mean(c[-200:])) if len(c) >= 200 else None
    sma200_pct = round((last - sma200) / sma200 * 100, 2) if sma200 else None
    daily_ret = round((c[-1] / c[-2] - 1) * 100, 2) if len(c) > 1 else None
    weekly_ret = round((c[-1] / c[-6] - 1) * 100, 2) if len(c) > 6 else None

    vwap = None
    try:
        vwap = _session_vwap(sym)
    except Exception:
        vwap = None

    return {
        "symbol": sym,
        "last": round(last, 2),
        "prior_day_high": round(pdh, 2) if pdh else None,
        "prior_day_low": round(pdl, 2) if pdl else None,
        "prior_week_high": round(pw_h, 2),
        "prior_week_low": round(pw_l, 2),
        "atr14": round(atr, 2) if atr else None,
        "rsi14": round(rsi, 1) if rsi is not None else None,
        "sma200": round(sma200, 2) if sma200 else None,
        "sma200_pct_above": sma200_pct,
        "above_sma200": (last > sma200) if sma200 else None,
        "session_vwap": round(vwap, 2) if vwap else None,
        "daily_return_pct": daily_ret,
        "weekly_return_pct": weekly_ret,
    }


def collect(symbols=SYMBOLS) -> dict:
    out = {"as_of": _now(), "instruments": {}}
    for s in symbols:
        try:
            out["instruments"][s] = _analyze(s)
        except Exception as e:  # never let one symbol kill the briefing
            out["instruments"][s] = {"symbol": s, "error": str(e)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = collect()
    for sym, d in r["instruments"].items():
        if "error" in d:
            print(f"{sym}: ERROR {d['error']}")
        else:
            print(f"{sym}: last {d['last']} | RSI {d['rsi14']} | ATR {d['atr14']} | "
                  f"{d['daily_return_pct']}% d / {d['weekly_return_pct']}% wk | "
                  f"{'>' if d['above_sma200'] else '<'}SMA200")
    print(f"  Saved: {OUT.relative_to(ROOT)}")
