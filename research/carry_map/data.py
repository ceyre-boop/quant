"""Data for the 20-year G10 carry map. yfinance spot (2003+), OECD immediate short rates via FRED
(IRSTCI01*, monthly, 2000+), VIX. Cached once. Read-only inputs; nothing here is a strategy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "carry_map"
CCY = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "SEK", "NOK"]
FRED = {"USD": "IRSTCI01USM156N", "EUR": "IRSTCI01EZM156N", "GBP": "IRSTCI01GBM156N", "JPY": "IRSTCI01JPM156N",
        "AUD": "IRSTCI01AUM156N", "NZD": "IRSTCI01NZM156N", "CAD": "IRSTCI01CAM156N", "CHF": "IRSTCI01CHM156N",
        "SEK": "IRSTCI01SEM156N", "NOK": "IRSTCI01NOM156N"}
# policy-rate fallbacks where the OECD series ends early (SEK 2020-10, CHF 2024-03, NZD 2024-12)
FRED_FALLBACK = {"SEK": "IR3TIB01SEM156N", "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N", "EUR": "IR3TIB01EZM156N"}
YF = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "USDJPY=X", "AUD": "AUDUSD=X", "NZD": "NZDUSD=X",
      "CAD": "USDCAD=X", "CHF": "USDCHF=X", "SEK": "USDSEK=X", "NOK": "USDNOK=X"}
INVERT = {"JPY", "CAD", "CHF", "SEK", "NOK"}          # quoted USD/XXX → convert to XXX per USD (USD value of 1 unit of ccy)


def _key() -> str:
    env = {k: v for k, v in (l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines() if "=" in l and not l.startswith("#"))}
    return env["FRED_API_KEY"]


def rates() -> pd.DataFrame:
    """Monthly short rates (% p.a.) per currency, month-start index, forward-filled ≤ 3 months."""
    CACHE.mkdir(parents=True, exist_ok=True); f = CACHE / "rates_monthly.parquet"
    if f.exists():
        return pd.read_parquet(f)
    out = {}
    for c, sid in FRED.items():
        s = _fred(sid)
        if c in FRED_FALLBACK:
            fb = _fred(FRED_FALLBACK[c]); s = s.combine_first(fb) if fb is not None else s
        out[c] = s
    df = pd.DataFrame(out).sort_index(); df.index = pd.to_datetime(df.index)
    df = df.ffill(limit=3); df.to_parquet(f); return df


def _fred(sid: str):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": sid, "api_key": _key(), "file_type": "json", "observation_start": "2000-01-01"}, timeout=60)
    if r.status_code != 200:
        return None
    o = [x for x in r.json()["observations"] if x["value"] != "."]
    return pd.Series({pd.Timestamp(x["date"]): float(x["value"]) for x in o})


def spot_usd() -> pd.DataFrame:
    """Daily USD value of one unit of each currency (USD column = 1). yfinance auto-adjusted closes."""
    CACHE.mkdir(parents=True, exist_ok=True); f = CACHE / "spot_usd_daily.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import yfinance as yf
    px = yf.download(list(YF.values()) + ["^VIX"], start="2003-01-01", end="2026-09-03", auto_adjust=True, progress=False)["Close"]
    out = pd.DataFrame(index=px.index)
    for c, t in YF.items():
        out[c] = (1.0 / px[t]) if c in INVERT else px[t]
    out["USD"] = 1.0; out["VIX"] = px["^VIX"]
    out = out.dropna(subset=[c for c in YF]).ffill()
    out.to_parquet(f); return out
