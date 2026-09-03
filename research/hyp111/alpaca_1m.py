"""Alpaca SIP 1-minute RTH bars, same frame contract as theta_v2.stock_1m (time 'HH:MM' ET, ohlcv).
Cache data/cache/alpaca_1m_rth/{SYM}_{DATE}.parquet. Key from ~/quant/.env; never printed."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "alpaca_1m_rth"
_H = None


def _headers() -> dict:
    global _H
    if _H is None:
        env = {k: v for k, v in (l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
                                 if "=" in l and not l.startswith("#"))}
        _H = {"APCA-API-KEY-ID": env.get("ALPACA_API_KEY") or env["APCA_API_KEY_ID"],
              "APCA-API-SECRET-KEY": env.get("ALPACA_SECRET_KEY") or env["APCA_API_SECRET_KEY"]}
    return _H


def stock_1m(sym: str, date: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sym}_{date}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows, token = [], None
    for _ in range(5):
        day = pd.Timestamp(date).tz_localize("America/New_York")
        p = {"timeframe": "1Min", "start": (day + pd.Timedelta(hours=9, minutes=25)).isoformat(),
             "end": (day + pd.Timedelta(hours=16, minutes=5)).isoformat(),
             "feed": "sip", "limit": 10000, "adjustment": "all"}
        if token: p["page_token"] = token
        r = requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/bars", headers=_headers(), params=p, timeout=60)
        if r.status_code == 429:
            time.sleep(3); continue
        r.raise_for_status()
        j = r.json(); rows += j.get("bars") or []; token = j.get("next_page_token")
        if not token: break
    ts = pd.to_datetime([b["t"] for b in rows], utc=True).tz_convert("America/New_York") if rows else []
    df = pd.DataFrame({"time": [t.strftime("%H:%M") for t in ts],
                       **{k: [float(b[k]) for b in rows] for k in ("o", "h", "l", "c", "v")}}
                      ).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    if len(df):
        df = df[(df["time"] >= "09:30") & (df["time"] <= "15:59")].reset_index(drop=True)
    df.to_parquet(f, index=False)
    return df
