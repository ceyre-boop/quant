"""ThetaData v2 REST client for the HYP-111/112 program. Local terminal on :25510, no auth
header (the terminal logs itself in). Never logs or prints credentials. Every response is
cached to parquet so a run never re-fetches; 471 (before first-access date) and 472
(no data) are cached as empty frames — the emptiness is a fact about the tier/data, not a
transient. 403 raises. No retry storm: one retry on connection error, then raise.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:25510"
CACHE_1M = ROOT / "data" / "cache" / "theta_1m"
CACHE_OPT = ROOT / "data" / "cache" / "theta_opt_eod"
CACHE_EXP = ROOT / "data" / "cache" / "theta_opt_eod" / "_expirations"
PACE_S = 0.1


class ThetaEntitlement(RuntimeError):
    pass


def _get(path: str, params: dict, timeout: int = 120) -> tuple[int, dict | str]:
    url = f"{BASE}{path}"
    for attempt in (0, 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            break
        except requests.RequestException:
            if attempt:
                raise
            time.sleep(2.0)
    time.sleep(PACE_S)
    if r.status_code == 200:
        return 200, r.json()
    return r.status_code, r.text.strip()[:200]


def _hhmm(ms: int) -> str:
    return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}"


def stock_1m(sym: str, date: str) -> pd.DataFrame:
    """RTH 1-minute OHLCV for one session. Frame contract mirrors backtester/data.py::_BAR_COLS:
    time 'HH:MM' ET string, open/high/low/close/volume float. Empty frame = not served."""
    CACHE_1M.mkdir(parents=True, exist_ok=True)
    f = CACHE_1M / f"{sym}_{date}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = date.replace("-", "")
    code, body = _get("/v2/hist/stock/ohlc", {"root": sym, "start_date": d, "end_date": d,
                                              "ivl": 60000, "rth": "true"})
    cols = ["time", "open", "high", "low", "close", "volume"]
    if code in (471, 472):
        df = pd.DataFrame(columns=cols)
    elif code != 200:
        raise ThetaEntitlement(f"{sym} {date}: HTTP {code} {body}")
    else:
        fmt = body["header"]["format"]
        rows = body["response"]
        ix = {k: fmt.index(k) for k in ("ms_of_day", "open", "high", "low", "close", "volume")}
        df = pd.DataFrame({
            "time": [_hhmm(r[ix["ms_of_day"]]) for r in rows],
            **{k: [float(r[ix[k]]) for r in rows] for k in ("open", "high", "low", "close", "volume")},
        })
        df = df[(df["time"] >= "09:30") & (df["time"] <= "15:59")].reset_index(drop=True)
    df.to_parquet(f, index=False)
    return df


def expirations(root: str) -> list[str]:
    CACHE_EXP.mkdir(parents=True, exist_ok=True)
    f = CACHE_EXP / f"{root}.json"
    if f.exists():
        return json.loads(f.read_text())
    code, body = _get("/v2/list/expirations", {"root": root})
    if code != 200:
        raise ThetaEntitlement(f"{root} expirations: HTTP {code} {body}")
    exps = [str(x) for x in body["response"]]
    f.write_text(json.dumps(exps))
    return exps


def option_bulk_eod(root: str, exp: str, start: str, end: str) -> pd.DataFrame:
    """All strikes/rights for one expiration over [start, end] (YYYYMMDD), last EOD NBBO tick per
    (date, strike, right). Columns: date, strike, right, bid, ask, close, volume."""
    CACHE_OPT.mkdir(parents=True, exist_ok=True)
    f = CACHE_OPT / f"{root}_{exp}_{start}_{end}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    code, body = _get("/v2/bulk_hist/option/eod", {"root": root, "exp": exp,
                                                   "start_date": start, "end_date": end}, timeout=180)
    cols = ["date", "strike", "right", "bid", "ask", "close", "volume"]
    if code in (471, 472):
        df = pd.DataFrame(columns=cols)
    elif code != 200:
        raise ThetaEntitlement(f"{root} {exp}: HTTP {code} {body}")
    else:
        fmt = body["header"]["format"]
        ix = {k: fmt.index(k) for k in ("ms_of_day", "bid", "ask", "close", "volume", "date")}
        recs = []
        for c in body["response"]:
            k, strike, right = c["contract"], c["contract"]["strike"] / 1000.0, c["contract"]["right"]
            last = {}
            for t in c["ticks"]:                       # keep the last tick per date
                last[t[ix["date"]]] = t
            for dt, t in last.items():
                recs.append((str(dt), strike, right, float(t[ix["bid"]]), float(t[ix["ask"]]),
                             float(t[ix["close"]]), float(t[ix["volume"]])))
        df = pd.DataFrame(recs, columns=cols)
    df.to_parquet(f, index=False)
    return df
