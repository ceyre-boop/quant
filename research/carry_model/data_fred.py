"""FRED-only data layer for the 36-year G10 panel (1985+). Daily spot (DEX*), OECD short rates
(IRSTCI01*), OECD CPI (*CPIALL*INMEI), EUR spliced from DEM before 1999. Cached once per series.

HOLDOUT GUARD: `monthly_panel()` refuses to return rows before HOLDOUT_END unless the environment
variable CARRY_HOLDOUT_UNLOCK=1 is set — the sealed HYP-117 test script sets it after gate zero.
The raw caches necessarily contain the holdout years; nothing above this layer can see them."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "carry_model"
HOLDOUT_END = "2005-12-31"                     # 1990-01 → 2005-12 is sealed
CCY = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "SEK", "NOK"]
SPOT = {"JPY": ("DEXJPUS", True), "GBP": ("DEXUSUK", False), "AUD": ("DEXUSAL", False), "NZD": ("DEXUSNZ", False),
        "CAD": ("DEXCAUS", True), "CHF": ("DEXSZUS", True), "SEK": ("DEXSDUS", True), "NOK": ("DEXNOUS", True),
        "EUR": ("DEXUSEU", False)}                        # (series, invert: True if quoted as ccy per USD)
DEM_SPOT = "EXGEUS"                                       # DEM per USD, monthly, pre-1999
DEM_PER_EUR = 1.95583
RATE = {"USD": "IRSTCI01USM156N", "EUR": "IRSTCI01EZM156N", "GBP": "IRSTCI01GBM156N", "JPY": "IRSTCI01JPM156N",
        "AUD": "IRSTCI01AUM156N", "NZD": "IRSTCI01NZM156N", "CAD": "IRSTCI01CAM156N", "CHF": "IRSTCI01CHM156N",
        "SEK": "IRSTCI01SEM156N", "NOK": "IRSTCI01NOM156N", "DEM": "IRSTCI01DEM156N"}
RATE_FB = {"SEK": "IR3TIB01SEM156N", "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N", "EUR": "IR3TIB01EZM156N", "AUD": "IR3TIB01AUM156N"}
CPI = {"USD": "USACPIALLMINMEI", "GBP": "GBRCPIALLMINMEI", "JPY": "JPNCPIALLMINMEI", "AUD": "AUSCPIALLQINMEI",
       "NZD": "NZLCPIALLQINMEI", "CAD": "CANCPIALLMINMEI", "CHF": "CHECPIALLMINMEI", "SEK": "SWECPIALLMINMEI",
       "NOK": "NORCPIALLMINMEI", "EUR": "CP0000EZ19M086NEST", "DEM": "DEUCPIALLMINMEI"}


def _key() -> str:
    env = {k: v for k, v in (l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines() if "=" in l and not l.startswith("#"))}
    return env["FRED_API_KEY"]


def fred(sid: str) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True); f = CACHE / f"{sid}.json"
    if f.exists():
        d = json.loads(f.read_text())
    else:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": sid, "api_key": _key(), "file_type": "json", "observation_start": "1984-01-01"}, timeout=60)
        r.raise_for_status(); d = {x["date"]: x["value"] for x in r.json()["observations"] if x["value"] != "."}
        f.write_text(json.dumps(d))
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in d.items()}).sort_index()
    return s


def month_end_spot() -> pd.DataFrame:
    """USD value of one unit of each currency, month-end. EUR before 1999-01 = DEM spot / 1.95583."""
    out = {}
    for c, (sid, inv) in SPOT.items():
        s = fred(sid); s = (1 / s) if inv else s; out[c] = s.resample("ME").last()
    dem = fred(DEM_SPOT).resample("ME").last()                  # DEM per USD → USD per DEM → per EUR
    eur_pre = (1 / dem) * DEM_PER_EUR
    out["EUR"] = out["EUR"].combine_first(eur_pre[eur_pre.index < "1999-01-01"])
    df = pd.DataFrame(out); df["USD"] = 1.0
    return df[CCY]


def month_rates() -> pd.DataFrame:
    out = {}
    for c, sid in RATE.items():
        if c == "DEM": continue
        s = fred(sid)
        if c in RATE_FB: s = s.combine_first(fred(RATE_FB[c]))
        if c == "EUR": s = s.combine_first(fred(RATE["DEM"])[lambda x: x.index < "1999-01-01"])
        out[c] = s
    df = pd.DataFrame(out).resample("ME").last().ffill(limit=6)
    return df[CCY]


def month_cpi() -> pd.DataFrame:
    out = {}
    for c, sid in CPI.items():
        if c == "DEM": continue
        s = fred(sid)
        if c == "EUR": s = s.combine_first(fred(CPI["DEM"])[lambda x: x.index < "1996-01-01"])
        out[c] = s.resample("ME").last().ffill(limit=4)       # quarterly AU/NZ forward-filled
    return pd.DataFrame(out)[CCY]


def guard(df: pd.DataFrame) -> pd.DataFrame:
    if os.environ.get("CARRY_HOLDOUT_UNLOCK") == "1":
        return df
    return df[df.index > HOLDOUT_END]
