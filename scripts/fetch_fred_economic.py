#!/usr/bin/env python3
"""
FRED US economic-state snapshot — DAILY pull for Oracle cognition context.

Complements (does NOT duplicate) the 4h markets snapshot (data/macro/macro_snapshot.json,
VIX/yields/curve/credit) and the forex carry FRED (per-country rates/CPI/GDP). This pull is the
US *economy* picture — growth, labor, inflation, sentiment — that neither of those carries.

Stored as a daily time series:  data/macro/fred_economic/YYYY-MM-DD.json
Plus a rolling latest:          data/macro/fred_economic_latest.json

Oracle reads the latest in reflect_cycle._load_daily_macro() as QUALITATIVE CONTEXT — it is
never wired to sizing, signals, or any gate (provenance.note records this).

Runs as the first step of the daily Oracle cycle (sovereign/oracle/oracle_cycle.py), and is
runnable standalone:  python3 scripts/fetch_fred_economic.py
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACRO_DIR = ROOT / "data" / "macro" / "fred_economic"
LATEST = ROOT / "data" / "macro" / "fred_economic_latest.json"

# .env loader — same pattern as scripts/fetch_macro_cache.py (works under launchd: no shell env).
def _load_env() -> None:
    env_path = ROOT / ".env"
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


# (series_id, human label, unit). YoY computed for index series; level reported otherwise.
SERIES: list[tuple[str, str, str]] = [
    ("A191RL1Q225SBEA", "Real GDP growth (q/q ann)", "%"),
    ("CPIAUCSL", "CPI", "idx"),
    ("CPILFESL", "Core CPI", "idx"),
    ("PCEPILFE", "Core PCE", "idx"),
    ("UNRATE", "Unemployment rate", "%"),
    ("PAYEMS", "Nonfarm payrolls", "k"),
    ("ICSA", "Initial jobless claims", "#"),
    ("FEDFUNDS", "Fed funds rate", "%"),
    ("DGS10", "10Y Treasury", "%"),
    ("DGS2", "2Y Treasury", "%"),
    ("T10Y2Y", "10Y-2Y spread", "%"),
    ("MORTGAGE30US", "30Y mortgage", "%"),
    ("RSAFS", "Retail sales", "$M"),
    ("UMCSENT", "Consumer sentiment (UMich)", "idx"),
    ("DTWEXBGS", "USD broad index", "idx"),
    ("VIXCLS", "VIX", "idx"),
]
YOY_SERIES = {"CPIAUCSL", "CPILFESL", "PCEPILFE"}   # index series we report as YoY %
TIMEOUT = 12


def _fred(series_id: str, key: str, **params) -> list[dict]:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
           f"&api_key={key}&file_type=json&{qs}")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode()).get("observations", [])


def _latest_and_anchors(series_id: str, key: str) -> dict:
    """Latest value + as-of values 1/2/5/10y ago (for trend), with a YoY % if an index series."""
    try:
        cur = _fred(series_id, key, sort_order="desc", limit=1)
        if not cur or cur[0]["value"] == ".":
            return {"value": None, "err": "no obs"}
        out = {"date": cur[0]["date"], "value": float(cur[0]["value"])}

        def asof(years: int):
            d = date.today().replace(year=date.today().year - years).isoformat()
            try:
                o = _fred(series_id, key, sort_order="desc", limit=1, observation_end=d)
                return float(o[0]["value"]) if o and o[0]["value"] != "." else None
            except Exception:
                return None

        out["y1"], out["y2"], out["y5"], out["y10"] = asof(1), asof(2), asof(5), asof(10)
        if series_id in YOY_SERIES and out.get("y1"):
            out["yoy_pct"] = round((out["value"] / out["y1"] - 1) * 100, 2)
        return out
    except urllib.error.HTTPError as e:
        return {"value": None, "err": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"value": None, "err": type(e).__name__}


def _regime_summary(m: dict) -> dict:
    """Compact derived read — context only, never a gate."""
    def v(sid):
        return (m.get(sid) or {}).get("value")
    curve = v("T10Y2Y")
    return {
        "gdp_growth_pct": v("A191RL1Q225SBEA"),
        "cpi_yoy_pct": (m.get("CPIAUCSL") or {}).get("yoy_pct"),
        "core_cpi_yoy_pct": (m.get("CPILFESL") or {}).get("yoy_pct"),
        "core_pce_yoy_pct": (m.get("PCEPILFE") or {}).get("yoy_pct"),
        "unemployment_pct": v("UNRATE"),
        "fed_funds_pct": v("FEDFUNDS"),
        "ten_year_pct": v("DGS10"),
        "yield_curve_10y2y": curve,
        "yield_curve_state": ("INVERTED" if curve is not None and curve < 0
                              else "FLAT" if curve is not None and curve < 0.25 else "NORMAL"),
        "consumer_sentiment": v("UMCSENT"),
        "vix": v("VIXCLS"),
    }


def fetch_economic_snapshot() -> dict:
    _load_env()
    key = os.environ.get("FRED_API_KEY", "")
    today = date.today().isoformat()
    metrics: dict = {}
    if key:
        for sid, _label, _unit in SERIES:
            metrics[sid] = _latest_and_anchors(sid, key)
    snapshot = {
        "date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": {sid: {"label": lab, "unit": u} for sid, lab, u in SERIES},
        "metrics": metrics,
        "summary": _regime_summary(metrics) if metrics else {},
        "provenance": {
            "source": "fred_economic",
            "verified": bool(key) and bool(metrics),
            "note": "Daily US economic-state context for Oracle cognition. NOT a trading input — "
                    "never wired to sizing, signals, or any gate.",
            "key_present": bool(key),
        },
    }
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    (MACRO_DIR / f"{today}.json").write_text(json.dumps(snapshot, indent=2))
    LATEST.write_text(json.dumps(snapshot, indent=2))
    return snapshot


if __name__ == "__main__":
    s = fetch_economic_snapshot()
    sm = s["summary"]
    ok = s["provenance"]["verified"]
    print(f"FRED economic snapshot {s['date']}: verified={ok}, series={len(s['metrics'])}")
    if ok:
        print(f"  GDP {sm.get('gdp_growth_pct')}% | CPI YoY {sm.get('cpi_yoy_pct')}% | "
              f"unemp {sm.get('unemployment_pct')}% | fed funds {sm.get('fed_funds_pct')}% | "
              f"10Y {sm.get('ten_year_pct')}% | curve {sm.get('yield_curve_state')} | "
              f"sentiment {sm.get('consumer_sentiment')} | VIX {sm.get('vix')}")
    else:
        print("  NOT verified — FRED_API_KEY missing or all series failed. Wrote empty snapshot.")
