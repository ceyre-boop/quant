"""sovereign/sentiment/cot_feed.py — CFTC Commitment of Traders positioning feeder.

Large-speculator (non-commercial) NET futures positioning per currency, from the FREE keyless CFTC
Socrata API (legacy futures-only COT, weekly, 1986+). The hypothesis feature is net_spec as a TRAILING
3-YEAR PERCENTILE — fade-at-extreme is contrarian, so the signal is the percentile, not the raw level.

⚠️ LOOK-AHEAD GUARD (load-bearing): COT is MEASURED each Tuesday but PUBLISHED ~Friday. Every row is dated
to the FRIDAY `publish_date` (measurement Tuesday + publish_lag_days); the board ASOF-joins on
publish_date so it can ONLY ever see a row on/after it was public. The percentile uses TRAILING weeks only
(no look-ahead). A feature dated to the Tuesday measurement would leak 3 days of future info — forbidden.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, upsert

API = "https://publicreporting.cftc.gov/resource/{ds}.json"
_COUNT_COLS = ["noncomm_long", "noncomm_short", "net_spec", "open_interest"]


def fetch_currency(code: str, dataset: str, start: str) -> list:
    """Full weekly COT history for one CFTC contract code from `start`. Paginated. [] on failure."""
    try:
        import requests
        rows, offset = [], 0
        while True:
            r = requests.get(API.format(ds=dataset), timeout=60, params={
                "cftc_contract_market_code": code,
                "$select": ("report_date_as_yyyy_mm_dd,noncomm_positions_long_all,"
                            "noncomm_positions_short_all,open_interest_all"),
                "$where": f"report_date_as_yyyy_mm_dd >= '{start}T00:00:00'",
                "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": 50000, "$offset": offset,
            })
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            rows += batch
            if len(batch) < 50000:
                break
            offset += 50000
        return rows
    except Exception as exc:
        print(f"  [cot] {code}: FETCH FAILED ({type(exc).__name__}: {exc})")
        return []


def update(con=None, start: str | None = None) -> dict:
    """Fetch per-pair COT, compute net/oi/3yr-percentile, Friday-publish-date it, upsert. Returns coverage."""
    cfg = params["sentiment"]["cot"]
    start = start or cfg.get("start", "1986-01-01")
    lag = int(cfg.get("publish_lag_days", 3))
    win = int(cfg.get("percentile_window_weeks", 156))
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    frames, coverage = [], {}
    for pair, code in cfg["pairs"].items():
        rows = fetch_currency(code, cfg["socrata_dataset"], start)
        if not rows:
            coverage[pair] = {"rows": 0}
            continue
        df = pd.DataFrame(rows)
        df["measurement_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce").dt.normalize()
        df["noncomm_long"] = pd.to_numeric(df["noncomm_positions_long_all"], errors="coerce")
        df["noncomm_short"] = pd.to_numeric(df["noncomm_positions_short_all"], errors="coerce")
        df["open_interest"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
        df = df.dropna(subset=["measurement_date", "noncomm_long", "noncomm_short", "open_interest"])
        df = df.sort_values("measurement_date").drop_duplicates("measurement_date", keep="last").reset_index(drop=True)
        df["net_spec"] = df["noncomm_long"] - df["noncomm_short"]
        df["net_oi"] = df["net_spec"] / df["open_interest"].replace(0, np.nan)
        # trailing 3yr percentile of net_spec — TRAILING window only (no look-ahead)
        df["net_pct"] = df["net_spec"].rolling(win, min_periods=52).apply(
            lambda x: float((x <= x[-1]).mean()), raw=True)
        df["publish_date"] = (df["measurement_date"] + pd.Timedelta(days=lag)).dt.date  # Tue → Fri
        df["measurement_date"] = df["measurement_date"].dt.date
        df["pair"] = pair
        df["fetched_at"] = now
        for c in _COUNT_COLS:
            df[c] = df[c].astype("int64")
        out = df[["measurement_date", "publish_date", "pair", "noncomm_long", "noncomm_short",
                  "net_spec", "net_oi", "net_pct", "open_interest", "fetched_at"]]
        frames.append(out)
        coverage[pair] = {"rows": int(len(out)), "start": str(out["measurement_date"].min()),
                          "end": str(out["measurement_date"].max())}
    if frames:
        cot_df = pd.concat(frames, ignore_index=True)
        upsert(con, "sentiment_cot_weekly", cot_df, ["measurement_date", "pair"])
    if own:
        con.close()
    return coverage
