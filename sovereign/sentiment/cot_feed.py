"""sovereign/sentiment/cot_feed.py — CFTC Commitment of Traders positioning feeder.

Two reports, one keying discipline:
- LEGACY futures-only (6dca-aqww): large-speculator (non-commercial) NET positioning → sentiment_cot_weekly.
- TFF futures-only (gpe5-46if): leveraged-funds NET positioning (the professional-speculator analog;
  for currencies "disaggregated" = the TFF report) → sentiment_cot_tff_weekly.

Features (all TRAILING-window, no look-ahead): net/oi level, 3y + 1y percentiles, 1y + 3y z-scores,
and flush_1w (WoW Δnet normalized by the trailing-1y Δ std — the forced-unwind feature, HYP-073).
Crosses (AUDNZD) are leg differences: net_spec = base_net − quote_net, net_oi = base_oi-norm − quote.

⚠️ LOOK-AHEAD GUARD (load-bearing): COT is MEASURED each Tuesday but PUBLISHED ~Friday 15:30 ET. Every
row is dated to the FRIDAY `publish_date` (measurement + publish_lag_days) with `release_ts` carrying
the 15:30 ET provenance; the board ASOF-joins on publish_date so it can ONLY see a row on/after it was
public. Holiday weeks slip the real release to Monday (~5/yr): the +3d approximation can run ≤1 business
day early those weeks — a documented, audited bias (scripts/audit_look_ahead.py counts affected weeks).
The feeder stays in CURRENCY space (6J net is JPY-vs-USD): USDJPY direction rules invert at TEST time
per the HYP-072 pre-registration — never here.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, upsert

API = "https://publicreporting.cftc.gov/resource/{ds}.json"
_LEGACY_SELECT = ("report_date_as_yyyy_mm_dd,noncomm_positions_long_all,"
                  "noncomm_positions_short_all,open_interest_all")
_TFF_SELECT = ("report_date_as_yyyy_mm_dd,lev_money_positions_long,"
               "lev_money_positions_short,open_interest_all")


def fetch_currency(code: str, dataset: str, start: str, select: str = _LEGACY_SELECT) -> list:
    """Full weekly COT history for one CFTC contract code from `start`. Paginated. [] on failure."""
    try:
        import requests
        rows, offset = [], 0
        while True:
            r = requests.get(API.format(ds=dataset), timeout=60, params={
                "cftc_contract_market_code": code,
                "$select": select,
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


def _frame(rows: list, long_col: str, short_col: str) -> pd.DataFrame | None:
    """Socrata rows → weekly frame with net/oi, deduped, measurement-sorted."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["measurement_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce").dt.normalize()
    df["long"] = pd.to_numeric(df[long_col], errors="coerce")
    df["short"] = pd.to_numeric(df[short_col], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
    df = df.dropna(subset=["measurement_date", "long", "short", "open_interest"])
    df = df.sort_values("measurement_date").drop_duplicates("measurement_date", keep="last").reset_index(drop=True)
    df["net"] = df["long"] - df["short"]
    df["net_oi"] = df["net"] / df["open_interest"].replace(0, np.nan)
    return df


def _trailing_pct(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).apply(lambda x: float((x <= x[-1]).mean()), raw=True)


def _trailing_z(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)


def _cross_frame(base: pd.DataFrame, quote: pd.DataFrame) -> pd.DataFrame | None:
    """Cross positioning = leg difference, aligned on measurement_date (inner join)."""
    if base is None or quote is None:
        return None
    m = base.merge(quote, on="measurement_date", suffixes=("_b", "_q"))
    if m.empty:
        return None
    out = pd.DataFrame({
        "measurement_date": m["measurement_date"],
        "long": 0, "short": 0,
        "net": m["net_b"] - m["net_q"],
        "net_oi": m["net_oi_b"] - m["net_oi_q"],
        "open_interest": 0,  # not meaningful for a leg difference — net_oi is leg-normalized
    })
    return out


def _date_stamp(df: pd.DataFrame, lag: int, release_time_et: str) -> pd.DataFrame:
    df = df.copy()
    pub = df["measurement_date"] + pd.Timedelta(days=lag)
    df["publish_date"] = pub.dt.date
    hh, mm = (int(x) for x in release_time_et.split(":"))
    df["release_ts"] = pub + pd.Timedelta(hours=hh, minutes=mm)  # naive ET provenance timestamp
    df["measurement_date"] = df["measurement_date"].dt.date
    return df


def update(con=None, start: str | None = None) -> dict:
    """Fetch legacy + TFF COT, compute trailing features, Friday-publish-date, upsert. Returns coverage."""
    cfg = params["sentiment"]["cot"]
    start = start or cfg.get("start", "1986-01-01")
    lag = int(cfg.get("publish_lag_days", 3))
    rel_t = str(cfg.get("release_time_et", "15:30"))
    w3 = int(cfg.get("percentile_window_weeks", 156))
    w1 = int(cfg.get("percentile_window_weeks_1y", 52))
    zmin = int(cfg.get("zscore_min_periods", 26))
    fw = int(cfg.get("flush_window_weeks", 52))
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    coverage: dict = {}

    codes = dict(cfg["pairs"])                      # pair -> code
    leg_codes = dict(cfg.get("currencies", {}))     # extra legs (NZD)
    crosses = dict(cfg.get("crosses", {}))          # cross pair -> {base_code, quote_code}

    for dataset_key, table, long_col, short_col, prefix in (
        ("socrata_dataset", "sentiment_cot_weekly",
         "noncomm_positions_long_all", "noncomm_positions_short_all", "legacy"),
        ("tff_socrata_dataset", "sentiment_cot_tff_weekly",
         "lev_money_positions_long", "lev_money_positions_short", "tff"),
    ):
        dataset = cfg.get(dataset_key)
        if not dataset:
            continue
        select = _LEGACY_SELECT if prefix == "legacy" else _TFF_SELECT
        by_code: dict[str, pd.DataFrame | None] = {}
        for code in set(list(codes.values()) + list(leg_codes.values())):
            by_code[code] = _frame(fetch_currency(code, dataset, start, select), long_col, short_col)

        frames = []
        series = {pair: by_code.get(code) for pair, code in codes.items()}
        for xpair, legs in crosses.items():
            series[xpair] = _cross_frame(by_code.get(str(legs["base_code"])),
                                         by_code.get(str(legs["quote_code"])))
        for pair, df in series.items():
            key = f"{prefix}:{pair}"
            if df is None or df.empty:
                coverage[key] = {"rows": 0}
                continue
            df = df.copy()
            df["pct_3y"] = _trailing_pct(df["net"], w3, 52)
            df["pct_1y"] = _trailing_pct(df["net"], w1, 26)
            if prefix == "legacy":
                df["net_z_1y"] = _trailing_z(df["net"], w1, zmin)
                df["net_z_3y"] = _trailing_z(df["net"], w3, zmin)
                dnet = df["net"].diff()
                df["flush_1w"] = dnet / dnet.rolling(fw, min_periods=zmin).std(ddof=0).replace(0, np.nan)
            df = _date_stamp(df, lag, rel_t)
            df["pair"] = pair
            df["fetched_at"] = now
            for c in ("long", "short", "net", "open_interest"):
                df[c] = df[c].astype("int64")
            if prefix == "legacy":
                out = df.rename(columns={"long": "noncomm_long", "short": "noncomm_short",
                                         "net": "net_spec", "pct_3y": "net_pct", "pct_1y": "net_pct_1y"})[
                    ["measurement_date", "publish_date", "pair", "noncomm_long", "noncomm_short",
                     "net_spec", "net_oi", "net_pct", "net_pct_1y", "net_z_1y", "net_z_3y",
                     "flush_1w", "release_ts", "open_interest", "fetched_at"]]
            else:
                out = df.rename(columns={"long": "lev_long", "short": "lev_short", "net": "lev_net",
                                         "net_oi": "lev_net_oi", "pct_3y": "lev_net_pct",
                                         "pct_1y": "lev_net_pct_1y"})[
                    ["measurement_date", "publish_date", "pair", "lev_long", "lev_short", "lev_net",
                     "lev_net_oi", "lev_net_pct", "lev_net_pct_1y", "release_ts",
                     "open_interest", "fetched_at"]]
            frames.append((table, out))
            coverage[key] = {"rows": int(len(out)), "start": str(out["measurement_date"].min()),
                             "end": str(out["measurement_date"].max())}
        for table_name, out in frames:
            upsert(con, table_name, out, ["measurement_date", "pair"])
        frames.clear()
    if own:
        con.close()
    return coverage
