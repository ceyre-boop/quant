"""sovereign/sentiment/surprise_feed.py — economic "release innovation" spine (FREE, honest proxy).

There is no FREE source of historical *consensus* forecasts, so this is NOT a consensus surprise (Citi/
Bloomberg style). It is a RELEASE INNOVATION: the FIRST-PRINT actual (FRED/ALFRED get_series_all_releases,
keyed on the PUBLISH date → no revision look-ahead) minus a naive baseline (prior first print), z-scored
over a trailing window. A real but weaker signal — labeled `release_innovation` everywhere so it is never
mistaken for a market-consensus surprise.

Per release it writes sentiment_surprise_release (audit). It then projects the standardized innovations
onto the trading-day calendar as a single EWMA-decayed `econ_surprise_z` (US releases → USD; broadcast
across pairs, the model learns each pair's sign). Foreign-CB surprises are a future per-pair extension.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, env_key, upsert


def first_prints(series_id: str, start: str) -> pd.DataFrame:
    """ALFRED first-print actuals for one series: [ref_date, publish_date, first_print], publish >= start.

    The first print = the row with the earliest realtime_start per reference date (what the market first
    saw). Empty DataFrame on failure.
    """
    try:
        from fredapi import Fred
        df = Fred(api_key=env_key("FRED_API_KEY")).get_series_all_releases(series_id)
    except Exception as exc:
        print(f"  [surprise] {series_id}: FETCH FAILED ({type(exc).__name__}: {exc})")
        return pd.DataFrame(columns=["ref_date", "publish_date", "first_print"])
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ref_date", "publish_date", "first_print"])
    df = df.copy()
    df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["realtime_start", "date", "value"])
    fp = df.sort_values("realtime_start").groupby("date", as_index=False).first()
    fp = fp.rename(columns={"date": "ref_date", "realtime_start": "publish_date", "value": "first_print"})
    fp = fp[fp["publish_date"] >= pd.Timestamp(start)].sort_values("ref_date").reset_index(drop=True)
    return fp[["ref_date", "publish_date", "first_print"]]


def compute_surprise(fp: pd.DataFrame, baseline: str, zscore_window: int) -> pd.DataFrame:
    """Release innovation per row: surprise = first_print − baseline; standardized over trailing window."""
    if fp.empty:
        return fp.assign(baseline=[], surprise=[], surprise_z=[])
    out = fp.copy()
    out["baseline"] = out["first_print"].shift(1)        # 'prior' = naive random-walk expectation
    out["surprise"] = out["first_print"] - out["baseline"]
    mean = out["surprise"].rolling(zscore_window, min_periods=max(8, zscore_window // 3)).mean()
    std = out["surprise"].rolling(zscore_window, min_periods=max(8, zscore_window // 3)).std()
    out["surprise_z"] = (out["surprise"] - mean) / std.replace(0.0, np.nan)
    return out


def _decay_accumulate(calendar: list, pulse_by_date: dict, halflife_days: float) -> np.ndarray:
    """Exponentially-decayed running sum of daily surprise pulses (a 'surprise climate')."""
    decay = 0.5 ** (1.0 / max(halflife_days, 1e-6))
    s = np.zeros(len(calendar))
    acc = 0.0
    for i, d in enumerate(calendar):
        acc = pulse_by_date.get(d, 0.0) + decay * acc
        s[i] = acc
    return s


def update(con=None, start: str | None = None) -> dict:
    """Build the release-innovation table + the daily econ_surprise_z climate. Returns coverage."""
    cfg = params["sentiment"]["surprise"]
    start = start or cfg.get("start", "2015-01-01")
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    frames, coverage = [], {}
    for sid in cfg["releases"]:
        fp = first_prints(sid, start)
        if fp.empty:
            coverage[sid] = {"releases": 0}
            continue
        s = compute_surprise(fp, cfg.get("baseline", "prior"), int(cfg.get("zscore_window", 36)))
        rel = pd.DataFrame({
            "publish_date": pd.to_datetime(s["publish_date"]).dt.date,
            "series": sid,
            "ref_date": pd.to_datetime(s["ref_date"]).dt.date,
            "first_print": s["first_print"].astype(float),
            "baseline": s["baseline"].astype(float),
            "surprise": s["surprise"].astype(float),
            "surprise_z": s["surprise_z"].astype(float),
            "fetched_at": now,
        }).dropna(subset=["surprise_z"])
        rel = rel.drop_duplicates(subset=["publish_date", "series"], keep="last")
        frames.append(rel)
        coverage[sid] = {"releases": int(len(rel)),
                         "start": str(rel["publish_date"].min()) if len(rel) else None,
                         "end": str(rel["publish_date"].max()) if len(rel) else None}
    if frames:
        rel_all = pd.concat(frames, ignore_index=True)
        upsert(con, "sentiment_surprise_release", rel_all, ["publish_date", "series"])

        # daily econ_surprise_z on the trading-day (VIX) calendar — exact alignment with the board spine
        cal = [r[0] for r in con.execute(
            "SELECT DISTINCT date FROM sentiment_vix_daily WHERE date >= ? ORDER BY date", [str(start)]).fetchall()]
        if not cal:                                       # tests / no VIX yet → business days
            cal = [d.date() for d in pd.bdate_range(start, now.date())]
        pulse = rel_all.groupby("publish_date")["surprise_z"].sum().to_dict()
        z = _decay_accumulate(cal, pulse, float(cfg.get("ewma_halflife_days", 5)))
        daily = pd.DataFrame({"date": cal, "econ_surprise_z": z, "fetched_at": now})
        upsert(con, "sentiment_surprise_daily", daily, ["date"])
        coverage["_daily"] = {"rows": int(len(daily)), "start": str(cal[0]), "end": str(cal[-1])}
    if own:
        con.close()
    return coverage
