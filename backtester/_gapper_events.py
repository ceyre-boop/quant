"""Shared helper: load the 234 HYP-093 qualifying events as an events_df with a
minute-bar data_cache. Used by the corrected reruns and MC/scanner drivers."""
import csv
from pathlib import Path

import pandas as pd

from . import data as _data

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data/research/gapper/per_candidate_enriched.csv"
MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]


def load_events():
    rows = []
    for r in csv.DictReader(open(CSV)):
        try:
            g, p, v = (float(r["gain_1030"]), float(r["price_1030"]),
                       float(r["cum_vol_1030"]))
        except (ValueError, KeyError):
            continue
        if g < 1.0 or p < 2.0 or v < 500_000:
            continue
        if any(m in (r.get("catalyst") or "").lower() for m in MNA):
            continue
        rows.append({"date": r["date"], "ticker": r["ticker"], "gain": g})
    return pd.DataFrame(rows)


def build_cache(events_df):
    return {(e["ticker"], e["date"]): _data.get_minute_bars(e["ticker"], e["date"])
            for _, e in events_df.iterrows()}
