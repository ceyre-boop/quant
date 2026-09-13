"""Builds the STATE events for both universes. Liquid: 23 continuous names in data/cache/minute_bars,
baseline = same name's prior 60 sessions (same-clock volume median for RVOL; cascade_raw distribution
for z). Microcap: the single-event-day names in the same cache, baseline = pooled cascade_raw of all
prior event-days (expanding, causal). Every event carries the bars needed for claims (a)-(c)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.cascade.state import raw_ratio, session_frame, W, Z_THR, HORIZON

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "minute_bars"
LIQUID = ["AAPL", "AI", "AMD", "AMZN", "BBAI", "BYND", "COIN", "FCEL", "GLD", "IWM", "META", "MSFT", "MSTR", "NVDA", "PLUG", "QBTS", "QQQ", "RGTI", "SMCI", "SPCE", "SPY", "TLT", "TSLA"]
BASE_SESSIONS = 60


def _files():
    by = defaultdict(list)
    for f in sorted(CACHE.glob("*.parquet")):
        sym, date = f.stem.rsplit("_", 1); by[sym].append((date, f))
    return by


def _events_from(b: pd.DataFrame, z: pd.Series, sym: str, date: str, universe: str) -> list[dict]:
    out = []; n = len(b); last = None
    st = (z > Z_THR).values
    for m in np.flatnonzero(st):
        if m + 1 >= n or m + 1 + HORIZON > n or m < W + 1:
            continue
        if last is not None and m - last < HORIZON:          # one event per 30-minute window
            continue
        last = m
        r = raw_ratio(b).iloc[m]
        out.append({"universe": universe, "sym": sym, "date": date, "m": int(m), "time": b["time"].iloc[m], "z": float(z.iloc[m]),
                    "direction": float(np.sign(r["ret5"])) if r["ret5"] != 0 else 0.0, "ret5": float(r["ret5"]),
                    "entry_open": float(b["open"].iloc[m + 1]), "prior_closes": b["close"].iloc[max(0, m - 5): m].astype(float).tolist(),
                    "bars_fwd": b.iloc[m + 1: m + 1 + HORIZON][["time", "open", "high", "low", "close", "volume"]].to_dict("list"),
                    "spread_proxy": float(r["spr"]), "minute_dollar_vol": float(r["dv5"] / W)})
    return out


def build() -> tuple[pd.DataFrame, dict]:
    files = _files(); events = []; nonstate_ranges = []
    # ── liquid ──
    for sym in LIQUID:
        hist_vol = []            # per session: same-clock volume arrays
        hist_raw = []            # per session: cascade_raw arrays
        for date, f in files.get(sym, []):
            b = session_frame(pd.read_parquet(f))
            if len(b) < 370: continue
            rr = raw_ratio(b)
            if len(hist_vol) >= BASE_SESSIONS:
                base_v = np.nanmedian(np.array([h[:len(b)] if len(h) >= len(b) else np.pad(h, (0, len(b) - len(h)), constant_values=np.nan) for h in hist_vol[-BASE_SESSIONS:]]), axis=0)
                rvol = rr["vol5"] / pd.Series(base_v).rolling(W).sum().shift(1).clip(lower=1)
                raw = (rvol * rr["ret5"]).abs() / (rr["dv5"] / rr["spr"]).clip(lower=1e-9)
                pool = np.concatenate([x[np.isfinite(x)] for x in hist_raw[-BASE_SESSIONS:]])
                z = (np.log(raw.clip(lower=1e-12)) - np.log(pool.clip(min=1e-12)).mean()) / max(np.log(pool.clip(min=1e-12)).std(), 1e-9)
                z = pd.Series(z, index=b.index)
                events += _events_from(b, z, sym, date, "liquid")
                st = (z > Z_THR).values
                fr = np.log(b["high"].rolling(HORIZON).max().shift(-HORIZON) / b["low"].rolling(HORIZON).min().shift(-HORIZON))
                nonstate_ranges.append({"universe": "liquid", "sym": sym, "date": date, "range": float(np.nanmedian(fr.values[~st])) if (~st).any() else np.nan})
            hist_vol.append(b["volume"].astype(float).values)
            rv0 = rr["vol5"] / rr["vol5"].rolling(60).median().shift(1).clip(lower=1)
            hist_raw.append(((rv0 * rr["ret5"]).abs() / (rr["dv5"] / rr["spr"]).clip(lower=1e-9)).values)
    # ── microcap: single-event-day names, pooled prior-day baseline ──
    micro = sorted([(d, s, f) for s, lst in files.items() if s not in LIQUID for d, f in lst])
    pool = np.array([])
    for date, sym, f in micro:
        b = session_frame(pd.read_parquet(f))
        if len(b) < 200: continue
        rr = raw_ratio(b); rv0 = rr["vol5"] / rr["vol5"].expanding().median().shift(1).clip(lower=1)   # within-day expanding baseline (causal)
        raw = ((rv0 * rr["ret5"]).abs() / (rr["dv5"] / rr["spr"]).clip(lower=1e-9))
        if len(pool) >= 5000:
            lp = np.log(pool.clip(min=1e-12)); z = pd.Series((np.log(raw.clip(lower=1e-12)) - lp.mean()) / max(lp.std(), 1e-9), index=b.index)
            events += _events_from(b, z, sym, date, "microcap")
            st = (z > Z_THR).values
            fr = np.log(b["high"].rolling(HORIZON).max().shift(-HORIZON) / b["low"].rolling(HORIZON).min().shift(-HORIZON))
            nonstate_ranges.append({"universe": "microcap", "sym": sym, "date": date, "range": float(np.nanmedian(fr.values[~st])) if (~st).any() else np.nan})
        pool = np.concatenate([pool, raw.values[np.isfinite(raw.values)]])
    return pd.DataFrame(events), {"nonstate": pd.DataFrame(nonstate_ranges), "n_liquid_files": sum(len(files.get(s, [])) for s in LIQUID), "n_micro_files": len(micro)}
