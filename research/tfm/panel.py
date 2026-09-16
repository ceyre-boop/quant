"""HYP-120 data: (1) daily log-return panel — ten ETFs (daily_universe) + nine G10 vs USD (FRED DEX*);
(2) the HYP-119 liquid cascade events with a 512-minute causal context rebuilt from the minute cache,
plus one matched non-state minute per event (seed 42). No forecast is computed here."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.carry_model.data_fred import fred, SPOT
from research.cascade.state import session_frame

ROOT = Path(__file__).resolve().parents[2]
ETFS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"]
G10 = ["EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "SEK", "NOK"]
CTX, H_DAILY, H_MIN = 512, 5, 30
MINUTE_CACHE = ROOT / "data" / "cache" / "minute_bars"


def daily_returns() -> pd.DataFrame:
    cols = {}
    for s in ETFS:
        d = pd.read_parquet(ROOT / "data" / "cache" / "daily_universe" / f"{s}.parquet"); d["date"] = pd.to_datetime(d["date"])
        cols[s] = np.log(d.set_index("date")["close"].astype(float)).diff()
    for c in G10:
        sid, inv = SPOT[c]; s = fred(sid); s = (1 / s) if inv else s
        cols[c] = np.log(s).diff()
    R = pd.DataFrame(cols).sort_index()
    return R[R.index >= "2014-01-01"]


def daily_windows(R: pd.DataFrame) -> list[dict]:
    """One window per (series, t) with t ≥ CTX and t+H ≤ n: context r[t−CTX..t−1], targets r[t..t+H−1]."""
    out = []
    for s in R.columns:
        r = R[s].dropna(); v = r.values.astype(np.float32); idx = r.index
        for t in range(CTX, len(v) - H_DAILY + 1):
            fwd = v[t: t + H_DAILY]
            out.append({"series": s, "date": idx[t], "ctx": v[t - CTX: t], "r_next": float(fwd[0]), "rv_next": float(fwd.std(ddof=1)),
                        "rv21": float(v[t - 21: t].std(ddof=1)), "ewma": float(np.sqrt(pd.Series(v[:t] ** 2).ewm(alpha=0.06).mean().iloc[-1]))})
    return out


def _session_bars(sym: str, dates: list[str]) -> dict[str, pd.DataFrame]:
    return {d: session_frame(pd.read_parquet(MINUTE_CACHE / f"{sym}_{d}.parquet")) for d in dates if (MINUTE_CACHE / f"{sym}_{d}.parquet").exists()}


def cascade_windows(events: pd.DataFrame, seed: int = 42) -> list[dict]:
    """For each liquid HYP-119 event and one matched non-state minute in the same session: 512-minute
    causal log-return context (previous sessions + current up to m−1) and the realized 30-min log range."""
    rng = np.random.default_rng(seed); out = []
    files = sorted(MINUTE_CACHE.glob("*.parquet"))
    by_sym: dict[str, list[str]] = {}
    for f in files:
        s, d = f.stem.rsplit("_", 1); by_sym.setdefault(s, []).append(d)
    for sym, g in events.groupby("sym"):
        dates = sorted(by_sym.get(sym, [])); cache: dict[str, pd.DataFrame] = {}
        for _, ev in g.iterrows():
            d = ev["date"]; i = dates.index(d) if d in dates else -1
            if i < 2: continue
            need = dates[max(0, i - 3): i + 1]
            for dd in need:
                if dd not in cache:
                    cache[dd] = session_frame(pd.read_parquet(MINUTE_CACHE / f"{sym}_{dd}.parquet"))
            cur = cache[d]; m = int(ev["m"])
            hist = pd.concat([cache[dd] for dd in need[:-1]] + [cur.iloc[:m]])
            lr = np.log(hist["close"].astype(float)).diff().dropna().values.astype(np.float32)
            if len(lr) < CTX or m + 1 + H_MIN > len(cur): continue
            fwd = cur.iloc[m + 1: m + 1 + H_MIN]; rng_real = float(np.log(fwd["high"].max() / fwd["low"].min()))
            out.append({"kind": "state", "sym": sym, "date": d, "m": m, "z": float(ev["z"]), "ctx": lr[-CTX:], "range": rng_real})
            # matched non-state minute: random m' in the session with enough history and horizon, not within 30 of the event
            cands = [k for k in range(CTX // 4, len(cur) - H_MIN - 1) if abs(k - m) > H_MIN]
            if not cands: continue
            k = int(rng.choice(cands)); hist2 = pd.concat([cache[dd] for dd in need[:-1]] + [cur.iloc[:k]])
            lr2 = np.log(hist2["close"].astype(float)).diff().dropna().values.astype(np.float32)
            fwd2 = cur.iloc[k + 1: k + 1 + H_MIN]
            if len(lr2) >= CTX:
                out.append({"kind": "nonstate", "sym": sym, "date": d, "m": k, "z": np.nan, "ctx": lr2[-CTX:], "range": float(np.log(fwd2["high"].max() / fwd2["low"].min()))})
    return out
