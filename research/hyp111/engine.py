"""Pure functions for HYP-111a (intraday retrace-then-continuation) and HYP-112 (post-shock
straddle). No I/O except `daily()`. Every rule here is the one frozen in research/HYP-111_SCOPE.md
and the two prereg JSONs; nothing is parameterised beyond what those documents freeze."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "data" / "cache" / "daily_universe"
INSTRUMENTS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"]


# ── daily layer (identical shock rule to HYP-109) ────────────────────────────

def daily(sym: str, window=("2014-01-02", "2026-07-16")) -> pd.DataFrame:
    df = pd.read_parquet(DAILY / f"{sym}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= window[0]) & (df["date"] <= window[1])].sort_values("date").set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["r"] = np.log(df["close"]).diff()
    a = df["r"].abs()
    thr = a.shift(1).rolling(252).quantile(0.90)            # t−252..t−1, t excluded
    df["shock"] = (a >= thr) & thr.notna()
    df["vol_med20"] = df["volume"].shift(1).rolling(20).median()
    return df


def events(df: pd.DataFrame, first_next: str, last_next: str, exclude_dates=()) -> list[dict]:
    """Shock events whose session t+1 falls in [first_next, last_next]. Levels fixed from day t."""
    idx = df.index
    out = []
    for i in np.flatnonzero(df["shock"].values):
        if i + 1 >= len(idx):
            continue
        t, t1 = idx[i], idx[i + 1]
        if not (pd.Timestamp(first_next) <= t1 <= pd.Timestamp(last_next)):
            continue
        if t1.strftime("%Y-%m-%d") in exclude_dates:
            continue
        row = df.iloc[i]
        s = 1.0 if row["r"] > 0 else -1.0
        rng = float(row["high"] - row["low"])
        C = float(row["close"])
        out.append({
            "t": t.strftime("%Y-%m-%d"), "t1": t1.strftime("%Y-%m-%d"), "i": int(i), "s": s,
            "C": C, "range": rng, "L": C - s * 0.382 * rng, "T": C + s * 0.382 * rng,
            "c1_volume": bool(row["volume"] >= 1.5 * row["vol_med20"]) if row["vol_med20"] > 0 else False,
            "c5_strong_close": bool(((row["close"] - row["low"]) / rng >= 0.75) if s > 0
                                    else ((row["high"] - row["close"]) / rng >= 0.75)) if rng > 0 else False,
        })
    return out


# ── intraday layer (HYP-111a) ────────────────────────────────────────────────

@dataclass
class Trade:
    triggered: bool
    tau1: str | None = None
    tau2: str | None = None
    entry: float | None = None
    exit: float | None = None
    exit_kind: str | None = None          # stop | target | time
    ret_gross: float = 0.0                # s*(exit-entry)/entry, 0 if not triggered
    R: float = 0.0


def simulate(bars: pd.DataFrame, s: float, C: float, L: float, T: float) -> Trade:
    """Retrace to L, reclaim C by 14:30, enter next bar open, 1R bracket, time exit 15:55 close.
    Only bars up to and including tau2 influence the entry decision (look-ahead canary tested)."""
    if bars.empty:
        return Trade(False)
    time = bars["time"].values
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))
    n = len(bars)
    against = (l <= L) if s > 0 else (h >= L)
    hit = np.flatnonzero(against)
    if len(hit) == 0:
        return Trade(False)
    tau1 = int(hit[0])
    reclaim = (c > C) if s > 0 else (c < C)
    cand = np.flatnonzero(reclaim[tau1 + 1:]) + tau1 + 1
    cand = [j for j in cand if time[j] <= "14:30"]
    if not cand or cand[0] + 1 >= n:
        return Trade(False, tau1=time[tau1])
    tau2 = int(cand[0])
    e = tau2 + 1
    entry = float(o[e])
    exit_px, kind = None, None
    for j in range(e, n):
        stop_hit = (l[j] <= L) if s > 0 else (h[j] >= L)
        tgt_hit = (h[j] >= T) if s > 0 else (l[j] <= T)
        if stop_hit:                      # both in one bar → stop (frozen, conservative)
            exit_px, kind = L, "stop"; break
        if tgt_hit:
            exit_px, kind = T, "target"; break
        if time[j] >= "15:55":
            exit_px, kind = float(c[j]), "time"; break
    if exit_px is None:                   # session ended before 15:55 bar (short day)
        exit_px, kind = float(c[n - 1]), "time"
    ret = s * (exit_px - entry) / entry
    return Trade(True, time[tau1], time[tau2], entry, exit_px, kind, float(ret),
                 float(ret / (abs(C - L) / C)) if C != L else 0.0)


def naive(bars: pd.DataFrame, s: float) -> float:
    """Incumbent: direction s from the 09:30 open to the 15:55 close (or last bar)."""
    if bars.empty:
        return 0.0
    o = float(bars["open"].iloc[0])
    late = bars[bars["time"] >= "15:55"]
    c = float(late["close"].iloc[0]) if len(late) else float(bars["close"].iloc[-1])
    return s * (c - o) / o


def confluence(bars: pd.DataFrame, proxy: pd.DataFrame, tr: Trade, s: float, C: float,
               c1: bool, c5: bool) -> dict:
    """Five frozen conditions, all knowable at tau2. Returns {c1..c5, count}."""
    if not tr.triggered:
        return {}
    j = int(np.flatnonzero(bars["time"].values == tr.tau2)[0])
    open_ = float(bars["open"].iloc[0])
    c2 = (open_ > C) if s > 0 else (open_ < C)
    sub = bars.iloc[: j + 1]
    tp = (sub["high"] + sub["low"] + sub["close"]) / 3
    vwap = float((tp * sub["volume"]).sum() / max(sub["volume"].sum(), 1e-9))
    close_j = float(sub["close"].iloc[-1])
    c3 = (close_j > vwap) if s > 0 else (close_j < vwap)
    c4 = False
    if proxy is not None and not proxy.empty:
        pj = proxy[proxy["time"] <= tr.tau2]
        if len(pj):
            pr = float(pj["close"].iloc[-1]) - float(proxy["open"].iloc[0])
            c4 = (pr > 0) if s > 0 else (pr < 0)
    d = {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "c4": bool(c4), "c5": bool(c5)}
    d["count"] = int(sum(d.values()))
    return d


# ── options layer (HYP-112) ──────────────────────────────────────────────────

def pick_expiration(exps: list[str], t: str, min_days: int = 7) -> str | None:
    """Nearest listed expiration ≥ t + min_days calendar days."""
    t0 = pd.Timestamp(t)
    for e in exps:
        if pd.Timestamp(e) >= t0 + pd.Timedelta(days=min_days):
            return e
    return None


def straddle(chain: pd.DataFrame, date_entry: str, date_exit: str, spot: float,
             commission_per_contract: float = 0.65) -> dict | None:
    """ATM straddle: strike nearest spot on date_entry. Buy call+put at ASK on entry, sell at BID
    on exit. Returns dict with premium_in, premium_out, ret_on_premium, ret_on_spot; None if
    either side unquoted (bid or ask == 0) on either date."""
    ce = chain[chain["date"] == date_entry]
    cx = chain[chain["date"] == date_exit]
    if ce.empty or cx.empty:
        return None
    strikes = np.sort(ce["strike"].unique())
    k = float(strikes[np.argmin(np.abs(strikes - spot))])
    def q(df, right):
        r = df[(df["strike"] == k) & (df["right"] == right)]
        return None if r.empty else (float(r["bid"].iloc[0]), float(r["ask"].iloc[0]))
    ce_c, ce_p, cx_c, cx_p = q(ce, "C"), q(ce, "P"), q(cx, "C"), q(cx, "P")
    if any(v is None for v in (ce_c, ce_p, cx_c, cx_p)):
        return None
    if min(ce_c[1], ce_p[1]) <= 0 or ce_c[0] <= 0 or ce_p[0] <= 0:
        return None                                      # unquoted / one-sided market at entry
    prem_in = ce_c[1] + ce_p[1]                          # pay the ask
    prem_out = max(cx_c[0], 0.0) + max(cx_p[0], 0.0)     # hit the bid (0 if no bid)
    comm = 4 * commission_per_contract / 100.0           # 2 legs × 2 sides, per share
    implied_move = (ce_c[1] + ce_p[1]) / 2 / spot        # mid-ish straddle / spot, descriptive
    return {"strike": k, "premium_in": prem_in, "premium_out": prem_out,
            "ret_on_premium": (prem_out - prem_in - comm) / prem_in,
            "ret_on_spot": (prem_out - prem_in - comm) / spot,
            "implied_move": implied_move}


def sharpe_dates(x: np.ndarray, dates_per_year: float) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(dates_per_year)) if s > 0 and len(x) > 2 else 0.0


def trade_dict(tr: Trade) -> dict:
    return asdict(tr)
