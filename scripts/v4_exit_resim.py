"""
V4 EXIT RE-SIMULATION ENGINE
=============================
Real price-path re-simulation of the v015 4-pair carry entries under alternative
exit policies. Replaces the ESTIMATED substitution used in V3 (which assumed
trailing-stopped trades would have earned the mean time-exit return — a
selection-bias trap, since those trades were stopped precisely because price
moved against them).

Entries are FIXED (taken from the sealed v015 log). Only the exit policy varies.
This is meta-labeling in the AFML ch.3 sense: entry set frozen, exit replayed.

Policy parameters:
  k_stop     hard stop distance in ATR multiples (0 = no hard stop)
  k_trail    trailing stop distance in ATR multiples (0 = no trail)
  delay      bars before the trail activates (0 = immediate)
  max_hold   time exit cap in bars

Costs included:
  - round-trip spread+slippage (constant across arms; affects level not ranking)
  - daily carry accrual from real rate differentials (MATTERS across arms, since
    longer holds accrue more carry — omitting it biases against long holds)

IS  = 2015-2022  (policy selection happens here ONLY)
OOS = 2023-2024  (touched once, after the policy is locked)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPOT = REPO / "data/research/positioning_family/spot_cache"
DIFF = REPO / "data/research/modern/spot_cache"
TRADES = REPO / "data/proof/backtest_trades_v015_2015_2024.csv"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
ATR_N = 14
ROUND_TRIP_COST_PCT = 0.0002   # 2bp round trip, spread+slippage
IS_END = 2022                  # IS = entries with year <= IS_END


# ─── DATA ────────────────────────────────────────────────────────────────────

def load_prices():
    px = {}
    for p in PAIRS:
        d = pd.read_parquet(SPOT / f"{p}_ohlc.parquet")
        d = d[["Open", "High", "Low", "Close"]].copy()
        d.columns = ["open", "high", "low", "close"]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="first")].sort_index()
        # ATR(14), Wilder
        pc = d["close"].shift(1)
        tr = pd.concat([d["high"] - d["low"],
                        (d["high"] - pc).abs(),
                        (d["low"] - pc).abs()], axis=1).max(axis=1)
        d["atr"] = tr.ewm(alpha=1 / ATR_N, adjust=False).mean()
        px[p] = d
    return px


def load_carry():
    """Daily carry in decimal per day: rate_differential / 365."""
    carry = {}
    for p in PAIRS:
        f = DIFF / f"{p}_differentials.parquet"
        if not f.exists():
            carry[p] = None
            continue
        d = pd.read_parquet(f)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="first")].sort_index()
        col = "rate_differential" if "rate_differential" in d.columns else None
        carry[p] = (d[col] / 100.0 / 365.0) if col else None
    return carry


def load_trades():
    t = pd.read_csv(TRADES)
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
    t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.normalize()
    t["year"] = t["entry_date"].dt.year
    t["pair_s"] = t["pair"].str.replace("=X", "", regex=False)
    t["recorded_r"] = t["pnl_pct"] / t["risk_pct"]
    return t


# ─── SIMULATION ──────────────────────────────────────────────────────────────

def simulate_trade(bars, direction, entry_px, atr0, carry_series,
                   k_stop, k_trail, delay, max_hold):
    """
    Walk bars forward from the bar AFTER entry. Returns (pnl_pct, hold_days, reason).
    Intrabar convention: stop/trail checked against the adverse extreme first
    (pessimistic — assumes the worst ordering within the bar).
    """
    if atr0 <= 0 or not np.isfinite(atr0):
        return None

    stop_px = None
    if k_stop > 0:
        stop_px = entry_px - direction * k_stop * atr0

    best = entry_px          # most favourable close reached
    trail_px = None
    exit_px, hold, reason = None, 0, "time"

    n = min(len(bars), max_hold)
    for i in range(n):
        hi = bars["high"].iat[i]
        lo = bars["low"].iat[i]
        cl = bars["close"].iat[i]
        hold = i + 1

        adverse = lo if direction == 1 else hi

        # hard stop
        if stop_px is not None:
            hit = adverse <= stop_px if direction == 1 else adverse >= stop_px
            if hit:
                exit_px, reason = stop_px, "stop"
                break

        # trailing stop (only after delay bars)
        if trail_px is not None:
            hit = adverse <= trail_px if direction == 1 else adverse >= trail_px
            if hit:
                exit_px, reason = trail_px, "trailing_stop"
                break

        # update favourable extreme and trail level on the close
        if direction == 1:
            best = max(best, cl)
        else:
            best = min(best, cl)

        if k_trail > 0 and hold >= delay:
            lvl = best - direction * k_trail * atr0
            if trail_px is None:
                trail_px = lvl
            else:
                trail_px = max(trail_px, lvl) if direction == 1 else min(trail_px, lvl)

    if exit_px is None:
        exit_px = bars["close"].iat[min(n, len(bars)) - 1]
        reason = "time"

    gross = direction * (exit_px - entry_px) / entry_px

    # carry accrual over the holding period
    carry_r = 0.0
    if carry_series is not None:
        seg = carry_series.reindex(bars.index[:hold]).ffill()
        if len(seg):
            carry_r = float(direction * seg.fillna(0).sum())

    net = gross + carry_r - ROUND_TRIP_COST_PCT
    return net, hold, reason


def run_policy(trades, px, carry, k_stop, k_trail, delay, max_hold):
    out = []
    for _, tr in trades.iterrows():
        p = tr["pair_s"]
        d = px[p]
        ed = tr["entry_date"]
        loc = d.index.searchsorted(ed)
        if loc >= len(d) - 2:
            continue
        atr0 = d["atr"].iat[loc]
        entry_px = float(tr["entry"])
        bars = d.iloc[loc + 1:loc + 1 + max_hold + 5]
        if len(bars) < 2:
            continue
        res = simulate_trade(bars, int(tr["direction"]), entry_px, atr0,
                             carry.get(p), k_stop, k_trail, delay, max_hold)
        if res is None:
            continue
        net, hold, reason = res
        out.append({
            "pair": p, "year": tr["year"], "entry_date": ed,
            "direction": tr["direction"], "risk_pct": tr["risk_pct"],
            "pnl_pct": net, "hold_days": hold, "exit_reason": reason,
            "r_mult": net / tr["risk_pct"],
            "recorded_r": tr["recorded_r"],
        })
    return pd.DataFrame(out)


# ─── METRICS ─────────────────────────────────────────────────────────────────

def stats(df, label=""):
    if df.empty:
        return {}
    r = df["r_mult"]
    ann = df.groupby("year")["r_mult"].sum()
    sharpe = r.mean() / r.std() * np.sqrt(len(r) / max(df["year"].nunique(), 1)) if r.std() > 0 else 0.0
    # equity path for drawdown, in R space
    eq = r.cumsum()
    dd = float((eq.cummax() - eq).max())
    return {
        "label": label, "n": len(r),
        "wr": float((r > 0).mean()),
        "mean_r": float(r.mean()),
        "sum_r": float(r.sum()),
        "sharpe_ann": float(sharpe),
        "max_dd_r": dd,
        "pos_years": int((ann > 0).sum()),
        "n_years": int(len(ann)),
    }


def fmt(s):
    return (f"n={s['n']:>4} WR={s['wr']:.1%} meanR={s['mean_r']:+.3f} "
            f"sumR={s['sum_r']:+7.1f} Sharpe={s['sharpe_ann']:+.2f} "
            f"maxDD={s['max_dd_r']:.1f}R yrs+={s['pos_years']}/{s['n_years']}")
