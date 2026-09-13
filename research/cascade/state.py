"""HYP-119 — the liquidity-cascade STATE, declared in the formal DSL and computed from bars ≤ m−1.

    K_m  = RVOL_m · ret5_m                    kinetic: relative volume × signed 5-minute return
    D_m  = Σ$vol(m−5..m−1) / spread_proxy_m    dampening: dollar volume per unit of range
    cascade_m = |K_m| / D_m ;  z-scored on a CAUSAL baseline (liquid: the same name's trailing 60 sessions;
    microcap: the pooled minutes of all PRIOR event-days);  STATE ⇔ z > 3 ;  direction = sign(ret5_m)

Frozen: 5-minute windows, 60-session / prior-day baselines, z > 3, 30-minute forward horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.formal.dsl import Rule, win, lag, fn

W, Z_THR, HORIZON = 5, 3.0, 30

RULE = Rule(
    "HYP-119 liquidity cascade",
    features={
        "rvol": fn("div", win("volume", -W, -1), win("volume_same_clock_prior_sessions", -60 * 390, -1)),
        "ret5": fn("logret", lag("close", 1), lag("close", W)),
        "dollar_vol": win("dollar_volume", -W, -1),
        "spread_proxy": win("range_over_close", -W, -1),
        "cascade_z": fn("z", fn("div", fn("abs", fn("mul", fn("div", win("volume", -W, -1), win("volume_same_clock_prior_sessions", -60 * 390, -1)),
                                                     fn("logret", lag("close", 1), lag("close", W)))),
                                          fn("div", win("dollar_volume", -W, -1), win("range_over_close", -W, -1))),
                        win("cascade_raw", -60 * 390, -1)),
    },
    universe=fn("ge", lag("session_count", 1), fn("const")),      # liquid: ≥60 prior sessions; microcap: ≥1 prior event-day
    decision_clock=None, entry_offset=1, hold_bars=HORIZON,
)


def raw_ratio(b: pd.DataFrame) -> pd.Series:
    """cascade_raw for every minute m of one session, using bars m−5..m−1 only (shifted)."""
    c, v, h, l = b["close"].astype(float), b["volume"].astype(float), b["high"].astype(float), b["low"].astype(float)
    ret5 = np.log(c.shift(1) / c.shift(W))
    vol5 = v.rolling(W).sum().shift(1)
    dv5 = (v * c).rolling(W).sum().shift(1)
    spr = ((h - l) / c).rolling(W).mean().shift(1).clip(lower=1e-5)
    return pd.DataFrame({"ret5": ret5, "vol5": vol5, "dv5": dv5, "spr": spr})


def session_frame(b: pd.DataFrame) -> pd.DataFrame:
    """RTH-only, zero-price bars dropped, minute index 0..n-1."""
    b = b[(b["time"] >= "09:30") & (b["time"] <= "15:59")]
    b = b[(b[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
    return b
