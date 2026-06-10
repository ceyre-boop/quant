"""ES/NQ structure confirmation gate — AMD sweep + VWAP-reclaim entry trigger.

Pure functions over 5-min RTH bars. NOT a standalone edge (ICT entry timing failed
permutation p=0.52 on forex) — a confirmation LAYER on top of a validated bias.

Bias UP:  watch PDL/ONL. Sweep = a bar's Low below level×(1−0.1%) followed by a
          Close back above the level within 3 bars. Confirmation = price touches
          session VWAP then one bar closes UP with Volume > 1.2× rolling-20 avg.
Bias DOWN: mirrored on PDH/ONH.

Entry next bar Open (+0.25 tick slip); stop = sweep extreme ∓ 2 ticks; T1/T2 =
1.5×/2.5× stop distance; half off at T1 → stop to breakeven; flat 15:55; no new
entries after 12:00 ET. All thresholds from config/es_nq_params.yml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from sovereign.es_nq.config import contract_spec, es_nq_params


@dataclass(frozen=True)
class Levels:
    pdh: float
    pdl: float
    onh: float
    onl: float


@dataclass(frozen=True)
class Sweep:
    level_name: str          # 'PDL' | 'ONL' | 'PDH' | 'ONH'
    level: float
    extreme: float           # min Low (max High) across sweep..reclaim bars
    sweep_bar_idx: int
    reclaim_bar_idx: int
    side: str                # 'low' | 'high'


@dataclass(frozen=True)
class TradePlan:
    direction: str           # LONG | SHORT
    entry: float             # includes entry slippage
    stop: float
    t1: float
    t2: float
    stop_points: float
    sweep: Sweep
    confirm_bar_idx: int
    entry_bar_idx: int


def session_levels(prior_daily_row: pd.Series, session_daily_row: pd.Series) -> Levels:
    """PDH/PDL from the prior session's RTH; ONH/ONL from this session's overnight."""
    return Levels(pdh=float(prior_daily_row["rth_high"]),
                  pdl=float(prior_daily_row["rth_low"]),
                  onh=float(session_daily_row["onh"]),
                  onl=float(session_daily_row["onl"]))


def session_vwap(bars5: pd.DataFrame) -> pd.Series:
    """Cumulative typical-price VWAP — identical math to
    sovereign/futures/scalp_strategy.compute_indicators."""
    typical = (bars5["High"] + bars5["Low"] + bars5["Close"]) / 3.0
    return (typical * bars5["Volume"]).cumsum() / bars5["Volume"].cumsum()


def _et_minutes(idx) -> "pd.Index":
    et = idx.tz_convert("America/New_York")
    return et.hour * 60 + et.minute


def detect_sweep(bars5: pd.DataFrame, levels: Levels, bias_dir: str,
                 params: Optional[dict] = None) -> Optional[Sweep]:
    """First completed sweep+reclaim consistent with the bias, else None.

    Checks both candidate levels for the bias side (UP → PDL and ONL; DOWN → PDH
    and ONH) and returns the earliest-reclaimed sweep.
    """
    p = (params or es_nq_params())["structure"]
    pct, max_reclaim = p["sweep_min_pct"], p["sweep_reclaim_bars"]
    if bias_dir == "UP":
        candidates = [("PDL", levels.pdl, "low"), ("ONL", levels.onl, "low")]
    elif bias_dir == "DOWN":
        candidates = [("PDH", levels.pdh, "high"), ("ONH", levels.onh, "high")]
    else:
        return None

    best: Optional[Sweep] = None
    for name, level, side in candidates:
        if level is None or not pd.notna(level):
            continue
        thresh = level * (1 - pct) if side == "low" else level * (1 + pct)
        for i in range(len(bars5)):
            bar = bars5.iloc[i]
            swept = bar["Low"] < thresh if side == "low" else bar["High"] > thresh
            if not swept:
                continue
            for j in range(i + 1, min(i + 1 + max_reclaim, len(bars5))):
                close = bars5["Close"].iloc[j]
                reclaimed = close > level if side == "low" else close < level
                if not reclaimed:
                    continue
                window = bars5.iloc[i:j + 1]
                extreme = float(window["Low"].min()) if side == "low" \
                    else float(window["High"].max())
                cand = Sweep(level_name=name, level=float(level), extreme=extreme,
                             sweep_bar_idx=i, reclaim_bar_idx=j, side=side)
                if best is None or cand.reclaim_bar_idx < best.reclaim_bar_idx:
                    best = cand
                break
            break  # only the FIRST sweep of each level counts
    return best


def detect_confirmation(bars5: pd.DataFrame, sweep: Sweep, bias_dir: str,
                        params: Optional[dict] = None) -> Optional[int]:
    """Index of the confirmation bar, else None.

    After the reclaim: (a) a bar whose Low..High range touches session VWAP, then
    (b) at/after the touch, one bar closing in the bias direction with Volume >
    confirm_volume_mult × rolling-20-bar average. Both must complete strictly
    before entry_deadline_et (entry happens on the NEXT bar's Open).
    """
    p = (params or es_nq_params())["structure"]
    deadline = _hhmm_minutes(p["entry_deadline_et"])
    vwap = session_vwap(bars5)
    vol_avg = bars5["Volume"].rolling(p["confirm_volume_lookback"], min_periods=1).mean()
    minutes = _et_minutes(bars5.index)

    touched = False
    for i in range(sweep.reclaim_bar_idx + 1, len(bars5)):
        if minutes[i] >= deadline:
            return None
        bar = bars5.iloc[i]
        v = vwap.iloc[i]
        if not touched:
            touched = bar["Low"] <= v <= bar["High"]
            if not touched:
                continue
        directional = (bar["Close"] > bar["Open"]) if bias_dir == "UP" \
            else (bar["Close"] < bar["Open"])
        volume_ok = bar["Volume"] > p["confirm_volume_mult"] * vol_avg.iloc[i]
        if directional and volume_ok:
            return i
    return None


def plan_trade(bars5: pd.DataFrame, confirm_idx: int, sweep: Sweep, bias_dir: str,
               instrument: str, params: Optional[dict] = None) -> Optional[TradePlan]:
    """Entry on the bar AFTER confirmation. None if no next bar before the deadline."""
    p = (params or es_nq_params())
    s = p["structure"]
    spec = contract_spec(instrument)
    tick = spec["tick"]
    entry_idx = confirm_idx + 1
    if entry_idx >= len(bars5):
        return None
    minutes = _et_minutes(bars5.index)
    if minutes[entry_idx] >= _hhmm_minutes(s["entry_deadline_et"]):
        return None

    raw_entry = float(bars5["Open"].iloc[entry_idx])
    slip = p["costs"]["slippage_ticks_entry"] * tick
    buffer = s["stop_buffer_ticks"] * tick
    if bias_dir == "UP":
        entry = raw_entry + slip
        stop = sweep.extreme - buffer
        stop_pts = entry - stop
        if stop_pts <= 0:
            return None
        t1, t2 = entry + s["t1_r"] * stop_pts, entry + s["t2_r"] * stop_pts
        direction = "LONG"
    else:
        entry = raw_entry - slip
        stop = sweep.extreme + buffer
        stop_pts = stop - entry
        if stop_pts <= 0:
            return None
        t1, t2 = entry - s["t1_r"] * stop_pts, entry - s["t2_r"] * stop_pts
        direction = "SHORT"
    return TradePlan(direction=direction, entry=round(entry, 4), stop=round(stop, 4),
                     t1=round(t1, 4), t2=round(t2, 4), stop_points=round(stop_pts, 4),
                     sweep=sweep, confirm_bar_idx=confirm_idx, entry_bar_idx=entry_idx)


def _hhmm_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)
