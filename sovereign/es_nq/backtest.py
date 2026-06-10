"""ES/NQ backtester — session-by-session replay over cached parquets. No network.

Modes:
  bias_only — Stage-1/2 baseline: bias ≠ NEUTRAL → enter at the 09:35 ET bar Open
              in bias direction, stop = 1.0 × daily ATR(14), same T1/T2 ratios.
  bias_gate — structure-gate trades (sweep + VWAP confirm), flat 0.5% risk.
  full      — the SAME structure-gate trade stream with the adaptive ladder.
              (bias_gate and full share trades by construction — Stage 3 compares
              sizing on identical trades.)

Fill conventions (pre-registered):
  - Entry at bar Open ± entry slippage (0.25 tick).
  - Intra-bar: if stop AND target are both touched in one bar, STOP fills first
    (conservative, worst case).
  - T1 = half off (+stop→breakeven), T2 = remainder. Half-split is the research
    idealization; live it needs ≥2 contracts.
  - Stop fills slip 0.5 tick; target/flat fills slip 0.25 tick.
  - Forced flat at the 15:55 ET bar Open. Roll-day sessions skipped for
    structure modes (spliced continuous series).
Costs: $0.35/side commission per contract.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sovereign.es_nq import data_store
from sovereign.es_nq.config import contract_spec, es_nq_params
from sovereign.es_nq.daily_bias_engine import build_feature_table
from sovereign.es_nq.session_sizing import SessionLadder
from sovereign.es_nq.structure_gate import (
    Levels, detect_confirmation, detect_sweep, plan_trade, session_levels,
)

ET_FLAT_DEFAULT = "15:55"


def _hhmm_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _bar_minutes(bars5: pd.DataFrame) -> np.ndarray:
    et = bars5.index.tz_convert("America/New_York")
    return np.asarray(et.hour * 60 + et.minute)


def simulate_bracket(bars5: pd.DataFrame, entry_idx: int, direction: str,
                     entry: float, stop: float, t1: float, t2: float,
                     instrument: str, params: Optional[dict] = None) -> dict:
    """Walk bars from entry_idx; return per-contract outcome dict.

    Identical machinery for real trades AND permutation nulls (Stage 2) — the
    null must flow through the same exits or the comparison is dishonest.
    """
    p = params or es_nq_params()
    spec = contract_spec(instrument)
    tick, dpp = spec["tick"], spec["dollars_per_point"]
    slip_t = p["costs"]["slippage_ticks_entry"] * tick     # target/flat fills
    slip_s = p["costs"]["slippage_ticks_stop"] * tick      # stop fills
    comm_rt = 2.0 * p["costs"]["commission_per_side_usd"]
    flat_min = _hhmm_minutes(p["structure"].get("flat_by_et", ET_FLAT_DEFAULT))
    minutes = _bar_minutes(bars5)
    long = direction in ("LONG", "UP")
    sgn = 1.0 if long else -1.0
    risk_pts = abs(entry - stop)

    def hit_stop(bar, s):
        return bar["Low"] <= s if long else bar["High"] >= s

    def hit_tgt(bar, t):
        return bar["High"] >= t if long else bar["Low"] <= t

    halves = [{"target": t1, "stop": stop, "pending_stop": None,
               "open": True, "exit": None, "reason": None},
              {"target": t2, "stop": stop, "pending_stop": None,
               "open": True, "exit": None, "reason": None}]
    t1_done = False
    exit_bar = entry_idx
    for i in range(entry_idx, len(bars5)):
        bar = bars5.iloc[i]
        forced_flat = minutes[i] >= flat_min
        for h in halves:
            # Breakeven stop becomes effective on the bar AFTER T1 fills —
            # same-bar application would degenerately stop the runner at entry
            # on nearly every entry bar (pre-registered convention).
            if h["pending_stop"] is not None:
                h["stop"] = h["pending_stop"]
                h["pending_stop"] = None
        for h in halves:
            if not h["open"]:
                continue
            if forced_flat:
                h["exit"] = float(bar["Open"]) - sgn * slip_t
                h["reason"] = "FLAT"
                h["open"] = False
            elif hit_stop(bar, h["stop"]):                # stop first — conservative
                h["exit"] = h["stop"] - sgn * slip_s
                h["reason"] = "BREAKEVEN" if (t1_done and h["stop"] == entry) else "STOP"
                h["open"] = False
            elif hit_tgt(bar, h["target"]):
                h["exit"] = h["target"] - sgn * slip_t
                h["reason"] = "T1" if h["target"] == t1 else "T2"
                h["open"] = False
                if h["target"] == t1 and not t1_done:
                    t1_done = True
                    for h2 in halves:
                        if h2["open"]:
                            h2["pending_stop"] = entry    # runner to breakeven NEXT bar
        if all(not h["open"] for h in halves):
            exit_bar = i
            break
        exit_bar = i

    pts = sum(0.5 * sgn * (h["exit"] - entry) for h in halves)
    usd_gross = pts * dpp
    usd_net = usd_gross - comm_rt
    risk_usd = risk_pts * dpp
    r_net = usd_net / risk_usd if risk_usd > 0 else 0.0
    return {
        "direction": "LONG" if long else "SHORT",
        "entry": round(entry, 4), "stop": round(stop, 4),
        "t1": round(t1, 4), "t2": round(t2, 4),
        "exit_reasons": [h["reason"] for h in halves],
        "exit_prices": [round(h["exit"], 4) for h in halves],
        "r_net": round(float(r_net), 4),
        "usd_net_per_contract": round(float(usd_net), 4),
        "risk_usd_per_contract": round(float(risk_usd), 4),
        "stop_points": round(float(risk_pts), 4),
        "entry_bar_idx": int(entry_idx), "exit_bar_idx": int(exit_bar),
        "bars_held": int(exit_bar - entry_idx + 1),
    }


def simulate_entry_at(bars5: pd.DataFrame, entry_idx: int, direction: str,
                      stop_points: float, instrument: str,
                      params: Optional[dict] = None) -> dict:
    """Bracket trade at an arbitrary bar with a given stop distance — the Stage-2
    permutation-null entry point. Same slippage/targets/exits as real trades."""
    p = params or es_nq_params()
    spec = contract_spec(instrument)
    slip = p["costs"]["slippage_ticks_entry"] * spec["tick"]
    s = p["structure"]
    long = direction in ("LONG", "UP")
    raw = float(bars5["Open"].iloc[entry_idx])
    entry = raw + slip if long else raw - slip
    stop = entry - stop_points if long else entry + stop_points
    t1 = entry + s["t1_r"] * stop_points * (1 if long else -1)
    t2 = entry + s["t2_r"] * stop_points * (1 if long else -1)
    return simulate_bracket(bars5, entry_idx, direction, entry, stop, t1, t2,
                            instrument, params)


def daily_atr(daily: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR on the daily session table (rth_high/rth_low/rth_close).
    Value at date t uses data through t (callers must use .shift(1) semantics —
    bias_only stops read atr.loc[<prior session>])."""
    h, l, c = daily["rth_high"], daily["rth_low"], daily["rth_close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def structure_trades_for_session(bars5: pd.DataFrame, levels: Levels, bias_dir: str,
                                 instrument: str, params: Optional[dict] = None,
                                 max_trades: Optional[int] = None) -> list[dict]:
    """Up to max_trades sequential structure-gate trades for one session.
    After a trade exits, scanning resumes from the next bar."""
    p = params or es_nq_params()
    cap = max_trades if max_trades is not None else p["sizing"]["max_trades_per_session"]
    trades: list[dict] = []
    offset = 0
    while len(trades) < cap and offset < len(bars5) - 2:
        window = bars5.iloc[offset:]
        sweep = detect_sweep(window, levels, bias_dir, p)
        if sweep is None:
            break
        confirm = detect_confirmation(window, sweep, bias_dir, p)
        if confirm is None:
            break
        plan = plan_trade(window, confirm, sweep, bias_dir, instrument, p)
        if plan is None:
            break
        result = simulate_bracket(window, plan.entry_bar_idx, plan.direction,
                                  plan.entry, plan.stop, plan.t1, plan.t2,
                                  instrument, p)
        result["sweep_level"] = sweep.level_name
        result["entry_bar_idx_session"] = int(offset + plan.entry_bar_idx)
        result["exit_bar_idx_session"] = int(offset + result["exit_bar_idx"])
        trades.append(result)
        offset = offset + result["exit_bar_idx"] + 1
    return trades


def run_backtest(start: str, end: str, mode: str,
                 params: Optional[dict] = None,
                 daily: Optional[pd.DataFrame] = None,
                 bars5_all: Optional[pd.DataFrame] = None,
                 feature_table: Optional[pd.DataFrame] = None,
                 instrument: Optional[str] = None) -> list[dict]:
    """One session record per non-NEUTRAL bias day in [start, end].

    Frames are injectable for tests; by default loads the parquet caches.
    """
    if mode not in ("bias_only", "bias_gate", "full"):
        raise ValueError(f"unknown mode: {mode}")
    p = params or es_nq_params()
    inst = instrument or p["meta"]["trade_instrument"]
    daily = daily if daily is not None else data_store.load_daily()
    bars5_all = bars5_all if bars5_all is not None else data_store.load_5min()
    if feature_table is None:
        aux = data_store.load_aux_daily()
        import json
        cal = json.loads((data_store.DATA_DIR / "econ_calendar_2018_2026.json").read_text())
        feature_table = build_feature_table(daily, aux, cal, start, end, p)

    atr = daily_atr(daily, p["bias_only_baseline"]["atr_period_days"])
    et_dates = bars5_all.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
    daily_dates = list(daily.index)
    sessions: list[dict] = []

    for date, frow in feature_table.iterrows():
        if not (start <= date <= end):
            continue
        bias_dir = frow["direction"]
        record = {
            "session_date": date, "mode": "BACKTEST", "backtest_mode": mode,
            "bias": {"direction": bias_dir, "confidence": float(frow["confidence"]),
                     "raw_score": float(frow["raw_score"]),
                     "event_day": bool(frow["event_day"]),
                     "roll_day": bool(frow["roll_day"])},
            "direction_real": frow["direction_real"], "move_pct": float(frow["move_pct"]),
            "trades": [], "session_r_total": 0.0, "session_usd_total": 0.0,
            "skipped": None,
        }
        if bias_dir == "NEUTRAL":
            record["skipped"] = "NEUTRAL_BIAS"
            sessions.append(record)
            continue
        if mode in ("bias_gate", "full") and bool(frow["roll_day"]):
            record["skipped"] = "ROLL_DAY"
            sessions.append(record)
            continue
        bars5 = bars5_all[et_dates == date]
        if len(bars5) < 6:
            record["skipped"] = "NO_BARS"
            sessions.append(record)
            continue

        if mode == "bias_only":
            trades = _bias_only_trade(bars5, daily, daily_dates, atr, date,
                                      bias_dir, inst, p)
        else:
            di = daily_dates.index(date)
            if di == 0:
                record["skipped"] = "NO_PRIOR_SESSION"
                sessions.append(record)
                continue
            levels = session_levels(daily.iloc[di - 1], daily.iloc[di])
            trades = structure_trades_for_session(bars5, levels, bias_dir, inst, p)
            record["levels"] = {"pdh": levels.pdh, "pdl": levels.pdl,
                                "onh": levels.onh, "onl": levels.onl}

        ladder = SessionLadder(account_usd=p["sizing"]["account_base_usd"],
                               flat_risk_pct=(None if mode == "full" else 0.005),
                               params=p)
        for t in trades:
            role = ladder.next_role()
            if role is None:
                break
            n = ladder.contracts(role, t["stop_points"], inst)
            t = dict(t)
            t["role"], t["contracts"] = role, n
            t["usd_net"] = round(t["usd_net_per_contract"] * n, 4)
            ladder.record(t["r_net"], t["usd_net"])
            record["trades"].append(t)
        record["session_r_total"] = round(sum(t["r_net"] for t in record["trades"]), 4)
        record["session_usd_total"] = round(sum(t["usd_net"] for t in record["trades"]), 4)
        record["bias_was_correct"] = (frow["direction_real"] == bias_dir
                                      if frow["direction_real"] else None)
        sessions.append(record)
    return sessions


def _bias_only_trade(bars5, daily, daily_dates, atr, date, bias_dir, inst, p) -> list[dict]:
    """The pre-registered Stage-2 baseline: one trade at the 09:35 bar Open."""
    b = p["bias_only_baseline"]
    minutes = _bar_minutes(bars5)
    target_min = _hhmm_minutes(b["entry_bar_et"])
    idxs = np.where(minutes == target_min)[0]
    if len(idxs) == 0:
        return []
    entry_idx = int(idxs[0])
    di = daily_dates.index(date)
    if di == 0:
        return []
    atr_prior = float(atr.iloc[di - 1])
    if not np.isfinite(atr_prior) or atr_prior <= 0:
        return []
    return [simulate_entry_at(bars5, entry_idx, bias_dir,
                              b["stop_atr_mult"] * atr_prior, inst, p)]
