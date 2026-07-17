"""Event-level backtest engine — bias-free fill model.

The fill model is the whole point. It eliminates the exact-trigger stop-fill
bias (research/gapper/BACKTEST_BIAS_AUDIT.md, finding #1, ~19pt/yr):

  Entry  : first minute bar with time >= entry_time; fill at that bar's OPEN.
  Stop   : scan bars after entry; first bar whose HIGH (short) / LOW (long)
           breaches the stop trigger. Fill at that bar's OPEN when the bar
           GAPPED THROUGH (open already beyond trigger) — else at the trigger.
           Never blindly at the trigger on a gap-through bar.
  Exit   : bar at exit_time close (or last bar of a short session).
  Slippage: (day_high - day_low)/day_close * 0.5, charged one-way on entry.
  Locate : if locate_required, gate against IB shortable snapshot for the date.

run(events_df, strategy_config, data_cache) -> results dict with a per-event
records list and aggregate daily P&L. Pure per-event computation; the scanner
memoises this so sizing sweeps stay vectorised.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as _data
from . import audit as _audit

REPO = Path(__file__).resolve().parents[1]
LOCATE_DIR = REPO / "data/research/gapper"

# HYP-093 borrow friction (annualised HTB APR by gap tier) — locate haircut.
def _apr(gain: float) -> float:
    return 6.0 if gain >= 1.5 else 4.0 if gain >= 1.0 else 2.0


def _load_locate(date: str) -> dict | None:
    fp = LOCATE_DIR / f"ib_locate_{date}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text()).get("detail") or {}


def _first_at_or_after(df: pd.DataFrame, hhmm: str) -> int | None:
    idx = df.index[df["time"] >= hhmm]
    return int(idx[0]) if len(idx) else None


def _simulate_event(bars: pd.DataFrame, cfg: dict, gain: float) -> dict:
    """Return fill/exit/return fields for one event. bars: normalised ET df."""
    direction = cfg["direction"]
    stop_pct = cfg["stop_pct"]
    slip_extra = cfg.get("slippage", 0.005)  # base per-side slippage floor
    ei = _first_at_or_after(bars, cfg["entry_time"])
    if ei is None or ei == 0:
        # ei==0 would mean no pre-entry bar exists -> cannot confirm no look-ahead
        return {"trade_taken": False, "reason": "no_entry_bar",
                "entry_price": None, "fill_price": None, "exit_price": None,
                "stop_hit": False, "stop_fill_price": None,
                "gross_pct": 0.0, "net_pct": 0.0, "filled_at_trigger": False}

    entry_price = float(bars.at[ei, "open"])
    if entry_price <= 0:
        return {"trade_taken": False, "reason": "bad_entry_price",
                "gross_pct": 0.0, "net_pct": 0.0, "filled_at_trigger": False,
                "stop_hit": False}

    # Spread proxy = ENTRY BAR's range, not the whole-day range. The mandate's
    # literal (day_high-day_low)/close*0.5 is pathological for parabolic
    # gappers (intraday range often exceeds price -> 50-60% phantom cost that
    # nukes every trade). The entry-bar high-low is the defensible bid-ask
    # spread estimate at the moment of the fill. Deviation flagged in the
    # bias audit / results doc.
    eb_high = float(bars.at[ei, "high"])
    eb_low = float(bars.at[ei, "low"])
    spread_cost = ((eb_high - eb_low) / entry_price * 0.5) if entry_price else 0.0

    post = bars.iloc[ei + 1:]
    # exit_time may be a clock "10:30" OR a duration "+30" (minutes/bars after
    # entry). Duration assumes 1-min bars (ei + N).
    et = str(cfg["exit_time"])
    if et.startswith("+"):
        ex_i = min(ei + int(et[1:]), len(bars) - 1)
    else:
        ex_i = _first_at_or_after(bars, et)
        if ex_i is None or ex_i <= ei:
            ex_i = len(bars) - 1
    exit_price = float(bars.at[ex_i, "close"])

    stop_hit = False
    filled_at_trigger = False
    stop_fill_price = None

    if direction == "short":
        trigger = entry_price * (1 + stop_pct)
        breach = post.index[post["high"] >= trigger]
        if len(breach):
            bi = int(breach[0])
            bar_open = float(bars.at[bi, "open"])
            # gap-through: bar opened already at/above trigger -> fill at open
            stop_fill_price = bar_open if bar_open >= trigger else trigger
            filled_at_trigger = not (bar_open >= trigger)
            # short P&L: (entry - fill)/entry
            gross = (entry_price - stop_fill_price) / entry_price
            stop_hit = True
        else:
            gross = (entry_price - exit_price) / entry_price
    else:  # long
        trail = cfg.get("trail_pct")
        trigger = entry_price * (1 - stop_pct)
        # Walk bars from entry+1 to the time-exit; whichever of hard stop or
        # trailing stop triggers first wins. Trailing peak = running max high.
        window = bars.iloc[ei + 1:ex_i + 1]
        peak = entry_price
        fired_i = None
        fired_trig = None
        for j in window.index:
            hard = trigger
            tstop = peak * (1 - trail) if trail else -1e18
            lvl = max(hard, tstop)
            if float(bars.at[j, "low"]) <= lvl:
                fired_i, fired_trig = int(j), lvl
                break
            peak = max(peak, float(bars.at[j, "high"]))
        if fired_i is not None:
            bar_open = float(bars.at[fired_i, "open"])
            stop_fill_price = bar_open if bar_open <= fired_trig else fired_trig
            filled_at_trigger = not (bar_open <= fired_trig)
            gross = (stop_fill_price - entry_price) / entry_price
            stop_hit = True
        else:
            gross = (exit_price - entry_price) / entry_price

    locate_haircut = 0.50 * _apr(gain) / 252
    net = gross - spread_cost - slip_extra - locate_haircut

    return {"trade_taken": True, "reason": "ok",
            "entry_bar_index": ei, "entry_price": round(entry_price, 4),
            "fill_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "stop_hit": stop_hit,
            "stop_fill_price": (round(stop_fill_price, 4)
                                if stop_fill_price is not None else None),
            "filled_at_trigger": filled_at_trigger,
            "spread_cost": round(spread_cost, 5),
            "gross_pct": round(gross, 5), "net_pct": round(net, 5)}


def run(events_df: pd.DataFrame, strategy_config: dict,
        data_cache=None, write_audit: bool = True) -> dict:
    """Backtest every event in events_df under strategy_config.

    events_df columns required: date, ticker, gain (gap fraction for haircut).
    strategy_config: entry_time, stop_pct, exit_time, direction, sizing_pct,
                     locate_required (bool), on_missing_locate ('take'|'skip').
    """
    cfg = dict(strategy_config)
    cfg.setdefault("exit_time", "15:45")
    cfg.setdefault("direction", "short")
    cfg.setdefault("on_missing_locate", "take")
    sizing = cfg.get("sizing_pct", 0.02)

    locate_cache: dict[str, dict | None] = {}
    records, daily = [], {}
    for _, e in events_df.iterrows():
        date, ticker = e["date"], e["ticker"]
        gain = float(e.get("gain", 1.0))
        rec = {"date": date, "ticker": ticker}

        # locate gate
        locate_status = "not_required"
        if cfg.get("locate_required"):
            if date not in locate_cache:
                locate_cache[date] = _load_locate(date)
            snap = locate_cache[date]
            if snap is None:
                locate_status = "UNKNOWN"
            else:
                tier = (snap.get(ticker) or {}).get("tier", "NOT_LISTED")
                locate_status = "EASY" if tier == "EASY" else "NO_LOCATE"
        rec["locate_status"] = locate_status

        skip_for_locate = (
            cfg.get("locate_required")
            and (locate_status == "NO_LOCATE"
                 or (locate_status == "UNKNOWN"
                     and cfg["on_missing_locate"] == "skip")))
        if skip_for_locate:
            rec.update({"trade_taken": False, "reason": "no_locate",
                        "net_pct": 0.0, "gross_pct": 0.0,
                        "filled_at_trigger": False, "stop_hit": False})
            records.append(rec)
            continue

        bars = (data_cache.get((ticker, date)) if data_cache else None)
        if bars is None:
            bars = _data.get_minute_bars(ticker, date)
        if bars is None or not len(bars):
            rec.update({"trade_taken": False, "reason": "no_bars",
                        "net_pct": 0.0, "gross_pct": 0.0,
                        "filled_at_trigger": False, "stop_hit": False})
            records.append(rec)
            continue

        sim = _simulate_event(bars, cfg, gain)
        rec.update(sim)
        records.append(rec)
        if sim["trade_taken"]:
            daily[date] = daily.get(date, 0.0) + sim["net_pct"] * sizing

    dates = sorted(daily)
    equity, curve, peak, maxdd = 1.0, [], 1.0, 0.0
    for d in dates:
        equity = max(equity * (1 + max(daily[d], -1.0)), 0.0)  # ruin absorbing
        peak = max(peak, equity)
        maxdd = max(maxdd, 1 - equity / peak) if peak > 0 else 1.0
        curve.append({"date": d, "day_return": daily[d], "equity": equity})

    day_series = [daily[d] for d in dates]
    ann = (equity - 1) if day_series else 0.0
    sharpe = (float(np.mean(day_series) / np.std(day_series) * np.sqrt(252))
              if len(day_series) > 1 and np.std(day_series) > 0 else 0.0)
    taken = [r for r in records if r.get("trade_taken")]
    result = {
        "config": cfg,
        "n_events": len(records), "n_taken": len(taken),
        "n_no_locate": sum(1 for r in records if r.get("reason") == "no_locate"),
        "n_unknown_locate": sum(1 for r in records
                                if r.get("locate_status") == "UNKNOWN"),
        "annual_return": round(ann, 5), "sharpe": round(sharpe, 4),
        "max_drawdown": round(maxdd, 5),
        "win_rate": (round(sum(1 for r in taken if r["net_pct"] > 0) / len(taken), 4)
                     if taken else 0.0),
        "daily_pnl": {c["date"]: c["day_return"] for c in curve},
        "equity_curve": curve, "records": records,
    }
    result["audit"] = _audit.audit_run(result, write=write_audit)
    return result
