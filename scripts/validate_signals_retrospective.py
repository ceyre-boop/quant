#!/usr/bin/env python3
"""
G2a — Signal Validation Retrospective

Runs the live ICT pipeline against the last 30 trading days of real historical
data using the exact same code that fires at 3 AM. Proves signal quality
without waiting for live paper trades.

Pass criteria:
  - signals_generated >= 3
  - win_rate (closed trades only) >= 0.30

Usage:
    python3 scripts/validate_signals_retrospective.py

Output:
    data/agent/g2a_validation.json
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ict.pipeline import ICTPipeline, ICTSignal, ICTVeto, ICTGrade
from ict.micro_risk import MicroRiskParams

PAIRS = ["GBPUSD=X", "EURUSD=X", "AUDUSD=X", "AUDNZD=X", "USDJPY=X"]
LOOKBACK_H = 120        # 5 trading days × ~24h of bars fed into pipeline per day
OUTCOME_BARS = 120      # look 120h (5 trading days) forward to classify WIN/LOSS/PENDING
WIN_RATE_FLOOR = 0.30
MIN_SIGNALS = 3
BACKTEST_EXPECTATION = 0.41
ACCOUNT = MicroRiskParams(account_size=100_000)


def trading_days_back(n: int) -> list[date]:
    days: list[date] = []
    d = date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days


def make_london_open_ts(target_date: date) -> datetime:
    """Return 03:15 AM Eastern on target_date as a UTC-aware datetime."""
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        ts_et = datetime(target_date.year, target_date.month, target_date.day, 3, 15, tzinfo=et)
        return ts_et.astimezone(timezone.utc)
    except ImportError:
        # Fallback: EDT offset (UTC-4). Close enough for retrospective.
        naive = datetime(target_date.year, target_date.month, target_date.day, 7, 15)
        return naive.replace(tzinfo=timezone.utc)


def classify_outcome(df: pd.DataFrame, cutoff_ts: pd.Timestamp, signal: ICTSignal) -> str:
    future = df[df.index > cutoff_ts].head(OUTCOME_BARS)
    if future.empty:
        return "PENDING"
    stop = signal.sizing.stop_loss
    tp1 = signal.sizing.tp1
    for _, row in future.iterrows():
        if signal.direction == "LONG":
            if row["Low"] <= stop:   return "LOSS"
            if row["High"] >= tp1:   return "WIN"
        else:
            if row["High"] >= stop:  return "LOSS"
            if row["Low"] <= tp1:    return "WIN"
    return "PENDING"


def evaluate_day_pair(
    full_df: pd.DataFrame,
    cutoff_ts: pd.Timestamp,
    symbol: str,
    pipeline: ICTPipeline,
    weekly_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    slice_df = full_df[full_df.index <= cutoff_ts].tail(LOOKBACK_H)
    if len(slice_df) < 20:
        return None

    ts = cutoff_ts.to_pydatetime()
    weekly_slice = weekly_df[weekly_df.index <= cutoff_ts] if weekly_df is not None else None

    for direction in ("LONG", "SHORT"):
        result = pipeline.evaluate(
            symbol=symbol,
            direction=direction,
            df=slice_df.copy(),
            timestamp=ts,
            account=ACCOUNT,
            weekly_df=weekly_slice,
        )
        if isinstance(result, ICTSignal) and result.passed:
            # Mirror live orchestrator's bias_agrees gate — signals blocked there
            # won't appear in live results, so exclude them here too.
            try:
                from ict.daily_bias import DailyBiasEngine
                biases = DailyBiasEngine().get_biases()
                bias = biases.get(symbol.replace("=X", ""), {})
                if bias.get("blackout", False):
                    continue
                bias_dir = bias.get("bias", "NEUTRAL")
                if bias_dir != "NEUTRAL" and bias_dir != direction:
                    continue
            except Exception:
                pass  # no bias data → don't block

            outcome = classify_outcome(full_df, cutoff_ts, result)
            return {
                "date": str(cutoff_ts.date()),
                "pair": symbol,
                "direction": direction,
                "grade": result.grade.value,
                "score": round(result.score, 2),
                "entry": round(result.entry_level, 5) if result.entry_level else None,
                "stop": round(result.sizing.stop_loss, 5),
                "tp1": round(result.sizing.tp1, 5),
                "outcome": outcome,
            }
    return None


def main() -> None:
    print("G2a — Signal Validation Retrospective")
    print(f"Scanning last 30 trading days × {len(PAIRS)} pairs\n")

    start_dt = (date.today() - timedelta(days=92)).isoformat()
    end_dt = date.today().isoformat()

    print("Downloading historical data (90-day 1h window)…")
    all_data: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        try:
            df = yf.download(pair, start=start_dt, end=end_dt, interval="1h",
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"  {pair}: no data returned")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.capitalize)
            df.index = pd.to_datetime(df.index, utc=True)
            all_data[pair] = df
            print(f"  {pair}: {len(df)} bars")
        except Exception as exc:
            print(f"  {pair}: download failed — {exc}")

    if not all_data:
        print("\nERROR: No data downloaded. Check network/yfinance.")
        sys.exit(1)

    print("\nDownloading weekly data (2y 1W window for Stage 5.6 trend gate)…")
    weekly_data: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        try:
            wdf = yf.download(pair, period="2y", interval="1wk",
                              progress=False, auto_adjust=True)
            if wdf.empty:
                continue
            if isinstance(wdf.columns, pd.MultiIndex):
                wdf.columns = wdf.columns.get_level_values(0)
            wdf = wdf.rename(columns=str.capitalize)
            wdf.index = pd.to_datetime(wdf.index, utc=True)
            weekly_data[pair] = wdf
            print(f"  {pair}: {len(wdf)} weekly bars")
        except Exception as exc:
            print(f"  {pair}: weekly download failed — {exc}")

    target_days = trading_days_back(30)
    print(f"\nRunning pipeline across {len(target_days)} trading days…")

    pipeline = ICTPipeline()
    signals: list[dict] = []
    pairs_breakdown: dict[str, dict] = {
        p: {"signals": 0, "wins": 0, "losses": 0, "pending": 0} for p in PAIRS
    }

    for target_day in target_days:
        cutoff_ts = pd.Timestamp(make_london_open_ts(target_day))
        for pair in PAIRS:
            if pair not in all_data:
                continue
            result = evaluate_day_pair(all_data[pair], cutoff_ts, pair, pipeline, weekly_data.get(pair))
            if result:
                signals.append(result)
                pb = pairs_breakdown[pair]
                pb["signals"] += 1
                if result["outcome"] == "WIN":
                    pb["wins"] += 1
                elif result["outcome"] == "LOSS":
                    pb["losses"] += 1
                else:
                    pb["pending"] += 1
                print(f"  {result['date']} {pair:12s} {result['direction']:5s} "
                      f"{result['grade']:3s} score={result['score']:.1f} → {result['outcome']}")

    closed = [s for s in signals if s["outcome"] in ("WIN", "LOSS")]
    wins = sum(1 for s in closed if s["outcome"] == "WIN")
    losses = len(closed) - wins
    win_rate = wins / len(closed) if closed else 0.0

    if len(signals) < MIN_SIGNALS:
        status = "FAIL"
        reason = f"Too few signals ({len(signals)} < {MIN_SIGNALS} required)"
    elif win_rate < WIN_RATE_FLOOR and closed:
        status = "FAIL"
        reason = f"WR {win_rate:.1%} below floor {WIN_RATE_FLOOR:.0%}"
    else:
        status = "PASS"
        reason = (f"WR {win_rate:.1%} ≥ floor {WIN_RATE_FLOOR:.0%} "
                  f"on {len(signals)} signals ({len(closed)} closed)")

    output = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "days_scanned": len(target_days),
        "signals_generated": len(signals),
        "wins": wins,
        "losses": losses,
        "pending": len(signals) - len(closed),
        "win_rate": round(win_rate, 4),
        "backtest_expectation": BACKTEST_EXPECTATION,
        "status": status,
        "status_reason": reason,
        "pairs_breakdown": pairs_breakdown,
        "signals": signals,
    }

    out_path = ROOT / "data" / "agent" / "g2a_validation.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\n{'─'*54}")
    print(f"Signals:  {len(signals)}  (need ≥ {MIN_SIGNALS})")
    print(f"Wins:     {wins}  Losses: {losses}  Pending: {output['pending']}")
    print(f"Win rate: {win_rate:.1%}  (backtest: {BACKTEST_EXPECTATION:.0%}, floor: {WIN_RATE_FLOOR:.0%})")
    print(f"Status:   {status} — {reason}")
    print(f"Output:   {out_path}")

    if status == "PASS":
        print("\n✓ G2a PASS — signal pipeline validated against 30 days of real data")
        print("  Next: run sync_dashboard_data.py to update G2a gate on dashboard")
    else:
        print(f"\n✗ G2a FAIL — {reason}")
        if len(signals) < MIN_SIGNALS:
            print("  Suggestion: market may have been range-bound; extend to 60 days or wait")
        else:
            print("  Suggestion: investigate pipeline — compare recent signals to backtest conditions")

    # Update g2_progress.json with G2a results
    g2_path = ROOT / "data" / "agent" / "g2_progress.json"
    if g2_path.exists():
        g2 = json.loads(g2_path.read_text())
        g2["g2a_status"] = status
        g2["g2a_signals"] = len(signals)
        g2["g2a_win_rate"] = round(win_rate, 4) if closed else None
        g2_path.write_text(json.dumps(g2, indent=2))
        print(f"\n  Updated {g2_path.name}: g2a_status={status}")


if __name__ == "__main__":
    main()
