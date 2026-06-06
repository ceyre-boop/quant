#!/usr/bin/env python3
"""
Interactive trade logger for MES/MNQ futures sandbox.

Prompts for each field, computes r_realized and bias_aligned,
then appends one JSONL entry to data/futures/trade_log.jsonl.

Also reads today's bias from bias_log.jsonl to pre-fill direction fields.

Usage:
    python3.13 scripts/futures_log.py
    python3.13 scripts/futures_log.py --list         # show last N trades
    python3.13 scripts/futures_log.py --summary      # session P&L summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"
BIAS_LOG  = ROOT / "data" / "futures" / "bias_log.jsonl"

INSTRUMENTS   = ["MES", "MNQ"]
DIRECTIONS    = ["LONG", "SHORT"]
EXIT_REASONS  = ["T1_HIT", "T2_HIT", "STOPPED", "MANUAL", "TIME_STOP"]
SIZE_RATIONALE = ["probe", "press", "reduce", "stand_down"]


# ── helpers ─────────────────────────────────────────────────────────────────

def _read_today_bias(instrument: str) -> dict | None:
    """Load the most recent bias entry for today's instrument."""
    if not BIAS_LOG.exists():
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last = None
    with open(BIAS_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("date") == today and rec.get("instrument", "").upper() == instrument.upper():
                    last = rec
            except Exception:
                pass
    return last


def _prompt(label: str, default=None, choices: list | None = None, cast=str, optional=False) -> any:
    """Single-field prompt with validation."""
    hint = ""
    if choices:
        hint = f" [{'/'.join(str(c) for c in choices)}]"
    if default is not None:
        hint += f" (default: {default})"
    elif optional:
        hint += " (blank to skip)"

    while True:
        raw = input(f"  {label}{hint}: ").strip()
        if not raw:
            if default is not None:
                return default
            if optional:
                return None
            print(f"    Required. Choose from: {choices}" if choices else "    Required.")
            continue
        if choices and raw.upper() not in [str(c).upper() for c in choices]:
            print(f"    Must be one of: {choices}")
            continue
        try:
            val = cast(raw.upper() if cast == str and choices else raw)
            return val
        except (ValueError, TypeError):
            print(f"    Invalid — expected {cast.__name__}")


def _compute_r(entry: float, stop: float, exit_price: float, direction: str) -> float:
    """R-multiple: 1R = distance from entry to stop."""
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    if direction == "LONG":
        return round((exit_price - entry) / risk, 3)
    else:
        return round((entry - exit_price) / risk, 3)


def _load_trades(date: str | None = None) -> list[dict]:
    if not TRADE_LOG.exists():
        return []
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            try:
                t = json.loads(line)
                if date is None or t.get("ts", "").startswith(date):
                    trades.append(t)
            except Exception:
                pass
    return trades


def _session_trade_num(instrument: str) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    trades = _load_trades(today)
    count = sum(1 for t in trades if t.get("instrument") == instrument)
    return count + 1


def _session_r_so_far(instrument: str) -> float:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    trades = _load_trades(today)
    return round(sum(
        (t.get("r_realized") or 0) * (t.get("size_contracts") or 1)
        for t in trades
        if t.get("instrument") == instrument
    ), 3)


# ── list / summary modes ─────────────────────────────────────────────────────

def _cmd_list(n: int) -> None:
    trades = _load_trades()
    if not trades:
        print("No trades logged yet.")
        return
    recent = trades[-n:]
    print(f"\nLast {len(recent)} trade(s):")
    for t in recent:
        r = t.get("r_realized", 0)
        sign = "+" if r >= 0 else ""
        print(f"  {t['ts'][:10]}  {t['instrument']:4s}  "
              f"{t['direction']:5s}  {sign}{r:.2f}R  "
              f"{t.get('exit_reason','?'):10s}  {t.get('sizing_rationale','?')}")
    print()


def _cmd_summary() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_trades = _load_trades()
    today_trades = _load_trades(today)

    def stats(trades: list[dict]) -> dict:
        if not trades:
            return {"n": 0}
        total_r = sum((t.get("r_realized") or 0) * (t.get("size_contracts") or 1) for t in trades)
        wins = sum(1 for t in trades if (t.get("r_realized") or 0) > 0)
        aligned = [t for t in trades if t.get("bias_aligned")]
        aligned_wins = sum(1 for t in aligned if (t.get("r_realized") or 0) > 0)
        return {
            "n": len(trades),
            "total_r": round(total_r, 3),
            "win_rate": round(wins / len(trades) * 100, 1),
            "bias_alignment_win_rate": round(aligned_wins / len(aligned) * 100, 1) if aligned else None,
            "aligned": len(aligned),
        }

    today_s = stats(today_trades)
    all_s   = stats(all_trades)

    print(f"\n{'═'*50}")
    print(f"  FUTURES SANDBOX — SUMMARY")
    print(f"{'═'*50}")
    print(f"  Today  ({today}): {today_s['n']} trades  "
          f"R={today_s.get('total_r', 0):+.2f}  "
          f"W={today_s.get('win_rate', 0):.0f}%")
    print(f"  All-time:          {all_s['n']} trades  "
          f"R={all_s.get('total_r', 0):+.2f}  "
          f"W={all_s.get('win_rate', 0):.0f}%")
    if all_s.get("aligned") and all_s.get("bias_alignment_win_rate") is not None:
        print(f"  Bias-aligned win%: {all_s['bias_alignment_win_rate']:.0f}%  "
              f"(n={all_s['aligned']})")
    bpb = sum(1 for t in all_trades if t.get("below_proven_bar"))
    if bpb:
        bpb_wins = sum(1 for t in all_trades
                       if t.get("below_proven_bar") and (t.get("r_realized") or 0) > 0)
        print(f"  BELOW_PROVEN_BAR:  {bpb_wins}/{bpb} wins  "
              f"({round(bpb_wins/bpb*100,1):.0f}%)")
    remaining = 150 - all_s["n"]
    print(f"\n  Progress to validation: {all_s['n']}/150 "
          f"({'DONE' if remaining <= 0 else f'{remaining} remaining'})")
    print(f"{'═'*50}\n")


# ── main log flow ─────────────────────────────────────────────────────────────

def _log_trade() -> None:
    print(f"\n{'─'*50}")
    print("  LOG TRADE — Futures Sandbox")
    print(f"{'─'*50}")

    # Instrument
    instrument = _prompt("Instrument", default="MES", choices=INSTRUMENTS)

    # Load today's bias for defaults
    bias = _read_today_bias(instrument)
    if bias:
        print(f"\n  Today's bias: {bias['bias']}  (conviction {bias['conviction']}/3)")
        bias_dir  = bias["bias"]
        bias_conv = bias["conviction"]
    else:
        print("\n  No bias logged for today — run futures_bias.py first for best results.")
        bias_dir  = _prompt("Bias direction (from today's bias)", choices=DIRECTIONS + ["NEUTRAL"])
        bias_conv = _prompt("Bias conviction (1/2/3)", choices=[1, 2, 3], cast=int)

    print()

    # Trade direction
    direction = _prompt("Trade direction", default=bias_dir if bias_dir in DIRECTIONS else None,
                        choices=DIRECTIONS)

    # Prices
    entry = _prompt("Entry price", cast=float)
    stop  = _prompt("Stop price", cast=float)
    t1    = _prompt("Target 1 (T1)", cast=float)
    t2    = _prompt("Target 2 (T2, blank to skip)", cast=float, optional=True)

    # Exit
    exit_price  = _prompt("Exit price", cast=float)
    exit_reason = _prompt("Exit reason", choices=EXIT_REASONS)

    # Size + rationale
    size = _prompt("Contracts", default=1, cast=int)

    # Suggest sizing rationale based on session history
    trade_num = _session_trade_num(instrument)
    r_so_far  = _session_r_so_far(instrument)
    if trade_num == 1:
        suggested_rationale = "probe"
    elif r_so_far > 0:
        suggested_rationale = "press"
    else:
        suggested_rationale = "reduce"
    rationale = _prompt("Sizing rationale", default=suggested_rationale, choices=SIZE_RATIONALE)

    # Notes
    notes = input("  Notes (optional, Enter to skip): ").strip() or None

    # Computed fields
    r_realized     = _compute_r(entry, stop, exit_price, direction)
    bias_aligned   = (bias_dir == direction) if bias_dir in DIRECTIONS else None
    below_proven   = (bias_conv < 2)  # conviction 1 = below proven bar for this sandbox
    session_r      = round(r_so_far + r_realized * size, 3)

    record = {
        "ts":                     datetime.now(timezone.utc).isoformat(),
        "instrument":             instrument,
        "trade_num_in_session":   trade_num,
        "bias_direction":         bias_dir,
        "bias_conviction":        bias_conv,
        "direction":              direction,
        "entry":                  entry,
        "stop":                   stop,
        "target_1":               t1,
        "target_2":               t2,
        "exit":                   exit_price,
        "exit_reason":            exit_reason,
        "r_realized":             r_realized,
        "size_contracts":         size,
        "sizing_rationale":       rationale,
        "bias_aligned":           bias_aligned,
        "below_proven_bar":       below_proven,
        "session_result_so_far_r": session_r,
        "notes":                  notes,
    }

    # Confirm
    r_sign = "+" if r_realized >= 0 else ""
    print(f"\n  ── Summary ──────────────────────────────")
    print(f"  {instrument}  {direction}  "
          f"entry={entry}  stop={stop}  exit={exit_price}")
    print(f"  R realized: {r_sign}{r_realized:.2f}  "
          f"(×{size} contracts = {r_sign}{r_realized*size:.2f}R)")
    print(f"  Bias aligned: {bias_aligned}  |  Below proven bar: {below_proven}")
    print(f"  Session R so far: {session_r:+.2f}R")
    if notes:
        print(f"  Notes: {notes}")

    confirm = input("\n  Log this trade? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("  Cancelled — nothing logged.")
        return

    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    all_trades = _load_trades()
    print(f"\n  Logged. Trade #{len(all_trades)} overall  |  {150 - len(all_trades)} to validation.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Futures trade logger")
    ap.add_argument("--list",    type=int, nargs="?", const=10, metavar="N",
                    help="Show last N trades (default 10)")
    ap.add_argument("--summary", action="store_true", help="Session P&L summary")
    args = ap.parse_args()

    if args.list is not None:
        _cmd_list(args.list)
    elif args.summary:
        _cmd_summary()
    else:
        _log_trade()


if __name__ == "__main__":
    main()
