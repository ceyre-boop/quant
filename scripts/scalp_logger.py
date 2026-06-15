#!/usr/bin/env python3
"""Discretionary ES/NQ scalping journal — fast trade logger.

PAPER ONLY. This has ZERO connection to live execution. Colin enters trades manually
(after taking them on OANDA paper) so he can track and develop his discretionary
scalping. It is NOT a signal generator and NOT read by the Oracle for systematic
learning — it lives in its own file, isolated from data/decision_logs/.

Storage: data/futures/scalp_log.jsonl  (one JSON record per line)

Usage (fast — log an open in one line while watching the chart):
  python3 scripts/scalp_logger.py open --instrument MES --direction LONG \
    --entry 5234.50 --stop 5231.00 --target 5240.00 --size 1 \
    --thesis "VWAP reclaim with volume confirmation" \
    --setup vwap_reclaim --session open [--emotional-state confident]

  python3 scripts/scalp_logger.py close --trade-id 17 --exit 5239.25 \
    --outcome MANUAL_EXIT --notes "Took early on momentum stall"

  python3 scripts/scalp_logger.py list --today
  python3 scripts/scalp_logger.py list --week
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.utils.timestamps import canonical_timestamp

SCALP_LOG = ROOT / "data" / "futures" / "scalp_log.jsonl"

# Dollars per point. Micros per the spec; full-size at standard CME multipliers.
POINT_VALUE = {"MES": 5.0, "MNQ": 2.0, "ES": 50.0, "NQ": 20.0}

SETUPS = ["open_range_break", "vwap_reclaim", "trend_pullback", "level_break",
          "reversal", "other"]
SESSIONS = ["premarket", "open", "midday", "close", "overnight"]
EMOTIONS = ["calm", "rushed", "uncertain", "confident"]
OUTCOMES = ["TARGET", "STOP", "MANUAL_EXIT", "OPEN"]


def _read_all() -> list[dict]:
    if not SCALP_LOG.exists():
        return []
    rows = []
    for line in SCALP_LOG.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append(record: dict) -> None:
    SCALP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALP_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _rewrite(rows: list[dict]) -> None:
    """Atomic rewrite (temp + replace) — used on close so a crash never truncates the log."""
    import os, tempfile
    SCALP_LOG.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(SCALP_LOG.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, SCALP_LOG)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _next_trade_id(rows: list[dict]) -> int:
    return (max((int(r.get("trade_id", 0)) for r in rows), default=0)) + 1


def _compute_r(entry: float, stop: float, exit_price: float, direction: str) -> float:
    """1R = entry→stop distance. Sign-correct for both directions."""
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    raw = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    return round(raw / risk, 3)


def _compute_pnl(entry: float, exit_price: float, size: int, direction: str, instrument: str) -> float:
    pv = POINT_VALUE.get(instrument.upper())
    if pv is None:
        raise SystemExit(f"unknown instrument '{instrument}' — expected one of {list(POINT_VALUE)}")
    pts = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    return round(pts * pv * size, 2)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_open(args) -> None:
    rows = _read_all()
    tid = _next_trade_id(rows)
    rec = {
        "trade_id": tid,
        "timestamp_entry": canonical_timestamp(),
        "timestamp_exit": None,
        "instrument": args.instrument.upper(),
        "direction": args.direction.upper(),
        "entry_price": args.entry,
        "stop_price": args.stop,
        "target_price": args.target,
        "size_contracts": args.size,
        "exit_price": None,
        "outcome": "OPEN",
        "R_realized": None,
        "pnl_dollars": None,
        "thesis": args.thesis,
        "setup_type": args.setup,
        "session": args.session,
        "emotional_state": args.emotional_state,
        "notes_post": None,
    }
    _append(rec)
    tgt = f" target {rec['target_price']}" if rec["target_price"] is not None else ""
    print(f"✓ OPEN #{tid}  {rec['instrument']} {rec['direction']} @ {rec['entry_price']} "
          f"stop {rec['stop_price']}{tgt}  [{rec['setup_type']}/{rec['session']}]")


def cmd_close(args) -> None:
    rows = _read_all()
    match = next((r for r in rows if int(r.get("trade_id", -1)) == args.trade_id), None)
    if match is None:
        raise SystemExit(f"trade_id {args.trade_id} not found in {SCALP_LOG.name}")
    if match.get("outcome") != "OPEN":
        raise SystemExit(f"trade_id {args.trade_id} is not OPEN (outcome={match.get('outcome')}) "
                         "— cannot close it again")
    match["timestamp_exit"] = canonical_timestamp()
    match["exit_price"] = args.exit
    match["outcome"] = args.outcome
    match["R_realized"] = _compute_r(match["entry_price"], match["stop_price"],
                                     args.exit, match["direction"])
    match["pnl_dollars"] = _compute_pnl(match["entry_price"], args.exit,
                                        match["size_contracts"], match["direction"],
                                        match["instrument"])
    if args.notes:
        match["notes_post"] = args.notes
    _rewrite(rows)
    sign = "✓" if match["R_realized"] >= 0 else "✗"
    print(f"{sign} CLOSE #{args.trade_id}  {match['instrument']} {match['direction']}  "
          f"{match['R_realized']:+.2f}R  ${match['pnl_dollars']:+.2f}  ({args.outcome})")


def _fmt_row(r: dict) -> str:
    rr = r.get("R_realized")
    pnl = r.get("pnl_dollars")
    rr_s = f"{rr:+.2f}R" if isinstance(rr, (int, float)) else "  open"
    pnl_s = f"${pnl:+.2f}" if isinstance(pnl, (int, float)) else "      —"
    mark = "·" if r.get("outcome") == "OPEN" else ("✓" if (rr or 0) >= 0 else "✗")
    t = (r.get("timestamp_entry") or "")[:16].replace("T", " ")
    return (f"  {mark} #{r.get('trade_id'):<3} {t}  {r.get('instrument'):<3} "
            f"{r.get('direction'):<5} {rr_s:>8} {pnl_s:>9}  "
            f"{r.get('setup_type','')}/{r.get('session','')}")


def cmd_list(args) -> None:
    rows = _read_all()
    now = datetime.now(timezone.utc)
    if args.week:
        cutoff = now - timedelta(days=7)
        label = "this week"
    else:  # default --today
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        label = "today"

    def _entry_dt(r):
        try:
            return datetime.fromisoformat(str(r.get("timestamp_entry")).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    sel = [r for r in rows if (_entry_dt(r) and _entry_dt(r) >= cutoff)]
    if not sel:
        print(f"No scalp trades {label}.")
        return
    closed = [r for r in sel if r.get("outcome") != "OPEN"]
    net_r = round(sum(r.get("R_realized") or 0 for r in closed), 2)
    net_pnl = round(sum(r.get("pnl_dollars") or 0 for r in closed), 2)
    wins = sum(1 for r in closed if (r.get("R_realized") or 0) > 0)
    wr = round(100 * wins / len(closed)) if closed else 0
    print(f"SCALP JOURNAL — {label}: {len(sel)} trade(s) "
          f"({len(closed)} closed)  {net_r:+.2f}R  ${net_pnl:+.2f}  WR {wr}%")
    for r in sorted(sel, key=lambda x: x.get("timestamp_entry") or ""):
        print(_fmt_row(r))


def main() -> None:
    ap = argparse.ArgumentParser(prog="scalp_logger",
                                 description="Discretionary ES/NQ scalping journal (paper only).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("open", help="log a new open scalp trade")
    o.add_argument("--instrument", required=True, choices=["MES", "MNQ", "ES", "NQ"])
    o.add_argument("--direction", required=True, choices=["LONG", "SHORT"])
    o.add_argument("--entry", type=float, required=True)
    o.add_argument("--stop", type=float, required=True)
    o.add_argument("--target", type=float, default=None)
    o.add_argument("--size", type=int, required=True)
    o.add_argument("--thesis", required=True, help="your reason for the trade")
    o.add_argument("--setup", required=True, choices=SETUPS)
    o.add_argument("--session", required=True, choices=SESSIONS)
    o.add_argument("--emotional-state", dest="emotional_state", default=None, choices=EMOTIONS)
    o.set_defaults(fn=cmd_open)

    c = sub.add_parser("close", help="close an open scalp trade")
    c.add_argument("--trade-id", dest="trade_id", type=int, required=True)
    c.add_argument("--exit", type=float, required=True)
    c.add_argument("--outcome", required=True, choices=["TARGET", "STOP", "MANUAL_EXIT"])
    c.add_argument("--notes", default=None, help="post-trade reflection")
    c.set_defaults(fn=cmd_close)

    l = sub.add_parser("list", help="list recent trades")
    g = l.add_mutually_exclusive_group()
    g.add_argument("--today", action="store_true", help="today's trades (default)")
    g.add_argument("--week", action="store_true", help="last 7 days")
    l.set_defaults(fn=cmd_list)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
