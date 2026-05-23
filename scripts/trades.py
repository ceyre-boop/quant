#!/usr/bin/env python3
"""
SOVEREIGN trade log CLI.

Usage:
  python3 scripts/trades.py              # last 20 events
  python3 scripts/trades.py --n 50       # last N events
  python3 scripts/trades.py --open       # open positions only
  python3 scripts/trades.py --summary    # P&L summary
  python3 scripts/trades.py --tail       # live tail (poll every 2s)
  python3 scripts/trades.py --json       # raw JSON output
  python3 scripts/trades.py --source ICT # filter by source
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_PATH = Path("data/ledger/live_trade_log.jsonl")

# ── ANSI colours ─────────────────────────────────────────────────────────────
G = "\033[92m"   # green
R = "\033[91m"   # red
Y = "\033[93m"   # yellow
B = "\033[94m"   # blue
C = "\033[96m"   # cyan
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

TYPE_COLOUR = {
    "ENTRY": B + BOLD,
    "TP1":   Y,
    "TP2":   G + BOLD,
    "STOP":  R,
    "BE":    Y,
    "SESSION_CLOSE": DIM,
    "TV_ALERT": C,
    "EXIT":  DIM,
}

SOURCE_COLOUR = {
    "ICT": C,
    "FOREX": B,
    "TRADINGVIEW": Y,
}


def _read(n: int = 200) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:] if l.strip()]


def _fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return iso[:16]


def _fmt_r(r) -> str:
    if r is None:
        return "      "
    r = float(r)
    colour = G if r > 0 else (R if r < 0 else DIM)
    return f"{colour}{r:+.2f}R{RESET}"


def _fmt_price(p) -> str:
    try:
        return f"{float(p):.5g}"
    except Exception:
        return str(p)


def _print_event(e: dict, wide: bool = False) -> None:
    tc = TYPE_COLOUR.get(e.get("type", ""), "")
    sc = SOURCE_COLOUR.get(e.get("source", ""), DIM)
    etype = f"{tc}{e.get('type','?'):14}{RESET}"
    src   = f"{sc}{e.get('source','?'):12}{RESET}"
    ts    = f"{DIM}{_fmt_ts(e.get('ts',''))}{RESET}"
    dir_  = e.get("direction", "?")
    dir_c = G if dir_ == "LONG" else (R if dir_ == "SHORT" else DIM)
    dirstr = f"{dir_c}{dir_:5}{RESET}"
    ticker = f"{BOLD}{e.get('ticker','?'):8}{RESET}"
    price = f"{_fmt_price(e.get('price', 0)):>10}"
    r_str = _fmt_r(e.get("r_value"))
    grade = e.get("grade") or ""
    print(f"  {ts}  {etype}  {src}  {ticker}  {dirstr}  {price}  {r_str}  {DIM}{grade}{RESET}")


def cmd_list(args):
    events = _read(args.n)
    if args.source:
        events = [e for e in events if e.get("source", "").upper() == args.source.upper()]
    if not events:
        print(f"  {DIM}No events in {LOG_PATH}{RESET}")
        return
    print(f"\n  {BOLD}SOVEREIGN TRADE LOG{RESET}  {DIM}({len(events)} events){RESET}\n")
    print(f"  {DIM}{'TIME':14}  {'TYPE':14}  {'SOURCE':12}  {'TICKER':8}  {'DIR':5}  {'PRICE':>10}  {'R':>6}  GRADE{RESET}")
    print(f"  {DIM}{'─'*90}{RESET}")
    for e in events:
        _print_event(e)
    print()


def cmd_open(args):
    try:
        from sovereign.ledger.live_trade_log import LiveTradeLog
        positions = LiveTradeLog.open_positions()
    except Exception:
        # Fallback: manual scan
        events = _read(500)
        open_map: dict[str, dict] = {}
        close_types = {"TP1", "TP2", "STOP", "SESSION_CLOSE", "EXIT", "BE"}
        for e in events:
            key = e["ticker"]
            if e["type"] == "ENTRY":
                open_map[key] = e
            elif e["type"] in close_types and key in open_map:
                del open_map[key]
        positions = list(open_map.values())

    if not positions:
        print(f"\n  {DIM}No open positions{RESET}\n")
        return
    print(f"\n  {BOLD}OPEN POSITIONS{RESET}  {DIM}({len(positions)}){RESET}\n")
    for e in positions:
        meta = e.get("meta", {})
        tp1 = meta.get("tp1", "—")
        tp2 = meta.get("tp2", "—")
        stop = meta.get("stop", "—")
        dir_ = e.get("direction", "?")
        dir_c = G if dir_ == "LONG" else R
        print(f"  {BOLD}{e.get('ticker'):8}{RESET}  "
              f"{dir_c}{dir_:5}{RESET}  "
              f"entry={_fmt_price(e.get('price'))}  "
              f"stop={_fmt_price(stop)}  "
              f"tp1={_fmt_price(tp1)}  "
              f"tp2={_fmt_price(tp2)}  "
              f"{DIM}{_fmt_ts(e.get('ts',''))}{RESET}  "
              f"{C}{e.get('source','')}{RESET}")
    print()


def cmd_summary(args):
    events = _read(1000)
    if args.source:
        events = [e for e in events if e.get("source", "").upper() == args.source.upper()]

    entries = [e for e in events if e["type"] == "ENTRY"]
    closes  = [e for e in events if e["type"] in {"TP1", "TP2", "STOP", "BE", "SESSION_CLOSE", "EXIT"}
               and e.get("r_value") is not None]

    total_r = sum(float(e["r_value"]) for e in closes)
    wins    = [e for e in closes if float(e["r_value"]) > 0]
    losses  = [e for e in closes if float(e["r_value"]) < 0]
    wr      = len(wins) / len(closes) if closes else 0
    avg_r   = total_r / len(closes) if closes else 0

    by_type: dict[str, int] = {}
    for e in closes:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1

    print(f"\n  {BOLD}P&L SUMMARY{RESET}\n")
    print(f"  Entries:   {len(entries)}")
    print(f"  Closed:    {len(closes)}")
    if closes:
        wr_c = G if wr >= 0.4 else (Y if wr >= 0.3 else R)
        tr_c = G if total_r > 0 else R
        ar_c = G if avg_r > 0 else R
        print(f"  Win rate:  {wr_c}{wr:.1%}{RESET}")
        print(f"  Total R:   {tr_c}{total_r:+.2f}R{RESET}")
        print(f"  Avg R:     {ar_c}{avg_r:+.3f}R{RESET}")
        print(f"  By type:   " + "  ".join(f"{k}={v}" for k, v in by_type.items()))

    # TV alerts
    tv = [e for e in events if e.get("source") == "TRADINGVIEW"]
    if tv:
        print(f"\n  {C}TradingView alerts: {len(tv)}{RESET}")
        strategies = {}
        for e in tv:
            s = e.get("meta", {}).get("strategy", "unknown")
            strategies[s] = strategies.get(s, 0) + 1
        for s, n in strategies.items():
            print(f"    {s}: {n}")
    print()


def cmd_tail(args):
    print(f"\n  {BOLD}SOVEREIGN — Live tail{RESET}  {DIM}(Ctrl+C to stop){RESET}\n")
    seen = set()

    def get_key(e):
        return (e.get("ts"), e.get("type"), e.get("ticker"))

    while True:
        try:
            events = _read(50)
            for e in events:
                k = get_key(e)
                if k not in seen:
                    seen.add(k)
                    _print_event(e)
            time.sleep(2)
        except KeyboardInterrupt:
            print(f"\n  {DIM}stopped{RESET}\n")
            break


def main():
    p = argparse.ArgumentParser(description="SOVEREIGN trade log CLI")
    p.add_argument("--n",       type=int, default=20, help="number of events (default 20)")
    p.add_argument("--open",    action="store_true",  help="show open positions")
    p.add_argument("--summary", action="store_true",  help="P&L summary")
    p.add_argument("--tail",    action="store_true",  help="live tail mode")
    p.add_argument("--json",    action="store_true",  help="raw JSON output")
    p.add_argument("--source",  default="",           help="filter: ICT | FOREX | TRADINGVIEW")
    args = p.parse_args()

    if args.json:
        events = _read(args.n)
        print(json.dumps(events, indent=2))
        return
    if args.open:
        cmd_open(args)
    elif args.summary:
        cmd_summary(args)
    elif args.tail:
        cmd_tail(args)
    else:
        cmd_list(args)


if __name__ == "__main__":
    main()
