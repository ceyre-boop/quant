#!/usr/bin/env python3
"""
alta — operator CLI for the system kill switch (master freeze for the live trading path).

    python3 scripts/alta.py freeze "reason" [--hard]   # freeze trading path (+ config if --hard)
    python3 scripts/alta.py thaw                        # resume
    python3 scripts/alta.py status                      # show state + what's blocked

soft freeze blocks the trading/signal path (forex_live_scan placement, DecisionChain).
hard freeze ALSO blocks approve_edge.py (live-config mutation).
Monitoring (pulse_check/loop_health) and Oracle cognition keep running either way.

Tip: add a shell alias so it works anywhere — `alias alta='python3 ~/quant/scripts/alta.py'`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.utils import kill_switch as ks


def cmd_freeze(args) -> None:
    p = ks.freeze(args.reason, hard=args.hard, by=args.by)
    blocks = "trading/signal path" + (" + approve_edge (live config)" if p["mode"] == "hard" else "")
    print(f"🧊 SYSTEM FROZEN ({p['mode']}) — {p['reason']}")
    print(f"   at {p['frozen_at']} by {p['by']}")
    print(f"   blocks: {blocks}")
    print(f"   still running: pulse_check/loop_health (monitoring), Oracle reflect/briefing (cognition)")
    print(f"   file: {ks.KILL_SWITCH}")
    print(f"   thaw with: python3 scripts/alta.py thaw")


def cmd_thaw(args) -> None:
    prior = ks.thaw(by=args.by)
    if prior:
        print(f"☀️  THAWED — was {prior.get('mode')} freeze ({prior.get('reason', '')})")
        print("   trading path resumes on the next scheduled cycle.")
    else:
        print("Not frozen — nothing to thaw.")


def cmd_status(args) -> None:
    s = ks.state()
    if not s:
        print("🟢 RUNNING — no freeze active. Trading path live.")
        return
    print(f"🧊 FROZEN ({s.get('mode')}) — {s.get('reason', '')}")
    print(f"   since {s.get('frozen_at')} by {s.get('by')}")
    print(f"   blocked: trading/signal path (forex_live_scan placement, DecisionChain.evaluate)")
    if s.get("mode") == "hard":
        print(f"   blocked: approve_edge.py (live-config mutation)")
    print(f"   running: pulse_check/loop_health (monitoring), Oracle reflect/briefing (cognition)")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="alta", description="System kill switch — master freeze for the live trading path.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="freeze the trading path")
    f.add_argument("reason", help="why (logged to data/agent/param_change_log.jsonl)")
    f.add_argument("--hard", action="store_true",
                   help="also block approve_edge.py (live-config mutation)")
    f.add_argument("--by", default="colin", help="who issued the freeze (default: colin)")
    f.set_defaults(fn=cmd_freeze)

    t = sub.add_parser("thaw", help="remove the freeze")
    t.add_argument("--by", default="colin")
    t.set_defaults(fn=cmd_thaw)

    s = sub.add_parser("status", help="show freeze state")
    s.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
