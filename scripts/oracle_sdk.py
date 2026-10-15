#!/usr/bin/env python3
"""Oracle v2 (Claude Agent SDK) runner.

  oracle_sdk run [--budget 2.0] [--model claude-opus-5-5] [--mission FILE] [--tag daily]
  oracle_sdk tools            exercise the read-only tools locally (no API call)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.oracle import sdk_agent as sa  # noqa: E402


def cmd_run(a):
    mission = Path(a.mission).read_text() if a.mission else None
    r = sa.run(mission=mission, model=a.model, budget=a.budget, max_turns=a.max_turns, tag=a.tag)
    print(r["report"])
    print(f"\n--- cost ${r.get('cost_usd') or 0:.4f}  turns {r.get('turns')}  stop {r.get('stop_reason')}  error {r.get('is_error')}")
    if r.get("errors"):
        print(f"errors: {r['errors']}")


def cmd_tools(a):
    print(sa.t_thesis_list()[:1500]); print("...")
    print(sa.t_loop_health()[:600]); print("...")
    print(sa.t_recent_decisions(7)[:600])


def main():
    ap = argparse.ArgumentParser(prog="oracle_sdk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--budget", type=float, default=sa.DEFAULT_BUDGET)
    p.add_argument("--model", default=sa.MODEL); p.add_argument("--mission"); p.add_argument("--tag", default="daily")
    p.add_argument("--max-turns", type=int, default=40); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("tools"); p.set_defaults(fn=cmd_tools)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
