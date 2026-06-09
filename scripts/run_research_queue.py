#!/usr/bin/env python3
"""
Run the WHOLE research queue — drain every QUEUED task in data/agent/research_queue.json.

Wraps sovereign.agent.research_agent.run_next_task() in a loop so the dashboard's
"▶ Run Research Queue" button can clear the backlog in one call. Each task picks the
highest-priority QUEUED item; we stop when the agent reports IDLE (no tasks) or after a
safety cap so a runaway never loops forever.

Usage:
  python3 scripts/run_research_queue.py            # drain up to 25 tasks
  python3 scripts/run_research_queue.py --max 5    # cap at 5
  python3 scripts/run_research_queue.py --dry-run  # show picks, run nothing
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from sovereign.agent.research_agent import run_next_task


def main() -> int:
    ap = argparse.ArgumentParser(description="Drain the research queue")
    ap.add_argument("--max", type=int, default=25, help="max tasks to run (safety cap)")
    ap.add_argument("--dry-run", action="store_true", help="print picks, execute nothing")
    args = ap.parse_args()

    ran = 0
    for i in range(args.max):
        result = run_next_task(dry_run=args.dry_run)
        print(f"[queue {i + 1}] {result}", flush=True)
        if not result or result.startswith("IDLE"):
            break
        ran += 1
    print(f"[queue] done — {ran} task(s) executed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
