"""
scripts/oracle_prompt.py
Alta Investments — Oracle → Claude Code Prompt Queue

Oracle queues prompts for Colin to run in Claude Code.
Claude Code reads this queue at session start and acts on the top item.

ORACLE QUEUES:     Called automatically from oracle_agent.py
COLIN READS:       python3 scripts/oracle_prompt.py --next
COLIN DISMISSES:   python3 scripts/oracle_prompt.py --done PQ-001
ORACLE RULE:       Only queues prompts when usage budget is healthy (not DANGER mode)

.env format:
  ANTHROPIC_WEEKLY_LIMIT=50000   # optional, for budget awareness
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

ROOT       = Path(__file__).parent.parent
QUEUE_PATH = ROOT / "data" / "agent" / "prompt_queue.json"


def load() -> dict:
    try:
        return json.loads(QUEUE_PATH.read_text())
    except Exception:
        return {"prompts": [], "stats": {"total_queued": 0, "total_sent": 0}}


def save(data: dict):
    QUEUE_PATH.write_text(json.dumps(data, indent=2))


def queue_prompt(prompt: str, reason: str, priority: str = "MEDIUM") -> str:
    """Called by oracle_agent.py — adds a prompt to the queue."""
    data = load()
    total = data["stats"].get("total_queued", 0) + 1
    pid = f"PQ-{total:03d}"
    data["prompts"].append({
        "id": pid,
        "prompt": prompt,
        "reason": reason,
        "priority": priority,
        "status": "QUEUED",
        "queued_at": datetime.now().isoformat(),
        "sent_at": None,
    })
    data["stats"]["total_queued"] = total
    save(data)
    return pid


def cmd_next(args):
    data = load()
    queued = [p for p in data["prompts"] if p["status"] == "QUEUED"]
    # Sort: HIGH first, then MEDIUM, then LOW
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    queued.sort(key=lambda p: order.get(p.get("priority","MEDIUM"), 1))

    if not queued:
        print("No prompts queued. Oracle will add some next cycle.")
        return

    p = queued[0]
    print(f"\n{'─'*60}")
    print(f"  {p['id']} [{p['priority']}] — queued {p['queued_at'][:16]}")
    print(f"  WHY: {p['reason']}")
    print(f"{'─'*60}")
    print(f"\n{p['prompt']}\n")
    print(f"  Run: python3 scripts/oracle_prompt.py --done {p['id']}")
    print(f"  Skip: python3 scripts/oracle_prompt.py --skip {p['id']}\n")

    # Mark as sent
    for item in data["prompts"]:
        if item["id"] == p["id"]:
            item["sent_at"] = datetime.now().isoformat()
    data["stats"]["total_sent"] = data["stats"].get("total_sent", 0) + 1
    save(data)


def cmd_done(args):
    data = load()
    pid = args.done.upper()
    for p in data["prompts"]:
        if p["id"] == pid:
            p["status"] = "DONE"
            print(f"✅ {pid} marked done.")
            save(data)
            return
    print(f"{pid} not found.")


def cmd_skip(args):
    data = load()
    pid = args.skip.upper()
    for p in data["prompts"]:
        if p["id"] == pid:
            p["status"] = "SKIPPED"
            print(f"⏭ {pid} skipped.")
            save(data)
            return
    print(f"{pid} not found.")


def cmd_list(args):
    data = load()
    queued = [p for p in data["prompts"] if p["status"] == "QUEUED"]
    done   = [p for p in data["prompts"] if p["status"] == "DONE"]
    print(f"\n  Prompt queue: {len(queued)} pending | {len(done)} done\n")
    for p in queued:
        pri = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(p["priority"],"○")
        print(f"  {pri} {p['id']} — {p['reason'][:70]}")
    if not queued:
        print("  No prompts queued.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle prompt queue")
    parser.add_argument("--next",  action="store_true", help="Show next prompt to run")
    parser.add_argument("--done",  metavar="ID",        help="Mark prompt done")
    parser.add_argument("--skip",  metavar="ID",        help="Skip prompt")
    parser.add_argument("--list",  action="store_true", help="List all pending")
    args = parser.parse_args()

    if args.done:   cmd_done(args)
    elif args.skip: cmd_skip(args)
    elif args.list: cmd_list(args)
    else:           cmd_next(args)
