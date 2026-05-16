"""
scripts/suggestions.py
Alta Investments — Sovereign Suggestions CLI

Action Oracle suggestions from the terminal.

USAGE:
  python3 scripts/suggestions.py --list
  python3 scripts/suggestions.py --implement SUG-001
  python3 scripts/suggestions.py --veto SUG-001
  python3 scripts/suggestions.py --veto SUG-001 --reason "already tried, edge decayed"
"""

import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

ROOT            = Path(__file__).parent.parent
SUGGESTIONS_PATH = ROOT / "data" / "agent" / "suggestions.json"
DASH_SYNC       = ROOT / "ict-dashboard" / "data" / "agent" / "suggestions.json"


def load() -> dict:
    try:
        return json.loads(SUGGESTIONS_PATH.read_text())
    except Exception:
        return {"suggestions": [], "vetoed_ids": [], "stats": {"total_generated": 0, "implemented": 0, "vetoed": 0}}


def save(data: dict):
    SUGGESTIONS_PATH.write_text(json.dumps(data, indent=2))
    # Sync to dashboard
    DASH_SYNC.parent.mkdir(parents=True, exist_ok=True)
    DASH_SYNC.write_text(json.dumps(data, indent=2))
    _git_push()


def _git_push():
    try:
        subprocess.run(["git", "add", "data/agent/suggestions.json", "ict-dashboard/data/agent/suggestions.json"],
                       cwd=str(ROOT), capture_output=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT))
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", f"Suggestions update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
                           cwd=str(ROOT), capture_output=True, check=True)
            subprocess.run(["git", "push"], cwd=str(ROOT), capture_output=True, check=True)
            print("Pushed to GitHub — dashboard updated.")
    except Exception as e:
        print(f"Warning: git push failed ({e}). Changes saved locally.")


def cmd_list(args):
    data = load()
    suggestions = data.get("suggestions", [])
    new = [s for s in suggestions if s["status"] == "NEW"]
    done = [s for s in suggestions if s["status"] != "NEW"]

    if not suggestions:
        print("No suggestions yet. Oracle generates them automatically.")
        return

    if new:
        print(f"\n{'─'*60}")
        print(f"  OPEN SUGGESTIONS ({len(new)})")
        print(f"{'─'*60}")
        for s in new:
            pri = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(s.get("priority",""), "○")
            print(f"\n  {pri} {s['id']} [{s.get('category','?')}]")
            print(f"  {s['title']}")
            print(f"  {s.get('detail','')}")
            print(f"  → {s.get('action','')}")
            print(f"  python3 scripts/suggestions.py --implement {s['id']}")
            print(f"  python3 scripts/suggestions.py --veto {s['id']}")

    if done:
        print(f"\n  ACTIONED ({len(done)} total)")
        for s in done[-5:]:
            icon = "✅" if s["status"] == "IMPLEMENTED" else "❌"
            print(f"  {icon} {s['id']} — {s['title'][:60]}")

    stats = data.get("stats", {})
    print(f"\n  Total: {stats.get('total_generated',0)} generated | "
          f"{stats.get('implemented',0)} implemented | {stats.get('vetoed',0)} vetoed\n")


def cmd_implement(args):
    data = load()
    sug_id = args.implement.upper()
    found = False
    for s in data["suggestions"]:
        if s["id"] == sug_id:
            if s["status"] != "NEW":
                print(f"{sug_id} is already {s['status']}.")
                return
            s["status"] = "IMPLEMENTED"
            s["implemented_at"] = datetime.now().isoformat()
            data["stats"]["implemented"] = data["stats"].get("implemented", 0) + 1
            found = True
            print(f"✅ Marked {sug_id} as IMPLEMENTED: {s['title']}")
            break
    if not found:
        print(f"Suggestion {sug_id} not found.")
        return
    save(data)


def cmd_veto(args):
    data = load()
    sug_id = args.veto.upper()
    reason = getattr(args, "reason", "") or ""
    found = False
    for s in data["suggestions"]:
        if s["id"] == sug_id:
            if s["status"] != "NEW":
                print(f"{sug_id} is already {s['status']}.")
                return
            s["status"] = "VETOED"
            s["vetoed_at"] = datetime.now().isoformat()
            s["veto_reason"] = reason
            data["stats"]["vetoed"] = data["stats"].get("vetoed", 0) + 1
            if sug_id not in data.get("vetoed_ids", []):
                data.setdefault("vetoed_ids", []).append(sug_id)
            found = True
            print(f"❌ Vetoed {sug_id}: {s['title']}")
            if reason:
                print(f"   Reason: {reason}")
            break
    if not found:
        print(f"Suggestion {sug_id} not found.")
        return
    save(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle suggestions CLI")
    parser.add_argument("--list",       action="store_true",    help="List all suggestions")
    parser.add_argument("--implement",  metavar="ID",           help="Mark suggestion as implemented")
    parser.add_argument("--veto",       metavar="ID",           help="Veto a suggestion")
    parser.add_argument("--reason",     metavar="TEXT",         help="Reason for veto (optional)")
    args = parser.parse_args()

    if args.implement:
        cmd_implement(args)
    elif args.veto:
        cmd_veto(args)
    else:
        cmd_list(args)
