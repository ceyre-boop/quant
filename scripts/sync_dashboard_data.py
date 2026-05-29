"""
sync_dashboard_data.py — refresh GitHub Pages static data files

Run this locally before committing + pushing to master.
Loads .env so API keys are visible, runs the real health check,
converts fills jsonl → json, and syncs oracle reflections.

Usage:
    python3 scripts/sync_dashboard_data.py
    # then: git add data/ && git commit && git push origin HEAD:master
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")  # MUST be before agent_scheduler import so os.environ is populated

from scripts.agent_scheduler import run_health_check  # noqa: E402  (after dotenv)


def _sync_health() -> dict:
    print("Running API health check (real connections)…")
    health = run_health_check()
    out = ROOT / "data" / "agent" / "health.json"
    out.write_text(json.dumps(health, indent=2))
    green = sum(1 for v in health["components"].values() if v["status"] == "GREEN")
    yellow = sum(1 for v in health["components"].values() if v["status"] == "YELLOW")
    red = sum(1 for v in health["components"].values() if v["status"] == "RED")
    print(f"  Health: {health['overall']} | GREEN={green} YELLOW={yellow} RED={red}")
    for k, v in health["components"].items():
        icon = "✓" if v["status"] == "GREEN" else ("!" if v["status"] == "YELLOW" else "✗")
        print(f"    {icon} {k:<25} {v['status']:<8} {v.get('detail','')[:60]}")
    return health


def _sync_fills() -> int:
    src = ROOT / "data" / "ledger" / "oanda_fills.jsonl"
    dst = ROOT / "data" / "ledger" / "oanda_fills.json"
    fills = []
    if src.exists():
        for line in src.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    fills.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    dst.write_text(json.dumps(fills, indent=2))
    print(f"  OANDA fills: {len(fills)} trades → data/ledger/oanda_fills.json")
    return len(fills)


def _sync_checklist(fill_count: int) -> None:
    path = ROOT / "data" / "agent" / "checklist_state.json"
    if not path.exists():
        return
    state = json.loads(path.read_text())
    for gate in state.get("gates", []):
        if gate.get("id") == "G2":
            needed = 30
            have = fill_count
            remaining = max(0, needed - have)
            gate["value"] = f"{have} London+GradeA trades logged"
            gate["status"] = "GREEN" if have >= needed else ("YELLOW" if have >= 10 else "RED")
            gate["detail"] = (
                f"Gate clear!" if have >= needed
                else f"Need {remaining} more — OANDA bridge live since 2026-05-28"
            )
    state["timestamp"] = datetime.now(timezone.utc).isoformat()
    green = sum(1 for g in state.get("gates", []) if g["status"] == "GREEN")
    state["gates_green"] = green
    state["gates_yellow"] = sum(1 for g in state.get("gates", []) if g["status"] == "YELLOW")
    state["gates_red"] = sum(1 for g in state.get("gates", []) if g["status"] == "RED")
    state["overall"] = "GO" if green == len(state.get("gates", [])) else "WAIT"
    path.write_text(json.dumps(state, indent=2))
    print(f"  Checklist: G2 updated to {fill_count}/30 trades")


def _sync_reflections() -> str | None:
    ref_dir = ROOT / "data" / "oracle" / "reflections"
    files = sorted(ref_dir.glob("20??-??-??.json"))
    if not files:
        return None
    latest = files[-1]
    print(f"  Latest reflection: {latest.name}")
    return latest.stem


def main() -> None:
    print(f"\n{'='*54}")
    print(f"Sovereign Dashboard Sync — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*54}\n")

    _sync_health()
    print()
    fill_count = _sync_fills()
    _sync_checklist(fill_count)
    ref_date = _sync_reflections()

    print(f"\n{'─'*54}")
    print("Files ready to commit:")
    print("  data/agent/health.json")
    print("  data/agent/checklist_state.json")
    print("  data/ledger/oanda_fills.json       (new)")
    if ref_date:
        print(f"  data/oracle/reflections/{ref_date}.json")
    print("\nNext:")
    print("  git add data/agent/health.json data/agent/checklist_state.json \\")
    print("          data/ledger/oanda_fills.json data/oracle/reflections/ \\")
    print("          scripts/sync_dashboard_data.py index.html")
    print("  git commit -m '[DASHBOARD] Surface live trading state'")
    print("  git push origin HEAD:master")


if __name__ == "__main__":
    main()
