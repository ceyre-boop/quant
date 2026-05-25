#!/usr/bin/env python3
"""
sovereign/oracle/oracle_agent.py
Oracle Dashboard Agent — reads trade ledger + bridge state, generates
suggestions and prompt queue items, posts messages to Colin.

Called by agent_scheduler.py every cycle.
Never modifies live trading parameters.
Writes to: data/agent/suggestions.json, data/agent/prompt_queue.json,
           data/agent/messages_to_colin.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_AGENT = ROOT / "data" / "agent"
SUGGESTIONS_PATH = DATA_AGENT / "suggestions.json"
PQ_PATH = DATA_AGENT / "prompt_queue.json"
MESSAGES_PATH = DATA_AGENT / "messages_to_colin.json"
LEDGER_PATH = ROOT / "data" / "ledger"
FORENSICS_PATH = ROOT / "data" / "forensics" / "cross_system_state.json"

def _load_dotenv(env_path: Path = None):
    """Load .env file into os.environ if present."""
    candidates = [env_path, ROOT / ".env", Path.home() / ".env"] if env_path else [ROOT / ".env", Path.home() / ".env"]
    for p in candidates:
        if p and Path(p).exists():
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return


# Research tasks that are safe to auto-execute (no live trading param changes)
_AUTO_EXECUTE_SAFE = {"backtest", "signal_scan", "forensics", "hypothesis_test", "shell"}


def _load(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _next_sug_id() -> str:
    data = _load(SUGGESTIONS_PATH)
    items = data if isinstance(data, list) else data.get("suggestions", [])
    existing = [s["id"] for s in items if s["id"].startswith("SUG-")]
    n = len(existing) + 1
    while f"SUG-{n:03d}" in existing:
        n += 1
    return f"SUG-{n:03d}"


def _next_pq_id() -> str:
    data = _load(PQ_PATH)
    prompts = data.get("prompts", []) if isinstance(data, dict) else []
    existing = [p["id"] for p in prompts if p["id"].startswith("PQ-")]
    n = len(existing) + 1
    while f"PQ-{n:03d}" in existing:
        n += 1
    return f"PQ-{n:03d}"


def add_suggestion(
    title: str,
    detail: str,
    category: str = "RESEARCH",
    priority: str = "MEDIUM",
    script: str = "",
    auto_queue: bool = True,
) -> str:
    """Add a new suggestion. Returns its ID. If auto_queue=True, sets status=PENDING so the
    research_agent picks it up immediately on next scheduler cycle."""
    data = _load(SUGGESTIONS_PATH)
    items = data if isinstance(data, list) else data.get("suggestions", [])

    sug_id = _next_sug_id()
    sug = {
        "id": sug_id,
        "category": category,
        "priority": priority,
        "title": title,
        "detail": detail,
        "script": script,
        "status": "PENDING" if auto_queue else "QUEUED",
        "created": datetime.now().isoformat(),
        "auto_queue": auto_queue,
    }
    items.append(sug)

    if isinstance(data, list):
        _save(SUGGESTIONS_PATH, items)
    else:
        data["suggestions"] = items
        _save(SUGGESTIONS_PATH, data)

    # If auto_queue, trigger research_agent immediately (don't wait for scheduler)
    if auto_queue:
        _trigger_research_agent()

    return sug_id


def add_prompt_queue(
    prompt: str,
    reason: str,
    script: str = "",
    priority: str = "MEDIUM",
    auto_execute: bool = True,
) -> str:
    """Add a PQ item. auto_execute=True means research_agent runs script without human approval.
    Safety: items that touch live trading params should always set auto_execute=False."""
    data = _load(PQ_PATH)
    if not isinstance(data, dict):
        data = {"prompts": [], "stats": {}}
    prompts = data.setdefault("prompts", [])

    pq_id = _next_pq_id()
    pq = {
        "id": pq_id,
        "prompt": prompt,
        "reason": reason,
        "script": script,
        "priority": priority,
        "auto_execute": auto_execute,
        "status": "QUEUED",
        "queued_at": datetime.now().isoformat(),
        "sent_at": None,
    }
    prompts.append(pq)
    data["prompts"] = prompts
    _save(PQ_PATH, data)

    if auto_execute and script:
        _trigger_research_agent()

    return pq_id


def _trigger_research_agent() -> None:
    """Fire research_agent --check-suggestions in background so new items run immediately."""
    import subprocess
    agent = ROOT / "sovereign" / "agent" / "research_agent.py"
    if not agent.exists():
        return
    try:
        subprocess.Popen(
            [sys.executable, str(agent), "--check-suggestions"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass


def _post_message(priority: str, text: str) -> None:
    emoji = {"URGENT": "🔴", "IMPORTANT": "🟡", "FYI": "🟢"}.get(priority, "🟢")
    data = _load(MESSAGES_PATH)
    if not isinstance(data, dict):
        data = {}
    msgs = data.setdefault("messages", [])
    msgs.insert(0, {
        "id": f"msg-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "priority": priority,
        "emoji": emoji,
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "read": False,
    })
    data["messages"] = msgs[:50]
    _save(MESSAGES_PATH, data)


def run_dashboard_analysis(dry_run: bool = False) -> dict:
    """
    Main agent analysis: read system state, identify opportunities,
    write suggestions and messages. Returns summary dict.
    """
    findings = []

    # Load bridge state
    bridge = _load(FORENSICS_PATH)
    threat = bridge.get("library_threat_score", 0.0)
    ict_mode = bridge.get("ict_mode", "UNKNOWN")

    # Load recent ICT trades
    ict_trades = []
    for ict_file in [ROOT / "data" / "ledger" / "ict_paper_trades.json"]:
        if ict_file.exists():
            d = _load(ict_file)
            ict_trades = d if isinstance(d, list) else d.get("trades", [])

    # Load hypothesis ledger — check for stale TESTING items
    ledger = _load(DATA_AGENT / "hypothesis_ledger.json")
    hypotheses = ledger.get("hypotheses", ledger) if isinstance(ledger, dict) else ledger
    if isinstance(hypotheses, list):
        stale_testing = [
            h for h in hypotheses
            if h.get("status") in ("TESTING", "🔄 TESTING")
        ]
        if len(stale_testing) > 2:
            finding = f"{len(stale_testing)} hypotheses stuck in TESTING state"
            findings.append(finding)
            if not dry_run:
                _post_message("IMPORTANT",
                    f"⚠ Oracle: {finding}. Auto-queuing execution runs.")
                for h in stale_testing[:3]:
                    h_id = h.get("id", "?")
                    h_name = h.get("name", h.get("title", h_id))
                    add_suggestion(
                        title=f"Complete TESTING hypothesis: {h_name}",
                        detail=f"Hypothesis {h_id} has been in TESTING state. Run backtest to confirm or reject.",
                        category="RESEARCH",
                        priority="HIGH",
                        script=f"python3 -c \"print('placeholder test run for {h_id}')\"",
                        auto_queue=False,  # Don't chain-trigger, scheduler will pick up
                    )

    # Report bridge state
    if threat >= 0.85:
        findings.append(f"Bridge threat {threat:.2f}: {ict_mode}")
        if not dry_run:
            _post_message("IMPORTANT",
                f"🌉 Bridge: threat={threat:.2f} mode={ict_mode}. ICT entries {'BLOCKED' if ict_mode == 'HALT_NEW' else 'RESTRICTED'}.")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bridge_threat": threat,
        "ict_mode": ict_mode,
        "ict_trades_analyzed": len(ict_trades),
        "stale_hypotheses": len(stale_testing) if isinstance(hypotheses, list) else 0,
        "findings": findings,
        "dry_run": dry_run,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Oracle Dashboard Agent")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_dashboard_analysis(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
