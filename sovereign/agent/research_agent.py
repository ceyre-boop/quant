#!/usr/bin/env python3
"""
sovereign/agent/research_agent.py
Oracle Research Agent — autonomous task executor.

Called by agent_scheduler.py as a subprocess:
  python3 sovereign/agent/research_agent.py --task RQ-NNN

Also handles:
  --dry-run       : print what would run, don't execute
  --check-suggestions : scan suggestions.json for PENDING items and queue them
  --next          : pick + run next QUEUED task from research_queue.json

Design: no Claude API calls. Pure local Python. All findings written to
data/agent/findings.jsonl and hypothesis_ledger.json.
"""

import argparse
import json
import shlex
import subprocess
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_AGENT = ROOT / "data" / "agent"
QUEUE_PATH = DATA_AGENT / "research_queue.json"
FINDINGS_PATH = DATA_AGENT / "findings.jsonl"
LEDGER_PATH = DATA_AGENT / "hypothesis_ledger.json"
SUGGESTIONS_PATH = DATA_AGENT / "suggestions.json"
MESSAGES_PATH = DATA_AGENT / "messages_to_colin.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AGENT] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("research_agent")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load(path: Path) -> dict | list:
    if not path.exists():
        return {} if path.suffix == ".json" else []
    with open(path) as f:
        return json.load(f)


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _append_finding(finding: dict) -> None:
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINDINGS_PATH, "a") as f:
        f.write(json.dumps(finding, default=str) + "\n")


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


# ── Ledger helpers ────────────────────────────────────────────────────────────

def _ledger_update(hyp_id: str, status: str, result_summary: str) -> None:
    """Update hypothesis status in hypothesis_ledger.json."""
    if not hyp_id:
        return
    ledger = _load(LEDGER_PATH)
    hypotheses = ledger.get("hypotheses", ledger) if isinstance(ledger, dict) else ledger
    if isinstance(hypotheses, list):
        for h in hypotheses:
            if h.get("id") == hyp_id:
                h["status"] = status
                h["last_result"] = result_summary
                h["updated_at"] = datetime.now().isoformat(timespec="seconds")
                break
        if isinstance(ledger, dict):
            ledger["hypotheses"] = hypotheses
            _save(LEDGER_PATH, ledger)
        else:
            _save(LEDGER_PATH, hypotheses)


# ── Queue helpers ──────────────────────────────────────────────────────────────

def _get_task_by_id(task_id: str) -> dict | None:
    data = _load(QUEUE_PATH)
    queue = data.get("queue", []) if isinstance(data, dict) else []
    for t in queue:
        if t["id"] == task_id:
            return t
    return None


def _mark_task(task_id: str, status: str, result: str = "") -> None:
    data = _load(QUEUE_PATH)
    if not isinstance(data, dict):
        return
    for t in data.get("queue", []):
        if t["id"] == task_id:
            t["status"] = status
            if result:
                t["last_result"] = result
            t["updated_at"] = datetime.now().isoformat(timespec="seconds")
            break
    _save(QUEUE_PATH, data)


def _add_queue_task(task: dict) -> str:
    """Add a new task to research_queue.json. Returns assigned ID."""
    data = _load(QUEUE_PATH)
    if not isinstance(data, dict):
        data = {"queue": []}
    queue = data.setdefault("queue", [])
    existing_ids = {t["id"] for t in queue}
    # Auto-assign ID
    if "id" not in task or task["id"] in existing_ids:
        n = len([t for t in queue if t["id"].startswith("RQ-")]) + 1
        task["id"] = f"RQ-{n:03d}"
    task.setdefault("status", "QUEUED")
    task.setdefault("priority", 99)
    task.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    queue.append(task)
    _save(QUEUE_PATH, data)
    return task["id"]


# ── Task dispatchers ──────────────────────────────────────────────────────────

def _run_script(script_path: str, timeout: int = 3600) -> tuple[bool, str]:
    """Run a shell command/script. Returns (ok, output).

    A bare Python script (e.g. "scripts/foo.py [args]") is run through the active
    interpreter, NOT the shell: `sh -c "scripts/foo.py"` fails with "Permission denied"
    because the .py file isn't an executable. Real shell commands (e.g. "echo no-op")
    and explicit argv lists are passed through unchanged.
    """
    try:
        if isinstance(script_path, list):
            cmd, use_shell = script_path, False
        else:
            parts = shlex.split(script_path)
            if parts and parts[0].endswith('.py'):
                cmd, use_shell = [sys.executable, *parts], False
            else:
                cmd, use_shell = script_path, True
        result = subprocess.run(
            cmd, shell=use_shell,
            capture_output=True, text=True,
            timeout=timeout, cwd=str(ROOT),
        )
        out = (result.stdout + result.stderr).strip()
        return result.returncode == 0, out[-2000:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: exceeded limit"
    except Exception as e:
        return False, f"EXCEPTION: {e}"


def _dispatch_backtest(task: dict, dry_run: bool) -> str:
    script = task.get("script")
    if not script:
        return "ERROR: no script field in task"
    if dry_run:
        log.info(f"[DRY RUN] would run: {script}")
        return f"DRY RUN: {script}"
    log.info(f"Running backtest: {script}")
    ok, out = _run_script(script)
    return ("OK: " if ok else "ERROR: ") + out[-500:]


def _dispatch_signal_scan(task: dict, dry_run: bool) -> str:
    pairs = task.get("pairs", ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "AUDNZD=X", "USDJPY=X"])
    if dry_run:
        return f"DRY RUN: signal scan on {pairs}"
    try:
        import yfinance as yf
        import pandas as pd
        sys.path.insert(0, str(ROOT))
        from sovereign.forex.signal_engine import build_signal_frame

        results = []
        for ticker in pairs:
            prices = yf.Ticker(ticker).history(period="1y", interval="1d")
            if prices.empty:
                continue
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            base = ticker.replace("=X", "")[:3]
            quote = ticker.replace("=X", "")[3:]
            df = build_signal_frame(ticker, prices, base, quote)
            last = df.iloc[-1]
            results.append(f"{ticker}: signal={int(last['signal'])} conviction={float(last.get('conviction', 0)):.3f}")
        return "OK: " + " | ".join(results)
    except Exception as e:
        return f"ERROR: {e}"


def _dispatch_forensics(task: dict, dry_run: bool) -> str:
    if dry_run:
        return "DRY RUN: unified forensics"
    ok, out = _run_script([sys.executable, str(ROOT / "sovereign" / "research" / "unified_forensics.py")])
    return ("OK: " if ok else "ERROR: ") + out[-500:]


def _dispatch_hypothesis_test(task: dict, dry_run: bool) -> str:
    # Generic: run whatever script is in task["script"]
    return _dispatch_backtest(task, dry_run)


def _dispatch_shell(task: dict, dry_run: bool) -> str:
    script = task.get("script", "echo no-op")
    if dry_run:
        return f"DRY RUN: {script}"
    ok, out = _run_script(script)
    return ("OK: " if ok else "ERROR: ") + out[-500:]


DISPATCHERS = {
    "backtest":         _dispatch_backtest,
    "signal_scan":      _dispatch_signal_scan,
    "forensics":        _dispatch_forensics,
    "hypothesis_test":  _dispatch_hypothesis_test,
    "shell":            _dispatch_shell,
}


# ── Decision log health monitor ───────────────────────────────────────────────

DECISION_LOG_DIR = ROOT / "data" / "decision_logs"


def check_decision_log_health() -> list[str]:
    """
    Inspect decision logs for staleness, missing outcomes, and schema gaps.
    Returns a list of issue strings — empty list means healthy.
    Called by agent_scheduler's run_health_check() every 2 hours.
    """
    issues = []
    import time
    now = time.time()

    # 1. Log file freshness — check ONLY the CURRENT-month file (the one the logger writes to;
    #    see decision_logger._log_path()). Previous-month files are stale BY DESIGN once the month
    #    ends — globbing the last 2 files flagged decisions_<prev-month>.jsonl forever (false alarm).
    if DECISION_LOG_DIR.exists():
        month = datetime.now(timezone.utc).strftime("%Y_%m")
        current = DECISION_LOG_DIR / f"decisions_{month}.jsonl"
        if not current.exists():
            issues.append(f"WARNING: {current.name} missing — no decisions logged this month yet")
        else:
            age_h = (now - current.stat().st_mtime) / 3600
            if age_h > 96:
                scanner_note = ""
                try:
                    from pathlib import Path as _P
                    ss = _P(__file__).resolve().parents[2] / "data" / "scanner_state.json"
                    if ss.exists():
                        scan_age_h = (now - ss.stat().st_mtime) / 3600
                        scanner_note = f" (scanner last run {scan_age_h:.1f}h ago)"
                except Exception:
                    pass
                issues.append(f"WARNING: {current.name} not updated in {age_h:.0f}h — no Grade A signals{scanner_note}")
    else:
        issues.append("WARNING: data/decision_logs/ directory missing — no trades logged yet")
        return issues

    # 2. Load records from last 30 days
    from datetime import timedelta
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_close = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    all_records: list[dict] = []
    for log_file in sorted(DECISION_LOG_DIR.glob("decisions_*.jsonl")):
        try:
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ts = (rec.get("entry_timestamp") or "")[:10]
                if ts >= cutoff_date:
                    all_records.append(rec)
        except Exception:
            continue

    if not all_records:
        return issues  # no data yet — not an error

    # 3. Outcome backfill — entries older than 7 days should be closed
    stale_open = [
        r for r in all_records
        if not r.get("outcome")
        and (r.get("entry_timestamp") or "")[:10] < cutoff_close
    ]
    if stale_open:
        pairs = ", ".join(r.get("pair", "?") for r in stale_open[:3])
        issues.append(
            f"WARNING: {len(stale_open)} trades >7d old missing outcome backfill ({pairs}...)"
        )

    # 4. Required field completeness
    required = {"why_this_trade", "why_this_size", "pair", "direction", "entry_level"}
    for rec in all_records[-20:]:
        missing_fields = {k for k in required if not rec.get(k)}
        if missing_fields:
            issues.append(
                f"DATA QUALITY: {rec.get('pair', '?')} "
                f"{(rec.get('entry_timestamp') or '')[:10]} "
                f"missing {missing_fields}"
            )

    # 5. Schema completeness — every record must have entry_timestamp
    no_ts = [r for r in all_records if not r.get("entry_timestamp")]
    if no_ts:
        issues.append(f"SCHEMA: {len(no_ts)} records missing entry_timestamp — logger may be broken")

    return issues


# ── Main task runner ──────────────────────────────────────────────────────────

def run_task(task_id: str, dry_run: bool = False) -> str:
    task = _get_task_by_id(task_id)
    if not task:
        return f"ERROR: task {task_id} not found in queue"

    task_type = task.get("task_type", task.get("type", "shell"))
    dispatcher = DISPATCHERS.get(task_type, _dispatch_shell)

    log.info(f"Running task {task_id}: {task.get('name', '?')} (type={task_type})")
    _mark_task(task_id, "RUNNING")

    result = dispatcher(task, dry_run)
    ok = result.startswith("OK")
    status = "DONE" if ok else "ERROR"

    _mark_task(task_id, status, result)

    # Update hypothesis ledger if linked
    hyp_id = task.get("hypothesis_id")
    if hyp_id:
        ledger_status = "CONFIRMED" if ok else "REJECTED"
        _ledger_update(hyp_id, ledger_status, result[:300])

    # Record finding
    _append_finding({
        "id": f"F-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "task_id": task_id,
        "task_name": task.get("name", "?"),
        "task_type": task_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "result": result,
        "ok": ok,
    })

    priority = "FYI" if ok else "IMPORTANT"
    _post_message(priority, f"[Oracle] {task.get('name', task_id)}: {result[:200]}")

    log.info(f"Summary: {result[:200]}")
    return result


def run_next_task(mode: str = "FULL", dry_run: bool = False) -> str:
    """Pick highest-priority QUEUED task and run it."""
    data = _load(QUEUE_PATH)
    queue = data.get("queue", []) if isinstance(data, dict) else []
    queued = [t for t in queue if t.get("status") == "QUEUED"]
    if not queued:
        log.info("No QUEUED tasks available")
        return "IDLE: no tasks"
    task = sorted(queued, key=lambda t: t.get("priority", 99))[0]
    return run_task(task["id"], dry_run=dry_run)


# ── Suggestion watcher ────────────────────────────────────────────────────────

def check_suggestions(dry_run: bool = False) -> int:
    """
    Scan suggestions.json for PENDING items. For each one:
    - Convert to a research_queue task
    - Mark suggestion as QUEUED
    - Immediately run it (the autonomous loop)

    Returns count of suggestions processed.
    """
    data = _load(SUGGESTIONS_PATH)
    suggestions = data if isinstance(data, list) else data.get("suggestions", [])

    processed = 0
    for sug in suggestions:
        if sug.get("status") != "PENDING":
            continue

        log.info(f"Auto-queuing suggestion {sug['id']}: {sug.get('title', sug.get('description', '?'))[:80]}")

        # Build queue task from suggestion
        task = {
            "name": sug.get("title", sug.get("description", sug["id"]))[:100],
            "description": sug.get("detail", sug.get("description", "")),
            "task_type": _suggestion_type(sug),
            "script": sug.get("script") or sug.get("action", "echo no-op"),
            "source": "oracle_suggestion",
            "suggestion_id": sug["id"],
            "needs_network": False,
            "estimated_minutes": 15,
            "priority": {"HIGH": 5, "MEDIUM": 10, "LOW": 20}.get(sug.get("priority", "LOW"), 15),
        }

        if not dry_run:
            task_id = _add_queue_task(task)
            # Mark suggestion as QUEUED immediately
            sug["status"] = "QUEUED"
            sug["queued_at"] = datetime.now().isoformat(timespec="seconds")
            sug["queue_task_id"] = task_id

            # Run it right now — this is the "millisecond" wiring
            result = run_task(task_id, dry_run=False)
            sug["status"] = "DONE" if result.startswith("OK") else "ERROR"
            sug["result"] = result[:300]
            log.info(f"Suggestion {sug['id']} auto-ran: {result[:100]}")
        else:
            log.info(f"[DRY RUN] would queue and run: {task['name']}")

        processed += 1

    # Save updated suggestions
    if not dry_run and processed > 0:
        if isinstance(data, list):
            _save(SUGGESTIONS_PATH, suggestions)
        else:
            data["suggestions"] = suggestions
            _save(SUGGESTIONS_PATH, data)

    return processed


def _suggestion_type(sug: dict) -> str:
    category = sug.get("category", "").upper()
    title = (sug.get("title", "") + sug.get("description", "")).lower()
    if "backtest" in title or category in ("FOREX", "RESEARCH"):
        return "backtest"
    if "signal" in title or "scan" in title:
        return "signal_scan"
    if "forensic" in title:
        return "forensics"
    return "shell"


# ── Prompt queue processor ────────────────────────────────────────────────────

def process_prompt_queue(dry_run: bool = False) -> int:
    """
    Run PQ-### items tagged auto_execute=true.
    Safety: never auto-execute items that modify live trading params.
    Returns count of items processed.
    """
    data = _load(DATA_AGENT / "prompt_queue.json")
    prompts = data.get("prompts", []) if isinstance(data, dict) else []

    processed = 0
    for pq in prompts:
        if pq.get("status") != "QUEUED":
            continue
        if not pq.get("auto_execute", False):
            continue
        # Safety gate: skip anything touching live params
        script = pq.get("script", "")
        if any(kw in script.lower() for kw in ["parameters.yml", "live_trade", "execute_daily", "ptj_gates"]):
            log.info(f"PQ {pq['id']}: skipping auto-execute (touches live trading params)")
            continue

        log.info(f"Auto-executing PQ {pq['id']}: {pq.get('prompt', '')[:80]}")

        if not dry_run:
            ok, out = _run_script(script or "echo no-op")
            pq["status"] = "DONE" if ok else "ERROR"
            pq["executed_at"] = datetime.now().isoformat(timespec="seconds")
            pq["result"] = out[:300]
            _append_finding({
                "id": f"F-PQ-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "source": "prompt_queue",
                "pq_id": pq["id"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "result": out[:500],
                "ok": ok,
            })
        else:
            log.info(f"[DRY RUN] would run: {script[:100]}")

        processed += 1

    if not dry_run and processed > 0:
        if isinstance(data, dict):
            data["prompts"] = prompts
        _save(DATA_AGENT / "prompt_queue.json", data)

    return processed


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Oracle Research Agent")
    parser.add_argument("--task", help="Run a specific task by ID (RQ-NNN)")
    parser.add_argument("--next", action="store_true", help="Run the next QUEUED task")
    parser.add_argument("--check-suggestions", action="store_true",
                        help="Scan suggestions.json for PENDING items and auto-run them")
    parser.add_argument("--process-pq", action="store_true",
                        help="Process auto_execute=true items from prompt_queue.json")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    if args.check_suggestions:
        n = check_suggestions(dry_run=args.dry_run)
        print(f"Processed {n} pending suggestions")
        return

    if args.process_pq:
        n = process_prompt_queue(dry_run=args.dry_run)
        print(f"Processed {n} auto-execute prompt queue items")
        return

    if args.task:
        result = run_task(args.task, dry_run=args.dry_run)
        print(f"Summary: {result[:300]}")
        return

    if args.next:
        result = run_next_task(dry_run=args.dry_run)
        print(f"Summary: {result[:300]}")
        return

    print("Usage: research_agent.py [--task RQ-NNN] [--next] [--check-suggestions] [--dry-run]")


if __name__ == "__main__":
    main()
