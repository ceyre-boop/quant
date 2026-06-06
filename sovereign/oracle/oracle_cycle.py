"""
Oracle Learning Cycle — Full Orchestrator
sovereign/oracle/oracle_cycle.py

Runs the complete daily cycle in sequence:
  2:00 AM — Phase 1: HARVEST  (free)
  2:15 AM — Phase 2: REFLECT  (≤2 cents)
  2:20 AM — Phase 3: TEST     (free)
  2:25 AM — Phase 4: CODIFY   (free if validated)

Monthly: Phase 5: MONITOR (free)

Called by agent_scheduler.py at 2:00 AM ET.
Direct usage: python3 sovereign/oracle/oracle_cycle.py [--monitor] [--dry-run]
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
# Allow direct invocation (`python3 sovereign/oracle/oracle_cycle.py`): when run as a script,
# sys.path[0] is this file's dir, not the repo root, so `import sovereign` would fail.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WISDOM_FILE     = ROOT / "I_am_a_good_trader.md"
ARCHIVE_FILE    = ROOT / "I_was_a_good_trader.md"
ORACLE_LOG      = ROOT / "logs" / "oracle_cycle.log"
HEARTBEAT       = ROOT / "logs" / ".heartbeat_oracle_reflection"
ORACLE_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(ORACLE_LOG, "a") as f:
        f.write(line + "\n")


def run_daily_cycle(dry_run: bool = False) -> dict:
    """Run harvest → reflect → test → codify. Returns cycle summary."""
    from sovereign.oracle.harvest_cycle import run_harvest, load_recent_harvests
    from sovereign.oracle.reflect_cycle import run_reflect
    from sovereign.oracle.validation_cycle import run_validation
    from sovereign.oracle.codify_cycle import run_codify

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Execution heartbeat FIRST (before any phase/gate) — loop_health measures EXECUTION,
    # not output. Written even if a phase later fails, so "did the cycle run" is unambiguous.
    try:
        HEARTBEAT.write_text(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass
    _log(f"Oracle daily cycle starting — {date}")

    # Phase 1: Harvest
    _log("Phase 1: HARVEST")
    try:
        harvest = run_harvest(date=date)
        _log(f"  Harvested {harvest['trades_closed']} trades. "
             f"WR={harvest.get('win_rate', 0)*100:.0f}% " if harvest.get('win_rate') else
             f"  Harvested {harvest['trades_closed']} trades.")
        if harvest.get("anomalies"):
            for a in harvest["anomalies"]:
                _log(f"  ⚠ Anomaly: {a}")
    except Exception as e:
        _log(f"  Harvest failed: {e}")
        return {"date": date, "status": "HARVEST_FAILED", "error": str(e)}

    # Phase 2: Reflect
    _log("Phase 2: REFLECT")
    harvests = load_recent_harvests(days=7)
    try:
        if dry_run:
            _log("  [dry-run] Skipping Oracle API call")
            reflection = {"date": date, "reflection": {
                "candidate_lesson": {
                    "lesson_text": "DRY RUN — no actual lesson proposed",
                    "testable_rule": "False",
                    "expected_impact": "N/A",
                    "mechanism": "N/A",
                    "sample_needed": 999999,
                },
                "retirement_flag": {"lesson_to_review": None, "reason": None},
                "system_health_note": "dry run",
            }}
        else:
            reflection = run_reflect(harvests, date=date)
            _log(f"  Cost: ${reflection['estimated_cost_usd']:.4f}")
            _log(f"  Proposed: {reflection['reflection'].get('candidate_lesson', {}).get('lesson_text', '')[:80]}")
    except Exception as e:
        _log(f"  Reflect failed: {e}")
        return {"date": date, "status": "REFLECT_FAILED", "error": str(e)}

    # Phase 3: Test
    _log("Phase 3: TEST")
    try:
        validation = run_validation(reflection, date=date)
        _log(f"  Verdict: {validation['verdict']}")
        _log(f"  Test A (delta Sharpe): {validation.get('test_a_delta_sharpe')} {'✓' if validation.get('test_a_passed') else '✗'}")
        _log(f"  Test B (p-value):      {validation.get('test_b_p_value')} {'✓' if validation.get('test_b_passed') else '✗'}")
        _log(f"  Test C (holdout):      {validation.get('test_c_replication_ratio')} {'✓' if validation.get('test_c_passed') else '✗'}")
    except Exception as e:
        _log(f"  Validation failed: {e}")
        return {"date": date, "status": "VALIDATION_FAILED", "error": str(e)}

    # Phase 4: Codify (only if validated)
    _log("Phase 4: CODIFY")
    try:
        codify_result = run_codify(validation, reflection, date=date)
        if codify_result["action"] == "CODIFIED":
            _log(f"  ✅ NEW LESSON {codify_result['lesson_number']} added to I_am_a_good_trader.md")
            if codify_result.get("retired_lesson"):
                _log(f"  📦 Lesson {codify_result['retired_lesson']} retired to I_was_a_good_trader.md")
            _log(f"  Implementation prompt: {codify_result['implementation_prompt']}")
        else:
            _log(f"  Skipped: {codify_result.get('reason', 'verdict not VALIDATED')}")
    except Exception as e:
        _log(f"  Codify failed (non-fatal): {e}")
        codify_result = {"action": "ERROR", "error": str(e)}

    summary = {
        "date": date,
        "status": "COMPLETE",
        "trades_harvested": harvest["trades_closed"],
        "anomalies": harvest.get("anomalies", []),
        "oracle_cost_usd": reflection.get("estimated_cost_usd", 0.0) if not dry_run else 0.0,
        "candidate_lesson": reflection.get("reflection", {}).get("candidate_lesson", {}).get("lesson_text", ""),
        "verdict": validation["verdict"],
        "new_lesson_number": codify_result.get("lesson_number"),
        "lesson_retired": codify_result.get("retired_lesson"),
    }

    _log(f"Daily cycle complete. Status: {summary['status']}")

    # Write dashboard-visible summary to data/agent/ (atomic: tmp → replace)
    import os as _os
    _AGENT_DIR = ROOT / "data" / "agent"
    _AGENT_DIR.mkdir(parents=True, exist_ok=True)
    _oracle_summary = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "harvest": {
            "trades_closed":    harvest["trades_closed"],
            "win_rate":         harvest.get("win_rate"),
            "failure_taxonomy": harvest.get("failure_taxonomy", {}),
        },
        "reflection": {
            "candidate_lesson": reflection.get("reflection", {}).get("candidate_lesson", {}).get("lesson_text", ""),
            "mechanism":        reflection.get("reflection", {}).get("candidate_lesson", {}).get("mechanism", ""),
            "testable_rule":    reflection.get("reflection", {}).get("candidate_lesson", {}).get("testable_rule", ""),
        },
        "validation": {
            "verdict": validation["verdict"],
            "reason":  validation.get("reason", ""),
        },
        "wisdom": {
            "active_lesson_count": codify_result.get("active_lessons_count"),
            "last_codified":       codify_result.get("lesson_number"),
            "codify_status":       codify_result.get("action", "UNKNOWN"),
            "codify_error":        codify_result.get("error"),
        },
        "cycle": {
            "cost_usd": summary["oracle_cost_usd"],
            "status":   summary["status"],
            "date":     summary["date"],
        },
    }
    _tmp = _AGENT_DIR / ".oracle_daily_summary.tmp"
    _tmp.write_text(json.dumps(_oracle_summary, indent=2))
    _os.replace(str(_tmp), str(_AGENT_DIR / "oracle_daily_summary.json"))
    _log("oracle_daily_summary.json written to data/agent/")

    return summary


def run_monthly_monitor() -> dict:
    """
    Phase 5: MONITOR — re-validate all active lessons against recent 30 days.
    Runs monthly. Free — no LLM calls.
    """
    from sovereign.oracle.validation_cycle import _load_all_forensics, _apply_rule, _compute_sharpe

    _log("Oracle monthly monitor starting")

    if not WISDOM_FILE.exists():
        _log("No wisdom file found. Skipping monitor.")
        return {"status": "NO_WISDOM_FILE"}

    content = WISDOM_FILE.read_text()

    # Extract rules from each lesson
    lessons = re.findall(
        r"### LESSON (\d+) — ([^\n]+).*?\*\*Rule:\*\* `([^`]+)`.*?\*\*Evidence:\*\* ([^\n]+)",
        content, re.DOTALL
    )

    if not lessons:
        _log("No lessons with parseable rules found.")
        return {"status": "NO_LESSONS", "lessons_checked": 0}

    all_records = _load_all_forensics()

    # Recent 30 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent = []
    for r in all_records:
        ts_str = r.get("entry_time", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts > cutoff:
                recent.append(r)
        except Exception:
            pass

    _log(f"  Records in last 30 days: {len(recent)} / {len(all_records)} total")

    results = []
    retirements_needed = []

    for num, title, rule, evidence in lessons:
        if len(recent) < 20:
            health = "INSUFFICIENT_DATA"
            decay_ratio = None
        else:
            filtered, all_r = _apply_rule(recent, rule)
            if len(filtered) < 5:
                health = "INSUFFICIENT_FILTERED"
                decay_ratio = None
            else:
                current_delta = _compute_sharpe(filtered) - _compute_sharpe(all_r)

                # Extract original delta from evidence text
                orig_match = re.search(r"delta.*?(-?\d+\.\d+)", evidence, re.IGNORECASE)
                orig_delta = float(orig_match.group(1)) if orig_match else 0.05

                decay_ratio = current_delta / orig_delta if orig_delta != 0 else 0.0

                if decay_ratio > 0.80:
                    health = "🟢 HEALTHY"
                elif decay_ratio > 0.50:
                    health = "🟡 MONITORING"
                elif decay_ratio > 0.00:
                    health = "🔴 DECAYING"
                else:
                    health = "💀 HARMFUL — retire immediately"
                    retirements_needed.append((int(num), current_delta))

        results.append({
            "lesson_num": int(num),
            "title": title[:60],
            "health": health,
            "decay_ratio": round(decay_ratio, 3) if decay_ratio is not None else None,
        })
        _log(f"  Lesson {num}: {health} (decay_ratio={decay_ratio:.3f})" if decay_ratio is not None
             else f"  Lesson {num}: {health}")

    # Phase 5b: Reasoning pattern analysis
    try:
        from sovereign.forensics.reasoning_analyzer import run_analysis
        reasoning_report = run_analysis()
        _log(f"  Reasoning analysis: {reasoning_report.n_trades_analyzed} trades analyzed, "
             f"{len(reasoning_report.best_chains)} best chains, "
             f"{len(reasoning_report.worst_chains)} worst chains")
    except Exception as e:
        _log(f"  Reasoning analysis failed (non-fatal): {e}")

    # Handle retirements
    from sovereign.oracle.codify_cycle import retire_lesson
    for lesson_num, delta in retirements_needed:
        _log(f"  ⚠ Retiring Lesson {lesson_num} (HARMFUL: current delta={delta:.3f})")
        retire_lesson(lesson_num, reason="HARMFUL — current performance is negative", latest_delta_sharpe=delta)

    # Update health status in wisdom file
    for r in results:
        if r["health"] != "INSUFFICIENT_DATA" and r["health"] != "INSUFFICIENT_FILTERED":
            content = re.sub(
                rf"(### LESSON {r['lesson_num']}.*?\*\*Health:\*\* )[^\n]+",
                rf"\g<1>{r['health']}",
                content, flags=re.DOTALL, count=1
            )
            content = re.sub(
                rf"(### LESSON {r['lesson_num']}.*?\*\*Last validated:\*\* )[^\n]+",
                rf"\g<1>{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                content, flags=re.DOTALL, count=1
            )

    WISDOM_FILE.write_text(content)
    _log("Monthly monitor complete.")

    return {
        "status": "COMPLETE",
        "lessons_checked": len(lessons),
        "records_in_window": len(recent),
        "retirements": len(retirements_needed),
        "lesson_health": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oracle learning cycle orchestrator")
    parser.add_argument("--monitor", action="store_true", help="Run monthly monitor instead of daily cycle")
    parser.add_argument("--dry-run", action="store_true", help="Skip Oracle API call")
    parser.add_argument("--status", action="store_true", help="Show current wisdom file summary")
    args = parser.parse_args()

    if args.status:
        if WISDOM_FILE.exists():
            content = WISDOM_FILE.read_text()
            lessons = re.findall(r"### LESSON (\d+) — ([^\n]+).*?\*\*Health:\*\* ([^\n]+)", content, re.DOTALL)
            print(f"\nI_am_a_good_trader.md — {len(lessons)} active lessons:")
            for num, title, health in lessons:
                print(f"  L{num}: {health.strip()} | {title[:60]}")
        else:
            print("No wisdom file found.")
    elif args.monitor:
        result = run_monthly_monitor()
        print(json.dumps(result, indent=2))
    else:
        result = run_daily_cycle(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
