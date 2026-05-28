"""
parallel_hypothesis_sweep.py — run all QUEUED hypotheses simultaneously

Reads data/agent/research_queue.json, runs every QUEUED item that has a
script and is not blocked, then writes results back and prints a summary.

Skips: requires_human=True, blocked_until!=None, script missing.

Usage:
    python3 scripts/parallel_hypothesis_sweep.py              # all QUEUED
    python3 scripts/parallel_hypothesis_sweep.py --dry-run    # show plan only
    python3 scripts/parallel_hypothesis_sweep.py --ids RQ-022 RQ-023
    python3 scripts/parallel_hypothesis_sweep.py --workers 8  # parallelism
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = ROOT / "data" / "agent" / "research_queue.json"

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def _load_queue() -> dict:
    with open(QUEUE_FILE) as f:
        return json.load(f)


def _save_queue(data: dict) -> None:
    with open(QUEUE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _is_runnable(item: dict) -> tuple[bool, str]:
    if item.get("requires_human"):
        return False, "requires human decision"
    if item.get("blocked_until"):
        return False, f"blocked_until={item['blocked_until']}"
    if not item.get("script"):
        return False, "no script defined"
    return True, ""


def _run_item(item: dict) -> dict:
    """Execute one hypothesis script; return result dict."""
    script = item["script"]
    timeout_secs = item.get("estimated_minutes", 10) * 60

    # Support both "python3 scripts/foo.py" and bare "scripts/foo.py"
    cmd = script.split() if script.startswith("python3 ") else ["python3", script]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            cwd=str(ROOT),
        )
        elapsed = round(time.monotonic() - t0, 1)
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            return {
                "id": item.get("id", "?"),
                "name": item.get("name", "?"),
                "status": "DONE",
                "last_result": output[:500] if output else "OK (no output)",
                "elapsed_s": elapsed,
            }
        else:
            return {
                "id": item.get("id", "?"),
                "name": item.get("name", "?"),
                "status": "ERROR",
                "last_result": (output[:500] if output else f"exit code {proc.returncode}"),
                "elapsed_s": elapsed,
            }
    except subprocess.TimeoutExpired:
        return {
            "id": item.get("id", "?"),
            "name": item.get("name", "?"),
            "status": "ERROR",
            "last_result": f"TIMEOUT after {timeout_secs}s",
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
    except Exception as exc:
        return {
            "id": item.get("id", "?"),
            "name": item.get("name", "?"),
            "status": "ERROR",
            "last_result": f"EXCEPTION: {exc}",
            "elapsed_s": round(time.monotonic() - t0, 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QUEUED hypotheses in parallel")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run, do not execute")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="Only run these IDs (e.g. RQ-022 RQ-023)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max parallel workers (default: 4)")
    args = parser.parse_args()

    data = _load_queue()
    queue: list[dict] = data["queue"]

    candidates = [q for q in queue if q.get("status") == "QUEUED"]
    if args.ids:
        candidates = [q for q in candidates if q.get("id") in args.ids]

    runnable, skipped = [], []
    for item in candidates:
        ok, reason = _is_runnable(item)
        (runnable if ok else skipped).append(
            item if ok else (item.get("id", "?"), item.get("name", "?"), reason)
        )

    print(f"\n{'=' * 62}")
    print(f"Parallel Hypothesis Sweep — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 62}")
    print(f"QUEUED: {len(candidates)}  |  Runnable: {len(runnable)}  |  Skipped: {len(skipped)}")

    if skipped:
        print("\nSkipped:")
        for sid, sname, reason in skipped:
            print(f"  {sid}: {sname} — {reason}")

    if not runnable:
        print("\nNothing to run.")
        return

    print(f"\nWill run ({args.workers} workers max):")
    for item in runnable:
        mins = item.get("estimated_minutes", 10)
        needs_net = " [network]" if item.get("needs_network") else ""
        print(f"  {item.get('id', '?'):16} ~{mins}min{needs_net}  {item.get('name', '?')}")

    if args.dry_run:
        print("\n[DRY RUN] Pass without --dry-run to execute.")
        return

    print()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_item, item): item for item in runnable}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            icon = "✓" if res["status"] == "DONE" else "✗"
            print(f"  {icon} {res['id']:16} {res['status']:6}  ({res['elapsed_s']}s)")

    # Write results back to queue
    now = datetime.now(timezone.utc).isoformat()
    result_map = {r["id"]: r for r in results}
    for item in queue:
        if item.get("id") in result_map:
            r = result_map[item["id"]]
            item["status"] = r["status"]
            item["last_result"] = r["last_result"]
            item["completed"] = now
    data["last_updated"] = now
    _save_queue(data)

    # Summary table
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"{'ID':<16}  {'Status':<8}  {'Time':>6}  Name")
    print(sep)
    for r in sorted(results, key=lambda x: x["id"]):
        print(f"{r['id']:<16}  {r['status']:<8}  {r['elapsed_s']:>5.1f}s  {r['name']}")
    print(sep)

    done = sum(1 for r in results if r["status"] == "DONE")
    error = sum(1 for r in results if r["status"] == "ERROR")
    print(f"Done: {done}  |  Error: {error}  |  Total: {len(results)}")

    if error:
        print("\nError details:")
        for r in results:
            if r["status"] == "ERROR":
                print(f"  {r['id']}: {r['last_result'][:300]}")

    sys.exit(1 if error else 0)


if __name__ == "__main__":
    main()
