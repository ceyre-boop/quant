#!/usr/bin/env python3
"""
Stray-file tripwire.

WHY: Five files (findings.jsonl, messages_to_colin.json, research_queue.json, usage.json, and a
junk "# GOALS.md") keep regenerating at the REPO ROOT instead of data/agent/. An exhaustive
static search proved NO writer in this repo is at fault — every path constant is correctly
anchored to repo-root/data/agent via Path(__file__).parent.parent.parent. So the writer is an
EXTERNAL/agentic process running with cwd=repo-root and writing bare filenames.

This tripwire catches it in the act: whenever a stray appears at root, it records best-effort
forensics (ctime, lsof, recent processes, content fingerprint vs the data/agent canonical) to a
JSONL log, then quarantines the stray so the root stays clean while the evidence is preserved.

Wired via launchd WatchPaths (scripts/com.alta.stray_tripwire.plist) so it fires the instant the
repo-root directory changes. Safe to run on every fire — it no-ops when no stray is present and
never touches the data/agent canonicals.

Usage:  python3 scripts/stray_tripwire.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "stray_tripwire.log"
QUARANTINE = ROOT / "logs" / "stray_quarantine"
CANONICAL_DIR = ROOT / "data" / "agent"

# Files that must never appear directly at the repo root (their home is data/agent/).
STRAY_NAMES = [
    "findings.jsonl",
    "messages_to_colin.json",
    "research_queue.json",
    "usage.json",
    "# GOALS.md",
]


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:  # noqa: BLE001 — forensics are best-effort
        return f"<{type(e).__name__}: {e}>"


def _fingerprint(p: Path) -> dict:
    data = p.read_bytes()
    first_line = ""
    try:
        first_line = data.split(b"\n", 1)[0].decode("utf-8", "replace")[:200]
    except Exception:
        pass
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "first_line": first_line,
    }


def _compare_canonical(name: str, stray_fp: dict) -> dict:
    """Compare the stray to its data/agent canonical — the strongest identity clue."""
    canon = CANONICAL_DIR / name
    if not canon.exists():
        return {"canonical_exists": False}
    cfp = _fingerprint(canon)
    rel = "identical_copy" if cfp["sha256"] == stray_fp["sha256"] else (
        "truncated_subset" if stray_fp["size"] < cfp["size"] else
        "larger_or_divergent" if stray_fp["size"] > cfp["size"] else "same_size_diff_content"
    )
    return {
        "canonical_exists": True,
        "canonical_size": cfp["size"],
        "canonical_sha256": cfp["sha256"],
        "relation": rel,
    }


def _process_snapshot(ctime: float) -> dict:
    """Full ps snapshot + the processes started shortly before the file's ctime."""
    ps = _run(["ps", "-axww", "-o", "pid,ppid,lstart,command"])
    suspects = []
    file_dt = datetime.fromtimestamp(ctime)
    for line in ps.splitlines()[1:]:
        # lstart is a fixed-width 24-char date like "Sat Jun  7 12:48:01 2026"
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        rest = parts[2]
        try:
            # command starts after the 24-char lstart field
            lstart_str = rest[:24]
            started = datetime.strptime(lstart_str, "%a %b %d %H:%M:%S %Y")
            delta = (file_dt - started).total_seconds()
            if -5 <= delta <= 120:  # started up to 2 min before (or just after) the write
                suspects.append({"pid": parts[0], "ppid": parts[1],
                                 "started": lstart_str, "cmd": rest[24:].strip()[:200],
                                 "delta_s": round(delta, 1)})
        except Exception:
            continue
    return {"suspects_near_ctime": suspects, "ps_lines": len(ps.splitlines())}


def main() -> int:
    present = [n for n in STRAY_NAMES if (ROOT / n).exists()]
    if not present:
        return 0  # no-op: cheap to fire often

    QUARANTINE.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)

    for name in present:
        p = ROOT / name
        try:
            st = p.stat()
            fp = _fingerprint(p)
            record = {
                "detected_utc": ts.isoformat(),
                "stray": name,
                "path": str(p),
                "ctime": datetime.fromtimestamp(st.st_ctime).isoformat(),
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "fingerprint": fp,
                "vs_canonical": _compare_canonical(name, fp),
                "lsof": _run(["lsof", "--", str(p)]),  # writer if handle still open
                "process_scan": _process_snapshot(st.st_ctime),
            }
            LOG_PATH.open("a").write(json.dumps(record) + "\n")

            # Quarantine: keep root clean, preserve evidence. Never touch data/agent canonicals.
            safe = name.replace("/", "_").replace(" ", "_").lstrip("#_") or "stray"
            dest = QUARANTINE / f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{safe}"
            shutil.move(str(p), str(dest))
            print(f"[stray_tripwire] caught + quarantined: {name} -> {dest.name}")
        except FileNotFoundError:
            continue  # vanished between scan and handling — ignore
        except Exception as e:  # noqa: BLE001
            print(f"[stray_tripwire] error handling {name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
