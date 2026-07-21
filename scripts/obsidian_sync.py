#!/usr/bin/env python3
"""
obsidian_sync.py — mirror live system state OUT of ~/quant INTO the Obsidian vault.

WHY THIS EXISTS
---------------
The scheduled Claude tasks (oracle-check-av-sentiment, eod-carry-debrief,
weekly-strategy-review, oracle-market-intelligence, the monthly carry reports)
run in a sandbox that mounts ONLY ~/Obsidian/Obsidian. They have no `/Users`
mount at all, so they cannot read this repo, and because they run unattended
they cannot request access mid-run. Every one of those tasks has therefore been
writing honest reports about its own blindness instead of doing its job.

We cannot change the sandbox mount from inside the repo. So we push the state
the other way: this script runs ON THE HOST as plain Python -- no `claude`
binary, no network dependency for the core path -- and writes a readable mirror
into the vault where the scheduled tasks CAN see it.

CONTRACT (non-negotiable, matches the discipline the existing logs already keep)
-------------------------------------------------------------------------------
1. Every file carries a `generated_at` UTC timestamp.
2. Every section carries an explicit status: OK | STALE | UNAVAILABLE.
3. UNAVAILABLE always carries a reason.
4. We NEVER write a zero, an empty list, or a neutral value in place of a value
   we could not read. "conviction UNREAD" is not "conviction 0.0". A downstream
   reader must never be able to mistake absence of data for a reading of zero.
5. This script never raises. A crashed sync that leaves yesterday's mirror in
   place while claiming freshness is worse than a mirror that says UNAVAILABLE.

Usage:
    python3 scripts/obsidian_sync.py            # write the mirror
    python3 scripts/obsidian_sync.py --dry-run  # print, write nothing
    python3 scripts/obsidian_sync.py --check    # exit 1 if mirror would be degraded
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
VAULT = Path(os.environ.get("ALTA_VAULT", "/Users/taboost/Obsidian/Obsidian"))
MIRROR = VAULT / "Trading" / "System" / "Mirror"

NOW = datetime.now(timezone.utc)

OK = "OK"
STALE = "STALE"
UNAVAILABLE = "UNAVAILABLE"

# How old a source file may be before its section is downgraded to STALE.
FRESHNESS_HOURS: dict[str, float] = {
    "oracle": 26.0,
    "positions": 26.0,
    "health": 6.0,
    "ledger": 48.0,
    "execution": 26.0,
    "prop": 48.0,
    "repo": 24.0 * 365,  # git state is never "stale", only unreadable
}


# --------------------------------------------------------------------------
# Section: a status-carrying unit of mirrored state.
# --------------------------------------------------------------------------
class Section:
    def __init__(self, name: str, kind: str = "oracle"):
        self.name = name
        self.kind = kind
        self.status = UNAVAILABLE
        self.reason = "not yet populated"
        self.source: str | None = None
        self.source_mtime: datetime | None = None
        self.data: Any = None
        self.notes: list[str] = []

    def unavailable(self, reason: str) -> "Section":
        self.status = UNAVAILABLE
        self.reason = reason
        self.data = None
        return self

    def ok(self, data: Any, source: Path | None = None) -> "Section":
        self.data = data
        self.reason = ""
        if source is not None:
            self.source = str(source.relative_to(REPO)) if _under(source, REPO) else str(source)
            try:
                self.source_mtime = datetime.fromtimestamp(source.stat().st_mtime, timezone.utc)
            except OSError:
                self.source_mtime = None
        self.status = OK
        if self.source_mtime is not None:
            age_h = (NOW - self.source_mtime).total_seconds() / 3600.0
            limit = FRESHNESS_HOURS.get(self.kind, 26.0)
            if age_h > limit:
                self.status = STALE
                self.reason = (
                    f"source is {age_h:.1f}h old (limit {limit:.0f}h) -- "
                    f"values are last-known, not current"
                )
        return self

    @property
    def age_hours(self) -> float | None:
        if self.source_mtime is None:
            return None
        return (NOW - self.source_mtime).total_seconds() / 3600.0

    def header_md(self) -> str:
        bits = [f"**Status:** `{self.status}`"]
        if self.source:
            bits.append(f"**Source:** `{self.source}`")
        if self.source_mtime:
            bits.append(
                f"**Source written:** {self.source_mtime.isoformat()} "
                f"({self.age_hours:.1f}h ago)"
            )
        if self.reason:
            bits.append(f"**Reason:** {self.reason}")
        return "  \n".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason or None,
            "source": self.source,
            "source_written_utc": self.source_mtime.isoformat() if self.source_mtime else None,
            "age_hours": round(self.age_hours, 2) if self.age_hours is not None else None,
        }


def _under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def load_json(rel: str, section: Section) -> Section:
    """Load a JSON file relative to the repo, recording provenance and freshness."""
    path = REPO / rel
    if not path.exists():
        return section.unavailable(f"`{rel}` does not exist in the repo")
    try:
        with path.open() as fh:
            return section.ok(json.load(fh), source=path)
    except json.JSONDecodeError as exc:
        return section.unavailable(f"`{rel}` exists but is not valid JSON: {exc}")
    except OSError as exc:
        return section.unavailable(f"`{rel}` could not be read: {exc}")


def load_jsonl_tail(rel: str, section: Section, limit: int = 25) -> Section:
    path = REPO / rel
    if not path.exists():
        return section.unavailable(f"`{rel}` does not exist in the repo")
    try:
        rows = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            return section.unavailable(f"`{rel}` exists but contains no parseable records")
        return section.ok(rows[-limit:], source=path)
    except OSError as exc:
        return section.unavailable(f"`{rel}` could not be read: {exc}")


# --------------------------------------------------------------------------
# Collectors
# --------------------------------------------------------------------------
def collect_oracle() -> dict[str, Section]:
    out: dict[str, Section] = {}
    out["loop_health"] = load_json(
        "data/oracle/loop_health_status.json", Section("Oracle loop health", "health")
    )
    out["daily_digest"] = load_json(
        "data/oracle/daily_digest.json", Section("Oracle daily digest", "oracle")
    )
    out["briefing"] = load_json(
        "data/agent/oracle_briefing_morning.json", Section("Morning briefing", "oracle")
    )

    # Today's bias record, if the bias job has run.
    bias = Section("Directional bias (per pair)", "oracle")
    today = NOW.strftime("%Y-%m-%d")
    bias_path = REPO / "data" / "bias" / f"bias_{today}.json"
    if bias_path.exists():
        load_json(str(bias_path.relative_to(REPO)), bias)
    else:
        candidates = sorted((REPO / "data" / "bias").glob("bias_*.json")) if (
            REPO / "data" / "bias"
        ).exists() else []
        if candidates:
            load_json(str(candidates[-1].relative_to(REPO)), bias)
            bias.notes.append(
                f"No bias record for {today}; showing most recent ({candidates[-1].name}). "
                "This is last-known, not a reading for today."
            )
        else:
            bias.unavailable("no data/bias/bias_*.json records exist at all")
    out["bias"] = bias
    return out


def collect_positions() -> dict[str, Section]:
    out: dict[str, Section] = {}
    out["paper_trades"] = load_json(
        "data/ledger/ict_paper_trades.json", Section("Paper trades (open/closed)", "positions")
    )
    out["oanda_fills"] = load_json(
        "data/ledger/oanda_fills.json", Section("OANDA fills", "positions")
    )
    out["live_trade_log"] = load_jsonl_tail(
        "data/ledger/live_trade_log.jsonl", Section("Live trade log (tail)", "ledger")
    )
    out["veto_ledger"] = load_jsonl_tail(
        "data/ledger/oanda_veto_ledger.jsonl", Section("OANDA veto ledger (tail)", "ledger")
    )

    # Account mode. This is the paper-vs-live question that has been open in the
    # vault -- answer it from the environment, definitively, every single run.
    mode = Section("Account mode (paper vs live)", "repo")
    env_path = REPO / ".env"
    if not env_path.exists():
        mode.unavailable("no .env file -- cannot determine whether OANDA is practice or live")
    else:
        try:
            live_flag = None
            base_url = None
            for line in env_path.read_text().splitlines():
                if line.startswith("OANDA_LIVE="):
                    live_flag = line.split("=", 1)[1].strip()
                elif line.startswith("OANDA_BASE_URL="):
                    base_url = line.split("=", 1)[1].strip()
            if live_flag is None and base_url is None:
                mode.unavailable(".env exists but defines neither OANDA_LIVE nor OANDA_BASE_URL")
            else:
                is_practice = (live_flag in ("0", "false", "False", "")) or (
                    base_url is not None and "fxpractice" in base_url
                )
                mode.ok(
                    {
                        "OANDA_LIVE": live_flag,
                        "OANDA_BASE_URL": base_url,
                        "verdict": "PAPER (practice endpoint)" if is_practice else "LIVE CAPITAL",
                        "confidence": "definitive -- read from .env at sync time",
                    },
                    source=env_path,
                )
        except OSError as exc:
            mode.unavailable(f".env could not be read: {exc}")
    out["account_mode"] = mode
    return out


def collect_execution() -> dict[str, Section]:
    out: dict[str, Section] = {}
    out["heartbeat"] = load_json(
        "data/execution/heartbeat.json", Section("Execution heartbeat", "execution")
    )
    out["fill_log"] = load_jsonl_tail(
        "data/execution/fill_log.jsonl", Section("Fill log (tail)", "execution")
    )

    summary = Section("Daily summary", "execution")
    path = REPO / "data" / "execution" / "daily_summary.csv"
    if not path.exists():
        summary.unavailable("data/execution/daily_summary.csv does not exist")
    else:
        try:
            with path.open() as fh:
                summary.ok(list(csv.DictReader(fh)), source=path)
        except OSError as exc:
            summary.unavailable(f"daily_summary.csv could not be read: {exc}")
    out["daily_summary"] = summary
    return out


def collect_health() -> dict[str, Section]:
    out: dict[str, Section] = {}
    out["system_health"] = load_json(
        "data/health/system_health.json", Section("System health", "health")
    )
    out["forex_data"] = load_json(
        "data/health/forex_data_status.json", Section("Forex data health", "health")
    )
    out["prop_challenge"] = load_json(
        "data/agent/prop_challenge_state.json", Section("Prop challenge state", "prop")
    )

    # launchd -- only meaningful on the host. This is precisely the check the
    # sandboxed tasks cannot perform, which is why running here matters.
    jobs = Section("launchd jobs (alta.* / clawd.*)", "health")
    if sys.platform != "darwin":
        jobs.unavailable("not running on macOS -- launchctl unavailable")
    else:
        try:
            proc = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True, timeout=20
            )
            if proc.returncode != 0:
                jobs.unavailable(f"launchctl list exited {proc.returncode}: {proc.stderr.strip()}")
            else:
                rows = []
                for line in proc.stdout.splitlines()[1:]:
                    parts = line.split(None, 2)
                    if len(parts) != 3:
                        continue
                    pid, status, label = parts
                    if any(k in label.lower() for k in ("alta", "clawd", "sovereign", "oracle")):
                        rows.append(
                            {
                                "label": label,
                                "pid": None if pid == "-" else pid,
                                "last_exit_status": status,
                                "running": pid != "-",
                            }
                        )
                if rows:
                    jobs.status = OK
                    jobs.reason = ""
                    jobs.data = rows
                else:
                    jobs.unavailable(
                        "launchctl ran but no alta.*/clawd.*/sovereign.*/oracle.* jobs are loaded"
                    )
        except FileNotFoundError:
            jobs.unavailable("launchctl binary not found")
        except subprocess.TimeoutExpired:
            jobs.unavailable("launchctl list timed out after 20s")
        except Exception as exc:  # noqa: BLE001 - never let a collector kill the sync
            jobs.unavailable(f"launchctl probe failed: {exc}")
    out["launchd"] = jobs

    # Recent log errors -- another host-only read.
    errs = Section("Recent log errors", "health")
    logdir = REPO / "logs"
    if not logdir.exists():
        errs.unavailable("logs/ directory does not exist")
    else:
        try:
            cutoff = NOW - timedelta(hours=24)
            found: list[dict[str, str]] = []
            for log in sorted(logdir.glob("*.log")):
                try:
                    if datetime.fromtimestamp(log.stat().st_mtime, timezone.utc) < cutoff:
                        continue
                    tail = log.read_text(errors="replace").splitlines()[-400:]
                except OSError:
                    continue
                hits = [
                    ln.strip()
                    for ln in tail
                    if any(k in ln for k in ("ERROR", "Traceback", "CRITICAL", "Exception"))
                ]
                if hits:
                    found.append({"log": log.name, "count": str(len(hits)), "last": hits[-1][:400]})
            if found:
                errs.status = OK
                errs.reason = ""
                errs.data = found
            else:
                errs.status = OK
                errs.reason = ""
                errs.data = []
                errs.notes.append(
                    "No ERROR/Traceback/CRITICAL lines in logs modified within 24h. "
                    "Note this is a genuine clean read, not a failure to look."
                )
        except Exception as exc:  # noqa: BLE001
            errs.unavailable(f"log scan failed: {exc}")
    out["log_errors"] = errs
    return out


def collect_repo() -> dict[str, Section]:
    out: dict[str, Section] = {}
    git = Section("Git state", "repo")
    try:
        def _git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(REPO), *args],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout.strip()

        git.status = OK
        git.reason = ""
        git.data = {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "--short", "HEAD"),
            "head_subject": _git("log", "-1", "--pretty=%s"),
            "head_date": _git("log", "-1", "--pretty=%cI"),
            "dirty_files": [ln for ln in _git("status", "--porcelain").splitlines() if ln],
            "unpushed": _git("log", "--oneline", "@{u}..HEAD").splitlines()
            if _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
            else [],
        }
    except subprocess.CalledProcessError as exc:
        git.unavailable(f"git command failed: {exc.stderr.strip() if exc.stderr else exc}")
    except Exception as exc:  # noqa: BLE001
        git.unavailable(f"git probe failed: {exc}")
    out["git"] = git

    nxt = Section("NEXT.md (tail)", "repo")
    path = REPO / "NEXT.md"
    if not path.exists():
        nxt.unavailable("NEXT.md does not exist")
    else:
        try:
            nxt.ok("\n".join(path.read_text(errors="replace").splitlines()[-120:]), source=path)
        except OSError as exc:
            nxt.unavailable(f"NEXT.md could not be read: {exc}")
    out["next_md"] = nxt
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_section(sec: Section, body_renderer=None) -> str:
    lines = [f"### {sec.name}", "", sec.header_md(), ""]
    for note in sec.notes:
        lines.append(f"> {note}")
        lines.append("")
    if sec.status == UNAVAILABLE:
        lines.append(
            "_No value is written for this section. Absence of data is not a reading of zero._"
        )
        lines.append("")
        return "\n".join(lines)
    if body_renderer is not None:
        lines.append(body_renderer(sec.data))
    else:
        lines.append("```json")
        lines.append(json.dumps(sec.data, indent=2, default=str)[:12000])
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_file(path: Path, title: str, sections: dict[str, Section], dry_run: bool) -> None:
    degraded = [s for s in sections.values() if s.status != OK]
    banner_status = OK if not degraded else (
        UNAVAILABLE if all(s.status == UNAVAILABLE for s in sections.values()) else STALE
    )
    parts = [
        "---",
        f"title: {title}",
        "tags: [mirror, generated, do-not-edit]",
        f"generated_at: {NOW.isoformat()}",
        f"mirror_status: {banner_status}",
        "---",
        "",
        f"# {title}",
        "",
        f"> **GENERATED FILE — do not edit.** Written by `scripts/obsidian_sync.py` on the host.",
        f"> Generated at **{NOW.isoformat()}**.",
        f"> Overall status: **{banner_status}**"
        + (f" — {len(degraded)} of {len(sections)} sections degraded." if degraded else "."),
        ">",
        "> Every section below carries its own status and source timestamp. A section marked",
        "> `UNAVAILABLE` has **no value**, which is not the same as a value of zero. If you are",
        "> a scheduled task reading this file, report the status honestly rather than",
        "> substituting a neutral reading.",
        "",
    ]
    for sec in sections.values():
        parts.append(render_section(sec))
    content = "\n".join(parts)
    if dry_run:
        print(f"--- would write {path} ({len(content)} bytes) ---")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  + {path.relative_to(VAULT)}  [{banner_status}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print instead of writing")
    ap.add_argument("--check", action="store_true", help="exit 1 if any section is degraded")
    args = ap.parse_args()

    if not VAULT.exists():
        print(f"FATAL: vault not found at {VAULT}", file=sys.stderr)
        return 2

    groups: dict[str, tuple[str, dict[str, Section]]] = {}
    for key, title, fn in [
        ("oracle-state", "Mirror — Oracle State", collect_oracle),
        ("positions", "Mirror — Positions & Ledger", collect_positions),
        ("execution", "Mirror — Execution", collect_execution),
        ("system-health", "Mirror — System Health", collect_health),
        ("repo-state", "Mirror — Repo State", collect_repo),
    ]:
        try:
            groups[key] = (title, fn())
        except Exception:  # noqa: BLE001 - a broken collector must not kill the sync
            broken = Section(title, "repo")
            broken.unavailable(
                "collector raised an exception:\n```\n" + traceback.format_exc()[-1500:] + "\n```"
            )
            groups[key] = (title, {"collector": broken})

    if not args.dry_run:
        MIRROR.mkdir(parents=True, exist_ok=True)

    for key, (title, sections) in groups.items():
        write_file(MIRROR / f"{key}.md", title, sections, args.dry_run)

    # Machine-readable status so tasks can gate on freshness without parsing prose.
    all_sections = {
        f"{key}.{name}": sec for key, (_, secs) in groups.items() for name, sec in secs.items()
    }
    counts = {
        OK: sum(1 for s in all_sections.values() if s.status == OK),
        STALE: sum(1 for s in all_sections.values() if s.status == STALE),
        UNAVAILABLE: sum(1 for s in all_sections.values() if s.status == UNAVAILABLE),
    }
    status_doc = {
        "generated_at": NOW.isoformat(),
        "repo": str(REPO),
        "overall": OK if counts[UNAVAILABLE] == 0 and counts[STALE] == 0 else "DEGRADED",
        "counts": counts,
        "sections": {k: s.to_dict() for k, s in all_sections.items()},
    }
    if args.dry_run:
        print(json.dumps(status_doc, indent=2))
    else:
        (MIRROR / "mirror_status.json").write_text(json.dumps(status_doc, indent=2))
        print(f"  + {(MIRROR / 'mirror_status.json').relative_to(VAULT)}")

        readme = [
            "---",
            "title: Mirror — README",
            "tags: [mirror, generated]",
            f"generated_at: {NOW.isoformat()}",
            "---",
            "",
            "# Mirror — README",
            "",
            "This folder is a **one-way projection of live system state** out of `~/quant`",
            "and into the vault, written by `scripts/obsidian_sync.py` running on the host.",
            "",
            "It exists because the scheduled Claude tasks mount only the vault -- they have",
            "no access to the repo and cannot request it mid-run. Without this mirror they",
            "are structurally blind and can only report their own blindness.",
            "",
            "## How to read it",
            "",
            "- Check `mirror_status.json` first: `generated_at` and `overall`.",
            "- **If `generated_at` is older than ~6 hours, treat everything here as stale**",
            "  and say so in your output. Do not present stale values as current readings.",
            "- Sections marked `UNAVAILABLE` have **no value**. That is not zero, not neutral,",
            "  not healthy. Report it as unread.",
            "",
            "## Files",
            "",
            f"- `oracle-state.md` — Oracle loop health, digest, morning briefing, per-pair bias",
            f"- `positions.md` — paper trades, OANDA fills, veto ledger, **account mode**",
            f"- `execution.md` — heartbeat, fill log tail, daily summary",
            f"- `system-health.md` — health JSON, launchd jobs, recent log errors",
            f"- `repo-state.md` — git branch/HEAD/dirty/unpushed, NEXT.md tail",
            f"- `mirror_status.json` — machine-readable status for every section",
            "",
            "## Regenerating",
            "",
            "```bash",
            "python3 ~/quant/scripts/obsidian_sync.py",
            "```",
            "",
            "Scheduled on the host via `scripts/com.alta.obsidian_sync.plist`.",
            "",
        ]
        (MIRROR / "README.md").write_text("\n".join(readme))
        print(f"  + {(MIRROR / 'README.md').relative_to(VAULT)}")

    print(
        f"\nmirror {status_doc['overall']}: "
        f"{counts[OK]} OK / {counts[STALE]} STALE / {counts[UNAVAILABLE]} UNAVAILABLE"
    )
    if args.check and status_doc["overall"] != OK:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nobsidian_sync FAILED -- mirror may be stale. Do NOT trust it.", file=sys.stderr)
        sys.exit(2)
