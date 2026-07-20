"""
Shared path + parsing helpers for the brain read/write modules.

Every path is resolved once, defensively. Nothing here raises on a missing
file — callers get empty/None and decide what to do.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

# --- Vault + repo roots -----------------------------------------------------

# Vault root, overridable for tests / alternate machines.
VAULT = Path(os.environ.get("ALTA_VAULT", str(Path.home() / "Obsidian" / "Obsidian")))

# Repo root = two levels up from this file (sovereign/brain/_paths.py -> repo).
REPO = Path(os.environ.get("ALTA_REPO", str(Path(__file__).resolve().parents[2])))

# Vault sub-locations the brain reads from and writes to.
BRAIN_DIR = VAULT / "00-BRAIN"
TRADING = VAULT / "Trading"
OPS = TRADING / "Ops"
RESEARCH = TRADING / "Research"
ORACLE_LOG = TRADING / "Oracle-Log"
HYPOTHESES = TRADING / "System"
MARKET_INTEL = TRADING / "Market-Intelligence"
PSYCH = VAULT / "Trading Psychology"

BRAIN_INDEX = VAULT / "BRAIN_INDEX.md"
WEAKNESS_LOG = PSYCH / "weakness_log.md"
HYPOTHESIS_LEDGER_NOTE = HYPOTHESES / "Hypothesis-Ledger.md"
REGIME_LOG = MARKET_INTEL / "regime_observations.md"
VERDICT_LOG = RESEARCH / "Verdict-Log.md"

# Repo data sources.
VERDICTS_JSONL = REPO / "data" / "research" / "auto_hypothesis_results.jsonl"
FACTORY_LEDGER = REPO / "data" / "research" / "factory_ledger.jsonl"
CONTEXT_DIR = REPO / "data" / "context"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def read_text(path: Path) -> str:
    """Read a file's text, returning '' if it is missing or unreadable."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def iter_jsonl(path: Path):
    """Yield parsed objects from a .jsonl file, skipping blank/broken lines."""
    import json

    text = read_text(path)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except (ValueError, TypeError):
            continue


def latest_matching(directory: Path, pattern: str, n: int = 1) -> list[Path]:
    """Return the n most-recently-modified files in `directory` matching glob."""
    try:
        files = sorted(
            Path(directory).glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    return files[:n]


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file we are about to write."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def append_line(path: Path, text: str) -> bool:
    """Append text to a file, creating it (and its dir) if needed."""
    ensure_parent(path)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
        return True
    except OSError:
        return False


# Match "## CONFIRMED (12)" style section headers and "- [[HYP-045]] — desc".
_SECTION_RE = re.compile(r"^##+\s+([A-Z_]+)\s*(?:\((\d+)\))?", re.MULTILINE)
_LEDGER_ITEM_RE = re.compile(r"^-\s*\[\[([^\]]+)\]\]\s*(?:[—-]\s*(.*))?$")


def parse_ledger_sections(text: str) -> dict[str, list[dict]]:
    """
    Parse the vault Hypothesis-Ledger note into {STATUS: [{id, desc}, ...]}.

    Tolerant of the generated format:
        ## CONFIRMED (12)
        - [[HYP-045]] — AUDNZD exclusion (4-pair)
    """
    out: dict[str, list[dict]] = {}
    current: str | None = None
    for raw in text.splitlines():
        header = _SECTION_RE.match(raw)
        if header:
            current = header.group(1).upper()
            out.setdefault(current, [])
            continue
        if current is None:
            continue
        item = _LEDGER_ITEM_RE.match(raw.strip())
        if item:
            hyp_id = item.group(1).strip()
            desc = (item.group(2) or "").strip()
            out[current].append({"id": hyp_id, "desc": desc})
    return out
