"""
sovereign/oracle/system_context.py
Loads goal-oriented system context for Oracle's reflect cycle.
Keeps Oracle aware of: current Sharpe, v-target gap, six tenets,
hypothesis ledger, queue status, bridge/regime state, prop gates.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CLAUDE_MD        = ROOT / "CLAUDE.md"
TRADING_PHIL     = ROOT / "TRADING_PHILOSOPHY.md"
HYPOTHESIS_LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
RESEARCH_QUEUE   = ROOT / "data" / "agent" / "research_queue.json"
BRIDGE_STATE     = ROOT / "data" / "forensics" / "cross_system_state.json"
SUGGESTIONS      = ROOT / "data" / "agent" / "suggestions.json"

# Six tenets — compact version (avoids re-reading full doc each cycle)
_TENETS_COMPACT = """1. Statistical Utility > Narrative Coherence — features must pass formal gates (IC>0.15 OOS, walk-forward, holdout).
2. Regime Appropriateness > Strategy Quality — right system at the right time; capital_allocator sizes by regime.
3. Systems Must Know When Unreliable — health metrics + consecutive-loss trigger UNRELIABLE freeze.
4. Premature Complexity Kills — every new component must answer a measurable question first.
5. Orchestration Is The Durable Edge — cross-system bridge + capital allocator outlast individual strategies.
6. Research Debt Is Existential Risk — every feature is either LIVE or GRAVEYARD; graveyard is permanent."""


def _extract_sharpe_status() -> dict:
    """Parse current Sharpe, version, and target from CLAUDE.md."""
    result = {
        "current_version": "v013",
        "current_sharpe": 1.8552,
        "target_version": "v014",
        "target_sharpe": 1.9052,
        "gap": 0.05,
    }
    if not CLAUDE_MD.exists():
        return result
    text = CLAUDE_MD.read_text()
    # Try to find version tracker lines
    m = re.search(r"(v\d+): [\d.]+ → ([\d.]+) ← CURRENT", text)
    if m:
        result["current_sharpe"] = float(m.group(2))
    m_ver = re.search(r"(v\d+): [\d.]+ → [\d.]+ ← CURRENT LIVE", text)
    if m_ver:
        result["current_version"] = m_ver.group(1)
    # target
    m_tgt = re.search(r"Target.*?Sharpe > ([\d.]+).*?\(v(\w+) gate", text)
    if m_tgt:
        result["target_sharpe"] = float(m_tgt.group(1))
        result["target_version"] = f"v{m_tgt.group(2)}"
        result["gap"] = round(result["target_sharpe"] - result["current_sharpe"], 4)
    return result


def _load_hypothesis_summary() -> str:
    if not HYPOTHESIS_LEDGER.exists():
        return "Hypothesis ledger not found."
    try:
        data = json.loads(HYPOTHESIS_LEDGER.read_text())
        ledger = data.get("ledger", [])
    except Exception:
        return "Could not parse hypothesis ledger."

    confirmed = [h for h in ledger if h["status"] == "CONFIRMED"]
    rejected  = [h for h in ledger if h["status"] == "REJECTED"]
    candidate = [h for h in ledger if h["status"] in ("CANDIDATE", "TESTING", "QUEUED")]

    lines = [f"Total: {len(ledger)} | Confirmed: {len(confirmed)} | Rejected: {len(rejected)} | In-flight: {len(candidate)}"]
    lines.append("\nRecent CONFIRMED (last 5):")
    for h in confirmed[-5:]:
        lines.append(f"  {h['id']} — {h['name']}: {h.get('result','')[:70]}")
    lines.append("\nRecent REJECTED (last 5):")
    for h in rejected[-5:]:
        lines.append(f"  {h['id']} — {h['name']}: {h.get('result','')[:70]}")
    if candidate:
        lines.append("\nCurrently testing/queued:")
        for h in candidate[-5:]:
            lines.append(f"  {h['id']} — {h['name']}")
    return "\n".join(lines)


def _load_queue_status() -> str:
    if not RESEARCH_QUEUE.exists():
        return "Research queue not found."
    try:
        data = json.loads(RESEARCH_QUEUE.read_text())
        queue = data.get("queue", []) if isinstance(data, dict) else data
    except Exception:
        return "Could not parse research queue."

    queued = [t for t in queue if t.get("status") == "QUEUED"]
    done   = [t for t in queue if t.get("status") == "DONE"]
    lines  = [f"Queue: {len(queued)} pending | {len(done)} done"]
    if queued:
        lines.append("Pending (don't re-suggest these):")
        for t in queued[:5]:
            lines.append(f"  {t['id']}: {t.get('description','')[:70]}")
    return "\n".join(lines)


def _load_bridge_state() -> str:
    if not BRIDGE_STATE.exists():
        return "Bridge state not found — assume NORMAL."
    try:
        state = json.loads(BRIDGE_STATE.read_text())
        threat = state.get("threat_score", 0.0)
        regime = state.get("regime", "UNKNOWN")
        ict_mode = state.get("ict_mode", "UNKNOWN")
        advisory = state.get("advisory", "")
        return (
            f"Threat: {threat:.2f} | Regime: {regime} | ICT mode: {ict_mode}\n"
            f"Advisory: {advisory}"
        )
    except Exception:
        return "Could not parse bridge state."


def _load_next_milestones() -> str:
    if not CLAUDE_MD.exists():
        return ""
    text = CLAUDE_MD.read_text()
    m = re.search(r"Next milestones:(.*?)(?:\n---|\Z)", text, re.DOTALL)
    if not m:
        return ""
    block = m.group(1).strip()
    lines = [l.strip() for l in block.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines[:6])


def build_system_context() -> dict:
    """Return all context as a dict for injection into Oracle prompt."""
    sharpe = _extract_sharpe_status()
    return {
        "tenets": _TENETS_COMPACT,
        "sharpe": sharpe,
        "sharpe_summary": (
            f"Current: {sharpe['current_version']} Sharpe={sharpe['current_sharpe']} | "
            f"Target: {sharpe['target_version']} Sharpe={sharpe['target_sharpe']} | "
            f"Gap remaining: +{sharpe['gap']:.4f}"
        ),
        "hypothesis_summary": _load_hypothesis_summary(),
        "queue_status": _load_queue_status(),
        "bridge_state": _load_bridge_state(),
        "next_milestones": _load_next_milestones(),
    }
