"""
Oracle Learning Cycle — Phase 4: CODIFY
sovereign/oracle/codify_cycle.py

When verdict == VALIDATED:
  - Adds lesson to I_am_a_good_trader.md
  - Enforces 10-lesson maximum (retires weakest to I_was_a_good_trader.md)
  - Generates Claude Code implementation prompt
  - Updates proven_research.json
  - Updates hypothesis_ledger.json

Cost: $0.00 (no LLM call — pure file writes)

Output:
  - I_am_a_good_trader.md (updated)
  - I_was_a_good_trader.md (if retirement occurred)
  - data/oracle/pending_implementations/YYYY_MM_DD.md
  - data/oracle/proven_research.json (updated)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
WISDOM_FILE     = ROOT / "I_am_a_good_trader.md"
ARCHIVE_FILE    = ROOT / "I_was_a_good_trader.md"
PROVEN_RESEARCH = ROOT / "data" / "oracle" / "proven_research.json"
IMPL_DIR        = ROOT / "data" / "oracle" / "pending_implementations"
IMPL_DIR.mkdir(parents=True, exist_ok=True)

MAX_ACTIVE_LESSONS = 10


def _count_active_lessons() -> int:
    if not WISDOM_FILE.exists():
        return 0
    content = WISDOM_FILE.read_text()
    return len(re.findall(r"### LESSON \d+", content))


def _get_lesson_delta_sharpes() -> dict[int, float]:
    """Extract delta Sharpe from each lesson for retirement ranking."""
    if not WISDOM_FILE.exists():
        return {}
    content = WISDOM_FILE.read_text()
    result = {}
    for match in re.finditer(r"### LESSON (\d+).*?\*\*Evidence:\*\* ([^\n]+)", content, re.DOTALL):
        num = int(match.group(1))
        evidence = match.group(2)
        # Try to extract delta sharpe from evidence text
        sharpe_match = re.search(r"delta.*?(\d+\.\d+)", evidence, re.IGNORECASE)
        if sharpe_match:
            result[num] = float(sharpe_match.group(1))
        else:
            result[num] = 0.05  # default if not parseable
    return result


def _get_weakest_lesson_num(delta_sharpes: dict[int, float]) -> Optional[int]:
    if not delta_sharpes:
        return None
    return min(delta_sharpes, key=lambda k: delta_sharpes[k])


def _extract_lesson_block(lesson_num: int) -> str:
    """Extract the full text of a lesson block from I_am_a_good_trader.md."""
    content = WISDOM_FILE.read_text()
    pattern = rf"(### LESSON {lesson_num} —.*?)(?=### LESSON \d+|---\n\n\*Next lesson|$)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""


def _remove_lesson_block(lesson_num: int) -> None:
    """Remove a lesson block from I_am_a_good_trader.md."""
    content = WISDOM_FILE.read_text()
    pattern = rf"### LESSON {lesson_num} —.*?(?=### LESSON \d+|---\n\n\*Next lesson|$)"
    new_content = re.sub(pattern, "", content, flags=re.DOTALL)
    WISDOM_FILE.write_text(new_content)


def retire_lesson(lesson_num: int, reason: str, latest_delta_sharpe: Optional[float] = None) -> None:
    """Move lesson from wisdom file to archive."""
    lesson_text = _extract_lesson_block(lesson_num)
    if not lesson_text:
        return

    # Extract original delta sharpe from lesson text
    orig_match = re.search(r"delta.*?(\d+\.\d+)", lesson_text, re.IGNORECASE)
    orig_sharpe = float(orig_match.group(1)) if orig_match else 0.0

    retirement_block = f"""
### RETIRED LESSON {lesson_num} — {reason}
**Retired:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
**Retirement reason:** {reason}
**Original evidence:**
{lesson_text}
**Final delta Sharpe:** {latest_delta_sharpe or 'not measured'}
**Original delta Sharpe:** {orig_sharpe}
**Decay ratio:** {round(latest_delta_sharpe / orig_sharpe, 3) if latest_delta_sharpe and orig_sharpe else 'N/A'}

---
"""

    # Append to archive
    archive = ARCHIVE_FILE.read_text() if ARCHIVE_FILE.exists() else ""
    # Remove "no retired lessons" placeholder
    archive = archive.replace("*No retired lessons yet. The system is 7 active lessons into its learning journey.*\n", "")
    ARCHIVE_FILE.write_text(archive + retirement_block)

    # Remove from wisdom file
    _remove_lesson_block(lesson_num)

    # Update active count line in wisdom file
    content = WISDOM_FILE.read_text()
    content = re.sub(r"\*\*Active lessons: \d+", f"**Active lessons: {_count_active_lessons()}", content)
    WISDOM_FILE.write_text(content)


def _write_lesson_to_wisdom(
    lesson_num: int,
    validation: dict,
    reflection: dict,
) -> None:
    candidate = reflection.get("reflection", {}).get("candidate_lesson", {})
    lesson_text = candidate.get("lesson_text", "")
    mechanism   = candidate.get("mechanism", candidate.get("lesson_text", ""))
    rule        = candidate.get("testable_rule", "")
    impact      = candidate.get("expected_impact", "")
    date_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    block = f"""
### LESSON {lesson_num} — {lesson_text[:60]}

**Discovered:** {validation.get('date', date_str)} (Oracle reflection)
**Validated:** {date_str}
**Evidence:** delta_sharpe={validation.get('test_a_delta_sharpe')}, p={validation.get('test_b_p_value')}, holdout_ratio={validation.get('test_c_replication_ratio')}
**Rule:** `{rule}`
**Impact:** {impact}
**Code location:** PENDING — see data/oracle/pending_implementations/{date_str}.md
**Health:** 🟢 ACTIVE
**Last validated:** {date_str}
**Linked hypothesis:** AUTO-{date_str}

*The mechanism:* {mechanism}

---
"""

    # Insert before the footer
    content = WISDOM_FILE.read_text()
    footer_marker = "*Next lesson will retire"
    if footer_marker in content:
        content = content.replace(f"*Next lesson will retire", block + "\n*Next lesson will retire")
    else:
        content += block

    # Update active count
    n_active = _count_active_lessons() + 1
    content = re.sub(r"\*\*Active lessons: \d+ / \d+\*\*", f"**Active lessons: {n_active} / {MAX_ACTIVE_LESSONS}**", content)
    content = re.sub(r"\*\*Last updated: [^\*]+\*\*", f"**Last updated: {date_str}**", content)

    WISDOM_FILE.write_text(content)


def _generate_implementation_prompt(validation: dict, reflection: dict, lesson_num: int) -> str:
    candidate = reflection.get("reflection", {}).get("candidate_lesson", {})
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return f"""# Claude Code Implementation Prompt — LESSON {lesson_num}
## Generated: {date_str} | Status: AWAITING COLIN REVIEW

**This prompt was auto-generated by the Oracle after statistical validation.**
**Colin must review and approve before implementation.**

---

## The Validated Lesson

**Lesson text:** {candidate.get('lesson_text', '')}

**Mechanism:** {candidate.get('mechanism', candidate.get('lesson_text', ''))}

**Rule:** `{candidate.get('testable_rule', '')}`

**Expected impact:** {candidate.get('expected_impact', '')}

## Statistical Evidence

- Test A (effect size): delta_sharpe = {validation.get('test_a_delta_sharpe')} (threshold: 0.05) {'✓' if validation.get('test_a_passed') else '✗'}
- Test B (significance): p = {validation.get('test_b_p_value')} (threshold: 0.05) {'✓' if validation.get('test_b_passed') else '✗'}
- Test C (holdout): ratio = {validation.get('test_c_replication_ratio')} (threshold: 0.80) {'✓' if validation.get('test_c_passed') else '✗'}

Trades tested: {validation.get('trades_tested')} | Trades filtered: {validation.get('trades_filtered')}

## Implementation Task for Claude Code

1. Locate the appropriate enforcement point in the codebase:
   - ICT rules → `ict/pipeline.py`
   - Forex sizing → `sovereign/forex/signal_engine.py`
   - Cross-system → `sovereign/intelligence/cross_system_bridge.py`
   - Risk management → `sovereign/risk/`

2. Implement the rule as a gate or size modifier:
   ```python
   # Rule to implement:
   # {candidate.get('testable_rule', '')}
   ```

3. Add the rule with a clear comment referencing LESSON {lesson_num}

4. Run existing tests: `python3 -m pytest tests/ -v`

5. Run the full forex backtest to confirm no regression:
   ```
   PYTHONPATH=. .venv/bin/python3 scripts/plot_research_brief.py
   ```

6. Update `I_am_a_good_trader.md` LESSON {lesson_num} Code Location field

7. Commit with message: `Lesson {lesson_num}: [brief description] — Oracle validated {date_str}`

## What NOT to do
- Do not implement if you disagree with the mechanism — flag it to Colin first
- Do not change live trading parameters without Colin's confirmation
- Do not skip the backtest regression check

---
*This file will be deleted after implementation. Archive in git commit message.*
"""


def run_codify(validation: dict, reflection: dict, date: Optional[str] = None) -> dict:
    """
    Codify a validated lesson. Returns summary of actions taken.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if validation.get("verdict") != "VALIDATED":
        return {
            "action": "SKIPPED",
            "reason": f"Verdict was {validation.get('verdict')}, not VALIDATED",
            "date": date,
        }

    n_active = _count_active_lessons()
    retired = None

    # Enforce 10-lesson maximum
    if n_active >= MAX_ACTIVE_LESSONS:
        delta_sharpes = _get_lesson_delta_sharpes()
        weakest = _get_weakest_lesson_num(delta_sharpes)
        if weakest:
            retire_lesson(weakest, reason="SUPERSEDED — new lesson takes its slot (max 10 rule)")
            retired = weakest
            n_active -= 1

    lesson_num = n_active + 1
    _write_lesson_to_wisdom(lesson_num, validation, reflection)

    # Generate implementation prompt
    impl_text = _generate_implementation_prompt(validation, reflection, lesson_num)
    impl_path = IMPL_DIR / f"{date}.md"
    impl_path.write_text(impl_text)

    # Update proven_research.json
    if PROVEN_RESEARCH.exists():
        research = json.loads(PROVEN_RESEARCH.read_text())
        candidate = reflection.get("reflection", {}).get("candidate_lesson", {})
        new_lesson = {
            "id": f"L-AUTO-{date}",
            "linked_hyp": f"AUTO-{date}",
            "lesson": candidate.get("lesson_text", ""),
            "mechanism": candidate.get("mechanism", ""),
            "rule": candidate.get("testable_rule", ""),
            "evidence": f"delta_sharpe={validation.get('test_a_delta_sharpe')} p={validation.get('test_b_p_value')}",
            "delta_sharpe": validation.get("test_a_delta_sharpe"),
            "discovered": date,
            "codified": date,
            "code_location": "PENDING",
            "health": "ACTIVE",
            "lesson_number_in_file": lesson_num,
        }
        research.setdefault("proven_lessons", []).append(new_lesson)
        research["_meta"]["last_updated"] = date
        PROVEN_RESEARCH.write_text(json.dumps(research, indent=2))

    return {
        "action": "CODIFIED",
        "lesson_number": lesson_num,
        "retired_lesson": retired,
        "implementation_prompt": str(impl_path),
        "date": date,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oracle codify cycle")
    parser.add_argument("--validation", required=True, help="Path to validation JSON")
    parser.add_argument("--reflection", required=True, help="Path to reflection JSON")
    parser.add_argument("--date", help="Date override (YYYY-MM-DD)")
    args = parser.parse_args()

    val = json.loads(Path(args.validation).read_text())
    ref = json.loads(Path(args.reflection).read_text())
    result = run_codify(val, ref, date=args.date)
    print(json.dumps(result, indent=2))
