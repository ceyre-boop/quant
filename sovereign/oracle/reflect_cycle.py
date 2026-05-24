"""
Oracle Learning Cycle — Phase 2: REFLECT
sovereign/oracle/reflect_cycle.py

Reads last 7 days of harvests + proven_research.json + I_am_a_good_trader.md.
Makes ONE Oracle (Claude haiku) call to propose a candidate lesson.
Cost: ≤2 cents per cycle.

Output: data/oracle/reflections/YYYY_MM_DD.json
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
PROVEN_RESEARCH  = ROOT / "data" / "oracle" / "proven_research.json"
WISDOM_FILE      = ROOT / "I_am_a_good_trader.md"
REFLECTIONS_DIR  = ROOT / "data" / "oracle" / "reflections"
REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)

ORACLE_SCHEMA = {
    "candidate_lesson": {
        "lesson_text": "string — the principle in one sentence",
        "mechanism":   "string — WHY this is true (institutional behavior, not pattern)",
        "testable_rule": "string — exact Python-like condition",
        "expected_impact": "string — which metric improves and by how much",
        "evidence_from_harvest": "string — what in the data supports this",
        "sample_needed": "int — minimum trades to validate statistically"
    },
    "retirement_flag": {
        "lesson_to_review": "string or null — lesson number from I_am_a_good_trader.md",
        "reason": "string or null — why this lesson may be decaying"
    },
    "system_health_note": "string — one sentence on current system state"
}

ORACLE_PROMPT_TEMPLATE = """You are the trading intelligence Oracle for Alta Investments — a two-person quant operation trading FX and equity using a systematic macro strategy.

## Your primary mission
Close the Sharpe gap. {sharpe_summary}
Every suggestion you make must move one needle: Sharpe, win rate, drawdown, or prop challenge readiness.

## The Six Trading Tenets (every suggestion MUST serve at least one)
{tenets}

## Current system state
{bridge_state}

## Next milestones (help get these done)
{next_milestones}

## Hypothesis ledger (what's been tested — do not duplicate)
{hypothesis_summary}

## Research queue (already queued — do not duplicate)
{queue_status}

## Proven research (already known — DO NOT re-propose these)
{proven_summary}

## Current active lessons (already codified — build on these, don't repeat them)
{active_lessons_summary}

## Last 7 days trade data
{harvest_summary}

## Instructions

A GOOD lesson is:
- Mechanistic: explains WHY something happens in terms of institutional behavior, not pattern matching
- Testable: can be expressed as an exact rule applied to trade data
- Specific: names exact thresholds, not vague directions
- Novel: not already in the proven research, active lessons, or hypothesis ledger above
- Goal-oriented: directly addresses the Sharpe gap or a next milestone

A BAD lesson is:
- "The market is uncertain" — not testable
- "Use wider stops" — not specific
- "EURUSD is volatile" — no mechanism
- Anything already confirmed, rejected, or queued in the hypothesis ledger

Examples of good lessons:
- "Trades entered within 3 bars of the London open have 2.1× higher R than trades entered after bar 6 — institutional displacement is concentrated in the first 30 minutes"
- "GBPUSD positions held through the NY_AM session (9-12 ET) give back 40% of London gains on average — the NY continuation bias reverses intraday"

## Rejected hypotheses (NEVER re-propose these)
{rejected_ids}

## Output — respond ONLY in this exact JSON schema, no other text:
{schema}
"""


def _load_proven_summary() -> str:
    if not PROVEN_RESEARCH.exists():
        return "No proven research file found."
    data = json.loads(PROVEN_RESEARCH.read_text())
    proven = data.get("proven_lessons", [])
    rejected = data.get("rejected_hypotheses", [])
    lines = ["PROVEN LESSONS:"]
    for l in proven:
        lines.append(f"  L-{l['id']}: {l['lesson'][:100]}")
    lines.append("\nKEY REJECTIONS (don't re-propose):")
    for r in rejected[:8]:
        lines.append(f"  {r['id']}: {r['finding'][:80]} → {r['reason_rejected'][:60]}")
    return "\n".join(lines)


def _load_active_lessons() -> str:
    if not WISDOM_FILE.exists():
        return "No wisdom file found."
    content = WISDOM_FILE.read_text()
    # Extract lesson summaries
    lessons = re.findall(r"### LESSON (\d+) — ([^\n]+)\n.*?\*\*Rule:\*\* ([^\n]+)", content, re.DOTALL)
    if not lessons:
        return content[:1000]
    return "\n".join(f"  L{n}: {title} | Rule: {rule[:80]}" for n, title, rule in lessons)


def _load_harvest_summary(harvests: list[dict]) -> str:
    if not harvests:
        return "No harvest data available. System may not have trades yet — propose a hypothesis about the existing backtest data instead."
    lines = []
    for h in harvests[:7]:
        fail_dist = h.get("failure_distribution", {})
        dominant = h.get("dominant_failure_mode", "UNKNOWN")
        lines.append(
            f"  {h['date']}: {h['trades_closed']} trades | "
            f"WR={h.get('win_rate', 0)*100:.0f}% | "
            f"avgR={h.get('avg_r', 0):.3f} | "
            f"dominant_fail={dominant} | "
            f"anomalies={len(h.get('anomalies', []))}"
        )
        if h.get("anomalies"):
            for a in h["anomalies"]:
                lines.append(f"    ⚠ {a}")
    return "\n".join(lines)


def _get_rejected_ids() -> str:
    if not PROVEN_RESEARCH.exists():
        return "None recorded"
    data = json.loads(PROVEN_RESEARCH.read_text())
    never_ids = data.get("oracle_instructions", {}).get("never_re_propose", [])
    return ", ".join(never_ids) if never_ids else "None"


def run_reflect(harvests: list[dict], date: Optional[str] = None) -> dict:
    """
    Call Oracle with compact prompt. Returns structured reflection.
    Saves to data/oracle/reflections/YYYY_MM_DD.json.
    Cost: ~1.5-2 cents.
    """
    from sovereign.agent.oracle_agent import _load_dotenv
    _load_dotenv()

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    from sovereign.oracle.system_context import build_system_context
    ctx = build_system_context()

    prompt = ORACLE_PROMPT_TEMPLATE.format(
        sharpe_summary=ctx["sharpe_summary"],
        tenets=ctx["tenets"],
        bridge_state=ctx["bridge_state"],
        next_milestones=ctx["next_milestones"],
        hypothesis_summary=ctx["hypothesis_summary"],
        queue_status=ctx["queue_status"],
        proven_summary=_load_proven_summary(),
        active_lessons_summary=_load_active_lessons(),
        harvest_summary=_load_harvest_summary(harvests),
        rejected_ids=_get_rejected_ids(),
        schema=json.dumps(ORACLE_SCHEMA, indent=2),
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()

    # Parse JSON from response
    try:
        reflection = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON block
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            reflection = json.loads(match.group())
        else:
            reflection = {"raw_response": raw_text, "parse_error": True}

    output = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "estimated_cost_usd": round(
            response.usage.input_tokens * 0.00000025 +
            response.usage.output_tokens * 0.00000125, 6
        ),
        "harvest_days_read": len(harvests),
        "reflection": reflection,
    }

    out_path = REFLECTIONS_DIR / f"{date}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    import argparse
    from sovereign.oracle.harvest_cycle import load_recent_harvests

    parser = argparse.ArgumentParser(description="Oracle reflection cycle")
    parser.add_argument("--date", help="Date (YYYY-MM-DD), default: today")
    parser.add_argument("--days", type=int, default=7, help="Days of harvest to read")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt, don't call API")
    args = parser.parse_args()

    harvests = load_recent_harvests(args.days)

    if args.dry_run:
        from sovereign.oracle.system_context import build_system_context
        ctx = build_system_context()
        print("=== ORACLE PROMPT (dry run) ===")
        print(ORACLE_PROMPT_TEMPLATE.format(
            sharpe_summary=ctx["sharpe_summary"],
            tenets=ctx["tenets"],
            bridge_state=ctx["bridge_state"],
            next_milestones=ctx["next_milestones"],
            hypothesis_summary=ctx["hypothesis_summary"],
            queue_status=ctx["queue_status"],
            proven_summary=_load_proven_summary(),
            active_lessons_summary=_load_active_lessons(),
            harvest_summary=_load_harvest_summary(harvests),
            rejected_ids=_get_rejected_ids(),
            schema=json.dumps(ORACLE_SCHEMA, indent=2),
        ))
    else:
        result = run_reflect(harvests, date=args.date)
        print(f"Reflection complete. Cost: ${result['estimated_cost_usd']:.4f}")
        print(json.dumps(result["reflection"], indent=2))
