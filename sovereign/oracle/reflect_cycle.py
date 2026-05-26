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
DECISION_LOG_DIR = ROOT / "data" / "decision_logs"
REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)

ORACLE_SCHEMA = {
    "candidate_lesson": {
        "lesson_text": "string — the principle in one sentence",
        "mechanism":   "string — WHY this is true (institutional behavior, not pattern)",
        "testable_rule": "string — exact Python-like condition",
        "expected_impact": "string — which metric improves and by how much",
        "evidence_from_harvest": "string — what in the data supports this",
        "evidence_from_reasoning": "string or null — which decision log reasoning components drove this lesson (e.g. 'trades with commitment_score < 0.70 and rate_diff_z < 1.2 failed at 80% rate')",
        "reasoning_component_targeted": "string or null — the specific decision field this lesson gates on (e.g. 'commitment_score', 'rate_differential_zscore', 'bars_since_signal', 'library_match')",
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

## Decision logs — reasoning behind each recent trade
These are the actual reasons the system chose to enter and how it sized each position.
The `outcome` and `r_realized` fields show what happened.
Look for patterns in the reasoning that PRECEDE failures.

{decision_log_summary}

## Instructions

You now have two inputs: aggregate harvest stats (what happened) and decision logs (why the system thought it should trade). The most valuable lessons come from connecting those two layers.

When you propose a candidate lesson, look for patterns in the reasoning chain that correlate with failure:
- Do trades with a specific `commitment_score` threshold fail at a higher rate?
- Do trades where `bars_since_signal` is high (stale entry) underperform fresh entries?
- Does a `library_match` at high similarity correlate with better or worse R?
- Do trades with `rate_differential_zscore` below a threshold fail even with high commitment?

Your lesson should reference the specific reasoning component that drives the pattern.

Not: "trades fail when commitment is low"
But: "trades with commitment_score < 0.70 that ALSO have rate_diff_z < 1.2 produce -0.XX avg R — the macro thesis is too weak to compensate for uncommitted price action"

A GOOD lesson is:
- Mechanistic: explains WHY this reasoning component predicts failure (institutional behavior)
- Testable: names the exact field and threshold from the decision log schema
- Novel: not already in the proven research, active lessons, or hypothesis ledger above
- Grounded: references actual entries from the decision logs above, not hypothetical ones

A BAD lesson is:
- "The market is uncertain" — not testable
- "Use wider stops" — not specific, not grounded in decision log fields
- Anything already confirmed, rejected, or queued in the hypothesis ledger
- A lesson about reasoning components if there are no decision log entries yet (fall back to harvest-only lessons in that case)

Examples of good reasoning-grounded lessons:
- "Trades with bars_since_signal > 3 produce -0.3R avg vs +0.6R for bars_since_signal ≤ 2 — the FVG has partially filled by bar 3 and the limit entry gets a worse fill"
- "rate_diff_z < 1.0 with commitment_score > 0.80 fails 75% of the time — commitment score captures price action quality, not macro strength; both must pass independently"

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


def _load_decision_log_summary(days: int = 7, max_entries: int = 20) -> str:
    """
    Load recent decision log entries that have outcomes filled.
    Returns a compact JSON block for the Oracle prompt.
    Caps at max_entries to keep token cost bounded (~$0.005 per 20 entries).
    """
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []

    if not DECISION_LOG_DIR.exists():
        return "No decision logs yet — system just wired. Logs will accumulate from this session forward."

    for log_file in sorted(DECISION_LOG_DIR.glob("decisions_*.jsonl")):
        try:
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Only include closed trades (outcome filled)
                if not rec.get("outcome"):
                    continue
                try:
                    ts = datetime.fromisoformat(rec["entry_timestamp"].replace("Z", "+00:00"))
                    if ts < cutoff:
                        continue
                except Exception:
                    pass
                # Compact projection — only fields Oracle needs for pattern analysis
                entries.append({
                    "pair":                    rec.get("pair"),
                    "system":                  rec.get("system"),
                    "direction":               rec.get("direction"),
                    "grade":                   rec.get("grade"),
                    "session":                 rec.get("session"),
                    "why_this_trade":          rec.get("why_this_trade"),
                    "why_this_size":           rec.get("why_this_size"),
                    "signal_layers_active":    rec.get("signal_layers_active"),
                    "commitment_score":        rec.get("commitment_score"),
                    "vix_at_entry":            rec.get("vix_at_entry"),
                    "rate_differential_zscore": rec.get("rate_differential_zscore"),
                    "bars_since_signal":       rec.get("bars_since_signal"),
                    "library_match":           rec.get("library_match"),
                    "risk_pct":                rec.get("risk_pct"),
                    "outcome":                 rec.get("outcome"),
                    "r_realized":              rec.get("r_realized"),
                })
        except Exception:
            continue

    if not entries:
        return (
            "No closed decision log entries in the last 7 days yet. "
            "The logger is wired and accumulating — entries will appear as trades close. "
            "Propose a harvest-based lesson instead."
        )

    # Newest first, cap at max_entries
    entries = entries[-max_entries:]
    n_wins  = sum(1 for e in entries if (e.get("r_realized") or 0) > 0)
    n_loss  = len(entries) - n_wins
    avg_r   = round(sum((e.get("r_realized") or 0) for e in entries) / len(entries), 3)

    header = (
        f"Recent decision logs: {len(entries)} closed trades "
        f"({n_wins}W / {n_loss}L, avg R={avg_r:+.3f})\n"
    )
    return header + json.dumps(entries, indent=2)


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
    from sovereign.oracle.oracle_agent import _load_dotenv
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
        decision_log_summary=_load_decision_log_summary(),
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
            decision_log_summary=_load_decision_log_summary(),
            rejected_ids=_get_rejected_ids(),
            schema=json.dumps(ORACLE_SCHEMA, indent=2),
        ))
    else:
        result = run_reflect(harvests, date=args.date)
        print(f"Reflection complete. Cost: ${result['estimated_cost_usd']:.4f}")
        print(json.dumps(result["reflection"], indent=2))
