"""
Oracle Tier 2 — MICRO CORRECTION
sovereign/oracle/micro_correct.py

Runs every 2 hours. Small Haiku call (~0.5 cents).
Asks 3 targeted questions: lesson violations, parameter adjustments, veto changes.
Queues corrections to data/oracle/pending_corrections/ — never auto-applies.

Cost: ~$0.005/run, ~$0.06/day (12 micro-corrections vs 4)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from sovereign.utils.timestamps import canonical_timestamp
from sovereign.oracle.pulse_check import (
    _load_entries_since,
    _load_pulse_state,
    _save_pulse_state,
    _write_messages,
    PULSE_DIR,
)

ROOT              = Path(__file__).resolve().parents[2]
WISDOM_FILE       = ROOT / "I_am_a_good_trader.md"
ORACLE_CYCLE_LOG  = ROOT / "logs" / "oracle_cycle.log"
MICRO_DIR         = ROOT / "data" / "oracle" / "micro_corrections"
CORRECTIONS_DIR   = ROOT / "data" / "oracle" / "pending_corrections"

MICRO_PROMPT = """You are Oracle running a 6-hour quick check. Be brief.

Last 6 hours of trades:
{trades_json}

Pulse anomalies flagged:
{anomalies_json}

Current prices (from last pulse):
{prices_json}

Active lessons: {lessons_list}

System state: {state}

Answer THREE questions only. Respond ONLY in this exact JSON (no other text):
{{
  "lesson_violations": ["lesson title — trade pair/direction that violated it"],
  "parameter_adjustments": ["field: current → suggested (one sentence reason)"],
  "veto_changes": ["veto name: tighten/suspend (one sentence reason)"],
  "urgent": false
}}

Rules: Empty lists if nothing found. No explanations outside the JSON. Max 3 items per list.
Only flag violations if you see a specific trade that clearly contradicts an active lesson.
Only suggest adjustments grounded in the trade data above, not hypothetical patterns."""


def _load_pulses_since(hours: int) -> list[dict]:
    if not PULSE_DIR.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    pulses = []
    for f in sorted(PULSE_DIR.glob("pulse_*.json")):
        try:
            data = json.loads(f.read_text())
            ts = datetime.fromisoformat(data.get("timestamp", "").replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                pulses.append(data)
        except Exception:
            continue
    return pulses


def _load_active_lessons_compact() -> list[str]:
    if not WISDOM_FILE.exists():
        return []
    content = WISDOM_FILE.read_text()
    lessons = re.findall(r"### LESSON \d+ — ([^\n]+)", content)
    return lessons


def _one_line_system_state() -> str:
    if not ORACLE_CYCLE_LOG.exists():
        return "nominal"
    try:
        lines = ORACLE_CYCLE_LOG.read_text().strip().splitlines()
        return lines[-1][22:].strip() if lines else "nominal"  # strip timestamp prefix
    except Exception:
        return "nominal"


def _queue_corrections(corrections: dict, input_tokens: int, output_tokens: int) -> None:
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = canonical_timestamp()[:16].replace(":", "").replace("-", "").replace("T", "")
    record = {
        "queued_at": canonical_timestamp(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "corrections": corrections,
        "status": "PENDING_HUMAN_REVIEW",
        "note": "Do not auto-apply. Review before touching parameters.yml or live config.",
    }
    (CORRECTIONS_DIR / f"correction_{ts_slug}.json").write_text(json.dumps(record, indent=2))


def run_micro_correction() -> dict:
    """Run 6-hour micro-correction. Returns result dict with corrections or skipped flag."""
    from sovereign.oracle.oracle_agent import _load_dotenv
    _load_dotenv()

    # 6h gate — check inside pulse state
    state = _load_pulse_state()
    last_micro = state.get("last_micro_time")
    if last_micro:
        try:
            dt = datetime.fromisoformat(last_micro)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
            if elapsed < 2 * 3600 - 300:  # 5-min tolerance
                return {"skipped": True, "reason": f"less than 2h since last micro ({elapsed/3600:.1f}h ago)"}
        except Exception:
            pass

    recent = _load_entries_since(datetime.now(timezone.utc) - timedelta(hours=2))
    pulses = _load_pulses_since(hours=2)
    all_anomalies = [a for p in pulses for a in p.get("anomalies", [])]

    # Extract live prices from most recent pulse
    prices_txt = "N/A — no recent price data"
    if pulses:
        lp = pulses[-1].get('live_prices', {})
        if lp:
            prices_txt = '\n'.join(
                f"{pair}: {v['current']} (H: {v['high_today']} L: {v['low_today']})"
                for pair, v in lp.items()
            )

    compact_trades = [
        {k: e[k] for k in ("pair", "direction", "why_this_trade", "commitment_score", "outcome", "r_realized")
         if k in e}
        for e in recent[:10]
    ]

    if not compact_trades and not all_anomalies:
        return {"skipped": True, "reason": "no trades or anomalies in last 2h"}

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": MICRO_PROMPT.format(
            trades_json=json.dumps(compact_trades, indent=2),
            anomalies_json=json.dumps(all_anomalies),
            prices_json=prices_txt,
            lessons_list=", ".join(_load_active_lessons_compact()) or "none yet",
            state=_one_line_system_state(),
        )}],
    )

    raw = response.content[0].text.strip()
    try:
        corrections = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        corrections = json.loads(m.group()) if m else {"parse_error": True, "raw": raw}

    if corrections.get("parameter_adjustments") or corrections.get("veto_changes"):
        _queue_corrections(corrections, response.usage.input_tokens, response.usage.output_tokens)

    if corrections.get("urgent"):
        _write_messages([{
            "type": "ORACLE_URGENT",
            "priority": "URGENT",
            "message": f"Micro-correction flagged urgent: {json.dumps(corrections)}",
        }])

    _save_pulse_state(last_micro_time=canonical_timestamp())

    result = {
        "timestamp": canonical_timestamp(),
        "cost_usd": round(
            response.usage.input_tokens * 0.00000025 +
            response.usage.output_tokens * 0.00000125, 6
        ),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "trades_checked": len(compact_trades),
        "anomalies_checked": len(all_anomalies),
        "corrections": corrections,
    }

    MICRO_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = result["timestamp"][:16].replace(":", "").replace("-", "").replace("T", "")
    (MICRO_DIR / f"micro_{ts_slug}.json").write_text(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    result = run_micro_correction()
    if result.get("skipped"):
        print(f"Skipped: {result['reason']}")
    else:
        print(f"Micro-correction complete. Cost: ${result['cost_usd']:.4f}")
        print(json.dumps(result["corrections"], indent=2))
