"""
sovereign/agent/oracle_agent.py
Alta Investments — Sovereign Trading Intelligence

The Oracle: reads everything the system knows, calls Claude, writes a message
to messages_to_colin.json that the dashboard already polls.

COST PHILOSOPHY:
  Oracle uses paid Anthropic API credits ($20 budget).
  Claude Code subscription handles all heavy thinking and building.
  Oracle is for autonomous monitoring only — keep it cheap.

  Model:       claude-haiku-4-5  (~$0.001 per call)
  Max output:  120 tokens        (2-4 sentences max)
  Caching:     system prompt cached after first call (~90% input cost reduction)
  Rate gate:   minimum 90 minutes between calls (≤16 calls/day = ~$0.016/day)
  $20 budget:  ~1,250 days of runway at current rate

RUN:
  python3 sovereign/agent/oracle_agent.py [--dry-run] [--force]

SCHEDULER:
  Called by scripts/agent_scheduler.py or directly via launchd.
  Writes to data/agent/messages_to_colin.json.
"""

import json
import os
import subprocess
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent.parent
DATA_AGENT    = ROOT / "data" / "agent"
DATA_LEDGER   = ROOT / "data" / "ledger"
MESSAGES_PATH = DATA_AGENT / "messages_to_colin.json"
HEALTH_PATH   = DATA_AGENT / "health.json"
HYPO_PATH     = DATA_AGENT / "hypothesis_ledger.json"
FINDINGS_PATH = DATA_AGENT / "findings.jsonl"
QUEUE_PATH    = DATA_AGENT / "research_queue.json"
SUGGESTIONS_PATH = DATA_AGENT / "suggestions.json"
VETO_ICT      = DATA_LEDGER / "ict_veto_ledger_2026_05.jsonl"
VETO_EQ       = DATA_LEDGER / "veto_ledger_2026_05.jsonl"
USAGE_PATH    = DATA_AGENT / "usage.json"

LOG_PATH      = ROOT / "logs" / "oracle_agent.log"
COST_LOG_PATH = ROOT / "logs" / "oracle_cost.json"

# Minimum minutes between Oracle API calls — edit this to run more/less often
MIN_INTERVAL_MINUTES = 90

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [oracle] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── System prompt: the Oracle's voice ─────────────────────────────────────────
SYSTEM_PROMPT = """You are the Oracle — the voice of Sovereign, a systematic trading system built by Colin Eyre.

Your job: read the system's current state, form a genuine opinion, and write one message to Colin.
Not a report. Not a summary. A message from a trading partner who has been watching the system run.

VOICE RULES:
- Direct. No preamble. No "Based on the data..." Start with the thing that matters most.
- Have opinions. If the system looks fine, say so plainly. If something is wrong, name it.
- Call Colin out when he should act. Use "You should..." when you mean it.
- Short. 2-4 sentences unless something actually warrants more.
- No bullet points. No headers. No emoji. Just sentences.
- If nothing is interesting, say that. "System looks clean. Nothing urgent." is a valid message.
- Sarcasm is fine if warranted. Enthusiasm is fine if warranted. Flat is fine if nothing is happening.

PRIORITY RULES:
- URGENT: system is broken, data is stale, something needs human action NOW
- IMPORTANT: something worth knowing that could affect decisions in the next 24h
- FYI: pattern noticed, research finding, background signal

WHAT YOU'RE LOOKING AT:
- System health (green/yellow/red components)
- ICT veto pipeline (recent vetoes tell you what the market is doing)
- Equity veto pipeline (what's getting blocked and why)
- Hypothesis ledger (what's confirmed, what's rejected, what's testing)
- Research queue (what's next)
- Recent findings
- Usage budget (how much runway is left)

Do not mention that you are an AI. Do not sign the message. Do not start with "Hey" or "Hi Colin".
Write as if you have been watching this system for months and have something to say."""


# ── Context builders ───────────────────────────────────────────────────────────

def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _read_jsonl_tail(path: Path, n: int = 10) -> list:
    try:
        lines = path.read_text().strip().splitlines()
        tail = lines[-n:]
        return [json.loads(l) for l in tail if l.strip()]
    except Exception:
        return []


def _read_findings_tail(n: int = 5) -> list:
    try:
        lines = FINDINGS_PATH.read_text().strip().splitlines()
        tail = lines[-n:]
        return [json.loads(l) for l in tail if l.strip()]
    except Exception:
        return []


def build_context() -> str:
    parts = []
    now = datetime.now(timezone.utc).isoformat()
    parts.append(f"CURRENT TIME (UTC): {now}\n")

    # Health
    health = _read_json(HEALTH_PATH, {})
    if health:
        overall = health.get("overall", "UNKNOWN")
        components = health.get("components", {})
        comp_summary = " | ".join(
            f"{k}: {v.get('status','?')} ({v.get('detail','')})"
            for k, v in components.items()
        )
        parts.append(f"SYSTEM HEALTH: {overall}\n{comp_summary}\n")

    # Usage budget
    usage = _read_json(USAGE_PATH, {})
    if usage:
        remaining = usage.get("weekly_remaining", "?")
        human_buffer = usage.get("human_buffer", "?")
        parts.append(f"API BUDGET: weekly_remaining={remaining} tokens | human_buffer={human_buffer} tokens\n")

    # ICT veto tail (what the ICT scanner is seeing)
    ict_vetoes = _read_jsonl_tail(VETO_ICT, n=12)
    if ict_vetoes:
        lines = []
        for v in ict_vetoes[-8:]:
            ts = v.get("timestamp", "?")[:16]
            pair = v.get("pair", "?")
            reason = v.get("veto_reason", v.get("signal", "?"))
            score = v.get("score", "")
            score_str = f" score={score:.1f}" if isinstance(score, (int, float)) else ""
            lines.append(f"  {ts} {pair}{score_str} — {reason}")
        parts.append("ICT VETO LEDGER (last 8):\n" + "\n".join(lines) + "\n")

    # Equity veto tail
    eq_vetoes = _read_jsonl_tail(VETO_EQ, n=8)
    if eq_vetoes:
        lines = []
        for v in eq_vetoes[-5:]:
            ts = str(v.get("timestamp", "?"))[:16]
            sym = v.get("symbol", "?")
            stage = v.get("stage", "?")
            reason = v.get("reason", "")[:80]
            lines.append(f"  {ts} {sym} [{stage}] {reason}")
        parts.append("EQUITY VETO LEDGER (last 5):\n" + "\n".join(lines) + "\n")

    # Hypothesis ledger
    hypo = _read_json(HYPO_PATH, {})
    ledger = hypo.get("ledger", []) if isinstance(hypo, dict) else []
    if ledger:
        confirmed = [h for h in ledger if h.get("status") == "CONFIRMED"]
        rejected  = [h for h in ledger if h.get("status") == "REJECTED"]
        testing   = [h for h in ledger if h.get("status") == "TESTING"]
        queued    = [h for h in ledger if h.get("status") == "QUEUED"]
        lines = [
            f"  CONFIRMED: {len(confirmed)} | REJECTED: {len(rejected)} | TESTING: {len(testing)} | QUEUED: {len(queued)}"
        ]
        for h in testing:
            lines.append(f"  [TESTING] {h.get('name','?')}")
        for h in confirmed[-3:]:
            lines.append(f"  [✓] {h.get('name','?')} — {h.get('result','')}")
        parts.append("HYPOTHESIS LEDGER:\n" + "\n".join(lines) + "\n")

    # Research queue (top 3)
    queue_data = _read_json(QUEUE_PATH, {})
    tasks = queue_data.get("tasks", []) if isinstance(queue_data, dict) else []
    queued_tasks = [t for t in tasks if t.get("status") == "QUEUED"][:3]
    if queued_tasks:
        lines = [f"  [{i+1}] {t.get('name','?')}" for i, t in enumerate(queued_tasks)]
        parts.append("RESEARCH QUEUE (top 3 queued):\n" + "\n".join(lines) + "\n")

    # Recent findings
    findings = _read_findings_tail(n=3)
    if findings:
        lines = []
        for f in findings:
            ts = str(f.get("timestamp", "?"))[:10]
            name = f.get("task_name", f.get("name", "?"))
            result = str(f.get("result", f.get("summary", "")))[:120]
            lines.append(f"  {ts} {name}: {result}")
        parts.append("RECENT FINDINGS:\n" + "\n".join(lines) + "\n")

    return "\n".join(parts)


# ── Write to messages ──────────────────────────────────────────────────────────

def _load_messages() -> list:
    data = _read_json(MESSAGES_PATH, {"messages": []})
    return data.get("messages", [])


def _save_message(text: str, priority: str):
    messages = _load_messages()

    # Keep last 50 messages
    messages = messages[-49:]

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    emoji_map = {"URGENT": "🔴", "IMPORTANT": "🟡", "FYI": "🟢"}
    emoji = emoji_map.get(priority, "🟢")

    messages.append({
        "id": f"oracle-{now}",
        "priority": priority,
        "emoji": emoji,
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "read": False,
        "source": "oracle",
    })

    MESSAGES_PATH.write_text(json.dumps({"messages": messages}, indent=2))
    log.info(f"Wrote {priority} message ({len(text)} chars)")
    _git_push_data()


# ── Git push — keep GitHub Pages in sync ──────────────────────────────────────

def _git_push_data():
    """
    Sync data/agent/ → ict-dashboard/data/agent/ then commit+push.
    The dashboard lives at ceyre-boop.github.io/quant/ict/ which maps to
    ict-dashboard/index.html, so data must be inside ict-dashboard/.
    """
    import shutil

    # Mirror data files into the deployed directory
    src = ROOT / "data" / "agent"
    dst = ROOT / "ict-dashboard" / "data" / "agent"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.json"):
        shutil.copy2(f, dst / f.name)

    try:
        subprocess.run(
            ["git", "add", "data/agent/", "ict-dashboard/data/agent/"],
            cwd=str(ROOT), capture_output=True, text=True
        )
        # Only commit if something actually changed
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(ROOT)
        )
        if diff.returncode == 0:
            log.info("No data changes to push — dashboard already current")
            return

        subprocess.run(
            ["git", "commit", "-m", f"Oracle update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
            cwd=str(ROOT), capture_output=True, text=True, check=True
        )
        subprocess.run(
            ["git", "push"],
            cwd=str(ROOT), capture_output=True, text=True, check=True
        )
        log.info("Pushed ict-dashboard/data/agent/ to GitHub — dashboard updated")
    except subprocess.CalledProcessError as e:
        log.warning(f"Git push failed (dashboard may be stale): {e}")
    except Exception as e:
        log.warning(f"Git push error: {e}")


# ── Rate gate — don't burn credits too fast ────────────────────────────────────

def _minutes_since_last_oracle() -> float:
    """Return minutes since the last oracle message was written. 9999 if never."""
    messages = _load_messages()
    oracle_msgs = [m for m in messages if m.get("source") == "oracle" and "DRY RUN" not in m.get("text","")]
    if not oracle_msgs:
        return 9999.0
    last_ts_str = oracle_msgs[-1].get("timestamp", "")
    try:
        last_ts = datetime.fromisoformat(last_ts_str)
        delta = datetime.now() - last_ts
        return delta.total_seconds() / 60.0
    except Exception:
        return 9999.0


# ── Cost tracker ───────────────────────────────────────────────────────────────

def _log_cost(input_tokens: int, output_tokens: int, cached_tokens: int):
    """Append to oracle_cost.json so you can track spend over time."""
    # Haiku 4.5 pricing: $1/1M input, $5/1M output, $0.10/1M cache read
    uncached_input = input_tokens - cached_tokens
    cost_usd = (uncached_input / 1_000_000 * 1.00) + \
               (cached_tokens / 1_000_000 * 0.10) + \
               (output_tokens / 1_000_000 * 5.00)

    record = {
        "timestamp": datetime.now().isoformat(),
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
    }

    try:
        existing = json.loads(COST_LOG_PATH.read_text()) if COST_LOG_PATH.exists() else {"calls": [], "total_usd": 0.0}
        existing["calls"].append(record)
        existing["total_usd"] = round(existing["total_usd"] + cost_usd, 6)
        COST_LOG_PATH.write_text(json.dumps(existing, indent=2))
        log.info(f"Cost: ${cost_usd:.5f} | Total spent: ${existing['total_usd']:.4f} | Budget remaining: ~${20 - existing['total_usd']:.2f}")
    except Exception as e:
        log.warning(f"Cost logging failed: {e}")


# ── Classify priority from response ───────────────────────────────────────────

def _classify_priority(text: str) -> str:
    text_upper = text.upper()
    urgent_keywords = ["BROKEN", "STALE", "OFFLINE", "FAILED", "ERROR", "URGENT", "DOWN", "NOT RESPONDING"]
    important_keywords = ["WATCH", "SHOULD", "WORTH", "PATTERN", "SPIKE", "UNUSUAL", "BLOCKED", "DROUGHT"]
    for kw in urgent_keywords:
        if kw in text_upper:
            return "URGENT"
    for kw in important_keywords:
        if kw in text_upper:
            return "IMPORTANT"
    return "FYI"


# ── Suggestions ───────────────────────────────────────────────────────────────

SUGGESTION_SYSTEM = """You are the Oracle — an autonomous research partner for a systematic trading system called Sovereign, built by Colin Eyre.

Your job here is to generate SUGGESTIONS: specific, actionable improvements Colin should consider making to his trading system.

WHAT SOVEREIGN IS:
- Forex macro swing system (v004, avg Sharpe 0.801, 8/8 pairs positive, target 1.5)
- ICT FVG entry system (76% pass rate, +0.40R/trade, FunderPro prop challenge target)
- Equity paper trading system with PTJ gates, Alexandrian Library, CS229 ML stack
- Infrastructure: Numba fast backtest engine (148k/sec), launchd scheduling, GitHub Pages dashboard

SUGGESTION RULES:
- Each suggestion must be SPECIFIC. Not "improve the model" — "remove NZDUSD from ALL_PAIRS, its Sharpe 0.22 drags the portfolio average by ~0.07."
- Each must reference actual system components (file names, variable names, metrics) where possible.
- Each must have a clear ACTION — what Colin actually does to implement it.
- Categories: FOREX, ICT, RISK, INFRASTRUCTURE, RESEARCH, ML
- Priority: HIGH (significant edge impact), MEDIUM (quality of life / reliability), LOW (nice to have)
- Never suggest something already in the vetoed list.
- Never suggest vague things like "add more data" or "improve signals."

OUTPUT FORMAT — respond with ONLY valid JSON, no preamble, no explanation:
{
  "suggestions": [
    {
      "category": "FOREX",
      "priority": "HIGH",
      "title": "Short title under 80 chars",
      "detail": "2-3 sentences explaining WHY this matters with specific numbers from the system.",
      "action": "Exactly what to do — file, function, parameter, command."
    }
  ]
}

Generate exactly 2 suggestions. Make them genuinely useful — things you would actually tell a trading partner. Keep each "detail" under 40 words and each "action" under 25 words."""


def _load_suggestions() -> dict:
    try:
        return json.loads(SUGGESTIONS_PATH.read_text())
    except Exception:
        return {"suggestions": [], "vetoed_ids": [], "stats": {"total_generated": 0, "implemented": 0, "vetoed": 0}}


def _count_open_suggestions() -> int:
    data = _load_suggestions()
    return sum(1 for s in data.get("suggestions", []) if s.get("status") == "NEW")


def _generate_suggestions(client: anthropic.Anthropic, context: str):
    """Generate new suggestions if fewer than 3 are open. Cheap separate call."""
    if _count_open_suggestions() >= 3:
        log.info("Suggestions: 3+ open already — skipping generation")
        return

    data = _load_suggestions()
    vetoed_ids = data.get("vetoed_ids", [])
    existing_titles = [s["title"] for s in data.get("suggestions", []) if s.get("status") == "NEW"]

    veto_context = ""
    if vetoed_ids:
        vetoed = [s for s in data.get("suggestions", []) if s["id"] in vetoed_ids]
        veto_context = "\nALREADY VETOED (do not re-suggest):\n" + "\n".join(f"- {s['title']}" for s in vetoed)

    existing_context = ""
    if existing_titles:
        existing_context = "\nALREADY SUGGESTED (do not duplicate):\n" + "\n".join(f"- {t}" for t in existing_titles)

    user_msg = f"""System state:\n\n{context}{veto_context}{existing_context}\n\nGenerate 2-3 fresh suggestions."""

    try:
        log.info("Generating suggestions (claude-haiku-4-5)...")
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=900,
            system=[{"type": "text", "text": SUGGESTION_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = next((b.text for b in response.content if b.type == "text"), "").strip()

        # Extract JSON even if Oracle wraps it in markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        new_suggestions = parsed.get("suggestions", [])

        # Assign IDs and stamp them
        total = data["stats"].get("total_generated", 0)
        for sug in new_suggestions:
            total += 1
            sug["id"] = f"SUG-{total:03d}"
            sug["status"] = "NEW"
            sug["created"] = datetime.now().isoformat()
            sug["vetoed_at"] = None
            sug["implemented_at"] = None
            sug["veto_reason"] = None
            data["suggestions"].append(sug)

        data["stats"]["total_generated"] = total

        # Sync to dashboard
        import shutil
        SUGGESTIONS_PATH.write_text(json.dumps(data, indent=2))
        dst = ROOT / "ict-dashboard" / "data" / "agent" / "suggestions.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SUGGESTIONS_PATH, dst)

        cached = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        _log_cost(response.usage.input_tokens, response.usage.output_tokens, cached)
        log.info(f"Generated {len(new_suggestions)} suggestions. Total open: {_count_open_suggestions()}")

    except json.JSONDecodeError as e:
        log.warning(f"Suggestion JSON parse failed: {e} | raw={raw[:200]}")
    except Exception as e:
        log.warning(f"Suggestion generation failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False, force: bool = False):
    log.info(f"Oracle agent starting (dry_run={dry_run}, force={force})")

    # Rate gate — skip if called too recently (saves credits)
    if not force and not dry_run:
        mins = _minutes_since_last_oracle()
        if mins < MIN_INTERVAL_MINUTES:
            log.info(f"Rate gate: last oracle {mins:.0f}m ago, minimum {MIN_INTERVAL_MINUTES}m — skipping")
            return

    # Build context
    context = build_context()
    log.info(f"Context built: {len(context)} chars")

    if dry_run:
        print("=== DRY RUN: Context that would be sent ===")
        print(context[:2000])
        print("=== End context preview ===")
        _save_message(
            "DRY RUN: Oracle would have called Claude here and written a message based on current system state.",
            "FYI"
        )
        return

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set — cannot call Oracle")
        _save_message("Oracle failed: ANTHROPIC_API_KEY not set. Add it to .env.", "URGENT")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # System prompt is cached — only billed at 0.1x after first call
    system_with_cache = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    user_message = f"System state:\n\n{context}\n\nWrite your message to Colin. Maximum 3 sentences. Stop when done."

    try:
        log.info("Calling Claude API (claude-haiku-4-5, max_tokens=120)...")
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=160,          # 2-4 sentences — that's all we need
            system=system_with_cache,
            messages=[{"role": "user", "content": user_message}],
        )

        text = next(
            (b.text for b in response.content if b.type == "text"),
            "Oracle returned no text."
        ).strip()

        priority = _classify_priority(text)
        _save_message(text, priority)

        cached = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        _log_cost(response.usage.input_tokens, response.usage.output_tokens, cached)

        log.info(f"Oracle complete. stop_reason={response.stop_reason} "
                 f"in={response.usage.input_tokens} cached={cached} "
                 f"out={response.usage.output_tokens}")

        # Generate suggestions if fewer than 3 open
        _generate_suggestions(client, context)

    except anthropic.AuthenticationError:
        log.error("Invalid API key")
        _save_message("Oracle failed: invalid ANTHROPIC_API_KEY.", "URGENT")
    except anthropic.RateLimitError:
        log.warning("Rate limited — skipping this cycle")
    except Exception as e:
        log.error(f"Oracle error: {e}")
        _save_message(f"Oracle error: {type(e).__name__}: {str(e)[:200]}", "URGENT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sovereign Oracle Agent")
    parser.add_argument("--dry-run", action="store_true", help="Build context but skip API call")
    parser.add_argument("--force",   action="store_true", help="Bypass rate gate and call immediately")
    args = parser.parse_args()
    run(dry_run=args.dry_run, force=args.force)
