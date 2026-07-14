#!/usr/bin/env python3
"""Oracle Daily Brief — the operator's morning read: what happened, what it
means, and where we stand on the ladder to funded trading.

READ-ONLY gather (NEXT.md top entries, hypothesis ledger tail, ICARUS shadow
state, latest reflection) -> ONE cost-capped LLM call (haiku-tier; graceful
template fallback with no key / on any error) -> data/oracle/daily_digest.json.

The progression ladder is the operator's own (2026-07-14): 30 clean shadow
days -> 1-day funded-account sim -> broker pass/fail -> earn -> compound.
Run: python3 -m sovereign.oracle.daily_digest   (daily, pre-market)
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/oracle/daily_digest.json"
SHADOW = REPO / "data/research/yield_frontier/shadow"
MODEL = "claude-haiku-4-5-20251001"      # cheap by design; brief != research
MAX_TOKENS = 900


def _env_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    envf = REPO / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def gather():
    g = {}
    nxt = (REPO / "NEXT.md").read_text(errors="ignore")
    entries = re.split(r"\n## ", nxt)[1:4]
    g["next_md_top"] = ["## " + e[:1400] for e in entries]
    try:
        led = json.loads((REPO / "data/agent/hypothesis_ledger.json").read_text())
        g["ledger_tail"] = [{k: e.get(k) for k in ("id", "status", "verdict", "result")}
                            for e in led[-6:]]
    except Exception:
        g["ledger_tail"] = []
    days = []
    if (SHADOW / "shadow_daily.jsonl").exists():
        days = [json.loads(x) for x in
                (SHADOW / "shadow_daily.jsonl").read_text().splitlines() if x.strip()]
    g["icarus_days"] = days
    g["icarus_green"] = sum(1 for d in days if d["constitutional_day_ret"] > 0)
    refl_dir = REPO / "data/oracle/reflections"
    refl = sorted(refl_dir.glob("*.json"))[-1:] if refl_dir.exists() else []
    if refl:
        try:
            r = json.loads(refl[0].read_text())
            g["latest_reflection"] = str(
                (r.get("reflection", r) or {}).get("candidate_lesson", {}).get(
                    "lesson_text", ""))[:300]
        except Exception:
            pass
    return g


def ladder(g):
    n = len(g["icarus_days"])
    clean = g["icarus_green"]
    return [
        {"gate": "G1", "label": "30 clean shadow days (ICARUS)",
         "state": "IN_PROGRESS", "pct": min(100, round(100 * n / 30)),
         "detail": f"{n}/30 days recorded · {clean} green"},
        {"gate": "G2", "label": "1-day funded-account simulation pass",
         "state": "LOCKED", "pct": 0, "detail": "unlocks at G1"},
        {"gate": "G3", "label": "TICK-024 swap cascade + clamps enforced (Jul 28)",
         "state": "PENDING_OPERATOR", "pct": 0,
         "detail": "gates any real dollar — unchanged"},
        {"gate": "G4", "label": "Broker account + live pass/fail",
         "state": "LOCKED", "pct": 0, "detail": "needs G1-G3 + explicit go"},
        {"gate": "G5", "label": "Earn -> compound via research + wisdom",
         "state": "LOCKED", "pct": 0, "detail": "the actual goal"},
    ]


def template_digest(g):
    days = g["icarus_days"]
    last = days[-1] if days else None
    return {
        "headline": (f"ICARUS shadow day {len(days)}: "
                     f"{last['constitutional_day_ret'] * 100:+.3f}% — "
                     f"{g['icarus_green']}/{len(days)} green so far"
                     if last else "No shadow days recorded yet"),
        "what_happened": [e.split("\n")[0].strip("# ") for e in g["next_md_top"]],
        "what_it_means": "Template digest (no LLM call) — see NEXT.md for detail.",
        "todays_one_thing": "Check the ICARUS calendar; the ladder does the rest.",
    }


def llm_digest(g, key):
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    prompt = (
        "You are the Oracle of a solo trader's quant system (Alta). Write his morning "
        "brief from the evidence below. He is a trader first. Be concrete, honest, "
        "brief. NEVER invent numbers; only use what is given. Return STRICT JSON: "
        '{"headline": one sentence, "what_happened": [3-5 short bullets], '
        '"what_it_means": 2-3 sentences tying events to the goal ladder, '
        '"todays_one_thing": the single highest-leverage action for the operator today}.'
        "\n\nEVIDENCE:\n" + json.dumps(g, default=str)[:14000])
    msg = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                 messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


def main():
    g = gather()
    key = _env_key()
    try:
        digest = llm_digest(g, key) if key else template_digest(g)
        source = "oracle-llm" if key else "template"
    except Exception as e:
        digest = template_digest(g)
        digest["what_it_means"] += f" (LLM fallback: {type(e).__name__})"
        source = "template-fallback"
    doc = {**digest, "progression": ladder(g), "source": source,
           "model": MODEL if source == "oracle-llm" else None,
           "generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"[daily-digest] {source}: {doc['headline'][:90]}")


if __name__ == "__main__":
    main()
