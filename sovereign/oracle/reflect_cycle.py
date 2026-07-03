"""
Oracle Learning Cycle — Phase 2: REFLECT
sovereign/oracle/reflect_cycle.py

Reads last 7 days of harvests + proven_research.json + I_am_a_good_trader.md.
Makes ONE Oracle (Claude Opus 4.8) call to propose a candidate lesson.
Micro-corrections stay on Haiku (micro_correct.py).
Cost: ≤8 cents per cycle.

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

REASONING_ANALYSIS_DIR = ROOT / "data" / "oracle" / "reasoning_analysis"
PULSE_DIR = ROOT / "data" / "oracle" / "pulses"


def _load_market_briefing() -> str:
    """Latest morning market briefing (Oracle's journal). Flagged unverified — qualitative
    regime context only. Returns a short note if none exists yet."""
    p = ROOT / "data" / "oracle" / "market_briefings" / "latest.json"
    try:
        d = json.loads(p.read_text())
        return (f"[{d.get('date')}] regime={d.get('meta_regime')} (verified={d['provenance']['verified']}). "
                f"{d.get('regime_read','')}\n{d.get('narrative','')[:1200]}")
    except Exception:
        return "No morning market briefing yet (run morning_market_briefing.py). Treat as no context."


def _load_daily_macro() -> str:
    """Daily US economic-state snapshot (FRED). Qualitative regime context for reflection —
    NOT a trading input (never wired to sizing/signals/gates). Mirrors the market-briefing
    pattern. Returns a short note if the snapshot is missing/unverified."""
    p = ROOT / "data" / "macro" / "fred_economic_latest.json"
    try:
        d = json.loads(p.read_text())
        if not d.get("provenance", {}).get("verified"):
            return "No verified macro snapshot (FRED key/fetch failed). Treat as no context."
        s = d.get("summary", {})

        def f(x, suf=""):
            return f"{x}{suf}" if x is not None else "n/a"
        return (
            f"[{d.get('date')}] US economic state (FRED):\n"
            f"  Growth: real GDP {f(s.get('gdp_growth_pct'),'%')} (q/q ann)\n"
            f"  Inflation: CPI {f(s.get('cpi_yoy_pct'),'%')} YoY | core CPI {f(s.get('core_cpi_yoy_pct'),'%')} | "
            f"core PCE {f(s.get('core_pce_yoy_pct'),'%')} (Fed target 2%)\n"
            f"  Labor: unemployment {f(s.get('unemployment_pct'),'%')}\n"
            f"  Rates: fed funds {f(s.get('fed_funds_pct'),'%')} | 10Y {f(s.get('ten_year_pct'),'%')} | "
            f"10Y-2Y {f(s.get('yield_curve_10y2y'),'%')} ({s.get('yield_curve_state','n/a')})\n"
            f"  Consumer: UMich sentiment {f(s.get('consumer_sentiment'))} | VIX {f(s.get('vix'))}"
        )
    except Exception:
        return "No macro snapshot yet (run scripts/fetch_fred_economic.py). Treat as no context."


def _load_recent_prices() -> str:
    """Load live_prices from the most recent pulse file."""
    if not PULSE_DIR.exists():
        return "N/A — no pulse data"
    pulse_files = sorted(PULSE_DIR.glob("pulse_*.json"))
    if not pulse_files:
        return "N/A — no pulse data"
    try:
        data = json.loads(pulse_files[-1].read_text())
        lp = data.get('live_prices', {})
        if not lp:
            return "N/A — price data not yet in pulse (upgrade pulse_check.py)"
        lines = [
            f"{pair}: {v['current']} (H: {v['high_today']} L: {v['low_today']})"
            for pair, v in lp.items()
        ]
        return '\n'.join(lines)
    except Exception:
        return "N/A — error reading pulse"


def _load_reasoning_summary() -> str:
    """Load the most recent monthly reasoning analysis as Oracle context."""
    if not REASONING_ANALYSIS_DIR.exists():
        return "No reasoning analysis yet — will populate after first monthly run."
    reports = sorted(REASONING_ANALYSIS_DIR.glob("*.json"))
    if not reports:
        return "No reasoning analysis yet — will populate after first monthly run."
    try:
        data = json.loads(reports[-1].read_text())
        summary = data.get("oracle_context_summary")
        if summary:
            return summary
        # Fallback: compact reconstruction
        month = data.get("month", "unknown")
        n = data.get("n_trades_analyzed", 0)
        best = data.get("best_chains", [])
        worst = data.get("worst_chains", [])
        lines = [f"Reasoning analysis ({month}, n={n}):"]
        if best:
            lines.append("  BEST: " + " | ".join(f"{b['condition']}→{b['avg_r']:+.2f}R" for b in best[:3]))
        if worst:
            lines.append("  WORST: " + " | ".join(f"{w['condition']}→{w['avg_r']:+.2f}R" for w in worst[:3]))
        return "\n".join(lines)
    except Exception:
        return "Reasoning analysis file present but unreadable."


ORACLE_PROMPT_TEMPLATE = """You are the trading intelligence Oracle for Alta Investments — a two-person quant operation trading FX and equity using a systematic macro strategy.

## Your primary mission
Close the Sharpe gap. {sharpe_summary}
Every suggestion you make must move one needle: Sharpe, win rate, drawdown, or prop challenge readiness.

## The Six Trading Tenets (every suggestion MUST serve at least one)
{tenets}

## Current system state
{bridge_state}

## Current market prices (from last pulse)
{prices_json}

## Morning market briefing (analyst narrative — UNVERIFIED qualitative context, NOT a data feed)
{market_briefing}

## US macro snapshot (FRED economic state — qualitative context, NOT a trading signal)
{macro_economic}

## External regime confirmation (TradingView vs internal)
{regime_alignment}

## 30-indicator consensus (live state vs historical memory)
{indicator_consensus}

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

## Monthly reasoning pattern analysis (updated once/month)
This is a decision-tree clustering of ALL closed trades by their reasoning fields.
Use this to find threshold combinations that consistently predict good or bad outcomes.
{reasoning_summary}

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
                # RED-1 fix: the Oracle must reason ONLY over strategy-authored
                # decisions. Records reconstructed from broker fills
                # (source="fills_backfill") were never authored by the strategy — they
                # carry forbidden pairs (USD_CAD, AUD_NZD) the strategy never traded and
                # have no entry reasoning to learn from. Synthetic test fills
                # (test_fill=True) are likewise not real decisions. Exclude both from
                # reflection input.
                if rec.get("source") == "fills_backfill" or rec.get("test_fill") is True:
                    continue
                # Only REAL closed trades — exclude OPEN and EXPIRED (never-filled
                # scan signals), which are not trade outcomes and would pollute analysis.
                if rec.get("outcome") in (None, "OPEN", "EXPIRED"):
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
                    # Loop 2: entry-time context — lets the Oracle reason over WHY,
                    # not just THAT, a trade won/lost.
                    "entry_context":           rec.get("present_state_snapshot") or {},
                    "active_lessons":          rec.get("active_lessons") or [],
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
        f"## Trade Outcome Analysis — {len(entries)} closed trades "
        f"({n_wins}W / {n_loss}L, avg R={avg_r:+.3f})\n"
        "Each entry includes `entry_context` (the state snapshot at entry). Look for "
        "context that CLUSTERS with losses — e.g. low constraint_score, weak consensus, "
        "trend-misaligned, FLAT regime. If you find a loss-clustering threshold, propose a "
        "lesson with a concrete `param_delta` (e.g. raise a min-score/threshold) so the "
        "EdgePipeline can auto-validate it. Do NOT over-fit to <10 trades.\n"
    )
    return header + json.dumps(entries, indent=2)


def _load_indicator_consensus() -> str:
    """Load live 30-indicator consensus from the latest pulse for Oracle context."""
    pulse_files = sorted(PULSE_DIR.glob("pulse_*.json"))
    if not pulse_files:
        return "No indicator consensus yet — run pulse_check.py first."
    try:
        data = json.loads(pulse_files[-1].read_text())
        ic = data.get("indicator_consensus", {})
        if not ic:
            return "Indicator consensus not yet computed — run build_indicator_ontology.py then pulse_check.py."
        lines = []
        for pair, c in ic.items():
            bull = c.get("bullish", 0)
            bear = c.get("bearish", 0)
            direction = c.get("direction", "?")
            conv = c.get("conviction", 0)
            green = c.get("matching_green", 0)
            hr = c.get("hit_rate", 0)
            lines.append(
                f"{pair}: {bull}/30 bull | {bear}/30 bear | dir={direction} "
                f"| conv={conv:.0%} | green_matches={green} | hist_hr={hr:.0%}"
            )
        ts = data.get("timestamp", "unknown")
        return f"[as of {ts[:16]}]\n" + "\n".join(lines)
    except Exception:
        return "Indicator consensus unavailable — check data/indicators/live_snapshot.json"


def _load_regime_alignment() -> str:
    """Load internal vs TradingView regime alignment for Oracle context."""
    try:
        from sovereign.intelligence.regime_confidence import score_regime_confidence
        conf = score_regime_confidence()
        age_str = f"{conf.tv_signal_age_min:.0f}min ago" if conf.tv_signal_age_min >= 0 else "none"
        return (
            f"Internal: {conf.internal_regime} | "
            f"TradingView: {conf.external_regime} ({conf.tv_signal_count} signals, newest {age_str})\n"
            f"Agreement: {conf.agreement} | Confidence: {conf.confidence:.0%} | "
            f"Sizing multiplier: {conf.sizing_multiplier:.0%}\n"
            f"Reason: {conf.reason}"
        )
    except Exception:
        return "Regime confidence unavailable — run sovereign/intelligence/regime_confidence.py"


def _load_harvest_summary(harvests: list[dict]) -> str:
    if not harvests:
        return "No harvest data available. System may not have trades yet — propose a hypothesis about the existing backtest data instead."
    lines = []
    for h in harvests[:7]:
        fail_dist = h.get("failure_distribution", {})
        dominant = h.get("dominant_failure_mode", "UNKNOWN")
        lines.append(
            f"  {h['date']}: {h['trades_closed']} trades | "
            f"WR={(h.get('win_rate') or 0)*100:.0f}% | "
            f"avgR={(h.get('avg_r') or 0):.3f} | "
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
        prices_json=_load_recent_prices(),
        market_briefing=_load_market_briefing(),
        macro_economic=_load_daily_macro(),
        regime_alignment=_load_regime_alignment(),
        indicator_consensus=_load_indicator_consensus(),
        next_milestones=ctx["next_milestones"],
        hypothesis_summary=ctx["hypothesis_summary"],
        queue_status=ctx["queue_status"],
        proven_summary=_load_proven_summary(),
        active_lessons_summary=_load_active_lessons(),
        reasoning_summary=_load_reasoning_summary(),
        harvest_summary=_load_harvest_summary(harvests),
        decision_log_summary=_load_decision_log_summary(),
        rejected_ids=_get_rejected_ids(),
        schema=json.dumps(ORACLE_SCHEMA, indent=2),
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-8",
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

    # Auto-validate a param-delta candidate through EdgePipeline (GATED — runs the
    # full significance + walk-forward + BH pipeline and stages a verdict; it NEVER
    # commits to live config. Text-only lessons carry no param_delta → skipped, so
    # this is fully non-breaking for the existing reflection flow).
    auto_validation = None
    try:
        cand = reflection.get("candidate_lesson") if isinstance(reflection, dict) else None
        if isinstance(cand, dict) and cand.get("param_delta"):
            from sovereign.oracle.edge_pipeline import EdgePipeline
            auto_validation = EdgePipeline().process({
                "id": cand.get("id", f"L-{date}"),
                "subsystem": cand.get("subsystem", "forex"),
                "param_delta": cand["param_delta"],
                "label": cand.get("lesson_text", ""),
            })
    except Exception as exc:
        auto_validation = {"status": "ERROR", "reason": f"auto-validation failed: {exc}"}

    output = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "estimated_cost_usd": round(
            response.usage.input_tokens * 0.000015 +
            response.usage.output_tokens * 0.000075, 6
        ),
        "harvest_days_read": len(harvests),
        "reflection": reflection,
        "auto_validation": auto_validation,
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
            prices_json=_load_recent_prices(),
        market_briefing=_load_market_briefing(),
            macro_economic=_load_daily_macro(),
            regime_alignment=_load_regime_alignment(),
            indicator_consensus=_load_indicator_consensus(),
            next_milestones=ctx["next_milestones"],
            hypothesis_summary=ctx["hypothesis_summary"],
            queue_status=ctx["queue_status"],
            proven_summary=_load_proven_summary(),
            active_lessons_summary=_load_active_lessons(),
            reasoning_summary=_load_reasoning_summary(),
            harvest_summary=_load_harvest_summary(harvests),
            decision_log_summary=_load_decision_log_summary(),
            rejected_ids=_get_rejected_ids(),
            schema=json.dumps(ORACLE_SCHEMA, indent=2),
        ))
    else:
        result = run_reflect(harvests, date=args.date)
        print(f"Reflection complete. Cost: ${result['estimated_cost_usd']:.4f}")
        print(json.dumps(result["reflection"], indent=2))
