"""Oracle v2 — the daily reflection as a Claude Agent SDK agent with repo tools.

What it may do: read the repo, score theses, read ledgers/logs/prices, search the web, and
write a thesis DRAFT (research/thesis/drafts/). What it may never do: seal, place orders, edit
config, or run shell. Enforced by the tool allow-list, not by the prompt.

Every run is bounded by `max_budget_usd` and logged to logs/oracle_cost.json."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "oracle" / "03-agent" / "output"
DRAFTS = ROOT / "research" / "thesis" / "drafts"
COST_LOG = ROOT / "logs" / "oracle_cost.json"
LEDGER = ROOT / "data" / "agent" / "oracle_sdk_ledger.jsonl"
MODEL = "claude-opus-5-5"
DEFAULT_BUDGET = 2.0

MISSION = """You are the Oracle of Alta Investments: the daily reflection over a trading repo.
Today is {today}. Work only from what the tools return; cite file paths for every claim.

Do, in order:
1. thesis_list, then thesis_score for every LIVE thesis. State basket vs SPY, drawdown vs the
   30% invalidation, and answer each manual invalidation question with evidence (WebSearch is
   allowed; name sources).
2. loop_health and recent_decisions(7). One paragraph: is the trading system producing closed
   trades? If not, say why in one sentence, from the data.
3. Read research/EDGE_LEDGER.md and research/HYPOTHESIS_LESSONS.md (Grep/Read). Propose ONE lesson
   for today: what the last 7 days of data say that the ledger does not already say. If nothing
   new, say "no new lesson" — that is a valid output.
4. Only if the evidence warrants it, write ONE thesis draft with propose_thesis_draft. It must
   name who is forced to pay and why, a basket of 4–8 liquid US tickers with weights summing to
   1, a 12-month horizon, and an invalidation. A draft is a proposal; Colin seals.

Finish with a report under 400 words with headers: THESES, SYSTEM, LESSON, DRAFT (or "no draft").
Never recommend an order. Never claim an edge the EDGE_LEDGER does not record."""


def _load_dotenv() -> None:
    for p in (ROOT / ".env", Path.home() / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return


# ---- plain functions (testable, network-free unless they say so) -------------------------

def t_thesis_list() -> str:
    from sovereign.thesis import ledger
    rows = []
    for r in ledger.read_all():
        rows.append({k: r.get(k) for k in ("id", "title", "status", "entry_date", "horizon_months",
                                           "hash_lock", "basket", "invalidation")})
    return json.dumps(rows, indent=1)


def t_thesis_score(thesis_id: str) -> str:
    from sovereign.thesis import ledger
    from sovereign.thesis.prices import load_daily
    from sovereign.thesis.score import format_scorecard, score
    rec = ledger.get(thesis_id)
    card = score(rec, load_daily, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    return format_scorecard(rec, card) + "\n\nmanual_check: " + str(rec.get("invalidation", {}).get("manual_check"))


def t_loop_health() -> str:
    from sovereign.oracle.loop_health import check_all_loops
    return json.dumps(check_all_loops(), indent=1, default=str)


def t_recent_decisions(days: int = 7) -> str:
    """Records from sovereign/intelligence/decision_logger (data/decision_logs/decisions_YYYY-MM.jsonl)."""
    base = ROOT / "data" / "decision_logs"
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for f in sorted(base.glob("decisions_*.jsonl"))[-2:] if base.exists() else []:
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = d.get("entry_timestamp") or d.get("timestamp") or d.get("ts") or ""
            try:
                if pd_ts(ts) < cutoff:
                    continue
            except Exception:
                pass
            out.append({k: d.get(k) for k in ("entry_timestamp", "system", "pair", "direction", "grade", "score",
                                              "session", "outcome", "exit_timestamp", "pnl_r", "trade_id")})
    return json.dumps({"days": days, "n": len(out), "source": str(base.relative_to(ROOT)),
                       "records": out[-80:]}, indent=1, default=str)


def pd_ts(ts: str) -> datetime:
    t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def t_price(ticker: str, days: int = 30) -> str:
    from sovereign.thesis.prices import load_daily
    s = load_daily(ticker).tail(days)
    return json.dumps({d.strftime("%Y-%m-%d"): round(float(v), 4) for d, v in s.items()})


def t_propose_thesis_draft(spec: dict[str, Any]) -> str:
    from sovereign.thesis.schema import validate
    spec = dict(spec)
    spec.setdefault("id", "THS-DRAFT")
    spec["source"] = "ai-proposed"
    spec.setdefault("benchmark", "SPY")
    spec.setdefault("entry_date", "TBD")
    spec.setdefault("sizing", {"account": "TBD", "dollars": 0})
    errs = [e for e in validate(spec) if "id must start" not in e]
    if errs:
        return "REJECTED: " + "; ".join(errs)
    DRAFTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = "".join(c if c.isalnum() else "-" for c in spec.get("title", "draft").lower())[:40].strip("-")
    f = DRAFTS / f"{stamp}_{slug}.json"
    f.write_text(json.dumps(spec, indent=2) + "\n")
    return f"DRAFT written: {f} (not sealed; Colin decides)"


# ---- SDK wiring --------------------------------------------------------------------------

def _server():
    from claude_agent_sdk import create_sdk_mcp_server, tool

    def txt(s: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": s}]}

    @tool("thesis_list", "All theses in the sealed thesis ledger (id, status, basket, invalidation).", {})
    async def thesis_list(args):
        return txt(t_thesis_list())

    @tool("thesis_score", "Score one thesis: basket vs SPY, drawdown, marks, invalidation state.", {"thesis_id": str})
    async def thesis_score(args):
        return txt(t_thesis_score(args["thesis_id"]))

    @tool("loop_health", "Liveness of every data/trading loop (ALIVE/STALE/DOWN with ages).", {})
    async def loop_health(args):
        return txt(t_loop_health())

    @tool("recent_decisions", "Live decision-log records from the last N days.", {"days": int})
    async def recent_decisions(args):
        return txt(t_recent_decisions(int(args.get("days", 7))))

    @tool("price", "Daily adjusted closes for a ticker over the last N days.", {"ticker": str, "days": int})
    async def price(args):
        return txt(t_price(args["ticker"], int(args.get("days", 30))))

    @tool("propose_thesis_draft",
          "Write a thesis DRAFT (never sealed). spec: {title, thesis, horizon_months, basket:[{ticker,weight}], invalidation:{rule,metric,threshold,manual_check}}",
          {"spec": dict})
    async def propose_thesis_draft(args):
        return txt(t_propose_thesis_draft(args["spec"]))

    return create_sdk_mcp_server("alta", tools=[thesis_list, thesis_score, loop_health,
                                                recent_decisions, price, propose_thesis_draft])


def _log_cost(cost: float, model: str, usage: dict[str, Any] | None) -> None:
    try:
        COST_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = json.loads(COST_LOG.read_text()) if COST_LOG.exists() else {"entries": []}
        if not isinstance(log, dict) or "entries" not in log:
            log = {"entries": []}
        u = usage or {}
        log["entries"].append({"at": datetime.now(timezone.utc).isoformat(), "source": "oracle_sdk_agent",
                               "model": model, "input_tokens": u.get("input_tokens"),
                               "output_tokens": u.get("output_tokens"), "cost_usd": round(cost or 0.0, 6)})
        log["entries"] = log["entries"][-500:]
        COST_LOG.write_text(json.dumps(log, indent=2))
    except Exception as e:  # cost logging must never kill the run, but say so
        print(f"[oracle_sdk] cost log failed: {e}", file=sys.stderr)


async def _run(mission: str, model: str, budget: float, max_turns: int) -> dict[str, Any]:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query
    opts = ClaudeAgentOptions(
        cwd=str(ROOT), model=model, max_budget_usd=budget, max_turns=max_turns,
        mcp_servers={"alta": _server()},
        allowed_tools=["Read", "Grep", "Glob", "WebSearch",
                       "mcp__alta__thesis_list", "mcp__alta__thesis_score", "mcp__alta__loop_health",
                       "mcp__alta__recent_decisions", "mcp__alta__price", "mcp__alta__propose_thesis_draft"],
        disallowed_tools=["Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "Agent", "WebFetch"],
        permission_mode="bypassPermissions",
        system_prompt="You are a read-only research agent inside a trading repo. You cannot place orders or change config.",
    )
    text, result = [], None
    async for m in query(prompt=mission, options=opts):
        if isinstance(m, AssistantMessage):
            for b in m.content:
                if isinstance(b, TextBlock):
                    text.append(b.text)
        elif isinstance(m, ResultMessage):
            result = m
    return {"report": result.result if (result and result.result) else "\n".join(text),
            "cost_usd": getattr(result, "total_cost_usd", None), "turns": getattr(result, "num_turns", None),
            "is_error": getattr(result, "is_error", None), "stop_reason": getattr(result, "stop_reason", None),
            "errors": getattr(result, "errors", None), "usage": getattr(result, "usage", None),
            "session_id": getattr(result, "session_id", None)}


def run(mission: str | None = None, model: str = MODEL, budget: float = DEFAULT_BUDGET,
        max_turns: int = 40, tag: str = "daily") -> dict[str, Any]:
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing (expected in .env)")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mission = mission or MISSION.format(today=today)
    r = asyncio.run(_run(mission, model, budget, max_turns))
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    (OUT / f"{stamp}_{tag}.md").write_text(r["report"] + "\n")
    meta = {k: v for k, v in r.items() if k != "report"}
    meta.update({"at": stamp, "tag": tag, "model": model, "budget_usd": budget, "report": f"oracle/03-agent/output/{stamp}_{tag}.md"})
    (OUT / f"{stamp}_{tag}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(meta, default=str) + "\n")
    _log_cost(r.get("cost_usd") or 0.0, model, r.get("usage"))
    return r
