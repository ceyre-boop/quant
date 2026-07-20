"""
obsidian_writer — the WRITE side of the brain.

Agents call these after acting to post structured knowledge back to the vault,
so the next agent (and the next session) reads it. Every writer is
append-oriented and creates missing files/dirs; none overwrite prior knowledge.
All functions return True on success, False if the write failed (never raise),
so a write failure can never crash a trading routine.

    from sovereign.brain import obsidian_writer as brain
    brain.write_regime_observation("USDJPY", "trending above 158 post-BoJ", "morning-scan")
"""

from __future__ import annotations

from . import _paths as P


def _ensure_header(path, header: str) -> None:
    """Seed a log file with a markdown H1 header the first time we touch it."""
    if not P.read_text(path).strip():
        P.append_line(path, header.rstrip() + "\n\n")


def write_verdict(hyp_id: str, verdict: str, stats=None, mechanism: str = "", notes: str = "") -> bool:
    """
    Write a hypothesis verdict to the vault with full context.

    Appends a dated block to Trading/Research/Verdict-Log.md. `stats` may be a
    dict (rendered as key: value lines) or a string.
    """
    _ensure_header(P.VERDICT_LOG, "# Verdict Log\n\n> Append-only. Written by sovereign.brain.obsidian_writer.")

    lines = [f"\n## {hyp_id} — {verdict}  ({P.now_iso()})\n"]
    if mechanism:
        lines.append(f"- **Mechanism:** {mechanism}\n")
    if isinstance(stats, dict):
        for k, v in stats.items():
            lines.append(f"- **{k}:** {v}\n")
    elif stats:
        lines.append(f"- **Stats:** {stats}\n")
    if notes:
        lines.append(f"- **Notes:** {notes}\n")
    return P.append_line(P.VERDICT_LOG, "".join(lines))


def write_regime_observation(pair_or_market: str, observation: str, source: str = "") -> bool:
    """
    Log a market-regime observation, e.g.
        write_regime_observation("USDJPY", "trending above 158 post-BoJ", "morning-scan")

    Appends one line to Trading/Market-Intelligence/regime_observations.md so
    load_regime_context() can surface it to the next morning routine.
    """
    _ensure_header(
        P.REGIME_LOG,
        "# Regime Observations\n\n> Append-only market-regime log. Read by obsidian_reader.load_regime_context().",
    )
    src = f" _(src: {source})_" if source else ""
    line = f"- [{P.today()}] **{pair_or_market}:** {observation}{src}\n"
    return P.append_line(P.REGIME_LOG, line)


def write_weakness_note(weakness_type: str, description: str, date: str = "") -> bool:
    """
    Log a trading-weakness observation, e.g.
        write_weakness_note("overtrading", "opened 4 trades during VIX spike")

    Appends to Trading Psychology/weakness_log.md in the exact format
    load_weakness_log() parses back: `- [DATE] (type) description`.
    """
    _ensure_header(
        P.WEAKNESS_LOG,
        "# Weakness Log\n\n> Observed behavioral patterns. The morning brief reads recent entries "
        "as 'watch for today'. Append-only; format: `- [DATE] (type) description`.",
    )
    line = f"- [{date or P.today()}] ({weakness_type}) {description}\n"
    return P.append_line(P.WEAKNESS_LOG, line)


def write_morning_brief(signals=None, regime: str = "", active_edges=None) -> bool:
    """
    Write the morning intelligence brief to that day's Ops agent log.
    Appends a MORNING BRIEF block to Trading/Ops/Agent-Log-{today}.md.
    """
    path = P.OPS / f"Agent-Log-{P.today()}.md"
    _ensure_header(path, f"# Agent Log — {P.today()}")

    lines = [f"\n## MORNING BRIEF — {P.now_iso()}\n"]
    if regime:
        lines.append(f"- **Regime:** {regime}\n")
    if active_edges:
        edge_str = ", ".join(str(e) for e in active_edges)
        lines.append(f"- **Active edges:** {edge_str}\n")
    if signals:
        if isinstance(signals, (list, tuple)):
            lines.append(f"- **Signals:** {', '.join(str(s) for s in signals)}\n")
        else:
            lines.append(f"- **Signals:** {signals}\n")
    return P.append_line(path, "".join(lines))


def write_eod_summary(fills=None, pnl=None, notes: str = "", lessons=None) -> bool:
    """
    Write the EOD summary with a structured lessons section, appended to that
    day's Ops agent log.

    fills   : list/int of fills or a rendered string
    pnl     : number or string
    lessons : list[str] — surfaced separately so the morning agent can read them
    """
    path = P.OPS / f"Agent-Log-{P.today()}.md"
    _ensure_header(path, f"# Agent Log — {P.today()}")

    lines = [f"\n## EOD SUMMARY — {P.now_iso()}\n"]
    if fills is not None:
        fills_str = fills if isinstance(fills, str) else (
            f"{len(fills)} fills" if isinstance(fills, (list, tuple)) else str(fills)
        )
        lines.append(f"- **Fills:** {fills_str}\n")
    if pnl is not None:
        lines.append(f"- **P&L:** {pnl}\n")
    if notes:
        lines.append(f"- **Notes:** {notes}\n")
    if lessons:
        lines.append("\n### Lessons\n")
        for lesson in (lessons if isinstance(lessons, (list, tuple)) else [lessons]):
            lines.append(f"- {lesson}\n")
    return P.append_line(path, "".join(lines))


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import os
    import tempfile

    # Redirect the vault to a temp dir so the smoke test writes nowhere real.
    os.environ["ALTA_VAULT"] = tempfile.mkdtemp()
    import importlib

    importlib.reload(P)
    print("regime:", write_regime_observation("USDJPY", "above 158", "test"))
    print("weakness:", write_weakness_note("overtrading", "4 trades in VIX spike"))
    print("verdict:", write_verdict("HYP-999", "NOT_SIGNIFICANT", {"p": 0.4}, "none", "smoke"))
    print("morning:", write_morning_brief(["EURUSD SHORT"], "carry-positive", ["HYP-045"]))
    print("eod:", write_eod_summary(fills=2, pnl="+0.3%", notes="quiet", lessons=["held discipline"]))
    print("wrote under:", P.VAULT)
