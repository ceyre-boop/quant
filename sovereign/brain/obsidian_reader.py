"""
obsidian_reader — the READ side of the brain.

Any agent imports this to load relevant long-term memory from the vault before
acting. Every function handles missing files gracefully: it returns an empty
structure and never raises. Nothing here mutates the vault.

    from sovereign.brain import obsidian_reader as brain
    ctx = brain.get_morning_context()
"""

from __future__ import annotations

from . import _paths as P

# Statuses in the vault ledger that count as "still-live confirmed edges".
_CONFIRMED_STATUSES = {"CONFIRMED", "DEPLOYED", "VALID_BUT_BELOW_FLOOR", "MARGINAL"}
_KILLED_STATUSES = {
    "NOT_SIGNIFICANT",
    "REJECTED",
    "REJECTED_OOS",
    "METRIC_ARTIFACT",
    "DATA_INSUFFICIENT",
}


def load_recent_verdicts(n: int = 10) -> list[dict]:
    """
    Load the last N hypothesis verdicts, newest first.

    Primary source is data/research/auto_hypothesis_results.jsonl (the machine's
    verdict stream). Falls back to factory_ledger.jsonl. Returns a list of
    normalized dicts: {hyp_id, verdict, oos_sharpe, p_value, timestamp, source}.
    """
    rows: list[dict] = []

    for obj in P.iter_jsonl(P.VERDICTS_JSONL):
        parsed = obj.get("parsed") or {}
        rows.append(
            {
                "hyp_id": obj.get("hypothesis_id", ""),
                "verdict": obj.get("verdict", ""),
                "oos_sharpe": parsed.get("oos_sharpe"),
                "p_value": parsed.get("perm_p"),
                "timestamp": obj.get("timestamp", ""),
                "validator": obj.get("validator", ""),
                "source": "auto_hypothesis_results",
            }
        )

    for obj in P.iter_jsonl(P.FACTORY_LEDGER):
        rows.append(
            {
                "hyp_id": obj.get("id", ""),
                "verdict": obj.get("status", ""),
                "oos_sharpe": None,
                "p_value": obj.get("p_value"),
                "timestamp": obj.get("tested_at", ""),
                "validator": "factory",
                "source": "factory_ledger",
            }
        )

    rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return rows[: max(0, n)]


def load_regime_context(lookback_days: int = 7) -> dict:
    """
    Load recent market-regime intelligence: what regime are we in?

    Reads (in priority order) the regime-observation log, the most recent repo
    morning-context JSON, and the latest vault Ops agent log. Returns:
        {regime_notes: [str], latest_context: dict|None, source_files: [str]}
    """
    out: dict = {"regime_notes": [], "latest_context": None, "source_files": []}

    # 1. Structured regime-observation log (written by obsidian_writer).
    regime_text = P.read_text(P.REGIME_LOG)
    if regime_text:
        notes = [ln.strip() for ln in regime_text.splitlines() if ln.strip().startswith("- ")]
        out["regime_notes"].extend(notes[-20:])
        out["source_files"].append(str(P.REGIME_LOG))

    # 2. Most recent repo morning-context JSON (rich, machine-written).
    latest_ctx = P.latest_matching(P.CONTEXT_DIR, "morning_context_*.json", n=1)
    if latest_ctx:
        import json

        try:
            out["latest_context"] = json.loads(P.read_text(latest_ctx[0]))
            out["source_files"].append(str(latest_ctx[0]))
        except (ValueError, TypeError):
            pass

    # 3. Latest vault Ops agent log — free-text regime commentary.
    latest_ops = P.latest_matching(P.OPS, "Agent-Log-*.md", n=1)
    if latest_ops:
        out["source_files"].append(str(latest_ops[0]))

    return out


def load_weakness_log() -> list[dict]:
    """
    Load the trading-psychology weakness log (the graveyard of behavioral
    mistakes). Returns a list of {date, type, description} newest last, or []
    if the log does not exist yet.
    """
    text = P.read_text(P.WEAKNESS_LOG)
    if not text:
        return []

    entries: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        # Format written by write_weakness_note:
        #   - [2026-07-19] (overtrading) description...
        if not line.startswith("- ["):
            continue
        try:
            date = line[3 : line.index("]")]
            rest = line[line.index("]") + 1 :].strip()
            wtype = ""
            if rest.startswith("("):
                wtype = rest[1 : rest.index(")")]
                rest = rest[rest.index(")") + 1 :].strip()
            entries.append({"date": date, "type": wtype, "description": rest})
        except ValueError:
            entries.append({"date": "", "type": "", "description": line[2:].strip()})
    return entries


def load_edge_summary() -> dict:
    """
    Load confirmed edges with current status, from the vault Hypothesis-Ledger
    note. Returns:
        {confirmed: [{id, desc, status}], killed: [{id, desc, status}],
         all_by_status: {STATUS: [...]}, source: str}
    Empty structure if the ledger note is absent.
    """
    text = P.read_text(P.HYPOTHESIS_LEDGER_NOTE)
    sections = P.parse_ledger_sections(text) if text else {}

    confirmed, killed = [], []
    for status, items in sections.items():
        for it in items:
            row = {"id": it["id"], "desc": it["desc"], "status": status}
            if status in _CONFIRMED_STATUSES:
                confirmed.append(row)
            elif status in _KILLED_STATUSES:
                killed.append(row)

    return {
        "confirmed": confirmed,
        "killed": killed,
        "all_by_status": sections,
        "source": str(P.HYPOTHESIS_LEDGER_NOTE) if text else "",
    }


def load_trading_psychology() -> list[str]:
    """
    Load the trader's observed behavioral patterns as flat strings — the
    "watch for this" list the morning brief surfaces. Newest-first.
    """
    return [
        f"[{e['date']}] {('(' + e['type'] + ') ') if e['type'] else ''}{e['description']}".strip()
        for e in reversed(load_weakness_log())
    ]


def get_morning_context() -> dict:
    """
    Composite morning brief: regime + active edges + recent verdicts + the top
    behavioral watch-items. This is what the 08:00 routine reads before scoring.
    """
    edges = load_edge_summary()
    return {
        "date": P.today(),
        "regime": load_regime_context(),
        "active_edges": edges["confirmed"],
        "recent_verdicts": load_recent_verdicts(n=5),
        "watch_for_today": load_trading_psychology()[:5],
    }


def get_research_context() -> dict:
    """
    Composite research brief: graveyard + confirmed edges + recent verdicts.
    The 21:00 research routine reads this so the hypothesis generator does not
    re-propose killed ideas and sweeps start from what already works.
    """
    edges = load_edge_summary()
    return {
        "date": P.today(),
        "graveyard": edges["killed"],
        "confirmed_edges": edges["confirmed"],
        "recent_verdicts": load_recent_verdicts(n=15),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json

    print("VAULT:", P.VAULT, "exists:", P.VAULT.exists())
    print("REPO:", P.REPO, "exists:", P.REPO.exists())
    print("\n--- recent verdicts (3) ---")
    print(json.dumps(load_recent_verdicts(3), indent=2))
    print("\n--- edge summary counts ---")
    es = load_edge_summary()
    print("confirmed:", len(es["confirmed"]), "killed:", len(es["killed"]))
    print("\n--- morning context keys ---")
    mc = get_morning_context()
    print({k: (len(v) if isinstance(v, list) else type(v).__name__) for k, v in mc.items()})
