#!/usr/bin/env python3
"""
Regenerate the vault BRAIN_INDEX.md — the one-shot situational-awareness file
every agent reads first. Run nightly by the EOD routine.

Reads live state through sovereign.brain.obsidian_reader (verdicts, edge ledger,
regime, weaknesses) and rewrites ~/Obsidian/Obsidian/BRAIN_INDEX.md. Degrades
gracefully: if the vault or a source is missing, the corresponding section shows
"(none)" rather than crashing.

    python3 scripts/refresh_brain_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sovereign.brain import _paths as P  # noqa: E402
from sovereign.brain import obsidian_reader as R  # noqa: E402


def _fmt(rows, limit=20, link=True):
    out = []
    for e in rows[:limit]:
        desc = f" — {e['desc']}" if e.get("desc") else ""
        suffix = f" [[{e['id']}]]" if link else ""
        out.append(f"- {e['id']}: {e['status']}{desc}{suffix}")
    return "\n".join(out) or "- (none)"


def build_index() -> str:
    es = R.load_edge_summary()
    verdicts = R.load_recent_verdicts(8)
    regime = R.load_regime_context()
    watch = R.load_trading_psychology()[:8]

    vlines = [
        f"- {v['hyp_id']}: {v['verdict']} ({v['validator']}, {(v['timestamp'] or '')[:10]})"
        for v in verdicts
    ] or ["- (none)"]
    watch_lines = [f"- {w}" for w in watch] or ["- (none yet — EOD agent appends to [[weakness_log]])"]

    return f"""---
date: {P.today()}
title: "Alta Trading Brain — Knowledge Index"
type: brain
read_first: true
generated: scripts/refresh_brain_index.py (EOD agent, nightly)
---

# Alta Trading Brain — Knowledge Index
Last updated: {P.today()}

> The one file every agent reads first for full situational awareness. Small by
> design. Regenerated nightly by `scripts/refresh_brain_index.py`. Canonical
> state of record still lives in `~/quant`; this is the fast index into it.
> Links: [[00-System-Index]] · [[Hypothesis-Ledger]] · [[Discovery-Ledger]] · [[CONTEXT]] · [[NEXT]]

## Confirmed / Live Edges
{_fmt(es['confirmed'])}

## Killed Hypotheses (graveyard — do not re-propose)
{_fmt(es['killed'], link=False)}

## Recent Verdicts
{chr(10).join(vlines)}

## Current Regime
- Carry (v015) is the only proven forex edge — regime-fragile (pays in rate-trending regimes only). Gapper-fade (HYP-093) is the live equity edge, VALID_BUT_BELOW_FLOOR.
- Latest machine context: {'present' if regime['latest_context'] else 'none'} ({len(regime['regime_notes'])} regime notes logged)

## Weakness Log (recent — "watch for today")
{chr(10).join(watch_lines)}

---
*Read from code:* `from sovereign.brain import obsidian_reader as brain; brain.get_morning_context()`
"""


def main() -> int:
    content = build_index()
    P.ensure_parent(P.BRAIN_INDEX)
    try:
        P.BRAIN_INDEX.write_text(content, encoding="utf-8")
    except OSError as exc:
        print(f"refresh_brain_index: could not write {P.BRAIN_INDEX}: {exc}", file=sys.stderr)
        return 1
    print(f"refresh_brain_index: wrote {P.BRAIN_INDEX} ({len(content.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
