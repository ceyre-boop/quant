"""experience/precedent_service.py — L2b stub: decision-time precedent lookups.

Inert by construction: nothing in the live decision path (or anywhere else in this repo)
imports this module. The Sunday review (L2a, experience/precedents.py, wired into
experience/weekly_review.py) is the only precedent-retrieval consumer today.

This stub exists so a future ticket can wire decision-time lookups (e.g. from
ict-engine/orchestrator.py at entry time) without inventing the interface from scratch — but
it stays gated behind config (experience.precedents.decision_time_enabled, default false) so
turning it on is a deliberate, logged config change (CLAUDE.md NON-NEGOTIABLE #4), not a
silent capability upgrade smuggled in by this ticket. query() returns [] unconditionally
while the flag is false, regardless of input.
"""
from __future__ import annotations

from experience import precedents


def query(board_state_row: dict, top_k: int = 3) -> list[dict]:
    """Precedents for a live sentiment_board_state row — [] while the L2b flag is off.

    board_state_row: one row shaped like sovereign.sentiment.store's sentiment_board_state
    table. Deliberately minimal tag extraction (categorical columns only) because the honest
    board->feature mapping (LIB-FEAT-1) doesn't exist yet; this stays a stub until that ticket
    lands and a real caller is wired up.
    """
    if not precedents.decision_time_enabled:
        return []
    tags = set()
    vix_regime = board_state_row.get("vix_regime")
    if vix_regime:
        tags.add(str(vix_regime).lower())
    return precedents.find_precedents(tags, top_k=top_k)
