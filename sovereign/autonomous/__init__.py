"""Autonomous closure layer — turns existing Alta signals into dispatched actions.

No new edges, no new strategies. Pure orchestration over what the rest of the
system already produces (loop_health, decision logs, the hypothesis ledger).

Modules:
  health_responder    — watch loop_health, dispatch fix requests (notify-only)
  hypothesis_generator— rule-based candidate hypotheses from closed reps + ledger gaps
  research_factory    — run validators against queued hypotheses (gated dry-run)
  escalation_router   — triage all outputs into RED/YELLOW/GREEN/AUTO-HANDLE

Hard rules (mirror CLAUDE.md / the build constraints):
  - NEVER auto-deploy to live trading. The layer proposes; Colin decides.
  - NEVER touch the forex live system.
  - EVERY decision logs. FAIL LOUD — no bare except.
"""
