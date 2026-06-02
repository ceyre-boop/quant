"""Morning Briefing Engine (ES/NQ).

A daily, self-scored, regime-aware market briefing for Oracle. Collects ES+NQ market
state, classifies the NQ/ES lead-lag regime, builds a volume profile, pulls news + the
event calendar, and (optionally) runs an Opus 4.8 synthesis — then scores yesterday's
call against reality so the briefing earns trust the way every edge does.

HONEST FRAME: this is INTELLIGENCE (a journal + research input), not a validated EDGE.
Output is flagged provenance.verified=false and must never auto-drive sizing until the
scorecard proves its confidence is calibrated over a real sample. ES/NQ are CME futures
(OANDA is forex-only) — no execution path here; this is data + classification only.
"""
