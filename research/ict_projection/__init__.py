"""Read-only 90-day ICT taken-trade projection (TICK-028).

READ-ONLY research module. Reads only data files under data/ledger/ and
data/decision_logs/. Never imports from ict.pipeline, ict.orchestrator, or
ict.ict_veto_ledger, and never touches the execution/exit path (shadow freeze).

Outputs (deterministic, seeded):
  - data/research/ict_projection/projection_90d.json
  - research/ict_projection/report.md
"""
