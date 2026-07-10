"""TICK-022 — Prop-Funnel EV Simulator (Phase A of the "$10k/month" program).

Measurement instrument, NOT edge validation. Simulates strategy families through
realistic prop-firm rulesets and reports the full funnel: pass rates, time-to-pass,
fees-to-funded, funded survival, monthly payout distribution, program EV.

No hypothesis-ledger writes. No live/execution-path/config changes. No launchd.
Outputs land only under data/research/prop_funnel/.

Plan: plans/TICK-022.md (approved 2026-07-10).
"""
