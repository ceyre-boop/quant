"""Sovereign self-play training loop (spec: research/SELF_PLAY_TRAINING_ARCHITECTURE.md).

Freeze-safe. Modules here MAY import sovereign/ but MUST NOT import ict/ — the
ICT isolation wall (CLAUDE.md NN #1) is enforced by
tests/test_pipeline_does_not_import_sovereign. The HYP-071 value board is consumed
READ-ONLY; only the policy is updated per cycle, and only when the ignition gate
is open (both blockers cleared). By default the gate is CLOSED and every run is
SCAFFOLD/DRY: full pipeline structure, no production policy, no parameter writes.
"""
