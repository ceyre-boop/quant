"""Petrules Gate — Phase 1 groundwork (free-data replay/feature plumbing).

Isolation: this package imports NOTHING from the live execution path (ict/, ict-engine/,
sovereign/, forex_exit_manager, decide_exit). It is pure research plumbing. See
research/petrules/PHASE1_GROUNDWORK.md.

Spec authority: research/PETRULES_GATE_SPEC.md (hash-locked pre-reg gates the model,
NOT this plumbing — no learning/model/calibration/sizer code lives here).
"""
