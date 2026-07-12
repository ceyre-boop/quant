"""HYP-090 "MODERN" (TICK-023) — pre-registered adaptive walk-forward study.

Does daily adaptive parameter selection (trailing 90/180/365d window ranking +
regime matching, full surface including pair selection) beat static v015 on
2015→2026 daily forex, after selection-bias correction?

Registered prior expectation: NOT_ROBUST. Prior family kills disclosed in the
prereg: HYP-065, HYP-066, HYP-067, the 180-config exit sweep, the regime router.

READ-ONLY STUDY: writes only under data/research/modern/ (+ the P0 prereg file
and hypothesis-ledger entries). Never touches live parameter files or any
broker/scheduler surface (see tests/test_isolation.py for the enforced wall).
Plan: plans/TICK-023.md (approved 2026-07-11).
"""
