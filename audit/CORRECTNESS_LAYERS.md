# The Four Correctness Layers

*The systems-level interpreter's map. A decision system fails the same four ways an operating
system does — and the principle every layer converges on is: **the system is not wrong when it
crashes; it is wrong when it silently succeeds.** This file records where each layer is enforced,
so the gaps are visible.*

| Layer | Failure it prevents | Standing enforcement (where) | Strength |
|---|---|---|---|
| **1 Reality** (Systems) | backtest ≠ live | shared `decide_exit` kernel `sovereign/forex/exit_machine.py` (backtester + live manager call the *same* function); parity test `tests/test_forex_exit_manager.py`; SHADOW_MODE lock test; scheduled self-escalating `audit/shadow_divergence.py` (`com.alta.shadow_audit`, weekdays 09:05) vs `audit/divergence_spec.md` (L1=100%, C5=0) | **STRONG** |
| **2 Signal** (Compiler) | meaning lost input→output / look-ahead | causality tests `tests/unit/test_vrp_causality.py`; look-ahead auditor `scripts/audit_look_ahead.py` (empirical board-leak check); COT Friday-keying; ALFRED first-print | **MEDIUM** — the production DB leak check is advisory (`update_sentiment.main()` prints but exits 0); the hard `==0` gate runs only in pytest against a fixture |
| **3 Environment** (Network) | edge is regime-fragile, not robust | per-year `wf_robust` `sovereign/discovery/gate.py`; CPCV `sovereign/discovery/cpcv.py` (test-locked); permutation/DSR/BH; standing methodology gate `research_factory.py` (4h, perm≥10k) | **MEDIUM-STRONG** — the scheduled gate checks the spec *declares* walk-forward/both-sides, not that per-regime robustness was re-verified; the real gate + prereg `--check` are on-demand |
| **4 Adversarial** (Security) | the system trusts a lie — incl. its own mistake | **this is the weak layer** — see below | **WEAK → being closed** |

## Layer 4 — why it was weak, and what closes it

Two adversarial failures **already materialized** and were caught **only by ad-hoc human audit**:

- **RED-1 — Oracle contamination.** `sovereign/oracle/reflect_cycle.py` selects trades for the
  daily reflection by outcome alone (`not in None/OPEN/EXPIRED`) — no source or pair filter — so
  backfilled probe records on forbidden pairs (USD_CAD/AUD_NZD) feed fabricated W/L into cognition.
  No test caught it; the one relevant test *asserted the vulnerable behavior was correct*.
- **Rogue OANDA writer.** 1-unit USD_CAD LONG sentinel probes (`stop=1.0/tp=2.0`) hit the broker
  with no matching decision record and no standing detector.

Before this, the only standing Layer-4 mechanisms were the `shadow_divergence` watchdog (scoped to
the exit-manager) and `stray_tripwire` (files only); the red-team skills (`quant-review`,
`security-auditor`) fire only when a human types the command.

**What closes it:** `audit/invariant_guard.py` — a read-only, spec-first, scheduled,
self-escalating detector (sibling to `shadow_divergence`), governed by `audit/invariants_spec.md`
and `com.alta.invariant_guard` (daily). It asserts three invariants, each mapped to a *materialized*
failure, and escalates URGENT on violation:

- **I1** no probe/forbidden record enters the Oracle reflection summary (RED-1),
- **I2** no rogue/sentinel OANDA fill in the ledgers (the rogue writer),
- **I3** no forbidden pair anywhere in the recent path (broad tripwire).

It reimplements the probe/insane-risk heuristics *independently* of the code it audits (an
adversarial check that imports the audited code shares its blind spots), and touches nothing on the
execution path. It does **not** fix RED-1 — that's the pre-registered Blue-Team change in
`reflect_cycle`; the guard is that fix's standing regression test, and until the fix lands, I1/I3
correctly keep the wound loud.

*This map is a snapshot; regenerate the ratings when a layer's enforcement changes.*
