# Adversarial Invariant Spec — the pre-registered integrity contract

The Layer-4 (Adversarial) companion to `audit/divergence_spec.md`. Where the divergence spec
proves **backtest ≡ live parity** for the exit manager, this spec asserts the **integrity
invariants** nothing else guards: that the system is not silently succeeding on contaminated
inputs, rogue broker writes, or forbidden instruments.

Read by `audit/invariant_guard.py` (read-only; imports nothing from the execution path). The
single machine-readable fence below is the source of every threshold — it is sha256-hashed and
version-checked at load, exactly like the divergence spec. Changing a threshold is a spec edit
with a git trail, not a code constant buried in the checker.

Full rationale and the four-layer map: `audit/CORRECTNESS_LAYERS.md`.

## The invariants

- **I1 — Oracle reflection purity.** No decision record that *would enter the Oracle reflection
  summary* (outcome attributed, i.e. not `None/OPEN/EXPIRED`, within `window_days`) may be on a
  `forbidden_pair`, bear a probe/`test_fill` tag, sit in a `probe_source`, or carry sentinel
  (insane-risk) levels. This is the standing catch for RED-1 (backfilled probe/forbidden records
  fabricating W/L into cognition). Hard fence: `i1_contaminated_allowed = 0`.
- **I2 — No rogue / unlogged OANDA writes.** In the recent fills ledgers, no fill on a
  `forbidden_pair` and no sentinel-probe fill (1-unit / insane stop). Catches the still-live
  unlogged USD_CAD 1-unit writer. If every fills ledger is stale past
  `fills_staleness_grace_hours`, the guard escalates "cannot verify" rather than passing silently.
  Hard fence: `i2_rogue_allowed = 0`.
- **I3 — Forbidden-pair guard (broad).** No `forbidden_pair` may appear *anywhere* in the recent
  decision logs or fills ledgers, regardless of outcome attribution — an earlier tripwire than I1
  (fires before an outcome is even attributed). Hard fence: `i3_forbidden_allowed = 0`.

Soft signals (escalate IMPORTANT, not a hard-fence FAIL): a record/fill on a pair **not in**
`allowed_pairs` and not explicitly forbidden (unknown instrument — review), and stale fills
ledgers.

```yaml audit-spec
spec_version: 1
window_days: 7
decision_log_dir: data/decision_logs
fills_paths:
  - data/ledger/oanda_fills.jsonl
  - data/execution/fills.jsonl
forbidden_pairs: [AUD_NZD, USD_CAD]
allowed_pairs: [EUR_USD, GBP_USD, USD_JPY, AUD_USD, GBP_JPY]
probe_sources: [test_fill, synthetic, proof_of_life]
insane_risk_fraction: 0.5
fills_staleness_grace_hours: 30
report_dir: audit/reports
messages_cap: 50
i1_contaminated_allowed: 0
i2_rogue_allowed: 0
i3_forbidden_allowed: 0
```

## Discipline

- The fence is the only source of thresholds. `invariant_guard.py` must find exactly one
  `yaml audit-spec` fence or it refuses to run (and still escalates the refusal).
- The guard reimplements the probe/insane-risk and fill↔decision matching heuristics
  *independently* of `scripts/backfill_decision_records.py` — an adversarial check that imports
  the audited code shares its blind spots. `tests/test_invariant_guard.py` cross-checks the two
  agree on the known cases.
- Read-only: the guard imports nothing from `sovereign/execution/` or the OANDA bridge, and
  writes only to `report_dir` and the escalation queue. Enforced by
  `test_invariant_guard_does_not_import_execution`.
- Expected day-1 state: I1/I3 escalate URGENT on the *existing* RED-1 contamination. That is
  correct — the guard makes the open wound loud and keeps it loud until the pre-registered Blue
  fix (source/pair exclusion in `reflect_cycle`) lands. The guard is that fix's regression test.
