# Petrules Gate — Phase 1 Groundwork (free-data plumbing)

**Date:** 2026-07-21 · **Status:** GROUNDWORK COMPLETE (free-data only) ·
**Spec:** `research/PETRULES_GATE_SPEC.md` · **Audit:** `research/PETRULES_GATE_data_audit.md`

Autonomous background groundwork done **before** the paid data + hash-locked pre-registration
land. It builds *only* the data/replay plumbing the spec permits pre-prereg. **No learning,
model, calibration, or sizer code exists here** — the spec forbids it until
`data/research/preregister/PETRULES_GATE_prereq.json` is filed and hash-locked (not yet present).

---

## What's built (all free / public data)

| Module | Role |
|---|---|
| `provenance.py` | The anti-lookahead core. `Provenanced(value, source, published_ts)` + `knowable_at(published_ts, freeze_ts) = published_ts < freeze_ts` (STRICT). No value can enter an example without a publication timestamp — so an un-audited value cannot exist. |
| `sources.py` | Free-source layer: Alpha Vantage `EARNINGS` (label + earnings spine, 1996+) and SEC EDGAR submissions (Form 4 / 13D-G / 13F filing dates). Live-fetch when reachable, else committed **real** fixtures. Returns `None` (→ ABSENT) when unavailable — never fabricates. |
| `free_features.py` | The earnings-surprise **label** and the free features, each emitted as a `Provenanced` timestamped by filing/print date only. |
| `replay_engine.py` | Enumerates earnings events, computes a conservative pre-print **freeze timestamp**, assembles `FrozenEvent`s, and runs a build-time `audit()` that raises on any leak. |
| `build_dataset.py` | Writes the small sample to `output/sample_events.{jsonl,csv}`. |
| `fixtures/` | Verbatim real probe responses (AV EARNINGS AAPL/DKS; 13F filing/period pairs). Not synthetic. |
| `tests/test_petrules_no_lookahead.py` | **The leakage audit** — 14 tests, all passing. |

**Features built free (with strict filing-timestamp gating):**
- `earnings_surprise` (LABEL) — AV `EARNINGS`, timestamped by `reportedDate` (the print → forward by construction).
- `earnings_surprise_history` (T2) — prior-quarter beats/mean, only quarters with `reportedDate < freeze`.
- `disclosed_form4_cluster` (T1) — ≥3 Form-4 filings within 30d, gated by **filing** date (direction/buy-sell parsing from the Form-4 XML is the one remaining Phase-1+ extension).
- `activist_disclosure_recent` (T1) — 13D/G filed within 90d; queries both `SC 13D` and `SCHEDULE 13D` form names (the 2025 rename caught in Phase 0).
- `institutional_accumulation` (T1) — latest 13F gated by **filing** date, never the period-of-report (the classic ~45-day 13F leak).

## Leakage-audit result

`python3 -m pytest tests/test_petrules_no_lookahead.py -v` → **14 passed**. It proves the guard
has teeth *and* audits the built sample:
- `knowable_at` is strict-before; a filing dated the freeze day is rejected; a value with no ts is never trusted.
- **Real-date 13F trap:** Berkshire period-end 2026-03-31 filed 2026-05-15 — at a freeze of 2026-04-15, timestamping by period-of-report would leak; by filing date it is correctly excluded.
- Paid interfaces raise `NotImplementedError` (no silent fabrication).
- Package imports nothing from `sovereign/` / `ict*` / the execution path.
- On the built sample (214 real AAPL+DKS events): every present feature's `published_ts < freeze_ts`; every label is at/after freeze; `FrozenEvent.audit()` re-raises on an injected post-freeze feature.

## Built-sample proof

Offline from real fixtures: **214 frozen earnings events** (AAPL + DKS), each with the real
surprise label and a `earnings_surprise_history` feature drawn strictly from prior quarters.
Disclosed-flow features are **ABSENT** offline (EDGAR needs network) — recorded as null with
their source named, never faked. With network + AV key on Colin's machine,
`python3 -m research.petrules.build_dataset AAPL MSFT ...` fills the EDGAR features live.

---

## What's stubbed, and exactly what each paid source unlocks

`paid_stubs.py` — each raises `NotImplementedError` with source, cost, coverage, and the
provenance timestamp the live impl must attach. Wiring each is a one-file change.

| Stub | Paid source · cost | Unlocks |
|---|---|---|
| `options_implied_move` | ThetaData options EOD — VALUE tier on file (2020-01→present); STANDARD (~paid) → 2012 | The **consensus baseline** `consensus_move_pct` for all setups. The Gate has no priced-in reference without it — the whole divergence label depends on it. |
| `options_skew_direction` | ThetaData (same) | T1 put/call skew sign vs consensus direction. |
| `implied_move_change_30d` | ThetaData (same) | T1 **replacement** for the dropped `consensus_revision_momentum` — the only honest expectation-dynamics signal that survived Phase 0. |
| `consensus_revision_momentum` | Refinitiv I/B/E/S or Zacks (NDL premium) — out of budget | The original Tier-1 revision velocity. DROPPED by the Phase-0 narrowing (no free point-in-time revision path); stub kept for a future paid pre-reg update. |

## Blockers / notes

- **Sandbox network is firewalled** (AV/EDGAR return 403 tunnel), so live EDGAR features
  could not be exercised here — the pipeline is built and unit-audited offline against real
  cached fixtures; the live path runs on Colin's machine where the probes originally ran.
- **Pre-registration not yet filed** — this is correct: only plumbing was built. No model,
  calibration, or sizer code was written. The moment `PETRULES_GATE_prereq.json` is
  hash-locked and paid data is wired, Phase 2 (training) can begin against this pipeline.
- Per the Phase-0 narrowing, `consensus_revision_momentum` and `congressional_trade_direction`
  remain out of Tier 1; the pre-reg must bake that in before hash-lock.

## Run

```bash
python3 -m research.petrules.build_dataset            # offline sample (AAPL, DKS)
python3 -m pytest tests/test_petrules_no_lookahead.py -v   # the leakage audit (14 tests)
```
