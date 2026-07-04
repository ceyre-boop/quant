# R4 — Factory + research harness retargeting audit (2026-07-03, static, read-only)

## Verdicts
- [fast_backtester] → feeds TODAY: benchmark telemetry only · could feed: the backbone
  of hypothesis testing + discovery sweeps · gap: run_positioning_family.py:360 loads
  pre-computed v015 trades (v015_replay.py:29-32) and runs 10k event-permutations;
  the engine idles · rewiring: signal→simulate wrapper for FUTURE families ·
  RETARGET — **SYNTHESIS RULING: NOT for HYP-072..081.** The family's primaries are
  de-overlapped event stats BY PRE-REGISTRATION (hash-locked manifest); swapping in a
  backtest mid-family is a prereg violation. The fast engine retargets the NEXT
  generation (post-family), plus the discovery bench.
- [run_positioning_family] → feeds ledger interim seals (2026-07-02 run manifest,
  10k perms, seed 42) · could feed faster verdicts for future families · gap: no
  simulation stage (by prereg design for THIS family) · LEAVE for the locked family;
  the successor harness is the retarget.
- [cpcv/discovery] → wired in factory/validation.py:66 (walk_forward_report),
  train.py deliberately defers labels until a CONFIRMED hypothesis defines them
  (train.py:59-62 — Article 6 by construction) · LEAVE.
- [bench job com.alta.bench] → telemetry to 3 files + ledger, ZERO consumers · could
  feed perf-regression alerting · gap: no reader · cheapest: 5-line regression check →
  health · ATTIC-CANDIDATE (Colin's ruling) or the cheap alert wire.
- [research_factory] → dry-run routing (config/autonomous.yml::live=false), keyword
  router, shadow results · LEAVE (drained until live flag + labels; safe by design).

## The performance truth (bench evidence, data/research/bench_history.jsonl)
- 2026-06-29 18:06 with Numba ACTIVE: 728k bt/s single-core, 1.259M parallel,
  1.67B bar-evals/s — the "1.26M/s" claim, real but Numba-dependent.
- 2026-06-29 22:02 Numba INACTIVE (python 3.14.4 incompat): 25.2k single / 123.2k
  parallel / 11.1M bar-evals/s. **~10× regression with a known cause.**
- Ticket: re-enable Numba (env pin or numba upgrade) — env change, not today mid-race.

## Headlines
1. Fast engine idles while research replays static trades — retarget applies to the
   NEXT family generation + discovery, never to the locked 072-081 protocol.
2. The 1.26M/s capability exists but is OFF (Numba × py3.14) — one env ticket
   recovers 10×.
3. factory/train's "missing" label stage is Article 6 working as intended — LEAVE;
   bench telemetry with no reader is the only true attic candidate here.
