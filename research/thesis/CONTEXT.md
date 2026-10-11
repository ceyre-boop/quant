# Thesis Engine — Layer 1 router

**What it is.** The process that produced Colin's best real-money result, made repeatable:
one structural thesis → an AI-built concentrated basket → sealed with a hash *before* entry →
held under a pre-defined invalidation → scored vs SPY at 1/3/6/12 months. Over several theses the
ledger separates a good year from a good process. Started 2026-10-10.

**Where things live.** Schema `shared/thesis_ledger_schema.md` (L3). Code `sovereign/thesis/`.
CLI `alta thesis …` (`scripts/thesis.py`). Ledger `data/thesis/thesis_ledger.jsonl`. Each thesis has
`research/thesis/THS-NNN/CONTEXT.md` (L2) and `output/` (L4). Broker fills `data/thesis/fills/`.

## Lifecycle

1. **Propose.** Write `research/thesis/THS-NNN/spec.json` (schema L3). The thesis must name the
   mechanism — who is forced to pay and why — and an invalidation. No invalidation, no thesis.
2. **Register + seal.** `alta thesis new spec.json` → `alta thesis seal THS-NNN`. The hash goes in
   the CONTEXT.md. From here the basket cannot change; a changed mind is a new thesis.
3. **Enter.** Fills happen *after* the seal. `alta thesis live THS-NNN --note "..."`.
4. **Hold.** No touching. Trims, adds, and "just this once" are deviations; they are allowed to
   happen in the account but they are recorded in `notes` and they do not change the score.
5. **Score.** `alta thesis score THS-NNN` whenever asked and at each horizon mark. The mechanical
   invalidation flips status to INVALIDATED by itself; the manual check is printed as a question.
6. **Close.** At horizon or invalidation: `alta thesis` status CLOSED, verdict in CONTEXT.md, one
   line in `research/EDGE_LEDGER.md`.

## Registry

| id | title | entry | status | last score |
|---|---|---|---|---|
| THS-001 | AI infrastructure — sellers of the compute bottleneck (margin) | 2026-08-28 | LIVE (retro) | see `THS-001/CONTEXT.md` |
| THS-001B | same thesis, managed RIA sleeve incl. AI-health | 2026-03-31 | LIVE (retro) | see `THS-001B/CONTEXT.md` |
| THS-002 | AI infrastructure — sellers of the power bottleneck (margin) | 2026-10-13 (planned) | SEALED, ai-proposed | see `THS-002/CONTEXT.md` |

## Rules (not restated elsewhere)

- Retro-registration is allowed **once per pre-existing position** and is labelled
  `source: retro-registered` forever. THS-001/001B are the only ones.
- The engine never places orders. Execution is the operator's, by hand, after the seal.
- Scoring measures the sealed basket. Reconcile measures the account. Both numbers are kept;
  neither is hidden behind the other.
- A thesis is judged at its horizon against SPY, not against the best day it had.
- THS-002 was AI-proposed and sealed 2026-10-11 before any fill. All later theses follow the same order.
- Fund with settled cash before sealing: a bounced deposit force-liquidates the thesis (THS-001B, twice).
