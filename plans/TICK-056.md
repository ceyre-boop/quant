# Plan — TICK-056: MT5 Execution Bridge (DEMO-only)

- **Spec (LAW):** `specs/mt5_bridge.md`
- **Ticket:** TICK-056 (`tickets/backlog.md`)
- **Effort:** E4 | **Est.:** 3–5 days, gated on the §Open Decision
- **Freeze status:** NEW infrastructure only; zero edits to frozen execution code. No unlock consumed.

## Approach

Build a transport that consumes a decoupled `order_intent` JSON contract and routes it to an MT5
**demo** account behind a hard guard and human approval. Isolate the terminal-specific connector
behind a `Connector` interface so the platform decision (below) is swappable and the guard / contract
/ idempotency / approval logic can be built and unit-tested now without a terminal.

## Files touched

All NEW (see spec §11). Nothing under the freeze is opened. `config/mt5.yml` holds placeholders only —
no credentials committed; any parameter added is logged to `param_change_log.jsonl` per NN#4.

## Build sequence

1. **Contract + validation** (`contract.py`, tests) — pure, no terminal. `OrderIntent.parse()` +
   `.validate()` with the spec §5 rules.
2. **Guard** (`guard.py`, tests) — demo-vs-live check against a `Connector`; live-unlock gate
   (env `ALTA_MT5_ALLOW_LIVE=1` AND unlock file, both required). Pure logic over the interface.
3. **Connector interface + MockConnector** (`connector.py`, tests) — `MT5Connector` is a thin real
   adapter (guarded by a lazy `import MetaTrader5` that fails loud on Darwin); `MockConnector` drives
   all unit tests.
4. **CLI + flow** (`mt5_bridge.py`) — `--selftest` / `--stage` / `--route --approve`, idempotency
   ledger, order card, append-only `mt5_routed.jsonl`.
5. **AST isolation test** — assert no import from the frozen execution path.
6. **Docs** — operator remediation + first-demo-validation steps into NEXT.md.

## Risks

- **Platform blocker (primary).** `MetaTrader5` is Windows-only; the Mac cannot import it. Mitigated
  by the `Connector` seam, but the real connector cannot be validated in-sandbox — only on Colin's
  chosen platform. No fabricated fill; `--selftest` is the honest first live-terminal check.
- **The5%ers symbol naming** (e.g. `EURUSD` vs `EURUSD.r`) differs per broker → `config/mt5.yml`
  symbol map; `--selftest` lists Market Watch symbols so the map is filled from truth, not guessed.
- **Signal-source mismatch.** `execute_daily.py` today is the *equities/Alpaca* path; the proven edge
  for a forex prop firm is `carry_v015`. Which producer emits `order_intent` is a follow-on wiring
  decision — deliberately out of scope so the bridge stays producer-agnostic.

## Open Decision (BLOCKS connector code — Colin's call)

Which MT5 host? **(A)** Windows box/VM, **(B)** Wine-prefix Python on the Mac, **(C)** socket-EA
bridge. Spec §8a has the tradeoffs. Steps 1–3 + 5 (contract, guard, mock, isolation) are identical
across all three and can proceed immediately; step 4's real connector and all live-terminal validation
wait on this answer.

## Acceptance

Ticket TICK-056 acceptance criteria, verbatim.
