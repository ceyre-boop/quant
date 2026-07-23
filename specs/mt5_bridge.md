# Spec — MT5 Execution Bridge (DEMO-only)

- **Ticket:** TICK-056
- **Status:** LAW (spec-first; connector code must not be written until this is reviewed)
- **Created:** 2026-07-22
- **Purpose:** Prerequisite for Step 3 (opening The5%ers $100K High Stakes challenge). Forcing
  date: FOMC 2026-07-29. The bridge must exist so a green-light FOMC outcome can be *acted on*.

---

## 1. Problem

The sovereign system produces trade signals but has no path to route them into a MetaTrader 5
account — the platform The5%ers requires for its High Stakes challenge. We need a bridge that turns
a signal into an MT5 order **on a demo/practice account only**, with a human in the loop, without
touching any frozen execution code, and without ever being able to fire on a live account by
accident.

## 2. Vision (ideal state)

A small, isolated, new-infrastructure module that:
1. Reads a decoupled **`order_intent`** JSON contract (defined here) — not by importing execution code.
2. Connects to an MT5 terminal, **verifies the account is DEMO**, and refuses everything otherwise.
3. Surfaces the fully-formed order for **human approval**.
4. On approval, calls `MetaTrader5.order_send(...)` against the demo account and records the result.
5. Fails loud — never fakes a fill — when the terminal, package, or credentials are absent.

## 3. Out of scope (explicit)

- **Any live-account routing.** Live is a separate, logged unlock (§8) — not built here.
- **Auto-trading / unattended firing.** Human approval is mandatory (§6).
- **Editing the signal producer** (`execute_daily.py`, `carry_engine`, `ict/pipeline.py`, exit path).
  Wiring a producer to *emit* `order_intent` is a follow-on ticket, not this one.
- **Strategy logic, sizing math, SL/TP derivation.** The bridge is a transport; the intent arrives
  pre-sized. It validates but does not compute edge.

## 4. Constraints (from CLAUDE.md — non-negotiable)

- **Execution-path freeze (until 2026-07-28).** The bridge is NEW files consuming a JSON contract.
  It MUST NOT import `forex_exit_manager`, `decide_exit`, `execution/harness.py`, `carry_engine`, or
  `ict/pipeline.py`. Enforced by AST isolation test.
- **DEMO-ONLY.** Physically incapable of routing live without an explicit, logged unlock (§8).
- **No auto-trading.** Surface → human approve → route. Never fire live automatically.
- **No silent mocking.** Missing `MetaTrader5` / terminal / creds → STOP with exact remediation.
- **Spec-first.** This document precedes connector code.

## 5. Order-routing contract (`order_intent`)

The bridge's only input. A producer writes one JSON file per intent to
`data/execution/mt5_intents/<intent_id>.json`; the bridge consumes it. Decoupling by data contract
(not import) is what keeps the bridge isolation-safe.

```json
{
  "intent_id": "2026-07-29T14:31:00Z_EURUSD_SHORT",
  "created_at": "2026-07-29T14:31:00Z",
  "symbol": "EURUSD",
  "side": "SELL",                    // BUY | SELL
  "order_type": "MARKET",            // MARKET only in v1 (PENDING deferred)
  "volume_lots": 0.50,               // broker lots, pre-sized upstream; > 0
  "sl_price": 1.0925,                // absolute price; required (challenge risk rule)
  "tp_price": 1.0840,                // absolute price; optional (null allowed)
  "comment": "v015_carry_HYP045",    // <= 31 chars (MT5 limit)
  "magic": 5015,                     // int strategy tag
  "source_strategy": "carry_v015",
  "signal_hash": "<frozen_hash>",    // provenance; not verified by bridge in v1
  "max_slippage_points": 20
}
```

**Validation rules (bridge rejects on any failure, loudly):**
- All required fields present and typed; `side ∈ {BUY,SELL}`; `order_type == MARKET`.
- `volume_lots > 0` and within `[min_lot, max_lot]` from config.
- `sl_price` present and on the correct side of an implied entry (SELL → SL above; BUY → SL below).
- `comment` ≤ 31 chars; `magic` is int.
- `intent_id` unique vs an idempotency ledger (`mt5_routed.jsonl`) — never route the same intent twice.

## 6. Approval + routing flow

1. `mt5_bridge.py --stage <intent_id>` — load + validate the intent, connect, verify DEMO, print a
   human-readable order card (symbol, side, lots, SL/TP, account login + **DEMO** banner, server).
   Writes a `pending/<intent_id>.json` staging record. **Does not route.**
2. Human reviews. Approval = `mt5_bridge.py --route <intent_id> --approve` (explicit flag; no flag =
   dry-run that re-prints the card and exits 0 without routing).
3. On `--approve`: re-verify DEMO (guard runs again immediately before `order_send`), submit
   `order_send`, capture the `retcode`/`order`/`deal`/price, append to `mt5_routed.jsonl` (append-only,
   idempotent by `intent_id`), and print the result.
4. Any non-`TRADE_RETCODE_DONE` result is reported verbatim — no retry loop that could double-fire.

## 7. Demo-vs-live guard (the load-bearing invariant)

```
account = MetaTrader5.account_info()
assert account is not None                      # no connection → abort
if account.trade_mode != ACCOUNT_TRADE_MODE_DEMO:   # 0 == DEMO, 2 == REAL, 1 == CONTEST
    ABORT LOUD: "REFUSING: account <login> on <server> is trade_mode=<n> (not DEMO)."
```

- The guard runs at **stage time and again immediately before every `order_send`** (TOCTOU-safe).
- A live route requires BOTH: (a) env `ALTA_MT5_ALLOW_LIVE=1`, and (b) a present, well-formed unlock
  file `data/execution/mt5_LIVE_UNLOCK.json` with a logged rationale + operator signature. Neither
  exists by default; the code path is dead until Colin creates them and records the unlock in NEXT.md.
- Absent the unlock, a non-DEMO account is a hard abort even with the env flag — both are required.

## 8. Failure modes (fail loud, never fake)

| Condition | Behavior |
|---|---|
| `import MetaTrader5` fails (e.g. macOS/Darwin) | STOP. Print exact remediation (§9). Exit non-zero. No fill. |
| Terminal not running / `initialize()` fails | STOP with `last_error()`. No fill. |
| `account_info()` is None | STOP — cannot verify DEMO → cannot route. |
| Account not DEMO | Hard abort (§7). |
| Symbol not found / not in Market Watch | STOP; instruct to enable symbol. No guess. |
| Malformed / missing-field intent | Reject with the failing field named. |
| Duplicate `intent_id` | Refuse (idempotency). |
| `order_send` returns non-DONE | Report `retcode` + comment verbatim; no auto-retry. |

### 8a. PLATFORM BLOCKER — `MetaTrader5` is Windows-only

The `MetaTrader5` pip package binds to the terminal via a Windows-only mechanism. This repo runs on
**macOS (Darwin)** where the package is **not importable** — verified in-sandbox. Colin's Mac cannot
run the native bridge as-is. Options, to be decided BEFORE connector code (see plan §Open Decision):

- **(A) Windows box / VM** running MT5 + Python — the bridge runs there; cleanest, package works natively.
- **(B) Wine-prefix Python** co-located with a Wine/CrossOver MT5 terminal on the Mac — the package can
  work inside the same prefix; fiddlier, but keeps everything on one machine.
- **(C) Socket-EA bridge** — an MT5 Expert Advisor listens on a local socket; a native-macOS Python
  client speaks a tiny JSON protocol to it. Most portable to Mac, but adds a second surface to build.

The demo guard, contract, validation, idempotency, and approval flow are identical across all three —
so those are built and unit-tested now (no terminal needed). Only the *connector* layer differs, and
it is deliberately isolated behind a `Connector` interface so the choice is swappable.

## 9. Operator remediation (what Colin runs on his Mac)

Depends on the §8a decision. Minimum for Option A/B (native package):
```
pip install MetaTrader5                 # Windows or Wine-prefix Python ONLY
# Install MetaTrader 5 terminal, log into The5%ers DEMO/practice server
# Enable "Algo Trading" in the terminal; add the target symbols to Market Watch
```
Then, first demo validation:
```
python mt5_bridge.py --selftest          # connects, prints account, asserts DEMO, routes NOTHING
python mt5_bridge.py --stage <intent_id> # stage a hand-written demo intent
python mt5_bridge.py --route <intent_id> --approve   # DEMO order_send
```

## 10. Test strategy

Unit-testable **without** a terminal (built now):
- Demo guard: mock connector returning DEMO → allowed; REAL/CONTEST → abort; None → abort.
- Contract validation: valid intent passes; each malformed variant rejected with the right field.
- Idempotency: second route of same `intent_id` refused.
- Live-unlock gate: with neither env+file → live path unreachable; with only one → still aborts.
- AST isolation: `mt5_bridge` imports nothing from the frozen execution path.

Requires a terminal (Colin's Mac, documented, not run in-sandbox):
- `--selftest` against The5%ers demo; one staged demo `order_send`; result recorded.

## 11. Files (all NEW)

- `mt5_bridge.py` — CLI + orchestration (stage / route / selftest).
- `sovereign/execution/mt5/connector.py` — `Connector` interface + `MT5Connector` (real) + `MockConnector` (tests).
- `sovereign/execution/mt5/contract.py` — `OrderIntent` parse/validate.
- `sovereign/execution/mt5/guard.py` — demo-vs-live guard + live-unlock gate.
- `config/mt5.yml` — server/login placeholders, min/max lot, symbol map, magic. **No secrets committed.**
- `tests/test_mt5_bridge.py` — the §10 unit tests.
- `data/execution/mt5_intents/` , `data/execution/mt5_routed.jsonl` — I/O (gitignored data).

## 12. Verification (done =)

All §10 unit tests green; AST isolation test green; `--selftest` documented for Colin; ticket
acceptance criteria checked; NEXT.md updated; pushed.
