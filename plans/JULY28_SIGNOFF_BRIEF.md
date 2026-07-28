# July 28 Unlock — Sign-Off Brief (read in 30 seconds)

## TICK-024 — swap/carry cost model was wrong

- **What was broken:** the live backtester's swap-cost table understated real OANDA
  financing charges by **~9x on average** (range 5.5x–11.9x), measured two independent
  ways: actual OANDA trade financing vs. the model, and today's OANDA rate quotes vs.
  the model. Both agree.
- **Extra bug:** EUR/USD short trades were modeled as a *cost* — OANDA actually *pays a
  credit*. Sign is flipped, not just the magnitude.
- **What this touches:** every headline number gated on the old cost model — the
  canonical Sharpe 0.6886 anchor, the OOS Sharpe 1.25, HYP-045 (AUDNZD exclusion).
  The exclusion logic itself is probably still right; the reported Sharpes will move.
- **Does the edge survive honest costs? UNKNOWN — that's the point of the impact study.**
  EUR/USD gets easier (credit not cost), GBP/AUD/JPY get harder (~7-12x more drag).
  Net direction on the live 4-pair blend is not knowable without re-running — which is
  exactly what TICK-024's acceptance criteria require before any table change lands.
- **Status tonight:** fix is staged but the diff file itself is **broken** — malformed
  hunk header, `git apply --check` fails with "corrupt patch." Needs to be regenerated
  before it can be applied tomorrow. Not a drift issue, a diff-authoring issue.

## TICK-044 — daily loss halt (-2%) has never been able to fire

- One-line version: the safety switch meant to halt trading after a -2% day was **inert**
  — nothing in the live path was ever updating the P&L counter it reads, so it always saw
  0.0% loss. Root-caused, fix staged, **verified clean apply** against current code.
  Ready to land tomorrow.

## Bottom line for sign-off

| | Status |
|---|---|
| TICK-024 patch | ❌ broken diff — needs regeneration before applying |
| TICK-044 patch | ✅ applies cleanly, ready |
| Ignition gate | CLOSED (correctly — both blockers still unmet) |
| Isolation test | green |
| Net edge after honest costs | unknown, pending impact study — no headline number should be trusted until then |
