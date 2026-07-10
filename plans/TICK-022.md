# TICK-022 — Prop-Funnel EV Simulator (Phase A of the "$10k/month" program)

**Approved:** 2026-07-10 (plan-mode approval, Claude Code / Molly session).
**Master plan (full detail):** `Plans/glistening-juggling-clover.md` — same directory on case-insensitive APFS; this file is the ticket-conventional pointer + build checklist.

## One-paragraph summary

Measurement instrument, not edge validation. New isolated module `research/prop_funnel/` simulates every strategy family (carry PROVEN/regime-fragile, ICT UNPROVEN, futures ORB UNVALIDATED, synthetic frontier) through realistic prop rulesets (FTMO-style static-DD 2-phase no-time-limit; APEX-style intraday-trailing with κ-bracket; Topstep-style EOD-trailing) and reports the funnel: pass rates (incl. p^100), time-to-pass, fees-to-funded, funded survival, monthly payout distribution, P(≥$10k every month ×12), 24-month program EV/month, plus a {challenge_mult × funded_mult} sizing optimization and the Sharpe×vol requirements frontier with the pass-vs-income tension chart. Parity-first gate: reproduce the three recorded MC artifacts before any new engine code. `sovereign/propfirm/rules_engine.py` stays byte-identical.

## Build order & gates

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | this ticket + plan | — |
| P1 | `_lib.py`, `parity.py`, isolation + parity tests | ALL 3 parity checks green |
| P2 | `rulesets.py`, `config/firms.yaml`, ruleset tests | hand-computed scenarios + PropFirmRules equivalence |
| P3 | `feeds.py` + tests | n==27 ict_live assert; fail-loud on missing files |
| P4 | `simulate.py`, `funnel.py`, determinism test | Sharpe-0 → 25–30% band; crude-MC anchors in CI |
| P5 | `sizing_opt.py` + frontier runs | runtime sane (`--fast` works) |
| P6 | `report.py`, `run_all.py`, artifacts, NEXT.md, push | verdict table stamps + caveats present |
| PR (optional, operator-gated) | futures replay regeneration via `--out-dir` on `scripts/futures_replay.py` | Colin approval; never writes `data/futures/` |

## Hard constraints

- No hypothesis-ledger writes. No live/execution-path/config changes. No launchd. No OANDA.
- Write-safety: monkeypatch `monte_carlo_prop.OUT`; nothing lands in `data/futures/`, `data/risk/prop_monte_carlo.json`, or `data/propfirm/`.
- Isolation AST wall + sovereign whitelist {propfirm, risk.monte_carlo_prop, discovery, reporting}.
- Every output row evidence-stamped; carry regime caveat verbatim; i.i.d.-attempts caveat on the table itself.

## Open questions for Colin (non-blocking, flagged in report)

1. Carry $-scale convention (`pnl_pct` vs `risk_adjusted_pnl_pct`) — default = `monte_carlo_prop` convention, R = pnl_pct/risk_pct.
2. Real fee/payout pricing — all `UNVERIFIED_PRICING` until confirmed; EV rankings provisional.
