# R6 — Dashboard parity audit (2026-07-03, read-only)

## The diff (condensed; full panel→source map in scout return)
| Colin-only (runtime) | Both | Machine-only (board, Colin never sees) |
|---|---|---|
| live conviction + best-trade rank | macro curve/spreads | gdelt_tone (+5d, volume) |
| oracle play (entry/SL/TP) | VIX level/regime | cot_net_pct/_oi/_1y/_z, cot_flush_1w |
| active setups + direction | — | tff_lev_net_pct |
| equity drawdown, health | — | vrp_signal/vrp_pct/iv/rv |
| ICT veto activity, R-multiples | — | rr25, bf25, atm_term_slope |
| session replay | — | econ_surprise_z |

## Verdicts
- [board→dashboard direction] → the board's institutional-positioning family (COT%,
  TFF, VRP, rr25/bf25, surprise_z) feeds NO panel · could feed: a "Positioning
  Board" section (display-only) so Colin sees what the machine sees · gap: one JSON
  export + one panel · cheapest: nightly board-row export → data/agent/
  positioning_board.json + panel · RETARGET (SAFE-NOW, display-only).
- [dashboard→machine direction] → Colin-only items are runtime state (conviction,
  account equity, plays); the honest machine-side capture is JOURNALING them at
  decision time (overlaps R3's decision-logger field preservation) · RETARGET via the
  memory organ, not via the board (board is EOD by design).
- [scout's "wire board into decision_chain Q2/Q3 + score_regime_confidence stubs,
  ~20 lines"] → **SYNTHESIS RULING: REFUSED as stated.** Those are live readiness
  gates; feeding them unproven board signals is live-capital allocation to unproven
  edges (Article 6) — requires a CONFIRMED hypothesis + logged param change, likely
  post-window. Display parity now; gate wiring only behind evidence.
- [live pages themselves] → serve Colin's monitoring well · LEAVE (staleness story
  already known: Render serves committed master data).

## Headlines
1. Machine-blindness is the SMALLER gap; Colin-blindness is bigger: the entire
   positioning board never reaches any panel. Display-only export = SAFE-NOW.
2. The machine's parity fix is memory-side (journal richer decision-time context),
   not board-side — the board is EOD by design.
3. Wiring board signals into live gates without verdicts is the tempting shortcut
   the constitution exists to refuse.
