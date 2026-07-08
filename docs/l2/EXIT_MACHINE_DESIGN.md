# L2 Exit-Machine — Live Port Design

> **Status:** DESIGN ONLY (approved 2026-06-27). No code written. Implementation is a separate,
> approved build. Source reading: `sovereign/forex/fast_backtester.py::_simulate_forex_core` (full),
> `sovereign/execution/oanda_bridge.py`, `scripts/forex_live_scan.py`.

---

## 1. The gap

The backtest manages exits via `_simulate_forex_core` — a deterministic state machine stepped once
per **daily bar**. Live execution (`forex_live_scan.py`, daily 08:00 ET, dry-run by default) only
**opens** trades with a static `stopLossOnFill` + `takeProfitOnFill` (`oanda_bridge.place_trade`) and
then **manages nothing**. `oanda_bridge` has **no stop-amend method**. The entire trailing / time /
reversal / cb_refresh exit layer that the backtest simulates is absent live. Most of the realizable
Sharpe in a slow-drift edge lives in capture/protection, so this is the #1 live-vs-backtest gap.

## 2. The 6 states → live mapping (canonical config: `strict_mode=False`)

Only **5** states fire in the proven v015 config. **Donchian is strict-mode-only → INACTIVE.**
Pyramiding is off.

| State | Backtest logic | Live mapping | Native / Poll |
|---|---|---|---|
| Initial stop | entry ∓ 2.0·ATR@entry (fixed) | `stopLossOnFill` (exists) | **NATIVE** |
| Trailing ATR | `best − trail_mult·ATR_today·best`, on daily closes | daily-amend stop to this price | **POLL (daily)** |
| Reversal | `signal_today ≠ 0 and ≠ direction` → close | daily signal flip → `close_trade` | **POLL (daily)** |
| Time | `hold_count ≥ 60` → close | daily hold counter → `close_trade` | **POLL (daily)** |
| CB refresh | `signal==dir and hold<30 and hold≥20` → continue | suppress time/reversal close when it holds | **POLL (daily)** |
| Donchian | strict-mode only | — | **N/A (inactive)** |

### Trailing fidelity — the central decision
OANDA's *native* `trailingStopLoss` uses a **fixed distance** and trails on **every tick**. The
backtest trail is **ATR-adaptive** and acts on **daily closes**. To match the backtest, **do not use
native trailing.** Instead, each day recompute the backtest's trail price
(`best_close − trail_mult · ATR_today · best_close`) and **amend a plain stop-loss to it** (Option C).

- **Option A** — native trailing, fixed distance @ entry ATR. Simplest; diverges (fixed-distance + tick-level).
- **Option B** — native trailing, daily-re-PATCHed distance = trail_mult·ATR_today. ATR-adaptive but still tick-level.
- **Option C (recommended)** — daily amend a *plain* stop to the computed daily-close trail price.
  Exact parity with `_simulate_forex_core` (daily-close, ATR-adaptive). The only residual divergence
  is intraday triggering (§5.1), which is inherent to any broker-held stop.

## 3. Polling loop — minimum frequency = DAILY

`_simulate_forex_core` steps once per daily bar, so the live manager needs **one pass per trading
day**, run right after `forex_live_scan` computes today's signal (~08:00 ET). New
`scripts/forex_exit_manager.py` (or a tail-call from `forex_live_scan`):

```
DAILY (after forex_live_scan writes today's per-pair carry signal):
  for each open OANDA forex trade (bridge.get_open_trades):
      load persisted state: entry_date, direction, best_price, hold_count
      inputs: today's ATR%[pair], today's signal[pair], today's close[pair]
      update: best_price = extreme(best_price, today_close); hold_count += 1
      decide (backtest priority order):
          reversal  : signal flipped against position      -> bridge.close_trade
          time      : hold_count >= 60 (unless cb_refresh)  -> bridge.close_trade
          cb_refresh: signal confirms & 20<=hold<30         -> hold (suppress time/reversal)
          trail     : new_stop = best - trail_mult*ATR_today*best
                      if new_stop tightens the existing stop -> bridge.set_stop(trade_id, new_stop)
      persist updated state
```

**Data each check needs:** today's ATR% per pair (reuse `get_historical_candles` → ATR, or
`signal_engine._compute_atr_pct`); today's carry signal per pair (from `forex_live_scan` /
`signal_engine.build_signal_frame`); and **persisted per-trade state** — OANDA does **not** track
"best price since entry", so the manager owns `data/agent/forex_exit_state.json` (best_price,
hold_count, entry_date keyed by trade_id).

## 4. New `oanda_bridge` capability required
`set_stop(trade_id, price)` — replace an open trade's stop-loss order (oandapyV20
`trades.TradeCRCDO`, Create/Replace/Cancel Dependent Orders; verify the exact endpoint at impl).
This is the **one** net-new bridge method. `close_trade` (exists) covers reversal/time exits.

## 5. Capture/protection parity gaps (live vs backtest)
1. **Intraday vs daily-close.** Backtest acts on daily closes; a broker stop fills *intraday*. An
   intraday spike to the stop that recovers by close → live exits, backtest holds. Live is slightly
   *more* protective. Bounded by the initial ATR stop. Documented & accepted (inherent to broker stops).
2. **Fills/slippage.** Backtest applies a modelled cost (`_apply_costs`); live fills at market
   (spread+slip). The replay-match tolerance (§6) must budget for this.
3. **Entry timing.** Backtest enters next-bar OPEN; live enters at the 08:00-ET scan price. Minor.
4. **cb_refresh re-entry.** Backtest may re-enter same-bar; live separates concerns — the exit-manager
   only CLOSES; the next daily `forex_live_scan` opens any new/opposite position. cb_refresh = "keep
   the position" (suppress the time/reversal close), not a re-open.
5. **Weekend/gaps.** OANDA trades 24/5; the daily poll skips weekends; Monday gaps are handled at the
   next poll.

## 6. Parity guarantee + validation + go-live discipline
- **Share ONE decision function.** Refactor the per-bar exit test out of `_simulate_forex_core` into a
  pure `decide_exit(state, bar) → action` that BOTH the backtester (loops over bars) and the live
  manager (calls once/day) import. Live == backtest *by construction*, not by re-implementation.
  Existing forex regression tests must stay green after the refactor.
- **Shadow first.** The manager DEFAULTS to shadow (logs intended amend/close, executes nothing), like
  `forex_live_scan`'s dry-run.
- **Validation gate (before any live flip):** replay `decide_exit` over the historical signal stream
  and confirm the resulting trades match `_simulate_forex_core` within tolerance (captured-Sharpe
  parity). Then shadow live for a period, comparing intended vs realized.
- **Live flip = logged `param_change`** (`data/agent/param_change_log.jsonl`) with rationale (project
  rule #4). This touches REAL positions — no silent enabling.

## 7. Implementation sequence (NOT YET BUILT — separate approval)
1. Refactor `_simulate_forex_core` → shared `decide_exit`; keep backtester output identical (regression test).
2. Add `oanda_bridge.set_stop(trade_id, price)` (TradeCRCDO) + a unit test against the practice account.
3. Build `scripts/forex_exit_manager.py` in **shadow** (state file, daily pass, logs intended actions).
4. Replay-match validation vs the backtester; shadow-live observation.
5. Flip to live only via a logged `param_change` rationale.

Nothing in steps 1–5 runs until separately approved.
