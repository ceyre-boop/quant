#!/usr/bin/env python3
"""
FTMO Swing Challenge Monte Carlo Simulation
Uses v013 forex system trade distribution (actual backtest parameters).

FTMO Swing Account Rules:
  - Profit target:       10%
  - Max drawdown:        10% trailing from equity high (not fixed)
  - Daily loss limit:    NONE (swing account)
  - Min trading days:    4 distinct calendar days with a trade
  - Max challenge days:  60
  - Weekend/overnight:   ALLOWED (swing account)

v013 Trade Distribution (from backtest results):
  Portfolio avg Sharpe: 1.8552
  5 pairs: USDJPY, AUDUSD, GBPUSD, EURUSD, AUDNZD
  Per-pair Sharpe: 1.558–1.770
  Hold periods: 5-7 days per pair
  Trades/pair/month: ~3-5 (inferred from Sharpe + hold period)
  WR: ~52% (from confirmed counter-momentum 52% WR finding)
  Avg Win:  +1.25R (trail 1.25x ATR; TP1 1.5R, TP2 3R structure)
  Avg Loss: -1.0R
  Risk per trade: 0.75% account (London+GradeA standard)
"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime

RNG = np.random.default_rng(42)

# ── Trade distribution parameters (v013, 5-pair portfolio) ──────────────────
WIN_RATE        = 0.52
AVG_WIN_R       = 1.25    # per PTJ structure: mix of TP1(1.5R) + TP2(3R) + runners
AVG_LOSS_R      = 1.00
RISK_PER_TRADE  = 0.0075  # 0.75% per trade
TRADES_PER_DAY  = 0.45    # portfolio: ~5 pairs × 3-4 trades/month / 21 trading days
                           # ≈ 0.45 trades/day across portfolio (realistic for macro swing)

# ── FTMO Swing rules ────────────────────────────────────────────────────────
PROFIT_TARGET_PCT   = 0.10
MAX_DD_PCT          = 0.10   # trailing from equity high
DAILY_LOSS_LIMIT    = None   # swing account: no daily limit
MIN_TRADING_DAYS    = 4
MAX_CHALLENGE_DAYS  = 60
INITIAL_EQUITY      = 100_000

N_TRIALS = 10_000


def simulate_trial(seed_offset: int) -> dict:
    rng = np.random.default_rng(42 + seed_offset)

    equity     = INITIAL_EQUITY
    equity_high = INITIAL_EQUITY
    trading_days_with_trade = set()
    day        = 0
    n_trades   = 0

    while day < MAX_CHALLENGE_DAYS:
        day += 1

        # Poisson draw: trades today
        n_today = rng.poisson(TRADES_PER_DAY)

        for _ in range(n_today):
            n_trades += 1
            trading_days_with_trade.add(day)

            risk_dollars = equity * RISK_PER_TRADE
            if rng.random() < WIN_RATE:
                pnl = risk_dollars * AVG_WIN_R
            else:
                pnl = -risk_dollars * AVG_LOSS_R

            equity += pnl
            equity_high = max(equity_high, equity)

            # Trailing drawdown bust check (checked after each trade)
            drawdown = (equity_high - equity) / equity_high
            if drawdown >= MAX_DD_PCT:
                return {
                    "result": "BUST",
                    "day": day,
                    "equity": round(equity, 2),
                    "n_trades": n_trades,
                    "trading_days": len(trading_days_with_trade),
                    "peak_equity": round(equity_high, 2),
                    "max_dd": round(drawdown * 100, 2),
                }

        # Check profit target after each day's trades
        gain = (equity - INITIAL_EQUITY) / INITIAL_EQUITY
        if gain >= PROFIT_TARGET_PCT:
            if len(trading_days_with_trade) >= MIN_TRADING_DAYS:
                return {
                    "result": "PASS",
                    "day": day,
                    "equity": round(equity, 2),
                    "n_trades": n_trades,
                    "trading_days": len(trading_days_with_trade),
                    "peak_equity": round(equity_high, 2),
                    "final_gain_pct": round(gain * 100, 2),
                }
            # Hit target but not enough trading days — keep going
            # (already past target, just need more days)

    # Timeout
    gain = (equity - INITIAL_EQUITY) / INITIAL_EQUITY
    return {
        "result": "TIMEOUT",
        "day": day,
        "equity": round(equity, 2),
        "n_trades": n_trades,
        "trading_days": len(trading_days_with_trade),
        "peak_equity": round(equity_high, 2),
        "final_gain_pct": round(gain * 100, 2),
    }


def run_simulation():
    print(f"\n{'═'*60}")
    print(f"  FTMO SWING CHALLENGE — Monte Carlo ({N_TRIALS:,} trials)")
    print(f"{'═'*60}")
    print(f"\nTrade distribution (v013 forex system):")
    print(f"  Win rate:         {WIN_RATE*100:.0f}%")
    print(f"  Avg win:          +{AVG_WIN_R:.2f}R")
    print(f"  Avg loss:         -{AVG_LOSS_R:.2f}R")
    print(f"  Risk/trade:       {RISK_PER_TRADE*100:.2f}%")
    print(f"  Trades/day (ptf): {TRADES_PER_DAY:.2f}")
    print(f"\nFTMO Swing rules:")
    print(f"  Profit target:    {PROFIT_TARGET_PCT*100:.0f}%")
    print(f"  Max drawdown:     {MAX_DD_PCT*100:.0f}% trailing")
    print(f"  Daily loss limit: NONE (swing)")
    print(f"  Min trading days: {MIN_TRADING_DAYS}")
    print(f"  Max days:         {MAX_CHALLENGE_DAYS}")
    print()

    results = [simulate_trial(i) for i in range(N_TRIALS)]

    passes   = [r for r in results if r["result"] == "PASS"]
    busts    = [r for r in results if r["result"] == "BUST"]
    timeouts = [r for r in results if r["result"] == "TIMEOUT"]

    pass_rate    = len(passes)   / N_TRIALS * 100
    bust_rate    = len(busts)    / N_TRIALS * 100
    timeout_rate = len(timeouts) / N_TRIALS * 100

    print(f"{'─'*60}")
    print(f"  PASS rate:     {pass_rate:6.1f}%  ({len(passes):,} trials)")
    print(f"  BUST rate:     {bust_rate:6.1f}%  ({len(busts):,} trials)")
    print(f"  TIMEOUT rate:  {timeout_rate:6.1f}%  ({len(timeouts):,} trials)")
    print(f"{'─'*60}")

    if passes:
        days_to_pass = [r["day"] for r in passes]
        trades_to_pass = [r["n_trades"] for r in passes]
        print(f"\n  When PASSING:")
        print(f"    Median days:    {np.median(days_to_pass):.0f}")
        print(f"    P25/P75 days:   {np.percentile(days_to_pass, 25):.0f} / {np.percentile(days_to_pass, 75):.0f}")
        print(f"    Median trades:  {np.median(trades_to_pass):.0f}")

    if busts:
        bust_days = [r["day"] for r in busts]
        print(f"\n  When BUSTING:")
        print(f"    Median day:     {np.median(bust_days):.0f}")
        print(f"    Max DD hit:     {np.mean([r['max_dd'] for r in busts]):.1f}% avg")

    # Verdict
    print(f"\n{'═'*60}")
    if pass_rate >= 70:
        verdict = f"✅ BUY THE CHALLENGE  (pass rate {pass_rate:.1f}% ≥ 70% gate)"
    elif pass_rate >= 55:
        verdict = f"⚠️  MARGINAL  (pass rate {pass_rate:.1f}%, review parameter below)"
    else:
        verdict = f"❌ DO NOT BUY  (pass rate {pass_rate:.1f}% < 70% gate)"
    print(f"  VERDICT: {verdict}")

    # Sensitivity: what if win rate or risk/trade changes?
    print(f"\n  Sensitivity (what matters most):")
    for wr_adj, label in [(WIN_RATE - 0.03, "WR 49%"), (WIN_RATE, "WR 52% (base)"), (WIN_RATE + 0.03, "WR 55%")]:
        sims = 2000
        p = sum(
            1 for i in range(sims)
            if _quick_sim(np.random.default_rng(1000+i), wr_adj) == "PASS"
        ) / sims * 100
        print(f"    {label}: {p:.1f}% pass rate")

    print(f"{'═'*60}\n")

    # Save results
    out = {
        "run_at": datetime.utcnow().isoformat(),
        "n_trials": N_TRIALS,
        "params": {
            "win_rate": WIN_RATE,
            "avg_win_r": AVG_WIN_R,
            "avg_loss_r": AVG_LOSS_R,
            "risk_per_trade": RISK_PER_TRADE,
            "trades_per_day": TRADES_PER_DAY,
        },
        "results": {
            "pass_rate": round(pass_rate, 2),
            "bust_rate": round(bust_rate, 2),
            "timeout_rate": round(timeout_rate, 2),
            "median_days_to_pass": round(np.median([r["day"] for r in passes]), 1) if passes else None,
        },
        "verdict": verdict,
    }
    out_path = Path("data/agent/ftmo_swing_mc.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Saved to {out_path}")
    return out


def _quick_sim(rng, win_rate):
    equity = 100_000
    eq_high = 100_000
    tdays = set()
    for day in range(1, 61):
        for _ in range(rng.poisson(TRADES_PER_DAY)):
            tdays.add(day)
            risk = equity * RISK_PER_TRADE
            equity += risk * AVG_WIN_R if rng.random() < win_rate else -risk * AVG_LOSS_R
            eq_high = max(eq_high, equity)
            if (eq_high - equity) / eq_high >= MAX_DD_PCT:
                return "BUST"
        if (equity - 100_000) / 100_000 >= PROFIT_TARGET_PCT and len(tdays) >= MIN_TRADING_DAYS:
            return "PASS"
    return "TIMEOUT"


if __name__ == "__main__":
    run_simulation()
