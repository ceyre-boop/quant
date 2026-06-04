"""Layer 7 — PROP REAL-TIME CEILING. The layer that passes or blows the challenge.

Returns the maximum risk_pct on THIS trade such that, if its stop is hit AND every currently-open
position hits its stop SIMULTANEOUSLY (the correlated worst case), resulting equity still stays above
BOTH the daily-loss floor and the max-drawdown floor (static or trailing per config), with a safety
buffer. If even zero additional risk would breach, returns 0.

Pure function of RiskState — no broker needed (the live path still calls PropRiskManager directly).
"""
from __future__ import annotations


def prop_ceiling(signal, state, cfg) -> float:
    p = cfg["prop"]
    acct = float(p["account_size"])
    equity = float(state.equity)
    buffer_abs = float(p["safety_buffer_pct"]) * acct

    # Daily-loss floor: day-start equity minus today's loss budget.
    today_pnl = state.daily_realized_pnl + state.daily_open_pnl
    day_start_equity = equity - today_pnl
    daily_loss_budget = float(p["daily_loss_limit_pct"]) * acct
    daily_floor = day_start_equity - daily_loss_budget

    # Max-drawdown floor (trailing from peak, or static from start).
    if p.get("drawdown_type", "trailing") == "trailing":
        dd_floor = float(state.peak_equity) * (1.0 - float(p["max_drawdown_pct"]))
    else:
        dd_floor = float(state.starting_balance) * (1.0 - float(p["max_drawdown_pct"]))

    # The binding floor is the highest one we must stay above, plus the safety buffer.
    binding_floor = max(daily_floor, dd_floor) + buffer_abs

    # Total dollar loss we can absorb from here before breaching.
    loss_budget_now = equity - binding_floor
    if loss_budget_now <= 0:
        return 0.0  # already at the edge — no new risk permitted

    # Worst case: all open positions stop out simultaneously (correlated to 1).
    open_risk_dollars = sum(float(getattr(pos, "risk_pct_at_entry", 0.0))
                            for pos in state.open_positions) * equity

    this_trade_budget = loss_budget_now - open_risk_dollars
    if this_trade_budget <= 0:
        return 0.0  # open positions already consume the entire budget

    return max(0.0, this_trade_budget / equity)
