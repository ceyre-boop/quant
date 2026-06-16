"""Iron-condor strategy simulator — INTENTIONALLY INERT.

The brief's pre-registered VRP harvesting strategy (short ~1-SD iron condor, 30-45 DTE,
manage at 50% profit / 21 DTE) requires HISTORICAL SPY/QQQ OPTION CHAINS (strike-level
bid/ask, 2018-2025). This system has none: yfinance serves only current chains and
data/polygon_client.py exposes no options endpoints. Per the brief's hard rule —
"Do not synthesize or interpolate options data. Use real prices or do not test." — this
function NEVER prices an option and NEVER returns a fabricated P&L. It returns the
DATA_INSUFFICIENT contract describing exactly what would unblock it.

The spec is frozen verbatim so the day real chains exist, the backtest is ready to write
against a known, pre-registered design (no post-hoc tuning).
"""
from __future__ import annotations

IRON_CONDOR_SPEC = {
    "structure": "short iron condor",
    "short_legs": "~1 standard deviation (prior 20d realized vol), symmetric around spot",
    "duration": "30-45 DTE entry, manage at 21 DTE",
    "entry": "open weekly on Monday",
    "exit": "close at 50% max profit OR 21 DTE, whichever first",
    "stop": "close at 2x credit received (defined risk anyway)",
    "sizing": "1% account risk per position on defined max loss (instrument-agnostic risk engine)",
}

COST_MODEL_SPEC = {
    "commission_per_contract_open": 0.65,
    "commission_per_contract_close": 0.65,
    "entry_slippage_pct_of_credit": 0.05,
    "exit_slippage_pct_of_credit": 0.10,
    "fill_basis": "mid-price for backtest; model 50% of bid-ask spread as slippage",
}

REQUIRED_DATA = {
    "instruments": ["SPY option chains", "QQQ option chains"],
    "fields": ["strike", "expiry", "right", "bid", "ask", "mid", "underlying_spot"],
    "history": "2018-2025 daily chains (IS 2018-2020 / OOS 2021-2023 / holdout 2024-2025)",
    "candidate_providers": [
        {"name": "Polygon.io Options", "approx_cost": "$29-199/mo",
         "note": "API-key infra already present; options tier not subscribed"},
        {"name": "ORATS", "approx_cost": "subscription", "note": "clean historical IV surface + greeks"},
        {"name": "CBOE DataShop", "approx_cost": "one-time historical purchase", "note": "authoritative SPX/SPY"},
        {"name": "IVolatility", "approx_cost": "subscription", "note": "academic-friendly historical IV"},
    ],
}


def iron_condor_simulate(*_args, **_kwargs) -> dict:
    """Inert by design. Returns the DATA_INSUFFICIENT contract; never fakes a backtest."""
    return {
        "status": "DATA_INSUFFICIENT",
        "reason": "No historical SPY/QQQ option chains available; brief forbids synthesizing option prices.",
        "strategy_spec": IRON_CONDOR_SPEC,
        "cost_model_spec": COST_MODEL_SPEC,
        "required_data": REQUIRED_DATA,
    }
