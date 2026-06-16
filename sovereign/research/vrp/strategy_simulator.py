"""Iron-condor strategy simulator — inert WITHOUT a loader, live WITH one.

The pre-registered VRP harvesting strategy (short ~1-SD iron condor, 30-45 DTE, manage at
50% profit / 21 DTE) requires HISTORICAL SPY/QQQ OPTION CHAINS. The system has none for
free (yfinance = current only); they arrive via ThetaData (Phase II). The brief's hard rule
holds — use real prices or do not test, never synthesize option prices.

Discipline reconciliation: `iron_condor_simulate(loader=None)` returns the DATA_INSUFFICIENT
contract (no loader => no fabricated P&L — the inert default the tests guard). Given a real
(or mock) loader implementing the ThetaDataLoader contract, it delegates to
`run_iron_condor_backtest`, which prices every leg from the supplied chains — no synthesis.

The frozen spec lives here verbatim so the day chains exist the build is fill-in-the-loader,
not design. Strike selection uses prior 20-day REALIZED vol of SPY daily (free), not VIX.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from clawd_trading.meta_evaluator.metrics_calculator import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe,
    calculate_win_rate,
)
from sovereign.research.vrp import vrp_calculator as vc

CONTRACT_MULTIPLIER = 100      # 1 equity option contract = 100 shares
STRIKE_INCREMENT = 1.0         # SPY strikes ~$1 near the money (frozen choice)
TRADING_DAYS = 252

IRON_CONDOR_SPEC = {
    "structure": "short iron condor",
    "short_legs": "1.0 SD expected move (prior 20d realized vol of SPY daily), symmetric around spot",
    "long_wings": "25 points outside each short strike (defined risk)",
    "duration": "30-45 DTE entry, manage at 21 DTE",
    "entry": "open weekly on Monday",
    "exit": "first of: spread value <= 50% of credit (take), 21 DTE, or spread value >= 2x credit (stop)",
    "stop_interpretation": "2x credit = close when spread value reaches 2x credit received "
                           "(realized loss = 1x credit); binds before defined max loss",
    "sizing": "1% account risk per position on defined max loss (instrument-agnostic)",
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


def _data_insufficient() -> dict:
    """The inert contract — returned whenever no loader (=> no real chains) is supplied."""
    return {
        "status": "DATA_INSUFFICIENT",
        "reason": "No historical SPY/QQQ option chains available; brief forbids synthesizing option prices.",
        "strategy_spec": IRON_CONDOR_SPEC,
        "cost_model_spec": COST_MODEL_SPEC,
        "required_data": REQUIRED_DATA,
    }


def iron_condor_simulate(loader=None, *, spy_daily=None, params=None, split=None,
                         vix_daily=None, account: float = 100_000.0) -> dict:
    """Public entry. INERT without a loader (preserves the no-fabrication guard); with a
    loader implementing the ThetaDataLoader contract it runs the real backtest.

    loader     : ThetaDataLoader | MockThetaDataLoader | None
    spy_daily  : pd.Series of SPY daily closes (free/yfinance) for realized-vol + spot
    params     : the FROZEN options_backtest['params'] dict (pre-registration)
    split      : (start, end) ISO dates of the window to test
    vix_daily  : optional pd.Series of VIX close for regime bucketing
    """
    if loader is None or spy_daily is None or params is None or split is None:
        return _data_insufficient()
    return run_iron_condor_backtest(loader, spy_daily, params, split,
                                    vix_daily=vix_daily, account=account)


# ── helpers (pure) ──────────────────────────────────────────────────────────────
def _nearest_strike(x: float, inc: float = STRIKE_INCREMENT) -> float:
    return round(round(x / inc) * inc, 2)


def _snap(avail: list[float], target: float) -> float:
    """Nearest strike that actually trades — real chains aren't a $1 grid far OTM."""
    return min(avail, key=lambda s: abs(s - target))


def _realized_vol_daily(spy_daily: pd.Series, asof, window: int = 20) -> float | None:
    """σ_daily = stdev of the last `window` daily log returns strictly BEFORE `asof`."""
    r = vc.log_returns(spy_daily)
    r = r[r.index < pd.Timestamp(asof)]
    if len(r) < window:
        return None
    return float(np.std(r.iloc[-window:].to_numpy(float), ddof=1))


def _leg_mid(chain: pd.DataFrame, strike: float, right: str) -> float | None:
    row = chain[chain["strike"] == strike]
    if row.empty:
        return None
    return float(row.iloc[0][f"{right}_mid"])


def _leg_spread(chain: pd.DataFrame, strike: float, right: str) -> float:
    """Bid-ask width of one leg (for slippage)."""
    row = chain[chain["strike"] == strike]
    if row.empty:
        return 0.0
    return float(row.iloc[0][f"{right}_ask"] - row.iloc[0][f"{right}_bid"])


def _condor_value(chain: pd.DataFrame, sc, sp, lc, lp) -> float | None:
    """Net mid value to CLOSE the short condor = (sc-lc) call spread + (sp-lp) put spread."""
    parts = [_leg_mid(chain, sc, "call"), _leg_mid(chain, lc, "call"),
             _leg_mid(chain, sp, "put"), _leg_mid(chain, lp, "put")]
    if any(p is None for p in parts):
        return None
    sc_m, lc_m, sp_m, lp_m = parts
    return (sc_m - lc_m) + (sp_m - lp_m)


def _select_expiry(chain: pd.DataFrame, dte_min: int, dte_max: int):
    """Earliest qualifying expiration whose dte ∈ [dte_min, dte_max]. Returns (expiration, dte)."""
    q = chain[(chain["dte"] >= dte_min) & (chain["dte"] <= dte_max)]
    if q.empty:
        return None, None
    dte = int(q["dte"].min())
    exp = q[q["dte"] == dte]["expiration"].iloc[0]
    return exp, dte


def _trading_mondays(spy_daily: pd.Series, start, end):
    """Each week's first available trading day on/after Monday, within [start, end]."""
    idx = spy_daily.index
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    seen, out = set(), []
    for ts in idx:
        key = (ts.isocalendar().year, ts.isocalendar().week)
        if key not in seen:
            seen.add(key)
            out.append(ts)
    return out


# ── the backtest (treats loader as the known ThetaDataLoader contract) ──
def run_iron_condor_backtest(loader, spy_daily: pd.Series, params: dict, split,
                             vix_daily=None, account: float = 100_000.0) -> dict:
    start, end = split
    sd_mult = float(params["short_legs_sd"])
    wing = float(params["wing_points"])
    dte_min, dte_max = params["dte_entry"]
    manage_dte = int(params["manage_dte"])
    take = float(params["profit_take_pct"])
    stop_x = float(params["stop_x_credit"])
    risk_pct = float(params["account_risk_pct"])
    comm = float(params["commission_per_contract_leg_side"])
    slip_pct = float(params["slippage_pct_of_bidask"])
    rv_window = int(params["rv_window_days"])

    trades, skips = [], []
    for monday in _trading_mondays(spy_daily, start, end):
        spot = float(spy_daily.loc[monday])
        sigma = _realized_vol_daily(spy_daily, monday, rv_window)
        if sigma is None:
            skips.append({"date": str(monday.date()), "reason": "insufficient realized-vol history"})
            continue
        entry_chain = loader.get_chain_for_dte_range("SPY", monday, dte_min, dte_max)
        exp, dte = _select_expiry(entry_chain, dte_min, dte_max)
        if exp is None:
            skips.append({"date": str(monday.date()), "reason": "no expiry in DTE window"})
            continue
        dte_trading = max(1, round(dte * TRADING_DAYS / 365.0))
        move = spot * sigma * math.sqrt(dte_trading)
        chain0 = entry_chain[entry_chain["expiration"] == exp]
        avail = sorted(float(s) for s in chain0["strike"].dropna().unique())
        if len(avail) < 4:
            skips.append({"date": str(monday.date()), "reason": "too few strikes in chain"})
            continue
        # snap target strikes to strikes that actually trade
        sc = _snap(avail, spot + sd_mult * move)
        sp = _snap(avail, spot - sd_mult * move)
        lc = _snap(avail, sc + wing)
        lp = _snap(avail, sp - wing)
        if not (lp < sp < sc < lc):
            skips.append({"date": str(monday.date()), "reason": "degenerate strikes after snap"})
            continue
        credit = _condor_value(chain0, sc, sp, lc, lp)
        if credit is None or credit <= 0:
            skips.append({"date": str(monday.date()), "reason": "no credit / missing strikes"})
            continue
        wing_width = max(lc - sc, sp - lp)         # actual defined-risk width from snapped strikes
        max_loss = wing_width - credit
        if max_loss <= 0:
            skips.append({"date": str(monday.date()), "reason": "non-positive max loss"})
            continue
        contracts = math.floor((account * risk_pct) / (max_loss * CONTRACT_MULTIPLIER))
        if contracts < 1:
            skips.append({"date": str(monday.date()), "reason": "size < 1 contract"})
            continue

        entry_slip = slip_pct * sum(_leg_spread(chain0, k, r)
                                    for k, r in [(sc, "call"), (lc, "call"), (sp, "put"), (lp, "put")])

        # ── daily MTM until first exit: take / stop / 21-DTE ──
        days = spy_daily.index[(spy_daily.index > monday) & (spy_daily.index <= pd.Timestamp(exp))]
        exit_value, exit_reason, exit_date, exit_chain, dte_at_exit = credit, "expiry", exp, chain0, 0
        for day in days:
            dte_now = (pd.Timestamp(exp) - day).days
            chain_t = loader.get_option_chain("SPY", day, exp)
            val = _condor_value(chain_t, sc, sp, lc, lp)
            if val is None:
                continue
            if val <= take * credit:
                exit_value, exit_reason, exit_date, exit_chain, dte_at_exit = val, "profit_take", day, chain_t, dte_now
                break
            if val >= stop_x * credit:
                exit_value, exit_reason, exit_date, exit_chain, dte_at_exit = val, "stop", day, chain_t, dte_now
                break
            if dte_now <= manage_dte:
                exit_value, exit_reason, exit_date, exit_chain, dte_at_exit = val, "manage_21dte", day, chain_t, dte_now
                break

        exit_slip = slip_pct * sum(_leg_spread(exit_chain, k, r)
                                   for k, r in [(sc, "call"), (lc, "call"), (sp, "put"), (lp, "put")])
        gross_pts = credit - exit_value                            # sold for credit, buy back at exit_value
        gross_usd = gross_pts * contracts * CONTRACT_MULTIPLIER
        commission = comm * 4 * 2 * contracts                      # 4 legs x open+close
        slip_usd = (entry_slip + exit_slip) * contracts * CONTRACT_MULTIPLIER
        costs = commission + slip_usd
        net = gross_usd - costs
        trades.append({
            "entry_date": str(monday.date()), "exit_date": str(pd.Timestamp(exit_date).date()),
            "expiration": str(pd.Timestamp(exp).date()), "dte_entry": dte, "dte_at_exit": dte_at_exit,
            "exit_reason": exit_reason, "spot": round(spot, 2), "sigma_daily": round(sigma, 5),
            "short_call": sc, "short_put": sp, "long_call": lc, "long_put": lp,
            "credit": round(credit, 4), "max_loss": round(max_loss, 4), "contracts": contracts,
            "gross_usd": round(gross_usd, 2), "costs": round(costs, 2), "net": round(net, 2),
            "ret_on_account": net / account,
            "entry_vix": (float(vix_daily.reindex([monday]).ffill().iloc[0])
                          if vix_daily is not None and len(vix_daily) else None),
        })

    return _summarize(trades, skips, account, split)


def _summarize(trades: list, skips: list, account: float, split) -> dict:
    if not trades:
        return {"status": "NO_TRADES", "split": list(split), "n_trades": 0,
                "skips": skips[:50], "n_skips": len(skips)}
    rets = [t["ret_on_account"] for t in trades]
    nets = [t["net"] for t in trades]
    equity = [account]
    for n in nets:
        equity.append(equity[-1] + n)
    wr, wins, total = calculate_win_rate(nets)
    max_dd, dd_dur = calculate_max_drawdown(equity)

    def _both_sides_by_sigma():
        sig = sorted(t["sigma_daily"] for t in trades)
        med = sig[len(sig) // 2]
        hi = [t["net"] for t in trades if t["sigma_daily"] > med]
        lo = [t["net"] for t in trades if t["sigma_daily"] <= med]
        return {"high_realized_vol_entry": {"n": len(hi), "mean_net": round(float(np.mean(hi)), 2) if hi else None},
                "low_realized_vol_entry": {"n": len(lo), "mean_net": round(float(np.mean(lo)), 2) if lo else None}}

    def _regime():
        v = [t for t in trades if t.get("entry_vix") is not None]
        if not v:
            return {"note": "no VIX series supplied"}
        hi = [t["net"] for t in v if t["entry_vix"] > 30]
        lo = [t["net"] for t in v if t["entry_vix"] <= 30]
        return {"entry_VIX_gt30": {"n": len(hi), "mean_net": round(float(np.mean(hi)), 2) if hi else None},
                "entry_VIX_le30": {"n": len(lo), "mean_net": round(float(np.mean(lo)), 2) if lo else None}}

    reasons: dict = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

    return {
        "status": "OK", "split": list(split), "n_trades": total, "n_skips": len(skips),
        "net_total": round(sum(nets), 2), "total_costs": round(sum(t["costs"] for t in trades), 2),
        "win_rate": wr, "wins": wins,
        "sharpe_weekly_ann": calculate_sharpe(rets, risk_free_rate=0.0, periods_per_year=52),
        "max_drawdown_pct": max_dd, "max_dd_duration": dd_dur,
        "profit_factor": calculate_profit_factor(nets),
        "exit_reason_counts": reasons,
        "both_sides_by_realized_vol": _both_sides_by_sigma(),
        "regime_conditional": _regime(),
        "trades": trades,
    }
