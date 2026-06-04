"""THE KILLER TEST — run decide() across 10,000 bootstrapped paths of REAL v015 trade returns.

Assert (a) engine P(breach) < threshold, and (b) engine P(breach) < naive fixed-fractional
P(breach) — i.e. the cascade demonstrably REDUCES ruin vs flat sizing. Reports both numbers and
the binding-constraint distribution (which layer actually governs risk).
"""
import collections
import copy
import os

import numpy as np
import pytest

from sovereign.risk.config.loader import load_risk_config
from sovereign.risk.monte_carlo_prop import load_pool
from sovereign.risk.risk_engine import decide
from sovereign.risk.risk_state import RiskState, Signal

N_SIMS = 10_000
N_TRADES = 120
EMBEDDED = 0.0075          # per-trade risk embedded in the pool's pnl_pct
NAIVE_RISK = 0.01          # naive fixed-fractional: flat 1% (same as A+ base, but no survival layers)
DD_FLOOR = 0.08            # 8% trailing-drawdown breach


def _cfg():
    c = copy.deepcopy(load_risk_config())
    c["audit"]["enabled"] = False        # no per-decision disk writes in a 1.2M-call test
    return c


def _edge_stats():
    """Pooled real v015 stats for Kelly (one strategy = the macro edge, n=103)."""
    pnls, _, _ = load_pool()
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    p = len(wins) / len(pnls)
    b = (wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else 2.0
    return {"forex_macro": {"win_rate": float(p), "payoff": float(b), "n_trades": int(len(pnls))}}


def _run():
    cfg = _cfg()
    pnls, _, _ = load_pool()
    edge = _edge_stats()
    rng = np.random.default_rng(7)
    draws = rng.choice(pnls, size=(N_SIMS, N_TRADES), replace=True)

    engine_breaches = 0
    naive_breaches = 0
    binding = collections.Counter()
    sig = Signal("EURUSD=X", 1, 1.1000, 1.0900, "A+", point_value=1.0)

    for s in range(N_SIMS):
        eq_e = eq_n = 100_000.0
        peak_e = peak_n = 100_000.0
        breached_e = breached_n = False
        for t in range(N_TRADES):
            pnl = draws[s, t]
            r_mult = pnl / EMBEDDED                       # trade outcome in R-multiples

            # ── engine path ──
            if not breached_e:
                dd_t = max(0.0, (peak_e - eq_e) / peak_e)
                st = RiskState(equity=eq_e, peak_equity=peak_e, starting_balance=100_000.0,
                               daily_realized_pnl=0.0, daily_open_pnl=0.0,
                               drawdown_trailing=dd_t, drawdown_static=max(0.0, (100_000 - eq_e) / 100_000),
                               edge_stats=edge, threat_score=0.0, health_ok=True)
                d = decide(sig, st, cfg)
                binding[d.binding_constraint] += 1
                eq_e *= (1.0 + r_mult * d.final_risk_pct)
                peak_e = max(peak_e, eq_e)
                if eq_e <= peak_e * (1 - DD_FLOOR) or eq_e <= 100_000 * (1 - DD_FLOOR):
                    breached_e = True

            # ── naive fixed-fractional path (flat, no survival layers) ──
            if not breached_n:
                eq_n *= (1.0 + r_mult * NAIVE_RISK)
                peak_n = max(peak_n, eq_n)
                if eq_n <= peak_n * (1 - DD_FLOOR) or eq_n <= 100_000 * (1 - DD_FLOOR):
                    breached_n = True

        engine_breaches += breached_e
        naive_breaches += breached_n

    return engine_breaches / N_SIMS, naive_breaches / N_SIMS, binding


def test_engine_reduces_ruin_vs_naive():
    p_engine, p_naive, binding = _run()
    total = sum(binding.values())
    dist = {k: round(v / total, 3) for k, v in binding.most_common()}
    print(f"\n  KILLER TEST — 10k bootstrapped real-v015 paths × {N_TRADES} trades")
    print(f"    engine  P(breach 8% DD): {p_engine:.4f}")
    print(f"    naive   P(breach 8% DD): {p_naive:.4f}  (flat {NAIVE_RISK:.1%} fixed-fractional)")
    print(f"    binding-constraint distribution: {dist}")

    # (a) engine breach below an absolute safety threshold
    assert p_engine < 0.05, f"engine P(breach)={p_engine:.4f} exceeds 5% safety threshold"
    # (b) engine demonstrably safer than naive flat sizing
    assert p_engine <= p_naive, f"engine P(breach)={p_engine:.4f} not <= naive {p_naive:.4f}"
