"""Phase 3 — statistical gauntlet for HYP-091.

Directional monthly-sign timing permutation (each pair's ACTUAL sign sequence is
permuted in time, preserving its long/short ratio exactly) recomputed
financing-consistently via the decomposition, plus Deflated-Sharpe and BH (both
near-vacuous for a family of one — the real guards are the permutation and the
Phase-0 hash-lock) and the OOS holdout. Verdict combines with the pre-registered
null (OOS Sharpe <= 0 OR |corr| > 0.5 => the diversification thesis fails).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.tsmom_hyp091 import backtest as bt
from research.tsmom_hyp091._lib import N_PERM, OOS_START, PAIRS, SEED
from sovereign.discovery.gate import benjamini_hochberg, deflated_sharpe_ratio
from sovereign.reporting.equity_curve import _sharpe


def _years(idx) -> float:
    return max((idx.max() - idx.min()).days / 365.25, 1e-9)


def _ann_sharpe(series: pd.Series) -> float:
    return _sharpe(series.tolist(), _years(series.index))


def run(dec: dict, port: pd.Series, corr_full_abs: float | None,
        n_perm: int = N_PERM, seed: int = SEED) -> dict:
    port = port.dropna()
    actual_S = _ann_sharpe(port)
    oos = port[port.index >= OOS_START]
    is_ = port[port.index < OOS_START]
    oos_S = _ann_sharpe(oos)
    is_S = _ann_sharpe(is_)

    # ── directional sign-timing permutation ──────────────────────────────────
    rng = np.random.default_rng(seed)
    actual_signs = {p: dec["comp"][p]["sign"] for p in PAIRS}
    ge = 0
    for _ in range(n_perm):
        perm = {p: rng.permutation(actual_signs[p]) for p in PAIRS}
        ps = bt.portfolio_returns(dec, perm).dropna()
        if _ann_sharpe(ps) >= actual_S:
            ge += 1
    perm_p = (ge + 1) / (n_perm + 1)

    # ── Deflated Sharpe (family of one -> n_trials=1, no deflation) ───────────
    mean, sd = float(port.mean()), float(port.std())
    sharpe_permonth = mean / sd if sd > 1e-12 else 0.0
    dsr_val, dsr_prob = deflated_sharpe_ratio(sharpe_permonth, n_trials=1, n_obs=12)

    # ── BH (family of one) ────────────────────────────────────────────────────
    bh_survives = benjamini_hochberg([perm_p])[0]

    # ── pre-registered null + verdict ────────────────────────────────────────
    null_sharpe = oos_S <= 0
    null_corr = (corr_full_abs is not None and corr_full_abs > 0.5)
    null_triggered = bool(null_sharpe or null_corr)
    passes = bool(
        (not null_triggered)
        and perm_p < 0.05 and dsr_prob > 0.95 and bh_survives and oos_S > 0
    )
    verdict = "VALID_EDGE" if passes else "NOT_SIGNIFICANT"

    reasons = []
    if null_sharpe:
        reasons.append(f"OOS Sharpe {oos_S:+.3f} <= 0 (null condition a)")
    if null_corr:
        reasons.append(f"|corr| {corr_full_abs:.3f} > 0.5 (null condition b)")
    if not null_triggered and perm_p >= 0.05:
        reasons.append(f"permutation p={perm_p:.4f} >= 0.05")
    if not null_triggered and dsr_prob <= 0.95:
        reasons.append(f"deflated-Sharpe prob={dsr_prob:.3f} <= 0.95")

    return {
        "full_sharpe": actual_S,
        "is_sharpe": is_S,
        "oos_sharpe": oos_S,
        "n_oos_months": int(len(oos)),
        "permutation_p": round(perm_p, 5),
        "n_perm": n_perm,
        "sharpe_permonth": round(sharpe_permonth, 5),
        "deflated_sharpe": round(float(dsr_val), 4),
        "deflated_sharpe_prob": round(float(dsr_prob), 4),
        "bh_survives": bool(bh_survives),
        "null_triggered": null_triggered,
        "null_sharpe_leg": bool(null_sharpe),
        "null_corr_leg": bool(null_corr),
        "verdict": verdict,
        "verdict_reasons": reasons,
    }
