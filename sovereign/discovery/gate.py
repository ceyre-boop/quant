"""
sovereign/discovery/gate.py
============================
THE methodology gate. Discovery proposes; this decides.

Mirrors the live antibody (sovereign/autonomous/research_factory._methodology_ok):
a candidate is only VALID_EDGE if it clears ALL of —
  • permutation test          : signal timing beats random entries at the same frequency
  • Deflated Sharpe Ratio     : survives the expected-max-Sharpe hurdle for n_trials candidates
                                (AFML Ch.8 — the multiple-testing correction the brute-force scan omits)
  • Benjamini-Hochberg        : survives FDR control across ALL candidates tested this run
  • frozen holdout            : positive on a window never used for selection

The funnel: cheap in-sample screen ranks candidates; only finalists pay for the
expensive permutation/holdout/DSR pass. Backtests here are ~1-10/sec (no Numba),
so `n_perms` defaults to a screening grade; promoting a survivor to the official
ledger means re-running at ≥10k via the existing research_factory.

DSR and BH are reimplemented inline (identical math to walk_forward_validation.
deflated_sharpe_ratio and derive_hypothesis_pvalues.benjamini_hochberg) to keep
this module dependency-light — importing those pulls xgboost/sklearn side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.stats as _stats

from sovereign.reporting.equity_curve import weighted_portfolio_sharpe


# ─── inlined corrections (mirror the canonical repo implementations) ──────────

def deflated_sharpe_ratio(sharpe_obs: float, n_trials: int,
                          sr_std: float = 0.30, n_obs: int = 252) -> tuple[float, float]:
    """AFML Ch.8 Deflated Sharpe. Returns (deflated_sr, prob_sr_above_zero).

    Hurdle rises with n_trials: a Sharpe of 1.0 on the 50th strategy tested is far
    less impressive than on the 1st. Identical to walk_forward_validation.deflated_sharpe_ratio.
    """
    gamma_em = 0.5772156649
    emc = np.e ** (-gamma_em)
    if n_trials <= 1:
        sr_star = 0.0
    else:
        sr_star = (
            (1 - gamma_em * emc) * _stats.norm.ppf(1 - 1.0 / n_trials)
            + gamma_em * emc * _stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        ) * sr_std
    sr_ann = sharpe_obs * np.sqrt(n_obs)
    deflated = (sr_ann - sr_star) / max(sr_std, 1e-8)
    return sr_ann - sr_star, float(_stats.norm.cdf(deflated))


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """FDR control. Returns a survives-mask aligned to `pvalues`.

    Identical algorithm to derive_hypothesis_pvalues.benjamini_hochberg.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    survive_rank = 0
    for rank, idx in enumerate(order, 1):
        if pvalues[idx] <= (rank / m) * alpha:
            survive_rank = rank
    survives = [False] * m
    for rank, idx in enumerate(order, 1):
        survives[idx] = rank <= survive_rank
    return survives


def bootstrap_diff_pvalue(a, b, n_boot: int = 5000, seed: int = 11) -> float:
    """One-sided bootstrap: p = P(mean(a) - mean(b) <= 0) — small p ⇒ a's mean
    return reliably exceeds b's (the regime gate beats the ungated base).

    Mirrors scripts/derive_hypothesis_pvalues.bootstrap_diff_pvalue.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return float(np.mean(diffs <= 0))


def bootstrap_sharpe_diff_pvalue(a, b, n_boot: int = 3000, seed: int = 11) -> float:
    """One-sided bootstrap on per-trade RISK-ADJUSTED return: p = P(sharpe(a) - sharpe(b) <= 0).

    A regime gate's job is to lift risk-adjusted return (Sharpe) by cutting adverse-regime
    trades — it can improve Sharpe while barely changing the mean. So a gate must be tested
    on Sharpe, not mean return. Small p ⇒ the gated config reliably beats the ungated base.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 5 or len(b) < 5:
        return 1.0
    rng = np.random.default_rng(seed)

    def _sh(x):
        s = x.std()
        return x.mean() / s if s > 1e-12 else 0.0

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = _sh(rng.choice(a, len(a), replace=True)) - _sh(rng.choice(b, len(b), replace=True))
    return float(np.mean(diffs <= 0))


# ─── candidate + verdict types ────────────────────────────────────────────────

@dataclass
class Candidate:
    id: str
    name: str
    description: str
    # signal_for(pair, price_df, feat_df) -> np.ndarray[int8] aligned to price_df.index
    signal_for: Callable
    meta: dict = field(default_factory=dict)


@dataclass
class Verdict:
    id: str
    name: str
    description: str
    verdict: str                 # VALID_EDGE | NOT_SIGNIFICANT | SCREENED_OUT
    train_sharpe: float
    full_sharpe: float
    holdout_sharpe: float
    n_trades: int
    perm_p: Optional[float]
    n_perms: int
    dsr_deflated: Optional[float]
    dsr_prob: Optional[float]
    bh_survives: Optional[bool]
    delta_p: Optional[float] = None        # vs ungated base (regime mode); None for generic discovery
    wf_robust: Optional[bool] = None       # positive costed Sharpe in every OOS year
    wf_by_year: Optional[dict] = None
    per_pair: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ─── the gate ─────────────────────────────────────────────────────────────────

class Gate:
    def __init__(self, adapter, *, train_window: tuple[str, str],
                 holdout_window: tuple[str, str], n_perms: int = 500, seed: int = 7,
                 perm_p_max: float = 0.05, dsr_prob_min: float = 0.95,
                 directional_perm: bool = False):
        self.adapter = adapter
        self.train_window = train_window
        self.holdout_window = holdout_window
        self.n_perms = n_perms
        self.rng = np.random.default_rng(seed)
        self.perm_p_max = perm_p_max
        self.dsr_prob_min = dsr_prob_min
        # directional_perm: shuffle the candidate's OWN long/short values (preserve the
        # directional ratio) instead of random ±1. Required for trending/directional assets
        # (equity indices) so a long-biased rule must beat random TIMING, not just capture beta.
        self.directional_perm = directional_perm
        self.pairs = adapter.pairs

    # -- per-candidate signal arrays (computed once, reused across windows) --
    def _signals(self, cand: Candidate, features_by_pair: dict) -> dict:
        out = {}
        for pair in self.pairs:
            pdf = self.adapter.price_df(pair)
            fdf = features_by_pair.get(pair)
            if pdf is None or fdf is None:
                continue
            try:
                sig = np.asarray(cand.signal_for(pair, pdf, fdf)).astype(np.int8)
            except Exception:
                continue
            if sig.shape[0] == len(pdf):
                out[pair] = sig
        return out

    def _portfolio(self, signals_by_pair: dict, window: Optional[tuple] = None):
        per_pair, trades = {}, []
        for pair, sig in signals_by_pair.items():
            res = self.adapter.eval_signals(pair, sig, window=window)
            per_pair[pair] = (res.sharpe, res.n_trades)
            trades.extend(res.trades)
        sharpe = weighted_portfolio_sharpe(list(per_pair.values()))
        n = sum(n for _, n in per_pair.values())
        return sharpe, n, per_pair, trades

    def _permutation_p(self, signals_by_pair: dict, real_sharpe: float) -> float:
        counts = {p: int(np.count_nonzero(s)) for p, s in signals_by_pair.items()}
        n_ge = 0
        for _ in range(self.n_perms):
            per_pair = {}
            for pair, sig in signals_by_pair.items():
                n_bars, n_sig = len(sig), counts[pair]
                perm = np.zeros(n_bars, dtype=np.int8)
                if n_sig > 0:
                    pos = self.rng.choice(n_bars, size=n_sig, replace=False)
                    if self.directional_perm:
                        perm[pos] = self.rng.permutation(sig[sig != 0])  # preserve long/short ratio
                    else:
                        perm[pos] = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=n_sig)
                res = self.adapter.eval_signals(pair, perm)
                per_pair[pair] = (res.sharpe, res.n_trades)
            null_sharpe = weighted_portfolio_sharpe(list(per_pair.values()))
            if null_sharpe >= real_sharpe:
                n_ge += 1
        return (n_ge + 1) / (self.n_perms + 1)  # +1 smoothing (never exactly 0)

    def _base_returns(self, base_signals_by_pair: dict) -> list:
        """Costed per-trade returns of the ungated base signal (the delta-test baseline)."""
        rets = []
        for pair, sig in base_signals_by_pair.items():
            res = self.adapter.eval_signals(pair, np.asarray(sig).astype(np.int8))
            rets.extend(t.get("pnl_pct", 0.0) for t in res.trades)
        return rets

    def evaluate(self, candidates: list[Candidate], features_by_pair: dict,
                 max_finalists: Optional[int] = None, screen_min_trades: int = 20,
                 progress: Optional[Callable] = None,
                 base_signals_by_pair: Optional[dict] = None,
                 wf_years: Optional[list] = None) -> list[Verdict]:
        n_trials = len(candidates)  # multiple-testing correction uses ALL screened
        base_returns = self._base_returns(base_signals_by_pair) if base_signals_by_pair else None
        regime_mode = base_returns is not None
        # ── stage 1: cheap screen on the train window ──
        screened = []
        for i, cand in enumerate(candidates):
            sigs = self._signals(cand, features_by_pair)
            if not sigs:
                continue
            tr_sharpe, tr_n, _, _ = self._portfolio(sigs, window=self.train_window)
            screened.append((cand, sigs, tr_sharpe, tr_n))
            if progress:
                progress(f"screen {i+1}/{n_trials}: {cand.name[:28]:28s} train_sharpe={tr_sharpe:+.2f} n={tr_n}")
        screened = [s for s in screened if s[3] >= screen_min_trades]
        screened.sort(key=lambda s: -s[2])
        finalists = screened if max_finalists is None else screened[:max_finalists]

        # ── stage 2: expensive gate on finalists ──
        results = []
        for cand, sigs, tr_sharpe, tr_n in finalists:
            full_sharpe, full_n, per_pair, trades = self._portfolio(sigs)
            ho_sharpe, _, _, _ = self._portfolio(sigs, window=self.holdout_window)
            if progress:
                progress(f"gate: {cand.name[:28]:28s} full={full_sharpe:+.2f} running {self.n_perms} perms...")
            perm_p = self._permutation_p(sigs, full_sharpe)
            dsr_def, dsr_prob = deflated_sharpe_ratio(full_sharpe, n_trials)
            delta_p = wf_robust = wf_by_year = None
            if regime_mode:
                cand_returns = [t.get("pnl_pct", 0.0) for t in trades]
                # Sharpe-difference (not mean) — a regime gate improves risk-adjusted return.
                delta_p = round(bootstrap_sharpe_diff_pvalue(cand_returns, base_returns), 4)
                if wf_years:
                    wf_by_year = {}
                    for y in wf_years:
                        ys, _, _, _ = self._portfolio(sigs, window=(f"{y}-01-01", f"{y}-12-31"))
                        wf_by_year[y] = round(ys, 3)
                    wf_robust = all(v > 0 for v in wf_by_year.values())
            results.append(Verdict(
                id=cand.id, name=cand.name, description=cand.description,
                verdict="PENDING", train_sharpe=round(tr_sharpe, 3), full_sharpe=round(full_sharpe, 3),
                holdout_sharpe=round(ho_sharpe, 3), n_trades=full_n, perm_p=round(perm_p, 4),
                n_perms=self.n_perms, dsr_deflated=round(dsr_def, 3), dsr_prob=round(dsr_prob, 4),
                bh_survives=None, delta_p=delta_p, wf_robust=wf_robust, wf_by_year=wf_by_year,
                per_pair=per_pair, trades=trades, meta=cand.meta,
            ))

        # ── stage 3: Benjamini-Hochberg across all finalists, then final verdict ──
        if results:
            bh = benjamini_hochberg([r.perm_p for r in results], alpha=self.perm_p_max)
            for r, surv in zip(results, bh):
                r.bh_survives = bool(surv)
                ok = (r.perm_p is not None and r.perm_p < self.perm_p_max
                      and r.dsr_prob is not None and r.dsr_prob > self.dsr_prob_min
                      and r.bh_survives and r.holdout_sharpe > 0)
                if regime_mode:
                    # higher bar: must beat the ungated base AND be positive every OOS year
                    ok = ok and (r.delta_p is not None and r.delta_p < 0.10) and bool(r.wf_robust)
                r.verdict = "VALID_EDGE" if ok else "NOT_SIGNIFICANT"

        # carry through screened-out candidates as a record (no gate run)
        finalist_ids = {c.id for c, *_ in finalists}
        for cand, sigs, tr_sharpe, tr_n in screened:
            if cand.id not in finalist_ids:
                results.append(Verdict(
                    id=cand.id, name=cand.name, description=cand.description,
                    verdict="SCREENED_OUT", train_sharpe=round(tr_sharpe, 3), full_sharpe=0.0,
                    holdout_sharpe=0.0, n_trades=tr_n, perm_p=None, n_perms=0,
                    dsr_deflated=None, dsr_prob=None, bh_survives=None, meta=cand.meta,
                ))
        return results
