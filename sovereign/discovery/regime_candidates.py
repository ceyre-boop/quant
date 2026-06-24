"""
sovereign/discovery/regime_candidates.py
=========================================
Regime-router candidates = the UNGATED base macro signal × one macro-state filter.
Each is judged not just "beats random" but "beats the ungated base" (delta-significance)
and "positive every OOS year" (walk-forward) — the bar that protects against the
HYP-044 cherry-picking failure.

Families (macro state = VIX / SPY-vs-200SMA / rate-trend):
  • deployed VIX gate (per-pair thresholds) — re-validate HYP-027 on current data
  • uniform VIX-bucket gates {15,18,20,22}
  • bull-only / bear-only (SPY vs 200SMA)
  • carry-widening (trade only when the rate differential is widening in the signal's direction)
  • low-VIX-only
"""
from __future__ import annotations

import numpy as np

from sovereign.discovery.gate import Candidate

# deployed HYP-027 thresholds (mirror signal_engine._VIX_GATES)
_DEPLOYED_VIX = {"USDJPY=X": 15.0, "AUDNZD=X": 15.0, "EURUSD=X": 18.0,
                 "GBPUSD=X": 18.0, "AUDUSD=X": 20.0}


def generate(adapter) -> list[Candidate]:
    def base(pair):
        return np.asarray(adapter.dataset(pair).signals).astype(np.int8)

    cands: list[Candidate] = []

    # 1) deployed VIX gate (per-pair thresholds) — the sanity / re-validation candidate
    def vixgate_deployed(pair, pdf, fdf):
        s = base(pair).copy()
        thr = _DEPLOYED_VIX.get(pair, 18.0)
        adverse = (fdf["spy_bull"].to_numpy() > 0) & (fdf["vix"].to_numpy() > thr)
        s[adverse] = 0
        return s
    cands.append(Candidate(
        "regime_vixgate_deployed", "vixgate_deployed",
        "Deployed HYP-027 VIX gate (per-pair thresholds) — re-validate vs ungated base",
        vixgate_deployed, meta={"family": "regime", "spec": "SPY>200SMA & VIX>pair_thr -> suppress"}))

    # 2) uniform VIX-bucket gates
    for thr in (15, 18, 20, 22):
        def mk(t):
            def fn(pair, pdf, fdf):
                s = base(pair).copy()
                adverse = (fdf["spy_bull"].to_numpy() > 0) & (fdf["vix"].to_numpy() > t)
                s[adverse] = 0
                return s
            return fn
        cands.append(Candidate(
            f"regime_vix{thr}", f"vixgate_{thr}", f"Suppress in bull + VIX>{thr} (uniform)",
            mk(thr), meta={"family": "regime", "spec": f"bull & VIX>{thr}"}))

    # 3) bull-only / bear-only
    def bull_only(pair, pdf, fdf):
        s = base(pair).copy(); s[fdf["spy_bull"].to_numpy() < 1] = 0; return s

    def bear_only(pair, pdf, fdf):
        s = base(pair).copy(); s[fdf["spy_bull"].to_numpy() > 0] = 0; return s
    cands += [
        Candidate("regime_bull_only", "bull_only", "Trade carry only when SPY>200SMA",
                  bull_only, meta={"family": "regime", "spec": "spy_bull only"}),
        Candidate("regime_bear_only", "bear_only", "Trade carry only when SPY<200SMA",
                  bear_only, meta={"family": "regime", "spec": "spy_bear only"}),
    ]

    # 4) carry-widening: keep signals only where the rate differential is widening
    #    in the signal's direction (mechanism: carry pays when the gap is opening, not just positive)
    def carry_widening(pair, pdf, fdf):
        s = base(pair).copy()
        mom = np.sign(fdf["rate_diff_mom"].to_numpy())
        keep = (np.sign(s) == mom) & (s != 0)
        out = np.zeros_like(s)
        out[keep] = s[keep]
        return out
    cands.append(Candidate(
        "regime_carry_widening", "carry_widening",
        "Trade only when the rate differential is widening in the signal's direction",
        carry_widening, meta={"family": "regime", "spec": "sign(signal)==sign(rate_diff_mom)"}))

    # 5) low-VIX only
    def lowvix(pair, pdf, fdf):
        s = base(pair).copy(); s[fdf["vix"].to_numpy() > 20] = 0; return s
    cands.append(Candidate(
        "regime_lowvix", "lowvix_only", "Trade carry only when VIX<20",
        lowvix, meta={"family": "regime", "spec": "VIX<20"}))

    return cands
