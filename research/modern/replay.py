"""P4: decade replay of all arms (HYP-090).

Replay span (prereg): 2016-07-01 -> 2026-06-30. Selection at day t (data <= t)
applies to day t+1: r_arm[i] = V[sel[t_i], t_i + 1]. All arms share the common
eval calendar, so Sharpes are directly comparable.

Switching-cost variant (verdict criterion 5): when the selected variant changes
t -> t+1, each pair whose implied position (direction x in-subset) differs is
charged a round-trip spread fraction cost_price_pair / close_pair[t], weighted
equal-notional (1/|subset_new|) into the portfolio return at t+1. Zero-cost
handover elsewhere is the prereg-stated idealization.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from research.modern import regimes
from research.modern import selection as sel
from research.modern._lib import OUT_DIR
from research.modern.signal_arrays import PAIRS

REPLAY_START, REPLAY_END = "2016-07-01", "2026-06-30"
N_PLACEBO = 500


def _cost_price_by_pair() -> np.ndarray:
    meta = json.loads((OUT_DIR / "daily_returns.npz.meta.json").read_text())
    return np.array([meta["cost_constants"][p]["cost_price"] for p in PAIRS])


class Replayer:
    def __init__(self, uni: sel.VariantUniverse):
        self.uni = uni
        mask = (uni.index >= REPLAY_START) & (uni.index <= REPLAY_END)
        days = np.where(mask)[0]
        self.sel_days = days[:-1]              # selection happens at t
        self.eval_days = days[1:]              # returns realized at t+1
        self.zfeat = regimes.build_features(uni.index)
        self.scores = {W: uni.window_scores(W) for W in sel.WINDOWS}
        self.vectors = {W: regimes.regime_vectors(self.zfeat, W) for W in sel.WINDOWS}
        self.cost_price = _cost_price_by_pair()

        # per-variant per-pair implied position at each day, resolved lazily
        self.variant_cfg = np.array([v[0] for v in uni.variants])
        self.subset_mask = np.zeros((uni.n_variants, len(PAIRS)), dtype=bool)
        for vi, (_, sub) in enumerate(uni.variants):
            self.subset_mask[vi, list(sub)] = True

    def returns_for(self, selections: np.ndarray) -> np.ndarray:
        return self.uni.V[selections, self.eval_days]

    def switching_costs(self, selections: np.ndarray) -> np.ndarray:
        """Portfolio-return deduction per eval day for config handovers."""
        uni = self.uni
        prev = np.concatenate([[uni.v015_variant], selections[:-1]])
        cost = np.zeros(len(selections))
        for i, (v_new, v_old, t) in enumerate(zip(selections, prev, self.sel_days)):
            if v_new == v_old:
                continue
            pos_new = np.where(self.subset_mask[v_new],
                               uni.position[self.variant_cfg[v_new], :, t], 0)
            pos_old = np.where(self.subset_mask[v_old],
                               uni.position[self.variant_cfg[v_old], :, t], 0)
            changed = pos_new != pos_old
            if not changed.any():
                continue
            w = 1.0 / self.subset_mask[v_new].sum()
            cost[i] = float(np.sum(changed * self.cost_price
                                   / uni.close_by_pair[:, t]) * w)
        return cost

    def run_arm(self, arm: str, W: int) -> dict:
        scores = self.scores[W]
        if arm == "A1":
            selections = sel.a1_select(scores, self.sel_days)
        elif arm == "A2":
            selections = sel.a2_select(scores, self.vectors[W], self.sel_days,
                                       self.uni.v015_variant)
        else:
            raise ValueError(arm)
        r = self.returns_for(selections)
        costs = self.switching_costs(selections)
        return {"arm": arm, "window": W, "selections": selections,
                "returns": r, "returns_costed": r - costs,
                "n_switches": int((np.diff(selections) != 0).sum())}

    def a0_returns(self) -> np.ndarray:
        return self.uni.V[self.uni.v015_variant, self.eval_days]

    def placebo_sharpes(self, W: int, n_seeds: int = N_PLACEBO) -> np.ndarray:
        from research.modern._lib import daily_sharpe
        out = np.empty(n_seeds)
        for s in range(n_seeds):
            selections = sel.a3_select(self.uni.n_variants, self.sel_days, (f"W{W}", s))
            out[s] = daily_sharpe(self.returns_for(selections))
        return out
