"""ES/NQ adaptive session sizing — the 3-trade ladder.

Trade 1 (probe, 0.5%): are we right about today?
Trade 2: press 1.0% if the probe won, pullback 0.25% if it lost.
Trade 3 (runner, 0.5%): only if trades 1 AND 2 both won.
Hard daily stop: −1.5% of account ends the session. Max 3 trades.

Stage-3 comparison arm: flat 0.5% on identical trades. If adaptive does not beat
flat with p<0.05, flat ships — simpler is better when results are equal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from sovereign.es_nq.config import contract_spec, es_nq_params


@dataclass
class SessionLadder:
    account_usd: float
    flat_risk_pct: Optional[float] = None      # set → flat sizing (Stage-3 baseline)
    realized_r: list = field(default_factory=list)
    realized_usd: float = 0.0
    params: Optional[dict] = None

    def _p(self) -> dict:
        return (self.params or es_nq_params())["sizing"]

    def next_role(self) -> Optional[str]:
        """Role of the next trade, or None when the session is done."""
        if self.halted():
            return None
        n = len(self.realized_r)
        if n >= self._p()["max_trades_per_session"]:
            return None
        if n == 0:
            return "probe"
        if n == 1:
            return "press" if self.realized_r[0] > 0 else "pullback"
        if self.realized_r[0] > 0 and self.realized_r[1] > 0:
            return "runner"
        return None

    def risk_pct(self, role: str) -> float:
        if self.flat_risk_pct is not None:
            return self.flat_risk_pct
        return float(self._p()["ladder"][role])

    def contracts(self, role: str, stop_points: float, instrument: str) -> int:
        """floor(risk_usd / (stop_points × $/pt)); 0 means skip the trade."""
        if stop_points <= 0:
            raise ValueError(f"stop_points must be positive, got {stop_points}")
        spec = contract_spec(instrument)
        risk_usd = self.account_usd * self.risk_pct(role)
        return max(0, math.floor(risk_usd / (stop_points * spec["dollars_per_point"])))

    def halted(self) -> bool:
        return self.realized_usd <= -self.account_usd * self._p()["daily_loss_cap_pct"]

    def record(self, r: float, usd: float) -> None:
        self.realized_r.append(float(r))
        self.realized_usd += float(usd)
