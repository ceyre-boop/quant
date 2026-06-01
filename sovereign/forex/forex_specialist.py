"""
ForexSpecialist — top-level orchestrator for the forex system.

Pipeline:
  1. ForexMacroEngine  → macro directional bias per pair
  2. ICTEngine         → price action structure (FVG, OB, sweeps, kill zones)
  3. ForexEntryEngine  → 6-point ICT checklist → entry/stop/target
  4. ForexPositionSizer → risk-based sizing with hard rules

Usage:
    specialist = ForexSpecialist(account_balance=10_000)
    report = specialist.run()
    for sig in report.tradeable:
        print(sig)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from sovereign.forex.entry_engine import ForexEntryEngine, ForexEntrySignal
from sovereign.forex.position_sizer import ForexPositionSizer, PositionSize

logger = logging.getLogger(__name__)


@dataclass
class ForexTradeCandidate:
    entry_signal: ForexEntrySignal
    position: PositionSize

    def summary(self) -> str:
        s = self.entry_signal
        p = self.position
        lines = [
            f"{'─'*60}",
            f"  {s.pair:12s}  {s.direction:5s}  Score: {s.score}/6  Conviction: {s.macro_conviction:.2f}",
            f"  Entry: {s.entry_price:.5f}  Stop: {s.stop_price:.5f}  "
            f"Risk: {p.risk_pct:.1%} (${p.risk_usd:.0f})",
            f"  T1: {s.t1:.5f}  T2: {s.t2:.5f}  T3: {s.t3:.5f}  (R:R T1={s.rr_t1:.1f})",
            f"  Units: {p.units:,.0f}  Size modifier: {s.size_modifier:.0%}",
            f"  Kill Zone: {s.ict_analysis.kill_zone_name or 'N/A'}",
            f"  Macro driver: {s.macro_signal.primary_driver}",
            f"  Rationale:",
        ]
        for r in s.rationale:
            lines.append(f"    {r}")
        return '\n'.join(lines)


@dataclass
class ForexScanReport:
    as_of: pd.Timestamp
    tradeable: List[ForexTradeCandidate] = field(default_factory=list)
    skipped: List[ForexEntrySignal] = field(default_factory=list)

    def print(self) -> None:
        print(f"\n{'═'*60}")
        print(f"  FOREX SCAN  —  {self.as_of.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'═'*60}")
        if not self.tradeable:
            print("  No tradeable setups at this time.")
        else:
            for c in self.tradeable:
                print(c.summary())
        print(f"\n  Evaluated: {len(self.tradeable) + len(self.skipped)} pairs  "
              f"|  Tradeable: {len(self.tradeable)}")
        print(f"{'═'*60}\n")


class ForexSpecialist:

    def __init__(self, account_balance: float = 10_000.0):
        self._entry = ForexEntryEngine()
        self._sizer = ForexPositionSizer(account_balance=account_balance)

    def run(self) -> ForexScanReport:
        """Full scan across all 11 pairs → returns ranked trade candidates."""
        from sovereign.forex.pair_universe import ALL_PAIRS
        report = ForexScanReport(as_of=pd.Timestamp.utcnow())

        # Single pass: evaluate all pairs, collect all signals
        entry_signals: list[ForexEntrySignal] = []
        for pair in ALL_PAIRS:
            try:
                sig = self._entry.evaluate(pair)
                if sig:
                    entry_signals.append(sig)
            except Exception as e:
                logger.debug(f"evaluate failed for {pair}: {e}")

        for sig in entry_signals:
            if not sig.is_tradeable:
                report.skipped.append(sig)
                continue

            pos = self._sizer.size(
                pair=sig.pair,
                entry=sig.entry_price,
                stop=sig.stop_price,
                t1=sig.t1,
                t2=sig.t2,
                t3=sig.t3,
                size_modifier=sig.size_modifier,
            )

            if pos.rejected:
                logger.info(f"{sig.pair}: position rejected — {pos.reject_reason}")
                report.skipped.append(sig)
                continue

            try:
                from sovereign.intelligence.decision_logger import log_forex_decision
                _macro = sig.macro_signal
                log_forex_decision(
                    pair=sig.pair,
                    direction=sig.direction,
                    entry_level=sig.entry_price,
                    stop_loss=sig.stop_price,
                    hold_days=getattr(_macro, "hold_period_estimate", 60),
                    risk_pct=pos.risk_pct,
                    signal_layers=sig.rationale[:6],
                    rate_diff_z=getattr(_macro, "irp_z", None),
                    vix_at_entry=None,   # not available at scan time; wire when VIX gate is live
                    cot_percentile=None,  # COT engine returns z-score not percentile; TODO
                    library_match=None,   # not yet wired into ForexEntrySignal
                    commitment_score=None,
                    freshness_mult=None,
                    kelly_fraction=None,
                    size_mult=sig.size_modifier,
                    extra={
                        "macro_conviction": sig.macro_conviction,
                        "score": sig.score,
                        "rate_differential": getattr(_macro, "rate_differential", None),
                        "ppp_z": getattr(_macro, "ppp_z", None),
                        "primary_driver": getattr(_macro, "primary_driver", None),
                    },
                    # Loop 2: forex entry-time snapshot (this path has no equity
                    # PresentState object; assemble the equivalent from the macro signal).
                    present_state_snapshot={
                        "score": sig.score,
                        "macro_conviction": sig.macro_conviction,
                        "rate_diff_z": getattr(_macro, "irp_z", None),
                        "rate_differential": getattr(_macro, "rate_differential", None),
                        "ppp_z": getattr(_macro, "ppp_z", None),
                        "primary_driver": getattr(_macro, "primary_driver", None),
                        "size_modifier": sig.size_modifier,
                        "hold_days_est": getattr(_macro, "hold_period_estimate", 60),
                    },
                    active_lessons=[],
                )
            except Exception:
                pass

            report.tradeable.append(ForexTradeCandidate(
                entry_signal=sig,
                position=pos,
            ))

        # Sort tradeable by score then macro conviction
        report.tradeable.sort(
            key=lambda c: (c.entry_signal.score, c.entry_signal.macro_conviction),
            reverse=True,
        )

        return report

    def evaluate_pair(self, pair: str) -> Optional[ForexTradeCandidate]:
        """Single-pair evaluation with position sizing."""
        sig = self._entry.evaluate(pair)
        if sig is None or not sig.is_tradeable:
            return None

        pos = self._sizer.size(
            pair=sig.pair,
            entry=sig.entry_price,
            stop=sig.stop_price,
            t1=sig.t1,
            t2=sig.t2,
            t3=sig.t3,
            size_modifier=sig.size_modifier,
        )

        if pos.rejected:
            return None

        return ForexTradeCandidate(entry_signal=sig, position=pos)
