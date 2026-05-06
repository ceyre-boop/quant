"""
PresentState — Unified Six-Dimension Snapshot.

Answers: *What is actually true about the market right now?*

The six dimensions:
  1. price_regime:     What price is doing      (Hurst, HMM, ADX, regime router)
  2. macro_regime:     Why price is doing it     (rates, inflation, CB policy)
  3. positioning:      Who is positioned how     (COT z-score, commercial net)
  4. narrative:        What the market believes  (TradingAgents / Qwen3 — stub)
  5. historical_match: What this has looked like (Alexandrian Library — stub)
  6. catalyst_window:  When resolution expected  (OU model + CB calendar)

When all six dimensions agree the future is as constrained as it gets without
a crystal ball.  The alignment_score (0–1) quantifies that convergence.

This is NOT a new gate.  It is the shared substrate from which all existing
gates already draw.  Building it once at the top of _run_symbol_session makes
the system's view of the present explicit, coherent, and observable.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Optional

from contracts.types import (
    CatalystWindowState,
    HistoricalMatchState,
    MacroRegimeState,
    NarrativeState,
    PositioningState,
    PresentState,
    PriceRegimeState,
    RouterOutput,
    SovereignFeatureRecord,
)

logger = logging.getLogger(__name__)

# COT symbol defaults for equity universe symbols
_EQUITY_COT_MAP: dict[str, str] = {
    'META': 'NQ', 'GOOGL': 'NQ', 'AMZN': 'NQ', 'NVDA': 'NQ', 'TSLA': 'NQ',
    'SPY':  'ES', 'QQQ':   'NQ',
    'TLT':  'ZN', 'GLD':   'GC', 'USO': 'CL',
}


def _safe_float(value: object, default: float = 0.0) -> float:
    """Return ``float(value)`` or ``default`` when value is NaN / None."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


class PresentStateBuilder:
    """
    Assembles a PresentState from the existing system components.

    All data comes from sources already computed before this is called:
      - SovereignFeatureRecord  (regime, momentum, macro)
      - RouterOutput            (regime label + confidence)
      - Optional z-score series for OU fit
      - cb_calendar             (queried live for catalyst window)
      - cot.py                  (optional live COT pull)

    Stubs are returned when a source isn't available yet (narrative,
    historical_match).  Stubs are marked source='none' so downstream
    code can weight them accordingly.
    """

    def build(
        self,
        symbol: str,
        feature_record: SovereignFeatureRecord,
        router_out: RouterOutput,
        *,
        current_price: float = 0.0,
        atr: float = 0.0,
        ou_half_life_days: Optional[float] = None,
        ou_reversion_days: Optional[float] = None,
        ou_confidence: Optional[str] = None,
        trade_direction: Optional[str] = None,  # 'LONG' | 'SHORT' | None
    ) -> PresentState:
        """Build and return the PresentState for one symbol at one moment."""
        ts = feature_record.timestamp

        price_state = self._build_price_regime(feature_record, router_out)
        macro_state = self._build_macro_regime(feature_record)
        position_state = self._build_positioning(symbol, feature_record)
        narrative_state = self._build_narrative()
        historical_state = self._build_historical_match()
        catalyst_state = self._build_catalyst_window(
            symbol, ou_half_life_days, ou_reversion_days, ou_confidence
        )

        active, score, label = self._compute_alignment(
            trade_direction,
            price_state,
            macro_state,
            position_state,
            narrative_state,
            historical_state,
            catalyst_state,
        )

        return PresentState(
            symbol=symbol,
            timestamp=ts if isinstance(ts, str) else ts.isoformat(),
            price_regime=price_state,
            macro_regime=macro_state,
            positioning=position_state,
            narrative=narrative_state,
            historical_match=historical_state,
            catalyst_window=catalyst_state,
            dimensions_active=active,
            alignment_score=score,
            alignment_label=label,
        )

    # ------------------------------------------------------------------
    # Dimension 1 — Price Regime
    # ------------------------------------------------------------------

    @staticmethod
    def _build_price_regime(
        record: SovereignFeatureRecord,
        router_out: RouterOutput,
    ) -> PriceRegimeState:
        r = record.regime
        return PriceRegimeState(
            hurst_short=_safe_float(r.hurst_short, 0.5),
            hurst_long=_safe_float(r.hurst_long, 0.5),
            hurst_signal=r.hurst_signal or 'NEUTRAL',
            hmm_state=int(r.hmm_state or 0),
            hmm_state_label=r.hmm_state_label or 'NORMAL',
            hmm_transition_prob=_safe_float(r.hmm_transition_prob, 0.2),
            adx=_safe_float(r.adx, 20.0),
            adx_signal=r.adx_signal or 'WEAK',
            regime=router_out.regime,
            regime_confidence=float(router_out.regime_confidence),
        )

    # ------------------------------------------------------------------
    # Dimension 2 — Macro Regime
    # ------------------------------------------------------------------

    @staticmethod
    def _build_macro_regime(record: SovereignFeatureRecord) -> MacroRegimeState:
        m = record.macro
        return MacroRegimeState(
            yield_curve_slope=_safe_float(m.yield_curve_slope),
            yield_curve_velocity=_safe_float(m.yield_curve_velocity),
            erp=_safe_float(m.erp),
            cape_zscore=_safe_float(m.cape_zscore),
            cot_zscore=_safe_float(m.cot_zscore),
            m2_velocity=_safe_float(m.m2_velocity, 1.5),
            hyg_spread_bps=_safe_float(m.hyg_spread_bps, 200.0),
            macro_signal=m.macro_signal or 'NEUTRAL',
        )

    # ------------------------------------------------------------------
    # Dimension 3 — Positioning
    # ------------------------------------------------------------------

    @staticmethod
    def _build_positioning(
        symbol: str,
        record: SovereignFeatureRecord,
    ) -> PositioningState:
        """
        Attempt a live COT pull via cot.py.  Falls back to the cot_zscore
        already present in macro features (which comes from the factor zoo).
        """
        cot_sym = _EQUITY_COT_MAP.get(symbol.upper(), 'NQ')
        cot_z = float('nan')
        source = 'none'

        try:
            from sovereign.features.macro.cot import get_cot_zscore
            cot_z = get_cot_zscore(cot_sym)
            source = 'CFTC'
        except Exception:
            pass

        # Fall back to macro feature cot_zscore if live pull failed
        if math.isnan(cot_z):
            fallback = _safe_float(record.macro.cot_zscore, float('nan'))
            if not math.isnan(fallback):
                cot_z = fallback
                source = 'macro_features'
            else:
                cot_z = 0.0
                source = 'none'

        # Classify bias
        if cot_z > 1.0:
            bias = 'LONG_HEAVY'
        elif cot_z < -1.0:
            bias = 'SHORT_HEAVY'
        else:
            bias = 'NEUTRAL'

        return PositioningState(
            cot_zscore=round(cot_z, 3),
            cot_symbol=cot_sym,
            positioning_bias=bias,
            source=source,
        )

    # ------------------------------------------------------------------
    # Dimension 4 — Narrative (stub until TradingAgents integration)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_narrative() -> NarrativeState:
        """
        Returns a neutral stub.  Replace this with a TradingAgents / Qwen3
        call once the integration is wired.  The source='none' flag tells
        alignment scoring to ignore this dimension until then.
        """
        return NarrativeState(
            summary='TradingAgents not yet integrated',
            sentiment='NEUTRAL',
            confidence=0.0,
            source='none',
        )

    # ------------------------------------------------------------------
    # Dimension 5 — Historical Match (stub until Alexandrian Library)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_historical_match() -> HistoricalMatchState:
        """
        Returns a neutral stub.  Replace this with an Alexandrian Library
        call once built.  The source='none' flag tells alignment scoring
        to ignore this dimension until then.
        """
        return HistoricalMatchState(
            regime_label='UNKNOWN',
            similarity_score=0.0,
            volumes_converging=0,
            typical_outcome='No historical match available',
            lookback_period_days=0,
            source='none',
        )

    # ------------------------------------------------------------------
    # Dimension 6 — Catalyst Window
    # ------------------------------------------------------------------

    @staticmethod
    def _build_catalyst_window(
        symbol: str,
        ou_half_life_days: Optional[float],
        ou_reversion_days: Optional[float],
        ou_confidence: Optional[str],
    ) -> CatalystWindowState:
        """
        OU model data comes from the caller (orchestrator computes it
        separately for each symbol).  CB calendar is queried fresh.

        The CB calendar is loaded via importlib rather than a normal package
        import so that sovereign.forex.__init__ (which eagerly imports heavy
        optional dependencies) is never executed.
        """
        # OU defaults when not provided
        half_life = float(ou_half_life_days or 20.0)
        reversion = float(ou_reversion_days or 20.0)
        confidence = ou_confidence or 'LOW'

        # CB calendar — find the nearest upcoming meeting across all banks
        nearest_bank: Optional[str] = None
        nearest_days: int = 999
        blackout = False

        try:
            # Load cb_calendar directly to avoid triggering sovereign.forex.__init__,
            # which imports optional forex dependencies not present in all environments.
            import importlib.util as _ilu
            import pathlib as _pl
            _cb_path = _pl.Path(__file__).parent / 'forex' / 'cb_calendar.py'
            _spec = _ilu.spec_from_file_location('_cb_calendar', _cb_path)
            _cb = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
            _spec.loader.exec_module(_cb)  # type: ignore[union-attr]

            today = date.today()
            for bank in _cb.CB_MEETINGS:
                try:
                    days = _cb.get_days_to_next_meeting(bank, today)
                    if days < nearest_days:
                        nearest_days = days
                        nearest_bank = bank
                except Exception:
                    continue
            if nearest_bank:
                blackout = _cb.is_in_blackout_period(nearest_bank, today)
        except Exception as exc:
            logger.debug(f"[PresentState] CB calendar unavailable: {exc}")

        # Urgency label
        if nearest_days <= 7:
            urgency = 'IMMINENT'
        elif nearest_days <= 21:
            urgency = 'NEAR'
        elif nearest_days < 999:
            urgency = 'DISTANT'
        else:
            urgency = 'NONE'

        return CatalystWindowState(
            ou_half_life_days=round(half_life, 1),
            ou_reversion_days=round(reversion, 1),
            ou_confidence=confidence,
            next_cb_event_bank=nearest_bank,
            next_cb_event_days=nearest_days,
            cb_blackout_active=blackout,
            catalyst_urgency=urgency,
        )

    # ------------------------------------------------------------------
    # Alignment scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alignment(
        trade_direction: Optional[str],
        price: PriceRegimeState,
        macro: MacroRegimeState,
        pos: PositioningState,
        narr: NarrativeState,
        hist: HistoricalMatchState,
        cat: CatalystWindowState,
    ) -> tuple[int, float, str]:
        """
        Count active dimensions and how many agree with trade_direction.

        Returns (dimensions_active, alignment_score, alignment_label).
        """
        active = 0
        aligned = 0
        direction = (trade_direction or '').upper()

        # Dimension 1 — price regime (always active)
        active += 1
        regime_ok = (
            (direction == 'LONG'  and price.regime == 'MOMENTUM') or
            (direction == 'SHORT' and price.regime == 'REVERSION') or
            price.regime != 'FLAT' or
            not direction
        )
        if regime_ok:
            aligned += 1

        # Dimension 2 — macro regime (always active)
        active += 1
        macro_ok = (
            (direction == 'LONG'  and macro.macro_signal in ('RISK_ON', 'NEUTRAL')) or
            (direction == 'SHORT' and macro.macro_signal in ('RISK_OFF', 'NEUTRAL')) or
            not direction
        )
        if macro_ok:
            aligned += 1

        # Dimension 3 — positioning (active when source != 'none')
        if pos.source != 'none':
            active += 1
            pos_ok = (
                (direction == 'LONG'  and pos.positioning_bias != 'SHORT_HEAVY') or
                (direction == 'SHORT' and pos.positioning_bias != 'LONG_HEAVY') or
                not direction
            )
            if pos_ok:
                aligned += 1

        # Dimension 4 — narrative (active when source != 'none')
        if narr.source != 'none':
            active += 1
            narr_ok = (
                (direction == 'LONG'  and narr.sentiment == 'BULLISH') or
                (direction == 'SHORT' and narr.sentiment == 'BEARISH') or
                narr.sentiment == 'NEUTRAL' or not direction
            )
            if narr_ok:
                aligned += 1

        # Dimension 5 — historical match (active when source != 'none')
        if hist.source != 'none':
            active += 1
            hist_ok = hist.similarity_score >= 0.5 or not direction
            if hist_ok:
                aligned += 1

        # Dimension 6 — catalyst window (always active; non-imminent = aligned)
        active += 1
        cat_ok = cat.catalyst_urgency != 'IMMINENT' or not direction
        if cat_ok:
            aligned += 1

        score = round(aligned / active, 3) if active > 0 else 0.0

        if score >= 0.99:
            label = 'FULL'
        elif score >= 0.75:
            label = 'STRONG'
        elif score >= 0.50:
            label = 'PARTIAL'
        elif score > 0.0:
            label = 'WEAK'
        else:
            label = 'NONE'

        return active, score, label
