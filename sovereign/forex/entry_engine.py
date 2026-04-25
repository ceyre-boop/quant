"""
Forex entry decision engine — combines macro bias + ICT price action.

Scoring:
  Criterion 1 — HTF bias aligned (macro ForexSignal direction)
  Criterion 2 — Price at OB or FVG zone
  Criterion 3 — Liquidity was hunted first (recent sweep)
  Criterion 4 — Market structure shift on entry TF (CHOCH or BOS)
  Criterion 5 — FVG present on entry TF
  Criterion 6 — Kill Zone timing active

Score 6/6 → full size  |  4–5/6 → 50–75% size  |  <4 → no trade

Buffett conviction gate:
  conviction < 0.35 → no trade (noise filter)
  conviction 0.35–0.70 → normal size
  conviction 0.70–0.85 → full size
  conviction > 0.85 → max size (1.5× modifier cap)

Commodity boost: AUD/CAD/NZD pairs get commodity lag signal as extra conviction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sovereign.forex.macro_engine import ForexMacroEngine, ForexSignal, CONVICTION_NEUTRAL_THRESHOLD
from sovereign.forex.ict_engine import ICTEngine, ICTAnalysis
from sovereign.forex.commodity_engine import CommodityEngine, COMMODITY_PAIRS

logger = logging.getLogger(__name__)


@dataclass
class ForexEntrySignal:
    pair: str
    direction: str          # LONG / SHORT / NO_TRADE
    score: int              # 0–6
    size_modifier: float    # 0.0 (no trade), 0.5, 0.75, or 1.0
    entry_price: float
    stop_price: float
    t1: float               # ~1.5:1 target
    t2: float               # ~3:1 target
    t3: float               # ~5:1 target (weekly liquidity)
    rr_t1: float
    rationale: list[str]
    macro_conviction: float
    ict_analysis: ICTAnalysis
    macro_signal: ForexSignal

    @property
    def is_tradeable(self) -> bool:
        return self.score >= 4 and self.direction != 'NO_TRADE'


class ForexEntryEngine:

    # Hard rules from ICT engine
    MIN_RR = 2.0
    STOP_ATR_MIN = 0.5
    STOP_ATR_MAX = 2.5
    MAX_CONCURRENT = 2

    def __init__(self):
        self._macro = ForexMacroEngine()
        self._ict = ICTEngine()
        self._commodity = CommodityEngine()
        self._open_count = 0

    def evaluate(
        self,
        pair: str,
        daily_df: Optional[pd.DataFrame] = None,
        intraday_df: Optional[pd.DataFrame] = None,
    ) -> Optional[ForexEntrySignal]:
        """
        Evaluate a pair for entry readiness.

        daily_df: OHLCV daily bars (for macro/structure context)
        intraday_df: OHLCV 1H or 4H bars (for entry timing / FVG / OB precision)
        If not provided, data is fetched from yfinance.
        """
        daily_df = daily_df if daily_df is not None else self._download(pair, '1d', '3y')
        if daily_df is None or len(daily_df) < 100:
            logger.warning(f"Insufficient daily data for {pair}")
            return None

        # Use daily for structure if no intraday provided
        entry_df = intraday_df if intraday_df is not None else daily_df

        # ── Macro signal ──────────────────────────────────────────────── #
        macro_sig = self._macro.score_pair(pair)
        if macro_sig is None:
            return None
        if macro_sig.direction == 'NEUTRAL':
            return ForexEntrySignal(
                pair=pair, direction='NO_TRADE', score=0,
                size_modifier=0.0, entry_price=0, stop_price=0,
                t1=0, t2=0, t3=0, rr_t1=0,
                rationale=['Macro neutral — no directional edge'],
                macro_conviction=0.0,
                ict_analysis=self._ict.analyse(pair, entry_df),
                macro_signal=macro_sig,
            )

        # ── Buffett conviction gate ───────────────────────────────────── #
        # Below 0.35 = noise, not a structural edge (cause-effect map Part 10)
        effective_conviction = macro_sig.conviction

        # Commodity boost for AUD/CAD/NZD pairs
        if pair in COMMODITY_PAIRS:
            try:
                comm_sig = self._commodity.score_pair(pair)
                if (comm_sig and comm_sig.direction == macro_sig.direction
                        and comm_sig.lag_detected):
                    boost = comm_sig.conviction * 0.15   # up to +12% conviction
                    effective_conviction = min(effective_conviction + boost, 1.0)
            except Exception:
                pass

        if effective_conviction < CONVICTION_NEUTRAL_THRESHOLD:
            return ForexEntrySignal(
                pair=pair, direction='NO_TRADE', score=0,
                size_modifier=0.0, entry_price=0, stop_price=0,
                t1=0, t2=0, t3=0, rr_t1=0,
                rationale=[
                    f'Conviction {effective_conviction:.2f} below threshold '
                    f'{CONVICTION_NEUTRAL_THRESHOLD} — Buffett filter'
                ],
                macro_conviction=effective_conviction,
                ict_analysis=self._ict.analyse(pair, entry_df),
                macro_signal=macro_sig,
            )

        target_direction = macro_sig.direction  # LONG / SHORT

        # ── ICT analysis ──────────────────────────────────────────────── #
        try:
            ict = self._ict.analyse(pair, entry_df)
        except ValueError as e:
            logger.warning(f"ICT analysis failed for {pair}: {e}")
            return None

        # ── 6-point checklist ─────────────────────────────────────────── #
        score, rationale = self._score_checklist(target_direction, macro_sig, ict)

        if score < 4:
            return ForexEntrySignal(
                pair=pair, direction='NO_TRADE', score=score,
                size_modifier=0.0, entry_price=ict.current_price,
                stop_price=0, t1=0, t2=0, t3=0, rr_t1=0,
                rationale=rationale,
                macro_conviction=effective_conviction,
                ict_analysis=ict, macro_signal=macro_sig,
            )

        # ICT score determines base size; Buffett conviction tiers cap it
        ict_size = 1.0 if score == 6 else (0.75 if score == 5 else 0.5)
        # Buffett tiers: conviction > 0.85 → 1.5×, 0.70–0.85 → 1×, 0.35–0.70 → 0.75×
        if effective_conviction >= 0.85:
            buffett_mod = 1.5
        elif effective_conviction >= 0.70:
            buffett_mod = 1.0
        else:
            buffett_mod = 0.75
        size_mod = min(ict_size * buffett_mod, 1.5)

        # ── Entry, stop, targets ──────────────────────────────────────── #
        entry, stop = self._entry_stop(target_direction, ict)
        if entry is None:
            return ForexEntrySignal(
                pair=pair, direction='NO_TRADE', score=score,
                size_modifier=0.0, entry_price=ict.current_price,
                stop_price=0, t1=0, t2=0, t3=0, rr_t1=0,
                rationale=rationale + ['No valid entry/stop zone found'],
                macro_conviction=effective_conviction,
                ict_analysis=ict, macro_signal=macro_sig,
            )

        risk = abs(entry - stop)
        if risk == 0:
            return None

        # Validate ATR range
        atr_ratio = risk / ict.atr_daily
        if atr_ratio < self.STOP_ATR_MIN or atr_ratio > self.STOP_ATR_MAX:
            return ForexEntrySignal(
                pair=pair, direction='NO_TRADE', score=score,
                size_modifier=0.0, entry_price=entry,
                stop_price=stop, t1=0, t2=0, t3=0, rr_t1=0,
                rationale=rationale + [f'Stop {atr_ratio:.1f}× ATR — outside 0.5–2.5× range'],
                macro_conviction=effective_conviction,
                ict_analysis=ict, macro_signal=macro_sig,
            )

        if target_direction == 'LONG':
            t1 = entry + 1.5 * risk
            t2 = entry + 3.0 * risk
            t3 = entry + 5.0 * risk
        else:
            t1 = entry - 1.5 * risk
            t2 = entry - 3.0 * risk
            t3 = entry - 5.0 * risk

        rr_t1 = 1.5
        if rr_t1 < self.MIN_RR:
            rationale.append(f'R:R {rr_t1:.1f} below 2:1 minimum — skipping')
            return ForexEntrySignal(
                pair=pair, direction='NO_TRADE', score=score,
                size_modifier=0.0, entry_price=entry,
                stop_price=stop, t1=t1, t2=t2, t3=t3, rr_t1=rr_t1,
                rationale=rationale,
                macro_conviction=effective_conviction,
                ict_analysis=ict, macro_signal=macro_sig,
            )

        return ForexEntrySignal(
            pair=pair,
            direction=target_direction,
            score=score,
            size_modifier=size_mod,
            entry_price=round(entry, 5),
            stop_price=round(stop, 5),
            t1=round(t1, 5),
            t2=round(t2, 5),
            t3=round(t3, 5),
            rr_t1=rr_t1,
            rationale=rationale,
            macro_conviction=effective_conviction,
            ict_analysis=ict,
            macro_signal=macro_sig,
        )

    def scan_all(self) -> list[ForexEntrySignal]:
        from sovereign.forex.pair_universe import ALL_PAIRS
        signals = []
        for pair in ALL_PAIRS:
            try:
                sig = self.evaluate(pair)
                if sig and sig.is_tradeable:
                    signals.append(sig)
            except Exception as e:
                logger.warning(f"evaluate failed for {pair}: {e}")

        signals.sort(key=lambda s: (s.score, s.macro_conviction), reverse=True)
        return signals

    # ── Checklist ─────────────────────────────────────────────────────── #

    def _score_checklist(
        self,
        direction: str,
        macro: ForexSignal,
        ict: ICTAnalysis,
    ) -> Tuple[int, list[str]]:
        score = 0
        rationale = []

        # 1. HTF bias aligned
        ms_trend = ict.market_structure.trend
        bias_ok = (
            (direction == 'LONG' and ms_trend == 'BULLISH') or
            (direction == 'SHORT' and ms_trend == 'BEARISH')
        )
        if bias_ok:
            score += 1
            rationale.append(f'✓ HTF bias: {ms_trend} aligned with macro {direction}')
        else:
            rationale.append(f'✗ HTF bias: market structure {ms_trend} vs macro {direction}')

        # 2. Price at OB or FVG zone (institutional order flow)
        at_zone = False
        zone_desc = ''
        if direction == 'LONG':
            ob = ict.nearest_bullish_ob
            fvg = ict.nearest_bullish_fvg
            price = ict.current_price
            if ob and ob.low <= price <= ob.high:
                at_zone = True
                zone_desc = f'Bullish OB [{ob.low:.5f}–{ob.high:.5f}]'
            elif fvg and fvg.bottom <= price <= fvg.top:
                at_zone = True
                zone_desc = f'Bullish FVG [{fvg.bottom:.5f}–{fvg.top:.5f}]'
        else:
            ob = ict.nearest_bearish_ob
            fvg = ict.nearest_bearish_fvg
            price = ict.current_price
            if ob and ob.low <= price <= ob.high:
                at_zone = True
                zone_desc = f'Bearish OB [{ob.low:.5f}–{ob.high:.5f}]'
            elif fvg and fvg.bottom <= price <= fvg.top:
                at_zone = True
                zone_desc = f'Bearish FVG [{fvg.bottom:.5f}–{fvg.top:.5f}]'

        if at_zone:
            score += 1
            rationale.append(f'✓ Price at institutional zone: {zone_desc}')
        else:
            rationale.append(f'✗ Price not at OB or FVG zone')

        # 3. Liquidity hunted (recent sweep in our direction)
        sweep_ok = False
        for s in ict.recent_sweeps:
            if direction == 'LONG' and s.direction == 'BULLISH_SWEEP':
                sweep_ok = True
                rationale.append(f'✓ Liquidity swept SSL at {s.swept_level:.5f}')
                break
            if direction == 'SHORT' and s.direction == 'BEARISH_SWEEP':
                sweep_ok = True
                rationale.append(f'✓ Liquidity swept BSL at {s.swept_level:.5f}')
                break
        if sweep_ok:
            score += 1
        else:
            rationale.append('✗ No recent liquidity sweep')

        # 4. Market structure shift (CHOCH or BOS)
        ms = ict.market_structure
        mss_ok = False
        if direction == 'LONG' and ms.last_choch == 'BULLISH':
            mss_ok = True
            rationale.append('✓ Bullish CHOCH on entry TF')
        elif direction == 'LONG' and ms.last_bos == 'BULLISH':
            mss_ok = True
            rationale.append('✓ Bullish BOS on entry TF')
        elif direction == 'SHORT' and ms.last_choch == 'BEARISH':
            mss_ok = True
            rationale.append('✓ Bearish CHOCH on entry TF')
        elif direction == 'SHORT' and ms.last_bos == 'BEARISH':
            mss_ok = True
            rationale.append('✓ Bearish BOS on entry TF')
        else:
            rationale.append('✗ No structure shift (CHOCH/BOS) in trade direction')

        if mss_ok:
            score += 1

        # 5. FVG present on entry TF
        fvg_ok = False
        if direction == 'LONG' and ict.nearest_bullish_fvg:
            fvg_ok = True
            f = ict.nearest_bullish_fvg
            rationale.append(f'✓ Bullish FVG present [{f.bottom:.5f}–{f.top:.5f}]')
        elif direction == 'SHORT' and ict.nearest_bearish_fvg:
            fvg_ok = True
            f = ict.nearest_bearish_fvg
            rationale.append(f'✓ Bearish FVG present [{f.bottom:.5f}–{f.top:.5f}]')
        else:
            rationale.append('✗ No FVG on entry timeframe')

        if fvg_ok:
            score += 1

        # 6. Kill Zone
        if ict.in_kill_zone:
            score += 1
            rationale.append(f'✓ In Kill Zone: {ict.kill_zone_name}')
        elif ict.in_ny_lunch:
            rationale.append('✗ NY Lunch — no new entries (hard rule)')
            score = min(score, 3)  # force below threshold during lunch
        else:
            rationale.append('✗ Outside Kill Zone (need +1 confluence to override)')

        return score, rationale

    # ── Entry & Stop ──────────────────────────────────────────────────── #

    def _entry_stop(
        self, direction: str, ict: ICTAnalysis
    ) -> Tuple[Optional[float], Optional[float]]:
        price = ict.current_price
        atr = ict.atr_daily

        if direction == 'LONG':
            # Priority: enter at OB midpoint or FVG midpoint
            ob = ict.nearest_bullish_ob
            fvg = ict.nearest_bullish_fvg

            if ob and ob.low <= price <= ob.high:
                entry = ob.midpoint
                stop = ob.low - 0.1 * atr
            elif fvg and fvg.bottom <= price <= fvg.top:
                entry = fvg.midpoint
                stop = fvg.bottom - 0.1 * atr
            elif ob:
                # Price approaching but not yet in zone — use zone boundary
                entry = ob.high
                stop = ob.low - 0.1 * atr
            else:
                entry = price
                stop = price - atr  # fallback: 1 ATR stop

            return entry, stop

        else:  # SHORT
            ob = ict.nearest_bearish_ob
            fvg = ict.nearest_bearish_fvg

            if ob and ob.low <= price <= ob.high:
                entry = ob.midpoint
                stop = ob.high + 0.1 * atr
            elif fvg and fvg.bottom <= price <= fvg.top:
                entry = fvg.midpoint
                stop = fvg.top + 0.1 * atr
            elif ob:
                entry = ob.low
                stop = ob.high + 0.1 * atr
            else:
                entry = price
                stop = price + atr

            return entry, stop

    # ── Data ──────────────────────────────────────────────────────────── #

    @staticmethod
    def _download(pair: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(pair, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna()
        except Exception as e:
            logger.warning(f"Download failed {pair} {interval}: {e}")
            return None
