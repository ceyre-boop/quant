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

CB_EVENT_TRIGGER (Edge 3 — Post-Decision Drift):
  Source: data/cache/cb_decisions.json (built by scripts/build_cb_decisions.py)
  Fires when: surprise_bps >= 25 AND within entry window (day +1 to +5)
  Conviction: 0.50 + abs(surprise_bps) / 200  [capped at 0.90]
  Hold: 10–20 days
  Adds ~6-8 signals per pair per year historically.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sovereign.forex.macro_engine import ForexMacroEngine, ForexSignal
from sovereign.forex.ict_engine import ICTEngine, ICTAnalysis
from sovereign.forex.commodity_engine import CommodityEngine, COMMODITY_PAIRS
from sovereign.forex.cot_engine import COTEngine
from sovereign.forex.dxy_engine import DXYEngine
from sovereign.forex.strategy import (
    CONVICTION_NEUTRAL_THRESHOLD,
    CONVICTION_FULL_SIZE,
    CONVICTION_MAX_SIZE,
    TARGET_RR_T1,
    TARGET_RR_T2,
    TARGET_RR_T3,
)

CB_DECISIONS_PATH = Path(__file__).parents[2] / 'data' / 'cache' / 'cb_decisions.json'
CB_MIN_SURPRISE_BPS = 25
CB_ENTRY_WINDOW_DAYS = 5   # entry valid for 5 business days after decision

logger = logging.getLogger(__name__)


class CBEventTrigger:
    """
    Edge 3 — Post-Decision Drift.

    Loads cb_decisions.json and answers: given a pair and a date,
    was there a central bank surprise in the last N days that generates
    a directional signal for this pair?

    Bank → country mapping determines which decisions affect which pairs.
    """

    # How each bank's surprise direction translates to pair direction:
    # surprise > 0 (hawkish) → base currency LONG; surprise < 0 (dovish) → SHORT
    BANK_TO_COUNTRY = {
        'FED': 'US', 'ECB': 'EU', 'BOE': 'UK',
        'BOJ': 'JP', 'SNB': 'CH', 'RBA': 'AU',
        'BOC': 'CA', 'RBNZ': 'NZ',
    }

    def __init__(self):
        self._decisions: list[dict] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            with open(CB_DECISIONS_PATH) as f:
                self._decisions = json.load(f)
            self._loaded = True
        except FileNotFoundError:
            logger.warning(
                f"cb_decisions.json not found at {CB_DECISIONS_PATH}. "
                "Run scripts/build_cb_decisions.py to generate it."
            )
            self._loaded = True  # don't retry

    def check(
        self,
        base_country: str,
        quote_country: str,
        as_of: pd.Timestamp,
        window_days: int = CB_ENTRY_WINDOW_DAYS,
    ) -> Optional[dict]:
        """
        Returns the strongest active CB surprise signal for this pair,
        or None if no qualifying event in the window.

        Returns:
          {'direction': 'LONG'|'SHORT', 'conviction': float,
           'surprise_bps': int, 'bank': str, 'date': str, 'hold_days': int}
        """
        self._load()
        if not self._decisions:
            return None

        window_start = as_of - pd.Timedelta(days=window_days + 4)  # +4 for weekends

        candidates = []
        for d in self._decisions:
            d_date = pd.Timestamp(d['date'])
            if not (window_start <= d_date <= as_of):
                continue
            if abs(d['surprise_bps']) < CB_MIN_SURPRISE_BPS:
                continue

            country = self.BANK_TO_COUNTRY.get(d['bank'])
            if country not in (base_country, quote_country):
                continue

            # Surprise direction: hawkish (+) strengthens the currency
            surprise = d['surprise_bps']
            if country == base_country:
                raw_direction = 'LONG' if surprise > 0 else 'SHORT'
            else:
                # Quote currency hawkish → pair goes DOWN (base weakens relatively)
                raw_direction = 'SHORT' if surprise > 0 else 'LONG'

            conviction = min(0.50 + abs(float(surprise)) / 200.0, 0.90)
            hold_days = int(10 + min(abs(float(surprise)) / 25.0, 2) * 5)  # 10–20 days

            candidates.append({
                'direction':    raw_direction,
                'conviction':   conviction,
                'surprise_bps': surprise,
                'bank':         d['bank'],
                'date':         d['date'],
                'hold_days':    hold_days,
            })

        if not candidates:
            return None

        # Return the highest-conviction (largest surprise) event
        return max(candidates, key=lambda c: abs(c['surprise_bps']))

    def check_historical(
        self,
        base_country: str,
        quote_country: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        min_surprise_bps: int = CB_MIN_SURPRISE_BPS,
    ) -> list[dict]:
        """
        Return all CB surprise events in [start, end] for this country pair.
        Used by the backtester to generate historical entry signals.
        Each event has 5-day entry window embedded as 'entry_start'/'entry_end'.
        """
        self._load()
        if not self._decisions:
            return []

        results = []
        for d in self._decisions:
            d_date = pd.Timestamp(d['date'])
            if not (start <= d_date <= end):
                continue
            if abs(d['surprise_bps']) < min_surprise_bps:
                continue
            country = self.BANK_TO_COUNTRY.get(d['bank'])
            if country not in (base_country, quote_country):
                continue

            surprise = d['surprise_bps']
            if country == base_country:
                direction = 'LONG' if surprise > 0 else 'SHORT'
            else:
                direction = 'SHORT' if surprise > 0 else 'LONG'

            conviction = min(0.50 + abs(float(surprise)) / 200.0, 0.90)
            hold_days = int(10 + min(abs(float(surprise)) / 25.0, 2) * 5)

            results.append({
                'decision_date': d_date,
                'entry_start':   d_date + pd.Timedelta(days=1),
                'entry_end':     d_date + pd.Timedelta(days=CB_ENTRY_WINDOW_DAYS + 2),
                'direction':     direction,
                'conviction':    conviction,
                'surprise_bps':  surprise,
                'bank':          d['bank'],
                'hold_days':     hold_days,
            })

        return results


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
    MIN_RR = TARGET_RR_T1
    STOP_ATR_MIN = 0.5
    STOP_ATR_MAX = 2.5
    MAX_CONCURRENT = 2

    def __init__(self):
        self._macro = ForexMacroEngine()
        self._ict = ICTEngine()
        self._commodity = CommodityEngine()
        self._cb = CBEventTrigger()
        self._cot = COTEngine()
        self._dxy = DXYEngine()
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

        # ── CB event trigger (Edge 3 — Post-Decision Drift) ─────────────── #
        # Check if a CB surprise event is active for this pair RIGHT NOW.
        # A CB surprise overrides the Buffett threshold — it's a direct causal event.
        cfg = __import__('sovereign.forex.pair_universe', fromlist=['PAIR_CONFIG']).PAIR_CONFIG
        from sovereign.forex.pair_universe import PAIR_CONFIG, CB_TO_COUNTRY
        pair_cfg = PAIR_CONFIG.get(pair)
        base_country = CB_TO_COUNTRY[pair_cfg.base_central_bank] if pair_cfg else None
        quote_country = CB_TO_COUNTRY[pair_cfg.quote_central_bank] if pair_cfg else None

        cb_event = None
        if base_country and quote_country:
            cb_event = self._cb.check(
                base_country, quote_country, as_of=pd.Timestamp.utcnow()
            )

        # ── Buffett conviction gate ───────────────────────────────────── #
        effective_conviction = macro_sig.conviction

        # CB event boost — takes precedence over everything
        if cb_event:
            # Post-decision drift: direction follows the surprise
            # Only enter if it aligns with macro or overrides on large surprise
            if (cb_event['direction'] == macro_sig.direction or
                    abs(cb_event['surprise_bps']) >= 50):
                effective_conviction = max(effective_conviction, cb_event['conviction'])

        # Commodity boost for AUD/CAD/NZD pairs
        if pair in COMMODITY_PAIRS:
            try:
                comm_sig = self._commodity.score_pair(pair)
                if (comm_sig and comm_sig.direction == macro_sig.direction
                        and comm_sig.lag_detected):
                    boost = comm_sig.conviction * 0.15
                    effective_conviction = min(effective_conviction + boost, 1.0)
            except Exception:
                pass

        if effective_conviction < CONVICTION_NEUTRAL_THRESHOLD and cb_event is None:
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

        # CB event can override macro direction if surprise is large enough
        target_direction = macro_sig.direction
        if cb_event and abs(cb_event['surprise_bps']) >= 50:
            target_direction = cb_event['direction']

        # ── ICT analysis ──────────────────────────────────────────────── #
        try:
            ict = self._ict.analyse(pair, entry_df)
        except ValueError as e:
            logger.warning(f"ICT analysis failed for {pair}: {e}")
            return None

        # ── 6-point checklist ─────────────────────────────────────────── #
        score, rationale = self._score_checklist(target_direction, macro_sig, ict)

        if cb_event:
            rationale.append(
                f'★ CB Event: {cb_event["bank"]} {cb_event["surprise_bps"]:+d}bp surprise '
                f'on {cb_event["date"]} → {cb_event["direction"]} '
                f'(conv={cb_event["conviction"]:.2f}, hold={cb_event["hold_days"]}d)'
            )
            # CB event counts as a bonus criterion — lowers score requirement to 3
            if score >= 3 and cb_event['direction'] == target_direction:
                score = max(score, 4)  # floor to tradeable threshold

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
        if effective_conviction >= CONVICTION_MAX_SIZE:
            buffett_mod = 1.5
        elif effective_conviction >= CONVICTION_FULL_SIZE:
            buffett_mod = 1.0
        else:
            buffett_mod = 0.75
        size_mod = min(ict_size * buffett_mod, 1.5)

        # ── COT positioning gate (Dalio Q3) ───────────────────────────── #
        cot_mult = self._cot.gate_signal(target_direction, base_country or '')
        if cot_mult < 1.0:
            size_mod = round(size_mod * cot_mult, 3)
            rationale.append(
                f'COT: speculators crowded {target_direction} on '
                f'{base_country} — size ×{cot_mult}'
            )

        # ── DXY smile overlay ─────────────────────────────────────────── #
        try:
            dxy_mult = self._dxy.get_modifier(pair, target_direction)
            if dxy_mult != 1.0:
                size_mod = round(min(size_mod * dxy_mult, 1.5), 3)
                td = self._dxy.get_trend()
                rationale.append(
                    f'DXY: {td["trend"]} ({td["smile_regime"]}) — size ×{dxy_mult}'
                )
        except Exception:
            pass

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
            t1 = entry + TARGET_RR_T1 * risk
            t2 = entry + TARGET_RR_T2 * risk
            t3 = entry + TARGET_RR_T3 * risk
        else:
            t1 = entry - TARGET_RR_T1 * risk
            t2 = entry - TARGET_RR_T2 * risk
            t3 = entry - TARGET_RR_T3 * risk

        rr_t1 = TARGET_RR_T1
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
