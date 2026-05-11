"""
ict/daily_bias.py
=================
Daily directional bias engine for the ICT system.

Combines three inputs:
  1. Alexandrian Library — macro regime + threat level
  2. ForexFactory calendar — today's high-impact events + blackout zones
  3. Trading Agents (optional, async) — analyst debate → narrative bias

Output per pair:
  {
    'pair':       'USDJPY',
    'bias':       'SHORT',        # LONG | SHORT | NEUTRAL
    'confidence': 0.71,           # 0–1
    'reason':     'BOJ rate...',
    'blackout':   False,          # True = no trades today (FOMC day etc.)
    'sources':    ['library', 'calendar'],
  }

Called once per scan. Cached for the session (recalculates at midnight ET).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Pairs the ICT engine trades
ICT_PAIRS = ['USDJPY', 'NZDUSD', 'EURUSD']

# Currency → pairs affected
CURRENCY_PAIRS = {
    'USD': ['USDJPY', 'NZDUSD', 'EURUSD'],
    'JPY': ['USDJPY'],
    'EUR': ['EURUSD'],
    'NZD': ['NZDUSD'],
    'GBP': [],
    'AUD': [],
    'CAD': [],
    'CHF': [],
}

# Events that trigger a full blackout (no ICT trades regardless of grade)
BLACKOUT_EVENTS = {
    'FOMC', 'Fed', 'Federal Reserve', 'Interest Rate Decision',
    'CPI', 'Non-Farm', 'NFP', 'GDP Flash',
}


@dataclass
class PairBias:
    pair:       str
    bias:       str    # LONG | SHORT | NEUTRAL
    confidence: float
    reason:     str
    blackout:   bool
    sources:    List[str]

    def to_dict(self):
        return asdict(self)


class DailyBiasEngine:

    def __init__(self):
        self._cache: Dict[str, dict] = {}   # pair → bias dict
        self._cache_date: Optional[date] = None

    def get_biases(self, library_ctx: Optional[dict] = None) -> Dict[str, dict]:
        """
        Return bias dict for all ICT pairs.
        Uses session-level cache — recalculates once per calendar day.
        """
        today = datetime.now(timezone.utc).date()
        if self._cache_date == today and self._cache:
            return self._cache

        calendar_events = self._fetch_calendar()
        biases = {}
        for pair in ICT_PAIRS:
            bias = self._compute_bias(pair, library_ctx or {}, calendar_events)
            biases[pair] = bias.to_dict()

        self._cache = biases
        self._cache_date = today
        return biases

    # ── Private ────────────────────────────────────────────────────────────── #

    def _fetch_calendar(self) -> List[dict]:
        """Pull today's high-impact events from ForexFactory."""
        try:
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from data.forex_factory_scraper import ForexFactoryScraper
            events = ForexFactoryScraper().fetch_today_events()
            high = [e for e in events if e.get('impact') == 'High']
            logger.info("Calendar: %d high-impact events today", len(high))
            return high
        except Exception as e:
            logger.debug("Calendar fetch failed: %s", e)
            return []

    def _compute_bias(
        self,
        pair: str,
        lib: dict,
        events: List[dict],
    ) -> PairBias:
        sources = []
        bias_votes: List[float] = []   # +1 = LONG, -1 = SHORT, 0 = neutral
        reasons: List[str] = []
        blackout = False

        base, quote = pair[:3], pair[3:]

        # ── 1. Calendar blackout / bias ────────────────────────────────────
        pair_events = [
            e for e in events
            if e.get('currency','') in (base, quote)
        ]
        for ev in pair_events:
            name = ev.get('event','') or ev.get('name','')
            if any(b.lower() in name.lower() for b in BLACKOUT_EVENTS):
                blackout = True
                reasons.append(f"BLACKOUT: {name} ({ev.get('currency','')})")
                break
            # Medium-impact: just add a neutral dampener
            reasons.append(f"Event today: {name}")
        if pair_events:
            sources.append('calendar')

        # ── 2. Alexandrian Library regime → pair directional tilt ─────────
        regime   = lib.get('regime', 'UNKNOWN')
        threat   = lib.get('threat', 'NORMAL')
        modifier = lib.get('size_modifier', 1.0)

        regime_vote = _regime_to_pair_vote(regime, pair)
        if regime_vote != 0:
            bias_votes.append(regime_vote * modifier)
            reasons.append(f"Library: {regime} ({threat}) → {'↑' if regime_vote > 0 else '↓'}")
            sources.append('library')

        # ── 3. Threat-level dampener ───────────────────────────────────────
        if threat in ('DANGER', 'CRITICAL'):
            bias_votes = [v * 0.3 for v in bias_votes]
            reasons.append(f"Threat {threat}: confidence reduced")

        # ── 4. Aggregate ──────────────────────────────────────────────────
        if not bias_votes:
            return PairBias(pair=pair, bias='NEUTRAL', confidence=0.5,
                           reason='No macro signal today', blackout=blackout,
                           sources=sources)

        net = sum(bias_votes) / len(bias_votes)
        bias = 'LONG' if net > 0.1 else 'SHORT' if net < -0.1 else 'NEUTRAL'
        confidence = min(0.95, 0.5 + abs(net) * 0.5)

        return PairBias(
            pair=pair, bias=bias, confidence=round(confidence, 2),
            reason=' | '.join(reasons[:3]),
            blackout=blackout, sources=list(set(sources)),
        )


def _regime_to_pair_vote(regime: str, pair: str) -> float:
    """
    Convert a library regime label to a directional vote for a forex pair.
    Returns +1 (LONG), -1 (SHORT), or 0 (no view).
    """
    r = regime.upper()

    # USD-positive regimes
    if any(x in r for x in ('DOLLAR_WRECKING', 'USD_STRENGTH', 'RISK_OFF', 'FLIGHT_TO_SAFETY',
                             'FED_HIKING', 'RATE_DIFFERENTIAL_USD')):
        if pair == 'USDJPY': return 1.0
        if pair in ('EURUSD', 'NZDUSD'): return -1.0

    # USD-negative / risk-on
    if any(x in r for x in ('DOLLAR_DECLINE', 'RISK_ON', 'MELT_UP', 'EARLY_RECOVERY',
                             'FED_CUTTING', 'FED_PIVOT')):
        if pair == 'USDJPY': return -1.0
        if pair in ('EURUSD', 'NZDUSD'): return 1.0

    # JPY-specific
    if any(x in r for x in ('ASIAN_CURRENCY', 'BOJ', 'JAPAN', 'YEN_CARRY_UNWIND')):
        if pair == 'USDJPY': return -1.0   # JPY strength

    if 'CARRY_TRADE' in r:
        if pair == 'USDJPY': return 1.0    # carry builds USDJPY
        if pair == 'NZDUSD': return 1.0

    # Contagion / crisis = USD + JPY safe haven
    if any(x in r for x in ('CONTAGION', 'CRISIS', 'CREDIT_STRESS', 'REPO_STRESS',
                             'LIQUIDITY_CRISIS')):
        if pair == 'USDJPY': return -0.5   # USD + JPY both safe haven, muted
        if pair in ('EURUSD', 'NZDUSD'): return -1.0

    return 0.0
