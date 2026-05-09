"""
Alexandrian Library — Historical Scenario Matching (V1.0)

Compares the current market feature vector against a catalogue of named
historical regimes.  When the best match exceeds ``SIMILARITY_HIGH`` (0.85),
the top-3 contributing features are logged so a human can evaluate whether
the causal structure actually matches or whether it is a surface-level false
positive.

Design goals (from the problem statement):
  • Transparency: black-box halving of positions is unacceptable.  Every
    high-similarity match must expose *why* it fired.
  • Humility: if the top features are generic (e.g. "VIX mildly elevated")
    rather than crisis-specific, the operator can override and restore normal
    sizing with a logged human decision.
  • Independence from signal logic: the Library does not gate trades.  It
    supplies the HistoricalMatchState dimension for PresentState alignment
    scoring; position-sizing decisions live in the risk engine.

Catalogue scenarios are defined as unit-normalised feature vectors.  Cosine
similarity is used because it is scale-invariant (we do not want a scenario
to match merely because one feature is large).  Feature weights are applied
before the dot product to reflect which features have the most explanatory
power for each scenario type.

Adding a new scenario: add an entry to ``SCENARIO_CATALOGUE`` below.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SIMILARITY_HIGH = 0.85   # above this → log top features + flag for operator review
SIMILARITY_LOW  = 0.50   # below this → treat as no match (source='none')

# ---------------------------------------------------------------------------
# Feature key constants
# ---------------------------------------------------------------------------
# Keys in the feature dict passed to ``match()``.  Callers must use the same
# names.  All values are expected to be floats in a roughly [-3, +3] range
# (z-scores or normalised ratios); raw values are normalised internally.

F_VIX           = 'vix_zscore'        # VIX vs trailing 1-yr z-score
F_VIX_TERM      = 'vix_term_struct'   # +1 contango / -1 backwardation
F_DXY           = 'dxy_zscore'        # DXY vs 252-day z-score
F_SPY_VS_200    = 'spy_vs_200sma'     # (price - 200SMA) / 200SMA, fraction
F_YIELD_CURVE   = 'yield_curve_slope' # 10Y-2Y spread, %
F_HYG_SPREAD    = 'hyg_spread_zscore' # HYG spread vs historical z-score
F_COT_SPEC      = 'cot_spec_zscore'   # speculative positioning z-score
F_ATR_PCT       = 'atr_pct_252d'      # ATR percentile vs 252-day window
F_CARRY_CROWD   = 'carry_crowd'       # COT crowding in carry pairs (0-1)

ALL_FEATURES: List[str] = [
    F_VIX, F_VIX_TERM, F_DXY, F_SPY_VS_200,
    F_YIELD_CURVE, F_HYG_SPREAD, F_COT_SPEC, F_ATR_PCT, F_CARRY_CROWD,
]


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A named historical regime and its representative feature fingerprint."""
    label: str
    description: str
    # Feature vector — only features relevant to this scenario need be set;
    # unset features default to 0.0 (neutral).
    features: Dict[str, float] = field(default_factory=dict)
    # Weight for each feature: higher = more important for this scenario.
    # Missing weights default to 1.0.
    weights: Dict[str, float] = field(default_factory=dict)
    # Typical market outcome observed in this regime.
    typical_outcome: str = ''
    # How many days the typical resolution window spans.
    lookback_period_days: int = 90


def _s(label: str, description: str, features: Dict[str, float],
        weights: Dict[str, float], outcome: str, days: int) -> Scenario:
    return Scenario(label=label, description=description, features=features,
                    weights=weights, typical_outcome=outcome,
                    lookback_period_days=days)


SCENARIO_CATALOGUE: List[Scenario] = [
    _s(
        label='ASIAN_CURRENCY_CONTAGION',
        description='1997-style emerging market currency crisis: '
                    'high VIX, dollar surge, carry unwind, credit spread blow-out.',
        features={
            F_VIX: +2.5, F_VIX_TERM: -1.0, F_DXY: +2.0,
            F_SPY_VS_200: -0.05, F_HYG_SPREAD: +2.0, F_COT_SPEC: -1.5,
            F_CARRY_CROWD: -1.0,
        },
        weights={
            F_VIX: 1.5, F_DXY: 1.5, F_HYG_SPREAD: 1.5, F_CARRY_CROWD: 1.2,
        },
        outcome='Broad EM sell-off; AUD/NZD down 15-30%; JPY/CHF rally. '
                'Typically resolves in 3-6 months with IMF intervention.',
        days=180,
    ),
    _s(
        label='GFC_2008',
        description='2008 Global Financial Crisis: VIX extreme, yield curve '
                    'inverted, credit spreads explosive, equity well below 200SMA.',
        features={
            F_VIX: +3.0, F_VIX_TERM: -1.0, F_DXY: +1.5,
            F_SPY_VS_200: -0.25, F_YIELD_CURVE: -0.5, F_HYG_SPREAD: +3.0,
            F_COT_SPEC: -2.0, F_CARRY_CROWD: -2.0,
        },
        weights={
            F_VIX: 2.0, F_HYG_SPREAD: 2.0, F_SPY_VS_200: 1.5,
            F_CARRY_CROWD: 1.5,
        },
        outcome='Risk asset collapse; carry completely unwound; JPY +20%, '
                'AUD -35%. Resolution took 18 months.',
        days=360,
    ),
    _s(
        label='SNB_FLOOR_REMOVAL_2015',
        description='2015 SNB CHF floor removal: sudden massive CHF spike, '
                    'low prior volatility (ATR compressed), no macro warning.',
        features={
            F_VIX: +0.5, F_VIX_TERM: +0.5, F_DXY: +0.8,
            F_SPY_VS_200: -0.02, F_ATR_PCT: -1.5,
        },
        weights={F_ATR_PCT: 2.0, F_DXY: 1.2},
        outcome='CHF +15% intraday; extreme tail risk. Low VIX pre-event '
                'is the signature — do NOT be short CHF when ATR is compressed.',
        days=5,
    ),
    _s(
        label='JPY_CARRY_UNWIND_2022',
        description='2022 yen carry unwind: extreme carry crowding, rising '
                    'VIX, DXY surge, BOJ divergence.',
        features={
            F_VIX: +1.5, F_VIX_TERM: -0.5, F_DXY: +2.5,
            F_SPY_VS_200: -0.15, F_YIELD_CURVE: -0.3,
            F_COT_SPEC: +2.0,   # specs were very long JPY carry
            F_CARRY_CROWD: +2.0,
        },
        weights={
            F_DXY: 1.5, F_CARRY_CROWD: 2.0, F_COT_SPEC: 1.5,
        },
        outcome='USD/JPY +30% in 12 months; BOJ eventually intervened. '
                'Carry crowding + DXY surge is the identifying signature.',
        days=252,
    ),
    _s(
        label='COVID_CRASH_2020',
        description='2020 COVID: sudden VIX spike from low base, dollar surge, '
                    'equity below 200SMA, credit blow-out.',
        features={
            F_VIX: +3.0, F_VIX_TERM: -1.0, F_DXY: +2.0,
            F_SPY_VS_200: -0.30, F_HYG_SPREAD: +2.5, F_CARRY_CROWD: -1.5,
        },
        weights={F_VIX: 2.0, F_SPY_VS_200: 1.5, F_HYG_SPREAD: 1.5},
        outcome='All risk assets -35%; recovery in 5 months. '
                'Unusually fast V-shape driven by fiscal/monetary response.',
        days=120,
    ),
    _s(
        label='RISK_ON_EXPANSION',
        description='Normal bull market: VIX low, yield curve positive, '
                    'equity above 200SMA, carry crowding building.',
        features={
            F_VIX: -1.0, F_VIX_TERM: +1.0, F_DXY: -0.5,
            F_SPY_VS_200: +0.08, F_YIELD_CURVE: +1.5,
            F_HYG_SPREAD: -1.0, F_COT_SPEC: +1.0, F_CARRY_CROWD: +1.0,
        },
        weights={F_VIX: 1.2, F_YIELD_CURVE: 1.2},
        outcome='Carry works; AUD/NZD bid; momentum strategies outperform.',
        days=252,
    ),
    _s(
        label='LATE_CYCLE_COMPRESSION',
        description='Pre-stress period: ATR compressed, VIX subdued but carry '
                    'crowding near 85th pct, yield curve flattening.',
        features={
            F_VIX: -0.5, F_VIX_TERM: +0.3, F_DXY: +0.3,
            F_SPY_VS_200: +0.03, F_YIELD_CURVE: +0.2,
            F_ATR_PCT: -1.5, F_CARRY_CROWD: +1.8,
        },
        weights={F_ATR_PCT: 2.0, F_CARRY_CROWD: 2.0},
        outcome='Geological stress: surface looks calm; spike risk building. '
                'Surgeon scales up before detonation.',
        days=60,
    ),
]


# ---------------------------------------------------------------------------
# Library engine
# ---------------------------------------------------------------------------

@dataclass
class LibraryMatch:
    """Result returned by AlexandrianLibrary.match()."""
    scenario: Scenario
    similarity: float              # 0..1 cosine similarity after weighting
    top_features: List[Tuple[str, float, float]]  # (feature, current_val, scenario_val)
    is_high_similarity: bool       # similarity >= SIMILARITY_HIGH
    volumes_converging: int        # features within 0.5 σ of scenario value


class AlexandrianLibrary:
    """
    Matches a current feature snapshot against the scenario catalogue and
    returns the closest historical analogue with full feature transparency.

    Usage::

        lib = AlexandrianLibrary()
        match = lib.match(features)
        if match:
            state = lib.to_historical_match_state(match)
    """

    def __init__(self, catalogue: Optional[List[Scenario]] = None):
        self._catalogue = catalogue or SCENARIO_CATALOGUE

    # ── Public API ──────────────────────────────────────────────────────── #

    def match(self, features: Dict[str, float]) -> Optional[LibraryMatch]:
        """
        Find the closest scenario in the catalogue.

        Returns ``None`` when the best similarity is below ``SIMILARITY_LOW``
        (no meaningful analogue found).

        When similarity exceeds ``SIMILARITY_HIGH``, the match is flagged and
        the top-3 contributing features are logged at INFO level so the operator
        can evaluate whether the structural match is genuine.
        """
        best_sim = -1.0
        best_match: Optional[LibraryMatch] = None

        for scenario in self._catalogue:
            sim, top_feats, converging = self._weighted_cosine(features, scenario)
            if sim > best_sim:
                best_sim = sim
                best_match = LibraryMatch(
                    scenario=scenario,
                    similarity=round(sim, 4),
                    top_features=top_feats,
                    is_high_similarity=(sim >= SIMILARITY_HIGH),
                    volumes_converging=converging,
                )

        if best_match is None or best_match.similarity < SIMILARITY_LOW:
            return None

        if best_match.is_high_similarity:
            self._log_high_similarity(best_match, features)

        return best_match

    def to_historical_match_state(self, match: Optional['LibraryMatch']):
        """
        Convert a LibraryMatch to the HistoricalMatchState dataclass used by
        PresentStateBuilder.  Returns a neutral stub when match is None.
        """
        # Import here to avoid circular imports (contracts → sovereign → contracts).
        from contracts.types import HistoricalMatchState

        if match is None:
            return HistoricalMatchState(
                regime_label='UNKNOWN',
                similarity_score=0.0,
                volumes_converging=0,
                typical_outcome='No historical match above similarity threshold.',
                lookback_period_days=0,
                source='none',
            )

        return HistoricalMatchState(
            regime_label=match.scenario.label,
            similarity_score=match.similarity,
            volumes_converging=match.volumes_converging,
            typical_outcome=match.scenario.typical_outcome,
            lookback_period_days=match.scenario.lookback_period_days,
            source='AlexandrianLibrary',
        )

    # ── Internal ────────────────────────────────────────────────────────── #

    @staticmethod
    def _weighted_cosine(
        current: Dict[str, float],
        scenario: Scenario,
    ) -> Tuple[float, List[Tuple[str, float, float]], int]:
        """
        Compute weighted cosine similarity between *current* and *scenario*.

        Returns (similarity, top_3_features, volumes_converging).

        ``volumes_converging`` counts how many features are within 0.5 of
        the scenario's value (i.e. they are "pointing the same way").
        """
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        contributions: List[Tuple[float, str, float, float]] = []
        converging = 0

        for feat in ALL_FEATURES:
            a = current.get(feat, 0.0)
            b = scenario.features.get(feat, 0.0)
            w = scenario.weights.get(feat, 1.0)

            wa = a * w
            wb = b * w
            c = wa * wb
            dot += c
            norm_a += wa * wa
            norm_b += wb * wb

            if abs(c) > 1e-9:
                contributions.append((abs(c), feat, a, b))

            if b != 0.0 and abs(a - b) <= 0.5:
                converging += 1

        denom = math.sqrt(norm_a * norm_b)
        sim = dot / denom if denom > 1e-9 else 0.0
        sim = max(0.0, min(1.0, sim))  # clamp to [0, 1]

        # Sort by contribution magnitude, return top 3
        contributions.sort(key=lambda x: x[0], reverse=True)
        top_3 = [(feat, cur, scen) for _, feat, cur, scen in contributions[:3]]

        return sim, top_3, converging

    @staticmethod
    def _log_high_similarity(match: 'LibraryMatch', current: Dict[str, float]) -> None:
        """
        Emit a structured INFO log when similarity exceeds the high threshold.

        This is the transparency layer called out in the problem statement:
        "when similarity > 0.85, print the top 3 features driving the match
        so you can evaluate whether the causal structure actually matches, or
        whether it's a false positive on surface features."
        """
        lines = [
            f'[AlexandrianLibrary] HIGH SIMILARITY ALERT — {match.scenario.label}',
            f'  Similarity : {match.similarity:.3f} (threshold={SIMILARITY_HIGH})',
            f'  Converging : {match.volumes_converging}/{len(ALL_FEATURES)} features',
            f'  Description: {match.scenario.description}',
            '  Top-3 driving features:',
        ]
        for rank, (feat, cur_val, scen_val) in enumerate(match.top_features, 1):
            direction = '↑ matches' if (cur_val * scen_val) > 0 else '↓ diverges'
            lines.append(
                f'    {rank}. {feat:<22}  current={cur_val:+.3f}  '
                f'scenario={scen_val:+.3f}  [{direction}]'
            )
        lines.append(
            '  ACTION: review whether the causal structure matches, '
            'not just the surface features.  Override sizing if superficial.'
        )
        logger.info('\n'.join(lines))
