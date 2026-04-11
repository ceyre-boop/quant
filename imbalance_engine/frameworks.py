"""Six mathematical frameworks for detecting structural imbalances 
weeks to months before price moves.

All frameworks are stateless computations. Feed them data, get a score.
The orchestrator (MacroImbalanceFramework) aggregates into a composite 
stress reading and emits a RegimeStressOutput.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import date, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Output Contracts
# =============================================================================

@dataclass
class ERPOutput:
    """Equity Risk Premium — overvaluation / undervaluation meter."""
    earnings_yield: float          # 1 / CAPE
    bond_yield_10yr: float
    erp: float                     # earnings_yield - bond_yield = ERP
    historical_avg_erp: float      # long-run avg (historically ~3.5%)
    z_score: float                 # std devs above/below historical avg
    signal: str                    # 'OVERVALUED' | 'FAIR' | 'UNDERVALUED'
    magnitude: float               # 0-10 severity


@dataclass
class CAPEOutput:
    """Shiller CAPE with 130-year z-score normalization."""
    cape_ratio: float
    historical_mean: float         # ~16.8 since 1881
    historical_std: float
    z_score: float                 # how many std devs from mean
    percentile: float              # 0-100 among all historical readings
    signal: str                    # 'EXTREME_OVERVALUED' | 'OVERVALUED' | 'FAIR' | 'CHEAP'
    implied_10yr_return: float     # regression-based forward return estimate


@dataclass
class PCARegimeOutput:
    """PCA on macro factors — stress before price moves."""
    pc1_score: float               # primary market stress component
    pc2_score: float               # secondary (yield/credit) component
    mahalanobis_distance: float    # how far from normal regime
    stress_regime: str             # 'NORMAL' | 'STRESS' | 'CRISIS'
    top_contributors: List[str]    # features driving the anomaly


@dataclass
class HMMOutput:
    """Hidden Markov Model — regime transition probability."""
    current_regime: int            # 0=Bull, 1=Bear, 2=Crash
    regime_label: str
    transition_probs: Dict[str, float]   # next-period probability per regime
    expected_duration_days: float        # expected days in current regime
    hmm_stress_score: float              # 0-1, higher = more likely to crack


@dataclass
class RecessionProbOutput:
    """NY Fed probit model — yield curve recession probability."""
    spread_3m_10yr: float          # 3-month / 10-year treasury spread (bps)
    spread_velocity: float          # rate of change over 3 months
    prob_12m: float                # probability of recession in 12 months (0-1)
    prob_signal: str               # 'GREEN' | 'YELLOW' | 'RED'
    historical_accuracy: str       # "Predicted every US recession since 1960"


@dataclass
class YieldCurveOutput:
    """Yield curve — velocity and acceleration, not just level."""
    spread_2_10: float             # 2yr-10yr spread
    spread_3m_10yr: float          # 3m-10yr spread
    velocity_30d: float            # rate of change over 30 days (bps/day)
    acceleration_30d: float        # second derivative (is steepening accelerating?)
    inversion_depth: float         # how inverted (negative = inverted)
    inversion_duration_days: int   # how long inverted
    signal: str                    # 'BULL_STEEP' | 'FLAT' | 'INVERTED' | 'CRASH_INVERSION'


@dataclass
class RegimeStressOutput:
    """Composite output from all six frameworks."""
    timestamp: str
    symbol: str                    # usually 'SPY' or index-level
    
    erp: ERPOutput
    cape: CAPEOutput
    pca: PCARegimeOutput
    hmm: HMMOutput
    recession_prob: RecessionProbOutput
    yield_curve: YieldCurveOutput
    
    # Aggregated signals
    composite_stress_score: float  # 0-10, weighted average across frameworks
    consensus_fault_detected: bool # True if >= 4/6 frameworks flag stress
    fault_summary: str             # Human-readable summary for Kimi prompt
    
    # XGBoost feature — emitted into FeatureVector
    hmm_regime_stress: float       # same as hmm.hmm_stress_score, explicit alias


# =============================================================================
# Framework 1: Equity Risk Premium
# =============================================================================

class ERPFramework:
    """Earnings yield vs 10-year bond yield.
    
    When ERP compresses to near zero or negative, equities price in 
    perfection — mathematically. No narrative needed.
    """
    
    HISTORICAL_AVG_ERP = 3.5       # %, long-run average since 1900
    HISTORICAL_STD_ERP = 2.1       # %, empirical standard deviation
    
    def compute(self, cape_ratio: float, bond_yield_10yr_pct: float) -> ERPOutput:
        earnings_yield = (1.0 / cape_ratio) * 100  # convert to %
        erp = earnings_yield - bond_yield_10yr_pct
        
        z = (erp - self.HISTORICAL_AVG_ERP) / self.HISTORICAL_STD_ERP
        # Negative z = ERP below historical avg = more overvalued than usual
        
        if erp < 0.5:
            signal = 'EXTREME_OVERVALUED'
            magnitude = min(10.0, 10 + z)  # z is negative, so magnitude > 10 capped
        elif erp < 2.0:
            signal = 'OVERVALUED'
            magnitude = max(5.0, 7 + z)
        elif erp > 5.0:
            signal = 'UNDERVALUED'
            magnitude = 2.0
        else:
            signal = 'FAIR'
            magnitude = 3.0
        
        magnitude = float(np.clip(magnitude, 0.0, 10.0))
        
        return ERPOutput(
            earnings_yield=round(earnings_yield, 3),
            bond_yield_10yr=bond_yield_10yr_pct,
            erp=round(erp, 3),
            historical_avg_erp=self.HISTORICAL_AVG_ERP,
            z_score=round(z, 3),
            signal=signal,
            magnitude=magnitude
        )


# =============================================================================
# Framework 2: Shiller CAPE
# =============================================================================

class CAPEFramework:
    """Shiller CAPE with 130-year normalization.
    
    CAPE alone is not a timing signal. The z-score relative to the 
    long historical distribution is the signal.
    """
    
    # Historical parameters (1881-2024 monthly data)
    HISTORICAL_MEAN = 16.8
    HISTORICAL_STD = 6.7
    
    # Regression: 10yr forward return ≈ 0.6/CAPE - 0.01 (rough Shiller regression)
    def compute(self, cape: float) -> CAPEOutput:
        z = (cape - self.HISTORICAL_MEAN) / self.HISTORICAL_STD
        
        # Percentile approximation from z-score (normal CDF)
        pct = float(0.5 * (1 + math.erf(z / math.sqrt(2)))) * 100
        
        # Signal tiers
        if z > 2.5:
            signal = 'EXTREME_OVERVALUED'
        elif z > 1.5:
            signal = 'OVERVALUED'
        elif z < -1.0:
            signal = 'CHEAP'
        else:
            signal = 'FAIR'
        
        # Implied 10yr real return (Shiller regression approximation)
        implied_return = (1.0 / cape) * 100 - 1.5  # rough: earnings yield - inflation drag
        
        return CAPEOutput(
            cape_ratio=round(cape, 2),
            historical_mean=self.HISTORICAL_MEAN,
            historical_std=self.HISTORICAL_STD,
            z_score=round(z, 3),
            percentile=round(pct, 1),
            signal=signal,
            implied_10yr_return=round(implied_return, 2)
        )


# =============================================================================
# Framework 3: PCA Macro Regime
# =============================================================================

class PCAFramework:
    """Principal Component Analysis on macro factor matrix.
    
    Inputs: credit spread, VIX, yield curve, USD index, commodity index,
            market breadth, CAPE z-score, ERP z-score.
    
    Mahalanobis distance flags when the multivariate state is far from 
    its historical distribution — before individual prices reprice.
    """
    
    # Historical covariance of macro factors (approximate empirical values)
    # In production: compute from rolling 252-day window
    FACTOR_NAMES = [
        'vix', 'credit_spread_hy', 'yield_curve_2_10', 
        'dxy', 'erp_z', 'cape_z', 'breadth'
    ]
    
    def compute(self, factor_vector: Dict[str, float]) -> PCARegimeOutput:
        """
        Args:
            factor_vector: dict with keys matching FACTOR_NAMES
                           Values should be z-scored relative to trailing 252d
        """
        vals = np.array([factor_vector.get(k, 0.0) for k in self.FACTOR_NAMES])
        
        # Mahalanobis distance (simplified — treat as identity cov for now)
        # Production: use fitted covariance matrix from historical period
        mahal = float(np.sqrt(np.sum(vals ** 2)))
        
        # PC1: dominant stress component (VIX + credit carry most weight)
        vix_z = factor_vector.get('vix', 0)
        credit_z = factor_vector.get('credit_spread_hy', 0)
        cape_z = factor_vector.get('cape_z', 0)
        
        pc1 = 0.40 * vix_z + 0.35 * credit_z + 0.25 * cape_z
        pc2 = 0.50 * factor_vector.get('yield_curve_2_10', 0) + \
              0.50 * factor_vector.get('erp_z', 0)
        
        # Regime classification
        if mahal > 4.0:
            regime = 'CRISIS'
        elif mahal > 2.5:
            regime = 'STRESS'
        else:
            regime = 'NORMAL'
        
        # Top contributors by absolute z-score
        contributions = {k: abs(factor_vector.get(k, 0)) for k in self.FACTOR_NAMES}
        top = sorted(contributions, key=contributions.get, reverse=True)[:3]
        
        return PCARegimeOutput(
            pc1_score=round(pc1, 3),
            pc2_score=round(pc2, 3),
            mahalanobis_distance=round(mahal, 3),
            stress_regime=regime,
            top_contributors=top
        )


# =============================================================================
# Framework 4: Hidden Markov Model (simplified 3-state)
# =============================================================================

class HMMFramework:
    """3-state Hidden Markov Model: Bull / Bear / Crash.
    
    State emission probabilities are estimated from VIX level + 
    momentum signature. Transition matrix is empirically fitted to 
    S&P 500 monthly data 1950-2023.
    
    The hmm_stress_score is emitted as a feature to XGBoost.
    """
    
    # Empirical transition matrix P[from][to]:
    # States: 0=Bull, 1=Bear, 2=Crash
    TRANSITION = np.array([
        [0.94, 0.05, 0.01],   # From Bull
        [0.15, 0.75, 0.10],   # From Bear
        [0.10, 0.40, 0.50],   # From Crash
    ])
    LABELS = {0: 'BULL', 1: 'BEAR', 2: 'CRASH'}
    EXPECTED_DURATIONS = {0: 730, 1: 240, 2: 90}  # avg calendar days
    
    def _infer_state(self, vix: float, momentum_1m: float, breadth: float) -> int:
        """Simple heuristic state assignment (production: Viterbi on full sequence)."""
        if vix > 35 or momentum_1m < -0.12:
            return 2  # Crash
        elif vix > 22 or momentum_1m < -0.04:
            return 1  # Bear
        else:
            return 0  # Bull
    
    def compute(self, vix: float, spy_return_1m: float, breadth: float) -> HMMOutput:
        state = self._infer_state(vix, spy_return_1m, breadth)
        
        trans = self.TRANSITION[state]
        transition_probs = {
            'BULL': round(float(trans[0]), 3),
            'BEAR': round(float(trans[1]), 3),
            'CRASH': round(float(trans[2]), 3),
        }
        
        # Stress score: P(Bear) + 2*P(Crash)
        hmm_stress = float(np.clip(trans[1] + 2 * trans[2], 0, 1))
        
        return HMMOutput(
            current_regime=state,
            regime_label=self.LABELS[state],
            transition_probs=transition_probs,
            expected_duration_days=float(self.EXPECTED_DURATIONS[state]),
            hmm_stress_score=round(hmm_stress, 4)
        )


# =============================================================================
# Framework 5: NY Fed Probit Recession Model
# =============================================================================

class RecessionProbFramework:
    """NY Fed yield curve probit model.
    
    Original paper: Estrella & Mishkin (1996).
    Uses 3-month / 10-year spread. Predicted every US recession since 1960.
    
    Probit: P(recession in 12m) = Φ(α + β * spread)
    Coefficients from Estrella et al. (α=1.54, β=-0.82)
    """
    
    ALPHA = 1.54
    BETA = -0.82
    
    def compute(self, spread_3m_10yr_bps: float, spread_velocity_bps_per_month: float) -> RecessionProbOutput:
        """
        Args:
            spread_3m_10yr_bps: 3m-10yr spread in basis points (negative = inverted)
            spread_velocity_bps_per_month: rate of spread change (bps/month)
        """
        spread_pct = spread_3m_10yr_bps / 100.0
        
        # Probit: Φ(α + β*spread)
        z = self.ALPHA + self.BETA * spread_pct
        prob = float(0.5 * (1 + math.erf(z / math.sqrt(2))))
        
        # Velocity amplifies the signal
        if spread_velocity_bps_per_month < -10:
            prob = min(prob + 0.10, 0.99)  # Rapidly inverting = extra risk
        
        if prob > 0.45:
            signal = 'RED'
        elif prob > 0.20:
            signal = 'YELLOW'
        else:
            signal = 'GREEN'
        
        return RecessionProbOutput(
            spread_3m_10yr=spread_3m_10yr_bps,
            spread_velocity=spread_velocity_bps_per_month,
            prob_12m=round(prob, 4),
            prob_signal=signal,
            historical_accuracy="Predicted every US recession since 1960 (Estrella & Mishkin)"
        )


# =============================================================================
# Framework 6: Yield Curve Velocity & Acceleration
# =============================================================================

class YieldCurveFramework:
    """Yield curve velocity and acceleration — the *rate of change* matters more.
    
    A sudden fast inversion is far more dangerous than a gradual one.
    Acceleration (second derivative) is the early-warning signal.
    """
    
    def compute(
        self,
        spread_2_10_history: pd.Series,  # daily 2yr-10yr spread in bps
        spread_3m_10yr_history: pd.Series  # daily 3m-10yr spread in bps
    ) -> YieldCurveOutput:
        """
        Args:
            spread_2_10_history: Series indexed by date, last 90 days minimum
            spread_3m_10yr_history: same
        """
        if len(spread_2_10_history) < 30:
            raise ValueError("Need at least 30 days of yield curve data")
        
        s2_10 = spread_2_10_history
        s3m_10 = spread_3m_10yr_history
        
        current_2_10 = float(s2_10.iloc[-1])
        current_3m_10 = float(s3m_10.iloc[-1])
        
        # Velocity: change over last 30 days (bps/day)
        velocity_30d = float((s2_10.iloc[-1] - s2_10.iloc[-30]) / 30)
        
        # Acceleration: change in velocity (second derivative)
        vel_now = (s2_10.iloc[-1] - s2_10.iloc[-15]) / 15
        vel_prev = (s2_10.iloc[-15] - s2_10.iloc[-30]) / 15
        acceleration_30d = float((vel_now - vel_prev) / 15)
        
        # Inversion stats
        inversion_depth = min(0.0, current_2_10)  # 0 if not inverted
        inverted_mask = s2_10 < 0
        inversion_duration = int(inverted_mask.iloc[-90:].sum()) if len(s2_10) >= 90 else int(inverted_mask.sum())
        
        # Signal classification
        if current_2_10 < -50 and velocity_30d < -2:
            signal = 'CRASH_INVERSION'
        elif current_2_10 < 0:
            signal = 'INVERTED'
        elif abs(current_2_10) < 30:
            signal = 'FLAT'
        else:
            signal = 'BULL_STEEP'
        
        return YieldCurveOutput(
            spread_2_10=round(current_2_10, 1),
            spread_3m_10yr=round(current_3m_10, 1),
            velocity_30d=round(velocity_30d, 3),
            acceleration_30d=round(acceleration_30d, 4),
            inversion_depth=round(inversion_depth, 1),
            inversion_duration_days=inversion_duration,
            signal=signal
        )


# =============================================================================
# Orchestrator
# =============================================================================

class MacroImbalanceFramework:
    """Orchestrates all six frameworks into a composite stress reading.
    
    Usage:
        framework = MacroImbalanceFramework()
        result = framework.analyze(macro_data)
        
        # Feed HMM stress to XGBoost
        features.hmm_regime_stress = result.hmm.hmm_stress_score
        
        # Gate Petroulas trades
        if result.consensus_fault_detected:
            petroulas_gate.evaluate(result, kimi_score)
    """
    
    def __init__(self):
        self.erp_fw = ERPFramework()
        self.cape_fw = CAPEFramework()
        self.pca_fw = PCAFramework()
        self.hmm_fw = HMMFramework()
        self.recession_fw = RecessionProbFramework()
        self.yc_fw = YieldCurveFramework()
    
    def analyze(
        self,
        cape: float,
        bond_yield_10yr_pct: float,
        vix: float,
        spy_return_1m: float,
        market_breadth: float,
        spread_3m_10yr_bps: float,
        spread_velocity_bps_month: float,
        spread_2_10_history: pd.Series,
        spread_3m_10yr_history: pd.Series,
        pca_factors: Optional[Dict[str, float]] = None,
        symbol: str = 'SPY'
    ) -> RegimeStressOutput:
        """Run all six frameworks and emit a composite RegimeStressOutput.
        
        Args:
            cape: Current Shiller CAPE ratio (e.g. 35.2)
            bond_yield_10yr_pct: 10yr treasury yield in % (e.g. 4.5)
            vix: Current VIX level
            spy_return_1m: SPY 1-month return (e.g. -0.06 = -6%)
            market_breadth: % of S&P stocks above 200-day MA (0-1)
            spread_3m_10yr_bps: 3m-10yr spread in basis points
            spread_velocity_bps_month: rate of spread change bps/month
            spread_2_10_history: pd.Series of daily 2yr-10yr spread
            spread_3m_10yr_history: pd.Series of daily 3m-10yr spread
            pca_factors: optional dict of z-scored macro factors for PCA
            symbol: index symbol label
        """
        logger.info(f"[ImbalanceEngine] Running 6-framework analysis for {symbol}")
        
        # 1. ERP
        erp_out = self.erp_fw.compute(cape, bond_yield_10yr_pct)
        
        # 2. CAPE
        cape_out = self.cape_fw.compute(cape)
        
        # 3. PCA
        if pca_factors is None:
            pca_factors = {
                'vix': (vix - 20) / 8.0,
                'credit_spread_hy': 0.0,  # neutral if not provided
                'yield_curve_2_10': (spread_2_10_history.iloc[-1] if len(spread_2_10_history) else 0) / 50.0,
                'dxy': 0.0,
                'erp_z': erp_out.z_score,
                'cape_z': cape_out.z_score,
                'breadth': (market_breadth - 0.5) / 0.2
            }
        pca_out = self.pca_fw.compute(pca_factors)
        
        # 4. HMM
        hmm_out = self.hmm_fw.compute(vix, spy_return_1m, market_breadth)
        
        # 5. Recession Probability
        recession_out = self.recession_fw.compute(spread_3m_10yr_bps, spread_velocity_bps_month)
        
        # 6. Yield Curve
        yc_out = self.yc_fw.compute(spread_2_10_history, spread_3m_10yr_history)
        
        # Composite stress scoring (0-10 scale per framework, weighted average)
        stress_scores = [
            erp_out.magnitude,                                          # ERP overvaluation
            min(10, cape_out.z_score * 2.5 + 5),                      # CAPE z-score mapped to 0-10
            min(10, pca_out.mahalanobis_distance * 2.5),               # PCA Mahalanobis
            hmm_out.hmm_stress_score * 10,                             # HMM stress
            recession_out.prob_12m * 10,                               # Recession probability
            {
                'CRASH_INVERSION': 9, 'INVERTED': 7,
                'FLAT': 4, 'BULL_STEEP': 1
            }.get(yc_out.signal, 4)                                    # Yield curve
        ]
        stress_scores = [float(np.clip(s, 0, 10)) for s in stress_scores]
        
        weights = [0.20, 0.20, 0.15, 0.15, 0.20, 0.10]
        composite = float(np.dot(stress_scores, weights))
        
        # Consensus fault: >= 4 of 6 frameworks in stress zone (score > 5)
        fault_count = sum(1 for s in stress_scores if s > 5.0)
        consensus_fault = fault_count >= 4
        
        # Human-readable fault summary for Kimi
        fault_summary = self._build_fault_summary(
            erp_out, cape_out, pca_out, hmm_out, recession_out, yc_out,
            stress_scores, composite, fault_count
        )
        
        logger.info(
            f"[ImbalanceEngine] Composite Stress: {composite:.1f}/10 | "
            f"Faults: {fault_count}/6 | Consensus: {consensus_fault}"
        )
        
        return RegimeStressOutput(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            erp=erp_out,
            cape=cape_out,
            pca=pca_out,
            hmm=hmm_out,
            recession_prob=recession_out,
            yield_curve=yc_out,
            composite_stress_score=round(composite, 2),
            consensus_fault_detected=consensus_fault,
            fault_summary=fault_summary,
            hmm_regime_stress=hmm_out.hmm_stress_score
        )
    
    def _build_fault_summary(self, erp, cape, pca, hmm, rec, yc, scores, composite, count) -> str:
        return (
            f"IMBALANCE ENGINE REPORT\n"
            f"=======================\n"
            f"Composite Stress Score: {composite:.1f}/10 | Faults Detected: {count}/6\n\n"
            f"1. ERP: {erp.signal} | ERP={erp.erp:.2f}% vs avg {erp.historical_avg_erp}% | "
            f"z={erp.z_score:.2f} | Score={scores[0]:.1f}\n"
            f"2. CAPE: {cape.signal} | CAPE={cape.cape_ratio} | "
            f"z={cape.z_score:.2f} ({cape.percentile:.0f}th pct) | "
            f"Implied 10yr return={cape.implied_10yr_return:.1f}% | Score={scores[1]:.1f}\n"
            f"3. PCA Macro: {pca.stress_regime} | Mahalanobis={pca.mahalanobis_distance:.2f} | "
            f"Drivers={pca.top_contributors} | Score={scores[2]:.1f}\n"
            f"4. HMM Regime: {hmm.regime_label} | "
            f"Trans->Bear={hmm.transition_probs.get('BEAR', 0):.2f} "
            f"Trans->Crash={hmm.transition_probs.get('CRASH', 0):.2f} | "
            f"Stress={hmm.hmm_stress_score:.3f} | Score={scores[3]:.1f}\n"
            f"5. Recession Prob: {rec.prob_signal} | "
            f"Spread={rec.spread_3m_10yr:.0f}bps | 12m prob={rec.prob_12m:.1%} | "
            f"Score={scores[4]:.1f}\n"
            f"6. Yield Curve: {yc.signal} | 2-10={yc.spread_2_10:.0f}bps | "
            f"Velocity={yc.velocity_30d:.2f}bps/d | "
            f"Inverted {yc.inversion_duration_days}d | Score={scores[5]:.1f}\n"
        )
