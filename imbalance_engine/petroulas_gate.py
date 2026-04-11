"""Petroulas Gate — Dual-confirmation structural fault detection.

Architecture:
  - XGBoost (Layer 1) produces baseline bias + confidence
  - MacroImbalanceFramework produces composite_stress_score + fault_summary
  - Kimi (Layer 1.5) receives the fault_summary and scores:
      magnitude: 1-10 (how large is the structural fault?)
      conviction: 1-10 (how confident is the asymmetric thesis?)
      consensus_blindspot: str (what does consensus miss?)
      petroulas_worthy: bool (worthy of a conviction-sized position?)
  - When BOTH XGBoost confidence > threshold AND Kimi petroulas_worthy=True:
      → calculate_petroulas_risk() scales position to 3-5% of account
  - Normal trades: 1-2% of account

Named after Petroulas — who bets ungodly amounts specifically because
the math is conclusive while consensus still believes there is no fault.

Historical case studies proved by arithmetic, not analysis:
  2021 Rate Shock:    M2=PQ arithmetic was conclusive 5 months before crash
                      while consensus said "transitory"
  SVB Crisis:         Duration × rate × portfolio math visible 12m before failure
  Petroulas Google:   FCF yield > 10yr bond yield + search monopoly math
                      vs narrative of AI destruction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import json
import os
import re

from imbalance_engine.frameworks import RegimeStressOutput

logger = logging.getLogger(__name__)

# =============================================================================
# Contracts
# =============================================================================

@dataclass
class KimiFaultScore:
    """Kimi's structural analysis of the fault."""
    magnitude: int                    # 1-10: size of structural imbalance
    conviction: int                   # 1-10: asymmetric thesis confidence
    consensus_blindspot: str          # What consensus is missing (specific)
    petroulas_worthy: bool            # Worthy of conviction-sized tranche?
    reasoning: str                    # Kimi's detailed reasoning
    arithmetic_proof: str             # The specific mathematical proof
    falsification_test: str           # Observable event that kills thesis within 30 days
    time_horizon_days: int            # Expected days for thesis to play out


@dataclass
class PetroulsasDecision:
    """Output of the dual-confirmation gate."""
    approved: bool                    # True = Petroulas trade approved
    position_size_pct: float          # % of account to risk (1-5%)
    normal_size_pct: float            # baseline (1-2%)
    fault_quality: float              # 0-10, composite quality score
    xgb_confidence: float             # XGBoost confidence at gate
    kimi_score: Optional[KimiFaultScore]
    stress_score: float               # from MacroImbalanceFramework
    reason: str                       # human-readable decision reason
    
    # Metadata for logging
    symbol: str
    timestamp: str
    thesis_id: str                    # unique ID for falsification tracking


# =============================================================================
# Kimi Prompt Builder
# =============================================================================

PETROULAS_KIMI_PROMPT = """You are acting as a structural fault detector. Your job is not to describe the market — it is to identify SPECIFIC MATHEMATICAL IMBALANCES that consensus has missed.

IMBALANCE ENGINE REPORT:
{fault_summary}

SYMBOL CONTEXT:
Symbol: {symbol}
Current XGBoost Bias: {direction} | Confidence: {xgb_confidence:.1%}
Entry Price: {entry_price}

YOUR TASK:
1. Score the MAGNITUDE of the structural fault (1=trivial, 10=systemic imbalance)
2. Score your CONVICTION in the asymmetric thesis (1=speculative, 10=arithmetic certainty)
3. Identify WHAT SPECIFICALLY consensus is missing (must be measurable)
4. Define the ONE observable test within 30 days that would FALSIFY this thesis
5. Estimate the time horizon for the thesis to resolve (days)

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "magnitude": <1-10>,
  "conviction": <1-10>,
  "consensus_blindspot": "<specific, measurable thing consensus ignores>",
  "petroulas_worthy": <true/false>,
  "reasoning": "<2-3 sentence max — the arithmetic, not the narrative>",
  "arithmetic_proof": "<the specific calculation that proves the fault>",
  "falsification_test": "<observable event within 30 days that kills the thesis>",
  "time_horizon_days": <integer>
}}

CRITICAL RULES:
- petroulas_worthy = true ONLY if magnitude >= 7 AND conviction >= 7
- The arithmetic_proof MUST contain actual numbers, not vague statements
- The falsification_test MUST be specific and observable in 30 days
- "AI will change everything" is not arithmetic. "ERP = 0.3% vs 3.5% historical avg" IS arithmetic.
"""


# =============================================================================
# The Gate
# =============================================================================

class PetroulsasGate:
    """Dual-confirmation gate for Petroulas-class structural trades.
    
    Usage in orchestrator:
        gate = PetroulsasGate()
        
        decision = gate.evaluate(
            symbol='AAPL',
            regime_stress=imbalance_result,
            xgb_confidence=0.73,
            direction='LONG',
            entry_price=185.0,
            kimi_client=self.kimi_brain  # optional, skips if None
        )
        
        if decision.approved:
            # Scale position
            size = calculate_petroulas_risk(decision, account_equity)
    """
    
    # Approval thresholds
    MIN_COMPOSITE_STRESS = 6.0       # Must have systemic context
    MIN_XGB_CONFIDENCE = 0.65        # XGBoost must be confident
    MIN_KIMI_MAGNITUDE = 7           # Kimi magnitude threshold
    MIN_KIMI_CONVICTION = 7          # Kimi conviction threshold
    
    # Position sizing
    NORMAL_SIZE_PCT = 1.5            # default per trade
    PETROULAS_BASE_PCT = 3.0         # base Petroulas size
    PETROULAS_MAX_PCT = 5.0          # max when fault quality = 10
    
    def __init__(self, kimi_client=None):
        self.kimi_client = kimi_client   # KimiBrain instance or None
    
    def evaluate(
        self,
        symbol: str,
        regime_stress: RegimeStressOutput,
        xgb_confidence: float,
        direction: str,
        entry_price: float
    ) -> PetroulsasDecision:
        """Run the dual-confirmation gate.
        
        Fast path: if stress < threshold or XGB confidence < threshold, 
        return normal-sized decision immediately without calling Kimi.
        """
        thesis_id = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Fast reject: insufficient macro context
        if regime_stress.composite_stress_score < self.MIN_COMPOSITE_STRESS:
            return self._normal_decision(
                symbol, xgb_confidence, regime_stress.composite_stress_score,
                f"Macro stress {regime_stress.composite_stress_score:.1f} < {self.MIN_COMPOSITE_STRESS} threshold",
                thesis_id
            )
        
        # Fast reject: XGBoost not confident enough
        if xgb_confidence < self.MIN_XGB_CONFIDENCE:
            return self._normal_decision(
                symbol, xgb_confidence, regime_stress.composite_stress_score,
                f"XGBoost confidence {xgb_confidence:.1%} < {self.MIN_XGB_CONFIDENCE:.1%} threshold",
                thesis_id
            )
        
        # Also require consensus fault across frameworks
        if not regime_stress.consensus_fault_detected:
            return self._normal_decision(
                symbol, xgb_confidence, regime_stress.composite_stress_score,
                "No consensus fault (< 4/6 frameworks in stress zone)",
                thesis_id
            )
        
        # Escalate to Kimi for fault scoring
        kimi_score = self._call_kimi(regime_stress, symbol, direction, xgb_confidence, entry_price)
        
        if kimi_score is None:
            # Kimi unavailable — approved as large-normal trade based on framework alone
            return self._framework_only_decision(
                symbol, xgb_confidence, regime_stress.composite_stress_score, thesis_id
            )
        
        # Dual confirmation check
        if (kimi_score.petroulas_worthy 
                and kimi_score.magnitude >= self.MIN_KIMI_MAGNITUDE 
                and kimi_score.conviction >= self.MIN_KIMI_CONVICTION):
            
            fault_quality = (kimi_score.magnitude + kimi_score.conviction) / 2.0
            size = self.calculate_petroulas_risk(fault_quality, regime_stress.composite_stress_score)
            
            reason = (
                f"PETROULAS APPROVED — "
                f"Fault magnitude={kimi_score.magnitude}/10, conviction={kimi_score.conviction}/10 | "
                f"Blindspot: {kimi_score.consensus_blindspot[:80]} | "
                f"Proof: {kimi_score.arithmetic_proof[:100]}"
            )
            
            logger.info(f"[PetroulsasGate] {symbol} APPROVED — size={size:.1f}% | {reason}")
            
            return PetroulsasDecision(
                approved=True,
                position_size_pct=size,
                normal_size_pct=self.NORMAL_SIZE_PCT,
                fault_quality=fault_quality,
                xgb_confidence=xgb_confidence,
                kimi_score=kimi_score,
                stress_score=regime_stress.composite_stress_score,
                reason=reason,
                symbol=symbol,
                timestamp=datetime.utcnow().isoformat(),
                thesis_id=thesis_id
            )
        
        # Disagreement or below threshold
        reason = (
            f"Dual confirmation failed — "
            f"Kimi magnitude={kimi_score.magnitude}/10, conviction={kimi_score.conviction}/10, "
            f"petroulas_worthy={kimi_score.petroulas_worthy}"
        )
        logger.info(f"[PetroulsasGate] {symbol} REJECTED — {reason}")
        
        return self._normal_decision(symbol, xgb_confidence, regime_stress.composite_stress_score, reason, thesis_id, kimi_score)
    
    def calculate_petroulas_risk(self, fault_quality: float, stress_score: float) -> float:
        """Scale position size based on fault quality.
        
        fault_quality: 0-10 (average of Kimi magnitude + conviction)
        stress_score:  0-10 (composite macro stress)
        
        Returns: position size as % of account (3.0 - 5.0)
        """
        # Linear interpolation from 3% to 5% as fault quality goes 7→10
        quality_factor = (fault_quality - 7.0) / 3.0  # 0.0 at 7, 1.0 at 10
        stress_factor = (stress_score - 6.0) / 4.0    # 0.0 at 6, 1.0 at 10
        
        # Both factors must contribute
        combined = (quality_factor * 0.6 + stress_factor * 0.4)
        combined = max(0.0, min(1.0, combined))
        
        size = self.PETROULAS_BASE_PCT + combined * (self.PETROULAS_MAX_PCT - self.PETROULAS_BASE_PCT)
        return round(size, 2)
    
    def _call_kimi(
        self,
        regime_stress: RegimeStressOutput,
        symbol: str,
        direction: str,
        xgb_confidence: float,
        entry_price: float
    ) -> Optional[KimiFaultScore]:
        """Call Kimi with the fault detection prompt."""
        if self.kimi_client is None:
            logger.warning("[PetroulsasGate] Kimi not available — skipping fault scoring")
            return None
        
        prompt = PETROULAS_KIMI_PROMPT.format(
            fault_summary=regime_stress.fault_summary,
            symbol=symbol,
            direction=direction,
            xgb_confidence=xgb_confidence,
            entry_price=entry_price
        )
        
        try:
            response = self.kimi_client.raw_completion(prompt)
            return self._parse_kimi_response(response)
        except Exception as e:
            logger.error(f"[PetroulsasGate] Kimi call failed: {e}")
            return None
    
    def _parse_kimi_response(self, response: str) -> Optional[KimiFaultScore]:
        """Parse Kimi's JSON response into KimiFaultScore."""
        try:
            # Extract JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in Kimi response")
            
            data = json.loads(json_match.group())
            
            return KimiFaultScore(
                magnitude=int(data.get('magnitude', 0)),
                conviction=int(data.get('conviction', 0)),
                consensus_blindspot=str(data.get('consensus_blindspot', '')),
                petroulas_worthy=bool(data.get('petroulas_worthy', False)),
                reasoning=str(data.get('reasoning', '')),
                arithmetic_proof=str(data.get('arithmetic_proof', '')),
                falsification_test=str(data.get('falsification_test', '')),
                time_horizon_days=int(data.get('time_horizon_days', 30))
            )
        except Exception as e:
            logger.error(f"[PetroulsasGate] Failed to parse Kimi response: {e}")
            return None
    
    def _normal_decision(
        self,
        symbol: str,
        xgb_confidence: float,
        stress_score: float,
        reason: str,
        thesis_id: str,
        kimi_score: Optional[KimiFaultScore] = None
    ) -> PetroulsasDecision:
        return PetroulsasDecision(
            approved=False,
            position_size_pct=self.NORMAL_SIZE_PCT,
            normal_size_pct=self.NORMAL_SIZE_PCT,
            fault_quality=0.0,
            xgb_confidence=xgb_confidence,
            kimi_score=kimi_score,
            stress_score=stress_score,
            reason=reason,
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            thesis_id=thesis_id
        )
    
    def _framework_only_decision(
        self,
        symbol: str,
        xgb_confidence: float,
        stress_score: float,
        thesis_id: str
    ) -> PetroulsasDecision:
        """Kimi unavailable but frameworks agree — approve at base size."""
        size = min(self.PETROULAS_BASE_PCT, self.NORMAL_SIZE_PCT * 1.5)
        return PetroulsasDecision(
            approved=True,
            position_size_pct=round(size, 2),
            normal_size_pct=self.NORMAL_SIZE_PCT,
            fault_quality=5.0,  # moderate — no Kimi to confirm
            xgb_confidence=xgb_confidence,
            kimi_score=None,
            stress_score=stress_score,
            reason=f"Framework-only approval (Kimi unavailable) — stress={stress_score:.1f}/10",
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            thesis_id=thesis_id
        )
