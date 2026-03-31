"""
Entry Selector Scoring for Clawd Trading

Scores entry models based on market conditions and historical performance.
Rule-based implementation (no ML required) for deterministic results.
"""
from typing import Any, Dict, List

import numpy as np


class EntryModelScore:
    """Score for a specific entry model."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.prob_select = 0.0  # Probability this model is best
        self.expected_r = 0.0   # Expected return (R)
        self.confidence = 0.0   # Confidence in score
        self.metadata: Dict[str, Any] = {}


def score_fvg_respect_continuation(
    layer1_output: Dict[str, Any],
    layer2_output: Dict[str, Any],
) -> EntryModelScore:
    """Score FVG Respect Continuation model."""
    score = EntryModelScore("FVG_RESPECT_CONTINUATION")
    
    # Check for FVG conditions
    has_fvg = layer1_output.get("fvg_detected", False)
    trend = layer2_output.get("trend", "neutral")
    
    if has_fvg and trend in ["uptrend", "downtrend"]:
        score.prob_select = 0.7
        score.expected_r = 2.5
        score.confidence = 0.75
    elif has_fvg:
        score.prob_select = 0.4
        score.expected_r = 1.5
        score.confidence = 0.55
    else:
        score.prob_select = 0.1
        score.expected_r = 0.5
        score.confidence = 0.3
    
    score.metadata = {"fvg_detected": has_fvg, "trend": trend}
    return score


def score_sweep_displacement_reversal(
    layer1_output: Dict[str, Any],
    layer2_output: Dict[str, Any],
) -> EntryModelScore:
    """Score Sweep + Displacement Reversal model."""
    score = EntryModelScore("SWEEP_DISPLACEMENT_REVERSAL")
    
    has_sweep = layer1_output.get("liquidity_sweep", False)
    has_displacement = layer1_output.get("displacement", False)
    manipulation = layer2_output.get("manipulation_score", 0)
    
    if has_sweep and has_displacement and manipulation > 0.6:
        score.prob_select = 0.8
        score.expected_r = 3.0
        score.confidence = 0.8
    elif has_sweep and has_displacement:
        score.prob_select = 0.5
        score.expected_r = 2.0
        score.confidence = 0.6
    else:
        score.prob_select = 0.15
        score.expected_r = 0.8
        score.confidence = 0.4
    
    score.metadata = {
        "sweep": has_sweep,
        "displacement": has_displacement,
        "manipulation": manipulation,
    }
    return score


def score_ob_continuation(
    layer1_output: Dict[str, Any],
    layer2_output: Dict[str, Any],
) -> EntryModelScore:
    """Score Order Block Continuation model."""
    score = EntryModelScore("OB_CONTINUATION")
    
    has_ob = layer1_output.get("order_block", False)
    trend_aligned = layer2_output.get("bias_aligned", False)
    
    if has_ob and trend_aligned:
        score.prob_select = 0.65
        score.expected_r = 2.2
        score.confidence = 0.7
    elif has_ob:
        score.prob_select = 0.35
        score.expected_r = 1.2
        score.confidence = 0.5
    else:
        score.prob_select = 0.1
        score.expected_r = 0.4
        score.confidence = 0.25
    
    score.metadata = {"order_block": has_ob, "trend_aligned": trend_aligned}
    return score


def score_ict_concept(
    layer1_output: Dict[str, Any],
    layer2_output: Dict[str, Any],
) -> EntryModelScore:
    """Score ICT Concept model (Breaker, Mitigation, etc.)."""
    score = EntryModelScore("ICT_CONCEPT")
    
    ict_setup = layer1_output.get("ict_setup", {})
    has_breaker = ict_setup.get("breaker", False)
    has_mitigation = ict_setup.get("mitigation", False)
    fvg_respected = ict_setup.get("fvg_respected", False)
    
    ict_score = sum([has_breaker, has_mitigation, fvg_respected])
    
    if ict_score >= 2:
        score.prob_select = 0.6
        score.expected_r = 2.0
        score.confidence = 0.65
    elif ict_score >= 1:
        score.prob_select = 0.3
        score.expected_r = 1.0
        score.confidence = 0.45
    else:
        score.prob_select = 0.05
        score.expected_r = 0.2
        score.confidence = 0.2
    
    score.metadata = {"ict_score": ict_score, "setup": ict_setup}
    return score


# Registry of all entry models
ENTRY_MODELS = {
    "FVG_RESPECT_CONTINUATION": score_fvg_respect_continuation,
    "SWEEP_DISPLACEMENT_REVERSAL": score_sweep_displacement_reversal,
    "OB_CONTINUATION": score_ob_continuation,
    "ICT_CONCEPT": score_ict_concept,
}


def score_all_entry_models(
    layer1_output: Dict[str, Any],
    layer2_output: Dict[str, Any],
    eligible_models: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Score all eligible entry models.
    
    Args:
        layer1_output: Layer 1 analysis
        layer2_output: Layer 2 bias/output
        eligible_models: List of model IDs to score (default: all)
    
    Returns:
        Dict mapping model_id to score dict with prob_select, expected_r, confidence
    """
    if eligible_models is None:
        eligible_models = list(ENTRY_MODELS.keys())
    
    results = {}
    
    for model_id in eligible_models:
        scorer_func = ENTRY_MODELS.get(model_id)
        if scorer_func:
            score = scorer_func(layer1_output, layer2_output)
            results[model_id] = {
                "prob_select": score.prob_select,
                "expected_R": score.expected_r,
                "confidence": score.confidence,
                "metadata": score.metadata,
            }
    
    return results


def select_best_entry_model(
    scores: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.5,
    min_expected_r: float = 1.0,
) -> Dict[str, Any]:
    """
    Select the best entry model from scored options.
    
    Args:
        scores: Output from score_all_entry_models()
        min_confidence: Minimum confidence threshold
        min_expected_r: Minimum expected return threshold
    
    Returns:
        Best model selection or None if none meet criteria
    """
    # Filter by thresholds
    qualified = {
        model_id: score
        for model_id, score in scores.items()
        if score["confidence"] >= min_confidence and score["expected_R"] >= min_expected_r
    }
    
    if not qualified:
        return {
            "selected": None,
            "reason": "no_models_meet_thresholds",
            "best_attempt": max(scores.items(), key=lambda x: x[1]["expected_R"])[0] if scores else None,
        }
    
    # Select by expected R weighted by confidence
    def weighted_score(score):
        return score["expected_R"] * score["confidence"] * score["prob_select"]
    
    best_model = max(qualified.items(), key=lambda x: weighted_score(x[1]))
    
    return {
        "selected": best_model[0],
        "score": best_model[1],
        "all_qualified": qualified,
        "selection_method": "weighted_expected_r",
    }
