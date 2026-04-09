"""
Composite Scorer — Aggregates all 5 layers into final swing bias
"""

import logging
from typing import Dict


class CompositeScorer:
    """
    Combines all 5 layer scores into composite signal.
    """

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["composite_scoring"]

    def calculate_composite(self, layer_scores: Dict[str, float]) -> float:
        """
        Calculate weighted composite score from all layers.
        """
        weights = self.config["layer_weights"]

        composite = 0
        total_weight = 0

        for layer, score in layer_scores.items():
            if layer in weights:
                composite += score * weights[layer]
                total_weight += weights[layer]

        if total_weight > 0:
            composite /= total_weight

        return round(composite, 4)

    def direction_from_score(self, composite: float) -> str:
        """
        Determine trade direction from composite score.
        """
        thresholds = self.config["thresholds"]

        if composite >= thresholds["strong_long"]:
            return "strong_long"
        elif composite >= thresholds["moderate_long"]:
            return "moderate_long"
        elif composite <= thresholds["strong_short"]:
            return "strong_short"
        elif composite <= thresholds["moderate_short"]:
            return "moderate_short"
        else:
            return "neutral"

    def confidence_from_alignment(self, layers_aligned: int, base_rate: float = None) -> float:
        """
        Calculate confidence based on layer alignment and base rate.
        """
        if base_rate:
            return base_rate

        # Fallback if no base rate
        alignment_confidence = {0: 0.50, 1: 0.52, 2: 0.55, 3: 0.60, 4: 0.70, 5: 0.75}

        return alignment_confidence.get(layers_aligned, 0.50)
