
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class ResearchSignal:
    hypothesis_id: str
    signal: float  # +1 (Long), -1 (Short), 0 (Neutral)
    confidence: float
    metadata: Dict[str, Any]

class ResearchEvaluator:
    """
    The Scientific Evaluator for the Desktop.
    Polls all active Tier 1-3 hypotheses without influencing current execution.
    """
    def __init__(self, registry_path: str = "research/registry.json"):
        self.logger = logging.getLogger(__name__)
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load research registry: {e}")
            return {"hypotheses": []}

    def evaluate_all(self, market_snapshot: pd.DataFrame, context: Any) -> List[ResearchSignal]:
        """
        Poll all hypotheses that are in 'IN_SAMPLE', 'PAPER_TRADE', or 'MICRO_DEPLOYMENT' states.
        """
        signals = []
        active_states = ["IN_SAMPLE", "PAPER_TRADE", "MICRO_DEPLOYMENT"]
        
        for hyp in self.registry.get("hypotheses", []):
            if hyp["state"] in active_states:
                # In a full implementation, we would dynamically load the tier module here
                # For now, we generate a 'Shadow Signal' for logging and discovery
                signal = self._mock_hypothesis_execution(hyp, market_snapshot)
                signals.append(signal)
        
        return signals

    def _mock_hypothesis_execution(self, hyp: Dict, df: pd.DataFrame) -> ResearchSignal:
        """Mock execution for the skeleton stage."""
        return ResearchSignal(
            hypothesis_id=hyp["id"],
            signal=0, # Neutral until module is implemented
            confidence=0.0,
            metadata={"description": hyp["description"], "tier": hyp["tier"]}
        )

    def log_signals(self, signals: List[ResearchSignal]):
        for s in signals:
            self.logger.info(f"[RESEARCH] {s.hypothesis_id} | Signal: {s.signal} | Conf: {s.confidence:.2f}")
