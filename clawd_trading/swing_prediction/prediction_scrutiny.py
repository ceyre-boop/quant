"""
Prediction Scrutiny - Chi-Squared Validation Gate

Validates predictions against actual outcomes using (O-E)²/E.
Only predictions that survive statistical scrutiny get through.
"""

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ScrutinyResult:
    """Result of prediction scrutiny."""

    prediction_id: str
    passed: bool
    chi_squared_stat: float
    p_value: float
    observed_wins: int
    expected_wins: float
    sample_size: int
    rejection_reason: str = ""


def validate_prediction_against_outcomes(
    prediction: Dict[str, Any],
    historical_outcomes: List[bool],
    expected_win_rate: float,
    alpha: float = 0.05,
) -> ScrutinyResult:
    """
    Validate a prediction using chi-squared test.

    Args:
        prediction: The prediction to validate
        historical_outcomes: List of actual outcomes (True=win, False=loss)
        expected_win_rate: Predicted win rate (0-1)
        alpha: Significance level (default 0.05)

    Returns:
        ScrutinyResult with pass/fail and statistics
    """
    n = len(historical_outcomes)

    if n < 30:
        return ScrutinyResult(
            prediction_id=prediction.get("id", "unknown"),
            passed=False,
            chi_squared_stat=0.0,
            p_value=1.0,
            observed_wins=sum(historical_outcomes),
            expected_wins=expected_win_rate * n,
            sample_size=n,
            rejection_reason="Insufficient sample size (< 30)",
        )

    observed_wins = sum(historical_outcomes)
    observed_losses = n - observed_wins

    expected_wins = expected_win_rate * n
    expected_losses = (1 - expected_win_rate) * n

    # Chi-squared: (O - E)² / E
    chi_sq = (observed_wins - expected_wins) ** 2 / expected_wins + (
        observed_losses - expected_losses
    ) ** 2 / expected_losses

    # Degrees of freedom = 1 (wins vs losses)
    p_value = 1 - stats.chi2.cdf(chi_sq, df=1)

    passed = p_value >= alpha

    if not passed:
        rejection_reason = f"Chi-squared p-value {p_value:.4f} < alpha {alpha}"
    else:
        rejection_reason = ""

    logger.info(
        f"Prediction {prediction.get('id', 'unknown')}: "
        f"chi²={chi_sq:.4f}, p={p_value:.4f}, passed={passed}"
    )

    return ScrutinyResult(
        prediction_id=prediction.get("id", "unknown"),
        passed=passed,
        chi_squared_stat=chi_sq,
        p_value=p_value,
        observed_wins=observed_wins,
        expected_wins=expected_wins,
        sample_size=n,
        rejection_reason=rejection_reason,
    )


class PredictionScrutinyGate:
    """Gate that blocks predictions failing chi-squared validation."""

    def __init__(self, alpha: float = 0.05, min_sample_size: int = 30):
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.logger = logging.getLogger(__name__)

    def scrutinize(self, prediction: Dict[str, Any], outcomes: List[bool]) -> bool:
        """
        Returns True if prediction passes scrutiny, False otherwise.
        """
        expected_wr = prediction.get("expected_win_rate", 0.5)

        result = validate_prediction_against_outcomes(
            prediction, outcomes, expected_wr, self.alpha
        )

        if not result.passed:
            self.logger.warning(f"Prediction BLOCKED: {result.rejection_reason}")

        return result.passed
