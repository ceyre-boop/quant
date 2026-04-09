"""Data Validator

Schema validation for all data records before Firebase write.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from pydantic import ValidationError

from data.schema import (
    FeatureRecordSchema,
    BiasOutputSchema,
    RiskOutputSchema,
    GameOutputSchema,
    EntrySignalSchema,
    PositionSchema,
    AccountStateSchema,
    DataQuality,
)

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation operation."""

    def __init__(self, is_valid: bool, errors: List[str], data: Optional[Dict] = None):
        self.is_valid = is_valid
        self.errors = errors
        self.data = data

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, errors={len(self.errors)})"


class DataValidator:
    """Validator for all system data records."""

    def __init__(self):
        self.validation_stats = {
            "total_validated": 0,
            "valid_count": 0,
            "invalid_count": 0,
        }

    def validate_feature_record(
        self, data: Dict[str, Any], strict: bool = True
    ) -> ValidationResult:
        """Validate a feature record against schema.

        Args:
            data: Feature record data
            strict: If True, raises on validation error

        Returns:
            ValidationResult with is_valid status and any errors
        """
        self.validation_stats["total_validated"] += 1

        try:
            # First check basic structure
            if not isinstance(data, dict):
                error = "Data must be a dictionary"
                self.validation_stats["invalid_count"] += 1
                return ValidationResult(False, [error])

            # Check required fields
            required = ["symbol", "timestamp", "timeframe", "features", "is_valid"]
            missing = [f for f in required if f not in data]
            if missing:
                error = f"Missing required fields: {missing}"
                self.validation_stats["invalid_count"] += 1
                return ValidationResult(False, [error])

            # Validate with Pydantic
            validated = FeatureRecordSchema(**data)

            # Additional sanity checks
            errors = self._sanity_check_features(validated.features.model_dump())
            if errors:
                self.validation_stats["invalid_count"] += 1
                return ValidationResult(False, errors)

            self.validation_stats["valid_count"] += 1
            return ValidationResult(True, [], validated.model_dump())

        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            self.validation_stats["invalid_count"] += 1
            logger.debug(f"Feature record validation failed: {errors}")
            return ValidationResult(False, errors)
        except Exception as e:
            self.validation_stats["invalid_count"] += 1
            return ValidationResult(False, [str(e)])

    def validate_bias_output(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate bias output against schema."""
        self.validation_stats["total_validated"] += 1

        try:
            validated = BiasOutputSchema(**data)
            self.validation_stats["valid_count"] += 1
            return ValidationResult(True, [], validated.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            self.validation_stats["invalid_count"] += 1
            return ValidationResult(False, errors)

    def validate_risk_structure(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate risk structure against schema."""
        self.validation_stats["total_validated"] += 1

        try:
            validated = RiskOutputSchema(**data)
            self.validation_stats["valid_count"] += 1
            return ValidationResult(True, [], validated.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            self.validation_stats["invalid_count"] += 1
            return ValidationResult(False, errors)

    def validate_game_output(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate game output against schema."""
        self.validation_stats["total_validated"] += 1

        try:
            validated = GameOutputSchema(**data)
            self.validation_stats["valid_count"] += 1
            return ValidationResult(True, [], validated.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            self.validation_stats["invalid_count"] += 1
            return ValidationResult(False, errors)

    def validate_entry_signal(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate entry signal against schema."""
        self.validation_stats["total_validated"] += 1

        try:
            validated = EntrySignalSchema(**data)
            self.validation_stats["valid_count"] += 1
            return ValidationResult(True, [], validated.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            self.validation_stats["invalid_count"] += 1
            return ValidationResult(False, errors)

    def validate_position(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate position state against schema."""
        self.validation_stats["total_validated"] += 1

        try:
            validated = PositionSchema(**data)
            self.validation_stats["valid_count"] += 1
            return ValidationResult(True, [], validated.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            self.validation_stats["invalid_count"] += 1
            return ValidationResult(False, errors)

    def validate_account_state(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate account state against schema."""
        self.validation_stats["total_validated"] += 1

        try:
            validated = AccountStateSchema(**data)
            self.validation_stats["valid_count"] += 1
            return ValidationResult(True, [], validated.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            self.validation_stats["invalid_count"] += 1
            return ValidationResult(False, errors)

    def validate_json_file(
        self, filepath: Union[str, Path], record_type: str = "feature_record"
    ) -> ValidationResult:
        """Validate a JSON file containing a record.

        Args:
            filepath: Path to JSON file
            record_type: Type of record to validate

        Returns:
            ValidationResult
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return ValidationResult(False, [f"File not found: {filepath}"])

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return ValidationResult(False, [f"Invalid JSON: {e}"])
        except Exception as e:
            return ValidationResult(False, [f"Error reading file: {e}"])

        # Route to appropriate validator
        validators = {
            "feature_record": self.validate_feature_record,
            "bias_output": self.validate_bias_output,
            "risk_structure": self.validate_risk_structure,
            "game_output": self.validate_game_output,
            "entry_signal": self.validate_entry_signal,
            "position": self.validate_position,
            "account_state": self.validate_account_state,
        }

        validator = validators.get(record_type)
        if not validator:
            return ValidationResult(False, [f"Unknown record type: {record_type}"])

        return validator(data)

    def _sanity_check_features(self, features: Dict[str, float]) -> List[str]:
        """Perform sanity checks on feature values.

        Returns:
            List of error messages (empty if all checks pass)
        """
        errors = []

        # Check ATR bounds (should be < 10% of typical price)
        atr = features.get("atr_14", 0)
        if atr > 10000:  # Unrealistic for any normal market
            errors.append(f"ATR {atr} exceeds sanity bounds")

        # Check RSI bounds
        rsi = features.get("rsi_14", 50)
        if not (0 <= rsi <= 100):
            errors.append(f"RSI {rsi} outside valid range [0, 100]")

        # Check ADX bounds
        adx = features.get("adx_14", 25)
        if not (0 <= adx <= 100):
            errors.append(f"ADX {adx} outside valid range [0, 100]")

        # Check correlation bounds
        corr = features.get("spy_correlation_20d", 0)
        if not (-1 <= corr <= 1):
            errors.append(f"Correlation {corr} outside valid range [-1, 1]")

        # Check volatility regime
        vol_reg = features.get("volatility_regime", 2)
        if vol_reg not in [1, 2, 3, 4]:
            errors.append(f"Invalid volatility regime: {vol_reg}")

        return errors

    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self.validation_stats.copy()

    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validated": 0,
            "valid_count": 0,
            "invalid_count": 0,
        }


# Convenience functions for quick validation


def validate_feature_record(data: Dict[str, Any]) -> ValidationResult:
    """Quick validation of feature record."""
    validator = DataValidator()
    return validator.validate_feature_record(data)


def validate_bias_output(data: Dict[str, Any]) -> ValidationResult:
    """Quick validation of bias output."""
    validator = DataValidator()
    return validator.validate_bias_output(data)


def validate_risk_structure(data: Dict[str, Any]) -> ValidationResult:
    """Quick validation of risk structure."""
    validator = DataValidator()
    return validator.validate_risk_structure(data)


def validate_game_output(data: Dict[str, Any]) -> ValidationResult:
    """Quick validation of game output."""
    validator = DataValidator()
    return validator.validate_game_output(data)


def validate_json_file(
    filepath: Union[str, Path], record_type: str = "feature_record"
) -> ValidationResult:
    """Quick validation of JSON file."""
    validator = DataValidator()
    return validator.validate_json_file(filepath, record_type)
