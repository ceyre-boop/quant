"""Unit tests for data/validator.py"""

import json
import pytest
from pathlib import Path

from data.validator import DataValidator, validate_feature_record, ValidationResult
from data.schema import DataQuality

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestValidationResult:
    """Test ValidationResult class."""

    def test_valid_result(self):
        result = ValidationResult(True, [])
        assert result.is_valid
        assert len(result.errors) == 0
        assert bool(result) is True

    def test_invalid_result(self):
        result = ValidationResult(False, ["error1", "error2"])
        assert not result.is_valid
        assert len(result.errors) == 2
        assert bool(result) is False

    def test_repr(self):
        result = ValidationResult(True, [])
        assert "VALID" in repr(result)


class TestValidateFeatureRecord:
    """Test feature record validation."""

    @pytest.fixture
    def valid_record(self):
        """Load valid feature record fixture."""
        with open(FIXTURES_DIR / "sample_feature_record.json") as f:
            return json.load(f)

    def test_valid_feature_record(self, valid_record):
        """Test that valid fixture passes validation."""
        validator = DataValidator()
        result = validator.validate_feature_record(valid_record)

        assert result.is_valid, f"Expected valid, got errors: {result.errors}"
        assert len(result.errors) == 0
        assert result.data is not None

    def test_valid_feature_record_quick_function(self, valid_record):
        """Test quick validation function."""
        result = validate_feature_record(valid_record)
        assert result.is_valid

    def test_missing_required_fields(self):
        """Test validation fails on missing required fields."""
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            # Missing timeframe, features, is_valid
        }
        validator = DataValidator()
        result = validator.validate_feature_record(data)

        assert not result.is_valid
        assert any("Missing required" in e for e in result.errors)

    def test_non_dict_input(self):
        """Test validation fails on non-dict input."""
        validator = DataValidator()
        result = validator.validate_feature_record("not a dict")

        assert not result.is_valid
        assert any("dictionary" in e for e in result.errors)


class TestBrokenFixtures:
    """Test that broken fixtures are correctly rejected."""

    @pytest.mark.parametrize(
        "fixture_name,expected_error",
        [
            ("broken_feature_record_atr_unrealistic.json", "ATR"),
            ("broken_feature_record_adx_out_of_range.json", "adx_14"),
            ("broken_feature_record_empty_symbol.json", "symbol"),
            ("broken_feature_record_invalid_timeframe.json", "timeframe"),
            ("broken_feature_record_invalid_vol_regime.json", "volatility_regime"),
        ],
    )
    def test_broken_fixtures_rejected(self, fixture_name, expected_error):
        """Test that each broken fixture is rejected with correct error."""
        fixture_path = FIXTURES_DIR / fixture_name

        validator = DataValidator()
        result = validator.validate_json_file(fixture_path, "feature_record")

        assert not result.is_valid, f"Expected {fixture_name} to be invalid"
        error_str = " ".join(result.errors).lower()
        assert (
            expected_error.lower() in error_str
        ), f"Expected error containing '{expected_error}' in {fixture_name}, got: {result.errors}"


class TestValidateBiasOutput:
    """Test bias output validation."""

    def test_valid_bias_output(self):
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            "direction": 1,
            "magnitude": 2,
            "confidence": 0.75,
            "regime_override": False,
            "rationale": ["TREND_STRENGTH", "MOMENTUM_SHIFT"],
            "model_version": "v1.0",
            "feature_snapshot": {
                "raw_features": {"f1": 1.0},
                "feature_group_tags": {},
                "regime_at_inference": {},
                "inference_timestamp": "2024-01-15T14:30:00",
            },
        }
        validator = DataValidator()
        result = validator.validate_bias_output(data)

        assert result.is_valid, f"Errors: {result.errors}"

    def test_invalid_rationale_group(self):
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            "direction": 1,
            "magnitude": 2,
            "confidence": 0.75,
            "regime_override": False,
            "rationale": ["INVALID_GROUP_NAME"],
            "model_version": "v1.0",
            "feature_snapshot": {
                "raw_features": {},
                "feature_group_tags": {},
                "regime_at_inference": {},
                "inference_timestamp": "2024-01-15T14:30:00",
            },
        }
        validator = DataValidator()
        result = validator.validate_bias_output(data)

        assert not result.is_valid
        assert any("Invalid rationale" in e for e in result.errors)

    def test_confidence_out_of_range(self):
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            "direction": 1,
            "magnitude": 2,
            "confidence": 1.5,  # Invalid: > 1
            "regime_override": False,
            "rationale": [],
            "model_version": "v1.0",
            "feature_snapshot": {
                "raw_features": {},
                "feature_group_tags": {},
                "regime_at_inference": {},
                "inference_timestamp": "2024-01-15T14:30:00",
            },
        }
        validator = DataValidator()
        result = validator.validate_bias_output(data)

        assert not result.is_valid
        assert any("confidence" in e.lower() for e in result.errors)


class TestValidateRiskStructure:
    """Test risk structure validation."""

    def test_valid_risk_structure(self):
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            "bias_id": "bias_001",
            "position_size": 1000.0,
            "kelly_fraction": 0.15,
            "stop_price": 19500.0,
            "stop_method": "atr",
            "tp1_price": 20000.0,
            "tp2_price": 20500.0,
            "trail_config": {"atr_multiple": 1.5},
            "expected_value": 0.05,
            "ev_positive": True,
            "size_breakdown": {"base_size": 1000},
        }
        validator = DataValidator()
        result = validator.validate_risk_structure(data)

        assert result.is_valid, f"Errors: {result.errors}"

    def test_invalid_stop_method(self):
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            "bias_id": "bias_001",
            "position_size": 1000.0,
            "kelly_fraction": 0.15,
            "stop_price": 19500.0,
            "stop_method": "invalid_method",
            "tp1_price": 20000.0,
            "tp2_price": 20500.0,
            "trail_config": {},
            "expected_value": 0.05,
            "ev_positive": True,
            "size_breakdown": {},
        }
        validator = DataValidator()
        result = validator.validate_risk_structure(data)

        assert not result.is_valid


class TestValidateGameOutput:
    """Test game output validation."""

    def test_valid_game_output(self):
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            "liquidity_map": {"equal_highs": [], "equal_lows": []},
            "nearest_unswept_pool": None,
            "trapped_positions": {},
            "forced_move_probability": 0.3,
            "nash_zones": [],
            "kyle_lambda": 0.5,
            "game_state_aligned": True,
            "game_state_summary": "Test summary",
            "adversarial_risk": "LOW",
        }
        validator = DataValidator()
        result = validator.validate_game_output(data)

        assert result.is_valid, f"Errors: {result.errors}"

    def test_invalid_adversarial_risk(self):
        data = {
            "symbol": "NAS100",
            "timestamp": "2024-01-15T14:30:00",
            "liquidity_map": {"equal_highs": [], "equal_lows": []},
            "nearest_unswept_pool": None,
            "trapped_positions": {},
            "forced_move_probability": 0.3,
            "nash_zones": [],
            "kyle_lambda": 0.5,
            "game_state_aligned": True,
            "game_state_summary": "Test",
            "adversarial_risk": "INVALID_RISK",
        }
        validator = DataValidator()
        result = validator.validate_game_output(data)

        assert not result.is_valid


class TestValidateEntrySignal:
    """Test entry signal validation."""

    def test_valid_entry_signal(self):
        data = {
            "symbol": "NAS100",
            "direction": 1,
            "entry_price": 20000.0,
            "position_size": 1000.0,
            "stop_loss": 19500.0,
            "tp1": 20500.0,
            "tp2": 21000.0,
            "confidence": 0.75,
            "rationale": ["TREND_STRENGTH"],
            "timestamp": "2024-01-15T14:30:00",
            "layer_context": {},
        }
        validator = DataValidator()
        result = validator.validate_entry_signal(data)

        assert result.is_valid, f"Errors: {result.errors}"


class TestValidatePosition:
    """Test position state validation."""

    def test_valid_position(self):
        data = {
            "trade_id": "trade_001",
            "symbol": "NAS100",
            "direction": 1,
            "entry_price": 20000.0,
            "position_size": 1000.0,
            "stop_loss": 19500.0,
            "tp1": 20500.0,
            "tp2": 21000.0,
            "current_price": 20100.0,
            "unrealized_pnl": 100.0,
            "realized_pnl": 0.0,
            "status": "OPEN",
            "opened_at": "2024-01-15T14:30:00",
        }
        validator = DataValidator()
        result = validator.validate_position(data)

        assert result.is_valid, f"Errors: {result.errors}"


class TestValidateAccountState:
    """Test account state validation."""

    def test_valid_account_state(self):
        data = {
            "account_id": "acc_001",
            "equity": 100000.0,
            "balance": 95000.0,
            "open_positions": 1,
            "daily_pnl": 500.0,
            "daily_loss_pct": 0.005,
            "margin_used": 10000.0,
            "margin_available": 40000.0,
            "timestamp": "2024-01-15T14:30:00",
        }
        validator = DataValidator()
        result = validator.validate_account_state(data)

        assert result.is_valid, f"Errors: {result.errors}"


class TestValidationStats:
    """Test validation statistics tracking."""

    def test_stats_tracking(self):
        validator = DataValidator()

        # Initial state
        stats = validator.get_stats()
        assert stats["total_validated"] == 0
        assert stats["valid_count"] == 0
        assert stats["invalid_count"] == 0

        # Validate some records
        validator.validate_feature_record({"symbol": "NAS100", "timeframe": "1h", "features": {}})  # Invalid

        # Load valid fixture for valid test
        with open(FIXTURES_DIR / "sample_feature_record.json") as f:
            valid_data = json.load(f)
        validator.validate_feature_record(valid_data)  # Valid

        stats = validator.get_stats()
        assert stats["total_validated"] == 2
        assert stats["invalid_count"] == 1

    def test_reset_stats(self):
        validator = DataValidator()
        validator.validate_feature_record({"invalid": "data"})

        validator.reset_stats()
        stats = validator.get_stats()
        assert stats["total_validated"] == 0


class TestValidateJsonFile:
    """Test JSON file validation."""

    def test_validate_valid_json_file(self):
        fixture_path = FIXTURES_DIR / "sample_feature_record.json"
        validator = DataValidator()
        result = validator.validate_json_file(fixture_path, "feature_record")

        assert result.is_valid

    def test_file_not_found(self):
        validator = DataValidator()
        result = validator.validate_json_file("nonexistent.json")

        assert not result.is_valid
        assert any("not found" in e.lower() for e in result.errors)

    def test_invalid_json(self, tmp_path):
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json")

        validator = DataValidator()
        result = validator.validate_json_file(invalid_file)

        assert not result.is_valid
        assert any("json" in e.lower() for e in result.errors)

    def test_unknown_record_type(self):
        fixture_path = FIXTURES_DIR / "sample_feature_record.json"
        validator = DataValidator()
        result = validator.validate_json_file(fixture_path, "unknown_type")

        assert not result.is_valid
        assert any("unknown" in e.lower() for e in result.errors)
