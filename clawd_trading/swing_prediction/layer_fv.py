"""
LAYER A — Fair Value Deviation
Produces z-score measuring how far asset price is from statistical fair value.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FairValueResult:
    """Result from fair value calculation."""

    symbol: str
    z_score: float  # negative = undervalued, positive = overvalued
    direction_bias: str  # 'LONG', 'SHORT', or 'NEUTRAL'
    signal_strength: str  # 'NONE' | 'WEAK' | 'MODERATE' | 'STRONG' | 'EXTREME'
    score: float  # Normalized score for composite (0-4 based on strength)
    methods_used: list
    component_scores: dict  # individual z-scores per method
    timestamp: str
    raw_values: Dict[str, float]


class FairValueLayer:
    """
    Layer A: Fair Value Deviation
    Routes to correct model based on asset class.
    """

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_a_fair_value"]

    async def compute(self, symbol: str, asset_class: str, data: Dict[str, Any]) -> FairValueResult:
        """
        Compute fair value z-score based on asset class.
        """
        self.logger.debug(f"Computing FV for {symbol} ({asset_class})")

        if asset_class == "equities":
            return self._compute_equity_fv(symbol, data)
        elif asset_class == "forex":
            return self._compute_forex_fv(symbol, data)
        elif asset_class == "commodities":
            return self._compute_commodity_fv(symbol, data)
        elif asset_class == "crypto":
            return self._compute_crypto_fv(symbol, data)
        else:
            self.logger.warning(f"Unknown asset class {asset_class}, using equity model")
            return self._compute_equity_fv(symbol, data)

    def _calculate_signal_strength(self, z_score: float) -> tuple[str, float]:
        """
        Calculate signal strength and composite score from z-score.
        Returns (strength_label, composite_score)
        """
        thresholds = self.config["z_score_thresholds"]
        abs_z = abs(z_score)

        if abs_z >= thresholds["extreme"]:
            return "EXTREME", 4.0
        elif abs_z >= thresholds["strong"]:
            return "STRONG", 3.0
        elif abs_z >= thresholds["moderate"]:
            return "MODERATE", 2.0
        elif abs_z >= thresholds["weak"]:
            return "WEAK", 1.0
        else:
            return "NONE", 0.0

    def _build_result(
        self,
        symbol: str,
        z_score: float,
        methods_used: list,
        component_scores: dict,
        raw_values: dict,
    ) -> FairValueResult:
        """Build FairValueResult with all metadata."""
        from datetime import datetime

        # Determine direction bias
        if z_score <= -self.config["z_score_thresholds"]["moderate"]:
            direction = "LONG"
        elif z_score >= self.config["z_score_thresholds"]["moderate"]:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # Calculate signal strength
        strength, score = self._calculate_signal_strength(z_score)

        return FairValueResult(
            symbol=symbol,
            z_score=round(z_score, 4),
            direction_bias=direction,
            signal_strength=strength,
            score=score,
            methods_used=methods_used,
            component_scores=component_scores,
            timestamp=datetime.now().isoformat(),
            raw_values=raw_values,
        )

    def _compute_equity_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Equity Fair Value Models:
        1. Price vs 200-week MA deviation (primary)
        2. Earnings yield vs bond yield spread (Fed Model)
        3. P/E z-score vs 5yr sector average
        """
        methods_used = []
        z_scores = []
        weights = []
        raw_values = {}

        # Method 1: Price vs 200-week MA
        if all(k in data for k in ["current_price", "ma_200w", "std_200w"]):
            deviation = (data["current_price"] - data["ma_200w"]) / data["std_200w"]
            z_scores.append(deviation)
            weights.append(self.config["equity_weights"]["ma200w_deviation"])
            methods_used.append("ma200w_deviation")
            raw_values["ma200w_zscore"] = deviation

        # Method 2: Fed Model (Earnings yield vs bond yield)
        if all(k in data for k in ["forward_pe", "treasury_10y_yield"]):
            earnings_yield = 1 / data["forward_pe"]
            spread = earnings_yield - data["treasury_10y_yield"]

            # Z-score of spread vs 5-year history
            if "spread_history_mean" in data and "spread_history_std" in data:
                spread_z = (spread - data["spread_history_mean"]) / data["spread_history_std"]
                z_scores.append(spread_z)
                weights.append(self.config["equity_weights"]["fed_model_spread"])
                methods_used.append("fed_model_spread")
                raw_values["fed_spread"] = spread
                raw_values["fed_spread_zscore"] = spread_z

        # Method 3: P/E z-score
        if all(k in data for k in ["current_pe", "sector_pe_mean_5yr", "sector_pe_std_5yr"]):
            pe_z = (data["current_pe"] - data["sector_pe_mean_5yr"]) / data["sector_pe_std_5yr"]
            z_scores.append(pe_z)
            weights.append(self.config["equity_weights"]["pe_zscore"])
            methods_used.append("pe_zscore")
            raw_values["pe_zscore"] = pe_z

        # Calculate weighted average z-score
        if not z_scores:
            self.logger.warning(f"No FV methods available for {symbol}")
            return FairValueResult(
                symbol=symbol,
                z_score=0.0,
                direction_bias="NEUTRAL",
                signal_strength="NONE",
                score=0.0,
                methods_used=[],
                component_scores={},
                timestamp=datetime.now().isoformat(),
                raw_values={},
            )

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))

        return self._build_result(
            symbol=symbol,
            z_score=weighted_z,
            methods_used=methods_used,
            component_scores=raw_values,
            raw_values=raw_values,
        )

    def _compute_forex_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Forex Fair Value Models:
        Method 1 — Real interest rate differential [Weight: 0.6]
        Method 2 — PPP deviation [Weight: 0.4]
        """
        methods_used = []
        z_scores = []
        weights = []
        component_scores = {}
        raw_values = {}

        # Method 1: Real rate differential
        # rate_diff = (base_rate - base_inflation) - (quote_rate - quote_inflation)
        if "real_rate_diff" in data:
            rate_diff = data["real_rate_diff"]
            raw_values["rate_diff"] = rate_diff

            if "rate_diff_3yr_mean" in data and "rate_diff_3yr_std" in data:
                z = (rate_diff - data["rate_diff_3yr_mean"]) / data["rate_diff_3yr_std"]
                z_scores.append(z)
                weights.append(0.6)  # Exact weight from spec
                methods_used.append("real_rate_differential")
                component_scores["real_rate_differential"] = round(z, 4)

        # Method 2: PPP deviation
        # ppp_rate from World Bank or OECD API (cached monthly)
        if "ppp_rate" in data and "current_rate" in data:
            ppp_rate = data["ppp_rate"]
            current_rate = data["current_rate"]
            raw_values["ppp_rate"] = ppp_rate
            raw_values["current_rate"] = current_rate

            if "ppp_5yr_std" in data:
                ppp_z = (current_rate - ppp_rate) / data["ppp_5yr_std"]
                z_scores.append(ppp_z)
                weights.append(0.4)  # Exact weight from spec
                methods_used.append("ppp_deviation")
                component_scores["ppp_deviation"] = round(ppp_z, 4)

        if not z_scores:
            return FairValueResult(
                symbol=symbol,
                z_score=0.0,
                direction_bias="NEUTRAL",
                signal_strength="NONE",
                score=0.0,
                methods_used=[],
                component_scores={},
                timestamp=datetime.now().isoformat(),
                raw_values={},
            )

        # Weighted average: [0.6, 0.4]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))

        return self._build_result(
            symbol=symbol,
            z_score=weighted_z,
            methods_used=methods_used,
            component_scores=component_scores,
            raw_values=raw_values,
        )

    def _compute_commodity_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Commodity Fair Value Models:
        Method 1 — Price vs cost of production [Weight: 0.4]
        Method 2 — Inventory vs seasonal norm [Weight: 0.4]
        Method 3 — Basis spread [Weight: 0.2]
        """
        methods_used = []
        z_scores = []
        weights = []
        component_scores = {}
        raw_values = {}

        # Method 1: Cost of production
        # z_score = (spot_price - production_cost) / std(spot_price, 5yr)
        if all(k in data for k in ["spot_price", "production_cost"]):
            spot = data["spot_price"]
            cost = data["production_cost"]
            raw_values["spot_price"] = spot
            raw_values["production_cost"] = cost

            if "spot_5yr_std" in data:
                z = (spot - cost) / data["spot_5yr_std"]
                z_scores.append(z)
                weights.append(0.4)  # Exact weight from spec
                methods_used.append("cost_of_production")
                component_scores["cost_of_production"] = round(z, 4)

        # Method 2: Inventory vs seasonal norm
        # inventory_deviation = (current - seasonal_mean) / seasonal_std
        # z_score = -1 * deviation (low inventory = price support = long signal)
        if all(k in data for k in ["current_inventory", "seasonal_mean", "seasonal_std"]):
            deviation = (data["current_inventory"] - data["seasonal_mean"]) / data["seasonal_std"]
            inv_z = -1 * deviation  # Invert: low inventory = bullish
            z_scores.append(inv_z)
            weights.append(0.4)  # Exact weight from spec
            methods_used.append("inventory_deviation")
            component_scores["inventory_deviation"] = round(inv_z, 4)
            raw_values["inventory_deviation"] = deviation

        # Method 3: Basis spread (spot vs front month futures)
        # z_score = (basis - mean_basis_3yr) / std_basis_3yr
        if "basis" in data and "basis_3yr_mean" in data and "basis_3yr_std" in data:
            basis_z = (data["basis"] - data["basis_3yr_mean"]) / data["basis_3yr_std"]
            z_scores.append(basis_z)
            weights.append(0.2)  # Exact weight from spec
            methods_used.append("basis_spread")
            component_scores["basis_spread"] = round(basis_z, 4)
            raw_values["basis"] = data["basis"]

        if not z_scores:
            return FairValueResult(
                symbol=symbol,
                z_score=0.0,
                direction_bias="NEUTRAL",
                signal_strength="NONE",
                score=0.0,
                methods_used=[],
                component_scores={},
                timestamp=datetime.now().isoformat(),
                raw_values={},
            )

        # Weighted average: [0.4, 0.4, 0.2]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))

        return self._build_result(
            symbol=symbol,
            z_score=weighted_z,
            methods_used=methods_used,
            component_scores=component_scores,
            raw_values=raw_values,
        )

    def _compute_crypto_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Crypto Fair Value Models:
        Method 1 — MVRV ratio [Weight: 0.4]
        Method 2 — NVT ratio [Weight: 0.3]
        Method 3 — Price vs realized price z-score [Weight: 0.3]
        """
        methods_used = []
        z_scores = []
        weights = []
        component_scores = {}
        raw_values = {}

        # Method 1: MVRV ratio
        # mvrv = market_cap / realized_cap (from Glassnode or CryptoQuant)
        if "mvrv_ratio" in data:
            mvrv = data["mvrv_ratio"]
            raw_values["mvrv_ratio"] = mvrv

            # z_score = (mvrv - mean(all_history)) / std(all_history)
            mvrv_mean = data.get("mvrv_all_history_mean", 1.0)
            mvrv_std = data.get("mvrv_all_history_std", 0.5)
            z = (mvrv - mvrv_mean) / mvrv_std
            z_scores.append(z)
            weights.append(0.4)  # Exact weight from spec
            methods_used.append("mvrv_ratio")
            component_scores["mvrv_ratio"] = round(z, 4)

        # Method 2: NVT ratio
        # nvt = market_cap / daily_transaction_volume_usd
        if "nvt_ratio" in data:
            nvt = data["nvt_ratio"]
            raw_values["nvt_ratio"] = nvt

            # z_score = (nvt - mean(2yr)) / std(2yr)
            nvt_mean = data.get("nvt_2yr_mean", 50)
            nvt_std = data.get("nvt_2yr_std", 20)
            z = (nvt - nvt_mean) / nvt_std
            z_scores.append(z)
            weights.append(0.3)  # Exact weight from spec
            methods_used.append("nvt_ratio")
            component_scores["nvt_ratio"] = round(z, 4)

        # Method 3: Price vs Realized Price
        # realized_price = realized_cap / coin_supply
        if all(k in data for k in ["price", "realized_price", "realized_diff_3yr_std"]):
            price = data["price"]
            realized = data["realized_price"]
            raw_values["price"] = price
            raw_values["realized_price"] = realized

            # z_score = (price - realized_price) / std(price - realized_price, 3yr)
            diff_z = (price - realized) / data["realized_diff_3yr_std"]
            z_scores.append(diff_z)
            weights.append(0.3)  # Exact weight from spec
            methods_used.append("price_vs_realized")
            component_scores["price_vs_realized"] = round(diff_z, 4)

        if not z_scores:
            return FairValueResult(
                symbol=symbol,
                z_score=0.0,
                direction_bias="NEUTRAL",
                signal_strength="NONE",
                score=0.0,
                methods_used=[],
                component_scores={},
                timestamp=datetime.now().isoformat(),
                raw_values={},
            )

        # Weighted average: [0.4, 0.3, 0.3]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))

        return self._build_result(
            symbol=symbol,
            z_score=weighted_z,
            methods_used=methods_used,
            component_scores=component_scores,
            raw_values=raw_values,
        )
