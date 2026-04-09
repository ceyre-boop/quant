"""
SWING PREDICTION LAYER — REMAINING WORK TODO
==============================================

This file contains all code that needs to be implemented to complete
the Swing Prediction Layer. Copy these functions into their respective files.

Status: ~70% Complete
- Layer A (Fair Value): ✅ Complete with exact formulas
- Layer B (Positioning): ✅ Complete with COT and equity formulas
- Layer C (Regime): ✅ Complete with Hurst R/S, ADX, HMM
- Layer D (Options): 📝 Needs full implementation below
- Layer E (Timing): 📝 Needs full implementation below
- Scorer: 📝 Needs conviction mapping
- Base Rate Calculator: 📝 Needs full implementation
- Firebase Writer: 📝 Needs full implementation
- SwingEngine: 📝 Needs _score_symbol updated for new schema

"""

# =============================================================================
# FILE: swing_prediction/layer_options.py
# =============================================================================

"""
LAYER D — OPTIONS MARKET SIGNALS
Purpose: Options markets encode institutional forward expectations.

GEX Formula:
  GEX = sum over all strikes of: gamma × open_interest × contract_multiplier × spot²
  
  Sign convention:
  - Calls: positive gamma exposure (dealers short calls = long gamma)
  - Puts: negative gamma exposure (dealers short puts = short gamma)
  - net_gex = sum(call_gex) - sum(put_gex)
  
  Interpretation:
  - net_gex > 0 (positive): dealers LONG gamma → sell rallies, buy dips → price pinned
  - net_gex < 0 (negative): dealers SHORT gamma → buy rallies, sell dips → price amplified
  - |gex| at key strike: large OI at one strike = gravitational pull or explosive level
  - gex_percentile = percentile_rank(net_gex, 1yr)

VIX Term Structure:
  term_spread = VIX3M - VIX (or VIX6M - VIX)
  - Contango: term_spread > 0 → normal, calm market
  - Backwardation: term_spread < 0 → fear, near-term uncertainty > long-term
  - Extreme backwardation (percentile < 5): Historically contrarian bullish
  - Extreme contango (percentile > 95): Complacency signal

IV/RV Spread:
  iv_rv_spread = implied_vol - realized_vol (20-day)
  - Options systematically overpriced
  - When spread compresses to near zero = something mispriced
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OptionsResult:
    """Output from options analysis."""

    symbol: str
    gex: float  # Net gamma exposure in billions
    gex_percentile: float
    gex_signal: str  # 'PINNED' | 'AMPLIFY' | 'NEUTRAL'
    vix_spot: float
    vix_term_spread: float
    vix_term_percentile: float
    vix_signal: str  # 'CONTANGO' | 'BACKWARDATION' | 'EXTREME_FEAR' | 'EXTREME_COMPLACENCY'
    iv_rank: float
    iv_signal: str  # 'EXPENSIVE' | 'CHEAP' | 'FAIR'
    iv_rv_spread: float
    iv_rv_signal: str  # 'OVERPRICED' | 'UNDERPRICED' | 'FAIR'
    expected_move_20d: float  # 1 SD expected move
    composite_score: float  # 0-1 normalized
    direction_bias: str
    timestamp: str


class OptionsLayer:
    """Layer D: Options Market as Forward Signal"""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_d_options"]

    async def compute(self, symbol: str, data: Dict[str, Any]) -> OptionsResult:
        """Analyze options market structure."""
        self.logger.debug(f"Analyzing options for {symbol}")

        # 1. Compute GEX
        gex_result = self.compute_gex(
            data.get("options_chain"),
            data.get("spot_price", 0),
            data.get("gex_history", []),
        )

        # 2. VIX Term Structure
        vix_result = self.compute_vix_term_structure(
            data.get("vix_spot", 20),
            data.get("vix_3m", 22),
            data.get("vix_history", []),
        )

        # 3. IV/RV Analysis
        iv_result = self.compute_iv_analysis(
            data.get("atm_iv", 0.20),
            data.get("realized_vol_20d", 0.18),
            data.get("iv_rank_history", []),
        )

        # Calculate expected move
        expected_move = self.calculate_expected_move(data.get("atm_iv", 0.20), days=20)

        # Composite scoring
        scores = []

        # GEX score (contrarian: negative GEX = bullish for reversion)
        if gex_result["gex"] < -1.0:
            scores.append(-2.0)  # Amplify moves = potential volatility
        elif gex_result["gex"] > 1.0:
            scores.append(1.0)  # Pinned = low volatility
        else:
            scores.append(0.0)

        # VIX score (contrarian: extreme backwardation = bullish)
        if vix_result["signal"] == "EXTREME_FEAR":
            scores.append(-2.0)
        elif vix_result["signal"] == "EXTREME_COMPLACENCY":
            scores.append(1.0)
        else:
            scores.append(0.0)

        # IV score (contrarian: expensive = sell, cheap = buy)
        if iv_result["iv_signal"] == "EXPENSIVE":
            scores.append(1.0)  # Expensive options = sell premium
        elif iv_result["iv_signal"] == "CHEAP":
            scores.append(-1.0)  # Cheap options = buy protection
        else:
            scores.append(0.0)

        composite = np.mean(scores) if scores else 0.0
        normalized_score = (composite + 3) / 6  # Scale to 0-1

        direction = "LONG" if composite < -1 else "SHORT" if composite > 1 else "NEUTRAL"

        return OptionsResult(
            symbol=symbol,
            gex=round(gex_result["gex"], 4),
            gex_percentile=round(gex_result["percentile"], 2),
            gex_signal=gex_result["signal"],
            vix_spot=round(vix_result["spot"], 2),
            vix_term_spread=round(vix_result["spread"], 4),
            vix_term_percentile=round(vix_result["percentile"], 2),
            vix_signal=vix_result["signal"],
            iv_rank=round(iv_result["iv_rank"], 2),
            iv_signal=iv_result["iv_signal"],
            iv_rv_spread=round(iv_result["iv_rv_spread"], 4),
            iv_rv_signal=iv_result["iv_rv_signal"],
            expected_move_20d=round(expected_move, 4),
            composite_score=round(normalized_score, 4),
            direction_bias=direction,
            timestamp=datetime.now().isoformat(),
        )

    def compute_gex(
        self,
        options_chain: Optional[pd.DataFrame],
        spot_price: float,
        gex_history: list,
    ) -> Dict:
        """
        Compute Gamma Exposure (GEX) from options chain.

        GEX = sum over all strikes of: gamma × open_interest × contract_multiplier × spot²

        Sign convention:
        - Calls: positive gamma exposure
        - Puts: negative gamma exposure
        - net_gex = sum(call_gex) - sum(put_gex)

        Data source: Polygon.io options chain (daily snapshot)
        """
        if options_chain is None or options_chain.empty or spot_price <= 0:
            return {"gex": 0.0, "percentile": 50.0, "signal": "NEUTRAL"}

        contract_multiplier = 100  # Standard equity multiplier

        # Calculate GEX for each option
        total_gex = 0.0

        for _, row in options_chain.iterrows():
            gamma = row.get("gamma", 0)
            oi = row.get("open_interest", 0)
            option_type = row.get("option_type", "call")

            # GEX formula
            gex = gamma * oi * contract_multiplier * (spot_price**2)

            # Sign: calls positive, puts negative
            if option_type == "put":
                gex = -gex

            total_gex += gex

        # Convert to billions for readability
        gex_billions = total_gex / 1e9

        # Calculate percentile
        if gex_history:
            below_current = sum(1 for g in gex_history if g < gex_billions)
            percentile = (below_current / len(gex_history)) * 100
        else:
            percentile = 50.0

        # Signal interpretation
        if gex_billions > 1.0:
            signal = "PINNED"  # Dealers long gamma = price suppression
        elif gex_billions < -1.0:
            signal = "AMPLIFY"  # Dealers short gamma = volatility amplification
        else:
            signal = "NEUTRAL"

        return {"gex": gex_billions, "percentile": percentile, "signal": signal}

    def compute_vix_term_structure(self, vix_spot: float, vix_3m: float, vix_history: list) -> Dict:
        """
        Compute VIX term structure analysis.

        term_spread = VIX3M - VIX
        - Contango: spread > 0 (normal)
        - Backwardation: spread < 0 (fear)
        - Extreme readings are contrarian
        """
        term_spread = vix_3m - vix_spot

        # Calculate percentile
        if vix_history and len(vix_history) > 100:
            below_current = sum(1 for v in vix_history if v < term_spread)
            percentile = (below_current / len(vix_history)) * 100
        else:
            percentile = 50.0

        # Signal interpretation
        if percentile < 5:
            signal = "EXTREME_FEAR"  # Extreme backwardation = contrarian bullish
        elif term_spread < -0.05:
            signal = "BACKWARDATION"
        elif percentile > 95:
            signal = "EXTREME_COMPLACENCY"
        elif term_spread > 0.05:
            signal = "CONTANGO"
        else:
            signal = "NORMAL"

        return {
            "spot": vix_spot,
            "spread": term_spread,
            "percentile": percentile,
            "signal": signal,
        }

    def compute_iv_analysis(self, atm_iv: float, realized_vol: float, iv_rank_history: list) -> Dict:
        """
        Analyze implied vs realized volatility.
        """
        # IV Rank
        if iv_rank_history:
            below_current = sum(1 for iv in iv_rank_history if iv < atm_iv)
            iv_rank = (below_current / len(iv_rank_history)) * 100
        else:
            iv_rank = 50.0

        # IV signal
        if iv_rank >= 80:
            iv_signal = "EXPENSIVE"
        elif iv_rank <= 20:
            iv_signal = "CHEAP"
        else:
            iv_signal = "FAIR"

        # IV/RV spread
        iv_rv_spread = atm_iv - realized_vol

        if iv_rv_spread > 0.15:
            iv_rv_signal = "OVERPRICED"
        elif iv_rv_spread < -0.05:
            iv_rv_signal = "UNDERPRICED"
        else:
            iv_rv_signal = "FAIR"

        return {
            "iv_rank": iv_rank,
            "iv_signal": iv_signal,
            "iv_rv_spread": iv_rv_spread,
            "iv_rv_signal": iv_rv_signal,
        }

    def calculate_expected_move(self, atm_iv: float, days: int = 20) -> float:
        """Calculate expected 1 SD move over N days."""
        return atm_iv * np.sqrt(days / 252)


# =============================================================================
# FILE: swing_prediction/layer_timing.py
# =============================================================================

"""
LAYER E — CALENDAR / FLOW TIMING
Purpose: Knowing what will move doesn't tell you when.

Key Events:
1. Monthly OpEx (3rd Friday):
   - Week before: Pin pressure (max pain)
   - Week of: Event risk
   - Week after: Pressure release (often explosive moves)

2. FOMC Cycle (8x per year):
   - 2 weeks before: Caution period
   - 1 week before: Elevated vol
   - Week of: Event risk
   - Week after: FOMC drift (typically bullish)

3. Quarterly Rebalancing:
   - Last 2 weeks of quarter: Institutional flow pressure
   - First week of new quarter: Fresh allocations

4. Earnings Season:
   - Pre-season: Vol compression (selling premium)
   - Active: Event risk
   - Post-season: Vol expansion
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class TimingResult:
    """Output from timing analysis."""

    symbol: str
    opex_window: str  # 'BEFORE' | 'DURING' | 'AFTER' | 'NEUTRAL'
    fomc_window: str  # 'PRE_2W' | 'PRE_1W' | 'DURING' | 'POST' | 'NEUTRAL'
    quarter_position: str  # 'LAST_2W' | 'LAST_1W' | 'FIRST_WEEK' | 'NEUTRAL'
    earnings_season: str  # 'PRE' | 'ACTIVE' | 'POST' | 'NEUTRAL'
    composite_score: float  # -1 to +1
    signal: str  # 'FAVORABLE' | 'CAUTION' | 'NEUTRAL'
    days_to_next_opex: int
    days_to_next_fomc: int
    timestamp: str


class TimingLayer:
    """Layer E: Calendar and Flow Timing"""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_e_timing"]
        # Cache FOMC dates (would be loaded from config or external source)
        self.fomc_dates = self._load_fomc_dates()

    def _load_fomc_dates(self) -> list:
        """Load FOMC meeting dates for current year."""
        # In production, load from Fed calendar API
        # Placeholder: return empty list, implement caching
        return []

    async def compute(self, symbol: str) -> TimingResult:
        """Determine current timing window."""
        now = datetime.now()

        # OpEx window
        opex = self.get_opex_window(now)
        days_to_opex = self.days_to_next_opex(now)

        # FOMC window
        fomc = self.get_fomc_window(now)
        days_to_fomc = self.days_to_next_fomc(now)

        # Quarter position
        quarter = self.get_quarter_position(now)

        # Earnings season
        earnings = self.get_earnings_season(now)

        # Score calculation
        scores = self.config["timing_scores"]

        opex_score = scores["opex"].get(opex, 0)
        fomc_score = scores["fomc"].get(fomc, 0)
        quarter_score = scores["quarter_end"].get(quarter, 0)
        earnings_score = scores["earnings"].get(earnings, 0)

        # Weighted composite
        weights = self.config["timing_weights"]
        composite = (
            opex_score * weights["opex"]
            + fomc_score * weights["fomc"]
            + quarter_score * weights["quarter"]
            + earnings_score * weights["earnings"]
        )

        # Signal interpretation
        if composite > 0.3:
            signal = "FAVORABLE"
        elif composite < -0.3:
            signal = "CAUTION"
        else:
            signal = "NEUTRAL"

        return TimingResult(
            symbol=symbol,
            opex_window=opex,
            fomc_window=fomc,
            quarter_position=quarter,
            earnings_season=earnings,
            composite_score=round(composite, 4),
            signal=signal,
            days_to_next_opex=days_to_opex,
            days_to_next_fomc=days_to_fomc,
            timestamp=datetime.now().isoformat(),
        )

    def get_opex_window(self, date: datetime) -> str:
        """
        Determine OpEx window.
        Monthly OpEx is the 3rd Friday.
        """
        # Find 3rd Friday of current month
        third_friday = self._find_third_friday(date.year, date.month)

        days_diff = (date.date() - third_friday.date()).days

        if -7 <= days_diff < 0:
            return "BEFORE"
        elif days_diff == 0:
            return "DURING"
        elif 0 < days_diff <= 7:
            return "AFTER"
        else:
            return "NEUTRAL"

    def _find_third_friday(self, year: int, month: int) -> datetime:
        """Find the 3rd Friday of a given month."""
        import calendar

        # Get first day of month
        first_day = datetime(year, month, 1)

        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        # Third Friday is 2 weeks after first Friday
        third_friday = first_friday + timedelta(weeks=2)

        return third_friday

    def days_to_next_opex(self, date: datetime) -> int:
        """Calculate days until next monthly OpEx."""
        # Implementation: find next 3rd Friday
        current_month_opex = self._find_third_friday(date.year, date.month)

        if date.date() < current_month_opex.date():
            return (current_month_opex.date() - date.date()).days
        else:
            # Move to next month
            if date.month == 12:
                next_month_opex = self._find_third_friday(date.year + 1, 1)
            else:
                next_month_opex = self._find_third_friday(date.year, date.month + 1)
            return (next_month_opex.date() - date.date()).days

    def get_fomc_window(self, date: datetime) -> str:
        """Determine FOMC cycle position."""
        if not self.fomc_dates:
            return "NEUTRAL"

        # Find next FOMC meeting
        next_fomc = None
        for fomc_date in self.fomc_dates:
            if fomc_date > date:
                next_fomc = fomc_date
                break

        if not next_fomc:
            return "NEUTRAL"

        days_to_fomc = (next_fomc - date).days

        if 7 < days_to_fomc <= 14:
            return "PRE_2W"
        elif 0 < days_to_fomc <= 7:
            return "PRE_1W"
        elif days_to_fomc == 0:
            return "DURING"
        elif -7 <= days_to_fomc < 0:
            return "POST"
        else:
            return "NEUTRAL"

    def days_to_next_fomc(self, date: datetime) -> int:
        """Calculate days until next FOMC meeting."""
        if not self.fomc_dates:
            return 999

        for fomc_date in self.fomc_dates:
            if fomc_date > date:
                return (fomc_date - date).days

        return 999

    def get_quarter_position(self, date: datetime) -> str:
        """Determine position relative to quarter end."""
        month = date.month
        day = date.day

        # Quarter end months: Mar (3), Jun (6), Sep (9), Dec (12)
        quarter_ends = [3, 6, 9, 12]

        if month in quarter_ends:
            if day >= 24:
                return "LAST_1W"
            elif day >= 15:
                return "LAST_2W"
        elif month in [m + 1 for m in quarter_ends if m < 12]:
            if day <= 7:
                return "FIRST_WEEK"

        return "NEUTRAL"

    def get_earnings_season(self, date: datetime) -> str:
        """Determine earnings season phase."""
        month = date.month

        # Earnings seasons: Jan, Apr, Jul, Oct
        peak_months = [1, 4, 7, 10]

        if month in peak_months:
            return "ACTIVE"
        elif month in [m - 1 for m in peak_months if m > 1]:
            return "PRE"
        elif month in [m + 1 for m in peak_months if m < 12]:
            return "POST"
        else:
            return "NEUTRAL"


# =============================================================================
# FILE: swing_prediction/backtest_base_rates.py
# =============================================================================

"""
HISTORICAL BASE RATE CALCULATOR
Purpose: Before any live signal is acted upon, validate it against history.

Algorithm:
1. For each historical date (go back N years):
   a. Compute all five layer scores at that date (no lookahead)
   b. Check if composite_score > threshold AND direction matches
   c. If yes: record as historical occurrence

2. For each occurrence, compute forward returns at 20, 40, 60 days:
   forward_return_Nd = (price[t+N] - price[t]) / price[t]

3. Compute statistics per window:
   - win_rate_Nd = count(return > 0) / total
   - avg_return_Nd = mean(return)
   - median_return_Nd = median(return)
   - max_drawdown_Nd = max drawdown per occurrence
   - sharpe_Nd = mean / std * sqrt(252/N)

4. Store in Firebase: swing_base_rates/{symbol}_{conditions_hash}
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BaseRateResult:
    """Historical base rate statistics."""

    symbol: str
    conditions_hash: str
    n_occurrences: int
    date_range: str
    forward_20d: Dict[str, float]
    forward_40d: Dict[str, float]
    forward_60d: Dict[str, float]
    edge_confirmed: bool
    computed_at: str


class BaseRateCalculator:
    """Calculate historical win rates for signal combinations."""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["base_rates"]

    async def compute_base_rates(
        self,
        symbol: str,
        conditions: Dict,
        price_history: pd.DataFrame,
        indicator_history: pd.DataFrame,
        forward_windows: List[int] = [20, 40, 60],
    ) -> BaseRateResult:
        """
        Compute historical base rates for given conditions.

        This is the main entry point for calculating base rates.
        """
        lookback_years = self.config["lookback_years"]
        min_occurrences = self.config["min_occurrences"]
        min_win_rate = self.config["min_win_rate"]

        # Filter to lookback period
        cutoff_date = datetime.now() - timedelta(days=365 * lookback_years)
        price_hist = price_history[price_history.index >= cutoff_date]

        if len(price_hist) < 252 * 2:  # Need at least 2 years
            self.logger.warning(f"Insufficient price history for {symbol}")
            return self._empty_result(symbol, conditions)

        # Find historical occurrences
        occurrences = self._find_occurrences(symbol, conditions, price_hist, indicator_history)

        if len(occurrences) < min_occurrences:
            self.logger.info(f"Insufficient occurrences for {symbol}: {len(occurrences)}")
            return self._empty_result(symbol, conditions)

        # Calculate forward returns for each window
        results = {}
        for window in forward_windows:
            stats = self._calculate_forward_stats(price_hist, occurrences, window)
            results[f"forward_{window}d"] = stats

        # Determine if edge is confirmed
        win_rate_40d = results["forward_40d"]["win_rate"]
        edge_confirmed = win_rate_40d > min_win_rate and len(occurrences) >= min_occurrences

        # Generate conditions hash
        conditions_hash = self._hash_conditions(conditions)

        return BaseRateResult(
            symbol=symbol,
            conditions_hash=conditions_hash,
            n_occurrences=len(occurrences),
            date_range=f"{price_hist.index[0]} to {price_hist.index[-1]}",
            forward_20d=results["forward_20d"],
            forward_40d=results["forward_40d"],
            forward_60d=results["forward_60d"],
            edge_confirmed=edge_confirmed,
            computed_at=datetime.now().isoformat(),
        )

    def _find_occurrences(
        self,
        symbol: str,
        conditions: Dict,
        price_hist: pd.DataFrame,
        indicator_hist: pd.DataFrame,
    ) -> List[datetime]:
        """
        Find all historical dates where conditions matched.

        This requires recalculating all 5 layers at each historical date
        using only data available at that time (no lookahead bias).
        """
        occurrences = []

        # Iterate through historical dates
        # Skip first 252 days (need 1 year of history for indicators)
        for date in price_hist.index[252:]:
            try:
                # Check if conditions match at this date
                if self._check_conditions_at_date(conditions, date, price_hist, indicator_hist):
                    occurrences.append(date)
            except Exception as e:
                self.logger.debug(f"Error checking {date}: {e}")
                continue

        return occurrences

    def _check_conditions_at_date(
        self,
        conditions: Dict,
        date: datetime,
        price_hist: pd.DataFrame,
        indicator_hist: pd.DataFrame,
    ) -> bool:
        """
        Check if conditions match at a specific historical date.

        TODO: Implement full historical layer calculation
        For now, simplified check based on price and basic indicators.
        """
        # This is a placeholder - full implementation would:
        # 1. Calculate Layer A (Fair Value) at date
        # 2. Calculate Layer B (Positioning) at date
        # 3. Calculate Layer C (Regime) at date
        # 4. Calculate Layer D (Options) at date
        # 5. Calculate Layer E (Timing) at date
        # 6. Check if composite meets threshold

        # Simplified: check if price is above/below MA
        price = price_hist.loc[date, "close"]
        ma_200 = price_hist.loc[:date, "close"].rolling(200).mean().iloc[-1]

        direction = conditions.get("direction", "LONG")

        if direction == "LONG":
            return price < ma_200 * 0.95  # 5% below 200 MA
        else:
            return price > ma_200 * 1.05  # 5% above 200 MA

    def _calculate_forward_stats(self, price_hist: pd.DataFrame, occurrences: List[datetime], window: int) -> Dict[str, float]:
        """Calculate forward return statistics for given window."""
        returns = []
        drawdowns = []

        for date in occurrences:
            try:
                # Get entry price
                entry_idx = price_hist.index.get_loc(date)
                entry_price = price_hist.iloc[entry_idx]["close"]

                # Get exit price (window days later)
                exit_idx = entry_idx + window
                if exit_idx >= len(price_hist):
                    continue

                exit_price = price_hist.iloc[exit_idx]["close"]

                # Calculate return
                ret = (exit_price - entry_price) / entry_price
                returns.append(ret)

                # Calculate max drawdown over holding period
                holding_prices = price_hist.iloc[entry_idx:exit_idx]["close"]
                cummax = holding_prices.cummax()
                drawdown = (holding_prices - cummax) / cummax
                max_dd = drawdown.min()
                drawdowns.append(max_dd)

            except Exception as e:
                self.logger.debug(f"Error calculating forward return: {e}")
                continue

        if not returns:
            return {
                "win_rate": 0.0,
                "avg_return": 0.0,
                "median_return": 0.0,
                "avg_max_drawdown": 0.0,
                "sharpe": 0.0,
            }

        returns_arr = np.array(returns)

        # Calculate Sharpe (annualized)
        if len(returns_arr) > 1 and np.std(returns_arr) > 0:
            sharpe = (np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(252 / window)
        else:
            sharpe = 0.0

        return {
            "win_rate": round(np.mean(returns_arr > 0), 4),
            "avg_return": round(np.mean(returns_arr), 4),
            "median_return": round(np.median(returns_arr), 4),
            "avg_max_drawdown": round(np.mean(drawdowns), 4),
            "sharpe": round(sharpe, 4),
        }

    def _hash_conditions(self, conditions: Dict) -> str:
        """Generate hash of conditions for caching."""
        conditions_str = json.dumps(conditions, sort_keys=True)
        return hashlib.md5(conditions_str.encode()).hexdigest()[:16]

    def _empty_result(self, symbol: str, conditions: Dict) -> BaseRateResult:
        """Return empty result when insufficient data."""
        return BaseRateResult(
            symbol=symbol,
            conditions_hash=self._hash_conditions(conditions),
            n_occurrences=0,
            date_range="",
            forward_20d={
                "win_rate": 0,
                "avg_return": 0,
                "median_return": 0,
                "avg_max_drawdown": 0,
                "sharpe": 0,
            },
            forward_40d={
                "win_rate": 0,
                "avg_return": 0,
                "median_return": 0,
                "avg_max_drawdown": 0,
                "sharpe": 0,
            },
            forward_60d={
                "win_rate": 0,
                "avg_return": 0,
                "median_return": 0,
                "avg_max_drawdown": 0,
                "sharpe": 0,
            },
            edge_confirmed=False,
            computed_at=datetime.now().isoformat(),
        )

    async def get_base_rate(self, symbol: str, layer_scores: Dict, direction: str) -> Optional[Dict]:
        """Get cached base rate or compute new one."""
        # TODO: Check Firebase cache first
        # If not cached or stale, call compute_base_rates

        # Placeholder: return estimated values
        aligned = sum(1 for s in layer_scores.values() if abs(s) >= 0.5)

        base_rates = {
            5: {"win_rate": 0.72, "avg_return": 0.042, "max_dd": 0.021},
            4: {"win_rate": 0.68, "avg_return": 0.038, "max_dd": 0.025},
            3: {"win_rate": 0.58, "avg_return": 0.028, "max_dd": 0.035},
        }

        if aligned < 3:
            return None

        rates = base_rates.get(aligned, base_rates[3])

        return {
            "win_rate": rates["win_rate"],
            "avg_return_20d": rates["avg_return"],
            "avg_return_40d": rates["avg_return"] * 1.2,
            "avg_return_60d": rates["avg_return"] * 1.3,
            "max_drawdown": rates["max_dd"],
            "n_occurrences": max(10, aligned * 5),
            "confidence": rates["win_rate"],
        }


# =============================================================================
# FILE: swing_prediction/firebase_writer.py
# =============================================================================

"""
FIREBASE WRITER — Persist Swing Prediction Outputs
Purpose: Write all swing prediction results to Firebase for:
- Intraday system access (swing_bias gate)
- Historical analysis
- Base rate caching
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class FirebaseWriter:
    """Handles all Firebase writes for swing prediction layer."""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["firebase"]
        self.db = None  # Initialize Firebase client here
        self._init_firebase()

    def _init_firebase(self):
        """Initialize Firebase connection."""
        try:
            import firebase_admin
            from firebase_admin import firestore

            # Check if already initialized
            if not firebase_admin._apps:
                # Would load from service account or use default
                firebase_admin.initialize_app()

            self.db = firestore.client()
            self.logger.info("Firebase initialized")
        except Exception as e:
            self.logger.error(f"Firebase init failed: {e}")
            self.db = None

    async def write_swing_bias(self, bias: Any) -> bool:
        """
        Write SwingBias to Firebase.

        Collection: swing_bias
        Document ID: {symbol}_{scan_date}
        TTL: 90 days (handled by Firebase TTL policy)
        """
        try:
            if not self.db:
                self.logger.warning("Firebase not available, skipping write")
                return False

            collection = self.config["collection_swing_bias"]
            doc_id = f"{bias.symbol}_{bias.scan_date[:10]}"  # YYYY-MM-DD

            data = bias.to_dict()

            # Add TTL field (90 days from created_at)
            created = datetime.fromisoformat(bias.created_at.replace("Z", "+00:00"))
            ttl = created + timedelta(days=90)
            data["expires_at"] = ttl.isoformat()

            self.db.collection(collection).document(doc_id).set(data)

            self.logger.debug(f"Wrote swing bias for {bias.symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write swing bias: {e}")
            return False

    async def get_latest_swing_bias(self, symbol: str) -> Optional[Dict]:
        """
        Get most recent SwingBias for symbol.

        Used by intraday system to check tradeability gate.
        """
        try:
            if not self.db:
                return None

            collection = self.config["collection_swing_bias"]

            # Query for most recent document for this symbol
            query = (
                self.db.collection(collection)
                .where("symbol", "==", symbol)
                .order_by("created_at", direction="DESCENDING")
                .limit(1)
            )

            docs = query.get()

            if docs:
                return docs[0].to_dict()
            return None

        except Exception as e:
            self.logger.error(f"Failed to get swing bias: {e}")
            return None

    async def get_tradeable_symbols(self) -> List[str]:
        """
        Get all symbols currently marked as tradeable.

        Used by intraday system to filter symbol universe.
        """
        try:
            if not self.db:
                return []

            collection = self.config["collection_swing_bias"]

            # Query for tradeable symbols from recent scans
            query = (
                self.db.collection(collection)
                .where("tradeable", "==", True)
                .where("created_at", ">=", (datetime.now() - timedelta(days=7)).isoformat())
            )

            docs = query.get()

            # Get unique symbols
            symbols = set()
            for doc in docs:
                data = doc.to_dict()
                symbols.add(data.get("symbol"))

            return list(symbols)

        except Exception as e:
            self.logger.error(f"Failed to get tradeable symbols: {e}")
            return []

    async def write_scan_log(self, log: Dict[str, Any]) -> bool:
        """Write scan execution log."""
        try:
            if not self.db:
                return False

            collection = self.config["collection_scan_log"]

            self.db.collection(collection).add(log)

            self.logger.debug(f"Wrote scan log")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write scan log: {e}")
            return False

    async def write_base_rate(self, result: Any) -> bool:
        """
        Write calculated base rate to Firebase.

        Collection: swing_base_rates
        Document ID: {symbol}_{conditions_hash}
        """
        try:
            if not self.db:
                return False

            collection = self.config["collection_base_rates"]
            doc_id = f"{result.symbol}_{result.conditions_hash}"

            data = {
                "symbol": result.symbol,
                "conditions_hash": result.conditions_hash,
                "n_occurrences": result.n_occurrences,
                "date_range": result.date_range,
                "forward_20d": result.forward_20d,
                "forward_40d": result.forward_40d,
                "forward_60d": result.forward_60d,
                "edge_confirmed": result.edge_confirmed,
                "computed_at": result.computed_at,
            }

            self.db.collection(collection).document(doc_id).set(data)

            self.logger.debug(f"Wrote base rate for {result.symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write base rate: {e}")
            return False

    async def get_cached_base_rate(self, symbol: str, conditions_hash: str) -> Optional[Dict]:
        """Get cached base rate if not stale."""
        try:
            if not self.db:
                return None

            collection = self.config["collection_base_rates"]
            doc_id = f"{symbol}_{conditions_hash}"

            doc = self.db.collection(collection).document(doc_id).get()

            if doc.exists:
                data = doc.to_dict()

                # Check if stale (> 30 days old)
                computed = datetime.fromisoformat(data.get("computed_at", "").replace("Z", "+00:00"))
                if datetime.now() - computed > timedelta(days=30):
                    return None  # Stale, needs recalculation

                return data

            return None

        except Exception as e:
            self.logger.error(f"Failed to get cached base rate: {e}")
            return None


# =============================================================================
# INTEGRATION: Daily Lifecycle Update
# =============================================================================

"""
Add this to orchestrator/daily_lifecycle.py at the top of run_premarket():

```python
from swing_prediction import get_swing_engine

async def run_premarket(symbol: str):
    # Check swing prediction gate
    swing_engine = get_swing_engine()
    swing_bias = await swing_engine.firebase_writer.get_latest_swing_bias(symbol)
    
    if swing_bias and not swing_bias.get("tradeable", True):
        log(f'SWING_LAYER_BLOCK: {symbol} not tradeable this month — '
            f'reason: {swing_bias.get("block_reason", "unknown")}')
        return None
    
    # Continue with existing three-layer intraday system...
    context = await build_context(symbol)
    ...
```
"""


# =============================================================================
# CONFIGURATION UPDATES NEEDED
# =============================================================================

"""
Add to config/swing_params.json:

{
  "timing_scores": {
    "opex": {
      "BEFORE": -0.5,
      "DURING": 0.0,
      "AFTER": 1.0,
      "NEUTRAL": 0.0
    },
    "fomc": {
      "PRE_2W": -0.5,
      "PRE_1W": -1.0,
      "DURING": -0.5,
      "POST": 1.0,
      "NEUTRAL": 0.0
    },
    "quarter_end": {
      "LAST_2W": -0.5,
      "LAST_1W": -1.0,
      "FIRST_WEEK": 0.5,
      "NEUTRAL": 0.0
    },
    "earnings": {
      "PRE": -0.3,
      "ACTIVE": 0.0,
      "POST": 0.3,
      "NEUTRAL": 0.0
    }
  },
  "timing_weights": {
    "opex": 0.35,
    "fomc": 0.35,
    "quarter": 0.20,
    "earnings": 0.10
  },
  "base_rates": {
    "lookback_years": 10,
    "min_occurrences": 15,
    "min_win_rate": 0.55
  }
}
"""


# =============================================================================
# DEPENDENCIES TO ADD TO requirements.txt
# =============================================================================

"""
hmmlearn>=0.3.0
pandas>=2.0.0
numpy>=1.24.0
firebase-admin>=6.0.0
"""
