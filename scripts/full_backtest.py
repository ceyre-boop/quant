"""
Full Statistical Backtest with Chi-Squared Gates

Runs 3-layer + 5-layer stack through history with chi-squared validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.alpaca_client import AlpacaDataClient
from training.feature_generator import FeatureGenerator
from walk_forward_validation import chi_squared_test


@dataclass
class Trade:
    """Single trade record"""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    confidence: float
    pnl: float
    pnl_pct: float
    exit_reason: str


class ChiSquaredGate:
    """
    Chi-squared gate: Only allow trades from confidence buckets
    that have passed statistical validation.
    """

    def __init__(self, min_confidence: float = 0.55, chi2_threshold: float = 0.05):
        self.min_confidence = min_confidence
        self.chi2_threshold = chi2_threshold
        self.validated_buckets = set()
        self.bucket_history = {}

    def update(self, predictions: np.ndarray, actuals: np.ndarray):
        """Update gate with recent predictions"""
        chi2_result = chi_squared_test(predictions, actuals, confidence_buckets=5)

        if chi2_result["significant"]:
            # Mark high-confidence buckets as validated
            for bucket in chi2_result["buckets"]:
                if bucket["actual_up_rate"] > 0.55:  # Better than random
                    self.validated_buckets.add(bucket["bucket"])

    def allow_trade(self, confidence: float, direction: str) -> bool:
        """Check if trade passes the chi-squared gate"""
        if confidence < self.min_confidence:
            return False

        # Map confidence to bucket
        bucket_idx = min(int(confidence * 5), 4)
        bucket_label = f"{bucket_idx*20}-{(bucket_idx+1)*20}%"

        # Only trade if bucket validated
        return bucket_label in self.validated_buckets


class StatisticalBacktest:
    """
    Full backtest with chi-squared gates and layer validation.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1D",
        initial_capital: float = 100000,
        position_size: float = 0.1,  # 10% per trade
        chi2_gate: bool = True,
    ):
        self.symbols = symbols
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.use_chi2_gate = chi2_gate

        self.client = AlpacaDataClient()
        self.feature_gen = FeatureGenerator()
        self.chi2_gate = ChiSquaredGate()

        # Results
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []

    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols"""
        print("Fetching historical data...")

        data = {}
        days = (self.end_date - self.start_date).days + 365  # Extra for feature calc

        for symbol in self.symbols:
            try:
                df = self.client.get_historical_bars(symbol, timeframe=self.timeframe, days=days)
                # Filter to date range
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                if len(df) > 50:
                    data[symbol] = df
                    print(f"  {symbol}: {len(df)} bars")
            except Exception as e:
                print(f"  {symbol}: Failed ({e})")

        return data

    def run_backtest(self) -> Dict:
        """
        Run full backtest.

        Returns:
            Dict with performance metrics
        """
        print("=" * 70)
        print("STATISTICAL BACKTEST")
        print("=" * 70)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Chi-Squared Gate: {'ON' if self.use_chi2_gate else 'OFF'}")
        print()

        # Fetch data
        data = self.fetch_historical_data()

        if not data:
            print("ERROR: No data fetched")
            return None

        # Load or train model
        model = self._load_model()

        # Run simulation
        capital = self.initial_capital
        self.equity_curve = [capital]

        # Get all trading days from first symbol
        first_symbol = list(data.keys())[0]
        trading_days = data[first_symbol].index

        print(f"\nSimulating {len(trading_days)} trading days...")

        for i, date in enumerate(trading_days):
            if i % 50 == 0:
                print(f"  Day {i}/{len(trading_days)}: ${capital:,.2f}")

            daily_pnl = 0

            # Check each symbol
            for symbol, df in data.items():
                if date not in df.index:
                    continue

                # Get data up to current date
                current_idx = df.index.get_loc(date)
                if current_idx < 30:
                    continue

                hist = df.iloc[: current_idx + 1]

                # Generate features and predict
                signal = self._generate_signal(symbol, hist, model)

                if signal and self._validate_signal(signal):
                    # Simulate trade
                    trade_pnl = self._simulate_trade(symbol, signal, df, current_idx)
                    daily_pnl += trade_pnl

            capital += daily_pnl
            self.equity_curve.append(capital)

            if i > 0:
                daily_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
                self.daily_returns.append(daily_return)

        # Calculate metrics
        return self._calculate_metrics()

    def _load_model(self):
        """Load or train XGBoost model"""
        from xgboost import XGBClassifier

        model_path = Path(__file__).parent / "training" / "xgb_model.pkl"

        if model_path.exists():
            import pickle

            with open(model_path, "rb") as f:
                saved = pickle.load(f)
                print("Loaded pre-trained model")
                return saved["model"]
        else:
            print("No model found - would need to train")
            return None

    def _generate_signal(self, symbol: str, df: pd.DataFrame, model) -> Dict:
        """Generate trading signal"""
        # Generate features
        features_df = self.feature_gen.generate_features(df)

        if features_df.empty or len(features_df) < 2:
            return None

        # Get latest
        latest = features_df.iloc[-1]

        # Build feature vector (simplified - would use actual feature cols)
        feature_cols = [c for c in features_df.columns if c not in ["open", "high", "low", "close", "volume"]]

        X = latest[feature_cols].fillna(0).values.reshape(1, -1)

        # Predict
        prob = model.predict_proba(X)[0]

        return {
            "symbol": symbol,
            "confidence": prob[1],
            "direction": "LONG" if prob[1] > 0.5 else "SHORT",
        }

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate signal through chi-squared gate"""
        if not self.use_chi2_gate:
            return signal["confidence"] > 0.55

        return self.chi2_gate.allow_trade(signal["confidence"], signal["direction"])

    def _simulate_trade(self, symbol: str, signal: Dict, df: pd.DataFrame, entry_idx: int) -> float:
        """Simulate a trade and return P&L"""
        # Simplified: 1-day hold for now
        entry_price = df["close"].iloc[entry_idx]

        if entry_idx + 1 >= len(df):
            return 0

        exit_price = df["close"].iloc[entry_idx + 1]

        position_value = self.initial_capital * self.position_size
        shares = position_value / entry_price

        if signal["direction"] == "LONG":
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        return pnl

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        returns = np.array(self.daily_returns)
        equity = np.array(self.equity_curve)

        total_return = (equity[-1] / equity[0]) - 1

        if len(returns) < 2:
            return {"error": "Insufficient returns data"}

        # Risk metrics
        sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Win rate
        wins = (returns > 0).sum()
        losses = (returns < 0).sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Annualized Return: {(total_return / len(returns) * 252)*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"Win Rate (daily): {win_rate*100:.2f}%")
        print(f"Final Equity: ${equity[-1]:,.2f}")
        print(f"Trades: {len(self.trades)}")

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "final_equity": equity[-1],
            "equity_curve": equity.tolist(),
            "daily_returns": returns.tolist(),
        }


if __name__ == "__main__":
    # Run 5-year backtest
    backtest = StatisticalBacktest(
        symbols=["SPY", "QQQ"],
        start_date="2020-01-01",
        end_date="2025-01-01",
        timeframe="1D",
        initial_capital=100000,
        chi2_gate=True,
    )

    results = backtest.run_backtest()
