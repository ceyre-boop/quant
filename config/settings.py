"""
Central configuration for Clawd Trading.
Reads from .env file and provides typed config values.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv(".env")


class Config:
    """Central configuration class."""

    # Firebase
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_API_KEY: str = os.getenv("FIREBASE_API_KEY", "")
    FIREBASE_RTDB_URL: str = os.getenv("FIREBASE_RTDB_URL", "")

    # Alpaca
    ALPACA_API_KEY: Optional[str] = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY: Optional[str] = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Polygon
    POLYGON_API_KEY: Optional[str] = os.getenv("POLYGON_API_KEY")

    # Trading
    TRADE_SYMBOLS: list = os.getenv("TRADE_SYMBOLS", "SPY").split(",")
    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")

    # Account - CONFIGURABLE (was hardcoded to 100000)
    STARTING_EQUITY: float = float(os.getenv("STARTING_EQUITY", "100000.00"))
    CURRENCY: str = os.getenv("CURRENCY", "USD")

    # Risk
    RISK_PCT_PER_TRADE: float = float(os.getenv("RISK_PCT_PER_TRADE", "0.02"))
    MAX_CONCURRENT_POSITIONS: int = int(os.getenv("MAX_CONCURRENT_POSITIONS", "5"))
    DAILY_LOSS_LIMIT_PCT: float = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.03"))

    @classmethod
    def get_starting_equity(cls) -> float:
        """Get starting equity from config (not hardcoded)."""
        return cls.STARTING_EQUITY

    @classmethod
    def validate(cls) -> bool:
        """Validate that critical config is present."""
        required = [
            cls.FIREBASE_PROJECT_ID,
            cls.FIREBASE_API_KEY,
            cls.FIREBASE_RTDB_URL,
        ]
        return all(required)


# Convenience function for getting equity
def get_starting_equity() -> float:
    """Get starting equity from environment/config."""
    return Config.get_starting_equity()
