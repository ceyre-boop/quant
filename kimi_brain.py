"""
KimiBrain - LLM-Powered Trading with Learning

Uses Kimi AI to analyze market data and make trading decisions.
Learns from trade outcomes to improve prompts and reasoning.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Load env vars before imports that need them
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import requests
import pandas as pd
import numpy as np

from ai_trading_bridge import AIBrain, Signal, AITradingBridge
from data.alpaca_client import AlpacaDataClient

logger = logging.getLogger(__name__)


@dataclass
class KimiTradeMemory:
    """Memory of a trade for learning"""

    symbol: str
    timestamp: str
    prompt: str
    reasoning: str
    decision: str  # LONG/SHORT/FLAT
    confidence: float
    actual_return: Optional[float] = None
    won: Optional[bool] = None
    feedback: str = ""  # What Kimi learned from this trade


class KimiBrain(AIBrain):
    """
    Kimi LLM-powered trading brain that learns from outcomes.

    Unlike XGBoost, this uses reasoning and can explain decisions.
    It maintains a memory of trades and learns what works.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "moonshot-v1-8k",  # Kimi model name
        temperature: float = 0.3,  # Lower = more consistent
        max_tokens: int = 2000,
        learning_mode: bool = True,
    ):
        self.api_key = api_key or os.getenv("KIMI_API_KEY")
        if not self.api_key:
            raise ValueError("KIMI_API_KEY required")

        # Kimi API base URL (without /v1 - that's part of the endpoint path)
        self.base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn")
        # Kimi model name - use official model identifier
        self.model = model  # Try: "moonshot-v1-8k" or "moonshot-v1-32k"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.learning_mode = learning_mode

        self.client = AlpacaDataClient()
        self.memory: List[KimiTradeMemory] = []
        self._name = "KimiBrain-v1.0"

        # Load previous memories
        self._load_memory()

    @property
    def name(self) -> str:
        return self._name

    def _load_memory(self):
        """Load trade memories from disk"""
        memory_path = Path("logs/kimi_memory.json")
        if memory_path.exists():
            with open(memory_path, "r") as f:
                data = json.load(f)
                self.memory = [KimiTradeMemory(**m) for m in data]
            logger.info("[KimiBrain] Loaded %d trade memories", len(self.memory))

    def _save_memory(self):
        """Save trade memories to disk"""
        memory_path = Path("logs/kimi_memory.json")
        memory_path.parent.mkdir(exist_ok=True)
        with open(memory_path, "w") as f:
            json.dump(
                [
                    {
                        "symbol": m.symbol,
                        "timestamp": m.timestamp,
                        "prompt": m.prompt,
                        "reasoning": m.reasoning,
                        "decision": m.decision,
                        "confidence": m.confidence,
                        "actual_return": m.actual_return,
                        "won": m.won,
                        "feedback": m.feedback,
                    }
                    for m in self.memory[-100:]
                ],
                f,
                indent=2,
            )  # Keep last 100

    def _build_prompt(self, symbol: str, df: pd.DataFrame, market_context: Dict) -> str:
        """Build trading prompt for Kimi"""

        # Calculate key stats
        current_price = df["close"].iloc[-1]
        price_5d_ago = df["close"].iloc[-5] if len(df) >= 5 else df["close"].iloc[0]
        price_20d_ago = df["close"].iloc[-20] if len(df) >= 20 else df["close"].iloc[0]

        return_1d = (current_price / df["close"].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
        return_5d = (current_price / price_5d_ago - 1) * 100
        return_20d = (current_price / price_20d_ago - 1) * 100

        volatility = df["close"].pct_change().std() * 100
        volume_trend = df["volume"].iloc[-5:].mean() / df["volume"].iloc[-20:].mean() if len(df) >= 20 else 1.0

        # Market context
        spy_return = market_context.get("SPY", {}).get("return_1d", 0)
        vix_level = market_context.get("VIXY", {}).get("price", 0)

        # Build learning context from memory
        learning_context = ""
        if self.memory and self.learning_mode:
            # Find similar trades (same symbol, similar conditions)
            similar = [m for m in self.memory if m.symbol == symbol][-5:]
            if similar:
                wins = sum(1 for m in similar if m.won)
                learning_context = f"""
Your track record on {symbol}: {wins}/{len(similar)} recent trades profitable.
Key lessons from past trades:
"""
                for m in similar[-3:]:
                    if m.feedback:
                        learning_context += f"- {m.feedback}\n"

        prompt = f"""You are an expert quantitative trader analyzing {symbol} for a potential trade.

MARKET DATA:
- Current Price: ${current_price:.2f}
- 1-Day Return: {return_1d:+.2f}%
- 5-Day Return: {return_5d:+.2f}%
- 20-Day Return: {return_20d:+.2f}%
- Volatility (daily): {volatility:.2f}%
- Volume Trend: {volume_trend:.2f}x average

MARKET CONTEXT:
- SPY 1-Day Return: {spy_return:+.2f}%
- VIXY Level: {vix_level:.2f} (fear gauge)

{learning_context}

TASK:
Analyze this data and decide: LONG, SHORT, or FLAT (no trade).

Provide your response in this exact format:
DECISION: [LONG/SHORT/FLAT]
CONFIDENCE: [0.0-1.0]
REASONING: [2-3 sentences explaining your analysis]
RISK_FACTORS: [What could make this trade fail?]

Rules:
- Only trade if you have strong conviction (confidence > 0.6)
- Consider momentum, mean reversion, and market regime
- FLAT is a valid decision if conditions aren't right
- Learn from your past trades (see track record above)"""

        return prompt

    def _call_kimi(self, prompt: str) -> Dict[str, Any]:
        """Call Kimi API"""
        # Kimi uses standard Bearer token auth
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert quantitative trader. Be concise, analytical, and data-driven.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            # Correct Kimi API endpoint - includes /v1 in path
            url = f"{self.base_url}/v1/chat/completions"
            logger.debug("[KimiBrain] Calling API: %s", url)
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            logger.debug("[KimiBrain] Response status: %d", response.status_code)
            if response.status_code != 200:
                logger.warning("[KimiBrain] Response body: %s", response.text[:200])
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("[KimiBrain] API error: %s", e)
            return None

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse Kimi's response"""
        result = {
            "decision": "FLAT",
            "confidence": 0.0,
            "reasoning": "",
            "risk_factors": "",
        }

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("DECISION:"):
                result["decision"] = line.split(":", 1)[1].strip().upper()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("RISK_FACTORS:"):
                result["risk_factors"] = line.split(":", 1)[1].strip()

        return result

    def predict(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals using Kimi LLM.

        Args:
            data: Dict of symbol -> OHLCV DataFrame

        Returns:
            List of Signal objects
        """
        signals = []

        logger.info("[KimiBrain] Analyzing %d symbols with LLM...", len(data))

        # Build market context
        market_context = {}
        for symbol, df in data.items():
            if len(df) >= 2:
                market_context[symbol] = {
                    "price": df["close"].iloc[-1],
                    "return_1d": (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100,
                }

        for symbol, df in data.items():
            logger.debug("\n--- Analyzing %s ---", symbol)

            if len(df) < 20:
                logger.debug("  Insufficient data (%d bars)", len(df))
                continue

            # Build prompt
            prompt = self._build_prompt(symbol, df, market_context)

            # Call Kimi
            response = self._call_kimi(prompt)
            if not response:
                logger.warning("  API call failed for %s", symbol)
                continue

            # Parse response
            content = response["choices"][0]["message"]["content"]
            parsed = self._parse_response(content)

            logger.info("  Decision: %s", parsed["decision"])
            logger.info("  Confidence: %.2f", parsed["confidence"])
            logger.debug("  Reasoning: %s", parsed["reasoning"][:80])

            # Store in memory
            memory = KimiTradeMemory(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                prompt=prompt,
                reasoning=parsed["reasoning"],
                decision=parsed["decision"],
                confidence=parsed["confidence"],
            )
            self.memory.append(memory)

            # Generate signal if confident
            if parsed["decision"] in ["LONG", "SHORT"] and parsed["confidence"] > 0.6:
                direction = parsed["decision"]

                # Size based on confidence (10-50 shares)
                size = int(10 + parsed["confidence"] * 40)

                signals.append(
                    Signal(
                        symbol=symbol,
                        direction=direction,
                        confidence=parsed["confidence"],
                        size=size,
                        metadata={
                            "brain": self.name,
                            "reasoning": parsed["reasoning"],
                            "risk_factors": parsed["risk_factors"],
                            "model": "kimi-latest",
                        },
                    )
                )

        # Save memories
        self._save_memory()

        logger.info("\n[KimiBrain] Generated %d signals", len(signals))
        return signals

    def learn_from_outcome(
        self,
        symbol: str,
        predicted_direction: str,
        actual_return: float,
        memory_index: int = -1,
    ):
        """
        Learn from trade outcome - update memory with results.

        Call this after trade closes to improve future decisions.
        """
        if not self.memory or abs(memory_index) > len(self.memory):
            return

        mem = self.memory[memory_index]
        if mem.symbol != symbol:
            return

        mem.actual_return = actual_return
        mem.won = (actual_return > 0 and predicted_direction == "LONG") or (
            actual_return < 0 and predicted_direction == "SHORT"
        )

        # Generate learning feedback
        if mem.won:
            mem.feedback = f"Successful {predicted_direction} trade. Trust similar setups."
        else:
            mem.feedback = f"Failed {predicted_direction} trade. {predicted_direction} bias was wrong - consider opposite signal next time."

        self._save_memory()
        logger.info(
            "[KimiBrain] Learned: %s %s → %s (%.2f%%)",
            symbol,
            predicted_direction,
            "WON" if mem.won else "LOSS",
            actual_return,
        )


def test_kimi_brain():
    """Test KimiBrain"""
    from ai_trading_bridge import AITradingBridge

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("=" * 60)
    logger.info("TESTING KIMIBRAIN")
    logger.info("=" * 60)

    try:
        brain = KimiBrain(learning_mode=True)

        bridge = AITradingBridge(brain=brain, symbols=["SPY", "QQQ"], timeframe="1D", paper=True)

        result = bridge.run_cycle()

        logger.info("=" * 60)
        logger.info("Signals: %s", result["signals"])
        logger.info("Executed: %s", result["executed"])

    except Exception as e:
        logger.error("Error: %s", e)


if __name__ == "__main__":
    test_kimi_brain()
