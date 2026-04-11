"""Kimi Brain - Layer 1.5 LLM Reasoner for Edge Cases.

Uses Moonshot AI (Kimi) to analyze market context for setups where the
XGBoost model confidence is borderline (0.5 to 0.7).
"""

import os
import logging
from typing import Dict, Any, Optional
import requests
import json

from contracts.types import Direction, MarketData
from layer1.feature_builder import FeatureVector

logger = logging.getLogger(__name__)

class KimiBiasResult:
    def __init__(self, direction: Direction, reasoning: str, confidence_boost: float = 0.0):
        self.direction = direction
        self.reasoning = reasoning
        self.confidence_boost = confidence_boost

class KimiBrain:
    """LLM wrapper for Moonshot API (Kimi)."""
    
    def __init__(self):
        # We assume load_dotenv() has been called before initializing this class
        self.api_key = os.getenv("KIMI_API_KEY", os.getenv("Kimi_Api_key"))
        self.base_url = "https://api.moonshot.ai/v1"
        self.model_id = "moonshot-v1-8k" # Default safe model ID for moonshot api 
        
        if not self.api_key:
            logger.warning("Kimi API key is missing. Hybrid fallback to XGBoost-only.")

    def analyze(self, symbol: str, features: FeatureVector, market_data: MarketData) -> Optional[KimiBiasResult]:
        """Ask Kimi to analyze the setup."""
        if not self.api_key:
            return None # Skip if not configured
            
        prompt = self._build_prompt(symbol, features, market_data)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a senior quantitative swing trader. Analyze the market structure and data to determine a directional bias. Respond ONLY with a JSON object in this exact format: {\"direction\": \"LONG\" or \"SHORT\" or \"NEUTRAL\", \"reasoning\": \"Your detailed explanation\"}"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3 # keep it low for consistency
        }
        
        try:
            logger.info(f"Escalating {symbol} to Kimi (Layer 1.5) for reasoning...")
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Basic parsing of json response
            # Sometimes LLMs wrap json in markdown blocks, we need to strip them
            content = content.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(content)
            
            direction_str = result_json.get("direction", "NEUTRAL").upper()
            if direction_str == "LONG":
                direction = Direction.LONG
            elif direction_str == "SHORT":
                direction = Direction.SHORT
            else:
                direction = Direction.NEUTRAL
                
            return KimiBiasResult(
                direction=direction,
                reasoning=result_json.get("reasoning", "No reasoning provided.")
            )
            
        except Exception as e:
            logger.error(f"Kimi API request failed: {e}")
            return None

    def _build_prompt(self, symbol: str, features: FeatureVector, market_data: MarketData) -> str:
        """Construct the prompt from features."""
        
        prompt = f"Analyze the following data for {symbol} and determine if the setup leans LONG, SHORT, or remains NEUTRAL based on ICT and quantitative concepts.\n\n"
        
        # Add key features that matter for swing trading context
        prompt += "--- CONTEXT & MARKET DATA ---\n"
        prompt += f"Current Price: {market_data.current_price}\n"
        prompt += f"ATR (Volatility): {market_data.atr_14:.2f}\n"
        prompt += f"Volume 24h: {market_data.volume_24h}\n\n"
        
        prompt += "--- QUANT FEATURES ---\n"
        
        fd = features.to_dict()
        prompt += f"Price vs SMA20: {fd.get('price_vs_sma_20', 0):.4f}\n"
        prompt += f"RSI (14): {fd.get('rsi_14', 50):.2f}\n"
        prompt += f"MACD Histogram: {fd.get('macd_histogram', 0):.4f}\n"
        prompt += f"ADX (Trend Strength): {fd.get('adx_14', 0):.2f}\n"
        
        # Keep it concise to save tokens
        return prompt
