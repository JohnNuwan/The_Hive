import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional

from shared.llm_client import LLMClient
from eva_banker.services.mt5 import MT5Service
from shared.indicators import IndicatorFactory

logger = logging.getLogger(__name__)

class Strategist:
    """
    The Cortex: High-Level Strategic planner using LLM (Gemma 3).
    Analyzes Macro Trends (M15/H1) to guide the Subconscious (Dreamer).
    """
    def __init__(self, mt5_service: MT5Service):
        self.mt5 = mt5_service
        self.cortex = LLMClient(model="gemma3:4b") # Default model
        self.latest_strategy: Dict[str, Any] = {}
        self.last_update: datetime = datetime.min

    async def analyze_market_context(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches M15 data, calculates macro indicators, and asks LLM for strategy.
        """
        logger.info(f"🧠 Cortex: Analyzing Macro Context for {symbol}...")
        
        # 1. Fetch M15 Data (Macro View)
        candles = await self.mt5.get_recent_candles(symbol, timeframe=15, count=50)
        if not candles:
            logger.warning("Cortex: No M15 data available.")
            return {"action": "NEUTRAL", "reason": "No Data"}

        # 2. Calculate Macro Indicators
        closes = [c["close"] for c in candles]
        rsi = IndicatorFactory.rsi(closes, 14).iloc[-1]
        trend_ema = IndicatorFactory.ema(closes, 50).iloc[-1]
        current_price = closes[-1]
        
        trend = "BULLISH" if current_price > trend_ema else "BEARISH"
        
        # 3. Formulate Prompt for Gemma
        context = {
            "symbol": symbol,
            "timeframe": "M15",
            "price": current_price,
            "trend_50ema": trend,
            "rsi_14": round(rsi, 2),
            "last_5_candles": [c["close"] for c in candles[-5:]]
        }
        
        prompt = (
            f"Review the M15 market structure for {symbol}. "
            f"Trend is {trend}. RSI is {rsi}. "
            "Determine the strategic bias: BULLISH (Buy Dips), BEARISH (Sell Rallies), or RANGING (Scalp Both). "
            "Output JSON: {\"bias\": \"...\", \"reason\": \"...\"}"
        )

        # 4. Ask Cortex
        response = await self.cortex.analyze(json.dumps(context), prompt)
        
        # 5. Parse Response (Simple heuristic parsing if JSON fails)
        bias = "NEUTRAL"
        if "BULLISH" in response.upper(): bias = "BULLISH"
        elif "BEARISH" in response.upper(): bias = "BEARISH"
        elif "RANGING" in response.upper(): bias = "RANGING"
        
        strategy = {
            "symbol": symbol,
            "bias": bias,
            "raw_thought": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.latest_strategy[symbol] = strategy
        logger.info(f"🧠 Cortex Strategy for {symbol}: {bias}")
        return strategy

    def get_bias(self, symbol: str) -> str:
        return self.latest_strategy.get(symbol, {}).get("bias", "NEUTRAL")
