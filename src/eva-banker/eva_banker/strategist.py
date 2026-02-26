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
        cortex_bias = "NEUTRAL"
        if "BULLISH" in response.upper(): cortex_bias = "BULLISH"
        elif "BEARISH" in response.upper(): cortex_bias = "BEARISH"
        elif "RANGING" in response.upper(): cortex_bias = "RANGING"
        
        # 6. Ask Proxmox GNN via REST API
        gnn_bias = "NEUTRAL"
        gnn_confidence = 0.0
        try:
            import aiohttp
            import os
            lab_host = os.getenv("LAB_HOST", "localhost")
            url = f"http://{lab_host}:8600/gnn/predict"
            # We mock the feature tensor here for the API structure
            payload = {
                "assets_data": {
                    symbol: [closes[-15:]] # Mock 15 features
                }
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        gnn_bias = data.get("bias", "NEUTRAL")
                        gnn_confidence = data.get("confidence", 0.0)
                    else:
                        logger.warning(f"⚠️ GNN a retourné HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de joindre le GNN sur le Lab: {e.__class__.__name__} - {e}")
            
        # 7. Merge Biases (If GNN is confident, it overrides/validates)
        final_bias = cortex_bias
        if gnn_bias != "NEUTRAL" and gnn_confidence > 0.5:
            if gnn_bias == cortex_bias:
                 # Synergy!
                 pass 
            else:
                 # GNN overrides or forces ranging if contradictions
                 final_bias = "RANGING" if gnn_confidence < 0.8 else gnn_bias

        strategy = {
            "symbol": symbol,
            "cortex_bias": cortex_bias,
            "gnn_bias": gnn_bias,
            "bias": final_bias,
            "raw_thought": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.latest_strategy[symbol] = strategy
        
        # --- COMMAND LINE LOGGING ---
        from colorama import Fore, Style, init
        init(autoreset=True)
        sym_color = Fore.CYAN if "XAU" in symbol else (Fore.YELLOW if "BTC" in symbol else Fore.WHITE)
        
        def get_color(b):
            if b == "BULLISH": return Fore.GREEN
            if b == "BEARISH": return Fore.RED
            if b == "RANGING": return Fore.MAGENTA
            return Fore.LIGHTBLACK_EX
            
        cb_color = get_color(cortex_bias)
        gb_color = get_color(gnn_bias)
        fb_color = get_color(final_bias)
        
        logger.info(
            f"🧠 {Fore.MAGENTA}Cortex Strategy{Style.RESET_ALL} -> {sym_color}{symbol:<8}{Style.RESET_ALL} | "
            f"Cortex: {cb_color}[{cortex_bias}]{Style.RESET_ALL} | "
            f"GNN: {gb_color}[{gnn_bias}]{Style.RESET_ALL} -> "
            f"Final: {fb_color}[{final_bias}]{Style.RESET_ALL} "
        )
        # ----------------------------
        
        return strategy

    def get_bias(self, symbol: str) -> str:
        return self.latest_strategy.get(symbol, {}).get("bias", "NEUTRAL")
