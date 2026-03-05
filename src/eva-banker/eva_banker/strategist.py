import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from shared.redis_client import get_redis_client

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
        candles = await self.mt5.get_recent_candles(symbol, timeframe=15, count=100)
        if not candles:
            logger.warning("Cortex: No M15 data available.")
            return {"action": "NEUTRAL", "reason": "No Data"}

        # 2. Calculate Macro Indicators
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["tick_volume"] for c in candles]
        
        rsi = IndicatorFactory.rsi(closes, 14).iloc[-1]
        trend_ema = IndicatorFactory.ema(closes, 50).iloc[-1]
        current_price = closes[-1]
        
        # New Advanced Indicators (VWAP, ADX, Fibs)
        vwap = IndicatorFactory.vwap(highs, lows, closes, volumes).iloc[-1]
        adx_data = IndicatorFactory.adx(highs, lows, closes, 14)
        adx_val = adx_data["adx"].iloc[-1]
        fibs = IndicatorFactory.get_fibonacci_levels(highs, lows, 50)
        
        trend = "BULLISH" if current_price > trend_ema else "BEARISH"
        
        # 3. Formulate Prompt for Gemma
        context = {
            "symbol": symbol,
            "timeframe": "M15",
            "price": current_price,
            "trend_50ema": trend,
            "rsi_14": round(rsi, 2),
            "vwap": round(vwap, 2) if not __import__("math").isnan(vwap) else current_price,
            "adx_trend_strength": round(adx_val, 2),
            "fib_382": round(fibs.get("fib_382", 0), 2),
            "fib_618": round(fibs.get("fib_618", 0), 2),
            "last_5_candles": [round(c["close"], 2) for c in candles[-5:]]
        }
        
        prompt = (
            f"Review the M15 market structure for {symbol}. "
            f"Trend is {trend}. RSI is {rsi:.1f}. ADX (Trend Strength) is {adx_val:.1f}. "
            f"Price is at {current_price:.2f} relative to VWAP {context['vwap']:.2f}. "
            f"Key Fib support/resistances lie at {context['fib_382']:.2f} and {context['fib_618']:.2f}. "
            "Determine the strategic bias: BULLISH (Buy Dips), BEARISH (Sell Rallies), or RANGING (Scalp Both). "
            "Consider VWAP rejections and ADX > 25 for strong trend continuation. "
            "Output JSON: {\"bias\": \"...\", \"reason\": \"...\"}"
        )

        # 4. Ask Cortex
        response = await self.cortex.analyze(json.dumps(context), prompt)
        
        # 5. Parse Response (Simple heuristic parsing if JSON fails)
        cortex_bias = "NEUTRAL"
        if "BULLISH" in response.upper(): cortex_bias = "BULLISH"
        elif "BEARISH" in response.upper(): cortex_bias = "BEARISH"
        elif "RANGING" in response.upper(): cortex_bias = "RANGING"
        
        # 6. Ask Proxmox MTF-GNN via REST API
        gnn_intraday = "NEUTRAL"
        gnn_scalp = "NEUTRAL"
        gnn_swing = "NEUTRAL"
        gnn_confidence = 0.0
        try:
            import aiohttp
            import os
            lab_host = os.getenv("LAB_HOST", "localhost")
            url = f"http://{lab_host}:8600/gnn/predict"
            
            # Send multi-timeframe payload using horizon-tagged keys
            # Each value is a [seq_len] list of recent closes (simplified feature vector)
            payload = {
                "assets_data": {
                    f"{symbol}_5": [closes[-15:]],   # M5 context = Scalping
                    f"{symbol}_60": [closes[-15:]],  # H1 context = Intraday
                    f"{symbol}_1440": [closes[-15:]], # D1 context = Swing
                }
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # New MTF response format
                        gnn_scalp = data.get("scalp", {}).get("bias", "NEUTRAL")
                        gnn_intraday = data.get("intraday", {}).get("bias", "NEUTRAL")
                        gnn_swing = data.get("swing", {}).get("bias", "NEUTRAL")
                        gnn_confidence = data.get("intraday", {}).get("confidence", 0.0)
                    else:
                        logger.warning(f"⚠️ GNN a retourné HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de joindre le MTF-GNN: {e.__class__.__name__}")
        
        # Reference the intraday bias for Cortex matching (M15 analysis runs on H1-like horizon)
        gnn_bias = gnn_intraday
            
        # 7. Merge Biases (Swing override on extreme conviction, else intraday GNN cross-check)
        final_bias = cortex_bias
        if gnn_swing != "NEUTRAL" and gnn_swing != cortex_bias:
            # D1 macro context forces RANGING if contradicting the Cortex's M15 reading
            final_bias = "RANGING"
        elif gnn_bias != "NEUTRAL" and gnn_confidence > 0.5:
            if gnn_bias == cortex_bias:
                pass  # Synergy!
            else:
                # GNN intraday contradicts Cortex M15: moderate by forcing RANGING
                final_bias = "RANGING" if gnn_confidence < 0.75 else gnn_bias

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
        from colorama import Fore, Style
        sym_color = Fore.CYAN if "XAU" in symbol else (Fore.YELLOW if "BTC" in symbol else Fore.WHITE)
        
        def get_color(b):
            if b == "BULLISH": return Fore.GREEN
            if b == "BEARISH": return Fore.RED
            if b == "RANGING": return Fore.MAGENTA
            return Fore.LIGHTBLACK_EX
            
        cb_color = get_color(cortex_bias)
        gb_color = get_color(gnn_bias)
        fb_color = get_color(final_bias)
        
        # --- RICH CONSOLE OUTPUT ---
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            
            console = Console()
            
            text = Text()
            text.append("📊 Technical Context:\n", style="bold yellow")
            text.append(f"  • Price: {current_price:.2f} (Trend: {trend})\n", style="white")
            text.append(f"  • RSI: {rsi:.1f} | ADX: {adx_val:.1f} | VWAP: {vwap:.2f}\n", style="white")
            text.append(f"  • Fibs: {fibs.get('fib_382', 0):.2f} / {fibs.get('fib_618', 0):.2f}\n\n", style="white")
            
            text.append("🧠 LLM Reasoning:\n", style="bold cyan")
            text.append(f"{response}\n\n", style="italic white")
            
            text.append(f"Cortex Bias: ", style="bold")
            text.append(f"[{cortex_bias}]\n", style=cb_color.replace('\x1b[', '').replace('m', '').lower() if hasattr(cb_color, 'replace') else "white")
            
            text.append(f"GNN Bias: ", style="bold")
            text.append(f"[{gnn_bias}]\n", style=gb_color.replace('\x1b[', '').replace('m', '').lower() if hasattr(gb_color, 'replace') else "white")
            
            text.append(f"Final Bias: ", style="bold")
            text.append(f"[{final_bias}]", style=fb_color.replace('\x1b[', '').replace('m', '').lower() if hasattr(fb_color, 'replace') else "white")
            
            panel = Panel(
                text,
                title=f"The Cortex: {symbol} (M15)",
                border_style="magenta",
                expand=False
            )
            console.print(panel)
            
        except ImportError:
            # Fallback if rich is somehow missing
            logger.info(
                f"🧠 Cortex Strategy -> {sym_color}{symbol:<8}{Style.RESET_ALL} | "
                f"Cortex: {cb_color}[{cortex_bias}]{Style.RESET_ALL} | "
                f"GNN: {gb_color}[{gnn_bias}]{Style.RESET_ALL} -> "
                f"Final: {fb_color}[{final_bias}]{Style.RESET_ALL} "
            )
        
        # PUBLISH TO AGENT FEED (UI)
        try:
            redis = get_redis_client()
            asyncio.create_task(redis.publish("eva.cortex.feed", {
                "id": str(uuid.uuid4()),
                "source_agent": "Strategist",
                "action": f"Cortex M15 Macro Bias for {symbol}: {final_bias} (GNN: {gnn_bias}) -> {response[:100]}...",
                "timestamp": datetime.now().isoformat(),
                "type": "thought"
            }))
        except Exception as e_redis:
            logger.debug(f"Failed to publish Cortex thought to Feed: {e_redis}")
            
        # ----------------------------
        
        return strategy

    def get_bias(self, symbol: str) -> str:
        return self.latest_strategy.get(symbol, {}).get("bias", "NEUTRAL")

    async def get_micro_reasoning(self, symbol: str, action: str, indicators: dict) -> str:
        """
        Generates a very short (1 sentence) reasoning for a specific trade in French.
        """
        rsi = indicators.get("RSI", 50)
        adx = indicators.get("adx", 25)
        vwap_dist = indicators.get("vwap_dist", 0)
        macd = indicators.get("MACD_Hist", 0)
        
        prompt = (
            f"En tant qu'expert trading, explique en UNE seule phrase courte et percutante en FRANCAIS "
            f"pourquoi on ouvre un {action} sur {symbol}. "
            f"Contexte: RSI={rsi:.1f}, ADX={adx:.1f}, "
            f"MACD Hist={macd:.4f}. "
            "Sois technique mais clair. Pas de blabla."
        )
        
        try:
            # Use a faster/smaller model for micro-reasoning if available, else default
            reasoning = await self.cortex.analyze(f"Action: {action} on {symbol}", prompt)
            # Cleanup unwanted characters or long responses
            return reasoning.strip().split('\n')[0][:200]
        except Exception as e:
            logger.warning(f"Failed to generate micro-reasoning: {e}")
            return "Analyse technique confirmée par le Swarm."
