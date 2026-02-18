import aiohttp
import json
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Local LLMs (Ollama/Gemma 3).
    Acts as the 'Language Center' of the Cortex.
    """
    def __init__(self, model: str = "gemma3:4b", host: str = "http://localhost:11434"):
        self.model = os.getenv("LLM_MODEL", model)
        self.host = os.getenv("LLM_HOST", host)
        self.api_url = f"{self.host}/api/generate"
        logger.info(f"🧠 Cortex (LLM) initialized on {self.host} with model {self.model}")

    async def analyze(self, context: str, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response.
        """
        full_prompt = f"Context: {context}\n\nTask: {prompt}\n\nResponse:"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2, # Low temp for analytical precision
                "num_ctx": 4096
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, timeout=30.0) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("response", "").strip()
                    else:
                        logger.error(f"LLM Error: {resp.status} - {await resp.text()}")
                        return "Error: LLM Unreachable"
        except Exception as e:
            logger.error(f"LLM Connection Failed: {e}")
            return "Error: LLM Connection Failed"

    async def get_strategy_signal(self, market_data: Dict[str, Any]) -> str:
        """
        Specialized prompt for Trading Strategy.
        """
        context = json.dumps(market_data, indent=2)
        prompt = (
            "You are a Senior Trading Strategist. Analyze the provided M15 Market Data. "
            "Identify the dominant trend (Bullish/Bearish/Neutral) and Key Support/Resistance levels. "
            "Output a concise directive for the execution engine: 'BUY_ONLY', 'SELL_ONLY', or 'NEUTRAL'. "
            "Justify directly."
        )
        return await self.analyze(context, prompt)
