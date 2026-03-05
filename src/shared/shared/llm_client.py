import aiohttp
import json
import logging
import os
from typing import Optional, Dict, Any

from shared.config import get_settings

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Local LLMs (Ollama or vLLM).
    Acts as the 'Language Center' of the Cortex.
    """
    def __init__(self, model: str = None, host: str = None):
        self.settings = get_settings()
        self.backend = self.settings.llm_backend
        
        if self.backend == "vllm":
            self.model = model or self.settings.vllm_model
            self.host = host or f"http://{self.settings.vllm_host}:{self.settings.vllm_port}"
            self.api_url = f"{self.host}/v1/chat/completions"
        else:
            self.model = model or self.settings.ollama_model
            self.host = host or f"http://{self.settings.ollama_host}:{self.settings.ollama_port}"
            self.api_url = f"{self.host}/api/generate"
            
        logger.info(f"🧠 Cortex (LLM) initialized on {self.host} ({self.backend}) with model {self.model}")

    async def analyze(self, context: str, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response.
        """
        full_prompt = f"Context: {context}\n\nTask: {prompt}\n\nResponse:"
        
        if self.backend == "vllm":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.2, # Low temp for analytical precision
                "max_tokens": 1024
            }
        else:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096
                }
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, timeout=30.0) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if self.backend == "vllm":
                            return result["choices"][0]["message"]["content"].strip()
                        else:
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
