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

        logger.info(
            "?? Cortex (LLM) initialized on %s (%s) with model %s",
            self.host,
            self.backend,
            self.model,
        )

    async def analyze(self, context: str, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response.
        """
        full_prompt = f"Context: {context}\n\nTask: {prompt}\n\nResponse:"

        if self.backend == "vllm":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        else:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096,
                },
            }

        try:
            async with aiohttp.ClientSession() as session:
                if self.backend == "vllm":
                    return await self._analyze_vllm(session, payload)

                async with session.post(self.api_url, json=payload, timeout=30.0) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("response", "").strip()

                    logger.error("LLM Error: %s - %s", resp.status, await resp.text())
                    return "Error: LLM Unreachable"
        except Exception as e:
            logger.error("LLM Connection Failed: %s", e)
            return "Error: LLM Connection Failed"

    async def _analyze_vllm(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> str:
        """
        Ex?cute un appel vLLM avec fallback automatique si le mod?le demand? est introuvable.
        """
        async with session.post(self.api_url, json=payload, timeout=30.0) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()

            error_text = await resp.text()
            if resp.status == 404 and self._looks_like_missing_model_error(error_text):
                fallback_model = await self._discover_vllm_available_model(session)
                if fallback_model and fallback_model != payload.get("model"):
                    old_model = payload.get("model")
                    logger.warning(
                        "Modele vLLM introuvable (%s). Bascule automatique vers %s.",
                        old_model,
                        fallback_model,
                    )
                    self.model = fallback_model
                    retry_payload = dict(payload)
                    retry_payload["model"] = fallback_model

                    async with session.post(self.api_url, json=retry_payload, timeout=30.0) as retry_resp:
                        if retry_resp.status == 200:
                            retry_result = await retry_resp.json()
                            return retry_result["choices"][0]["message"]["content"].strip()

                        logger.error(
                            "LLM Error apres fallback: %s - %s",
                            retry_resp.status,
                            await retry_resp.text(),
                        )
                        return "Error: LLM Unreachable"

            logger.error("LLM Error: %s - %s", resp.status, error_text)
            return "Error: LLM Unreachable"

    async def _discover_vllm_available_model(self, session: aiohttp.ClientSession) -> Optional[str]:
        """
        Interroge /v1/models pour trouver un mod?le disponible sur vLLM.
        """
        models_url = f"{self.host}/v1/models"
        try:
            async with session.get(models_url, timeout=10.0) as resp:
                if resp.status != 200:
                    logger.warning("Impossible de lister les modeles vLLM (HTTP %s)", resp.status)
                    return None

                data = await resp.json()
                entries = data.get("data", []) if isinstance(data, dict) else []
                available_ids = [entry.get("id") for entry in entries if isinstance(entry, dict) and entry.get("id")]
                if not available_ids:
                    return None

                preferred = [
                    os.getenv("BANKER_CORTEX_MODEL", "").strip(),
                    os.getenv("COUNCIL_MODEL_BANKER", "").strip(),
                    os.getenv("VLLM_MODEL_NAME", "").strip(),
                    os.getenv("VLLM_MODEL", "").strip(),
                ]
                for model_name in preferred:
                    if model_name and model_name in available_ids:
                        return model_name

                return available_ids[0]
        except Exception as exc:
            logger.warning("Echec de decouverte du modele vLLM: %s", exc)
            return None

    @staticmethod
    def _looks_like_missing_model_error(error_text: str) -> bool:
        """
        D?tecte les erreurs 404 li?es ? un mod?le absent.
        """
        lowered = (error_text or "").lower()
        return "does not exist" in lowered or "model" in lowered and "notfound" in lowered

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
