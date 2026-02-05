"""
Service LLM - Client pour vLLM / Ollama
Gère les appels au modèle de langage pour la génération de réponses
"""

import logging
from functools import lru_cache

import httpx

from shared import ChatMessage, get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Client pour les serveurs LLM (vLLM ou Ollama).
    
    Supporte:
    - Ollama (développement) - API /api/generate
    - vLLM (production) - API OpenAI-compatible
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "llama3:8b",
        use_ollama: bool = True,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.use_ollama = use_ollama
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"LLMService initialisé: {self.base_url} (model={model})")

    async def generate_response(
        self,
        messages: list[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """
        Génère une réponse à partir d'une liste de messages.
        """
        try:
            if self.use_ollama:
                return await self._generate_ollama(
                    messages, system_prompt, max_tokens, temperature
                )
            else:
                return await self._generate_vllm(
                    messages, system_prompt, max_tokens, temperature
                )
        except httpx.ConnectError:
            logger.warning("LLM non disponible - mode mock")
            return self._mock_response(messages)
        except Exception as e:
            logger.exception(f"Erreur LLM: {e}")
            return f"Désolé, j'ai rencontré une erreur: {str(e)}"

    async def _generate_ollama(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Génération via Ollama API"""
        # Construire le prompt
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = msg.role.value.capitalize()
            prompt_parts.append(f"{role}: {msg.content}\n")

        prompt_parts.append("Assistant: ")
        prompt = "".join(prompt_parts)

        response = await self._client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def _generate_vllm(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Génération via vLLM (API OpenAI-compatible)"""
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _mock_response(self, messages: list[ChatMessage]) -> str:
        """Réponse mock quand le LLM n'est pas disponible"""
        last_msg = messages[-1].content if messages else ""
        return (
            f"[Mode Dev] J'ai bien reçu ton message: '{last_msg[:50]}...'. "
            "Le serveur LLM (Ollama) n'est pas encore démarré. "
            "Lance `ollama serve` puis `ollama pull llama3:8b` pour l'activer."
        )


@lru_cache
def get_llm_service() -> LLMService:
    """Retourne l'instance LLM configurée"""
    settings = get_settings()
    return LLMService(
        host=settings.ollama_host if settings.use_ollama else settings.vllm_host,
        port=settings.ollama_port if settings.use_ollama else settings.vllm_port,
        model=settings.ollama_model if settings.use_ollama else settings.vllm_model,
        use_ollama=settings.use_ollama,
    )
