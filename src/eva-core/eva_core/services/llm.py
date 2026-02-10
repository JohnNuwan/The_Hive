"""
Service LLM - Client pour vLLM / Ollama
Gère les appels au modèle de langage pour la génération de réponses
"""

import logging
from functools import lru_cache

import httpx

from shared import ChatMessage, get_settings

from eva_core.services.council import get_council_service
from shared.memory_bridge import get_memory_bridge

logger = logging.getLogger(__name__)


class LLMService:
    """
    Client pour les serveurs LLM (vLLM ou Ollama).
    
    Supporte:
    - Ollama (développement) - API /api/generate
    - vLLM (production) - API OpenAI-compatible
    - The Council - Gestion des rôles et Model Swapping
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "llama3.2:1b",
        use_ollama: bool = True,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.use_ollama = use_ollama
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(timeout=120.0)
        self.council = get_council_service()
        self.memory = get_memory_bridge()
        logger.info(f"LLMService initialisé: {self.base_url} (model={model})")

    async def generate_response(
        self,
        messages: list[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        role: str = "general",
    ) -> tuple[str, str | None]:
        """
        Génère une réponse à partir d'une liste de messages.
        Utilise le Council pour préparer le modèle selon le rôle.
        Retourne (réponse_nettoyée, trace_de_raisonnement).
        """
        try:
            # 1. Préparer le modèle via le Council (Swap si nécessaire)
            target_model = await self.council.prepare_model(role)

            # 1.5. Récupération Contextuelle (Mem0 / Neo4j)
            try:
                # On utilise le dernier message utilisateur comme requête
                last_user_msg = next((m for m in reversed(messages) if m.role.value == "user"), None)
                if last_user_msg:
                    memories = await self.memory.search(last_user_msg.content, limit=3)
                    if memories:
                        memory_block = "\n".join([f"- {m}" for m in memories])
                        system_prompt += f"\n\n[CONTEXTE MÉMORIEL (Souvenirs Pertinents)]:\n{memory_block}\nInfluence ta réponse avec ces faits si nécessaire."
            except Exception as e:
                logger.warning(f"Memory retrieval passed: {e}")
            
            # Ajouter une instruction de raisonnement si le rôle est "research"
            if role == "research":
                system_prompt += "\nUtilise les balises <thought>...</thought> pour détailler ton raisonnement avant de répondre."

            if self.use_ollama:
                raw_response = await self._generate_ollama(
                    messages, system_prompt, max_tokens, temperature, model_override=target_model
                )
            else:
                raw_response = await self._generate_vllm(
                    messages, system_prompt, max_tokens, temperature, model_override=target_model
                )
            
            # 2. Extraire les pensées (<thought>...</thought>)
            thoughts = None
            if "<thought>" in raw_response and "</thought>" in raw_response:
                start = raw_response.find("<thought>") + len("<thought>")
                end = raw_response.find("</thought>")
                thoughts = raw_response[start:end].strip()
                clean_response = raw_response[end + len("</thought>"):].strip()
                return clean_response, thoughts
            
            return raw_response, None

        except httpx.ConnectError:
            logger.warning("LLM non disponible - mode mock")
            return self._mock_response(messages), None
        except Exception as e:
            logger.exception(f"Erreur LLM: {e}")
            return f"Désolé, j'ai rencontré une erreur: {str(e)}", None

    async def _generate_ollama(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        model_override: str = None,
    ) -> str:
        """Génération via Ollama API"""
        # Construire le prompt
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role_str = msg.role.value.capitalize()
            prompt_parts.append(f"{role_str}: {msg.content}\n")

        prompt_parts.append("Assistant: ")
        prompt = "".join(prompt_parts)

        response = await self._client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model_override or self.model,
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
        model_override: str = None,
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
                "model": model_override or self.model,
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
    
    use_vllm = settings.llm_backend == "vllm"
    
    # Priority to settings.llm_backend, fallback to settings.use_ollama if legacy
    if settings.llm_backend == "vllm":
        host = settings.vllm_host
        port = settings.vllm_port
        model = settings.vllm_model
    else:
        host = settings.ollama_host
        port = settings.ollama_port
        model = settings.ollama_model
        
    return LLMService(
        host=host,
        port=port,
        model=model,
        use_ollama=not use_vllm,
    )
