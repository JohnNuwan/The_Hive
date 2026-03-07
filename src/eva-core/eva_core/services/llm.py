"""Service LLM pour EVA: inference vLLM/Ollama avec routage MoE."""

import logging
from functools import lru_cache
from types import SimpleNamespace

import httpx

from shared import ChatMessage, get_settings
from shared.memory_bridge import get_memory_bridge

from eva_core.services.council import get_council_service

logger = logging.getLogger(__name__)


class LLMService:
    """Client unifie pour les backends LLM avec routage par role."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "llama3.2:1b",
        use_ollama: bool = True,
    ) -> None:
        """Initialise le client LLM avec un endpoint par defaut."""
        self.host = host
        self.port = port
        self.model = model
        self.use_ollama = use_ollama
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(timeout=120.0)
        self.council = get_council_service()
        self.memory = get_memory_bridge()
        logger.info("LLMService initialise: endpoint=%s model=%s", self.base_url, model)

    def _resolve_route(self, role: str, target_model: str):
        """Resout la route Conseil ou retourne une route locale par defaut."""
        backend = "ollama" if self.use_ollama else "vllm"
        fallback = SimpleNamespace(
            role=role,
            backend=backend,
            host=self.host,
            port=self.port,
            model=target_model or self.model,
        )

        resolve_fn = getattr(self.council, "resolve_route", None)
        if callable(resolve_fn):
            try:
                route = resolve_fn(role)
                if hasattr(route, "backend") and hasattr(route, "host"):
                    return route
            except Exception as exc:
                logger.warning("Echec resolve_route(%s): %s", role, exc)

        return fallback

    async def generate_response(
        self,
        messages: list[ChatMessage],
        system_prompt: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        role: str = "general",
    ) -> tuple[str, str | None]:
        """Genere une reponse via le role cible et retourne (reponse, thoughts)."""
        try:
            target_model = await self.council.prepare_model(role)
            route = self._resolve_route(role, target_model)

            # RAG contextuel
            try:
                last_user_msg = next((m for m in reversed(messages) if m.role.value == "user"), None)
                if last_user_msg:
                    memories = await self.memory.search(last_user_msg.content, limit=3)
                    if memories:
                        memory_block = "\n".join([f"- {memory}" for memory in memories])
                        system_prompt += (
                            "\n\n[CONTEXTE MEMORIEL (Souvenirs pertinents)]:\n"
                            f"{memory_block}\n"
                            "Influence ta reponse avec ces faits si necessaire."
                        )
            except Exception as exc:
                logger.warning("Memory retrieval ignored: %s", exc)

            if role == "research":
                system_prompt += (
                    "\nUtilise les balises <thought>...</thought> pour detailler "
                    "ton raisonnement avant la reponse finale."
                )

            if route.backend == "ollama":
                raw_response = await self._generate_ollama(
                    messages=messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_override=target_model,
                    host=route.host,
                    port=route.port,
                )
            else:
                try:
                    raw_response = await self._generate_vllm(
                        messages=messages,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model_override=target_model,
                        host=route.host,
                        port=route.port,
                    )
                except httpx.HTTPStatusError as exc:
                    # Fallback MoE: role -> general si modele role indisponible.
                    if role.lower() != "general":
                        logger.warning(
                            "Echec modele role '%s' (%s): fallback sur role general.",
                            role,
                            exc,
                        )
                        fallback_model = await self.council.prepare_model("general")
                        fallback_route = self._resolve_route("general", fallback_model)
                        raw_response = await self._generate_vllm(
                            messages=messages,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            model_override=fallback_model,
                            host=fallback_route.host,
                            port=fallback_route.port,
                        )
                    else:
                        raise

            thoughts = None
            if "<thought>" in raw_response and "</thought>" in raw_response:
                start = raw_response.find("<thought>") + len("<thought>")
                end = raw_response.find("</thought>")
                thoughts = raw_response[start:end].strip()
                clean_response = raw_response[end + len("</thought>") :].strip()
                return clean_response, thoughts

            return raw_response, None

        except httpx.ConnectError:
            logger.warning("LLM indisponible - mode mock")
            return self._mock_response(messages), None
        except Exception as exc:
            logger.exception("Erreur LLM: %s", exc)
            return f"Desole, j'ai rencontre une erreur: {exc}", None

    async def _generate_ollama(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        model_override: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> str:
        """Generation via API Ollama /api/generate."""
        prompt_parts: list[str] = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        for message in messages:
            role_str = message.role.value.capitalize()
            prompt_parts.append(f"{role_str}: {message.content}\n")

        prompt_parts.append("Assistant: ")
        prompt = "".join(prompt_parts)

        resolved_host = host or self.host
        resolved_port = port or self.port
        base_url = f"http://{resolved_host}:{resolved_port}"

        response = await self._client.post(
            f"{base_url}/api/generate",
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
        model_override: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> str:
        """Generation via API OpenAI-compatible de vLLM."""
        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            api_messages.append({"role": message.role.value, "content": message.content})

        resolved_host = host or self.host
        resolved_port = port or self.port
        base_url = f"http://{resolved_host}:{resolved_port}"

        response = await self._client.post(
            f"{base_url}/v1/chat/completions",
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
        """Retourne un message de secours quand aucun backend LLM ne repond."""
        last_msg = messages[-1].content if messages else ""
        return (
            f"[Mode Dev] Message recu: '{last_msg[:50]}...'. "
            "Le backend LLM est indisponible sur "
            f"{self.host}:{self.port}."
        )


@lru_cache
def get_llm_service() -> LLMService:
    """Construit l'instance LLM selon la configuration active."""
    settings = get_settings()
    use_vllm = settings.llm_backend == "vllm"

    if use_vllm:
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
