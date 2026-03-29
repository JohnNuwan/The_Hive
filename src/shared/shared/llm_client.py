"""Client LLM partage pour THE HIVE avec gestion des indisponibilites transitoires."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

import aiohttp

from shared.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Dialogue avec le backend LLM actif et amortit les indisponibilites transitoires.

    Le `banker` appelle ce client tres frequemment. Lorsqu'un redemarrage `vLLM`
    est en cours, un simple refus TCP peut sinon polluer la console sur chaque
    actif scanne. Ce client ajoute donc des tentatives courtes puis un cooldown
    temporaire pour laisser le serveur revenir sans noyer les logs.

    Args:
        model (str | None): Modele cible a utiliser. Si absent, la configuration
            globale determine le modele par defaut.
        host (str | None): Hote ou URL de base a utiliser. Si absent, la
            configuration globale determine l'endpoint par defaut.
        backend (str | None): Backend explicite (`vllm` ou `ollama`). Si absent,
            la configuration globale reste la source de verite.
        request_timeout_seconds (float | None): Delai HTTP maximal. Si absent,
            un delai par defaut raisonnable est applique.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        backend: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> None:
        """
        Initialise le client LLM partage.

        Args:
            model (str | None): Surcharge optionnelle du modele.
            host (str | None): Surcharge optionnelle de l'endpoint.
            backend (str | None): Surcharge optionnelle du backend.
            request_timeout_seconds (float | None): Delai maximal par appel.
        """
        self.settings = get_settings()
        normalized_backend = str(backend or self.settings.llm_backend or "vllm").strip().lower()
        self.backend = normalized_backend if normalized_backend in {"vllm", "ollama"} else "vllm"
        self.retry_attempts = max(1, self._env_int("LLM_RETRY_ATTEMPTS", 4))
        self.retry_delay_seconds = max(1.0, self._env_float("LLM_RETRY_DELAY_SECONDS", 5.0))
        self.failure_cooldown_seconds = max(0.0, self._env_float("LLM_FAILURE_COOLDOWN_SECONDS", 45.0))
        default_timeout_seconds = max(1.0, self._env_float("LLM_REQUEST_TIMEOUT_SECONDS", 30.0))
        self.request_timeout_seconds = max(
            1.0,
            float(request_timeout_seconds) if request_timeout_seconds is not None else default_timeout_seconds,
        )
        self._cooldown_until = 0.0
        self._last_failure_reason = ""

        if self.backend == "vllm":
            self.model = model or self.settings.vllm_model
            self.host = self._build_base_url(host, self.settings.vllm_host, self.settings.vllm_port)
            self.api_url = f"{self.host}/v1/chat/completions"
        else:
            self.model = model or self.settings.ollama_model
            self.host = self._build_base_url(host, self.settings.ollama_host, self.settings.ollama_port)
            self.api_url = f"{self.host}/api/generate"

        logger.info(
            "Client LLM initialise sur %s (%s) avec le modele %s.",
            self.host,
            self.backend,
            self.model,
        )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        """
        Lit un entier depuis l'environnement.

        Args:
            name (str): Nom de la variable d'environnement.
            default (int): Valeur de repli si la variable est absente ou invalide.

        Returns:
            int: Valeur convertie ou repli.
        """
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except ValueError:
            logger.warning("Variable %s invalide (%s). Repli sur %s.", name, raw_value, default)
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        """
        Lit un flottant depuis l'environnement.

        Args:
            name (str): Nom de la variable d'environnement.
            default (float): Valeur de repli.

        Returns:
            float: Valeur convertie ou repli.
        """
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return float(raw_value)
        except ValueError:
            logger.warning("Variable %s invalide (%s). Repli sur %s.", name, raw_value, default)
            return default

    @staticmethod
    def _build_base_url(host: str | None, default_host: str, default_port: int) -> str:
        """
        Construit une URL de base a partir d'un hote ou d'une URL brute.

        Args:
            host (str | None): Hote ou URL potentiellement fournie.
            default_host (str): Hote par defaut issu des settings.
            default_port (int): Port par defaut issu des settings.

        Returns:
            str: URL de base normalisee, sans slash final.
        """
        candidate = (host or "").strip()
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate.rstrip("/")
        resolved_host = candidate or default_host
        return f"http://{resolved_host}:{default_port}".rstrip("/")

    def _cooldown_active(self) -> bool:
        """
        Indique si le backend est encore dans une fenetre de reprise.

        Returns:
            bool: True si un cooldown est actif.
        """
        return time.time() < self._cooldown_until

    def _mark_failure(self, reason: str) -> None:
        """
        Active un cooldown apres une indisponibilite repetee.

        Args:
            reason (str): Motif principal de l'indisponibilite.
        """
        self._last_failure_reason = reason
        self._cooldown_until = time.time() + self.failure_cooldown_seconds

    def _clear_failure(self) -> None:
        """Supprime l'etat de cooldown apres un appel reussi."""
        self._last_failure_reason = ""
        self._cooldown_until = 0.0

    async def analyze(self, context: str, prompt: str) -> str:
        """
        Soumet une analyse au backend LLM actif.

        Args:
            context (str): Contexte deja serialise a transmettre au modele.
            prompt (str): Instruction utilisateur ou systeme a appliquer.

        Returns:
            str: Reponse textuelle du backend, ou message de repli si le LLM est
            indisponible.
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

        if self._cooldown_active():
            remaining = max(1, int(self._cooldown_until - time.time()))
            logger.warning(
                "Backend LLM %s en phase de reprise pendant encore %ss (%s). Repli heuristique conserve.",
                self.backend,
                remaining,
                self._last_failure_reason or "indisponibilite reseau",
            )
            return f"Mode heuristique temporaire: {self.backend} indisponible"

        try:
            async with aiohttp.ClientSession() as session:
                if self.backend == "vllm":
                    return await self._analyze_vllm_with_retries(session, payload)

                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=self.request_timeout_seconds,
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        self._clear_failure()
                        return result.get("response", "").strip()

                    logger.error("Erreur LLM: HTTP %s - %s", resp.status, await resp.text())
                    self._mark_failure(f"http_{resp.status}")
                    return f"Mode heuristique temporaire: {self.backend} indisponible"
        except Exception as exc:
            logger.error("Connexion LLM impossible: %s", exc)
            self._mark_failure(str(exc))
            return f"Mode heuristique temporaire: {self.backend} indisponible"

    async def _analyze_vllm_with_retries(
        self,
        session: aiohttp.ClientSession,
        payload: dict[str, Any],
    ) -> str:
        """
        Tente plusieurs appels `vLLM` avant d'activer un cooldown.

        Args:
            session (aiohttp.ClientSession): Session HTTP reutilisee.
            payload (dict[str, Any]): Charge utile OpenAI-compatible.

        Returns:
            str: Reponse du modele ou message de repli.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = await self._analyze_vllm(session, payload)
                self._clear_failure()
                return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exception = exc
                if attempt < self.retry_attempts:
                    logger.warning(
                        "vLLM indisponible (%s/%s): %s. Nouvelle tentative dans %.1fs.",
                        attempt,
                        self.retry_attempts,
                        exc,
                        self.retry_delay_seconds,
                    )
                    await asyncio.sleep(self.retry_delay_seconds)
                    continue

        if last_exception is not None:
            self._mark_failure(str(last_exception))
            logger.warning(
                "Connexion vLLM impossible apres %s tentatives: %s",
                self.retry_attempts,
                last_exception,
            )
        return f"Mode heuristique temporaire: {self.backend} indisponible"

    async def _analyze_vllm(self, session: aiohttp.ClientSession, payload: dict[str, Any]) -> str:
        """
        Execute un appel `vLLM` avec repli automatique si le modele est absent.

        Args:
            session (aiohttp.ClientSession): Session HTTP reutilisee.
            payload (dict[str, Any]): Charge utile OpenAI-compatible.

        Returns:
            str: Reponse brute du modele.

        Raises:
            aiohttp.ClientError: Si la connexion ou la requete echoue.
            asyncio.TimeoutError: Si l'appel depasse le delai.
        """
        async with session.post(
            self.api_url,
            json=payload,
            timeout=self.request_timeout_seconds,
        ) as resp:
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

                    async with session.post(
                        self.api_url,
                        json=retry_payload,
                        timeout=self.request_timeout_seconds,
                    ) as retry_resp:
                        if retry_resp.status == 200:
                            retry_result = await retry_resp.json()
                            return retry_result["choices"][0]["message"]["content"].strip()

                        logger.error(
                            "Erreur LLM apres fallback: HTTP %s - %s",
                            retry_resp.status,
                            await retry_resp.text(),
                        )
                        return f"Mode heuristique temporaire: {self.backend} indisponible"

            logger.error("Erreur LLM: HTTP %s - %s", resp.status, error_text)
            return f"Mode heuristique temporaire: {self.backend} indisponible"

    async def _discover_vllm_available_model(self, session: aiohttp.ClientSession) -> Optional[str]:
        """
        Interroge `/v1/models` pour trouver un modele disponible sur `vLLM`.

        Args:
            session (aiohttp.ClientSession): Session HTTP reutilisee.

        Returns:
            Optional[str]: Identifiant d'un modele disponible si la decouverte
            reussit, sinon `None`.
        """
        models_url = f"{self.host}/v1/models"
        try:
            async with session.get(models_url, timeout=10.0) as resp:
                if resp.status != 200:
                    logger.warning("Impossible de lister les modeles vLLM (HTTP %s).", resp.status)
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
        Detecte les erreurs 404 liees a un modele absent.

        Args:
            error_text (str): Corps de reponse renvoye par l'API.

        Returns:
            bool: True si le texte ressemble a une erreur de modele introuvable.
        """
        lowered = (error_text or "").lower()
        return "does not exist" in lowered or "model" in lowered and "notfound" in lowered

    async def get_strategy_signal(self, market_data: dict[str, Any]) -> str:
        """
        Produit un signal strategique simple pour un contexte de marche.

        Args:
            market_data (dict[str, Any]): Donnees de marche deja preparees.

        Returns:
            str: Reponse textuelle du LLM ou message de repli.
        """
        context = json.dumps(market_data, indent=2)
        prompt = (
            "You are a Senior Trading Strategist. Analyze the provided M15 Market Data. "
            "Identify the dominant trend (Bullish/Bearish/Neutral) and Key Support/Resistance levels. "
            "Output a concise directive for the execution engine: 'BUY_ONLY', 'SELL_ONLY', or 'NEUTRAL'. "
            "Justify directly."
        )
        return await self.analyze(context, prompt)
