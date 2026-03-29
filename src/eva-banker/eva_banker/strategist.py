"""Strategiste macro du banker.

Ce module centralise la lecture du contexte macro local, l'appel optionnel au
LLM de contexte et la fusion prudente des biais ``Cortex + GNN``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import unicodedata
import uuid
from datetime import datetime
from typing import Any

from eva_banker.services.mt5 import MT5Service
from shared import get_settings
from shared.indicators import IndicatorFactory
from shared.llm_client import LLMClient
from shared.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class Strategist:
    """Analyse le contexte macro M15 et produit un biais exploitable.

    En mode standard, le strategist interroge le Cortex LLM puis fusionne son
    biais avec le GNN multi-horizon.

    En mode `cpu_live`, le strategist peut soit desactiver completement le
    Cortex, soit utiliser un Cortex `ollama` sur CPU. Dans tous les cas, ce
    Cortex reste strictement consultatif : le live continue a etre pilote par
    MuZero et aucun veto directionnel fort n'est ajoute.
    """

    def __init__(self, mt5_service: MT5Service):
        """Initialise le strategist.

        Args:
            mt5_service (MT5Service): Service MT5 utilise pour lire les bougies.
        """
        self.mt5 = mt5_service
        settings = get_settings()
        self._training_compat_mode = self._resolve_training_compat_mode(
            os.getenv("BANKER_TRAINING_COMPAT_MODE", "disabled")
        )
        self._cpu_live_mode = self._training_compat_mode == "cpu_live"
        self._cpu_live_cortex_mode = self._resolve_cpu_live_cortex_mode(
            os.getenv("BANKER_CPU_LIVE_CORTEX_MODE", "ollama")
        )
        self._cpu_live_cortex_timeout_seconds = self._resolve_cpu_live_cortex_timeout_seconds()
        self._cpu_live_cortex_endpoint = self._resolve_cpu_live_cortex_endpoint(settings)
        self.cortex_model = self._resolve_cortex_model(settings)
        self.cortex_backend = settings.llm_backend
        self.cortex = self._build_cortex_client(settings)
        self.latest_strategy: dict[str, dict[str, Any]] = {}
        self.last_update: datetime = datetime.min

        if self._cpu_live_mode:
            if self.cortex is None:
                logger.info(
                    "Mode cpu_live actif: Cortex desactive, GNN consultatif, contexte local uniquement."
                )
            else:
                logger.info(
                    "Mode cpu_live actif: Cortex consultatif via %s (%s), timeout %.1fs.",
                    self.cortex_backend,
                    self.cortex_model,
                    self._cpu_live_cortex_timeout_seconds,
                )
        else:
            logger.info("Cortex bancaire initialise avec le modele: %s", self.cortex_model)

    @staticmethod
    def _resolve_training_compat_mode(raw_mode: str) -> str:
        """Normalise le mode de compatibilite trading/training.

        Args:
            raw_mode (str): Valeur brute issue de l'environnement.

        Returns:
            str: Mode retenu (`disabled` ou `cpu_live`).
        """
        normalized = str(raw_mode or "").strip().lower()
        if normalized in {"", "0", "false", "off", "none"}:
            return "disabled"
        if normalized == "cpu_live":
            return normalized
        logger.warning(
            "Mode de compatibilite banker inconnu (%s). Repli sur disabled.",
            raw_mode,
        )
        return "disabled"

    @staticmethod
    def _resolve_cpu_live_cortex_mode(raw_mode: str) -> str:
        """Normalise le mode du Cortex consultatif en `cpu_live`.

        Args:
            raw_mode (str): Valeur brute issue de l'environnement.

        Returns:
            str: Mode retenu (`disabled` ou `ollama`).
        """
        normalized = str(raw_mode or "").strip().lower()
        if normalized in {"", "0", "false", "off", "none", "disabled"}:
            return "disabled"
        if normalized == "ollama":
            return normalized
        logger.warning(
            "Mode BANKER_CPU_LIVE_CORTEX_MODE inconnu (%s). Repli sur disabled.",
            raw_mode,
        )
        return "disabled"

    @staticmethod
    def _resolve_cpu_live_cortex_timeout_seconds() -> float:
        """Lit le timeout du Cortex CPU consultatif.

        Returns:
            float: Delai maximal d'un appel CPU consultatif.
        """
        raw_timeout = os.getenv("BANKER_CPU_LIVE_CORTEX_TIMEOUT_SECONDS", "4.0").strip()
        try:
            return max(1.0, float(raw_timeout))
        except ValueError:
            logger.warning(
                "Timeout BANKER_CPU_LIVE_CORTEX_TIMEOUT_SECONDS invalide (%s). Repli sur 4.0s.",
                raw_timeout,
            )
            return 4.0

    @staticmethod
    def _resolve_cpu_live_cortex_endpoint(settings) -> str:
        """Construit l'endpoint du Cortex CPU consultatif.

        Args:
            settings: Configuration partagee.

        Returns:
            str: URL de base a utiliser pour `ollama`.
        """
        explicit_url = os.getenv("BANKER_CPU_LIVE_CORTEX_URL", "").strip()
        if explicit_url:
            return explicit_url

        host = os.getenv("BANKER_CPU_LIVE_CORTEX_HOST", "").strip() or "127.0.0.1"
        port = os.getenv("BANKER_CPU_LIVE_CORTEX_PORT", "").strip() or str(settings.ollama_port)
        return f"http://{host}:{port}"

    def _build_cortex_client(self, settings) -> LLMClient | None:
        """Construit le client Cortex adapte au mode courant.

        Args:
            settings: Configuration partagee.

        Returns:
            LLMClient | None: Client LLM actif ou `None` si le Cortex reste coupe.
        """
        if self._cpu_live_mode:
            if self._cpu_live_cortex_mode != "ollama":
                self.cortex_backend = "disabled"
                return None

            self.cortex_backend = "ollama"
            self.cortex_model = (
                os.getenv("BANKER_CPU_LIVE_CORTEX_MODEL", "").strip()
                or settings.ollama_model
            )
            return LLMClient(
                model=self.cortex_model,
                host=self._cpu_live_cortex_endpoint,
                backend="ollama",
                request_timeout_seconds=self._cpu_live_cortex_timeout_seconds,
            )

        self.cortex_backend = settings.llm_backend
        return LLMClient(model=self.cortex_model)

    @staticmethod
    def _resolve_cortex_model(settings) -> str:
        """Resout le modele LLM du Cortex.

        Args:
            settings: Configuration partagee.

        Returns:
            str: Identifiant du modele a utiliser.
        """
        direct_model = os.getenv("BANKER_CORTEX_MODEL", "").strip()
        if direct_model:
            return direct_model

        if settings.llm_backend == "vllm":
            candidate = os.getenv("COUNCIL_MODEL_BANKER", settings.council_model_banker).strip()
            if candidate and ":" in candidate and "/" not in candidate:
                logger.warning(
                    "COUNCIL_MODEL_BANKER=%s ressemble a un modele Ollama; repli sur %s.",
                    candidate,
                    settings.vllm_model,
                )
                return settings.vllm_model
            return candidate or settings.vllm_model

        return settings.ollama_model

    def get_runtime_status(self) -> dict[str, Any]:
        """Expose l'etat du Cortex pour les APIs du banker.

        Returns:
            dict[str, Any]: Mode, backend, exigence et endpoint du Cortex.
        """
        if self._cpu_live_mode:
            if self.cortex is None:
                return {
                    "mode": "disabled",
                    "backend": "none",
                    "required": False,
                    "consultative": True,
                    "model": None,
                    "endpoint": None,
                    "timeout_seconds": self._cpu_live_cortex_timeout_seconds,
                }
            return {
                "mode": "consultatif_cpu",
                "backend": self.cortex_backend,
                "required": False,
                "consultative": True,
                "model": self.cortex_model,
                "endpoint": self._cpu_live_cortex_endpoint,
                "timeout_seconds": self._cpu_live_cortex_timeout_seconds,
            }

        return {
            "mode": "live",
            "backend": self.cortex_backend,
            "required": True,
            "consultative": False,
            "model": self.cortex_model,
            "endpoint": getattr(self.cortex, "host", None) if self.cortex is not None else None,
            "timeout_seconds": getattr(self.cortex, "request_timeout_seconds", None),
        }

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        """Retire les balises Markdown autour d'un JSON eventuel.

        Args:
            content (str): Texte brut renvoye par le LLM.

        Returns:
            str: Texte nettoye.
        """
        cleaned = (content or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return cleaned.strip()

    @staticmethod
    def _sanitize_reasoning_text(content: str) -> str:
        """Normalise un texte libre pour les logs et Telegram.

        Args:
            content (str): Texte a nettoyer.

        Returns:
            str: Texte ASCII compact.
        """
        cleaned = " ".join(str(content or "").replace("\n", " ").split()).strip(" -:")
        ascii_cleaned = (
            unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
        )
        return ascii_cleaned.strip(" -:")

    @classmethod
    def _parse_cortex_response(cls, response: str) -> tuple[str, str]:
        """Extrait un biais et une synthese depuis la reponse du Cortex.

        Args:
            response (str): Reponse brute du modele.

        Returns:
            tuple[str, str]: Biais normalise et raison courte.
        """
        cleaned = cls._strip_json_fence(response)
        if not cleaned:
            return "NEUTRAL", "Synthese Cortex indisponible."

        payload_text = cleaned
        if "{" in cleaned and "}" in cleaned:
            payload_text = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            parsed = {}

        bias_source = str(parsed.get("bias", cleaned)).upper()
        reason = str(parsed.get("reason", cleaned)).strip()
        details = str(parsed.get("details", "")).strip()

        bias = "NEUTRAL"
        if "BULLISH" in bias_source:
            bias = "BULLISH"
        elif "BEARISH" in bias_source:
            bias = "BEARISH"
        elif "RANGING" in bias_source or "RANGE" in bias_source:
            bias = "RANGING"

        normalized_reason = cls._sanitize_reasoning_text(reason)
        normalized_details = cls._sanitize_reasoning_text(details)
        if (
            normalized_details
            and normalized_details.lower() != normalized_reason.lower()
            and normalized_details.lower() not in normalized_reason.lower()
        ):
            normalized_reason = f"{normalized_reason}. {normalized_details}".strip(". ")
        if not normalized_reason:
            normalized_reason = "Synthese Cortex indisponible."

        return bias, normalized_reason

    async def analyze_market_context(self, symbol: str) -> dict[str, Any]:
        """Analyse le contexte M15 et retourne un biais macro fusionne.

        Args:
            symbol (str): Symbole a analyser.

        Returns:
            dict[str, Any]: Biais Cortex, biais GNN, biais final et metadonnees.
        """
        logger.info("Cortex: analyse du contexte macro pour %s.", symbol)

        candles = await self.mt5.get_recent_candles(symbol, timeframe=15, count=100)
        if not candles:
            logger.warning("Cortex: aucune donnee M15 disponible pour %s.", symbol)
            strategy = {
                "symbol": symbol,
                "cortex_bias": "NEUTRAL",
                "gnn_bias": "NEUTRAL",
                "gnn_scalp_bias": "NEUTRAL",
                "gnn_intraday_bias": "NEUTRAL",
                "gnn_swing_bias": "NEUTRAL",
                "gnn_confidence": 0.0,
                "bias": "NEUTRAL",
                "bias_alignment": "no_data",
                "bias_strength": "weak",
                "raw_thought": "Aucune donnee M15 disponible.",
                "raw_response": "",
                "compat_mode": self._training_compat_mode,
                "cortex_required": not self._cpu_live_mode,
                "timestamp": datetime.now().isoformat(),
            }
            self.latest_strategy[symbol] = strategy
            return strategy

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["tick_volume"] for c in candles]

        rsi = float(IndicatorFactory.rsi(closes, 14).iloc[-1])
        trend_ema = float(IndicatorFactory.ema(closes, 50).iloc[-1])
        current_price = float(closes[-1])
        vwap = float(IndicatorFactory.vwap(highs, lows, closes, volumes).iloc[-1])
        adx_data = IndicatorFactory.adx(highs, lows, closes, 14)
        adx_val = float(adx_data["adx"].iloc[-1])
        fibs = IndicatorFactory.get_fibonacci_levels(highs, lows, 50)

        trend = "BULLISH" if current_price > trend_ema else "BEARISH"
        context = {
            "symbol": symbol,
            "timeframe": "M15",
            "price": current_price,
            "trend_50ema": trend,
            "rsi_14": round(rsi, 2),
            "vwap": round(vwap, 2) if not math.isnan(vwap) else current_price,
            "adx_trend_strength": round(adx_val, 2),
            "fib_382": round(fibs.get("fib_382", 0), 2),
            "fib_618": round(fibs.get("fib_618", 0), 2),
            "last_5_candles": [round(c["close"], 2) for c in candles[-5:]],
        }

        response = ""
        if self.cortex is None:
            cortex_bias = "NEUTRAL"
            cortex_reason = (
                "Mode CPU live: Cortex desactive pendant l'entrainement GPU. "
                "Le GNN reste consultatif et ne force aucune direction."
            )
        else:
            prompt = (
                f"Analyse la structure M15 de {symbol}. "
                f"La tendance 50 EMA est {trend}. RSI={rsi:.1f}. ADX={adx_val:.1f}. "
                f"Le prix vaut {current_price:.2f} pour une VWAP a {context['vwap']:.2f}. "
                f"Les niveaux de Fibonacci clefs sont {context['fib_382']:.2f} et {context['fib_618']:.2f}. "
                "Determine le biais strategique parmi BULLISH, BEARISH ou RANGING. "
                "Tiens compte des rejets de VWAP et du seuil ADX > 25 pour la poursuite de tendance. "
                "Reponds en JSON strict sur une seule ligne, sans Markdown, avec "
                "{\"bias\":\"BULLISH|BEARISH|RANGING\",\"reason\":\"premiere phrase courte en francais\","
                "\"details\":\"seconde phrase courte en francais avec le niveau ou le signal cle\"}."
            )
            response = await self.cortex.analyze(json.dumps(context), prompt)
            cortex_bias, cortex_reason = self._parse_cortex_response(response)

        gnn_intraday = "NEUTRAL"
        gnn_scalp = "NEUTRAL"
        gnn_swing = "NEUTRAL"
        gnn_confidence = 0.0
        try:
            import aiohttp

            lab_host = os.getenv("LAB_HOST", "localhost")
            url = f"http://{lab_host}:8600/gnn/predict"
            payload = {
                "assets_data": {
                    f"{symbol}_5": [closes[-15:]],
                    f"{symbol}_60": [closes[-15:]],
                    f"{symbol}_1440": [closes[-15:]],
                }
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        gnn_scalp = str(data.get("scalp", {}).get("bias", "NEUTRAL"))
                        gnn_intraday = str(data.get("intraday", {}).get("bias", "NEUTRAL"))
                        gnn_swing = str(data.get("swing", {}).get("bias", "NEUTRAL"))
                        gnn_confidence = float(
                            data.get("intraday", {}).get("confidence", 0.0) or 0.0
                        )
                    else:
                        logger.warning("GNN indisponible pour %s: HTTP %s.", symbol, resp.status)
        except Exception as exc:
            logger.warning("Connexion GNN impossible pour %s: %s.", symbol, exc.__class__.__name__)

        gnn_bias = gnn_intraday
        final_bias = cortex_bias
        bias_alignment = "cortex_only"
        bias_strength = "weak"

        if cortex_bias in {"BULLISH", "BEARISH"}:
            bias_strength = "moderate"

        if gnn_swing != "NEUTRAL" and cortex_bias in {"BULLISH", "BEARISH"}:
            if gnn_swing == cortex_bias:
                bias_alignment = "swing_confirme"
                bias_strength = "strong"
            else:
                final_bias = "RANGING"
                bias_alignment = "swing_conflict"
                bias_strength = "weak"
        elif gnn_bias != "NEUTRAL" and gnn_confidence >= 0.55:
            if gnn_bias == cortex_bias and cortex_bias in {"BULLISH", "BEARISH"}:
                bias_alignment = "aligned"
                bias_strength = "strong" if gnn_confidence >= 0.75 else "moderate"
            elif cortex_bias in {"NEUTRAL", "RANGING"}:
                if gnn_confidence >= 0.90 and gnn_swing in {"NEUTRAL", gnn_bias}:
                    final_bias = gnn_bias
                    bias_alignment = "gnn_confirmed"
                    bias_strength = "moderate"
                else:
                    final_bias = cortex_bias
                    bias_alignment = "gnn_soft_hint"
                    bias_strength = "weak"
            else:
                final_bias = "RANGING"
                bias_alignment = "intraday_conflict"
                bias_strength = "weak"

        if self._cpu_live_mode:
            if self.cortex is not None and cortex_bias in {"BULLISH", "BEARISH"}:
                if gnn_bias == cortex_bias and gnn_confidence >= 0.55:
                    final_bias = "RANGING"
                    bias_alignment = "cpu_live_conseil_aligne"
                elif gnn_bias != "NEUTRAL" and gnn_confidence >= 0.55 and gnn_bias != cortex_bias:
                    final_bias = "RANGING"
                    bias_alignment = "cpu_live_conseil_conflit"
                else:
                    final_bias = "NEUTRAL"
                    bias_alignment = "cpu_live_cortex_consultatif"
            elif gnn_bias != "NEUTRAL" and gnn_confidence >= 0.55:
                final_bias = "RANGING"
                bias_alignment = "cpu_live_gnn_consultatif"
            else:
                final_bias = "NEUTRAL"
                bias_alignment = "cpu_live_neutral"
            bias_strength = "weak"

        strategy = {
            "symbol": symbol,
            "cortex_bias": cortex_bias,
            "gnn_bias": gnn_bias,
            "gnn_scalp_bias": gnn_scalp,
            "gnn_intraday_bias": gnn_intraday,
            "gnn_swing_bias": gnn_swing,
            "gnn_confidence": float(gnn_confidence or 0.0),
            "bias": final_bias,
            "bias_alignment": bias_alignment,
            "bias_strength": bias_strength,
            "raw_thought": cortex_reason,
            "raw_response": response,
            "compat_mode": self._training_compat_mode,
            "cortex_required": not self._cpu_live_mode,
            "cortex_backend": self.cortex_backend,
            "cortex_mode": self.get_runtime_status().get("mode"),
            "timestamp": datetime.now().isoformat(),
        }
        self.latest_strategy[symbol] = strategy
        self.last_update = datetime.now()

        logger.info(
            "Strategie %s: cortex=%s gnn=%s final=%s alignement=%s mode=%s",
            symbol,
            cortex_bias,
            gnn_bias,
            final_bias,
            bias_alignment,
            self._training_compat_mode,
        )

        try:
            redis = get_redis_client()
            asyncio.create_task(
                redis.publish(
                    "eva.cortex.feed",
                    {
                        "id": str(uuid.uuid4()),
                        "source_agent": "Strategist",
                        "action": (
                            f"Cortex M15 {symbol}: {final_bias} "
                            f"(Cortex={cortex_bias}, GNN={gnn_bias}) -> {cortex_reason[:100]}"
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "type": "thought",
                    },
                )
            )
        except Exception as exc:
            logger.debug("Publication feed strategist impossible: %s", exc)

        return strategy

    def get_bias(self, symbol: str) -> str:
        """Retourne le dernier biais connu pour un symbole.

        Args:
            symbol (str): Symbole cible.

        Returns:
            str: Biais courant ou `NEUTRAL`.
        """
        return self.latest_strategy.get(symbol, {}).get("bias", "NEUTRAL")

    async def get_micro_reasoning(self, symbol: str, action: str, indicators: dict) -> str:
        """Genere une synthese courte pour Telegram.

        Args:
            symbol (str): Symbole traite.
            action (str): Action candidate.
            indicators (dict): Indicateurs utiles au commentaire.

        Returns:
            str: Phrase courte exploitable dans Telegram.
        """
        rsi = float(indicators.get("RSI", 50) or 50)
        adx = float(indicators.get("adx", 25) or 25)
        macd = float(indicators.get("MACD_Hist", 0) or 0)

        if self._cpu_live_mode:
            last_strategy = self.latest_strategy.get(symbol, {})
            cortex_bias = str(last_strategy.get("cortex_bias") or "NEUTRAL")
            gnn_bias = str(last_strategy.get("gnn_bias") or "NEUTRAL")
            if self.cortex is not None:
                rationale = self._sanitize_reasoning_text(last_strategy.get("raw_thought", ""))[:140]
                detail = rationale or f"RSI {rsi:.1f}, ADX {adx:.1f} et MACD {macd:.4f}."
                return (
                    f"Mode CPU live: {action} sur {symbol} avec Cortex CPU {cortex_bias}, "
                    f"GNN {gnn_bias}. {detail}"
                )
            return (
                f"Mode CPU live: {action} sur {symbol} conserve en demo avec RSI {rsi:.1f}, "
                f"ADX {adx:.1f} et MACD {macd:.4f}."
            )

        if self.cortex is None:
            return "Le signal technique reste coherent avec le contexte courant."

        prompt = (
            f"Explique en une seule phrase courte, en francais, pourquoi ouvrir un {action} sur {symbol}. "
            f"Contexte: RSI={rsi:.1f}, ADX={adx:.1f}, MACD hist={macd:.4f}. "
            "Reponds uniquement avec la phrase finale, sans JSON, sans puce et sans titre."
        )

        try:
            reasoning = await self.cortex.analyze(f"Action: {action} on {symbol}", prompt)
            cleaned = " ".join(str(reasoning).replace("*", " ").split())
            return (
                cleaned.split(". ")[0][:200].strip(" -:")
                or "Le signal technique reste coherent avec le contexte courant."
            )
        except Exception as exc:
            logger.warning("Generation du micro-raisonnement impossible: %s", exc)
            return "Le signal technique reste coherent avec le contexte courant."
