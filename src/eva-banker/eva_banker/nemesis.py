"""
Nemesis System - Apprentissage adaptatif des defaites.

Inspire du Nemesis System de "Shadow of Mordor".
Chaque perte en trading est analysee, classifiee et memorisee.
Quand un type d'ennemi bat EVA plusieurs fois dans une fenetre recente,
une phase de meditation est declenchee pour suspendre temporairement le
trading et lancer un apprentissage correctif.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NemesisSystem:
    """
    Gere la memoire des defaites et le blocage anti-tilt adaptatif.

    Le systeme opere sur une fenetre glissante recente plutot qu'un compteur a
    vie. Cela evite de retrigger une meditation sur des pertes anciennes deja
    traitees.
    """

    def __init__(self) -> None:
        """
        Initialise les seuils et les structures de suivi Nemesis.
        """
        self.block_duration_seconds = max(
            300,
            int(os.getenv("BANKER_NEMESIS_BLOCK_SECONDS", "3600")),
        )
        self.trigger_threshold = max(
            2,
            int(os.getenv("BANKER_NEMESIS_TRIGGER_THRESHOLD", "3")),
        )
        self.lookback_hours = max(
            1,
            int(os.getenv("BANKER_NEMESIS_LOOKBACK_HOURS", "12")),
        )
        self.max_ledger_entries = max(
            50,
            int(os.getenv("BANKER_NEMESIS_MAX_LEDGER_ENTRIES", "200")),
        )
        self.defeat_ledger: List[Dict[str, Any]] = []
        self.known_nemeses: Dict[str, int] = {}
        self.lifetime_nemeses: Dict[str, int] = {}
        self.last_meditation_by_type: Dict[str, str] = {}
        self.trading_blocked_until: Optional[datetime] = None
        self._meditation_in_progress = False

    async def report_loss(
        self,
        trade_id: str,
        loss_amount: float,
        market_context: Dict[str, Any],
    ) -> None:
        """
        Enregistre une defaite et met a jour l'etat Nemesis.

        Args:
            trade_id (str): Identifiant unique du trade cloture.
            loss_amount (float): Montant de la perte constatee.
            market_context (Dict[str, Any]): Contexte de marche associe.
        """
        if any(entry.get("trade_id") == trade_id for entry in self.defeat_ledger):
            logger.info("Nemesis ignore un doublon de defaite pour le trade %s.", trade_id)
            return

        now = datetime.now()
        nemesis_type = self._classify_nemesis(market_context)
        defeat_entry = {
            "timestamp": now.isoformat(),
            "trade_id": trade_id,
            "loss": loss_amount,
            "context": market_context,
            "nemesis_type": nemesis_type,
        }
        self.defeat_ledger.append(defeat_entry)
        self._trim_ledger()

        self.lifetime_nemeses[nemesis_type] = self.lifetime_nemeses.get(nemesis_type, 0) + 1
        self._refresh_recent_nemeses(now=now)
        recent_count = self.known_nemeses.get(nemesis_type, 0)

        logger.warning(
            "Nemesis '%s' detecte (trade %s). Defaites recentes: %s/%s.",
            nemesis_type,
            trade_id,
            recent_count,
            self.trigger_threshold,
        )

        if recent_count >= self.trigger_threshold:
            await self._trigger_meditation(nemesis_type, now=now)

        await self._save_state()

    def _classify_nemesis(self, context: Dict[str, Any]) -> str:
        """
        Classifie la cause probable de la perte.

        Args:
            context (Dict[str, Any]): Contexte capture lors de la perte.

        Returns:
            str: Type de Nemesis detecte.
        """
        volatility = context.get("volatility", 0)
        news_event = context.get("news_event", False)
        trend_reversal = context.get("trend_reversal", False)

        if news_event:
            return "BLACK_SWAN_NEMESIS"
        if trend_reversal:
            return "TREND_REVERSAL"
        if volatility > 0.03:
            return "WHIPLASH_VOLATILITY"
        return "LIQUIDITY_TRAP"

    async def _trigger_meditation(
        self,
        nemesis_type: str,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Declenche une meditation si aucun blocage n'est deja actif.

        Args:
            nemesis_type (str): Type de pattern responsable du blocage.
            now (Optional[datetime]): Horodatage de reference si deja calcule.
        """
        if self.should_block_trading():
            logger.info(
                "Meditation deja active. Blocage conserve pour le type %s.",
                nemesis_type,
            )
            return

        now = now or datetime.now()
        self._meditation_in_progress = True
        self.trading_blocked_until = now + timedelta(seconds=self.block_duration_seconds)
        self.last_meditation_by_type[nemesis_type] = now.isoformat()
        self._refresh_recent_nemeses(now=now)
        logger.info(
            "Meditation declenchee pour Nemesis '%s'. Trading bloque jusqu'a %s.",
            nemesis_type,
            self.trading_blocked_until.strftime("%H:%M"),
        )

        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            await redis.broadcast_to_swarm(
                source="banker",
                action="NEMESIS_MEDITATION",
                payload={
                    "nemesis_type": nemesis_type,
                    "defeats": self.known_nemeses.get(nemesis_type, 0),
                    "blocked_until": self.trading_blocked_until.isoformat(),
                },
            )
        except Exception as exc:
            logger.error("Erreur notification meditation: %s", exc)

    def should_block_trading(self) -> bool:
        """
        Indique si le trading doit rester bloque.

        Returns:
            bool: True si une meditation est active.
        """
        if self.trading_blocked_until and datetime.now() < self.trading_blocked_until:
            return True
        if self.trading_blocked_until and datetime.now() >= self.trading_blocked_until:
            self._meditation_in_progress = False
            self.trading_blocked_until = None
            self._refresh_recent_nemeses()
        return False

    async def _save_state(self) -> None:
        """
        Persiste l'etat courant dans Redis si disponible.
        """
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            await redis.cache_set(
                "nemesis:state",
                {
                    "defeat_ledger": self.defeat_ledger[-self.max_ledger_entries :],
                    "known_nemeses": self.known_nemeses,
                    "lifetime_nemeses": self.lifetime_nemeses,
                    "last_meditation_by_type": self.last_meditation_by_type,
                    "trading_blocked_until": (
                        self.trading_blocked_until.isoformat()
                        if self.trading_blocked_until
                        else None
                    ),
                },
                ttl_seconds=86400,
            )
        except Exception:
            pass

    async def load_state(self) -> None:
        """
        Recharge l'etat depuis Redis au demarrage.
        """
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            state = await redis.cache_get("nemesis:state")
            if not state:
                return

            self.defeat_ledger = state.get("defeat_ledger", [])
            self.known_nemeses = state.get("known_nemeses", {})
            self.lifetime_nemeses = state.get(
                "lifetime_nemeses",
                state.get("known_nemeses", {}),
            )
            self.last_meditation_by_type = state.get("last_meditation_by_type", {})
            if state.get("trading_blocked_until"):
                self.trading_blocked_until = datetime.fromisoformat(
                    state["trading_blocked_until"]
                )
                self._meditation_in_progress = datetime.now() < self.trading_blocked_until

            if not self.last_meditation_by_type and self.known_nemeses:
                now_iso = datetime.now().isoformat()
                self.last_meditation_by_type = {
                    nemesis_type: now_iso
                    for nemesis_type, defeats in self.known_nemeses.items()
                    if int(defeats) >= self.trigger_threshold
                }

            self._trim_ledger()
            self._refresh_recent_nemeses()
            logger.info("Etat Nemesis charge depuis Redis.")
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """
        Retourne l'etat courant du systeme Nemesis.

        Returns:
            Dict[str, Any]: Blocage, compteurs recents et historique recent.
        """
        self._refresh_recent_nemeses()
        return {
            "total_defeats": len(self.defeat_ledger),
            "known_nemeses": self.known_nemeses,
            "lifetime_nemeses": self.lifetime_nemeses,
            "trading_blocked": self.should_block_trading(),
            "blocked_until": (
                self.trading_blocked_until.isoformat()
                if self.trading_blocked_until
                else None
            ),
            "meditation_active": self._meditation_in_progress,
            "trigger_threshold": self.trigger_threshold,
            "lookback_hours": self.lookback_hours,
            "recent_defeats": self.defeat_ledger[-5:],
        }

    def _trim_ledger(self) -> None:
        """
        Tronque le journal des defaites pour garder une taille bornee.
        """
        if len(self.defeat_ledger) > self.max_ledger_entries:
            self.defeat_ledger = self.defeat_ledger[-self.max_ledger_entries :]

    def _refresh_recent_nemeses(self, now: Optional[datetime] = None) -> None:
        """
        Recalcule les compteurs operationnels sur une fenetre glissante.

        Args:
            now (Optional[datetime]): Horodatage de reference optionnel.
        """
        now = now or datetime.now()
        lookback_start = now - timedelta(hours=self.lookback_hours)
        recent_counts: Dict[str, int] = {}

        for entry in self.defeat_ledger:
            entry_time = self._parse_timestamp(entry.get("timestamp"))
            if entry_time is None or entry_time < lookback_start:
                continue

            nemesis_type = str(entry.get("nemesis_type") or "UNKNOWN")
            meditation_start = self._parse_timestamp(
                self.last_meditation_by_type.get(nemesis_type)
            )
            if meditation_start and entry_time <= meditation_start:
                continue
            recent_counts[nemesis_type] = recent_counts.get(nemesis_type, 0) + 1

        self.known_nemeses = recent_counts

    @staticmethod
    def _parse_timestamp(raw_value: Optional[str]) -> Optional[datetime]:
        """
        Convertit un timestamp ISO en datetime.

        Args:
            raw_value (Optional[str]): Horodatage au format ISO.

        Returns:
            Optional[datetime]: Horodatage parse ou None si invalide.
        """
        if not raw_value:
            return None
        try:
            return datetime.fromisoformat(raw_value)
        except ValueError:
            logger.warning("Timestamp Nemesis invalide ignore: %s", raw_value)
            return None


_nemesis_instance: Optional[NemesisSystem] = None


def get_nemesis_system() -> NemesisSystem:
    """
    Retourne l'instance singleton du systeme Nemesis.

    Returns:
        NemesisSystem: Instance partagee du systeme Nemesis.
    """
    global _nemesis_instance
    if _nemesis_instance is None:
        _nemesis_instance = NemesisSystem()
    return _nemesis_instance