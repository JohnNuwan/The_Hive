"""
Nemesis System — Apprentissage adaptatif des défaites
════════════════════════════════════════════════════

Inspiré du Nemesis System de 'Shadow of Mordor'.
Chaque perte en trading est analysée, classifiée, et mémorisée.
Quand un type d'ennemi (pattern de marché) bat EVA 3+ fois,
une phase de Méditation (retraining ciblé) est déclenchée.

Types de Nemesis :
  - BLACK_SWAN_NEMESIS : Événement news non prévu
  - WHIPLASH_VOLATILITY : Volatilité extrême (>3%)
  - LIQUIDITY_TRAP : Piège de liquidité / slippage
  - TREND_REVERSAL : Retournement de tendance brutal
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NemesisSystem:
    """
    Système Nemesis — Mémoire des défaites et adaptation autonome.
    """

    BLOCK_DURATION_SECONDS = 3600  # 1h de blocage lors de méditation

    def __init__(self):
        self.defeat_ledger: List[Dict[str, Any]] = []
        self.known_nemeses: Dict[str, int] = {}
        self.trading_blocked_until: Optional[datetime] = None
        self._meditation_in_progress = False

    async def report_loss(
        self, trade_id: str, loss_amount: float, market_context: Dict[str, Any]
    ) -> None:
        """
        Enregistre une défaite et classifie le type d'ennemi.
        Si un Nemesis atteint 3 défaites, déclenche la Méditation.
        """
        defeat_entry = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade_id,
            "loss": loss_amount,
            "context": market_context,
            "nemesis_type": self._classify_nemesis(market_context),
        }
        self.defeat_ledger.append(defeat_entry)

        n_type = defeat_entry["nemesis_type"]
        self.known_nemeses[n_type] = self.known_nemeses.get(n_type, 0) + 1

        logger.warning(
            f"⚔️ Nemesis '{n_type}' détecté (trade {trade_id}). "
            f"Défaites contre cet ennemi: {self.known_nemeses[n_type]}"
        )

        if self.known_nemeses[n_type] >= 3:
            await self._trigger_meditation(n_type)

        # Persister dans Redis si disponible
        await self._save_state()

    def _classify_nemesis(self, context: Dict[str, Any]) -> str:
        """Classifie la cause de la perte."""
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

    async def _trigger_meditation(self, nemesis_type: str) -> None:
        """
        Déclenche une phase de Méditation : bloque le trading et
        demande un retraining ciblé au Lab.
        """
        self._meditation_in_progress = True
        self.trading_blocked_until = datetime.now() + timedelta(
            seconds=self.BLOCK_DURATION_SECONDS
        )
        logger.info(
            f"🧘 MÉDITATION déclenchée pour Nemesis '{nemesis_type}'. "
            f"Trading bloqué jusqu'à {self.trading_blocked_until.strftime('%H:%M')}"
        )

        # Notifier le Lab pour retraining via Redis
        try:
            from shared.redis_client import get_redis_client
            redis = get_redis_client()
            await redis.broadcast_to_swarm(
                source="banker",
                action="NEMESIS_MEDITATION",
                payload={
                    "nemesis_type": nemesis_type,
                    "defeats": self.known_nemeses[nemesis_type],
                    "blocked_until": self.trading_blocked_until.isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Erreur notification méditation: {e}")

    def should_block_trading(self) -> bool:
        """Vérifie si le trading doit être bloqué (méditation en cours)."""
        if self.trading_blocked_until and datetime.now() < self.trading_blocked_until:
            return True
        # Auto-libération
        if self.trading_blocked_until and datetime.now() >= self.trading_blocked_until:
            self._meditation_in_progress = False
            self.trading_blocked_until = None
        return False

    async def _save_state(self) -> None:
        """Persiste l'état dans Redis."""
        try:
            from shared.redis_client import get_redis_client
            redis = get_redis_client()
            await redis.cache_set(
                "nemesis:state",
                {
                    "defeat_ledger": self.defeat_ledger[-50:],  # Garder les 50 derniers
                    "known_nemeses": self.known_nemeses,
                    "trading_blocked_until": (
                        self.trading_blocked_until.isoformat()
                        if self.trading_blocked_until
                        else None
                    ),
                },
                ttl_seconds=86400,
            )
        except Exception:
            pass  # Redis optionnel

    async def load_state(self) -> None:
        """Charge l'état depuis Redis au démarrage."""
        try:
            from shared.redis_client import get_redis_client
            redis = get_redis_client()
            state = await redis.cache_get("nemesis:state")
            if state:
                self.defeat_ledger = state.get("defeat_ledger", [])
                self.known_nemeses = state.get("known_nemeses", {})
                if state.get("trading_blocked_until"):
                    self.trading_blocked_until = datetime.fromisoformat(
                        state["trading_blocked_until"]
                    )
                logger.info("📂 Nemesis state chargé depuis Redis")
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état complet du Nemesis System."""
        return {
            "total_defeats": len(self.defeat_ledger),
            "known_nemeses": self.known_nemeses,
            "trading_blocked": self.should_block_trading(),
            "blocked_until": (
                self.trading_blocked_until.isoformat()
                if self.trading_blocked_until
                else None
            ),
            "meditation_active": self._meditation_in_progress,
            "recent_defeats": self.defeat_ledger[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_nemesis_instance: Optional[NemesisSystem] = None


def get_nemesis_system() -> NemesisSystem:
    """Retourne l'instance singleton du Nemesis System."""
    global _nemesis_instance
    if _nemesis_instance is None:
        _nemesis_instance = NemesisSystem()
    return _nemesis_instance
