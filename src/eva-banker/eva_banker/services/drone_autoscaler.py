"""
Drone Autoscaler — Ajustement dynamique des drones de trading
════════════════════════════════════════════════════════════

Règles d'auto-scaling basées sur la volatilité du marché :
  - Volatilité haute (>2x seuil) → Scale UP (plus de drones pour diversifier)
  - Volatilité basse (<0.5x seuil) → Scale DOWN (moins de drones, économie GPU)
  - Volatilité normale → Pas de changement

Notifie le Swarm via Redis pour coordination avec les autres agents.
"""

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DroneAutoscaler:
    """
    Service d'auto-scaling des drones de surveillance et trading.
    """

    def __init__(
        self,
        min_drones: int = 1,
        max_drones: int = 5,
        volatility_threshold: float = 0.01,
    ):
        self.current_drones = min_drones
        self.min_drones = min_drones
        self.max_drones = max_drones
        self.volatility_threshold = volatility_threshold
        self._scale_history: list[Dict[str, Any]] = []

        logger.info(
            f"🤖 DroneAutoscaler initialisé "
            f"(min={min_drones}, max={max_drones}, seuil_vol={volatility_threshold})"
        )

    async def evaluate_and_scale(self, market_volatility: float) -> int:
        """
        Évalue la volatilité et ajuste le nombre de drones.

        Args:
            market_volatility: Volatilité actuelle du marché (ex: 0.015 = 1.5%)

        Returns:
            Nombre de drones actifs après ajustement.
        """
        old_count = self.current_drones
        action = "NONE"

        if market_volatility > self.volatility_threshold * 2:
            # Haute volatilité → scale up
            if self.current_drones < self.max_drones:
                self.current_drones = min(self.current_drones + 1, self.max_drones)
                action = "SCALE_UP"
                logger.warning(
                    f"⬆️ SCALE UP: Volatilité {market_volatility:.4f} > seuil. "
                    f"Drones: {old_count} → {self.current_drones}"
                )
        elif market_volatility < self.volatility_threshold * 0.5:
            # Basse volatilité → scale down
            if self.current_drones > self.min_drones:
                self.current_drones = max(self.current_drones - 1, self.min_drones)
                action = "SCALE_DOWN"
                logger.info(
                    f"⬇️ SCALE DOWN: Volatilité {market_volatility:.4f} < seuil/2. "
                    f"Drones: {old_count} → {self.current_drones}"
                )

        if action != "NONE":
            self._scale_history.append({
                "action": action,
                "volatility": market_volatility,
                "old_count": old_count,
                "new_count": self.current_drones,
            })
            await self._notify_swarm(action, market_volatility)

        return self.current_drones

    async def _notify_swarm(self, action: str, volatility: float) -> None:
        """Notifie le Swarm du changement de capacité via Redis."""
        try:
            from shared.redis_client import get_redis_client
            redis = get_redis_client()
            await redis.broadcast_to_swarm(
                source="banker",
                action="DRONE_SCALE_EVENT",
                payload={
                    "action": action,
                    "new_drone_count": self.current_drones,
                    "volatility": volatility,
                },
            )
        except Exception as e:
            logger.error(f"Erreur notification Swarm: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_drones": self.current_drones,
            "min_drones": self.min_drones,
            "max_drones": self.max_drones,
            "volatility_threshold": self.volatility_threshold,
            "recent_actions": self._scale_history[-5:],
        }
