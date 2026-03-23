"""
Service GhostShield pour l'obfuscation optionnelle des executions.
"""

import asyncio
import logging
from decimal import Decimal
from random import randint, uniform
from typing import Any

from shared import get_settings
from shared.models import TradeOrder

logger = logging.getLogger(__name__)


class GhostShield:
    """
    Obfusque l'execution des ordres quand le mode stealth est active.

    En mode demo, la fragmentation reste desactivee par defaut afin de
    privilegier un suivi lisible du risque, de la marge et des positions.
    """

    def __init__(self, mt5_service: Any) -> None:
        """
        Initialise GhostShield.

        Args:
            mt5_service (Any): Service MT5 utilise pour l'envoi final.
        """
        self.mt5 = mt5_service
        self.settings = get_settings()
        self.fragmentation_enabled = self.settings.banker_ghost_fragmentation_enabled
        self.fragmentation_min_volume = Decimal(
            str(self.settings.banker_ghost_fragmentation_min_volume)
        )

    async def execute_obfuscated_order(self, order: TradeOrder) -> dict[str, Any]:
        """
        Execute un ordre avec delai aleatoire et fragmentation optionnelle.

        Args:
            order (TradeOrder): Ordre a executer.

        Returns:
            dict[str, Any]: Resultat retourne par le service MT5.
        """
        total_volume = order.volume

        if not self.fragmentation_enabled or total_volume < self.fragmentation_min_volume:
            delay = uniform(0.1, 1.5)
            await asyncio.sleep(delay)
            return await self.mt5.execute_order(order)

        num_fragments = randint(2, 4)
        logger.info(
            "GhostShield fragmente %s en %s morceaux pour brouiller la signature.",
            total_volume,
            num_fragments,
        )

        remaining = total_volume
        first_result: dict[str, Any] | None = None

        for index in range(num_fragments):
            if index == num_fragments - 1:
                fragment_volume = remaining
            else:
                fragment_volume = (
                    total_volume * Decimal(str(uniform(0.2, 0.4)))
                ).quantize(Decimal("0.01"))
                if fragment_volume >= remaining:
                    fragment_volume = (remaining / 2).quantize(Decimal("0.01"))

            if fragment_volume <= 0:
                break

            fragment_order = order.model_copy(update={"volume": fragment_volume})
            delay = uniform(0.5, 5.0)
            logger.info(
                "GhostShield execute le fragment %s/%s (%s lot) apres %.2fs.",
                index + 1,
                num_fragments,
                fragment_volume,
                delay,
            )
            await asyncio.sleep(delay)

            result = await self.mt5.execute_order(fragment_order)
            if first_result is None:
                first_result = result
            if not result.get("success"):
                return result

            remaining -= fragment_volume

        return first_result or {
            "success": False,
            "message": "Aucun fragment valide n'a pu etre execute.",
        }
