import asyncio
import logging
from decimal import Decimal
from random import uniform, randint
from typing import List
from shared.models import TradeOrder

logger = logging.getLogger(__name__)

class GhostShield:
    """
    Service d'Obfuscation de signature de marché (Anti-HFT).
    Fragment les ordres et déale l'exécution pour rester invisible.
    """
    def __init__(self, mt5_service):
        self.mt5 = mt5_service

    async def execute_obfuscated_order(self, order: TradeOrder):
        """
        Fragment un gros ordre en plusieurs petits ordres pour cacher la main de l'IA.
        """
        total_volume = order.volume
        
        # Si le volume est petit, on l'exécute normalement mais avec un délais aléatoire
        if total_volume <= Decimal("0.05"):
            delay = uniform(0.1, 1.5)
            await asyncio.sleep(delay)
            return await self.mt5.execute_order(order)

        # Fragmentation (entre 2 et 4 morceaux)
        num_fragments = randint(2, 4)
        logger.info(f"GhostShield: Fragmenting order of {total_volume} into {num_fragments} pieces.")
        
        remaining = total_volume
        results = []
        
        for i in range(num_fragments):
            if i == num_fragments - 1:
                frag_volume = remaining
            else:
                # Portion aléatoire (ex: 20% à 40%)
                frag_volume = (total_volume * Decimal(str(uniform(0.2, 0.4)))).quantize(Decimal("0.01"))
                if frag_volume >= remaining: frag_volume = remaining / 2
            
            if frag_volume <= 0: break
                
            frag_order = order.model_copy(update={"volume": frag_volume})
            
            # Ajout du délais "Fantôme" pour briser le pattern
            delay = uniform(0.5, 5.0)
            logger.info(f"GhostShield: Executing fragment {i+1} ({frag_volume}) after {delay:.2f}s delay.")
            await asyncio.sleep(delay)
            
            res = await self.mt5.execute_order(frag_order)
            results.append(res)
            remaining -= frag_volume

        return results[0] # On retourne le premier résultat pour la continuité
