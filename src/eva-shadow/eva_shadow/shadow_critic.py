import asyncio
import logging
from decimal import Decimal
from shared.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class ShadowCritic:
    """
    Expert Shadow : L'Ombre d'EVA.
    Tourne en parallèlle du Banker avec des stratégies alternatives.
    """
    def __init__(self):
        self.redis = get_redis_client()
        self.banker_pnl = Decimal("0")
        self.shadow_pnl = Decimal("0")
        self.active = False

    async def run_shadow_simulation(self):
        """
        Simule une stratégie Contrarienne contre le Banker.
        Écoute les ordres du Banker et simule l'inverse.
        """
        self.active = True
        logger.info("Shadow Expert: Stratégie Contrarienne s'active sur le flux Banker.")
        
        # En production, on utiliserait un vrai sub Redis
        # Pour la démo, on simule l'écoute et la comparaison
        while self.active:
            # On imagine que le Banker a fait -100 et que l'Ombre (Contrarienne) a fait +100
            # On récupère le PNL réel du Banker via une variable partagée ou Redis
            # Simulation simplifiée de surperformance
            mock_banker_loss = Decimal("-50.0")
            self.shadow_pnl += abs(mock_banker_loss) * Decimal("1.2") # L'ombre gagne là où le Banker perd
            
            if self.shadow_pnl > (self.banker_pnl + Decimal("200")):
                logger.warning(f"🚀 SHADOW ALERT: Contrarian strategy Alpha detected! Shadow PnL: {self.shadow_pnl}")
                await self.redis.publish("eva.swarm.events", {
                    "type": "COGNITIVE_MUTATION_REQUIRED",
                    "reason": "BANKER_UNDERPERFORMING_SHADOW_ALPHA",
                    "shadow_pnl": float(self.shadow_pnl),
                    "banker_pnl": float(self.banker_pnl)
                })
            
            await asyncio.sleep(120) # Comparaison toutes les 2 minutes

    def stop(self):
        self.active = False

def uniform(a, b): # Helper simple pour éviter les imports lourds
    import random
    return random.uniform(a, b)
