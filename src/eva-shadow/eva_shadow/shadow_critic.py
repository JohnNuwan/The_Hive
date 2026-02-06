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
        Simule une stratégie "Adverse" ou "Alternative" en temps réel.
        """
        self.active = True
        logger.info("Shadow Expert: Ombre activée. Analyse comparative en cours.")
        
        while self.active:
            # Simulation d'un trade de l'ombre
            # Ici on pourrait appeler une version différente du GNN
            shadow_gain = Decimal(str(round(float(uniform(-10, 15)), 2)))
            self.shadow_pnl += shadow_gain
            
            # Comparaison avec les données du Banker (récupérées via Redis)
            # En prod: on écoute eva.banker.events
            
            if self.shadow_pnl > (self.banker_pnl + Decimal("100")):
                logger.warning("🚀 SHADOW ALERT: Alternative strategy is outperforming the main Banker!")
                await self.redis.publish("eva.swarm.events", {
                    "type": "COGNITIVE_MUTATION_REQUIRED",
                    "reason": "Shadow strategy showing higher Alpha",
                    "shadow_pnl": float(self.shadow_pnl),
                    "banker_pnl": float(self.banker_pnl)
                })
            
            await asyncio.sleep(300) # Comparaison toutes les 5 minutes

    def stop(self):
        self.active = False

def uniform(a, b): # Helper simple pour éviter les imports lourds
    import random
    return random.uniform(a, b)
