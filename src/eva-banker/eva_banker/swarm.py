import asyncio
import logging
from decimal import Decimal
from shared.swarm_helper import DroneRunner
from shared.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class BankerSwarm(DroneRunner):
    """
    Extension de Banker pour gérer les drones de surveillance financière.
    """
    def __init__(self):
        super().__init__(agent_name="banker")

    async def run_gold_surveillance(self, threshold: Decimal = Decimal("2050.0")):
        """
        Mission du drone : Surveiller le prix de l'or et alerter en cas de dépassement.
        """
        logger.info(f"GoldSurveillance mission started. Threshold: {threshold}")
        redis = get_redis_client()
        
        try:
            while True:
                # Simulation de récupération de prix (en prod: MT5Service.get_price)
                # On simule une variation
                fake_price = Decimal("2048.50") 
                
                if fake_price >= threshold:
                    await redis.publish("eva.swarm.events", {
                        "type": "MARKET_ALERT",
                        "drone": "GoldSurveillance",
                        "message": f"🚨 GOLD target reached: {fake_price} >= {threshold}"
                    })
                
                # Le DroneRunner s'occupe du heartbeat automatiquement via sa boucle interne
                await asyncio.sleep(60) # Vérification chaque minute
        except asyncio.CancelledError:
            logger.info("GoldSurveillance mission cancelled.")
            raise
