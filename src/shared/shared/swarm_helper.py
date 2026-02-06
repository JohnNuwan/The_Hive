import asyncio
import logging
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, Optional

from shared.models import AgentMessage, AgentMessageType, SwarmDrone
from shared.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class DroneRunner:
    """
    Classe de base pour transformer un expert en contrôleur de Drones.
    Permet de lancer des tâches autonomes persistantes.
    """
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.active_drones: Dict[str, asyncio.Task] = {}
        self.redis = get_redis_client()

    async def spawn_drone(self, name: str, mission: str, coro, metadata: Optional[Dict] = None):
        """
        Lance un drone de fond (coroutine) et l'enregistre dans le swarm.
        """
        drone_id = str(uuid4())
        drone_info = SwarmDrone(
            id=drone_id,
            name=name,
            parent_agent=self.agent_name,
            mission=mission,
            metadata=metadata or {}
        )
        
        # Enregistrement initial dans Redis
        await self.redis.cache_set(f"swarm:drone:{drone_id}", drone_info.model_dump(), ttl_seconds=3600)
        
        # Lancement de la tâche
        task = asyncio.create_task(self._run_drone_loop(drone_info, coro))
        self.active_drones[drone_id] = task
        
        logger.info(f"Drone [{name}] (ID: {drone_id}) launched by {self.agent_name}. Mission: {mission}")
        return drone_id

    async def _run_drone_loop(self, drone: SwarmDrone, coro):
        """
        Boucle de vie du drone avec heartbeat.
        """
        try:
            # On lance la mission en parallèle
            mission_task = asyncio.create_task(coro)
            
            while not mission_task.done():
                # Heartbeat toutes les 30 secondes
                drone.last_callback = datetime.now()
                await self.redis.cache_set(f"swarm:drone:{drone.id}", drone.model_dump(), ttl_seconds=60)
                
                # Notification de progression au swarm
                await self.redis.publish(f"eva.swarm.events", {
                    "type": "DRONE_HEARTBEAT",
                    "drone_id": str(drone.id),
                    "status": drone.status
                })
                
                await asyncio.sleep(30)
                
            # Fin de mission
            result = await mission_task
            drone.status = "completed"
            drone.metadata["result"] = str(result)
            await self.redis.cache_set(f"swarm:drone:{drone.id}", drone.model_dump(), ttl_seconds=3600)
            logger.info(f"Drone [{drone.name}] finished mission successfully.")
            
        except Exception as e:
            logger.error(f"Drone [{drone.name}] failed: {e}")
            drone.status = "error"
            drone.metadata["error"] = str(e)
            await self.redis.cache_set(f"swarm:drone:{drone.id}", drone.model_dump(), ttl_seconds=3600)
        finally:
            if str(drone.id) in self.active_drones:
                del self.active_drones[str(drone.id)]

    async def kill_drone(self, drone_id: str):
        """Arrête un drone de force."""
        if drone_id in self.active_drones:
            self.active_drones[drone_id].cancel()
            logger.warning(f"Drone {drone_id} killed by parent.")
            return True
        return False
