"""
Self-Healing Service — THE HIVE
Monitors Docker containers and restarts them if they fail (Phoenix Protocol).
"""

import asyncio
import logging
from typing import List
from eva_core.services.docker_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class SelfHealingService:
    """
    Service responsible for the 'Self-Healing' of the Hive Swarm.
    It monitors unhealthy or exited containers and attempts to resurrect them.
    
    Integration RLM (Sprint 4):
        Chaque résurrection est loggée dans `resurrection_events` pour que
        le RLM Evaluator puisse analyser les crashs récurrents.
    """

    def __init__(self):
        self.monitor = SystemMonitor()
        self.critical_services = [
            "hive-infra-redis",
            "hive-infra-qdrant",
            "hive-core",
            "hive-banker",
            "hive-sentinel",
            "hive-kernel"
        ]
        # RLM Hook: collect resurrection events for deeper analysis
        self.resurrection_events: list = []

    async def start_monitoring(self, interval_seconds: int = 30):
        """
        Starts the infinite loop of health monitoring and healing.
        """
        logger.info(f"🔥 Phoenix Protocol (Self-Healing) active — Scan interval: {interval_seconds}s")
        
        while True:
            try:
                await self.heal_swarm()
            except Exception as e:
                logger.error(f"Self-Healing failure: {e}")
            
            await asyncio.sleep(interval_seconds)

    async def heal_swarm(self):
        """
        Scans the swarm and performs healing actions if necessary.
        """
        containers = await self.monitor.get_docker_containers()
        
        for container in containers:
            name = container.get("name")
            status = container.get("status")
            
            # If a critical container is not running, we heal it
            if name in self.critical_services and status not in ["running", "restarting"]:
                logger.warning(f"⚠️ Service {name} detected in state '{status}'. Initiating Phoenix Protocol...")
                await self._resurrect_container(name)

    async def _resurrect_container(self, name: str):
        """
        Attempts to restart a container using the Docker SDK.
        Logs the event for RLM analysis.
        """
        if not self.monitor._docker_client:
            logger.error("Cannot resurrect: Docker client not connected.")
            return

        try:
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                None, lambda: self.monitor._docker_client.containers.get(name)
            )
            
            logger.info(f"⚡ Restarting {name}...")
            await loop.run_in_executor(None, container.restart)
            logger.info(f"✅ {name} successfully resurrected.")
            
            # RLM Hook: enregistrer l'événement pour analyse RLM
            self.resurrection_events.append({
                "service": name,
                "event": "resurrection",
                "status": "success",
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })
            
            # Notifier la ruche via Redis (pour alerte Telegram)
            from shared.redis_client import get_redis_client
            redis = get_redis_client()
            await redis.publish("eva.swarm.healing", {
                "service": name,
                "event": "resurrection",
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Failed to resurrect {name}: {e}")
            # RLM Hook: enregistrer l'échec aussi
            self.resurrection_events.append({
                "service": name,
                "event": "resurrection_failed",
                "error": str(e),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })

