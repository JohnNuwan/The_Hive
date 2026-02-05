"""
Service Monitoring - Surveillance Hardware THE HIVE
Monitoring CPU, RAM, Disk et GPU (simulation)
"""

import asyncio
import logging
from datetime import datetime

import psutil
from shared import GPUMetrics, HardwareMetrics

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Surveille l'utilisation des ressources hardware"""

    def __init__(self, interval_seconds: int = 5):
        self.interval = interval_seconds
        self._running = False
        self._task = None
        self._current_metrics = None

    async def start(self):
        """Lance la boucle de monitoring"""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("SystemMonitor démarré")

    async def stop(self):
        """Arrête le monitoring"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Boucle de collecte des métriques"""
        while self._running:
            try:
                self._current_metrics = self._collect_metrics()
                # TODO: Envoyer via Redis pour Grafana/Panopticon
            except Exception as e:
                logger.error(f"Erreur collecte métriques: {e}")
            await asyncio.sleep(self.interval)

    def _collect_metrics(self) -> HardwareMetrics:
        """Collecte les métriques système via psutil"""
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        
        # Simulation GPU (à remplacer par nvidia-smi en prod)
        gpu = GPUMetrics(
            temperature_celsius=62.0,
            utilization_percent=12.5,
            memory_used_mb=4200,
            memory_total_mb=24576
        )

        return HardwareMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            ram_used_gb=round(ram.used / (1024**3), 2),
            ram_total_gb=round(ram.total / (1024**3), 2),
            disk_used_percent=psutil.disk_usage('/').percent,
            gpu=gpu
        )

    async def get_current_metrics(self) -> HardwareMetrics:
        """Retourne les dernières métriques collectées"""
        if self._current_metrics is None:
            return self._collect_metrics()
        return self._current_metrics
