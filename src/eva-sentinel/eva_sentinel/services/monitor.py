"""
Service Monitoring — Surveillance Hardware THE HIVE.

Collecte en continu les métriques système (CPU, RAM, Disque, GPU)
via psutil et les expose à Sentinel, au Core et à Grafana.

Compatible Windows et Linux (détection automatique du disque racine).
"""

import asyncio
import logging
import sys
from datetime import datetime

import psutil
from shared import GPUMetrics, HardwareMetrics

logger = logging.getLogger(__name__)


def _get_disk_root() -> str:
    """
    Retourne le chemin racine du disque selon le système d'exploitation.

    Returns:
        str: 'C:\\\\' sur Windows, '/' sur Linux/macOS.
    """
    if sys.platform == "win32":
        return "C:\\"
    return "/"


class SystemMonitor:
    """
    Surveille l'utilisation des ressources hardware en temps réel.

    Collecte périodiquement (par défaut toutes les 5 secondes) :
    - CPU : pourcentage d'utilisation global.
    - RAM : utilisée / totale en Go.
    - Disque : pourcentage d'occupation.
    - GPU : température, utilisation, VRAM (simulation en mode lite).

    Attributes:
        interval: Intervalle de collecte en secondes.
    """

    def __init__(self, interval_seconds: int = 5):
        self.interval = interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None
        self._current_metrics: HardwareMetrics | None = None

    async def start(self):
        """Lance la boucle de monitoring en tâche de fond."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("SystemMonitor démarré")

    async def stop(self):
        """Arrête proprement le monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Boucle de collecte des métriques à intervalle régulier."""
        while self._running:
            try:
                self._current_metrics = self._collect_metrics()
            except Exception as e:
                logger.error(f"Erreur collecte métriques: {e}")
            await asyncio.sleep(self.interval)

    def _collect_metrics(self) -> HardwareMetrics:
        """
        Collecte les métriques système via psutil.

        Utilise le bon chemin racine selon l'OS (Windows vs Linux).
        Le GPU est simulé en mode lite ; en production, remplacer
        par nvidia-smi ou pynvml.

        Returns:
            HardwareMetrics: Snapshot des métriques hardware.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        # Disque — compatible Windows et Linux
        disk_root = _get_disk_root()
        try:
            disk_percent = psutil.disk_usage(disk_root).percent
        except Exception:
            disk_percent = 0.0

        # Simulation GPU (à remplacer par nvidia-smi / pynvml en prod)
        gpu = GPUMetrics(
            temperature_celsius=62.0,
            utilization_percent=12.5,
            memory_used_mb=4200,
            memory_total_mb=24576,
        )

        return HardwareMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            ram_used_gb=round(ram.used / (1024**3), 2),
            ram_total_gb=round(ram.total / (1024**3), 2),
            disk_used_percent=disk_percent,
            gpu=gpu,
        )

    async def get_current_metrics(self) -> HardwareMetrics:
        """
        Retourne les dernières métriques collectées.

        Si aucune collecte n'a encore eu lieu, effectue une collecte
        immédiate et la retourne.

        Returns:
            HardwareMetrics: Métriques système actuelles.
        """
        if self._current_metrics is None:
            return self._collect_metrics()
        return self._current_metrics
