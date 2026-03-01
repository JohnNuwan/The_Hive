"""
EVA Sentinel — System Metrics Collector.

Collecte les métriques hardware via psutil pour le monitoring Sentinel.
"""

import logging
import sys
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)


def _get_disk_root() -> str:
    if sys.platform == "win32":
        return "C:\\"
    return "/"


class SystemMetricsCollector:
    """
    Collecteur de métriques système pour Sentinel.

    Fournit une méthode `collect()` qui retourne un dictionnaire
    de métriques hardware (CPU, RAM, Disque, GPU).
    """

    def collect(self) -> dict:
        """
        Collecte toutes les métriques système.

        Returns:
            dict: Métriques avec clés cpu, ram_used_gb, ram_total_gb,
                  disk_percent, gpu, timestamp.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        disk_root = _get_disk_root()
        try:
            disk = psutil.disk_usage(disk_root)
            disk_percent = disk.percent
        except Exception:
            disk_percent = 0.0

        # GPU via nvidia-smi si disponible
        gpu_data = self._collect_gpu()

        return {
            "cpu": cpu_percent,
            "ram_used_gb": round(ram.used / (1024 ** 3), 2),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "ram_percent": ram.percent,
            "disk_percent": disk_percent,
            "gpu": gpu_data,
            "timestamp": datetime.now().isoformat(),
        }

    def _collect_gpu(self) -> dict:
        """Tente de collecter les métriques GPU via nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "temperature": float(parts[0]),
                    "utilization": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                }
        except Exception:
            pass

        return {
            "temperature": 0,
            "utilization": 0,
            "memory_used_mb": 0,
            "memory_total_mb": 0,
            "note": "nvidia-smi not available",
        }
