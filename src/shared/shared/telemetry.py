"""
Telemetry — Observabilité et métriques pour THE HIVE
════════════════════════════════════════════════════

Collecte des métriques clés de chaque microservice :
  - Uptime
  - Compteurs de requêtes / erreurs
  - Latence moyenne
  - CPU / RAM (via psutil si disponible)
  - Métriques custom par service

Conçu pour être compatible OpenTelemetry dans le futur.
"""

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

# Import optionnel de psutil pour les métriques système réelles
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class Telemetry:
    """
    Service de télémétrie pour un microservice THE HIVE.

    Usage:
        telemetry = Telemetry("eva-core")
        telemetry.record_request()
        telemetry.record_latency(0.042)
        metrics = telemetry.get_metrics()
    """

    def __init__(self, service_name: str, latency_window: int = 100):
        self.service_name = service_name
        self.start_time = time.time()

        # Compteurs
        self.requests_total = 0
        self.errors_total = 0
        self.warnings_total = 0

        # Latence (fenêtre glissante)
        self._latency_window = latency_window
        self._latencies: deque = deque(maxlen=latency_window)

        # Métriques custom
        self._custom_metrics: Dict[str, Any] = {}

        # Dernière requête
        self.last_request_time: Optional[float] = None

        logger.info(f"📊 Telemetry initialisée pour '{service_name}'")

    def record_request(self) -> None:
        """Enregistre une requête entrante"""
        self.requests_total += 1
        self.last_request_time = time.time()

    def record_error(self) -> None:
        """Enregistre une erreur"""
        self.errors_total += 1

    def record_warning(self) -> None:
        """Enregistre un warning"""
        self.warnings_total += 1

    def record_latency(self, seconds: float) -> None:
        """Enregistre la latence d'une requête"""
        self._latencies.append(seconds)

    def set_metric(self, key: str, value: Any) -> None:
        """Définit une métrique custom"""
        self._custom_metrics[key] = value

    def increment_metric(self, key: str, amount: int = 1) -> None:
        """Incrémente un compteur custom"""
        self._custom_metrics[key] = self._custom_metrics.get(key, 0) + amount

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques système réelles (si psutil est disponible)"""
        if not PSUTIL_AVAILABLE:
            return {
                "cpu_percent": 0.0,
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "psutil_available": False,
            }

        try:
            mem = psutil.virtual_memory()
            return {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_used_mb": round(mem.used / (1024 * 1024)),
                "memory_total_mb": round(mem.total / (1024 * 1024)),
                "memory_percent": mem.percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "psutil_available": True,
            }
        except Exception:
            return {"cpu_percent": 0.0, "memory_used_mb": 0, "psutil_available": False}

    def _get_latency_stats(self) -> Dict[str, float]:
        """Calcule les statistiques de latence"""
        if not self._latencies:
            return {
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)

        return {
            "avg_ms": round(sum(sorted_lat) / n * 1000, 2),
            "min_ms": round(sorted_lat[0] * 1000, 2),
            "max_ms": round(sorted_lat[-1] * 1000, 2),
            "p95_ms": round(sorted_lat[min(int(n * 0.95), n - 1)] * 1000, 2),
            "p99_ms": round(sorted_lat[min(int(n * 0.99), n - 1)] * 1000, 2),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne toutes les métriques du service"""
        uptime = time.time() - self.start_time
        error_rate = (
            (self.errors_total / self.requests_total * 100)
            if self.requests_total > 0
            else 0.0
        )

        return {
            "service_name": self.service_name,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 1),
            "uptime_human": self._format_duration(uptime),
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "warnings_total": self.warnings_total,
            "error_rate_percent": round(error_rate, 2),
            "last_request_time": (
                datetime.fromtimestamp(self.last_request_time).isoformat()
                if self.last_request_time
                else None
            ),
            "latency": self._get_latency_stats(),
            "system": self._get_system_metrics(),
            "custom": self._custom_metrics,
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Formate une durée en lisible humain"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        elif minutes > 0:
            return f"{minutes}m{secs:02d}s"
        else:
            return f"{secs}s"
