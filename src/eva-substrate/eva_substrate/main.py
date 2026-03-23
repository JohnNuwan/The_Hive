"""
EVA Substrate — Le Corps Biologique de THE HIVE.

Ce module gère les « fonctions vitales » de l'infrastructure :
- Monitoring de la consommation énergétique (CPU, GPU, réseau).
- Rythme circadien : adaptation jour/nuit des ressources.
- Allocation dynamique des accélérateurs (TPU/GPU).
- Monitoring GPU temps réel (température, VRAM, utilisation).
- Historique des métriques et alertes de seuil.
- Planification tâches éco-mode (heures creuses).

Architecture :
    - Passif : collecte et expose les métriques.
    - Communique avec le Core via Redis heartbeat.
    - Réduit les charges la nuit (mode éco).
"""

import logging
import asyncio
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from eva_substrate.energy_monitor import EnergyMonitor
from eva_substrate.circadian_rhythm import CircadianRhythm
from eva_substrate.resource_allocator import ResourceAllocator
from shared.redis_client import init_redis, get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════


class GpuMetrics(BaseModel):
    """Métriques GPU temps réel."""
    name: str = "N/A"
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0


class SystemMetrics(BaseModel):
    """Snapshot complet des métriques système."""
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    disk_used_gb: float
    disk_total_gb: float
    gpu: GpuMetrics | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AlertThreshold(BaseModel):
    """Configuration d'alerte de seuil."""
    metric: str = Field(..., description="Métrique: cpu, ram, gpu_temp, gpu_vram, disk")
    threshold: float = Field(..., gt=0, description="Seuil de déclenchement")
    action: str = Field(default="notify", description="Action: notify, throttle, shutdown")


class EcoSchedule(BaseModel):
    """Planification de tâche en mode éco (heures creuses)."""
    task_name: str = Field(..., min_length=2)
    preferred_hours: list[int] = Field(default=[0, 1, 2, 3, 4, 5], description="Heures préférées (0-23)")
    gpu_required: bool = False
    estimated_duration_min: int = 30


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Substrate.

    Initialise la connexion Redis, instancie les services de monitoring
    énergétique, rythme circadien et allocation de ressources.
    """
    logger.info("🌿 Démarrage EVA Substrate (Le Corps)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.monitor = EnergyMonitor()
    app.state.rhythm = CircadianRhythm()
    app.state.allocator = ResourceAllocator()
    app.state.metrics_history: deque[dict[str, Any]] = deque(maxlen=1000)
    app.state.alerts: list[dict[str, Any]] = []
    app.state.thresholds: list[dict[str, Any]] = [
        {"metric": "cpu", "threshold": 90.0, "action": "notify"},
        {"metric": "gpu_temp", "threshold": 85.0, "action": "throttle"},
        {"metric": "ram", "threshold": 90.0, "action": "notify"},
        {"metric": "disk", "threshold": 95.0, "action": "notify"},
    ]
    app.state.eco_queue: list[dict[str, Any]] = []

    # Heartbeat + Metrics collector
    asyncio.create_task(hard_heartbeat(app.state.rhythm))
    asyncio.create_task(metrics_collector())

    logger.info("✅ EVA Substrate actif")
    yield
    logger.info("🛑 Arrêt EVA Substrate")


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def hard_heartbeat(rhythm: CircadianRhythm):
    """Signal de présence avec mode circadien courant."""
    redis = get_redis_client()
    while True:
        try:
            mode_info = rhythm.get_current_mode()
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "substrate",
                "mode": mode_info["mode"],
                "is_night": mode_info["is_night"],
            }
            await redis.cache_set("eva.substrate.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(2.0)


async def metrics_collector():
    """Collecte périodique de métriques (toutes les 30s) et vérification des seuils."""
    while True:
        try:
            gpu = _get_gpu_metrics()
            import psutil
            metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_used_gb": round(psutil.disk_usage("/").used / (1024**3), 2),
                "disk_total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
                "gpu": gpu,
                "timestamp": datetime.now().isoformat(),
            }
            app.state.metrics_history.append(metrics)

            # Vérification des seuils
            for threshold in app.state.thresholds:
                _check_threshold(threshold, metrics)

        except Exception as e:
            logger.debug(f"Metrics collector: {e}")
        await asyncio.sleep(30)


def _get_gpu_metrics() -> dict[str, Any] | None:
    """Tente de récupérer les métriques GPU via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        pynvml.nvmlShutdown()
        return {
            "name": name,
            "utilization_percent": util.gpu,
            "memory_used_mb": round(info.used / (1024**2)),
            "memory_total_mb": round(info.total / (1024**2)),
            "temperature_c": temp,
            "power_draw_w": round(power, 1),
        }
    except Exception:
        return None


def _check_threshold(threshold: dict, metrics: dict):
    """Vérifie si un seuil est dépassé et génère une alerte."""
    metric = threshold["metric"]
    limit = threshold["threshold"]
    value = None

    if metric == "cpu":
        value = metrics.get("cpu_percent", 0)
    elif metric == "ram":
        total = metrics.get("ram_total_gb", 1)
        used = metrics.get("ram_used_gb", 0)
        value = (used / total) * 100 if total > 0 else 0
    elif metric == "gpu_temp" and metrics.get("gpu"):
        value = metrics["gpu"].get("temperature_c", 0)
    elif metric == "disk":
        total = metrics.get("disk_total_gb", 1)
        used = metrics.get("disk_used_gb", 0)
        value = (used / total) * 100 if total > 0 else 0

    if value is not None and value > limit:
        alert = {
            "metric": metric,
            "value": value,
            "threshold": limit,
            "action": threshold["action"],
            "timestamp": datetime.now().isoformat(),
        }
        app.state.alerts.append(alert)
        logger.warning(f"🚨 Seuil dépassé: {metric}={value:.1f}% (max {limit}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA Substrate API",
    description="Le Corps de THE HIVE — Énergie, GPU & Rythme Circadien",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    """Vérifie la santé du module Substrate."""
    return {"status": "online", "service": "substrate"}


@app.get("/metrics", tags=["Énergie"])
async def get_metrics():
    """Retourne les métriques énergétiques et hardware actuelles."""
    monitor: EnergyMonitor = app.state.monitor
    base = monitor.get_current_consumption()
    gpu = _get_gpu_metrics()
    if gpu:
        base["gpu"] = gpu
    return base


@app.get("/gpu", tags=["GPU"])
async def get_gpu():
    """Métriques GPU détaillées (température, VRAM, utilisation, puissance)."""
    gpu = _get_gpu_metrics()
    if gpu is None:
        return {"status": "unavailable", "message": "Pas de GPU NVIDIA détecté (pynvml)"}
    return {"status": "ok", "gpu": gpu}


@app.get("/mode", tags=["Circadien"])
async def get_mode():
    """Retourne le mode circadien actuel (Jour/Nuit)."""
    rhythm: CircadianRhythm = app.state.rhythm
    return rhythm.get_current_mode()


@app.post("/allocate", tags=["Ressources"])
async def allocate_tpus(profile: str):
    """Alloue les accélérateurs (TPU/GPU) selon un profil spécifique."""
    allocator: ResourceAllocator = app.state.allocator
    return allocator.set_profile(profile)


@app.get("/metrics/history", tags=["Énergie"])
async def get_metrics_history(limit: int = Query(default=60, ge=1, le=1000)):
    """Historique des métriques système (max 1000 snapshots à 30s chacun)."""
    history = list(app.state.metrics_history)[-limit:]
    return {"history": history, "count": len(history)}


@app.get("/alerts", tags=["Alertes"])
async def get_alerts(limit: int = Query(default=50, ge=1, le=500)):
    """Retourne les alertes de seuil dépassé."""
    alerts = app.state.alerts[-limit:]
    return {"alerts": alerts, "total": len(app.state.alerts)}


@app.get("/thresholds", tags=["Alertes"])
async def get_thresholds():
    """Liste les seuils d'alerte configurés."""
    return {"thresholds": app.state.thresholds}


@app.post("/thresholds", tags=["Alertes"])
async def add_threshold(threshold: AlertThreshold):
    """Ajoute ou modifie un seuil d'alerte."""
    # Mettre à jour si existe déjà
    for i, t in enumerate(app.state.thresholds):
        if t["metric"] == threshold.metric:
            app.state.thresholds[i] = threshold.model_dump()
            return {"status": "updated", "threshold": threshold.model_dump()}
    app.state.thresholds.append(threshold.model_dump())
    return {"status": "created", "threshold": threshold.model_dump()}


@app.post("/eco/schedule", tags=["Éco-Mode"])
async def schedule_eco_task(schedule: EcoSchedule):
    """Planifie une tâche pour exécution en heures creuses (mode éco)."""
    task = {
        **schedule.model_dump(),
        "status": "queued",
        "created_at": datetime.now().isoformat(),
    }
    app.state.eco_queue.append(task)
    return {"status": "queued", "task": task}


@app.get("/eco/queue", tags=["Éco-Mode"])
async def get_eco_queue():
    """Liste les tâches en attente d'exécution éco."""
    return {"queue": app.state.eco_queue, "total": len(app.state.eco_queue)}
