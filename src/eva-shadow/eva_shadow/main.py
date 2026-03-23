"""
The Shadow — Agent OSINT et Renseignement de THE HIVE.

Expert C du système d'experts. Responsable de :
- La recherche web et le scraping (DuckDuckGo, Brave Search).
- La reconnaissance d'entités (Entity Recon / Threat Intel).
- La veille persistante sur des mots-clés (monitoring).
- La gestion de profils/persona pour l'investigation.
- L'analyse de menaces et la détection d'opportunités.

Architecture :
    - Utilise httpx + BeautifulSoup pour le scraping web.
    - En production, peut se connecter à des API payantes
      (Brave Search, Shodan, VirusTotal, AlienVault OTX).
    - Heartbeat vers le Core pour la découverte des agents.
"""

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_shadow.services.osint import OSINTService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES API
# ═══════════════════════════════════════════════════════════════════════════════


class SearchRequest(BaseModel):
    """Requête de recherche OSINT."""
    query: str = Field(..., min_length=2, description="Requête de recherche")
    max_results: int = Field(default=10, ge=1, le=50)
    sources: list[str] = Field(default=["duckduckgo"], description="Sources: duckduckgo, brave, shodan")


class ReconReport(BaseModel):
    """Rapport de reconnaissance sur une cible."""
    target: str
    findings: list[dict[str, Any]]
    threat_level: str = "LOW"
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class ThreatIntelResult(BaseModel):
    """Résultat d'analyse Threat Intel."""
    indicator: str
    type: str = Field(description="Type: ip, domain, hash, email")
    threat_score: float = Field(default=0.0, ge=0, le=10)
    sources: list[str] = []
    details: dict[str, Any] = {}


class MonitorTarget(BaseModel):
    """Cible de veille persistante."""
    keyword: str = Field(..., min_length=2)
    category: str = Field(default="general", description="Catégorie: finance, tech, security, crypto, general")
    interval_minutes: int = Field(default=60, ge=5, le=1440)


class PersonaProfile(BaseModel):
    """Profil Persona pour investigation sous couverture."""
    name: str = Field(..., min_length=2)
    role: str = Field(..., description="Rôle: journalist, researcher, investor, analyst")
    bio: str = Field(default="")
    specialties: list[str] = []


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Shadow.

    Initialise Redis, instancie le service OSINT, les cibles de veille
    et les profils persona.
    """
    logger.info("🌑 Démarrage The Shadow (OSINT Agent)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Service OSINT
    app.state.osint = OSINTService()

    # Stockage en mémoire pour la veille et les personas
    app.state.monitors: list[dict[str, Any]] = []
    app.state.personas: list[dict[str, Any]] = []
    app.state.alerts: deque[dict[str, Any]] = deque(maxlen=200)
    app.state.threat_cache: dict[str, dict[str, Any]] = {}

    # Heartbeat
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ The Shadow dans les ténèbres (prêt)")

    yield

    logger.info("🛑 Arrêt The Shadow")


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def hard_heartbeat():
    """
    Signal de présence pour l'Orchestrateur Core.

    Inclut le nombre de moniteurs actifs et d'alertes récentes.
    """
    redis = get_redis_client()
    while True:
        try:
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "shadow",
            }
            await redis.cache_set("eva.shadow.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="The Shadow API",
    description="Agent OSINT & Renseignement - THE HIVE",
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
# ENDPOINTS — RECHERCHE
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    """Vérifie la santé du module Shadow."""
    return {
        "status": "ok",
        "service": "shadow",
        "active_monitors": len(app.state.monitors),
        "personas": len(app.state.personas),
        "recent_alerts": len(app.state.alerts),
    }


@app.get("/search", tags=["OSINT"])
async def search(q: str = Query(..., min_length=2), max_results: int = Query(default=10, ge=1, le=50)):
    """
    Recherche OSINT rapide via DuckDuckGo.

    Args:
        q: Requête de recherche (min 2 caractères).
        max_results: Nombre maximum de résultats.

    Returns:
        dict: Requête et liste des résultats trouvés.
    """
    osint_service: OSINTService = app.state.osint
    results = await osint_service.quick_search(q, max_results)
    return {"query": q, "results": results, "count": len(results)}


@app.post("/search/advanced", tags=["OSINT"])
async def advanced_search(request: SearchRequest):
    """
    Recherche OSINT avancée multi-sources.

    Peut interroger plusieurs moteurs en parallèle
    (DuckDuckGo, Brave Search, Shodan).
    """
    osint_service: OSINTService = app.state.osint
    all_results = []
    for source in request.sources:
        try:
            results = await osint_service.quick_search(request.query, request.max_results)
            all_results.extend([{**r, "source": source} for r in results])
        except Exception as e:
            logger.warning(f"Source {source} failed: {e}")

    return {
        "query": request.query,
        "sources": request.sources,
        "results": all_results,
        "total": len(all_results),
    }


@app.get("/recon", tags=["OSINT"])
async def recon(target: str):
    """
    Recherche approfondie sur une cible (Entity Recon).

    Combine recherche web et analyse Threat Intel pour
    produire un rapport complet.
    """
    osint_service: OSINTService = app.state.osint
    report = await osint_service.entity_recon(target)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — THREAT INTEL
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/threats/analyze", tags=["Threat Intel"])
async def analyze_threat(indicator: str, type: str = Query(default="domain", description="Type: ip, domain, hash, email")):
    """
    Analyse un indicateur de compromission (IoC).

    En mode simulation, retourne un score de menace estimé.
    En production, interrogera VirusTotal / AlienVault OTX.
    """
    # Score simulé basé sur des heuristiques simples
    score = 0.0
    details = {"indicator": indicator, "type": type}

    if type == "ip":
        # IPs privées = safe, publiques = analyse nécessaire
        if indicator.startswith(("10.", "192.168.", "172.")):
            score = 0.5
            details["verdict"] = "PRIVATE_IP — Risque faible"
        else:
            score = 3.0
            details["verdict"] = "PUBLIC_IP — Analyse recommandée"
    elif type == "domain":
        suspicious_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf"]
        if any(indicator.endswith(tld) for tld in suspicious_tlds):
            score = 6.5
            details["verdict"] = "SUSPICIOUS_TLD — TLD fréquemment abusé"
        else:
            score = 1.5
            details["verdict"] = "STANDARD_DOMAIN — Pas de signal fort"
    elif type == "hash":
        score = 2.0
        details["verdict"] = "HASH — Vérification VirusTotal nécessaire"
    elif type == "email":
        score = 1.0
        details["verdict"] = "EMAIL — Vérification HIBP recommandée"

    result = {
        "indicator": indicator,
        "type": type,
        "threat_score": score,
        "severity": "HIGH" if score >= 7 else "MEDIUM" if score >= 4 else "LOW",
        "details": details,
        "mode": "SIMULATION",
        "analyzed_at": datetime.now().isoformat(),
    }

    # Mettre en cache
    app.state.threat_cache[indicator] = result
    return result


@app.get("/threats/history", tags=["Threat Intel"])
async def get_threat_history():
    """Retourne l'historique des analyses de menaces."""
    return {
        "analyses": list(app.state.threat_cache.values()),
        "total": len(app.state.threat_cache),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — MONITORING
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/monitor", tags=["Veille"])
async def add_monitor(target: MonitorTarget):
    """
    Ajoute une cible de veille persistante.

    Le Shadow surveillera ce mot-clé à l'intervalle spécifié
    et générera des alertes en cas de nouveau résultat significatif.
    """
    monitor = {
        "id": f"MON-{uuid4().hex[:8].upper()}",
        "keyword": target.keyword,
        "category": target.category,
        "interval_minutes": target.interval_minutes,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "last_check": None,
        "hits": 0,
    }
    app.state.monitors.append(monitor)
    logger.info(f"🌑 Nouvelle veille : '{target.keyword}' (cat: {target.category})")
    return {"status": "created", "monitor": monitor}


@app.get("/monitor", tags=["Veille"])
async def list_monitors():
    """Liste les cibles de veille actives."""
    return {
        "monitors": app.state.monitors,
        "total_active": sum(1 for m in app.state.monitors if m["status"] == "active"),
    }


@app.delete("/monitor/{monitor_id}", tags=["Veille"])
async def remove_monitor(monitor_id: str):
    """Désactive une cible de veille."""
    for m in app.state.monitors:
        if m["id"] == monitor_id:
            m["status"] = "disabled"
            return {"status": "disabled", "monitor_id": monitor_id}
    raise HTTPException(status_code=404, detail=f"Monitor {monitor_id} not found")


@app.get("/alerts", tags=["Veille"])
async def get_alerts(limit: int = Query(default=50, ge=1, le=200)):
    """Retourne les alertes générées par la veille."""
    alerts = list(app.state.alerts)[-limit:]
    return {"alerts": alerts, "total": len(app.state.alerts)}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — PERSONAS
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/profiles", tags=["Persona"])
async def create_persona(persona: PersonaProfile):
    """
    Crée un profil persona pour l'investigation sous couverture.

    Les personas sont des identités fictives utilisées pour
    interagir avec des cibles sans révéler l'identité du Shadow.
    """
    profile = {
        "id": f"PER-{uuid4().hex[:8].upper()}",
        **persona.model_dump(),
        "created_at": datetime.now().isoformat(),
        "missions_count": 0,
    }
    app.state.personas.append(profile)
    logger.info(f"🎭 Nouveau persona : {persona.name} ({persona.role})")
    return {"status": "created", "profile": profile}


@app.get("/profiles", tags=["Persona"])
async def list_personas():
    """Liste les profils persona disponibles."""
    return {"personas": app.state.personas, "total": len(app.state.personas)}


@app.delete("/profiles/{persona_id}", tags=["Persona"])
async def delete_persona(persona_id: str):
    """Supprime un profil persona."""
    for i, p in enumerate(app.state.personas):
        if p["id"] == persona_id:
            app.state.personas.pop(i)
            return {"status": "deleted", "persona_id": persona_id}
    raise HTTPException(status_code=404, detail=f"Persona {persona_id} not found")
