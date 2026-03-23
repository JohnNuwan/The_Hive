"""
EVA RWA — The Sovereign : Gestionnaire d'Actifs Réels de THE HIVE.

Expert K du système d'experts. Responsable de :
- La gestion du portefeuille d'actifs réels tokenisés (RealT, Centrifuge).
- Le suivi de la stratégie Sovereign Fund (Immobilier, Énergie, DeFi).
- Le monitoring IoT des actifs physiques (panneaux solaires, capteurs).
- Les recommandations d'acquisition basées sur la phase stratégique.

Architecture :
    - Portfolio sur disque (JSON) avec CRUD complet.
    - Stratégie d'investissement en 3 phases (Énergie → Industrie → Diplomatie).
    - Heartbeat Redis pour la découverte par le Core.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_rwa.token_bridge import TokenBridge
from eva_rwa.iot_controller import IotController
from eva_rwa.services.sovereign_fund import SovereignFund
from eva_rwa.services.asset_tracker import AssetTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES API
# ═══════════════════════════════════════════════════════════════════════════════


class AssetRecord(BaseModel):
    """Représente un actif réel dans le portefeuille."""
    id: str = Field(default_factory=lambda: f"RWA-{uuid4().hex[:8].upper()}")
    name: str = Field(..., description="Nom de l'actif (ex: Terrain Solaire Var)")
    category: str = Field(..., description="Catégorie : real_estate, energy, defi, land, industrial")
    valuation: float = Field(..., gt=0, description="Valorisation actuelle en EUR")
    annual_yield: float = Field(default=0.0, ge=0, le=1.0, description="Rendement annuel (0.0 à 1.0)")
    acquisition_date: datetime = Field(default_factory=datetime.now)
    location: str | None = Field(None, description="Localisation physique (si applicable)")
    tokenized: bool = Field(default=False, description="Si l'actif est tokenisé on-chain")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssetAcquisition(BaseModel):
    """Requête d'acquisition d'un nouvel actif."""
    name: str = Field(..., min_length=2)
    category: str = Field(..., description="Catégorie : real_estate, energy, defi, land, industrial")
    valuation: float = Field(..., gt=0)
    annual_yield: float = Field(default=0.0, ge=0, le=1.0)
    location: str | None = None
    tokenized: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortfolioReport(BaseModel):
    """Rapport consolidé du portefeuille RWA."""
    total_assets: int
    total_valuation: float
    weighted_yield: float
    by_category: dict[str, dict[str, Any]]
    phase: str
    phase_progress: float
    assets: list[dict[str, Any]]
    currency: str = "EUR"
    generated_at: datetime = Field(default_factory=datetime.now)


class InvestmentRecommendation(BaseModel):
    """Recommandation d'investissement du Sovereign Fund."""
    priority: str
    category: str
    reason: str
    target_amount: float
    expected_yield: float


class EnergyTelemetry(BaseModel):
    """Données de télémétrie des capteurs IoT."""
    solar_production_w: float
    battery_level: float
    external_temp: float
    daily_production_kwh: float
    monthly_savings_eur: float
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application RWA (The Sovereign).

    Initialise Redis, le portfolio d'actifs, le contrôleur IoT,
    le Sovereign Fund et l'AssetTracker.
    """
    logger.info("👑 Démarrage EVA RWA (The Sovereign)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.bridge = TokenBridge()
    app.state.iot = IotController()
    app.state.fund = SovereignFund(app.state.bridge)
    app.state.tracker = AssetTracker(app.state.bridge)

    # Heartbeat
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ EVA RWA — The Sovereign est éveillé")
    yield
    logger.info("🛑 Arrêt EVA RWA")


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def hard_heartbeat():
    """
    Signal de présence pour l'Orchestrateur Core.

    Publie l'état « online » dans Redis sous la clé `eva.rwa.status`
    avec la valorisation totale du portefeuille.
    """
    try:
        redis = get_redis_client()
    except Exception:
        redis = None

    while True:
        try:
            if redis:
                payload = {
                    "status": "online",
                    "ts": datetime.now().timestamp(),
                    "expert": "rwa",
                }
                await redis.cache_set("eva.rwa.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA RWA — The Sovereign API",
    description="Gestion d'Actifs Réels & Stratégie Souveraine - THE HIVE",
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
    """Vérifie la santé du module RWA (The Sovereign)."""
    bridge: TokenBridge = app.state.bridge
    portfolio = bridge.get_portfolio()
    return {
        "status": "online",
        "service": "rwa",
        "total_assets": len(portfolio["assets"]),
        "total_valuation": portfolio["total_valuation"],
    }


@app.get("/portfolio", tags=["Portfolio"], response_model=PortfolioReport)
async def get_portfolio():
    """
    Rapport consolidé du portefeuille d'actifs réels.

    Inclut la répartition par catégorie, le rendement pondéré,
    la phase stratégique actuelle et la progression.
    """
    tracker: AssetTracker = app.state.tracker
    return tracker.get_full_report()


@app.post("/portfolio/acquire", tags=["Portfolio"])
async def acquire_asset(acquisition: AssetAcquisition):
    """
    Enregistre l'acquisition d'un nouvel actif réel.

    Vérifie la cohérence avec la stratégie du Sovereign Fund
    avant d'ajouter l'actif au portefeuille.
    """
    bridge: TokenBridge = app.state.bridge
    fund: SovereignFund = app.state.fund

    # Créer l'enregistrement
    asset = AssetRecord(
        name=acquisition.name,
        category=acquisition.category,
        valuation=acquisition.valuation,
        annual_yield=acquisition.annual_yield,
        location=acquisition.location,
        tokenized=acquisition.tokenized,
        metadata=acquisition.metadata,
    )

    # Vérifier l'alignement stratégique
    alignment = fund.check_alignment(asset.category)

    # Ajouter au portfolio
    bridge.add_asset(asset.model_dump(mode="json"))
    logger.info(f"👑 Actif acquis : {asset.name} ({asset.category}) — {asset.valuation}€")

    return {
        "status": "acquired",
        "asset_id": asset.id,
        "strategic_alignment": alignment,
        "portfolio_total": bridge.get_portfolio()["total_valuation"],
    }


@app.delete("/portfolio/{asset_id}", tags=["Portfolio"])
async def remove_asset(asset_id: str):
    """Retire un actif du portefeuille (cession)."""
    bridge: TokenBridge = app.state.bridge
    result = bridge.remove_asset(asset_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Actif {asset_id} non trouvé")
    return {"status": "removed", "asset_id": asset_id}


@app.put("/portfolio/{asset_id}/valuate", tags=["Portfolio"])
async def update_valuation(asset_id: str, new_valuation: float = Query(..., gt=0)):
    """Met à jour la valorisation d'un actif existant."""
    bridge: TokenBridge = app.state.bridge
    result = bridge.update_valuation(asset_id, new_valuation)
    if not result:
        raise HTTPException(status_code=404, detail=f"Actif {asset_id} non trouvé")
    return {"status": "updated", "asset_id": asset_id, "new_valuation": new_valuation}


@app.get("/strategy", tags=["Sovereign Fund"])
async def get_strategy():
    """
    Retourne la stratégie d'investissement actuelle du Sovereign Fund.

    Inclut la phase en cours, les objectifs, les recommandations
    d'acquisition et la progression vers la souveraineté.
    """
    fund: SovereignFund = app.state.fund
    return fund.get_strategy_report()


@app.get("/strategy/recommendations", tags=["Sovereign Fund"])
async def get_recommendations():
    """
    Génère des recommandations d'investissement basées sur la phase actuelle
    et la composition du portefeuille.
    """
    fund: SovereignFund = app.state.fund
    return fund.get_recommendations()


@app.get("/iot/telemetry", tags=["IoT"])
async def get_telemetry():
    """Récupère les données de télémétrie des capteurs IoT (énergie solaire)."""
    iot: IotController = app.state.iot
    data = iot.get_telemetry()
    return data


@app.get("/iot/energy/history", tags=["IoT"])
async def get_energy_history(days: int = Query(default=7, ge=1, le=90)):
    """Historique de production énergétique sur N jours."""
    iot: IotController = app.state.iot
    return iot.get_history(days)


@app.get("/categories", tags=["Portfolio"])
async def get_categories():
    """Liste les catégories d'actifs et leur répartition."""
    tracker: AssetTracker = app.state.tracker
    return tracker.get_category_breakdown()
