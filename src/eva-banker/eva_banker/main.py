"""
Application FastAPI de Trading et Gestion Financière (The Banker).

Ce module est l'Expert B du système MoE. Il est responsable de :
- L'exécution des ordres de trading sur MetaTrader 5 (via `eva_banker.services.mt5`).
- La validation stricte du risque avant exécution (Loi 2 - Constitution).
- La surveillance en temps réel des positions et du drawdown.
- L'activation du Kill-Switch en cas de dépassement des limites.

Architecture :
    - FastAPI pour l'interface REST.
    - Redis pour la communication avec le Core et la réception des signaux.
    - MetaTrader 5 (Windows) comme moteur d'exécution (via service dédié).
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared import (
    AccountBalance,
    Position,
    PropFirmAccount,
    RiskStatus,
    TradeAction,
    TradeOrder,
    get_settings,
    symlog,
    inv_symlog,
    calculate_var,
    calculate_cvar,
)
from shared.redis_client import get_redis_client, init_redis

from eva_banker.services.mt5 import MT5Service, get_mt5_service
from eva_banker.services.risk import RiskValidator, get_risk_validator
from eva_banker.skill_library import SkillLibrary, SkilledBehavior
from eva_banker.models.gnn_model import TFTGNNModel
from eva_banker.swarm import BankerSwarm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE HIÉRARCHIQUE (SPlaTES)
# ═══════════════════════════════════════════════════════════════════════════════

class BankerManager:
    """
    NIVEAU HAUT : Le Manager (Abstract World Model).
    Planifie les stratégies en utilisant TFT-GNN et la conscience du risque.
    """
    def __init__(self, library: SkillLibrary):
        self.library = library
        # Initialisation du modèle (dims fictives pour l'exemple)
        self.brain = TFTGNNModel(asset_dim=5, temporal_dim=64, hidden_dim=128)

    def plan_strategy(self, market_history: dict) -> SkilledBehavior:
        """
        Analyse le marché via TFT-GNN et injecte VaR/CVaR.
        """
        # 1. Calcul des métriques de risque adaptatives (Inhibiteur interne)
        returns = market_history.get("returns", [])
        var = calculate_var(returns)
        cvar = calculate_cvar(returns)
        
        # 2. Préparation des données pour le modèle (Normalisées via Symlog)
        price = symlog(market_history.get("price", 0))
        
        logger.info(f"Manager decision core triggered. Price: {price}, VaR: {var}, CVaR: {cvar}")
        
        # Si le risque (VaR) est trop élevé, on bascule en mode conservateur
        if var < -0.02: # Perte potentielle > 2% attendue
            logger.warning("High VaR detected. Selecting HEDGING skill.")
            return SkilledBehavior.HEDGING
            
        return SkilledBehavior.SCALPING

class BankerWorker:
    """
    NIVEAU BAS : L'Exécutant (Worker).
    Support de GhostShield pour l'invisibilité HFT.
    """
    def __init__(self, mt5_service: MT5Service, ghost_shield=None):
        self.mt5 = mt5_service
        self.ghost = ghost_shield

    async def execute_skill(self, skill: SkilledBehavior, order: TradeOrder):
        logger.info(f"Worker executing skill: {skill}")
        if self.ghost and skill != SkilledBehavior.HEDGING: # Le hedging doit être direct
            return await self.ghost.execute_obfuscated_order(order)
        return await self.mt5.execute_skill(skill, order)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES API
# ═══════════════════════════════════════════════════════════════════════════════


class OrderRequest(BaseModel):
    """Requête d'ordre de trading"""
    symbol: str = Field(..., description="Symbole (ex: XAUUSD)")
    action: TradeAction
    volume: Decimal = Field(..., gt=0, le=5)
    stop_loss: Decimal | None = Field(None, description="Prix Stop Loss (obligatoire)")
    take_profit: Decimal | None = None
    account_id: UUID | None = None


class OrderResponse(BaseModel):
    """Réponse après exécution d'ordre"""
    success: bool
    ticket: int | None = None
    order_id: UUID | None = None
    message: str
    risk_check: dict[str, Any] = {}


class RiskCheckRequest(BaseModel):
    """Requête de vérification de risque"""
    symbol: str
    action: TradeAction
    volume: Decimal
    stop_loss: Decimal
    account_id: UUID | None = None


class RiskCheckResponse(BaseModel):
    """Réponse de vérification de risque"""
    allowed: bool
    risk_percent: Decimal
    reason: str | None = None
    details: dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Réponse de santé"""
    status: str
    mt5_connected: bool
    paper_trading: bool
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application Banker.
    """
    logger.info("🏦 Démarrage The Banker (Hierarchical Architecture)...")
    settings = get_settings()

    # Redis
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.settings = settings
    app.state.mt5_service = get_mt5_service()
    app.state.risk_validator = get_risk_validator()
    
    # Hiérarchie
    app.state.skill_library = SkillLibrary()
    app.state.manager = BankerManager(app.state.skill_library)
    from eva_banker.services.ghost_shield import GhostShield
    app.state.ghost_shield = GhostShield(app.state.mt5_service)
    app.state.worker = BankerWorker(app.state.mt5_service, app.state.ghost_shield)

    # Intégration SWARM
    app.state.swarm = BankerSwarm()
    await app.state.swarm.init_mqtt()
    
    # Tâche de fond pour écouter les ordres Swarm
    asyncio.create_task(swarm_listener())
    asyncio.create_task(hard_heartbeat())

    # Connexion MT5
    mt5_service: MT5Service = app.state.mt5_service
    if await mt5_service.connect():
        logger.info("✅ MT5 connecté")
    else:
        logger.warning("⚠️ MT5 en mode mock")

    logger.info("✅ The Banker (SWARM MODE) READY")

    yield

    # Shutdown
    logger.info("🛑 Arrêt The Banker...")
    await mt5_service.disconnect()


async def hard_heartbeat():
    """
    Signal haute fréquence pour le Watchdog Rust (Loi 0) et l'Orchestrateur Core.
    Persiste l'état dans Redis pour la découverte des agents.
    """
    from shared.redis_client import get_redis_client
    redis = get_redis_client()
    while True:
        payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "banker"}
        # Publication Pub/Sub (temps réel)
        await redis.publish("eva.banker.heartbeat", payload)
        # Persistence (découverte)
        await redis.cache_set("eva.banker.status", payload, ttl_seconds=10)
        await asyncio.sleep(0.3)


async def swarm_listener():
    """
    Écoute les commandes broadcast de l'essaim.
    """
    from shared.redis_client import get_redis_client
    redis = get_redis_client()
    swarm: BankerSwarm = app.state.swarm
    
    async def handle_swarm(channel, message):
        action = message.get("action")
        if action == "SWARM_SURVEILLANCE":
            # Lancement automatique d'un drone de surveillance
            await swarm.spawn_drone(
                name="GoldSurveillance",
                mission="Surveiller XAUUSD avec le Swarm",
                coro=swarm.run_gold_surveillance(Decimal("2050.0"))
            )

    await redis.subscribe(["eva.all.swarm_command", "eva.banker.swarm_command"], handle_swarm)
    await redis.listen()


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="The Banker API",
    description="Expert Trading - THE HIVE",
    version="0.1.0",
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


@app.get("/health", response_model=HealthResponse, tags=["Système"])
async def health_check() -> HealthResponse:
    """
    Vérifie la santé du module Banker et la connexion MT5.

    Returns:
        HealthResponse: Statut global, état de la connexion MT5 et mode (Paper/Live).
    """
    mt5_service: MT5Service = app.state.mt5_service
    settings = app.state.settings
    return HealthResponse(
        status="ok",
        mt5_connected=mt5_service.is_connected,
        paper_trading=settings.paper_trading,
    )


@app.post("/orders", response_model=OrderResponse, tags=["Trading"])
async def create_order(request: OrderRequest) -> OrderResponse:
    """
    Traite une demande d'ordre de trading via l'architecture hiérarchique.
    """
    # 1. Vérification Stop Loss obligatoire
    if request.stop_loss is None:
        raise HTTPException(
            status_code=400,
            detail="Stop Loss obligatoire (ROE Trading: aucun trade sans SL)",
        )

    # 2. Le Manager définit la stratégie (Skill)
    manager: BankerManager = app.state.manager
    # Simulation de données de marché pour le manager
    skill = manager.plan_strategy({"price": 2034.50})

    # 3. Conversion en TradeOrder
    order = TradeOrder(
        symbol=request.symbol,
        action=request.action,
        volume=request.volume,
        stop_loss_price=request.stop_loss,
        take_profit_price=request.take_profit,
        account_id=request.account_id,
        comment=f"Skill: {skill}"
    )

    # 4. Vérification des risques (Loi 2)
    risk_validator: RiskValidator = app.state.risk_validator
    risk_result = await risk_validator.validate_order(order)

    if not risk_result["allowed"]:
        return OrderResponse(
            success=False,
            message=f"Ordre rejeté: {risk_result['reason']}",
            risk_check=risk_result,
        )

    # 5. Le Worker exécute la compétence
    worker: BankerWorker = app.state.worker
    result = await worker.execute_skill(skill, order)

    return OrderResponse(
        success=result["success"],
        ticket=result.get("ticket"),
        order_id=order.id,
        message=f"Exécuté avec succès via {skill}",
        risk_check=risk_result,
    )


@app.get("/positions", response_model=list[Position], tags=["Trading"])
async def get_positions() -> list[Position]:
    """
    Récupère la liste des positions actuellement ouvertes sur MT5.

    Returns:
        list[Position]: Liste des positions avec P&L latent, Swap et Ticket.
    """
    mt5_service: MT5Service = app.state.mt5_service
    return await mt5_service.get_open_positions()


@app.delete("/positions/{ticket}", tags=["Trading"])
async def close_position(ticket: int) -> dict[str, Any]:
    """
    Ferme une position spécifique via son ticket MT5.

    Args:
        ticket (int): Identifiant unique MT5 de la position à fermer.

    Returns:
        dict[str, Any]: Résultat de la clôture (Succès, Prix de clôture, Profit réalisé).
    """
    mt5_service: MT5Service = app.state.mt5_service
    result = await mt5_service.close_position(ticket)
    
    # Intégration Compliance (Juriste / Loi 5)
    # Si le trade est profitable, on informe l'expert Compliance pour provisionnement URSSAF
    if result.get("success") and result.get("profit", 0) > 0:
        from shared.redis_client import get_redis_client
        redis = get_redis_client()
        await redis.publish("eva.compliance.trades", {
            "ticket_id": ticket,
            "profit": result.get("profit"),
            "symbol": result.get("symbol", "UNKNOWN"),
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"⚖️ Trade profit envoyé à Compliance pour provisionnement")
        
    return result


@app.get("/account", response_model=AccountBalance, tags=["Compte"])
async def get_account_balance() -> AccountBalance:
    """
    Récupère les informations financières du compte de trading (Equity, Balance, Marge).

    Returns:
        AccountBalance: Données financières temps réel.
    """
    mt5_service: MT5Service = app.state.mt5_service
    return await mt5_service.get_account_info()


@app.get("/risk/status", response_model=RiskStatus, tags=["Risque"])
async def get_risk_status() -> RiskStatus:
    """
    Fournit un audit instantané de l'état des risques (Loi 2).

    Inclut le pourcentage de Drawdown journalier, le nombre de positions ouvertes
    et l'état des filtres (Anti-Tilt, News Trading).

    Returns:
        RiskStatus: Rapport complet de conformité risque.
    """
    risk_validator: RiskValidator = app.state.risk_validator
    return await risk_validator.get_current_status()


@app.post("/risk/check", response_model=RiskCheckResponse, tags=["Risque"])
async def check_risk(request: RiskCheckRequest) -> RiskCheckResponse:
    """
    Simule une prise de position pour vérifier sa conformité sans l'exécuter.

    Utilisé par le Core ou l'UI pour pré-valider une stratégie avant d'envoyer
    l'ordre réel.

    Args:
        request (RiskCheckRequest): Paramètres de l'ordre simulé.

    Returns:
        RiskCheckResponse: Booléen `allowed` et raison du refus si applicable.
    """
    order = TradeOrder(
        symbol=request.symbol,
        action=request.action,
        volume=request.volume,
        stop_loss_price=request.stop_loss,
        account_id=request.account_id,
    )

    risk_validator: RiskValidator = app.state.risk_validator
    result = await risk_validator.validate_order(order)

    return RiskCheckResponse(
        allowed=result["allowed"],
        risk_percent=result.get("risk_percent", Decimal("0")),
        reason=result.get("reason"),
        details=result,
    )


@app.post("/risk/kill-switch", tags=["Risque"])
async def trigger_kill_switch() -> dict[str, str]:
    """
    🚨 KILL-SWITCH D'URGENCE.

    Ferme IMMÉDIATEMENT toutes les positions ouvertes, annule les ordres en attente
    et bloque toute nouvelle activité de trading.
    Doit être appelé en cas de perte critique (>4% DD) ou d'anomalie système majeure.

    Returns:
        dict[str, str]: Rapport des fermetures effectuées.
    """
    mt5_service: MT5Service = app.state.mt5_service
    positions = await mt5_service.get_open_positions()

    closed = 0
    for pos in positions:
        result = await mt5_service.close_position(pos.ticket)
        if result.get("success"):
            closed += 1

    logger.warning(f"🚨 KILL-SWITCH: {closed}/{len(positions)} positions fermées")

    return {
        "status": "kill_switch_triggered",
        "message": f"{closed} positions fermées sur {len(positions)}",
    }
