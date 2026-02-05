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
)
from shared.redis_client import get_redis_client, init_redis

from eva_banker.services.mt5 import MT5Service, get_mt5_service
from eva_banker.services.risk import RiskValidator, get_risk_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application Banker.

    Charge la configuration, connecte Redis et établit la liaison critique avec
    le terminal MetaTrader 5. Si MT5 n'est pas disponible, bascule en mode
    'Mock' pour permettre les tests hors-ligne (Simulation).

    Args:
        app (FastAPI): Instance de l'application.

    Yields:
        None: Rend le contrôle après initialisation.
    """
    logger.info("🏦 Démarrage The Banker...")
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

    # Connexion MT5
    mt5_service: MT5Service = app.state.mt5_service
    if await mt5_service.connect():
        logger.info("✅ MT5 connecté")
    else:
        logger.warning("⚠️ MT5 en mode mock")

    logger.info("✅ The Banker prêt")

    yield

    # Shutdown
    logger.info("🛑 Arrêt The Banker...")
    await mt5_service.disconnect()


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
    Traite une demande d'ordre de trading (Achat/Vente).

    Cette fonction est critique et suit un protocole strict :
    1. **Validation Initiale** : Vérifie la présence d'un Stop Loss (Obligatoire par ROE).
    2. **Contrôle des Risques** : Appelle le `RiskValidator` pour simuler l'impact
       sur le drawdown journalier et l'exposition totale.
    3. **Exécution** : Si le risque est validé, transmet l'ordre au terminal MT5.
    4. **Confirmation** : Retourne le ticket MT5 ou la raison du rejet.

    Args:
        request (OrderRequest): Détails de l'ordre (Symbole, Volume, SL, TP).

    Returns:
        OrderResponse: Résultat de l'exécution, incluant le ticket et l'audit risque.

    Raises:
        HTTPException(400): Si le Stop Loss est absent ou invalide.
    """
    # Vérification Stop Loss obligatoire (ROE Trading)
    if request.stop_loss is None:
        raise HTTPException(
            status_code=400,
            detail="Stop Loss obligatoire (ROE Trading: aucun trade sans SL)",
        )

    # Conversion en TradeOrder
    order = TradeOrder(
        symbol=request.symbol,
        action=request.action,
        volume=request.volume,
        stop_loss_price=request.stop_loss,
        take_profit_price=request.take_profit,
        account_id=request.account_id,
    )

    # Vérification des risques
    risk_validator: RiskValidator = app.state.risk_validator
    risk_result = await risk_validator.validate_order(order)

    if not risk_result["allowed"]:
        return OrderResponse(
            success=False,
            message=f"Ordre rejeté: {risk_result['reason']}",
            risk_check=risk_result,
        )

    # Exécution de l'ordre
    mt5_service: MT5Service = app.state.mt5_service
    result = await mt5_service.execute_order(order)

    return OrderResponse(
        success=result["success"],
        ticket=result.get("ticket"),
        order_id=order.id,
        message=result.get("message", "Ordre exécuté"),
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
