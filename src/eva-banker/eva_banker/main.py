"""
Application FastAPI de Trading et Gestion FinanciÃ¨re (The Banker).

Ce module est l'Expert B du systÃ¨me MoE. Il est responsable de :
- L'exÃ©cution des ordres de trading sur MetaTrader 5 (via `eva_banker.services.mt5`).
- La validation stricte du risque avant exÃ©cution (Loi 2 - Constitution).
- La surveillance en temps rÃ©el des positions et du drawdown.
- L'activation du Kill-Switch en cas de dÃ©passement des limites.

Architecture :
    - FastAPI pour l'interface REST.
    - Redis pour la communication avec le Core et la rÃ©ception des signaux.
    - MetaTrader 5 (Windows) comme moteur d'exÃ©cution (via service dÃ©diÃ©).
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

from eva_banker.nemesis import NemesisSystem, get_nemesis_system
from eva_banker.services.binance_service import BinanceService
from eva_banker.services.traderepublic_client import TradeRepublicService
from eva_banker.services.mt5 import MT5Service, get_mt5_service
from eva_banker.services.news_filter import NewsFilterService
from eva_banker.services.risk import RiskValidator, get_risk_validator
from eva_banker.skill_library import SkillLibrary
from eva_banker.swarm import BankerSwarm
from shared import (
    AccountBalance,
    Position,
    RiskStatus,
    TradeAction,
    TradeOrder,
    get_settings,
    BaseHealthResponse,
)
from shared.auth_middleware import InternalAuthMiddleware
from shared.probes import check_cognitive_sincerity
from shared.redis_client import get_redis_client, init_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ARCHITECTURE HIÃ‰RARCHIQUE (SPlaTES)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

from eva_banker.brain import AutoTradingEngine, BankerManager, BankerWorker

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODÃˆLES API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class OrderRequest(BaseModel):
    """
    RequÃªte d'exÃ©cution d'ordre de trading.

    Attributes:
        symbol (str): Le symbole financier (ex: XAUUSD).
        action (TradeAction): BUY (Achat) ou SELL (Vente).
        volume (Decimal): La taille du lot.
        stop_loss (Decimal | None): Prix du Stop Loss (Obligatoire).
        take_profit (Decimal | None): Prix du Take Profit (Optionnel).
    """
    symbol: str = Field(..., description="Symbole (ex: XAUUSD)")
    action: TradeAction
    volume: Decimal = Field(..., gt=0, le=5)
    stop_loss: Decimal | None = Field(None, description="Prix Stop Loss (obligatoire)")
    take_profit: Decimal | None = None
    account_id: UUID | None = None


class OrderResponse(BaseModel):
    """
    RÃ©sultat de la tentative d'exÃ©cution d'un ordre.

    Attributes:
        success (bool): Indique si l'ordre a Ã©tÃ© placÃ© avec succÃ¨s.
        ticket (int | None): Le ticket MT5 gÃ©nÃ©rÃ©.
        message (str): Message descriptif du rÃ©sultat.
        risk_check (dict): DÃ©tails de la validation des risques.
    """
    success: bool
    ticket: int | None = None
    order_id: UUID | None = None
    message: str
    risk_check: dict[str, Any] = {}


class RiskCheckRequest(BaseModel):
    """
    ParamÃ¨tres pour une simulation de risque (Pre-Trade).

    Attributes:
        symbol (str): Symbole concernÃ©.
        action (TradeAction): Sens du trade.
        volume (Decimal): Taille du lot.
        stop_loss (Decimal): Niveau de stop-loss envisagÃ©.
    """
    symbol: str
    action: TradeAction
    volume: Decimal
    stop_loss: Decimal
    account_id: UUID | None = None


class RiskCheckResponse(BaseModel):
    """
    RÃ©sultat de l'audit de risque.

    Attributes:
        allowed (bool): Si True, le trade respecte la Constitution (Loi 2).
        risk_percent (Decimal): Pourcentage du capital risquÃ©.
        reason (str | None): Motif du refus si applicable.
    """
    allowed: bool
    risk_percent: Decimal
    reason: str | None = None
    details: dict[str, Any] = {}


class HealthResponse(BaseHealthResponse):
    """
    RÃ©ponse Ã©tendue pour le Health Check du Banker.

    Attributes:
        mt5_connected (bool): Ã‰tat de la connexion au terminal de trading.
        paper_trading (bool): Si True, les ordres ne sont pas exÃ©cutÃ©s rÃ©ellement.
    """
    mt5_connected: bool
    paper_trading: bool


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LIFECYCLE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application Banker.

    Args:
        app (FastAPI): Instance de l'application.

    Yields:
        None: Rend le contrÃ´le aprÃ¨s initialisation.
    """
    logger.info("ðŸ¦ DÃ©marrage The Banker (Architecture HiÃ©rarchique)...")
    settings = get_settings()

    # Redis
    try:
        await init_redis()
        logger.info("âœ… Redis connectÃ©")
    except Exception as e:
        logger.warning(f"âš ï¸ Redis non disponible: {e}")

    # Services
    app.state.settings = settings
    app.state.mt5_service = get_mt5_service()
    app.state.risk_validator = get_risk_validator()
    app.state.binance_service = BinanceService()
    app.state.tr_service = TradeRepublicService()
    
    # CCXT Init
    await app.state.binance_service.initialize()
    # TR Init
    await app.state.tr_service.initialize()

    # HiÃ©rarchie
    app.state.skill_library = SkillLibrary()
    app.state.manager = BankerManager(app.state.skill_library)
    from eva_banker.services.ghost_shield import GhostShield
    app.state.ghost_shield = GhostShield(app.state.mt5_service)
    app.state.worker = BankerWorker(app.state.mt5_service, app.state.ghost_shield)

    # Auto-Trading Engine (Weekend Drift -> DÃ©rive de Week-end)
    app.state.auto_engine = AutoTradingEngine(
        manager=app.state.manager,
        worker=app.state.worker,
        mt5=app.state.mt5_service,
        risk=app.state.risk_validator
    )

    # SystÃ¨me Nemesis
    app.state.nemesis = get_nemesis_system()
    await app.state.nemesis.load_state()

    # Filtre de Nouvelles (News Filter)
    app.state.news_filter = NewsFilterService(
        filter_minutes=settings.risk_news_filter_minutes
    )

    # Telemetry
    app.state.start_time = datetime.now()
    app.state.request_count = 0
    app.state.error_count = 0

    # IntÃ©gration SWARM (Essaim)
    app.state.swarm = BankerSwarm()
    await app.state.swarm.init_mqtt()

    # TÃ¢ches de fond
    asyncio.create_task(swarm_listener())
    asyncio.create_task(hard_heartbeat())
    asyncio.create_task(app.state.news_filter.start_monitoring())

    # Connexion MT5
    mt5_service: MT5Service = app.state.mt5_service
    if await mt5_service.connect():
        logger.info("âœ… MT5 connectÃ©")
        # Detecter l'univers reel puis preparer le premier lot de scan.
        await app.state.auto_engine.refresh_symbol_universe(force=True)
        await mt5_service.initialize_symbols(app.state.auto_engine.get_symbol_batch(advance=False))
        
        # DÃ‰MARRAGE AU LANCEMENT (Seulement aprÃ¨s connexion rÃ©ussie)
        await app.state.auto_engine.start()
        logger.info("ðŸš€ Auto-Trading Engine Started")
    else:
        logger.warning("âš ï¸ MT5 en mode mock (Reconnexion possible en arriÃ¨re-plan)")

    logger.info("âœ… The Banker (SWARM MODE) READY")

    yield

    # ArrÃªt (Shutdown)
    logger.info("ðŸ›‘ ArrÃªt The Banker...")
    if hasattr(app.state, 'auto_engine'):
        await app.state.auto_engine.stop()
    await mt5_service.disconnect()


async def hard_heartbeat():
    """
    Signal haute frÃ©quence pour le Watchdog Rust (Loi 0) et l'Orchestrateur Core.

    Persiste l'Ã©tat dans Redis pour la dÃ©couverte des agents.
    Inclut dÃ©sormais l'Ã©quitÃ© pour le Kill-Switch financier.
    """
    from shared.redis_client import get_redis_client
    redis = get_redis_client()
    mt5_service = app.state.mt5_service

    while True:
        try:
            account = await mt5_service.get_account_info()
            if account is not None:
                payload = {
                    "status": "online",
                    "ts": datetime.now().timestamp(),
                    "expert": "banker",
                    "equity": float(account.equity),
                    "balance": float(account.balance),
                    "currency": account.currency
                }
                # Publication Pub/Sub (temps rÃ©el pour le Kernel)
                await redis.publish("eva.banker.heartbeat", payload)
                # Persistence (dÃ©couverte)
                await redis.cache_set("eva.banker.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

        await asyncio.sleep(0.3)


async def swarm_listener():
    """
    Ã‰coute les commandes broadcast de l'essaim.

    Cette tÃ¢che de fond permet au Banker de rÃ©agir aux ordres globaux
    ou de dÃ©ployer des drones de surveillance.
    """
    from shared.redis_client import get_redis_client
    redis = get_redis_client()
    swarm: BankerSwarm = app.state.swarm

    async def handle_swarm(channel, message):
        action = message.get("action")
        command = message.get("command")
        
        if command == "GLOBAL_STOP":
            logger.critical(f"ðŸ›‘ KILL-SWITCH RECEIVED: {message.get('reason')}")
            # ArrÃªt du moteur d'exÃ©cution The Hive Mind
            if hasattr(app.state, 'auto_engine') and app.state.auto_engine.is_active:
                await app.state.auto_engine.stop()
                
            # Notification d'urgence
            if hasattr(app.state.auto_engine, 'telegram'):
                app.state.auto_engine.telegram.send_sync(
                    f"ðŸš¨ *URGENCE : KILL-SWITCH DÃ‰CLENCHÃ‰* ðŸš¨\n"
                    f"â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”\n"
                    f"ðŸ‘¤ Par : {message.get('issuer', 'Unknown')}\n"
                    f"âš ï¸ Raison : {message.get('reason')}\n"
                    f"â¸ï¸ Le Bot E.V.A est totalement HALTÃ‰."
                )
                
        elif action == "SWARM_SURVEILLANCE":
            # Lancement automatique d'un drone de surveillance
            await swarm.spawn_drone(
                name="GoldSurveillance",
                mission="Surveiller XAUUSD avec le Swarm",
                coro=swarm.run_gold_surveillance(Decimal("2050.0"))
            )

    await redis.subscribe(["eva.all.swarm_command", "eva.banker.swarm_command"], handle_swarm)
    await redis.listen()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# APPLICATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


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

# SecuritÃ© Inter-Agents
app.add_middleware(InternalAuthMiddleware)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENDPOINTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@app.get("/health", response_model=HealthResponse, tags=["SystÃ¨me"])
async def health_check() -> HealthResponse:
    """
    VÃ©rifie la santÃ© du module Banker et la connexion MT5.

    Returns:
        HealthResponse: Statut global, Ã©tat de la connexion MT5 et mode (Paper/Live).
    """
    mt5_service: MT5Service = app.state.mt5_service
    settings = app.state.settings
    return HealthResponse(
        status="ok",
        mt5_connected=mt5_service.is_connected,
        paper_trading=settings.paper_trading,
    )


class AutoTradingRequest(BaseModel):
    """
    ModÃ¨le pour activer/dÃ©sactiver le trading automatique.

    Attributes:
        enable (bool): True pour dÃ©marrer, False pour arrÃªter.
    """
    enable: bool


@app.post("/trading/auto", tags=["Trading"])
async def set_auto_trading(request: AutoTradingRequest):
    """
    Active ou dÃ©sactive le mode Auto-Trading (DÃ©rive de Week-end).

    Ce mode lance une boucle autonome qui analyse et trade pÃ©riodiquement.

    Args:
        request (AutoTradingRequest): Ã‰tat dÃ©sirÃ© (enable=True/False).

    Returns:
        dict: Nouvel Ã©tat du moteur de trading.
    """
    engine: AutoTradingEngine = app.state.auto_engine
    if request.enable:
        await engine.start()
        status = "STARTED"
    else:
        await engine.stop()
        status = "STOPPED"

    return {
        "status": status,
        "active": engine.is_active,
        "symbols": engine.symbols
    }


@app.post("/orders", response_model=OrderResponse, tags=["Trading"])
async def create_order(request: OrderRequest) -> OrderResponse:
    """
    Traite une demande d'ordre de trading via l'architecture hiÃ©rarchique.

    Args:
        request (OrderRequest): DÃ©tails de l'ordre (symbole, volume, SL...).

    Returns:
        OrderResponse: RÃ©sultat de l'exÃ©cution (succÃ¨s/Ã©chec, ticket).

    Raises:
        HTTPException: Si le Stop Loss est manquant (RÃ¨gle ROE).
    """
    # 1. VÃ©rification Stop Loss obligatoire
    if request.stop_loss is None:
        raise HTTPException(
            status_code=400,
            detail="Stop Loss obligatoire (ROE Trading: aucun trade sans SL)",
        )

    # 2. Le Manager dÃ©finit la stratÃ©gie (Skill)
    manager: BankerManager = app.state.manager
    # Simulation de donnÃ©es de marchÃ© pour le manager (incluant VaR)
    market_data = {"price": 2034.50, "returns": [0.001, -0.002, 0.005]}
    skill = manager.plan_strategy(market_data)

    # 3. VÃ©rification de la "SincÃ©ritÃ© Cognitive"
    # On simule l'obtention des activations du LLM
    import torch
    mock_activations = torch.randn(1, 4096)
    is_sincere, sincerity_msg = check_cognitive_sincerity(
        mock_activations,
        "The market shows a strong bullish trend on H4.",
        request.action
    )

    if not is_sincere:
        logger.warning(f"ðŸš« BLOCKING ORDER: {sincerity_msg}")
        return OrderResponse(
            success=False,
            message=sincerity_msg,
            risk_check={"allowed": False, "reason": "COGNITIVE_SINCERITY_FAILURE"}
        )

    # 4. Conversion en TradeOrder
    order = TradeOrder(
        symbol=request.symbol,
        action=request.action,
        volume=request.volume,
        stop_loss_price=request.stop_loss,
        take_profit_price=request.take_profit,
        account_id=request.account_id,
        comment=f"Skill: {skill}"
    )

    # 5. VÃ©rification des risques (Loi 2)
    risk_validator: RiskValidator = app.state.risk_validator
    risk_result = await risk_validator.validate_order(order)

    if not risk_result["allowed"]:
        return OrderResponse(
            success=False,
            message=f"Ordre rejetÃ©: {risk_result['reason']}",
            risk_check=risk_result,
        )

    # 6. Le Worker exÃ©cute la compÃ©tence
    worker: BankerWorker = app.state.worker
    result = await worker.execute_skill(skill, order)

    return OrderResponse(
        success=result["success"],
        ticket=result.get("ticket"),
        order_id=order.id,
        message=f"ExÃ©cutÃ© avec succÃ¨s via {skill}",
        risk_check=risk_result,
    )


@app.get("/positions", response_model=list[Position], tags=["Trading"])
async def get_positions() -> list[Position]:
    """
    RÃ©cupÃ¨re la liste des positions actuellement ouvertes sur MT5.

    Returns:
        list[Position]: Liste des positions avec P&L latent, Swap et Ticket.
    """
    mt5_service: MT5Service = app.state.mt5_service
    return await mt5_service.get_open_positions()


@app.delete("/positions/{ticket}", tags=["Trading"])
async def close_position(ticket: int) -> dict[str, Any]:
    """
    Ferme une position spÃ©cifique via son ticket MT5.

    Args:
        ticket (int): Identifiant unique MT5 de la position Ã  fermer.

    Returns:
        dict[str, Any]: RÃ©sultat de la clÃ´ture (SuccÃ¨s, Prix de clÃ´ture, Profit rÃ©alisÃ©).
    """
    mt5_service: MT5Service = app.state.mt5_service
    result = await mt5_service.close_position(ticket)

    # IntÃ©gration Compliance (Juriste / Loi 5)
    # Si le trade est profitable, on informe l'expert Compliance pour provisionnement URSSAF
    try:
        redis = get_redis_client()
        profit = result.get("profit", 0)

        if profit and float(profit) != 0:
            # Signal pour Compliance (URSSAF)
            await redis.publish("eva.compliance.trades", {
                "ticket_id": ticket,
                "profit": profit,
                "symbol": result.get("symbol", "UNKNOWN"),
                "timestamp": datetime.now().isoformat()
            })

            # Signal pour Master Notification (Sentinel/Telegram)
            await redis.publish("eva.banker.trades", {
                "ticket_id": ticket,
                "profit": profit,
                "symbol": result.get("symbol", "UNKNOWN")
            })

            logger.info("âš–ï¸ Trade profit envoyÃ© Ã  Compliance et Sentinel")
    except Exception as e:
        logger.error(f"Erreur notification trade: {e}")

    return result


@app.get("/account", response_model=AccountBalance, tags=["Compte"])
async def get_account_balance() -> AccountBalance:
    """
    RÃ©cupÃ¨re les informations financiÃ¨res du compte de trading (Equity, Balance, Marge).

    Returns:
        AccountBalance: DonnÃ©es financiÃ¨res temps rÃ©el.
    """
    mt5_service: MT5Service = app.state.mt5_service
    return await mt5_service.get_account_info()


@app.get("/ticks/{symbol}", tags=["Trading"])
async def get_tick(symbol: str):
    """
    RÃ©cupÃ¨re le dernier prix (tick) pour un symbole.

    Args:
        symbol (str): Le symbole financier (ex: EURUSD).

    Returns:
        dict: Dernier tick (bid, ask, time).
    """
    mt5_service: MT5Service = app.state.mt5_service
    return await mt5_service.get_symbol_tick(symbol.upper())


@app.get("/risk/status", response_model=RiskStatus, tags=["Risque"])
async def get_risk_status() -> RiskStatus:
    """
    Fournit un audit instantanÃ© de l'Ã©tat des risques (Loi 2).

    Inclut le pourcentage de Drawdown journalier, le nombre de positions ouvertes
    et l'Ã©tat des filtres (Anti-Tilt, News Trading).

    Returns:
        RiskStatus: Rapport complet de conformitÃ© risque.
    """
    risk_validator: RiskValidator = app.state.risk_validator
    return await risk_validator.get_current_status()


@app.post("/risk/check", response_model=RiskCheckResponse, tags=["Risque"])
async def check_risk(request: RiskCheckRequest) -> RiskCheckResponse:
    """
    Simule une prise de position pour vÃ©rifier sa conformitÃ© sans l'exÃ©cuter.

    UtilisÃ© par le Core ou l'UI pour prÃ©-valider une stratÃ©gie avant d'envoyer
    l'ordre rÃ©el.

    Args:
        request (RiskCheckRequest): ParamÃ¨tres de l'ordre simulÃ©.

    Returns:
        RiskCheckResponse: BoolÃ©en `allowed` et raison du refus si applicable.
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
    ðŸš¨ KILL-SWITCH D'URGENCE.

    Ferme IMMÃ‰DIATEMENT toutes les positions ouvertes, annule les ordres en attente
    et bloque toute nouvelle activitÃ© de trading.
    Doit Ãªtre appelÃ© en cas de perte critique (>4% DD) ou d'anomalie systÃ¨me majeure.

    Returns:
        dict[str, str]: Rapport des fermetures effectuÃ©es.
    """
    mt5_service: MT5Service = app.state.mt5_service
    positions = await mt5_service.get_open_positions()

    # ExÃ©cution parallÃ¨le pour la vitesse et la robustesse (Loi 2 - Kill Switch)
    tasks = [mt5_service.close_position(pos.ticket) for pos in positions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    closed = 0
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Kill switch error: {result}")
            continue
        if result.get("success"):
            closed += 1

    logger.warning(f"ðŸš¨ KILL-SWITCH: {closed}/{len(positions)} positions fermÃ©es")

    return {
        "status": "kill_switch_triggered",
        "message": f"{closed} positions fermÃ©es sur {len(positions)}",
    }


@app.get("/status/crypto", tags=["Compte"])
async def get_crypto_status():
    """
    RÃ©cupÃ¨re l'Ã©tat des comptes Crypto (Binance).

    Returns:
        dict: Soldes par actif.
    """
    from eva_banker.services.binance_service import BinanceService
    binance: BinanceService = app.state.binance_service
    balances = await binance.get_account_balances()
    return {k: float(v) for k, v in balances.items()}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENDPOINTS NEMESIS & NEWS FILTER & TELEMETRY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@app.get("/nemesis/status", tags=["Nemesis"])
async def get_nemesis_status():
    """
    Retourne l'Ã©tat du Nemesis System (mÃ©moire des dÃ©faites).

    Returns:
        dict: Statistiques du systÃ¨me Nemesis.
    """
    nemesis: NemesisSystem = app.state.nemesis
    return nemesis.get_status()


@app.get("/news/filter", tags=["News"])
async def get_news_filter():
    """
    Retourne l'Ã©tat du filtre de nouvelles Ã©conomiques.

    Returns:
        dict: Ã‰tat actif/inactif et prochains Ã©vÃ©nements majeurs.
    """
    news: NewsFilterService = app.state.news_filter
    status = news.get_status()
    # Reformatter pour le frontend
    upcoming = status.get("upcoming_events", [])
    return {
        "is_active": status["is_active"],
        "blocked_until": status["blocked_until"],
        "next_high_impact_events": [
            {
                "event": e["name"],
                "impact": e["impact"],
                "time": e["time"]
            }
            for e in upcoming[:5]
        ]
    }


@app.get("/trading/status", tags=["Trading"])
async def get_trading_status():
    """
    AgrÃ¨ge les donnÃ©es de trading pour le frontend.

    Returns:
        dict: Vue globale (Compte, Positions, Risque).
    """
    mt5_service: MT5Service = app.state.mt5_service
    risk_validator: RiskValidator = app.state.risk_validator

    account = await mt5_service.get_account_info()
    positions = await mt5_service.get_open_positions()
    risk = await risk_validator.get_current_status()

    return {
        "account": {
            "equity": float(account.equity),
            "balance": float(account.balance),
            "margin": float(account.margin),
            "free_margin": float(account.free_margin),
            "currency": account.currency,
            "leverage": account.leverage,
        },
        "positions": [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "action": p.action.value if hasattr(p.action, 'value') else str(p.action),
                "volume": float(p.volume),
                "profit": float(p.profit),
                "open_price": float(p.open_price),
                "current_price": float(p.current_price),
            }
            for p in positions
        ],
        "risk": {
            "daily_drawdown_percent": float(risk.daily_drawdown_percent),
            "trading_allowed": risk.trading_allowed,
            "open_positions": risk.open_positions_count,
            "anti_tilt_active": risk.anti_tilt_active,
            "news_filter_active": risk.news_filter_active,
        },
        "decisions": app.state.auto_engine.latest_decisions,
        "universe": {
            "dynamic": getattr(app.state.auto_engine, "_dynamic_universe_enabled", False),
            "symbols_total": len(app.state.auto_engine.symbols),
            "batch_size": len(app.state.auto_engine.get_symbol_batch(advance=False)),
        }
    }


@app.get("/", tags=["SystÃ¨me"])
async def root():
    """
    Endpoint racine pour health check simple.

    Returns:
        dict: Statut de base du service.
    """
    return {"status": "ok", "service": "eva-banker"}


@app.get("/telemetry", tags=["SystÃ¨me"])
async def get_telemetry():
    """
    Retourne les mÃ©triques de tÃ©lÃ©mÃ©trie du Banker.

    Returns:
        dict: Uptime et compteurs de requÃªtes.
    """
    start_time: datetime = app.state.start_time
    uptime = (datetime.now() - start_time).total_seconds()
    return {
        "service_name": "banker",
        "uptime_seconds": int(uptime),
        "requests_total": app.state.request_count,
        "errors_total": app.state.error_count,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/circuit-breaker/status", tags=["SystÃ¨me"])
async def get_circuit_breaker():
    """
    Retourne l'Ã©tat du circuit-breaker du Banker.

    Le circuit est ouvert si le Nemesis System ou le filtre de news bloque le trading.

    Returns:
        dict: Ã‰tat OPEN/CLOSED et compteurs.
    """
    nemesis: NemesisSystem = app.state.nemesis
    news: NewsFilterService = app.state.news_filter

    # Le circuit-breaker est "OPEN" si Nemesis ou News bloquent le trading
    trading_blocked = nemesis.should_block_trading() or news.should_block_trading()

    if trading_blocked:
        state = "OPEN"
        failures = sum(nemesis.known_nemeses.values())
    else:
        state = "CLOSED"
        failures = 0

    return {
        "name": "banker_trading",
        "state": state,
        "failures": failures,
        "failure_threshold": 3,  # Nemesis threshold
    }


@app.get("/accounts/propfirm", tags=["Compte"])
async def get_propfirm_accounts():
    """
    Retourne les comptes Prop Firm (Hydra Protocol).

    Returns:
        list[dict]: Liste des comptes financÃ©s.
    """
    mt5_service: MT5Service = app.state.mt5_service
    account = await mt5_service.get_account_info()

    # En mode lite, on retourne le compte principal comme un "prop firm account"
    return [
        {
            "id": str(account.login),
            "name": f"Account {account.login}",
            "server": account.server,
            "balance": float(account.balance),
            "equity": float(account.equity),
            "phase": "CHALLENGE",
            "status": "active",
            "max_drawdown": 4.0,
            "daily_drawdown": 0.0,
        }
    ]
