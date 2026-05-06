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
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from eva_banker.nemesis import NemesisSystem, get_nemesis_system
from eva_banker.services.binance_service import BinanceService
from eva_banker.services.copy_trading import CopyTradingRouter
from eva_banker.services.traderepublic_client import TradeRepublicService
from eva_banker.services.mt5 import MT5Service, get_mt5_service
from eva_banker.services.news_filter import NewsFilterService
from eva_banker.services.risk import RiskValidator, get_risk_validator
from eva_banker.skill_library import SkillLibrary
from eva_banker.swarm import BankerSwarm
from shared import (
    AccountBalance,
    ConnectorMode,
    Position,
    RiskStatus,
    RuntimeMode,
    TradeAction,
    TradeOrder,
    TradingDecisionEnvelope,
    get_settings,
    BaseHealthResponse,
    OrderSource,
    )
from shared.auth_middleware import InternalAuthMiddleware
from shared.probes import check_cognitive_sincerity
from shared.redis_client import get_redis_client, init_redis


def configure_logging() -> None:
    """Configure une journalisation console lisible pour le banker local.

    Le banker tourne surtout sur Windows a cote de MT5. On force donc une
    sortie UTF-8 et on prefere ``rich`` quand il est disponible afin de
    supprimer le bruit visuel et de rendre les niveaux de logs plus lisibles.
    """
    level_name = os.getenv("BANKER_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    rich_enabled = os.getenv("BANKER_RICH_LOGS", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    if rich_enabled:
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            console = Console(stderr=True, soft_wrap=True)
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                datefmt="[%H:%M:%S]",
                handlers=[
                    RichHandler(
                        console=console,
                        rich_tracebacks=False,
                        show_path=False,
                        markup=False,
                    )
                ],
                force=True,
            )
            logging.captureWarnings(True)
            return
        except Exception:
            pass

    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )


configure_logging()
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
        account_id (UUID | None): Identifiant d'une cible distante optionnelle.
        comment (str): Commentaire libre associe a l'ordre.
        source (OrderSource): Origine metier de l'ordre.
    """
    symbol: str = Field(..., description="Symbole (ex: XAUUSD)")
    action: TradeAction
    volume: Decimal = Field(..., gt=0, le=5)
    stop_loss: Decimal | None = Field(None, description="Prix Stop Loss (obligatoire)")
    take_profit: Decimal | None = None
    account_id: UUID | None = None
    comment: str = Field(default="", description="Commentaire libre pour l'ordre")
    source: OrderSource = Field(default=OrderSource.CHAT, description="Origine de l'ordre")


class OrderResponse(BaseModel):
    """
    RÃ©sultat de la tentative d'exÃ©cution d'un ordre.

    Attributes:
        success (bool): Indique si l'ordre a Ã©tÃ© placÃ© avec succÃ¨s.
        ticket (int | None): Le ticket MT5 gÃ©nÃ©rÃ©.
        message (str): Message descriptif du rÃ©sultat.
        risk_check (dict): DÃ©tails de la validation des risques.
        copy_results (list[dict[str, Any]]): Resultats des copies distantes.
    """
    success: bool
    ticket: int | None = None
    order_id: UUID | None = None
    message: str
    risk_check: dict[str, Any] = {}
    copy_results: list[dict[str, Any]] = []


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


class PositionModifyRequest(BaseModel):
    """
    Parametres de modification d'une position existante.

    Attributes:
        stop_loss (Decimal | None): Nouveau niveau de stop loss.
        take_profit (Decimal | None): Nouveau niveau de take profit.
    """

    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None


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


async def _read_mt5_snapshot(mt5_service: MT5Service) -> tuple[AccountBalance | None, list[Position]]:
    """
    Lit un instantane MT5 sans laisser de positions `None`.

    Args:
        mt5_service (MT5Service): Service MT5 actif.

    Returns:
        tuple[AccountBalance | None, list[Position]]: Compte courant si disponible
        et liste de positions ouverte, vide si MT5 ne repond plus.
    """
    account = await mt5_service.get_account_info()
    positions = await mt5_service.get_open_positions()
    return account, positions or []


def _is_mt5_live_offline(mt5_service: MT5Service) -> bool:
    """
    Indique si le mode reel MT5 est hors ligne.

    Args:
        mt5_service (MT5Service): Service MT5 actif.

    Returns:
        bool: True si aucun mock n'est actif et que la connexion MT5 est perdue.
    """
    return not mt5_service.mock_mode and not mt5_service.is_connected


def _derive_runtime_mode(auto_engine) -> RuntimeMode:
    """
    Derive le mode runtime canonique du banker.

    Args:
        auto_engine: Instance du moteur d'auto-trading.

    Returns:
        RuntimeMode: Mode stable expose aux autres services.
    """
    if not getattr(auto_engine, "is_active", False):
        return RuntimeMode.MAINTENANCE
    if bool(getattr(auto_engine, "_cpu_live_mode", False)):
        return RuntimeMode.TRAINING_CPU_LIVE
    return RuntimeMode.DEMO_LIVE


def _build_connector_status(app: FastAPI) -> dict[str, Any]:
    """
    Construit une vue explicite des connecteurs du banker.

    Args:
        app (FastAPI): Application Banker courante.

    Returns:
        dict[str, Any]: Etat explicite des connecteurs critiques.
    """
    mt5_service: MT5Service = app.state.mt5_service
    auto_engine = app.state.auto_engine
    binance: BinanceService = app.state.binance_service
    trade_republic: TradeRepublicService = app.state.tr_service
    gnn_available = not bool(getattr(getattr(auto_engine, "manager", None), "brain", None) is None)
    gnn_stub = bool(getattr(getattr(auto_engine, "manager", None), "brain", None) and getattr(getattr(auto_engine.manager, "brain", None), "gnn", None) and getattr(auto_engine.manager.brain.gnn, "stub", False))
    live_inference_url = ""
    if hasattr(auto_engine, "_resolve_live_inference_url"):
        try:
            live_inference_url = str(auto_engine._resolve_live_inference_url() or "")
        except Exception:
            live_inference_url = ""

    mt5_mode = ConnectorMode.LIVE if mt5_service.is_connected and not mt5_service.mock_mode else ConnectorMode.PAPER
    if not mt5_service.is_connected and not mt5_service.mock_mode:
        mt5_mode = ConnectorMode.DISABLED

    binance_mode = ConnectorMode.PAPER
    if getattr(binance, "api_key", None) and getattr(binance, "api_secret", None):
        binance_mode = ConnectorMode.LIVE

    trade_republic_mode = ConnectorMode.PAPER
    if getattr(trade_republic, "is_connected", False):
        trade_republic_mode = ConnectorMode.LIVE
    elif getattr(trade_republic, "phone", None) and getattr(trade_republic, "pin", None):
        trade_republic_mode = ConnectorMode.PAPER

    gnn_mode = ConnectorMode.DISABLED
    if gnn_available and not gnn_stub:
        gnn_mode = ConnectorMode.LIVE

    vllm_mode = ConnectorMode.DISABLED if bool(getattr(auto_engine, "_cpu_live_mode", False)) else ConnectorMode.LIVE
    live_inference_mode = ConnectorMode.LIVE if live_inference_url else ConnectorMode.DISABLED

    return {
        "mt5": {
            "mode": mt5_mode.value,
            "connected": mt5_service.is_connected,
            "mock_mode": mt5_service.mock_mode,
        },
        "binance": {
            "mode": binance_mode.value,
            "testnet": bool(getattr(binance, "testnet", False)),
        },
        "traderepublic": {
            "mode": trade_republic_mode.value,
            "connected": bool(getattr(trade_republic, "is_connected", False)),
        },
        "gnn": {
            "mode": gnn_mode.value,
            "stub": gnn_stub,
            "role": "consultatif" if bool(getattr(auto_engine, "_cpu_live_mode", False)) else "fusionne",
        },
        "vllm": {
            "mode": vllm_mode.value,
            "required": not bool(getattr(auto_engine, "_cpu_live_mode", False)),
        },
        "live_inference": {
            "mode": live_inference_mode.value,
            "required": bool(getattr(auto_engine, "_cpu_live_mode", False)),
            "url": live_inference_url or None,
        },
    }


async def _publish_trading_status_snapshot(payload: dict[str, Any]) -> None:
    """
    Met en cache la vue de trading agregee pour EVA Core.

    Args:
        payload (dict[str, Any]): Etat trading agrege pret a serialiser.
    """
    try:
        redis = get_redis_client()
        await redis.cache_set("eva:state:trading:status", payload, ttl_seconds=30)
        latest_events = list((payload.get("decision_audit") or {}).get("recent", []) or [])
        latest_event = latest_events[-1] if latest_events else {}
        runtime_mode_value = str((payload.get("runtime") or {}).get("runtime_mode") or RuntimeMode.MAINTENANCE.value)
        try:
            runtime_mode = RuntimeMode(runtime_mode_value)
        except ValueError:
            runtime_mode = RuntimeMode.MAINTENANCE
        decision_envelope = TradingDecisionEnvelope(
            runtime_mode=runtime_mode,
            symbol=str(latest_event.get("symbol") or "GLOBAL"),
            horizon=str((latest_event.get("horizon") or (payload.get("universe") or {}).get("lab_live", {}).get("horizon") or "unknown")),
            raw_model_action=str(latest_event.get("raw_model_action") or "HOLD"),
            post_veto_action=str(latest_event.get("post_veto_action") or "HOLD"),
            selection=str(latest_event.get("selection") or "none"),
            checkpoint=str(latest_event.get("checkpoint") or "") or None,
            final_bias=str(latest_event.get("final_bias") or "NEUTRAL"),
            veto_reason=str(latest_event.get("veto_reason") or "") or None,
            connectors=dict(payload.get("connectors") or {}),
            payload=latest_event,
            metadata={"snapshot": "trading_status"},
        )
        await redis.publish("eva.trading.status", payload)
        await redis.publish("eva.trading.decision", decision_envelope.model_dump())
    except Exception as exc:
        logger.debug("Publication du statut trading ignoree: %s", exc)


async def _read_consultative_market_context(symbol: str, family: str | None) -> dict[str, Any] | None:
    """
    Lit un contexte marche consultatif structure depuis Redis.

    Args:
        symbol (str): Symbole cible.
        family (str | None): Famille d'actifs associee.

    Returns:
        dict[str, Any] | None: Snapshot de contexte si disponible.
    """
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_family = str(family or "mixed").strip().lower() or "mixed"
    if not normalized_symbol:
        return None
    try:
        redis = get_redis_client()
        payload = await redis.cache_get(
            f"eva:state:intelligence:market_context:{normalized_family}:{normalized_symbol}"
        )
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.debug("Lecture du contexte consultatif ignoree pour %s: %s", normalized_symbol, exc)
        return None


async def _cancel_background_tasks(tasks: list[asyncio.Task[Any]]) -> None:
    """
    Annule proprement les taches de fond du Banker.

    Args:
        tasks (list[asyncio.Task[Any]]): Liste des taches a stopper.
    """
    active_tasks = [task for task in tasks if task is not None and not task.done()]
    if not active_tasks:
        return

    for task in active_tasks:
        task.cancel()

    results = await asyncio.gather(*active_tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            logger.warning("Erreur lors de l'arret d'une tache de fond: %s", result)


async def _close_optional_service(service_name: str, service: Any) -> None:
    """
    Ferme un service s'il expose une methode `close` ou `disconnect`.

    Args:
        service_name (str): Nom fonctionnel du service.
        service (Any): Instance a fermer si disponible.
    """
    if service is None:
        return

    closer = getattr(service, "close", None) or getattr(service, "disconnect", None)
    if not callable(closer):
        return

    try:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:
        logger.warning("Fermeture partielle du service %s: %s", service_name, exc)


class FollowerAutoEngineStub:
    """
    Fournit un moteur minimal lorsque le banker tourne en mode follower.

    Ce stub preserve les endpoints transverses sans lancer la pile complete
    d'auto-trading, inutile pour une instance de copy trading.
    """

    def __init__(self) -> None:
        self.is_active = False
        self._cpu_live_mode = False
        self.symbols: list[str] = []
        self.latest_decisions: list[dict[str, Any]] = []

    async def start(self) -> None:
        """N'effectue aucune action en mode follower."""

    async def stop(self) -> None:
        """N'effectue aucune action en mode follower."""

    async def refresh_symbol_universe(self, force: bool = False) -> None:
        """
        Ignore tout recalcul d'univers en mode follower.

        Args:
            force (bool): Parametre conserve pour compatibilite.
        """

    def get_symbol_batch(self, advance: bool = False) -> list[str]:
        """
        Retourne une liste vide en mode follower.

        Args:
            advance (bool): Parametre conserve pour compatibilite.

        Returns:
            list[str]: Liste vide.
        """
        return []

    def get_runtime_mode_status(self) -> dict[str, Any]:
        """
        Retourne un statut runtime minimal pour les endpoints de supervision.

        Returns:
            dict[str, Any]: Statut de maintenance pour une instance follower.
        """
        return {
            "runtime_mode": RuntimeMode.MAINTENANCE.value,
            "runtime_profile": "follower",
            "shadow_learning_mode": "disabled",
            "force_maintenance": False,
        }

    def get_execution_mechanics_status(self) -> dict[str, Any]:
        """
        Retourne un statut d'execution minimal.

        Returns:
            dict[str, Any]: Vue minimale des mecanismes d'execution.
        """
        return {
            "live_family": "follower",
            "selection_policy_required": "copy_only",
            "ensemble_ready": False,
            "ensemble_active": False,
            "muzero_can_activate_live": False,
            "dreamer_can_activate_live": False,
            "active_live_engine": None,
        }

    def get_decision_audit_snapshot(self) -> dict[str, Any]:
        """
        Retourne un audit vide des decisions.

        Returns:
            dict[str, Any]: Snapshot vide.
        """
        return {
            "recent": [],
            "ensemble_decision_stats": {},
        }

    def get_live_universe_status(self) -> dict[str, Any]:
        """
        Retourne un etat d'univers live neutre.

        Returns:
            dict[str, Any]: Statut minimal de l'univers live.
        """
        return {
            "source": "follower",
            "horizon": "copy",
            "symbols": [],
        }

    def get_latest_trading_review_metadata(self) -> dict[str, Any]:
        """
        Retourne l'absence de revue live.

        Returns:
            dict[str, Any]: Metadonnees vides.
        """
        return {}

    def get_latest_trading_review(self) -> dict[str, Any]:
        """
        Retourne une reponse neutre pour les endpoints de revue.

        Returns:
            dict[str, Any]: Reponse de mode follower.
        """
        return {
            "status": "disabled",
            "reason": "follower_mode",
        }

    async def generate_trading_review(
        self,
        period_start: datetime,
        period_end: datetime,
        period_name: str,
        report_kind: str,
    ) -> dict[str, Any]:
        """
        Retourne une reponse neutre pour une revue demandee en mode follower.

        Args:
            period_start (datetime): Debut demande.
            period_end (datetime): Fin demandee.
            period_name (str): Libelle demande.
            report_kind (str): Type de rapport demande.

        Returns:
            dict[str, Any]: Reponse de neutralisation.
        """
        return {
            "status": "disabled",
            "reason": "follower_mode",
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "period_name": period_name,
            "report_kind": report_kind,
        }


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
    follower_mode = bool(getattr(settings, "banker_follower_mode", False))

    # Redis
    try:
        await init_redis()
        logger.info("âœ… Redis connectÃ©")
    except Exception as e:
        logger.warning(f"âš ï¸ Redis non disponible: {e}")

    # Services
    app.state.settings = settings
    app.state.primary_mt5_service = get_mt5_service()
    app.state.mt5_service = CopyTradingRouter(app.state.primary_mt5_service)
    await app.state.mt5_service.initialize()
    app.state.risk_validator = get_risk_validator()
    app.state.binance_service = BinanceService()
    app.state.tr_service = TradeRepublicService()
    app.state.background_tasks = []
    app.state.follower_mode = follower_mode
    
    if not follower_mode:
        # CCXT Init
        await app.state.binance_service.initialize()
        # TR Init
        await app.state.tr_service.initialize()
    else:
        logger.info("Mode follower actif: connecteurs secondaires non initialises.")

    # HiÃ©rarchie
    app.state.skill_library = SkillLibrary()
    app.state.manager = BankerManager(app.state.skill_library)
    from eva_banker.services.ghost_shield import GhostShield
    app.state.ghost_shield = GhostShield(app.state.mt5_service)
    app.state.worker = BankerWorker(app.state.mt5_service, app.state.ghost_shield)

    if follower_mode:
        app.state.auto_engine = FollowerAutoEngineStub()
    else:
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

    if follower_mode:
        app.state.swarm = None
    else:
        # IntÃ©gration SWARM (Essaim)
        app.state.swarm = BankerSwarm()
        await app.state.swarm.init_mqtt()

    # Connexion MT5
    mt5_service = app.state.mt5_service
    if await mt5_service.connect():
        logger.info("âœ… MT5 connectÃ©")
        if not follower_mode:
            # Detecter l'univers reel puis preparer le premier lot de scan.
            await app.state.auto_engine.refresh_symbol_universe(force=True)
            await mt5_service.initialize_symbols(app.state.auto_engine.get_symbol_batch(advance=False))
            
            # DÃ‰MARRAGE AU LANCEMENT (Seulement aprÃ¨s connexion rÃ©ussie)
            await app.state.auto_engine.start()
            logger.info("ðŸš€ Auto-Trading Engine Started")
        else:
            logger.info("Mode follower actif: auto-trading et univers live desactives.")
    else:
        logger.error(
            "MT5 indisponible: auto-trading non demarre et aucun repli mock n'est autorise en mode reel."
        )

    app.state.background_tasks = [
        asyncio.create_task(hard_heartbeat(), name="banker_hard_heartbeat"),
    ]
    if not follower_mode:
        app.state.background_tasks.insert(
            0,
            asyncio.create_task(swarm_listener(), name="banker_swarm_listener"),
        )
        app.state.background_tasks.append(
            asyncio.create_task(
                app.state.news_filter.start_monitoring(),
                name="banker_news_filter",
            ),
        )

    logger.info("âœ… The Banker (SWARM MODE) READY")

    yield

    # ArrÃªt (Shutdown)
    logger.info("ðŸ›‘ ArrÃªt The Banker...")
    if hasattr(app.state, "news_filter"):
        app.state.news_filter.stop()
    if hasattr(app.state, 'auto_engine'):
        await app.state.auto_engine.stop()
    await _cancel_background_tasks(getattr(app.state, "background_tasks", []))
    await _close_optional_service("binance", getattr(app.state, "binance_service", None))
    await _close_optional_service(
        "mqtt",
        getattr(getattr(app.state, "swarm", None), "mqtt", None),
    )
    with suppress(Exception):
        await get_redis_client().disconnect()
    await mt5_service.disconnect()


async def hard_heartbeat():
    """
    Signal haute fr?quence pour le Watchdog Rust (Loi 0) et l'Orchestrateur Core.

    Persiste l'?tat dans Redis pour la d?couverte des agents.
    Inclut d?sormais l'?quit? pour le Kill-Switch financier.
    """
    from shared.redis_client import get_redis_client

    redis = get_redis_client()
    mt5_service = app.state.mt5_service
    heartbeat_interval = max(
        1.0,
        float(getattr(app.state.settings, "banker_heartbeat_interval_seconds", 3.0)),
    )
    redis_unavailable = False

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
                    "currency": account.currency,
                }
                await redis.publish("eva.banker.heartbeat", payload)
                await redis.cache_set("eva.banker.status", payload, ttl_seconds=10)
                if redis_unavailable:
                    logger.info("Heartbeat Redis r?tabli.")
                    redis_unavailable = False
        except Exception as e:
            if not redis_unavailable:
                logger.warning(f"Heartbeat Redis indisponible: {e}")
                redis_unavailable = True
            try:
                await redis.disconnect()
            except Exception:
                pass
            await asyncio.sleep(min(heartbeat_interval, 2.0))
            continue

        await asyncio.sleep(heartbeat_interval)


async def swarm_listener():
    """
    ?coute les commandes broadcast de l'essaim.

    Cette t?che de fond permet au Banker de r?agir aux ordres globaux
    ou de d?ployer des drones de surveillance.
    """
    from shared.redis_client import get_redis_client

    swarm: BankerSwarm = app.state.swarm

    async def handle_swarm(channel, message):
        action = message.get("action")
        command = message.get("command")

        if command == "GLOBAL_STOP":
            logger.critical(f"KILL-SWITCH recu: {message.get('reason')}")
            if hasattr(app.state, 'auto_engine') and app.state.auto_engine.is_active:
                await app.state.auto_engine.stop()

            if hasattr(app.state.auto_engine, 'telegram'):
                app.state.auto_engine.telegram.send_sync(
                    f"URGENCE: KILL-SWITCH DECLENCHE\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Par: {message.get('issuer', 'Unknown')}\n"
                    f"Raison: {message.get('reason')}\n"
                    f"Le Bot E.V.A est totalement HALTE."
                )

        elif action == "SWARM_SURVEILLANCE":
            await swarm.spawn_drone(
                name="GoldSurveillance",
                mission="Surveiller XAUUSD avec le Swarm",
                coro=swarm.run_gold_surveillance(Decimal("2050.0"))
            )

    redis_unavailable = False

    while True:
        redis = get_redis_client()
        try:
            await redis.subscribe(["eva.all.swarm_command", "eva.banker.swarm_command"], handle_swarm)
            if redis_unavailable:
                logger.info("?coute SWARM Redis r?tablie.")
                redis_unavailable = False
            await redis.listen()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not redis_unavailable:
                logger.warning(f"SWARM Redis indisponible: {e}")
                redis_unavailable = True
            try:
                await redis.disconnect()
            except Exception:
                pass
            await asyncio.sleep(5.0)


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
        status="degraded" if _is_mt5_live_offline(mt5_service) else "ok",
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
        if _is_mt5_live_offline(app.state.mt5_service):
            raise HTTPException(
                status_code=503,
                detail="MT5 hors ligne: impossible d'activer l'auto-trading en mode reel.",
            )
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

    if request.source == OrderSource.COPY:
        skill = "COPY_ROUTER"
    else:
        # 2. Le Manager definit la strategie (Skill)
        manager: BankerManager = app.state.manager
        # Simulation de donnees de marche pour le manager (incluant VaR)
        market_data = {"price": 2034.50, "returns": [0.001, -0.002, 0.005]}
        skill = manager.plan_strategy(market_data)

        # 3. Verification de la sincerite cognitive
        # Le filtre reste reserve aux ordres decides localement.
        import torch

        mock_activations = torch.randn(1, 4096)
        is_sincere, sincerity_msg = check_cognitive_sincerity(
            mock_activations,
            "The market shows a strong bullish trend on H4.",
            request.action
        )

        if not is_sincere:
            logger.warning("Ordre bloque par sincerite cognitive: %s", sincerity_msg)
            return OrderResponse(
                success=False,
                message=sincerity_msg,
                risk_check={"allowed": False, "reason": "COGNITIVE_SINCERITY_FAILURE"},
            )

    # 4. Conversion en TradeOrder
    order = TradeOrder(
        symbol=request.symbol,
        action=request.action,
        volume=request.volume,
        stop_loss_price=request.stop_loss,
        take_profit_price=request.take_profit,
        account_id=request.account_id,
        comment=request.comment or f"Skill: {skill}",
        source=request.source,
    )

    # 5. Verification des risques (Loi 2)
    risk_validator: RiskValidator = app.state.risk_validator
    mt5_service: MT5Service = app.state.mt5_service

    # La validation doit utiliser l'etat reel du compte pour eviter les faux
    # rejets de copy trading apres un redemarrage local du follower.
    account, positions = await _read_mt5_snapshot(mt5_service)
    if account is not None:
        equity_or_balance = getattr(account, "equity", None) or getattr(account, "balance", None)
        if equity_or_balance is not None:
            risk_validator.update_account_balance(Decimal(str(equity_or_balance)))
    risk_validator.update_positions_count(len(positions))

    # Le calcul de risque doit s'appuyer sur le dernier tick MT5 plutot que sur
    # un prix mock statique, sinon le follower surestime les copies valides.
    if order.entry_price is None:
        tick_payload = await mt5_service.get_symbol_tick(order.symbol)
        if isinstance(tick_payload, dict) and tick_payload.get("success", True):
            live_price = (
                tick_payload.get("ask")
                if order.action == TradeAction.BUY
                else tick_payload.get("bid")
            )
            if live_price is not None and float(live_price) > 0.0:
                order.entry_price = Decimal(str(live_price))

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
        message=result.get("message", f"Execution traitee via {skill}"),
        risk_check=risk_result,
        copy_results=result.get("copy_results", []),
    )


@app.get("/positions", response_model=list[Position], tags=["Trading"])
async def get_positions() -> list[Position]:
    """
    RÃ©cupÃ¨re la liste des positions actuellement ouvertes sur MT5.

    Returns:
        list[Position]: Liste des positions avec P&L latent, Swap et Ticket.
    """
    mt5_service: MT5Service = app.state.mt5_service
    return await mt5_service.get_open_positions() or []


@app.delete("/positions/{ticket}", tags=["Trading"])
async def close_position(
    ticket: int,
    volume: Decimal | None = Query(default=None, gt=0),
) -> dict[str, Any]:
    """
    Ferme une position spÃ©cifique via son ticket MT5.

    Args:
        ticket (int): Identifiant unique MT5 de la position Ã  fermer.

    Returns:
        dict[str, Any]: RÃ©sultat de la clÃ´ture (SuccÃ¨s, Prix de clÃ´ture, Profit rÃ©alisÃ©).
    """
    mt5_service: MT5Service = app.state.mt5_service
    result = await mt5_service.close_position(ticket, volume=volume)

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


@app.post("/positions/{ticket}/modify", tags=["Trading"])
async def modify_position(ticket: int, request: PositionModifyRequest) -> dict[str, Any]:
    """
    Modifie les niveaux de protection d'une position ouverte.

    Args:
        ticket (int): Ticket MT5 de la position a modifier.
        request (PositionModifyRequest): Nouveaux niveaux SL/TP demandes.

    Returns:
        dict[str, Any]: Resultat local et, si actif, details de propagation.
    """
    mt5_service = app.state.mt5_service
    stop_loss = float(request.stop_loss) if request.stop_loss is not None else 0.0
    take_profit = float(request.take_profit) if request.take_profit is not None else 0.0
    return await mt5_service.modify_position(ticket, sl=stop_loss, tp=take_profit)


@app.get("/account", response_model=AccountBalance, tags=["Compte"])
async def get_account_balance() -> AccountBalance:
    """
    RÃ©cupÃ¨re les informations financiÃ¨res du compte de trading (Equity, Balance, Marge).

    Returns:
        AccountBalance: DonnÃ©es financiÃ¨res temps rÃ©el.
    """
    mt5_service: MT5Service = app.state.mt5_service
    account = await mt5_service.get_account_info()
    if account is None:
        raise HTTPException(
            status_code=503,
            detail="MT5 hors ligne: informations de compte indisponibles.",
        )
    return account


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
    mt5_service: MT5Service = app.state.mt5_service
    risk_validator: RiskValidator = app.state.risk_validator
    account, positions = await _read_mt5_snapshot(mt5_service)
    if account is not None:
        risk_validator.update_account_balance(Decimal(str(account.balance)))
    risk_validator.update_positions_count(len(positions))
    risk = await risk_validator.get_current_status()
    if account is None:
        return risk.model_copy(update={"trading_allowed": False})
    return risk


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
    positions = await mt5_service.get_open_positions() or []

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

    account, positions = await _read_mt5_snapshot(mt5_service)
    if account is not None:
        risk_validator.update_account_balance(Decimal(str(account.balance)))
    risk_validator.update_positions_count(len(positions))
    risk = await risk_validator.get_current_status()
    if account is None:
        risk = risk.model_copy(update={"trading_allowed": False})
    runtime_status = app.state.auto_engine.get_runtime_mode_status()
    execution_mechanics = app.state.auto_engine.get_execution_mechanics_status()
    decision_audit = app.state.auto_engine.get_decision_audit_snapshot()
    live_universe_status = app.state.auto_engine.get_live_universe_status()
    latest_review = app.state.auto_engine.get_latest_trading_review_metadata()
    nemesis_status = app.state.nemesis.get_status()
    connectors = _build_connector_status(app)
    live_family = str(
        execution_mechanics.get("live_family")
        or live_universe_status.get("live_family")
        or "mixed"
    ).strip().lower() or "mixed"
    tracked_symbols: list[str] = []
    for symbol in list(execution_mechanics.get("live_top_symbols") or []):
        normalized_symbol = str(symbol).strip().upper()
        if normalized_symbol and normalized_symbol not in tracked_symbols:
            tracked_symbols.append(normalized_symbol)
    for position in positions:
        normalized_symbol = str(position.symbol).strip().upper()
        if normalized_symbol and normalized_symbol not in tracked_symbols:
            tracked_symbols.append(normalized_symbol)
    market_context: dict[str, Any] = {}
    for symbol in tracked_symbols[:8]:
        payload = await _read_consultative_market_context(symbol, live_family)
        if payload:
            market_context[symbol] = payload
    event_blockers = {
        symbol: context
        for symbol, context in market_context.items()
        if bool(context.get("blocked", False))
    }
    risk_overrides = {
        symbol: {
            "blocked": bool(context.get("blocked", False)),
            "event_risk": context.get("event_risk"),
            "geo_risk": context.get("geo_risk"),
            "macro_bias": context.get("macro_bias"),
            "confidence": context.get("confidence"),
        }
        for symbol, context in market_context.items()
    }

    payload = {
        "status": "offline" if account is None else "online",
        "runtime_profile": runtime_status.get("runtime_profile"),
        "shadow_learning_mode": runtime_status.get("shadow_learning_mode", "shadow_only"),
        "runtime": runtime_status,
        "connection": {
            "mt5_connected": mt5_service.is_connected,
            "mock_mode": mt5_service.mock_mode,
        },
        "copy_trading": {
            "targets": getattr(app.state.mt5_service, "get_targets_status", lambda: [])(),
        },
        "topology": {
            "execution_authority": "banker_local_mt5",
            "server_role": "modeles_supervision_memoire",
            "runtime_mode": _derive_runtime_mode(app.state.auto_engine).value,
            "runtime_profile": runtime_status.get("runtime_profile"),
        },
        "connectors": connectors,
        "vllm": connectors.get("vllm", {}),
        "gnn": connectors.get("gnn", {}),
        "nemesis": nemesis_status,
        "account": {
            "equity": float(account.equity) if account is not None else 0.0,
            "balance": float(account.balance) if account is not None else 0.0,
            "margin": float(account.margin) if account is not None else 0.0,
            "free_margin": float(account.free_margin) if account is not None else 0.0,
            "currency": account.currency if account is not None else "USD",
            "leverage": account.leverage if account is not None else 0,
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
        "execution_mechanics": execution_mechanics,
        "decisions": app.state.auto_engine.latest_decisions,
        "decision_audit": decision_audit,
        "ensemble_decision_stats": decision_audit.get("ensemble_decision_stats", {}),
        "live_family": execution_mechanics.get("live_family"),
        "live_champion_id_muzero": execution_mechanics.get("live_champion_id_muzero"),
        "live_champion_id_dreamer": execution_mechanics.get("live_champion_id_dreamer"),
        "active_live_engine": execution_mechanics.get("active_live_engine"),
        "registered_live_champion_muzero": execution_mechanics.get("registered_live_champion_muzero"),
        "registered_live_champion_dreamer": execution_mechanics.get("registered_live_champion_dreamer"),
        "muzero_promotion_state": execution_mechanics.get("muzero_promotion_state"),
        "dreamer_promotion_state": execution_mechanics.get("dreamer_promotion_state"),
        "muzero_can_activate_live": bool(execution_mechanics.get("muzero_can_activate_live", False)),
        "dreamer_can_activate_live": bool(execution_mechanics.get("dreamer_can_activate_live", False)),
        "dreamer_live_enabled": bool(execution_mechanics.get("dreamer_live_enabled", False)),
        "ensemble_ready": bool(execution_mechanics.get("ensemble_ready", False)),
        "ensemble_active": bool(execution_mechanics.get("ensemble_active", False)),
        "force_maintenance": bool(runtime_status.get("force_maintenance", False)),
        "research_mode": "consultatif",
        "latest_review": latest_review,
        "live_data_source": live_universe_status.get("source") or execution_mechanics.get("selection_policy_required"),
        "market_context": market_context,
        "event_blockers": event_blockers,
        "risk_overrides": risk_overrides,
        "degraded_fallback_reason": (
            ((decision_audit.get("recent") or [{}])[-1]).get("degraded_fallback_reason")
            if decision_audit.get("recent")
            else None
        ),
        "universe": {
            "dynamic": getattr(app.state.auto_engine, "_dynamic_universe_enabled", False),
            "symbols_total": len(app.state.auto_engine.symbols),
            "batch_size": len(app.state.auto_engine.get_symbol_batch(advance=False)),
            "lab_live": live_universe_status,
        }
    }
    await _publish_trading_status_snapshot(payload)
    return payload


@app.get("/trading/review/latest", tags=["Trading"])
async def get_latest_trading_review():
    """
    Retourne la derniere revue de trading persistee.

    Returns:
        dict[str, Any]: Rapport complet si disponible, sinon statut d'absence.
    """

    return app.state.auto_engine.get_latest_trading_review()


@app.post("/trading/review/generate", tags=["Trading"])
async def generate_trading_review(
    hours: int = Query(default=12, ge=1, le=72),
    period_name: str | None = Query(default=None),
):
    """
    Genere une revue de trading structuree sur une fenetre glissante.

    Args:
        hours (int): Taille de la fenetre analysee en heures.
        period_name (str | None): Libelle optionnel du rapport.

    Returns:
        dict[str, Any]: Rapport persiste et chemins de stockage.
    """

    period_end = datetime.now()
    period_start = period_end - timedelta(hours=hours)
    review_label = str(period_name or f"Fenetre {hours}h").strip() or f"Fenetre {hours}h"
    review = await app.state.auto_engine.generate_trading_review(
        period_start=period_start,
        period_end=period_end,
        period_name=review_label,
        report_kind="manual",
    )
    return review



@app.get("/performance/models", tags=["Trading"])
async def get_model_performance(
    days: int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=5, ge=1, le=20),
):
    """
    Retourne le PnL realise par moteur de decision sur une fenetre glissante.

    Args:
        days (int): Nombre de jours a analyser.
        limit (int): Nombre maximal de strategies retournees par classement.

    Returns:
        dict[str, Any]: Resume global, details par modele et derniers trades clotures.
    """
    mt5_service: MT5Service = app.state.mt5_service
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days)
    performance = await mt5_service.get_strategy_performance(
        from_dt=from_dt,
        to_dt=to_dt,
        limit=limit,
    )
    summary = dict(performance.get("summary") or {})
    summary.setdefault("realized_pnl", summary.get("net_profit", 0.0))
    summary.setdefault("window_label", f"{days}j")
    performance["summary"] = summary
    return {
        "status": "ok",
        "window_days": days,
        **performance,
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
    mt5_service = app.state.mt5_service
    account = await mt5_service.get_account_info()
    local_accounts = []
    if account is not None:
        local_accounts.append(
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
                "copy_role": "master",
            }
        )

    remote_accounts = []
    target_status_getter = getattr(mt5_service, "get_targets_status", None)
    if callable(target_status_getter):
        for target in target_status_getter():
            remote_accounts.append(
                {
                    "id": target["id"],
                    "name": target["name"],
                    "server": target.get("server"),
                    "balance": None,
                    "equity": None,
                    "phase": target.get("phase", "funded"),
                    "status": "active" if target.get("enabled", False) else "disabled",
                    "max_drawdown": None,
                    "daily_drawdown": None,
                    "copy_role": "follower",
                    "banker_base_url": target.get("banker_base_url"),
                    "allocation_ratio": target.get("allocation_ratio"),
                    "broker": target.get("broker"),
                    "terminal_label": target.get("terminal_label"),
                }
            )

    return local_accounts + remote_accounts


@app.get("/copy-trading/status", tags=["Trading"])
async def get_copy_trading_status() -> dict[str, Any]:
    """
    Retourne l'etat du routage de copy trading multi-instances.

    Returns:
        dict[str, Any]: Liste des cibles configurees et activation du module.
    """
    mt5_service = app.state.mt5_service
    target_status_getter = getattr(mt5_service, "get_targets_status", None)
    targets = target_status_getter() if callable(target_status_getter) else []
    return {
        "enabled": bool(targets),
        "targets": targets,
    }


@app.get("/symbols/discover", tags=["Trading"])
async def discover_available_symbols(
    include_forex: bool = Query(default=True),
    include_cfd: bool = Query(default=True),
    include_crypto: bool = Query(default=True),
    max_symbols: int = Query(default=0, ge=0, le=5000),
) -> dict[str, Any]:
    """
    Expose l'univers de symboles tradables vu par une instance Banker.

    Cet endpoint sert principalement au copy trading inter-brokers pour
    traduire les symboles quand deux terminaux n'emploient pas la meme
    nomenclature.

    Args:
        include_forex (bool): Inclut les paires Forex si True.
        include_cfd (bool): Inclut les CFD si True.
        include_crypto (bool): Inclut les cryptos si True.
        max_symbols (int): Limite optionnelle du nombre de symboles.

    Returns:
        dict[str, Any]: Liste des symboles visibles et options d'appel.
    """
    mt5_service: MT5Service = app.state.mt5_service
    symbols = await mt5_service.discover_symbols(
        include_forex=include_forex,
        include_cfd=include_cfd,
        include_crypto=include_crypto,
        max_symbols=max_symbols,
    )
    return {
        "status": "ok",
        "count": len(symbols),
        "symbols": symbols,
        "filters": {
            "include_forex": include_forex,
            "include_cfd": include_cfd,
            "include_crypto": include_crypto,
            "max_symbols": max_symbols,
        },
    }
