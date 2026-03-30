"""
Executeur Hydra dedie a un terminal MT5 unique.

Cette API legere est concue pour etre lancee une fois par compte esclave,
souvent sous Wine. Chaque instance pilote un seul terminal MT5 et expose
des routes minimales pour la copie master/slave.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eva_banker.services.mt5 import MT5Service
from shared import HydraTerminalHealth, OrderSource, TradeAction, TradeOrder, get_settings

logger = logging.getLogger(__name__)


def _resolve_account_uuid() -> UUID:
    """
    Resol l'identifiant Hydra du terminal depuis l'environnement.

    Returns:
        UUID: Identifiant stable du compte terminal.
    """
    raw_value = os.getenv("HYDRA_ACCOUNT_UUID", "00000000-0000-0000-0000-000000000000")
    try:
        return UUID(str(raw_value))
    except ValueError:
        logger.warning("Hydra Terminal: HYDRA_ACCOUNT_UUID invalide, UUID nul utilise.")
        return UUID("00000000-0000-0000-0000-000000000000")


class HydraTerminalOrderRequest(BaseModel):
    """
    Charge utile de copie pour un terminal Hydra unique.
    """

    symbol: str
    action: TradeAction
    volume: Decimal = Field(..., gt=0)
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    entry_price: Decimal | None = None
    master_ticket: int | None = None
    comment: str | None = None


class HydraTerminalCloseRequest(BaseModel):
    """
    Demande de cloture d'une position sur un terminal Hydra.
    """

    source_ticket: int
    symbol: str
    action: TradeAction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gere la connexion MT5 de l'executeur Hydra.
    """

    settings = get_settings()
    app.state.settings = settings
    app.state.mt5_service = MT5Service(
        mock_mode=settings.mock_mt5,
        login=settings.mt5_login,
        password=settings.mt5_password.get_secret_value(),
        server=settings.mt5_server,
    )
    await app.state.mt5_service.connect()
    yield
    await app.state.mt5_service.disconnect()


app = FastAPI(
    title="Hydra Terminal Executor",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def get_health() -> dict[str, Any]:
    """
    Retourne la sante minimum du terminal Hydra.
    """

    mt5_service: MT5Service = app.state.mt5_service
    account = await mt5_service.get_account_info()
    return {
        "status": "ok" if mt5_service.is_connected else "degraded",
        "process_alive": True,
        "mt5_connected": mt5_service.is_connected,
        "autotrading_enabled": True,
        "login": account.login if account else None,
        "server": account.server if account else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/hydra/terminal/health", response_model=HydraTerminalHealth)
async def get_hydra_terminal_health() -> HydraTerminalHealth:
    """
    Retourne un snapshot detaille du terminal MT5.
    """

    mt5_service: MT5Service = app.state.mt5_service
    account = await mt5_service.get_account_info()
    account_uuid = _resolve_account_uuid()
    if account is None:
        return HydraTerminalHealth(
            account_id=account_uuid,
            process_alive=True,
            mt5_connected=False,
            autotrading_enabled=False,
            terminal_path=os.getenv("HYDRA_TERMINAL_PATH"),
            wineprefix=os.getenv("WINEPREFIX"),
        )
    return HydraTerminalHealth(
        account_id=account_uuid,
        process_alive=True,
        mt5_connected=mt5_service.is_connected,
        autotrading_enabled=True,
        terminal_path=os.getenv("HYDRA_TERMINAL_PATH"),
        wineprefix=os.getenv("WINEPREFIX"),
    )


@app.post("/hydra/terminal/order")
async def execute_hydra_order(request: HydraTerminalOrderRequest) -> dict[str, Any]:
    """
    Execute un ordre copie sur le terminal local.
    """

    mt5_service: MT5Service = app.state.mt5_service
    order = TradeOrder(
        symbol=request.symbol,
        action=request.action,
        volume=request.volume,
        entry_price=request.entry_price,
        stop_loss_price=request.stop_loss_price,
        take_profit_price=request.take_profit_price,
        source=OrderSource.COPY,
        comment=request.comment or f"Hydra copy {request.master_ticket or 'na'}",
    )
    result = await mt5_service.execute_order(order)
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("message") or "Echec de copie.")
    return result


@app.post("/hydra/terminal/close")
async def close_hydra_position(request: HydraTerminalCloseRequest) -> dict[str, Any]:
    """
    Ferme une position sur le terminal Hydra.

    La V1 cloture le premier ticket local matching le symbole et le sens
    inverse attendu, car le maitre et le slave n'ont pas encore de mapping
    durable ticket->ticket.
    """

    mt5_service: MT5Service = app.state.mt5_service
    positions = await mt5_service.get_open_positions() or []
    target = None
    for position in positions:
        if str(position.symbol).strip().upper() != str(request.symbol).strip().upper():
            continue
        target = position
        break
    if target is None:
        raise HTTPException(status_code=404, detail="Aucune position cible n'a ete trouvee pour la cloture.")
    result = await mt5_service.close_position(target.ticket)
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("message") or "Echec de cloture.")
    return result


@app.get("/hydra/terminal/metrics")
async def get_hydra_metrics() -> dict[str, Any]:
    """
    Retourne un resume rapide du compte local pilote par l'executeur.
    """

    mt5_service: MT5Service = app.state.mt5_service
    account = await mt5_service.get_account_info()
    positions = await mt5_service.get_open_positions() or []
    return {
        "status": "ok" if mt5_service.is_connected else "degraded",
        "account": account.model_dump(mode="json") if account else None,
        "open_positions": len(positions),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "eva_banker.hydra_terminal_main:app",
        host=os.getenv("HYDRA_TERMINAL_HOST", "0.0.0.0"),
        port=int(os.getenv("HYDRA_TERMINAL_PORT", "19100")),
        reload=False,
    )
