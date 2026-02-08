"""
Service MT5 - Client MetaTrader 5
Gère la connexion et l'exécution des ordres sur MT5
"""

import logging
import sys
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from shared import AccountBalance, Position, TradeAction, TradeOrder, get_settings

logger = logging.getLogger(__name__)

# MT5 ne fonctionne que sur Windows
MT5_AVAILABLE = sys.platform == "win32"

if MT5_AVAILABLE:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        MT5_AVAILABLE = False
        logger.warning("MetaTrader5 non installé")


class MT5Service:
    """
    Client MetaTrader 5 pour exécution des ordres.
    
    Supporte:
    - Mode réel (Windows avec MT5 installé)
    - Mode mock (développement / paper trading)
    """

    def __init__(self, mock_mode: bool = True, login: int = 0, password: str = "", server: str = ""):
        self.mock_mode = mock_mode or not MT5_AVAILABLE
        self.is_connected = False
        self._mock_positions: list[Position] = []
        self._mock_balance = Decimal("100000.00")
        self._next_ticket = 12345678
        # Credentials pour login automatique
        self._login = login
        self._password = password
        self._server = server
        logger.info(f"MT5Service initialise (mock={self.mock_mode}, login={login}, server={server})")

    async def connect(self) -> bool:
        """Connexion a MT5"""
        if self.mock_mode:
            self.is_connected = True
            logger.info("MT5 Mock: connecte")
            return True

        try:
            # Initialisation MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

            # Verifier si le terminal est deja connecte au bon compte
            account_info = mt5.account_info()
            if account_info and account_info.login == self._login:
                logger.info(f"MT5 deja connecte: compte {account_info.login} sur {account_info.server} "
                           f"(Balance: {account_info.balance}, Equity: {account_info.equity})")
            elif self._login and self._password and self._server:
                # Login automatique si credentials fournis et pas encore connecte
                authorized = mt5.login(
                    login=self._login,
                    password=self._password,
                    server=self._server
                )
                if authorized:
                    account_info = mt5.account_info()
                    logger.info(f"MT5 login reussi: compte {self._login} sur {self._server}")
                else:
                    # Le terminal est peut-etre deja connecte mais login() echoue
                    account_info = mt5.account_info()
                    if account_info:
                        logger.info(f"MT5 terminal deja actif: compte {account_info.login} sur {account_info.server} "
                                   f"(Balance: {account_info.balance})")
                    else:
                        logger.error(f"MT5 login echoue pour {self._login}@{self._server}: {mt5.last_error()}")
                        mt5.shutdown()
                        return False
            elif account_info:
                logger.info(f"MT5 connecte: compte {account_info.login} sur {account_info.server}")
            else:
                logger.warning("MT5 initialise mais aucun compte connecte")

            self.is_connected = True
            return True
        except Exception as e:
            logger.exception(f"Erreur connexion MT5: {e}")
            return False

    async def disconnect(self) -> None:
        """Déconnexion de MT5"""
        if not self.mock_mode and MT5_AVAILABLE:
            mt5.shutdown()
        self.is_connected = False
        logger.info("MT5 déconnecté")

    async def get_account_info(self) -> AccountBalance:
        """Récupère les informations du compte"""
        if self.mock_mode:
            return AccountBalance(
                login=12345678,
                server="Mock-Server",
                balance=self._mock_balance,
                equity=self._mock_balance + self._get_mock_pnl(),
                margin=Decimal("0"),
                free_margin=self._mock_balance,
                leverage=100,
            )

        info = mt5.account_info()
        if info is None:
            raise RuntimeError("Impossible de récupérer les infos du compte")

        return AccountBalance(
            login=info.login,
            server=info.server,
            balance=Decimal(str(info.balance)),
            equity=Decimal(str(info.equity)),
            margin=Decimal(str(info.margin)),
            free_margin=Decimal(str(info.margin_free)),
            margin_level=info.margin_level,
            currency=info.currency,
            leverage=info.leverage,
        )

    async def get_open_positions(self) -> list[Position]:
        """Récupère les positions ouvertes"""
        if self.mock_mode:
            return self._mock_positions

        positions_data = mt5.positions_get()
        if positions_data is None:
            return []

        positions = []
        for pos in positions_data:
            positions.append(
                Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    action=TradeAction.BUY if pos.type == 0 else TradeAction.SELL,
                    volume=Decimal(str(pos.volume)),
                    open_price=Decimal(str(pos.price_open)),
                    current_price=Decimal(str(pos.price_current)),
                    stop_loss=Decimal(str(pos.sl)) if pos.sl > 0 else None,
                    take_profit=Decimal(str(pos.tp)) if pos.tp > 0 else None,
                    profit=Decimal(str(pos.profit)),
                    swap=Decimal(str(pos.swap)),
                    commission=Decimal(str(pos.commission)),
                    magic_number=pos.magic,
                    open_time=datetime.fromtimestamp(pos.time),
                )
            )
        return positions

    async def execute_order(self, order: TradeOrder) -> dict[str, Any]:
        """Exécute un ordre de trading"""
        if self.mock_mode:
            return await self._execute_mock_order(order)

        # Préparation de la requête MT5
        symbol_info = mt5.symbol_info(order.symbol)
        if symbol_info is None:
            return {"success": False, "message": f"Symbole {order.symbol} non trouvé"}

        price = mt5.symbol_info_tick(order.symbol)
        if price is None:
            return {"success": False, "message": "Prix non disponible"}

        order_type = mt5.ORDER_TYPE_BUY if order.action == TradeAction.BUY else mt5.ORDER_TYPE_SELL
        exec_price = price.ask if order.action == TradeAction.BUY else price.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": float(order.volume),
            "type": order_type,
            "price": exec_price,
            "sl": float(order.stop_loss_price) if order.stop_loss_price else 0.0,
            "tp": float(order.take_profit_price) if order.take_profit_price else 0.0,
            "deviation": 10,
            "magic": order.magic_number,
            "comment": order.comment or "EVA Banker",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "message": f"Erreur MT5: {result.comment}",
                "retcode": result.retcode,
            }

        return {
            "success": True,
            "ticket": result.order,
            "message": f"Ordre exécuté: {order.action.value} {order.volume} {order.symbol}",
        }

    async def close_position(self, ticket: int) -> dict[str, Any]:
        """Ferme une position par son ticket"""
        if self.mock_mode:
            self._mock_positions = [p for p in self._mock_positions if p.ticket != ticket]
            return {"success": True, "message": f"Position {ticket} fermée (mock)"}

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "message": f"Position {ticket} non trouvée"}

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol)
        close_price = price.bid if pos.type == 0 else price.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": 10,
            "magic": pos.magic,
            "comment": "EVA Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "message": f"Erreur fermeture: {result.comment}"}

        return {"success": True, "ticket": ticket, "message": "Position fermée"}

    async def _execute_mock_order(self, order: TradeOrder) -> dict[str, Any]:
        """Exécute un ordre en mode mock"""
        ticket = self._next_ticket
        self._next_ticket += 1

        # Prix simulé
        mock_prices = {
            "XAUUSD": Decimal("2080.00"),
            "EURUSD": Decimal("1.0850"),
            "GBPUSD": Decimal("1.2650"),
            "USDJPY": Decimal("150.50"),
        }
        price = mock_prices.get(order.symbol, Decimal("100.00"))

        position = Position(
            ticket=ticket,
            symbol=order.symbol,
            action=order.action,
            volume=order.volume,
            open_price=price,
            current_price=price,
            stop_loss=order.stop_loss_price,
            take_profit=order.take_profit_price,
            profit=Decimal("0"),
            magic_number=order.magic_number,
            open_time=datetime.now(),
        )
        self._mock_positions.append(position)

        logger.info(f"📊 Mock Order: {order.action.value} {order.volume} {order.symbol} @ {price}")

        return {
            "success": True,
            "ticket": ticket,
            "message": f"[MOCK] {order.action.value} {order.volume} {order.symbol}",
        }

    async def execute_skill(self, skill, order: TradeOrder) -> dict[str, Any]:
        """
        Execute un ordre en utilisant une compétence (Skill) spécifique.

        Dispatche l'exécution en fonction du type de skill sélectionné
        par le Manager (niveau haut de l'architecture hiérarchique SPlaTES).
        En mode lite, délègue simplement à execute_order().

        Args:
            skill: Le type de compétence à utiliser (SkilledBehavior enum).
            order: L'ordre de trading à exécuter.

        Returns:
            dict[str, Any]: Résultat de l'exécution avec 'success', 'ticket', 'message'.
        """
        logger.info(f"Exécution via skill: {skill} pour {order.symbol}")
        # En production, chaque skill aurait sa propre logique d'exécution
        # (timing, slicing, obfuscation, etc.)
        # En mode lite, on délègue directement à execute_order()
        result = await self.execute_order(order)
        result["skill_used"] = str(skill)
        return result

    def _get_mock_pnl(self) -> Decimal:
        """Calcule le P&L mock total"""
        return sum(p.profit for p in self._mock_positions)


@lru_cache
def get_mt5_service() -> MT5Service:
    """Retourne l'instance MT5 configuree avec credentials FTMO"""
    settings = get_settings()
    return MT5Service(
        mock_mode=settings.mock_mt5,
        login=settings.mt5_login,
        password=settings.mt5_password.get_secret_value(),
        server=settings.mt5_server,
    )
