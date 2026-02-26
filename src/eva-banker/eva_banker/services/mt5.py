"""
Service MT5 - Client MetaTrader 5
Gère la connexion et l'exécution des ordres sur MT5
"""

import asyncio
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
            if not await asyncio.to_thread(mt5.initialize):
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

            # Verifier si le terminal est deja connecte au bon compte
            account_info = await asyncio.to_thread(mt5.account_info)
            if account_info and account_info.login == self._login:
                logger.info(f"MT5 deja connecte: compte {account_info.login} sur {account_info.server} "
                           f"(Balance: {account_info.balance}, Equity: {account_info.equity})")
            elif self._login and self._password and self._server:
                # Login automatique si credentials fournis et pas encore connecte
                authorized = await asyncio.to_thread(
                    mt5.login,
                    login=self._login,
                    password=self._password,
                    server=self._server
                )
                if authorized:
                    account_info = await asyncio.to_thread(mt5.account_info)
                    logger.info(f"MT5 login reussi: compte {self._login} sur {self._server}")
                else:
                    # Le terminal est peut-etre deja connecte mais login() echoue
                    account_info = await asyncio.to_thread(mt5.account_info)
                    if account_info:
                        logger.info(f"MT5 terminal deja actif: compte {account_info.login} sur {account_info.server} "
                                   f"(Balance: {account_info.balance})")
                    else:
                        logger.error(f"MT5 login echoue pour {self._login}@{self._server}: {mt5.last_error()}")
                        await asyncio.to_thread(mt5.shutdown)
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

    async def initialize_symbols(self, symbols: list[str]) -> None:
        """S'assure que les symboles sont sélectionnés dans le Market Watch"""
        if self.mock_mode:
            return
        
        for symbol in symbols:
            selected = await asyncio.to_thread(mt5.symbol_select, symbol, True)
            if not selected:
                logger.warning(f"Impossible de sélectionner le symbole {symbol}: {mt5.last_error()}")
            else:
                logger.info(f"Symbole {symbol} sélectionné avec succès.")

    async def disconnect(self) -> None:
        """Déconnexion de MT5"""
        if not self.mock_mode and MT5_AVAILABLE:
            await asyncio.to_thread(mt5.shutdown)
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

        info = await asyncio.to_thread(mt5.account_info)
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

        positions_data = await asyncio.to_thread(mt5.positions_get)
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
                    swap=Decimal(str(getattr(pos, "swap", 0.0))),
                    commission=Decimal(str(getattr(pos, "commission", 0.0))),
                    magic_number=pos.magic,
                    open_time=datetime.fromtimestamp(pos.time),
                )
            )
        return positions

    async def get_recent_candles(self, symbol: str, timeframe: int = 1, count: int = 20) -> list[dict]:
        """Récupère les dernières bougies (M1 par défaut)"""
        # Mapping timeframe int -> MT5 constant if needed, but assuming M1=1 for now implies simple mapping logic or direct use if caller passes constant.
        # Actually, let's just default to M1 (1 minute) if 1 is passed.
        # MT5 constants: TIMEFRAME_M1 = 1, etc.
        
        if self.mock_mode:
            # Generate fake candles
            import random
            candles = []
            base_price = 2080.0
            for i in range(count):
                close = base_price + random.uniform(-5, 5)
                candles.append({
                    "time": datetime.now().timestamp() - (count - i) * 60,
                    "open": base_price,
                    "high": max(base_price, close) + 1,
                    "low": min(base_price, close) - 1,
                    "close": close,
                    "tick_volume": 100,
                })
                base_price = close
            return candles

        # Real MT5
        tf_map = {1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15}
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
        
        rates = None
        for attempt in range(3):
            rates = await asyncio.to_thread(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, count)
            if rates is not None and len(rates) > 0:
                break
            # Trigger download and wait
            logger.debug(f"MT5: Waiting for data {symbol} (Attempt {attempt+1}/3)...")
            await asyncio.sleep(0.5)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No rates found for {symbol}")
            return []

        # Convert to list of dicts
        candles = []
        for rate in rates:
            candles.append({
                "time": rate['time'],
                "open": rate['open'],
                "high": rate['high'],
                "low": rate['low'],
                "close": rate['close'],
                "tick_volume": rate['tick_volume'],
            })
        return candles

    async def get_symbol_tick(self, symbol: str) -> dict[str, Any]:
        """Récupère le dernier tick pour un symbole"""
        if self.mock_mode:
            # Prix simulés
            mock_prices = {
                "XAUUSD": Decimal("2080.50"),
                "EURUSD": Decimal("1.0855"),
                "GBPUSD": Decimal("1.2655"),
                "USDJPY": Decimal("150.55"),
            }
            price = mock_prices.get(symbol, Decimal("100.00"))
            return {
                "symbol": symbol,
                "bid": float(price),
                "ask": float(price) + 0.0001,
                "time": datetime.now().timestamp()
            }

        tick = await asyncio.to_thread(mt5.symbol_info_tick, symbol)
        if tick is None:
            return {"success": False, "message": f"Dernier tick non disponible pour {symbol}"}

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "time": tick.time
        }

    def _get_deviation(self, symbol: str) -> int:
        """Retourne la déviation (slippage) recommandée selon la volatilité de l'actif."""
        if any(v in symbol.upper() for v in ["XAU", "BTC", "US30", "NAS100", "GER40"]):
            return 50  # 5 pips pour les actifs volatils
        return 20  # 2 pips par défaut pour le Forex

    async def execute_order(self, order: TradeOrder) -> dict[str, Any]:
        """Exécute un ordre de trading avec gestion des Requotes et Slippage."""
        if self.mock_mode:
            return await self._execute_mock_order(order)

        symbol_info = await asyncio.to_thread(mt5.symbol_info, order.symbol)
        if symbol_info is None:
            return {"success": False, "message": f"Symbole {order.symbol} non trouvé"}

        deviation = self._get_deviation(order.symbol)
        
        # Retry Loop pour gérer les Requotes/Busy terminal
        for attempt in range(3):
            price = await asyncio.to_thread(mt5.symbol_info_tick, order.symbol)
            if price is None:
                await asyncio.sleep(0.5)
                continue
                
            # --- NEW (Sprint 10): Dynamic Spread Filter ---
            # Mesure du spread actuel en points (tick_size dépendant)
            current_spread = (price.ask - price.bid) / mt5.symbol_info(order.symbol).point
            
            # Limites de spread (Valeurs empiriques max)
            max_spread = 25 # Forex default (2.5 pips)
            sym = order.symbol.upper()
            if "XAU" in sym:
                max_spread = 40 # Or: 40 points = 4 pips ($0.40)
            elif "US30" in sym or "NAS100" in sym or "GER40" in sym:
                max_spread = 60 # Indices: 60 points
                
            if current_spread > max_spread:
                logger.warning(f"❌ Spread trop élevé sur {order.symbol}: {current_spread:.1f} > {max_spread}. Ordre avorté.")
                return {
                    "success": False,
                    "message": f"Échec Sécurité: Spread ({current_spread:.1f} pts) > Max autorisé ({max_spread} pts).",
                    "retcode": 99999, # Code d'erreur interne
                }
            # ---------------------------------------------

            order_type = mt5.ORDER_TYPE_BUY if order.action == TradeAction.BUY else mt5.ORDER_TYPE_SELL
            exec_price = price.ask if order.action == TradeAction.BUY else price.bid

            # Sanitize comment
            raw_comment = order.comment or "EVA"
            safe_comment = "".join(c for c in raw_comment if c.isalnum() or c in " -_.")[:31]
            if not safe_comment: safe_comment = "EVA"

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": float(order.volume),
                "type": order_type,
                "price": exec_price,
                "sl": float(order.stop_loss_price) if order.stop_loss_price else 0.0,
                "tp": float(order.take_profit_price) if order.take_profit_price else 0.0,
                "deviation": deviation,
                "magic": order.magic_number,
                "comment": safe_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = await asyncio.to_thread(mt5.order_send, request)
            
            if result is None:
                logger.warning(f"MT5 order_send returned None for {order.symbol} (Attempt {attempt+1})")
                await asyncio.sleep(0.5)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    "success": True,
                    "ticket": result.order,
                    "message": f"Ordre exécuté: {order.action.value} {order.volume} {order.symbol} (Att: {attempt+1})",
                }
            
            # Requotes (10004) ou Request Rejected (10006) ou Price Changed (10021)
            if result.retcode in [10004, 10006, 10021]:
                logger.info(f"Retrying order {order.symbol} due to {result.comment} (Retcode: {result.retcode})")
                await asyncio.sleep(0.5)
                continue
            else:
                return {
                    "success": False,
                    "message": f"Erreur MT5: {result.comment}",
                    "retcode": result.retcode,
                }

        return {"success": False, "message": "Échec de l'ordre après 3 tentatives (Slippage/Requotes)"}

    async def close_position(self, ticket: int) -> dict[str, Any]:
        """Ferme une position par son ticket"""
        if self.mock_mode:
            pos = next((p for p in self._mock_positions if p.ticket == ticket), None)
            if not pos:
                return {"success": False, "message": f"Position {ticket} non trouvée"}
            
            self._mock_positions = [p for p in self._mock_positions if p.ticket != ticket]
            # Simulation de profit pour le mock (entre -50 et +150)
            import random
            profit = Decimal(str(random.uniform(-50, 150)))
            return {
                "success": True, 
                "message": f"Position {ticket} fermée (mock)",
                "profit": float(profit),
                "symbol": pos.symbol
            }

        position = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if not position:
            return {"success": False, "message": f"Position {ticket} non trouvée"}

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        deviation = self._get_deviation(pos.symbol)
        
        for attempt in range(3):
            price = await asyncio.to_thread(mt5.symbol_info_tick, pos.symbol)
            if price is None:
                await asyncio.sleep(0.5)
                continue
                
            close_price = price.bid if pos.type == 0 else price.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": deviation,
                "magic": pos.magic,
                "comment": "EVA Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = await asyncio.to_thread(mt5.order_send, request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    "success": True, 
                    "ticket": ticket, 
                    "message": f"Position fermée (Att: {attempt+1})",
                    "profit": pos.profit,
                    "symbol": pos.symbol
                }
            
            if result.retcode in [10004, 10006, 10021]:
                await asyncio.sleep(0.5)
                continue
            else:
                return {"success": False, "message": f"Erreur fermeture: {result.comment}"}

        return {"success": False, "message": "Échec de fermeture après 3 tentatives (Slippage/Requotes)"}

    async def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> dict[str, Any]:
        """Modifie le SL/TP d'une position"""
        if self.mock_mode:
            pos = next((p for p in self._mock_positions if p.ticket == ticket), None)
            if pos:
                if sl > 0: pos.stop_loss = Decimal(str(sl))
                if tp > 0: pos.take_profit = Decimal(str(tp))
                return {"success": True, "message": f"Position {ticket} modified (mock)"}
            return {"success": False, "message": "Position not found"}

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(sl),
            "tp": float(tp),
        }
        
        result = await asyncio.to_thread(mt5.order_send, request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "message": f"Erreur modification: {result.comment}"}
            
        return {"success": True, "message": f"Position {ticket} modified SL={sl} TP={tp}"}

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
        """
        logger.info(f"Executing skill {skill} for {order.symbol}")
        return await self.execute_order(order)

    async def get_deal_history(self, from_dt: datetime, to_dt: datetime) -> list[dict]:
        """Récupère l'historique des deals (trades fermés) sur une période."""
        if self.mock_mode:
            return []
        
        try:
            deals = await asyncio.to_thread(mt5.history_deals_get, from_dt, to_dt)
            if deals is None:
                return []
            
            result = []
            for deal in deals:
                if deal.entry == 1:  # DEAL_ENTRY_OUT = fermeture
                    result.append({
                        "ticket": deal.ticket,
                        "order": deal.order,
                        "position_id": deal.position_id,
                        "symbol": deal.symbol,
                        "type": "BUY" if deal.type == 0 else "SELL",
                        "volume": deal.volume,
                        "price": deal.price,
                        "profit": deal.profit,
                        "swap": deal.swap,
                        "commission": deal.commission,
                        "time": datetime.fromtimestamp(deal.time),
                        "comment": deal.comment,
                        "magic": deal.magic,
                    })
            return result
        except Exception as e:
            logger.error(f"Error fetching deal history: {e}")
            return []

    async def get_account_summary(self) -> dict:
        """Récupère un résumé du compte (pour Daily Report)."""
        info = await self.get_account_info()
        if not info:
            return {}
        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "free_margin": float(info.free_margin),
            "profit": float(info.equity) - float(info.balance),
        }

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
