"""
Service MT5 - Client MetaTrader 5
GÃ¨re la connexion et l'exÃ©cution des ordres sur MT5
"""

import asyncio
import logging
import sys
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, List, Optional

from shared import AccountBalance, Position, TradeAction, TradeOrder, get_settings

logger = logging.getLogger(__name__)

# MT5 ne fonctionne que sur Windows
MT5_AVAILABLE = sys.platform == "win32"

if MT5_AVAILABLE:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        MT5_AVAILABLE = False
        logger.warning("MetaTrader5 non installÃ©")

MOCK_DISCOVERY_SYMBOLS = [
    {"name": "BTCUSD", "path": "Crypto\\Majors", "trade_mode": 1},
    {"name": "ETHUSD", "path": "Crypto\\Majors", "trade_mode": 1},
    {"name": "SOLUSD", "path": "Crypto\\Altcoins", "trade_mode": 1},
    {"name": "EURUSD", "path": "Forex\\Majors", "trade_mode": 1},
    {"name": "GBPUSD", "path": "Forex\\Majors", "trade_mode": 1},
    {"name": "USDJPY", "path": "Forex\\Majors", "trade_mode": 1},
    {"name": "AUDUSD", "path": "Forex\\Majors", "trade_mode": 1},
    {"name": "USDCAD", "path": "Forex\\Majors", "trade_mode": 1},
    {"name": "XAUUSD", "path": "CFD\\Metals", "trade_mode": 1},
    {"name": "XAGUSD", "path": "CFD\\Metals", "trade_mode": 1},
    {"name": "US30.cash", "path": "CFD\\Indices", "trade_mode": 1},
    {"name": "US100.cash", "path": "CFD\\Indices", "trade_mode": 1},
    {"name": "GER40.cash", "path": "CFD\\Indices", "trade_mode": 1},
    {"name": "UK100.cash", "path": "CFD\\Indices", "trade_mode": 1},
    {"name": "AAPL.cash", "path": "CFD\\Stocks", "trade_mode": 1},
]

FOREX_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "CNH",
    "EUR",
    "GBP",
    "HKD",
    "JPY",
    "MXN",
    "NOK",
    "NZD",
    "SEK",
    "SGD",
    "TRY",
    "USD",
    "ZAR",
}

CRYPTO_BASES = {
    "ADA",
    "AVAX",
    "BNB",
    "BTC",
    "DOGE",
    "DOT",
    "ETH",
    "LINK",
    "LTC",
    "MATIC",
    "SOL",
    "UNI",
    "XBT",
    "XRP",
}

CRYPTO_QUOTES = ("USDT", "USDC", "USD", "EUR", "BTC", "ETH")



class MT5Service:
    """
    Client MetaTrader 5 pour exÃ©cution des ordres.
    
    Supporte:
    - Mode rÃ©el (Windows avec MT5 installÃ©)
    - Mode mock (dÃ©veloppement / paper trading)
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
                self.mock_mode = True
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
                        self.mock_mode = True
                        return False
            elif account_info:
                logger.info(f"MT5 connecte: compte {account_info.login} sur {account_info.server}")
            else:
                logger.warning("MT5 initialise mais aucun compte connecte")

            self.is_connected = True
            return True
        except Exception as e:
            logger.exception(f"Erreur connexion MT5: {e}")
            self.mock_mode = True
            return False

    async def initialize_symbols(self, symbols: list[str]) -> None:
        """
        S'assure que les symboles sont selectionnes dans le Market Watch.

        Args:
            symbols (list[str]): Symboles a rendre disponibles dans MT5.
        """
        if self.mock_mode:
            return

        for symbol in symbols:
            await self.ensure_symbol_selected(symbol)

    async def ensure_symbol_selected(self, symbol: str) -> bool:
        """
        Selectionne un symbole dans MT5 si necessaire.

        Args:
            symbol (str): Symbole a rendre visible.

        Returns:
            bool: True si le symbole est disponible, sinon False.
        """
        if self.mock_mode:
            return True

        info = await asyncio.to_thread(mt5.symbol_info, symbol)
        if info is None:
            logger.warning("Symbole %s introuvable dans MT5.", symbol)
            return False

        if getattr(info, "visible", False):
            return True

        selected = await asyncio.to_thread(mt5.symbol_select, symbol, True)
        if not selected:
            logger.warning(
                "Impossible de selectionner le symbole %s: %s",
                symbol,
                mt5.last_error(),
            )
            return False
        return True

    async def discover_symbols(
        self,
        include_forex: bool = True,
        include_cfd: bool = True,
        include_crypto: bool = True,
        max_symbols: int = 0,
    ) -> list[str]:
        """
        Decouvre les symboles tradables disponibles sur le broker.

        Args:
            include_forex (bool): Inclut les paires Forex si True.
            include_cfd (bool): Inclut les CFD indices, metaux, matieres premieres et actions si True.
            include_crypto (bool): Inclut les cryptos si True.
            max_symbols (int): Limite optionnelle du nombre de symboles retournes. `0` desactive la limite.

        Returns:
            list[str]: Liste dedoublonnee et triee de symboles.
        """
        if self.mock_mode:
            raw_symbols = MOCK_DISCOVERY_SYMBOLS
        else:
            raw_symbols = await asyncio.to_thread(mt5.symbols_get)
            if raw_symbols is None:
                logger.warning("MT5: impossible de recuperer l'univers de symboles.")
                return []

        discovered: list[str] = []
        for entry in raw_symbols:
            symbol_info = self._normalize_symbol_entry(entry)
            name = symbol_info["name"]
            if not name:
                continue

            trade_mode = symbol_info.get("trade_mode")
            if trade_mode == 0:
                continue

            asset_class = self.classify_symbol(name, symbol_info.get("path", ""))
            if asset_class == "forex" and not include_forex:
                continue
            if asset_class == "cfd" and not include_cfd:
                continue
            if asset_class == "crypto" and not include_crypto:
                continue
            if asset_class is None:
                continue

            discovered.append(name)

        ordered = self._sort_symbol_universe(discovered)
        if max_symbols > 0:
            ordered = ordered[:max_symbols]
        return ordered

    def classify_symbol(self, symbol: str, path: str = "") -> str | None:
        """
        Classe un symbole dans une famille de marche exploitable par le banker.

        Args:
            symbol (str): Symbole brut du broker.
            path (str): Chemin ou groupe Market Watch fourni par MT5.

        Returns:
            str | None: `crypto`, `forex`, `cfd` ou `None` si inconnu.
        """
        symbol_upper = symbol.upper()
        path_upper = (path or "").upper()

        if "CRYPTO" in path_upper or self._looks_like_crypto_symbol(symbol_upper):
            return "crypto"

        if any(keyword in path_upper for keyword in ["CFD", "INDEX", "INDICES", "METAL", "METALS", "COMMOD", "ENER", "STOCK", "SHARE", "FUTURE"]):
            return "cfd"
        if self._looks_like_cfd_symbol(symbol_upper):
            return "cfd"

        if "FOREX" in path_upper or "FX" in path_upper:
            return "forex"
        if self._looks_like_forex_symbol(symbol_upper):
            return "forex"

        return None

    def _normalize_symbol_entry(self, entry: Any) -> dict[str, Any]:
        """Normalise un symbole MT5 ou mock vers un dictionnaire simple."""
        if isinstance(entry, dict):
            return {
                "name": entry.get("name", ""),
                "path": entry.get("path", ""),
                "trade_mode": entry.get("trade_mode"),
            }

        return {
            "name": getattr(entry, "name", ""),
            "path": getattr(entry, "path", ""),
            "trade_mode": getattr(entry, "trade_mode", None),
        }

    def _looks_like_crypto_symbol(self, symbol: str) -> bool:
        """Retourne True si le symbole ressemble a une paire crypto."""
        clean = "".join(char for char in symbol if char.isalnum())
        for quote in CRYPTO_QUOTES:
            if clean.endswith(quote) and len(clean) > len(quote):
                base = clean[: -len(quote)]
                if base in CRYPTO_BASES and base not in FOREX_CODES:
                    return True
        return False

    def _looks_like_forex_symbol(self, symbol: str) -> bool:
        """Retourne True si le symbole ressemble a une paire Forex."""
        clean = "".join(char for char in symbol if char.isalpha())
        if len(clean) < 6:
            return False
        base = clean[:3]
        quote = clean[3:6]
        if base in {"XAU", "XAG", "XPT", "XPD"}:
            return False
        return base in FOREX_CODES and quote in FOREX_CODES

    def _looks_like_cfd_symbol(self, symbol: str) -> bool:
        """Retourne True si le symbole ressemble a un CFD."""
        return any(
            token in symbol
            for token in [
                ".CASH",
                "US30",
                "US100",
                "GER40",
                "UK100",
                "NAS100",
                "SPX500",
                "XAU",
                "XAG",
                "BRENT",
                "WTI",
            ]
        )

    def _sort_symbol_universe(self, symbols: list[str]) -> list[str]:
        """Trie les symboles de facon stable avec priorite aux actifs liquides."""
        preferred = {
            "BTCUSD": 0,
            "ETHUSD": 1,
            "EURUSD": 2,
            "GBPUSD": 3,
            "USDJPY": 4,
            "XAUUSD": 5,
            "US30.CASH": 6,
            "US100.CASH": 7,
            "GER40.CASH": 8,
        }
        asset_weight = {"crypto": 0, "forex": 1, "cfd": 2, None: 9}
        unique_symbols = list(dict.fromkeys(symbols))
        return sorted(
            unique_symbols,
            key=lambda item: (
                preferred.get(item.upper(), 999),
                asset_weight.get(self.classify_symbol(item), 9),
                item,
            ),
        )


    async def disconnect(self) -> None:
        """DÃ©connexion de MT5"""
        if not self.mock_mode and MT5_AVAILABLE:
            await asyncio.to_thread(mt5.shutdown)
        self.is_connected = False
        logger.info("MT5 dÃ©connectÃ©")

    async def get_account_info(self) -> Optional[AccountBalance]:
        """RÃ©cupÃ¨re les informations du compte"""
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
            logger.warning("MT5: Impossible de rÃ©cupÃ©rer les infos du compte (terminal occupÃ© ou dÃ©connectÃ©)")
            return None

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
        """RÃ©cupÃ¨re les positions ouvertes"""
        if self.mock_mode:
            return self._mock_positions

        positions_data = await asyncio.to_thread(mt5.positions_get)
        if positions_data is None:
            # None indicates a terminal/connection error, not "no positions"
            return None

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

    async def get_mtf_candles(self, symbol: str, timeframes: list[int] = [5, 60, 1440], count: int = 100) -> dict[int, list[dict]]:
        """
        RÃ©cupÃ¨re les bougies OMNI-STATE (Multi-Timeframe) synchronisÃ©es.
        Renvoie un dictionnaire: {5: [candles_m5], 60: [candles_h1], 1440: [candles_d1]}
        """
        import random
        from datetime import datetime
        
        tf_map = {
            1: 1 if self.mock_mode else mt5.TIMEFRAME_M1, 
            5: 5 if self.mock_mode else mt5.TIMEFRAME_M5, 
            15: 15 if self.mock_mode else mt5.TIMEFRAME_M15,
            60: 60 if self.mock_mode else mt5.TIMEFRAME_H1,
            1440: 1440 if self.mock_mode else mt5.TIMEFRAME_D1
        }
        
        result = {}
        
        for tf in timeframes:
            mt5_tf = tf_map.get(tf, tf_map[1])
            
            if self.mock_mode:
                # Generate fake candles
                candles = []
                base_price = 2080.0
                for i in range(count):
                    close = base_price + random.uniform(-5, 5)
                    candles.append({
                        "time": datetime.now().timestamp() - (count - i) * (tf * 60),
                        "open": base_price,
                        "high": max(base_price, close) + 1,
                        "low": min(base_price, close) - 1,
                        "close": close,
                        "tick_volume": 100,
                    })
                    base_price = close
                result[tf] = candles
                continue

            # Real MT5
            rates = None
            for attempt in range(3):
                rates = await asyncio.to_thread(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, count)
                if rates is not None and len(rates) > 0:
                    break
                logger.debug(f"MT5: Waiting for MTF data {symbol} (TF={tf}, Attempt {attempt+1}/3)...")
                await asyncio.sleep(0.5)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No rates found for {symbol} on TF {tf}")
                result[tf] = []
            else:
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
                result[tf] = candles
                
        return result

    async def get_recent_candles(self, symbol: str, timeframe: int = 15, count: int = 20) -> list[dict]:
        """Wrapper de compatibilitÃ© (Legacy 1D)"""
        res = await self.get_mtf_candles(symbol, [timeframe], count)
        return res.get(timeframe, [])

    async def get_symbol_tick(self, symbol: str) -> dict[str, Any]:
        """RÃ©cupÃ¨re le dernier tick pour un symbole"""
        if self.mock_mode:
            # Prix simulÃ©s
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
        """Retourne la dÃ©viation (slippage) recommandÃ©e selon la volatilitÃ© de l'actif."""
        if any(v in symbol.upper() for v in ["XAU", "BTC", "US30", "US100", "GER40"]):
            return 50  # 5 pips pour les actifs volatils
        return 20  # 2 pips par dÃ©faut pour le Forex

    async def execute_order(self, order: TradeOrder) -> dict[str, Any]:
        """ExÃ©cute un ordre de trading avec gestion des Requotes et Slippage."""
        if self.mock_mode:
            return await self._execute_mock_order(order)

        symbol_info = await asyncio.to_thread(mt5.symbol_info, order.symbol)
        if symbol_info is None:
            return {"success": False, "message": f"Symbole {order.symbol} non trouvÃ©"}

        deviation = self._get_deviation(order.symbol)
        
        # Retry Loop pour gÃ©rer les Requotes/Busy terminal
        for attempt in range(3):
            price = await asyncio.to_thread(mt5.symbol_info_tick, order.symbol)
            if price is None:
                await asyncio.sleep(0.5)
                continue
                
            # --- NEW (Sprint 10): Dynamic Spread Filter ---
            # Mesure du spread actuel en points (tick_size dÃ©pendant)
            current_spread = (price.ask - price.bid) / mt5.symbol_info(order.symbol).point
            
            # Limites de spread (Valeurs empiriques max)
            max_spread = 25 # Forex default (2.5 pips)
            sym = order.symbol.upper()
            if "XAU" in sym:
                max_spread = 60 # Gold: 60 points ($0.60)
            elif "BTC" in sym or "ETH" in sym:
                max_spread = 1500 # Crypto: 1500 points ($15.00) for BTC
            elif "US30" in sym or "US100" in sym or "GER40" in sym:
                max_spread = 150 # Indices: 150 points
                
            if current_spread > max_spread:
                logger.warning(f"âŒ Spread trop Ã©levÃ© sur {order.symbol}: {current_spread:.1f} > {max_spread}. Ordre avortÃ©.")
                return {
                    "success": False,
                    "message": f"Ã‰chec SÃ©curitÃ©: Spread ({current_spread:.1f} pts) > Max autorisÃ© ({max_spread} pts).",
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
                    "message": f"Ordre exÃ©cutÃ©: {order.action.value} {order.volume} {order.symbol} (Att: {attempt+1})",
                }
            
            # Requotes (10004) ou Request Rejected (10006) ou Price Changed (10021)
            if result.retcode in [10004, 10006, 10021, 10031]:
                logger.info(f"Retrying order {order.symbol} due to {result.comment} (Retcode: {result.retcode})")
                if result.retcode == 10031: # No connection
                    logger.warning("ðŸŒ MT5 Connection lost (10031). Attempting emergency reconnection...")
                    await self.connect()
                await asyncio.sleep(1.0)
                continue
            else:
                return {
                    "success": False,
                    "message": f"Erreur MT5: {result.comment}",
                    "retcode": result.retcode,
                }

        return {"success": False, "message": "Ã‰chec de l'ordre aprÃ¨s 3 tentatives (Slippage/Requotes)"}

    async def close_position(self, ticket: int) -> dict[str, Any]:
        """Ferme une position par son ticket"""
        if self.mock_mode:
            pos = next((p for p in self._mock_positions if p.ticket == ticket), None)
            if not pos:
                return {"success": False, "message": f"Position {ticket} non trouvÃ©e"}
            
            self._mock_positions = [p for p in self._mock_positions if p.ticket != ticket]
            # Simulation de profit pour le mock (entre -50 et +150)
            import random
            profit = Decimal(str(random.uniform(-50, 150)))
            return {
                "success": True, 
                "message": f"Position {ticket} fermÃ©e (mock)",
                "profit": float(profit),
                "symbol": pos.symbol
            }

        position = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if not position:
            return {"success": False, "message": f"Position {ticket} non trouvÃ©e"}

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
                    "message": f"Position fermÃ©e (Att: {attempt+1})",
                    "profit": pos.profit,
                    "symbol": pos.symbol
                }
            
            if result.retcode in [10004, 10006, 10021]:
                await asyncio.sleep(0.5)
                continue
            else:
                return {"success": False, "message": f"Erreur fermeture: {result.comment}"}

        return {"success": False, "message": "Ã‰chec de fermeture aprÃ¨s 3 tentatives (Slippage/Requotes)"}

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

    async def get_margin_required(self, symbol: str, action: TradeAction, volume: float) -> Optional[float]:
        """Estime la marge requise pour un ordre (Sprint 13)."""
        if self.mock_mode:
            # Estimation pifomÃ©trique pour le mock
            return volume * 500.0  # $500 de marge par lot
            
        order_type = mt5.ORDER_TYPE_BUY if action == TradeAction.BUY else mt5.ORDER_TYPE_SELL
        
        # RÃ©cupÃ©rer le tick actuel pour le calcul
        tick = await asyncio.to_thread(mt5.symbol_info_tick, symbol)
        if tick is None:
            return None
            
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        margin = await asyncio.to_thread(mt5.order_calc_margin, order_type, symbol, volume, price)
        if margin is None:
            logger.warning(f"Calcul de marge Ã©chouÃ© pour {symbol} {volume} lots")
            return None
            
        return float(margin)

    async def _execute_mock_order(self, order: TradeOrder) -> dict[str, Any]:
        """ExÃ©cute un ordre en mode mock"""
        ticket = self._next_ticket
        self._next_ticket += 1

        # Prix simulÃ©
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

        logger.info(f"ðŸ“Š Mock Order: {order.action.value} {order.volume} {order.symbol} @ {price}")

        return {
            "success": True,
            "ticket": ticket,
            "message": f"[MOCK] {order.action.value} {order.volume} {order.symbol}",
        }

    async def execute_skill(self, skill, order: TradeOrder) -> dict[str, Any]:
        """
        Execute un ordre en utilisant une compÃ©tence (Skill) spÃ©cifique.
        Dispatche l'exÃ©cution en fonction du type de skill sÃ©lectionnÃ©
        par le Manager (niveau haut de l'architecture hiÃ©rarchique SPlaTES).
        """
        logger.info(f"Executing skill {skill} for {order.symbol}")
        return await self.execute_order(order)

    async def get_deal_history(self, from_dt: datetime, to_dt: datetime) -> list[dict]:
        """RÃ©cupÃ¨re l'historique des deals (trades fermÃ©s) sur une pÃ©riode."""
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
        """RÃ©cupÃ¨re un rÃ©sumÃ© du compte (pour Daily Report)."""
        info = await self.get_account_info()
        if not info:
            return {}
        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "margin_free": float(info.free_margin),
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
