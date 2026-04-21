"""
Service MT5 - Client MetaTrader 5
GÃ¨re la connexion et l'exÃ©cution des ordres sur MT5
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_FLOOR
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

    def __init__(
        self,
        mock_mode: bool = True,
        login: int = 0,
        password: str = "",
        server: str = "",
        terminal_path: str = "",
        terminal_portable: bool = False,
        terminal_timeout_ms: int = 60000,
    ):
        """
        Initialise le service MT5 avec les identifiants et le terminal cible.

        Args:
            mock_mode (bool): Active le mode mock si True.
            login (int): Login MT5 a utiliser en mode reel.
            password (str): Mot de passe du compte MT5.
            server (str): Serveur MT5 associe au compte.
            terminal_path (str): Chemin du terminal MT5 a piloter.
            terminal_portable (bool): Active le mode portable du terminal si True.
            terminal_timeout_ms (int): Timeout d'initialisation du terminal en millisecondes.
        """
        settings = get_settings()
        self._explicit_mock_mode = bool(mock_mode)
        self.mock_mode = self._explicit_mock_mode
        self.is_connected = False
        self._mock_positions: list[Position] = []
        self._mock_balance = Decimal("100000.00")
        self._next_ticket = 12345678
        self._duplicate_order_cooldown = timedelta(
            seconds=max(5, settings.mt5_duplicate_order_cooldown_seconds)
        )
        self._recent_order_signatures: dict[str, datetime] = {}
        self._inflight_order_signatures: set[str] = set()
        self._order_guard_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._reconnect_cooldown = timedelta(
            seconds=max(5, getattr(settings, "mt5_reconnect_cooldown_seconds", 15))
        )
        self._warning_cooldown = timedelta(
            seconds=max(10, getattr(settings, "mt5_warning_cooldown_seconds", 30))
        )
        self._last_reconnect_attempt: datetime | None = None
        self._last_offline_warning: datetime | None = None
        self._last_disconnect_reason: str | None = None
        # Credentials pour login automatique
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = str(terminal_path or "").strip()
        self._terminal_portable = bool(terminal_portable)
        self._terminal_timeout_ms = max(1000, int(terminal_timeout_ms or 60000))
        logger.info(
            "MT5Service initialise (mock_explicite=%s, mock_actif=%s, login=%s, server=%s, terminal=%s, portable=%s)",
            self._explicit_mock_mode,
            self.mock_mode,
            login,
            server,
            self._terminal_path or "<terminal-auto>",
            self._terminal_portable,
        )

    def _build_initialize_kwargs(self) -> dict[str, Any]:
        """
        Construit les parametres d'initialisation du terminal MT5.

        Returns:
            dict[str, Any]: Parametres compatibles avec `MetaTrader5.initialize`.
        """
        init_kwargs: dict[str, Any] = {
            "timeout": self._terminal_timeout_ms,
        }
        if self._terminal_path:
            init_kwargs["path"] = self._terminal_path
        if self._terminal_portable:
            init_kwargs["portable"] = True
        return init_kwargs

    def _live_mode_requested(self) -> bool:
        """
        Indique si le service doit fonctionner en mode reel.

        Returns:
            bool: True si aucun mode mock explicite n'a ete demande.
        """
        return not self._explicit_mock_mode

    def _mark_live_disconnected(self, reason: str) -> None:
        """
        Marque le service MT5 comme hors ligne sans activer le mock.

        Args:
            reason (str): Raison principale de la perte de connexion.
        """
        was_connected = self.is_connected
        self.mock_mode = self._explicit_mock_mode
        self.is_connected = False
        if was_connected or reason != self._last_disconnect_reason:
            self._log_offline_warning("MT5 hors ligne: %s", reason)
        self._last_disconnect_reason = reason

    def _should_emit_offline_warning(self) -> bool:
        """
        Indique si un nouveau warning hors ligne peut etre emis.

        Returns:
            bool: True si le cooldown de warning est ecoule.
        """
        now = datetime.now()
        if self._last_offline_warning is None:
            self._last_offline_warning = now
            return True
        if now - self._last_offline_warning >= self._warning_cooldown:
            self._last_offline_warning = now
            return True
        return False

    def _log_offline_warning(self, message: str, *args: Any) -> None:
        """
        Emet un warning hors ligne avec anti-spam.

        Args:
            message (str): Message a journaliser.
            *args (Any): Arguments de formatage du logger.
        """
        if self._should_emit_offline_warning():
            logger.warning(message, *args)

    async def _ensure_live_connection(self, reason: str, *, force: bool = False) -> bool:
        """
        Tente de retablir automatiquement la connexion MT5 si elle est perdue.

        Args:
            reason (str): Contexte de la tentative de reconnexion.
            force (bool): Ignore temporairement le cooldown de reconnexion.

        Returns:
            bool: True si la connexion est de nouveau disponible.
        """
        if self.mock_mode or self.is_connected:
            return True
        if not self._live_mode_requested():
            return False

        async with self._reconnect_lock:
            if self.mock_mode or self.is_connected:
                return True

            now = datetime.now()
            if (
                not force
                and self._last_reconnect_attempt is not None
                and now - self._last_reconnect_attempt < self._reconnect_cooldown
            ):
                self._log_offline_warning(
                    "MT5: reconnexion automatique en attente (%s).",
                    reason,
                )
                return False

            self._last_reconnect_attempt = now
            logger.warning("MT5: tentative de reconnexion automatique (%s).", reason)
            if MT5_AVAILABLE:
                try:
                    await asyncio.to_thread(mt5.shutdown)
                except Exception:
                    pass
            connected = await self.connect()
            if connected:
                self._last_offline_warning = None
                logger.info("MT5: reconnexion automatique reussie (%s).", reason)
                return True

            self._log_offline_warning(
                "MT5: echec de reconnexion automatique (%s).",
                reason,
            )
            return False

    def _build_order_signature(self, order: TradeOrder) -> str:
        """
        Construit une signature stable pour detecter un doublon d'ordre.

        Args:
            order (TradeOrder): Ordre en preparation d'execution.

        Returns:
            str: Signature exploitable pour les garde-fous anti-doublon.
        """
        return (
            f"{order.symbol.upper()}:{order.action.value}:"
            f"{float(order.volume):.4f}:{order.magic_number}"
        )

    def _prune_recent_order_signatures(self, now: datetime) -> None:
        """
        Purge les signatures d'ordres expirees du cache anti-doublon.

        Args:
            now (datetime): Horodatage courant utilise comme reference.
        """
        expired = [
            signature
            for signature, timestamp in self._recent_order_signatures.items()
            if now - timestamp >= self._duplicate_order_cooldown
        ]
        for signature in expired:
            self._recent_order_signatures.pop(signature, None)

    async def _register_order_attempt(self, signature: str) -> dict[str, Any] | None:
        """
        Verrouille une signature d'ordre avant envoi vers MT5.

        Args:
            signature (str): Signature calculee pour l'ordre.

        Returns:
            dict[str, Any] | None: Un resultat d'echec si l'ordre est bloque,
                sinon `None` pour autoriser l'execution.
        """
        async with self._order_guard_lock:
            now = datetime.now()
            self._prune_recent_order_signatures(now)

            if signature in self._inflight_order_signatures:
                logger.warning("Ordre duplique bloque pendant une execution deja en cours: %s", signature)
                return {
                    "success": False,
                    "message": "Ordre duplique bloque pendant l'execution.",
                    "retcode": 99001,
                }

            last_execution = self._recent_order_signatures.get(signature)
            if last_execution is not None:
                remaining = self._duplicate_order_cooldown - (now - last_execution)
                logger.warning(
                    "Ordre recent duplique bloque: %s (fenetre restante %.1fs)",
                    signature,
                    max(0.0, remaining.total_seconds()),
                )
                return {
                    "success": False,
                    "message": "Ordre recent duplique bloque.",
                    "retcode": 99002,
                }

            self._inflight_order_signatures.add(signature)
        return None

    async def _release_order_attempt(self, signature: str, remember_execution: bool) -> None:
        """
        Libere un verrou d'ordre et memorise l'execution si elle a reussi.

        Args:
            signature (str): Signature calculee pour l'ordre.
            remember_execution (bool): True si l'ordre a ete execute et doit
                etre protege pendant la fenetre anti-doublon.
        """
        async with self._order_guard_lock:
            self._inflight_order_signatures.discard(signature)
            if remember_execution:
                self._recent_order_signatures[signature] = datetime.now()

    async def connect(self) -> bool:
        """Connexion a MT5"""
        if self._explicit_mock_mode:
            self.mock_mode = True
            self.is_connected = True
            logger.info("MT5 Mock: connecte")
            return True

        self.mock_mode = False

        if not MT5_AVAILABLE:
            self._mark_live_disconnected(
                "MetaTrader 5 est indisponible sur cette machine alors que le mode reel est demande."
            )
            return False

        try:
            # Initialisation MT5
            init_kwargs = self._build_initialize_kwargs()
            if not await asyncio.to_thread(mt5.initialize, **init_kwargs):
                self._mark_live_disconnected(f"Echec d'initialisation MT5: {mt5.last_error()}")
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
                        self._mark_live_disconnected(
                            f"Impossible d'ouvrir la session MT5 {self._login}@{self._server}."
                        )
                        return False
            elif account_info:
                logger.info(f"MT5 connecte: compte {account_info.login} sur {account_info.server}")
            else:
                self._mark_live_disconnected("MT5 initialise mais aucun compte n'est connecte.")
                await asyncio.to_thread(mt5.shutdown)
                return False

            self.mock_mode = False
            self.is_connected = True
            if self._last_disconnect_reason:
                logger.info("MT5: connexion retablie sur le compte %s.", account_info.login)
            self._last_disconnect_reason = None
            self._last_offline_warning = None
            return True
        except Exception as e:
            logger.exception(f"Erreur connexion MT5: {e}")
            self._mark_live_disconnected(str(e))
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

        if not self.is_connected and not await self._ensure_live_connection("lecture des informations compte"):
            self._log_offline_warning("MT5: infos compte indisponibles car la connexion est hors ligne.")
            return None

        info = await asyncio.to_thread(mt5.account_info)
        if info is None:
            self._mark_live_disconnected("Informations compte indisponibles depuis le terminal.")
            if not await self._ensure_live_connection("lecture des informations compte apres echec", force=True):
                return None
            info = await asyncio.to_thread(mt5.account_info)
            if info is None:
                self._mark_live_disconnected(
                    "Informations compte toujours indisponibles apres reconnexion."
                )
                return None

        self.is_connected = True
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

    async def get_open_positions(self) -> Optional[list[Position]]:
        """RÃ©cupÃ¨re les positions ouvertes.

        Returns:
            Optional[list[Position]]: Liste des positions si la lecture MT5
            est disponible. Renvoie ``None`` en cas de rupture de connexion
            pour eviter de confondre un incident reseau avec une absence
            reelle de position.
        """
        if self.mock_mode:
            return self._mock_positions

        if not self.is_connected and not await self._ensure_live_connection("lecture des positions"):
            self._log_offline_warning("MT5: positions indisponibles car la connexion est hors ligne.")
            return None

        positions_data = await asyncio.to_thread(mt5.positions_get)
        if positions_data is None:
            # None indicates a terminal/connection error, not "no positions"
            self._mark_live_disconnected("Lecture des positions indisponible depuis le terminal.")
            if not await self._ensure_live_connection("lecture des positions apres echec", force=True):
                return None
            positions_data = await asyncio.to_thread(mt5.positions_get)
            if positions_data is None:
                self._mark_live_disconnected("Positions toujours indisponibles apres reconnexion.")
                return None

        self.is_connected = True
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

    async def get_mtf_candles(self, symbol: str, timeframes: list[int] = [5, 15, 60, 1440], count: int = 100) -> dict[int, list[dict]]:
        """
        RÃ©cupÃ¨re les bougies OMNI-STATE (Multi-Timeframe) synchronisÃ©es.
        Renvoie un dictionnaire: {5: [candles_m5], 15: [candles_m15], 60: [candles_h1], 1440: [candles_d1]}
        """
        import random
        from datetime import datetime
        
        tf_map = {
            1: 1 if self.mock_mode else mt5.TIMEFRAME_M1, 
            5: 5 if self.mock_mode else mt5.TIMEFRAME_M5, 
            15: 15 if self.mock_mode else mt5.TIMEFRAME_M15,
            60: 60 if self.mock_mode else mt5.TIMEFRAME_H1,
            1440: 1440 if self.mock_mode else mt5.TIMEFRAME_D1,
            10080: 10080 if self.mock_mode else mt5.TIMEFRAME_W1,
        }
        
        result = {}

        if (
            not self.mock_mode
            and not self.is_connected
            and not await self._ensure_live_connection(f"lecture des bougies {symbol}")
        ):
            self._log_offline_warning("MT5: bougies indisponibles pour %s car la connexion est hors ligne.", symbol)
            return {tf: [] for tf in timeframes}
        
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

        if not self.is_connected and not await self._ensure_live_connection(f"lecture du tick {symbol}"):
            return {"success": False, "message": f"MT5 hors ligne pour {symbol}"}

        tick = await asyncio.to_thread(mt5.symbol_info_tick, symbol)
        if tick is None:
            self.is_connected = False
            await self._ensure_live_connection(f"lecture du tick {symbol} apres echec", force=True)
            return {"success": False, "message": f"Dernier tick non disponible pour {symbol}"}

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "time": tick.time
        }

    async def get_symbol_volume_constraints(self, symbol: str) -> dict[str, Decimal]:
        """
        Retourne les contraintes de volume imposees par le broker pour un symbole.

        Args:
            symbol (str): Symbole MT5 a inspecter.

        Returns:
            dict[str, Decimal]: Bornes `min`, `step` et `max` du symbole.
        """
        if self.mock_mode:
            return {
                "min": Decimal("0.01"),
                "step": Decimal("0.01"),
                "max": Decimal("1.00"),
            }
        if not self.is_connected and not await self._ensure_live_connection(f"lecture des contraintes {symbol}"):
            return {
                "min": Decimal("0.01"),
                "step": Decimal("0.01"),
                "max": Decimal("1.00"),
            }

        symbol_info = await asyncio.to_thread(mt5.symbol_info, symbol)
        if symbol_info is None:
            logger.warning(
                "MT5: contraintes de volume indisponibles pour %s, repli sur les bornes par defaut.",
                symbol,
            )
            return {
                "min": Decimal("0.01"),
                "step": Decimal("0.01"),
                "max": Decimal("1.00"),
            }

        volume_min = Decimal(str(getattr(symbol_info, "volume_min", 0.01) or 0.01))
        volume_step = Decimal(str(getattr(symbol_info, "volume_step", 0.01) or 0.01))
        volume_max = Decimal(str(getattr(symbol_info, "volume_max", volume_min) or volume_min))

        if volume_step <= 0:
            volume_step = Decimal("0.01")
        if volume_max < volume_min:
            volume_max = volume_min

        return {
            "min": volume_min,
            "step": volume_step,
            "max": volume_max,
        }

    def _get_deviation(self, symbol: str) -> int:
        """Retourne la dÃ©viation (slippage) recommandÃ©e selon la volatilitÃ© de l'actif."""
        if any(v in symbol.upper() for v in ["XAU", "BTC", "US30", "US100", "GER40"]):
            return 50  # 5 pips pour les actifs volatils
        return 20  # 2 pips par dÃ©faut pour le Forex

    @staticmethod
    def _resolve_filling_mode(symbol_info: Any) -> int:
        """
        Selectionne un mode de remplissage compatible avec le broker.

        Args:
            symbol_info (Any): Meta-donnees du symbole MT5.

        Returns:
            int: Constante ``ORDER_FILLING_*`` exploitable par MT5.
        """
        filling_mode = int(getattr(symbol_info, "filling_mode", 0) or 0)
        supported_modes = [
            ("IOC", getattr(mt5, "ORDER_FILLING_IOC", None), getattr(mt5, "SYMBOL_FILLING_IOC", None)),
            ("RETURN", getattr(mt5, "ORDER_FILLING_RETURN", None), getattr(mt5, "SYMBOL_FILLING_RETURN", None)),
            ("FOK", getattr(mt5, "ORDER_FILLING_FOK", None), getattr(mt5, "SYMBOL_FILLING_FOK", None)),
        ]
        for _, order_mode, symbol_mode in supported_modes:
            if order_mode is None:
                continue
            if symbol_mode is None:
                return order_mode
            if filling_mode & symbol_mode:
                return order_mode
        return getattr(mt5, "ORDER_FILLING_IOC", 1)

    @staticmethod
    def _normalize_volume(symbol_info: Any, requested_volume: Decimal) -> Decimal:
        """
        Normalise un volume selon les bornes et le pas imposes par le broker.

        Args:
            symbol_info (Any): Meta-donnees du symbole MT5.
            requested_volume (Decimal): Volume initial demande par le moteur.

        Returns:
            Decimal: Volume compatible avec ``volume_min`` / ``volume_step`` / ``volume_max``.

        Raises:
            ValueError: Si les bornes du broker sont incoherentes.
        """
        volume_min = Decimal(str(getattr(symbol_info, "volume_min", 0.01) or 0.01))
        volume_step = Decimal(str(getattr(symbol_info, "volume_step", 0.01) or 0.01))
        volume_max = Decimal(str(getattr(symbol_info, "volume_max", volume_min) or volume_min))

        if volume_step <= 0:
            volume_step = Decimal("0.01")
        if volume_max < volume_min:
            raise ValueError("Les bornes de volume MT5 sont incoherentes.")

        bounded_volume = min(max(Decimal(str(requested_volume)), volume_min), volume_max)
        steps = ((bounded_volume - volume_min) / volume_step).to_integral_value(rounding=ROUND_FLOOR)
        normalized_volume = volume_min + (steps * volume_step)
        if normalized_volume < volume_min:
            normalized_volume = volume_min
        if normalized_volume > volume_max:
            normalized_volume = volume_max

        precision = max(0, -volume_step.normalize().as_tuple().exponent)
        quantum = Decimal("1").scaleb(-precision)
        return normalized_volume.quantize(quantum)

    @staticmethod
    def _is_order_check_valid(check_result: Any) -> bool:
        """
        Indique si ``order_check`` valide la requete courante.

        Args:
            check_result (Any): Reponse brute de ``mt5.order_check``.

        Returns:
            bool: ``True`` si la requete peut etre transmise a ``order_send``.
        """
        if check_result is None:
            return False
        retcode = getattr(check_result, "retcode", None)
        if retcode in (0, getattr(mt5, "TRADE_RETCODE_DONE", 10009)):
            return True
        comment = str(getattr(check_result, "comment", "") or "").strip().lower()
        return comment == "done"

    async def execute_order(self, order: TradeOrder) -> dict[str, Any]:
        """
        Exécute un ordre de trading avec garde-fous de marché et anti-doublon.

        Args:
            order (TradeOrder): Ordre à transmettre à MetaTrader 5.

        Returns:
            dict[str, Any]: Résultat d'exécution contenant au minimum
                `success` et `message`, ainsi que `ticket` ou `retcode`
                selon le cas.
        """
        signature = self._build_order_signature(order)
        duplicate_block = await self._register_order_attempt(signature)
        if duplicate_block is not None:
            return duplicate_block

        order_executed = False

        try:
            if self.mock_mode:
                result = await self._execute_mock_order(order)
                order_executed = bool(result.get("success"))
                return result

            if not self.is_connected and not await self._ensure_live_connection(f"execution ordre {order.symbol}"):
                return {
                    "success": False,
                    "message": "MT5 hors ligne: ordre refuse tant que la connexion n'est pas retablie.",
                    "retcode": 99010,
                }

            symbol_info = await asyncio.to_thread(mt5.symbol_info, order.symbol)
            if symbol_info is None:
                return {"success": False, "message": f"Symbole {order.symbol} non trouvÃ©"}

            deviation = self._get_deviation(order.symbol)
            point = getattr(symbol_info, "point", 0.0) or 1.0
            normalized_volume = self._normalize_volume(symbol_info, order.volume)
            filling_mode = self._resolve_filling_mode(symbol_info)
            if normalized_volume != Decimal(str(order.volume)):
                logger.info(
                    "Volume normalise pour %s: demande=%s, envoye=%s, min=%s, step=%s, max=%s",
                    order.symbol,
                    order.volume,
                    normalized_volume,
                    getattr(symbol_info, "volume_min", "?"),
                    getattr(symbol_info, "volume_step", "?"),
                    getattr(symbol_info, "volume_max", "?"),
                )

            # Retry Loop pour gÃ©rer les Requotes/Busy terminal
            for attempt in range(3):
                price = await asyncio.to_thread(mt5.symbol_info_tick, order.symbol)
                if price is None:
                    await asyncio.sleep(0.5)
                    continue

                # Le filtre de spread doit rester proche de l'envoi reel pour
                # eviter un ordre valide au calcul mais invalide au moment du deal.
                current_spread = (price.ask - price.bid) / point

                max_spread = 25
                sym = order.symbol.upper()
                if "XAU" in sym:
                    max_spread = 60
                elif "BTC" in sym or "ETH" in sym:
                    max_spread = 1500
                elif "US30" in sym or "US100" in sym or "GER40" in sym:
                    max_spread = 150

                if current_spread > max_spread:
                    logger.warning(
                        "Spread trop eleve sur %s: %.1f > %s. Ordre annule.",
                        order.symbol,
                        current_spread,
                        max_spread,
                    )
                    return {
                        "success": False,
                        "message": (
                            f"Echec securite: spread ({current_spread:.1f} pts) > "
                            f"max autorise ({max_spread} pts)."
                        ),
                        "retcode": 99999,
                    }

                order_type = mt5.ORDER_TYPE_BUY if order.action == TradeAction.BUY else mt5.ORDER_TYPE_SELL
                exec_price = price.ask if order.action == TradeAction.BUY else price.bid

                raw_comment = order.comment or "EVA"
                safe_comment = "".join(c for c in raw_comment if c.isalnum() or c in " -_.")[:31]
                if not safe_comment:
                    safe_comment = "EVA"

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": order.symbol,
                    "volume": float(normalized_volume),
                    "type": order_type,
                    "price": exec_price,
                    "sl": float(order.stop_loss_price) if order.stop_loss_price else 0.0,
                    "tp": float(order.take_profit_price) if order.take_profit_price else 0.0,
                    "deviation": deviation,
                    "magic": order.magic_number,
                    "comment": safe_comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }

                check_result = await asyncio.to_thread(mt5.order_check, request)
                if not self._is_order_check_valid(check_result):
                    check_comment = str(getattr(check_result, "comment", "") or "Pre-validation refusee").strip()
                    logger.warning(
                        "Pre-validation MT5 refusee pour %s %s: %s (volume=%s)",
                        order.symbol,
                        order.action.value,
                        check_comment,
                        normalized_volume,
                    )
                    return {
                        "success": False,
                        "message": (
                            "Pre-check MT5 refuse: "
                            f"{check_comment} | volume={normalized_volume} "
                            f"(min={getattr(symbol_info, 'volume_min', '?')}, "
                            f"step={getattr(symbol_info, 'volume_step', '?')}, "
                            f"max={getattr(symbol_info, 'volume_max', '?')})"
                        ),
                        "normalized_volume": float(normalized_volume),
                        "retcode": getattr(check_result, "retcode", 0),
                    }

                result = await asyncio.to_thread(mt5.order_send, request)

                if result is None:
                    logger.warning("MT5 order_send a retourne None pour %s (tentative %s)", order.symbol, attempt + 1)
                    await asyncio.sleep(0.5)
                    continue

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    order_executed = True
                    return {
                        "success": True,
                        "ticket": result.order,
                        "normalized_volume": float(normalized_volume),
                        "message": (
                            f"Ordre execute: {order.action.value} {normalized_volume} "
                            f"{order.symbol} (tentative {attempt + 1})"
                        ),
                    }

                if result.retcode in [10004, 10006, 10021, 10031]:
                    logger.info(
                        "Nouvelle tentative sur %s apres %s (retcode %s)",
                        order.symbol,
                        result.comment,
                        result.retcode,
                    )
                    if result.retcode == 10031:
                        logger.warning("Connexion MT5 perdue (10031). Reconnexion d'urgence.")
                        await self.connect()
                    await asyncio.sleep(1.0)
                    continue

                return {
                    "success": False,
                    "message": f"Erreur MT5: {result.comment}",
                    "normalized_volume": float(normalized_volume),
                    "retcode": result.retcode,
                }

            return {"success": False, "message": "Echec de l'ordre apres 3 tentatives (slippage/requotes)"}
        finally:
            await self._release_order_attempt(signature, remember_execution=order_executed)

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

        if not self.is_connected and not await self._ensure_live_connection(f"fermeture position {ticket}"):
            return {"success": False, "message": "MT5 hors ligne: fermeture impossible."}

        position = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if not position:
            return {"success": False, "message": f"Position {ticket} non trouvÃ©e"}

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        deviation = self._get_deviation(pos.symbol)
        symbol_info = await asyncio.to_thread(mt5.symbol_info, pos.symbol)
        filling_mode = self._resolve_filling_mode(symbol_info) if symbol_info else getattr(mt5, "ORDER_FILLING_IOC", 1)
        
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
                "type_filling": filling_mode,
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

        if not self.is_connected and not await self._ensure_live_connection(f"modification position {ticket}"):
            return {"success": False, "message": "MT5 hors ligne: modification impossible."}

        position = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if not position:
            return {"success": False, "message": f"Position {ticket} introuvable pour modification."}

        pos = position[0]
        current_sl = float(getattr(pos, "sl", 0.0) or 0.0)
        current_tp = float(getattr(pos, "tp", 0.0) or 0.0)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": float(sl) if sl > 0 else current_sl,
            "tp": float(tp) if tp > 0 else current_tp,
            "magic": getattr(pos, "magic", 0),
        }

        result = await asyncio.to_thread(mt5.order_send, request)
        if result is None:
            return {"success": False, "message": f"MT5 n'a retourne aucune reponse pour la modification de {ticket}."}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "message": f"Erreur modification: {result.comment}"}

        return {
            "success": True,
            "message": (
                f"Position {ticket} modifiee "
                f"SL={request['sl']:.5f} TP={request['tp']:.5f}"
            ),
        }

    async def get_margin_required(self, symbol: str, action: TradeAction, volume: float) -> Optional[float]:
        """Estime la marge requise pour un ordre (Sprint 13)."""
        if self.mock_mode:
            # Estimation pifomÃ©trique pour le mock
            return volume * 500.0  # $500 de marge par lot

        if not self.is_connected and not await self._ensure_live_connection(f"calcul de marge {symbol}"):
            self._log_offline_warning("Calcul de marge impossible sur %s: MT5 hors ligne.", symbol)
            return None
            
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

    async def get_deal_history(
        self,
        from_dt: datetime,
        to_dt: datetime,
        closed_only: bool = True,
    ) -> list[dict]:
        """
        Recupere l'historique des deals MT5 sur une periode.

        Args:
            from_dt (datetime): Debut de la plage temporelle.
            to_dt (datetime): Fin de la plage temporelle.
            closed_only (bool): Si True, ne retourne que les deals de sortie.

        Returns:
            list[dict]: Liste normalisee des deals MT5.
        """
        if self.mock_mode:
            return []

        if not self.is_connected and not await self._ensure_live_connection("lecture de l'historique MT5"):
            self._log_offline_warning("Historique MT5 indisponible: connexion hors ligne.")
            return []
        
        try:
            deals = await asyncio.to_thread(mt5.history_deals_get, from_dt, to_dt)
            if deals is None:
                return []
            
            entry_map = {
                0: "IN",
                1: "OUT",
                2: "INOUT",
                3: "OUT_BY",
            }
            result = []
            for deal in deals:
                if closed_only and deal.entry != 1:
                    continue

                result.append({
                    "ticket": deal.ticket,
                    "order": deal.order,
                    "position_id": deal.position_id,
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == 0 else "SELL",
                    "entry": deal.entry,
                    "entry_label": entry_map.get(deal.entry, f"UNKNOWN_{deal.entry}"),
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

    @staticmethod
    def _normalize_strategy_label(comment: str | None) -> dict[str, str]:
        """
        Normalise le commentaire MT5 pour l'analyse de performance.

        Args:
            comment (str | None): Commentaire brut issu du deal MT5.

        Returns:
            dict[str, str]: Libelle exact et famille de strategie associee.
        """
        raw_comment = str(comment or "").strip()
        if not raw_comment:
            return {"label": "Inconnu", "family": "Inconnu"}

        lowered = raw_comment.lower()
        if lowered.startswith("eva close"):
            return {"label": "Cloture systeme", "family": "Systeme"}
        if "muzero" in lowered:
            return {"label": raw_comment, "family": "MuZero JAX"}
        if "dreamer" in lowered:
            return {"label": raw_comment, "family": "Dreamer V3"}
        if "gnn" in lowered:
            return {"label": raw_comment, "family": "GNN"}
        if "manual" in lowered:
            return {"label": raw_comment, "family": "Manuel"}
        return {"label": raw_comment, "family": raw_comment[:32]}

    async def get_strategy_performance(
        self,
        from_dt: datetime,
        to_dt: datetime,
        limit: int = 5,
    ) -> dict[str, Any]:
        """
        Agrège les performances realisees par moteur de decision.

        La methode groupe les deals par ``position_id`` afin de reconstruire
        chaque trade ferme, puis consolide le PnL net par commentaire
        d'execution et par famille de strategie.

        Args:
            from_dt (datetime): Debut de la plage d'analyse.
            to_dt (datetime): Fin de la plage d'analyse.
            limit (int): Nombre maximal de strategies retournees.

        Returns:
            dict[str, Any]: Resume global et details agreges par strategie.
        """
        if self.mock_mode:
            return {
                "summary": {
                    "closed_trades": 0,
                    "net_profit": 0.0,
                    "win_rate": 0.0,
                    "from": from_dt.isoformat(),
                    "to": to_dt.isoformat(),
                },
                "by_model": [],
                "by_family": [],
                "recent_trades": [],
            }

        deals = await self.get_deal_history(from_dt, to_dt, closed_only=False)
        grouped_positions: dict[int, dict[str, Any]] = {}

        for deal in sorted(deals, key=lambda item: item["time"]):
            position_id = int(deal.get("position_id") or deal.get("order") or deal.get("ticket") or 0)
            if position_id <= 0:
                continue

            trade_state = grouped_positions.setdefault(
                position_id,
                {
                    "position_id": position_id,
                    "symbol": deal.get("symbol"),
                    "action": None,
                    "entry_time": None,
                    "close_time": None,
                    "entry_price": None,
                    "exit_price": None,
                    "entry_comment": "",
                    "exit_comment": "",
                    "volume": 0.0,
                    "profit": 0.0,
                    "swap": 0.0,
                    "commission": 0.0,
                    "magic": deal.get("magic", 0),
                },
            )

            entry = int(deal.get("entry", -1))
            trade_state["symbol"] = trade_state["symbol"] or deal.get("symbol")
            trade_state["volume"] = max(float(trade_state["volume"]), float(deal.get("volume") or 0.0))
            trade_state["profit"] += float(deal.get("profit") or 0.0)
            trade_state["swap"] += float(deal.get("swap") or 0.0)
            trade_state["commission"] += float(deal.get("commission") or 0.0)

            if entry in (0, 2):
                if trade_state["entry_time"] is None:
                    trade_state["entry_time"] = deal["time"]
                trade_state["entry_price"] = trade_state["entry_price"] or float(deal.get("price") or 0.0)
                trade_state["action"] = trade_state["action"] or deal.get("type")
                if deal.get("comment"):
                    trade_state["entry_comment"] = str(deal["comment"])

            if entry in (1, 2, 3):
                trade_state["close_time"] = deal["time"]
                trade_state["exit_price"] = float(deal.get("price") or 0.0)
                if deal.get("comment"):
                    trade_state["exit_comment"] = str(deal["comment"])

        closed_trades: list[dict[str, Any]] = []
        for trade_state in grouped_positions.values():
            if trade_state["close_time"] is None:
                continue

            strategy_comment = trade_state["entry_comment"] or trade_state["exit_comment"]
            if not strategy_comment and trade_state["magic"]:
                strategy_comment = f"Magic {trade_state['magic']}"

            strategy_info = self._normalize_strategy_label(strategy_comment)
            net_profit = (
                float(trade_state["profit"])
                + float(trade_state["swap"])
                + float(trade_state["commission"])
            )
            closed_trades.append(
                {
                    "position_id": trade_state["position_id"],
                    "symbol": trade_state["symbol"],
                    "action": trade_state["action"] or "UNKNOWN",
                    "label": strategy_info["label"],
                    "family": strategy_info["family"],
                    "entry_time": trade_state["entry_time"].isoformat() if trade_state["entry_time"] else None,
                    "close_time": trade_state["close_time"].isoformat() if trade_state["close_time"] else None,
                    "entry_price": trade_state["entry_price"],
                    "exit_price": trade_state["exit_price"],
                    "volume": float(trade_state["volume"]),
                    "net_profit": net_profit,
                    "gross_profit": float(trade_state["profit"]),
                    "swap": float(trade_state["swap"]),
                    "commission": float(trade_state["commission"]),
                    "magic": trade_state["magic"],
                }
            )

        def aggregate_by(key_name: str) -> list[dict[str, Any]]:
            buckets: dict[str, dict[str, Any]] = {}
            for trade in closed_trades:
                bucket_key = str(trade[key_name] or "Inconnu")
                bucket = buckets.setdefault(
                    bucket_key,
                    {
                        "label": bucket_key,
                        "closed_trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "net_profit": 0.0,
                        "gross_profit": 0.0,
                        "symbols": set(),
                        "last_closed_at": None,
                    },
                )
                bucket["closed_trades"] += 1
                bucket["net_profit"] += float(trade["net_profit"])
                bucket["gross_profit"] += float(trade["gross_profit"])
                if float(trade["net_profit"]) > 0:
                    bucket["wins"] += 1
                elif float(trade["net_profit"]) < 0:
                    bucket["losses"] += 1
                bucket["symbols"].add(str(trade["symbol"]))
                if trade["close_time"]:
                    bucket["last_closed_at"] = max(
                        bucket["last_closed_at"] or trade["close_time"],
                        trade["close_time"],
                    )

            ranked = []
            for bucket in buckets.values():
                closed_count = bucket["closed_trades"]
                ranked.append(
                    {
                        "label": bucket["label"],
                        "closed_trades": closed_count,
                        "wins": bucket["wins"],
                        "losses": bucket["losses"],
                        "win_rate": round((bucket["wins"] / closed_count) * 100.0, 2) if closed_count else 0.0,
                        "net_profit": round(bucket["net_profit"], 2),
                        "avg_profit": round(bucket["net_profit"] / closed_count, 2) if closed_count else 0.0,
                        "gross_profit": round(bucket["gross_profit"], 2),
                        "symbols": sorted(bucket["symbols"]),
                        "last_closed_at": bucket["last_closed_at"],
                    }
                )
            ranked.sort(key=lambda item: (item["net_profit"], item["closed_trades"]), reverse=True)
            return ranked[: max(1, limit)]

        closed_count = len(closed_trades)
        wins = sum(1 for trade in closed_trades if float(trade["net_profit"]) > 0)
        net_profit = round(sum(float(trade["net_profit"]) for trade in closed_trades), 2)
        recent_trades = sorted(
            closed_trades,
            key=lambda item: item["close_time"] or "",
            reverse=True,
        )[:10]

        return {
            "summary": {
                "closed_trades": closed_count,
                "wins": wins,
                "losses": max(0, closed_count - wins),
                "win_rate": round((wins / closed_count) * 100.0, 2) if closed_count else 0.0,
                "net_profit": net_profit,
                "from": from_dt.isoformat(),
                "to": to_dt.isoformat(),
            },
            "by_model": aggregate_by("label"),
            "by_family": aggregate_by("family"),
            "recent_trades": recent_trades,
        }

    async def get_candles_range(
        self,
        symbol: str,
        timeframe: int,
        from_dt: datetime,
        to_dt: datetime,
    ) -> list[dict[str, Any]]:
        """
        Recupere des bougies sur une plage temporelle explicite.

        Args:
            symbol (str): Symbole a charger.
            timeframe (int): Timeframe MT5 simplifie (1, 5, 15, 60, 1440).
            from_dt (datetime): Debut de la plage.
            to_dt (datetime): Fin de la plage.

        Returns:
            list[dict[str, Any]]: Bougies OHLCV triees chronologiquement.
        """
        if self.mock_mode:
            return []

        tf_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            60: mt5.TIMEFRAME_H1,
            1440: mt5.TIMEFRAME_D1,
            10080: mt5.TIMEFRAME_W1,
        }
        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            raise ValueError(f"Timeframe non supporte pour MT5: {timeframe}")

        if not await self.ensure_symbol_selected(symbol):
            return []

        try:
            rates = await asyncio.to_thread(
                mt5.copy_rates_range,
                symbol,
                mt5_tf,
                from_dt,
                to_dt,
            )
            if rates is None or len(rates) == 0:
                return []

            candles: list[dict[str, Any]] = []
            for rate in rates:
                candles.append(
                    {
                        "time": datetime.fromtimestamp(rate["time"]),
                        "open": float(rate["open"]),
                        "high": float(rate["high"]),
                        "low": float(rate["low"]),
                        "close": float(rate["close"]),
                        "tick_volume": float(rate["tick_volume"]),
                        "spread": float(rate["spread"]),
                    }
                )
            return candles
        except Exception as exc:
            logger.error("Erreur recuperation bougies %s sur plage: %s", symbol, exc)
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
        terminal_path=settings.mt5_terminal_path,
        terminal_portable=settings.mt5_terminal_portable,
        terminal_timeout_ms=settings.mt5_terminal_timeout_ms,
    )
