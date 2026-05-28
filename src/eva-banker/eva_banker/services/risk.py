"""
Service de validation des risques pour The Banker.

Ce module applique les garde-fous de la Constitution (Loi 2) et centralise
les regles de session de marche utilisees par l'automate de trading.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_FLOOR
from functools import lru_cache
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from shared import Position, RiskStatus, TradeOrder, calculate_var, get_settings

logger = logging.getLogger(__name__)


def resolve_effective_max_open_positions(settings: Any) -> int:
    """
    Calcule le plafond de positions effectivement applique.

    Les followers ont besoin d'un plafond plus haut que le master, car ils
    conservent un reliquat de position apres les clotures gagnantes et peuvent
    donc empiler plusieurs runners simultanement.

    Args:
        settings (Any): Objet de configuration pydantic ou equivalent.

    Returns:
        int: Nombre maximal de positions a autoriser pour cette instance.
    """
    master_limit = int(getattr(settings, "risk_max_open_positions", 3) or 3)
    if bool(getattr(settings, "banker_follower_mode", False)):
        follower_limit = int(
            getattr(settings, "risk_follower_max_open_positions", master_limit)
            or master_limit
        )
        return max(master_limit, follower_limit)
    return master_limit


class RiskValidator:
    """
    Valide les ordres selon les limites de risque du projet.

    Le validateur controle notamment:
    - le risque maximal par trade ;
    - les drawdowns journalier et total ;
    - l'anti-tilt ;
    - la disponibilite de la session de marche.
    """

    def __init__(
        self,
        max_risk_per_trade: Decimal = Decimal("1.0"),
        max_daily_drawdown: Decimal = Decimal("4.0"),
        max_total_drawdown: Decimal = Decimal("8.0"),
        max_open_positions: int = 3,
        anti_tilt_losses: int = 2,
        anti_tilt_hours: int = 24,
        max_daily_profit: Decimal = Decimal("2.0"),
        giveback_activation: Decimal = Decimal("1.0"),
        giveback_tolerance: Decimal = Decimal("0.5"),
    ) -> None:
        """
        Initialise le validateur.

        Args:
            max_risk_per_trade (Decimal): Risque maximal autorise par trade.
            max_daily_drawdown (Decimal): Drawdown journalier maximal.
            max_total_drawdown (Decimal): Drawdown total maximal.
            max_open_positions (int): Nombre maximal de positions ouvertes.
            anti_tilt_losses (int): Nombre de pertes consecutives avant pause.
            anti_tilt_hours (int): Duree de la pause anti-tilt en heures.
            max_daily_profit (Decimal): Seuil cible de profit quotidien.
            giveback_activation (Decimal): Seuil d'activation du trailing profit.
            giveback_tolerance (Decimal): Amortisseur de baisse autorisée depuis le pic.
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.max_open_positions = max_open_positions
        self.anti_tilt_losses = anti_tilt_losses
        self.anti_tilt_hours = anti_tilt_hours
        self.max_daily_profit = max_daily_profit
        self.giveback_activation = giveback_activation
        self.giveback_tolerance = giveback_tolerance
        self.settings = get_settings()

        self._consecutive_losses = 0
        self._anti_tilt_until: datetime | None = None
        self._daily_pnl = Decimal("0")
        self._daily_pnl_peak = Decimal("0")
        self._daily_profit_locked = False
        self._daily_giveback_locked = False
        self._total_pnl = Decimal("0")
        self._open_positions_count = 0
        self._total_positions_count = 0
        self._hold_positions_count = 0
        self._ignored_positions_count = 0
        self._open_positions_pnl = Decimal("0")
        self._account_balance = Decimal("100000")
        self._session_log_states: dict[str, str] = {}
        self._symbol_asset_classes: dict[str, str] = {}

        logger.info(
            "RiskValidator initialise: max_risk=%s%%, max_dd_daily=%s%%, max_dd_total=%s%%",
            max_risk_per_trade,
            max_daily_drawdown,
            max_total_drawdown,
        )

    @staticmethod
    def _normalize_symbol_token(symbol: str) -> str:
        """
        Normalise un symbole pour les heuristiques de risque.

        Cette normalisation retire les suffixes brokers les plus courants
        afin de raisonner sur le sous-jacent reel (`BTCUSD.e` -> `BTCUSD`,
        `DE40.e` -> `DE40`, `US30.cash` -> `US30`).

        Args:
            symbol (str): Symbole brut a normaliser.

        Returns:
            str: Symbole en majuscules sans suffixe broker.
        """
        normalized_symbol = str(symbol or "").strip().upper()
        for suffix in (".CASH", ".E", ".M"):
            if normalized_symbol.endswith(suffix):
                normalized_symbol = normalized_symbol[: -len(suffix)]
        return normalized_symbol

    async def validate_order(self, order: TradeOrder) -> dict[str, Any]:
        """
        Valide un ordre selon les regles de risque.

        Args:
            order (TradeOrder): Ordre a verifier.

        Returns:
            dict[str, Any]: Resultat de validation et details des controles.
        """
        result = {
            "allowed": True,
            "reason": None,
            "risk_percent": Decimal("0"),
            "checks": [],
        }

        if order.stop_loss_price is None:
            result["allowed"] = False
            result["reason"] = "SL obligatoire (ROE Trading)"
            result["checks"].append(("stop_loss", False, "Stop Loss manquant"))
            return result
        result["checks"].append(("stop_loss", True, "Stop Loss present"))

        if not self.is_within_trading_session(order.symbol):
            result["allowed"] = False
            result["reason"] = "Session de trading fermee (week-end ou rollover)"
            result["checks"].append(("session", False, result["reason"]))
            return result
        result["checks"].append(("session", True, "Session de trading ouverte"))

        risk_percent = self._calculate_risk_percent(order)
        result["risk_percent"] = risk_percent

        if risk_percent > self.max_risk_per_trade:
            result["allowed"] = False
            result["reason"] = (
                f"Risque {risk_percent:.2f}% > max {self.max_risk_per_trade}%"
            )
            result["checks"].append(("risk_per_trade", False, result["reason"]))
            return result
        result["checks"].append(
            ("risk_per_trade", True, f"Risque {risk_percent:.2f}% OK")
        )

        if self._is_anti_tilt_active():
            result["allowed"] = False
            result["reason"] = f"Anti-Tilt actif jusqu'a {self._anti_tilt_until}"
            result["checks"].append(("anti_tilt", False, result["reason"]))
            return result
        result["checks"].append(("anti_tilt", True, "Anti-Tilt inactif"))

        if self._daily_profit_locked:
            result["allowed"] = False
            result["reason"] = "Trading bloque: cible de profit journalier atteinte"
            result["checks"].append(("daily_profit_lock", False, result["reason"]))
            return result
        result["checks"].append(("daily_profit_lock", True, "Verrou de profit journalier inactif"))

        if self._daily_giveback_locked:
            result["allowed"] = False
            result["reason"] = "Trading bloque: protection anti-giveback declenchee"
            result["checks"].append(("daily_giveback_lock", False, result["reason"]))
            return result
        result["checks"].append(("daily_giveback_lock", True, "Verrou anti-giveback inactif"))

        if self._get_daily_drawdown_percent() >= self.max_daily_drawdown:
            result["allowed"] = False
            result["reason"] = (
                f"Drawdown journalier limite atteinte ({self.max_daily_drawdown}%)"
            )
            result["checks"].append(("daily_drawdown", False, result["reason"]))
            return result
        result["checks"].append(
            (
                "daily_drawdown",
                True,
                f"DD journalier {self._get_daily_drawdown_percent():.2f}% OK",
            )
        )

        if self._get_total_drawdown_percent() >= self.max_total_drawdown:
            result["allowed"] = False
            result["reason"] = (
                f"Drawdown total limite atteinte ({self.max_total_drawdown}%)"
            )
            result["checks"].append(("total_drawdown", False, result["reason"]))
            return result
        result["checks"].append(
            (
                "total_drawdown",
                True,
                f"DD total {self._get_total_drawdown_percent():.2f}% OK",
            )
        )

        if self._open_positions_count >= self.max_open_positions:
            result["allowed"] = False
            result["reason"] = f"Max positions atteint ({self.max_open_positions})"
            result["checks"].append(("max_positions", False, result["reason"]))
            return result
        result["checks"].append(
            (
                "max_positions",
                True,
                f"Positions {self._open_positions_count}/{self.max_open_positions}",
            )
        )

        mock_returns = [0.001, -0.002, 0.005, -0.01, 0.002]
        var = calculate_var(mock_returns)
        if var < -0.025:
            result["allowed"] = False
            result["reason"] = f"VaR trop elevee ({var:.4f}). Marche instable."
            result["checks"].append(("var_check", False, result["reason"]))
            return result
        result["checks"].append(("var_check", True, f"VaR OK ({var:.4f})"))

        logger.info(
            "Ordre valide: %s %s - risque %s%%",
            order.symbol,
            order.action.value,
            risk_percent,
        )
        return result

    @staticmethod
    def is_hold_position(position: Position) -> bool:
        """
        Indique si une position doit etre consideree comme un runner HOLD.

        Une position HOLD est un reliquat protege au break-even ou mieux.
        Elle ne consomme plus le meme budget de risque qu'une entree fraiche,
        et ne doit donc plus compter dans le plafond d'ouvertures simultanees.

        Args:
            position (Position): Position ouverte a evaluer.

        Returns:
            bool: True si la position est protegee au break-even ou mieux.
        """
        if position is None:
            return False

        comment = str(getattr(position, "comment", "") or "").strip().upper()
        if "HOLD" in comment:
            return True
        if comment.startswith("EVA CLOSE") or comment.startswith("CLAUSE"):
            return True

        stop_loss = getattr(position, "stop_loss", None)
        if stop_loss is None:
            return False

        open_price = Decimal(str(getattr(position, "open_price", "0") or "0"))
        stop_loss_price = Decimal(str(stop_loss))
        if open_price <= 0:
            return False

        if position.action == position.action.BUY:
            return stop_loss_price >= open_price
        if position.action == position.action.SELL:
            return stop_loss_price <= open_price
        return False

    def is_ignored_position(self, position: Position) -> bool:
        """
        Indique si une position ouverte doit etre ignoree du plafond EVA.

        Ce garde-fou sert aux comptes followers qui contiennent deja des
        positions manuelles historiques. Quand le flag runtime est actif, seules
        les positions sans commentaire MT5 sont ignorees ; les positions COPY,
        EVA Close et HOLD restent pilotees par EVA.

        Args:
            position (Position): Position ouverte a evaluer.

        Returns:
            bool: True si la position ne doit pas consommer le plafond EVA.
        """
        if position is None:
            return False
        if not bool(getattr(self.settings, "risk_ignore_uncommented_positions", False)):
            return False
        comment = str(getattr(position, "comment", "") or "").strip()
        return comment == ""

    def summarize_positions(self, positions: list[Position]) -> dict[str, int]:
        """
        Calcule les compteurs de positions utilises par le module de risque.

        Args:
            positions (list[Position]): Positions ouvertes a analyser.

        Returns:
            dict[str, int]: Totaux `total`, `hold` et `counted`.
        """
        total_positions_count = len(positions or [])
        ignored_positions_count = sum(
            1 for position in (positions or []) if self.is_ignored_position(position)
        )
        hold_positions_count = sum(
            1
            for position in (positions or [])
            if not self.is_ignored_position(position) and self.is_hold_position(position)
        )
        counted_positions_count = max(
            0,
            total_positions_count - hold_positions_count - ignored_positions_count,
        )
        return {
            "total": total_positions_count,
            "hold": hold_positions_count,
            "ignored": ignored_positions_count,
            "counted": counted_positions_count,
        }

    def update_positions_snapshot(self, positions: list[Position]) -> None:
        """
        Met a jour les compteurs a partir d'un snapshot complet de positions.

        Args:
            positions (list[Position]): Positions ouvertes courantes.
        """
        counters = self.summarize_positions(positions)
        self._total_positions_count = counters["total"]
        self._hold_positions_count = counters["hold"]
        self._ignored_positions_count = counters["ignored"]
        self._open_positions_count = counters["counted"]

        # Calculer le PnL latent total (Floating P&L) des positions actives non ignorees
        floating_pnl = Decimal("0")
        for pos in (positions or []):
            if not self.is_ignored_position(pos):
                floating_pnl += Decimal(str(pos.profit))
        self._open_positions_pnl = floating_pnl

    def register_symbol_universe(self, asset_classes: dict[str, str]) -> None:
        """
        Enregistre les classes d'actifs du dernier univers decouvert.

        Args:
            asset_classes (dict[str, str]): Mapping symbole -> classe d'actif.
        """
        self._symbol_asset_classes = {
            symbol.upper(): asset_class
            for symbol, asset_class in asset_classes.items()
            if asset_class
        }

    def _calculate_risk_percent(self, order: TradeOrder) -> Decimal:
        """
        Calcule le pourcentage de risque d'un ordre.

        Args:
            order (TradeOrder): Ordre a evaluer.

        Returns:
            Decimal: Pourcentage de risque estime.
        """
        if order.stop_loss_price is None:
            return Decimal("100")

        normalized_symbol = self._normalize_symbol_token(order.symbol)

        if order.entry_price is not None:
            current_price = Decimal(str(order.entry_price))
        else:
            mock_prices = {
                "XAUUSD": Decimal("4900.00"),
                "EURUSD": Decimal("1.0850"),
                "BTCUSD": Decimal("67000.00"),
                "ETHUSD": Decimal("3200.00"),
                "SOLUSD": Decimal("140.00"),
                "US30.CASH": Decimal("39000.00"),
                "GBPUSD": Decimal("1.2700"),
                "USDJPY": Decimal("150.00"),
                "AUDUSD": Decimal("0.6600"),
                "XAGUSD": Decimal("23.50"),
                "US100.CASH": Decimal("18000.00"),
                "GER40.CASH": Decimal("17000.00"),
                "US30": Decimal("39000.00"),
                "US100": Decimal("18000.00"),
                "GER40": Decimal("17000.00"),
                "DE40": Decimal("17000.00"),
                "US500": Decimal("5200.00"),
            }
            current_price = mock_prices.get(
                normalized_symbol,
                mock_prices.get(order.symbol.upper(), Decimal("100")),
            )

        sl_distance = abs(current_price - order.stop_loss_price)
        if sl_distance <= 0:
            return Decimal("100")
        contract_size = self._get_estimated_contract_size(order.symbol)
        potential_loss = sl_distance * Decimal(str(order.volume)) * contract_size

        if self._account_balance > 0:
            risk_percent = (potential_loss / self._account_balance) * 100
        else:
            risk_percent = Decimal("100")

        return risk_percent.quantize(Decimal("0.01"))

    def _get_estimated_contract_size(self, symbol: str) -> Decimal:
        """
        Estime la taille de contrat pour le calcul de risque.

        Args:
            symbol (str): Symbole financier.

        Returns:
            Decimal: Taille de contrat approximee.
        """
        symbol_upper = symbol.upper()
        normalized_symbol = self._normalize_symbol_token(symbol_upper)
        asset_class = self._symbol_asset_classes.get(symbol_upper) or self._symbol_asset_classes.get(
            normalized_symbol
        )

        if asset_class == "crypto" or self._is_crypto_symbol(normalized_symbol):
            return Decimal("1")
        if "XAU" in normalized_symbol:
            return Decimal("100")
        if (
            "US30" in normalized_symbol
            or "GER40" in normalized_symbol
            or "DE40" in normalized_symbol
            or "US100" in normalized_symbol
            or "USTEC" in normalized_symbol
            or "US500" in normalized_symbol
            or "CASH" in symbol_upper
        ):
            return Decimal("1")
        if "JPY" in normalized_symbol:
            return Decimal("1000")
        if "USD" in normalized_symbol or "EUR" in normalized_symbol:
            return Decimal("100000")
        return Decimal("10")

    def is_within_trading_session(self, symbol: str) -> bool:
        """
        Verifie si le symbole est tradable au moment courant.

        Regles appliquees:
        - les cryptos restent ouvertes 24/7 ;
        - tous les autres actifs sont bloques le samedi et le dimanche ;
        - tous les autres actifs sont bloques de 23:00 a 00:59 heure de Paris.

        Args:
            symbol (str): Symbole a verifier.

        Returns:
            bool: True si la session est ouverte, sinon False.
        """
        symbol_upper = symbol.upper()
        normalized_symbol = self._normalize_symbol_token(symbol_upper)
        asset_class = self._symbol_asset_classes.get(symbol_upper) or self._symbol_asset_classes.get(
            normalized_symbol
        )
        if asset_class == "crypto" or self._is_crypto_symbol(normalized_symbol):
            self._session_log_states.pop(symbol_upper, None)
            return True

        now = self._get_market_now()
        if now.weekday() >= 5:
            self._log_session_block(
                symbol_upper,
                "weekend",
                f"Filtre de session: trading bloque sur {symbol} (week-end).",
            )
            return False

        if now.hour in {23, 0}:
            self._log_session_block(
                symbol_upper,
                "rollover",
                f"Filtre de session: trading bloque sur {symbol} (rollover).",
            )
            return False

        self._session_log_states.pop(symbol_upper, None)
        return True

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """
        Detecte heuristiquement un symbole crypto quand le metadata manque.

        Args:
            symbol (str): Symbole normalise.

        Returns:
            bool: True si le symbole ressemble a une paire crypto.
        """
        clean = "".join(char for char in symbol.upper() if char.isalnum())
        fiat_codes = {
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
        crypto_bases = {
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
            "XRP",
            "XBT",
        }
        quotes = ("USDT", "USDC", "USD", "EUR", "BTC", "ETH")

        for quote in quotes:
            if clean.endswith(quote) and len(clean) > len(quote):
                base = clean[: -len(quote)]
                if base in crypto_bases and base not in fiat_codes:
                    return True
        return False

    def _get_market_now(self) -> datetime:
        """
        Retourne l'heure de reference des sessions de marche.

        Returns:
            datetime: Heure locale dans le fuseau du projet.
        """
        timezone_name = getattr(self.settings, "timezone", "Europe/Paris")
        try:
            return datetime.now(ZoneInfo(timezone_name))
        except Exception:
            return datetime.now()

    def _log_session_block(self, symbol: str, state: str, message: str) -> None:
        """
        Evite de spammer les logs quand un symbole reste bloque.

        Args:
            symbol (str): Symbole normalise.
            state (str): Motif de blocage.
            message (str): Message a journaliser.
        """
        if self._session_log_states.get(symbol) == state:
            return

        self._session_log_states[symbol] = state
        logger.warning(message)

    def _is_anti_tilt_active(self) -> bool:
        """
        Indique si l'anti-tilt est actif.

        Returns:
            bool: True si la pause anti-tilt est en cours.
        """
        if self._anti_tilt_until is None:
            return False
        return datetime.now() < self._anti_tilt_until

    def _get_daily_drawdown_percent(self) -> Decimal:
        """
        Retourne le drawdown journalier en pourcentage.

        Returns:
            Decimal: Drawdown journalier.
        """
        if self._account_balance <= 0:
            return Decimal("0")
        combined_pnl = self._daily_pnl + self._open_positions_pnl
        if combined_pnl >= 0:
            return Decimal("0")
        return (abs(combined_pnl) / self._account_balance * 100).quantize(
            Decimal("0.01")
        )

    def _get_total_drawdown_percent(self) -> Decimal:
        """
        Retourne le drawdown total en pourcentage.

        Returns:
            Decimal: Drawdown total.
        """
        if self._account_balance <= 0:
            return Decimal("0")
        combined_total_pnl = self._total_pnl + self._open_positions_pnl
        if combined_total_pnl >= 0:
            return Decimal("0")
        return (abs(combined_total_pnl) / self._account_balance * 100).quantize(
            Decimal("0.01")
        )

    def record_trade_result(self, profit: Decimal) -> None:
        """
        Enregistre le resultat d'un trade.

        Args:
            profit (Decimal): Profit ou perte du trade.
        """
        self._daily_pnl += profit
        self._total_pnl += profit

        if self._daily_pnl > self._daily_pnl_peak:
            self._daily_pnl_peak = self._daily_pnl

        if profit < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.anti_tilt_losses:
                self._activate_anti_tilt()
        else:
            self._consecutive_losses = 0

    def check_gains_protection(self, open_positions_pnl: Decimal) -> dict[str, Any]:
        """
        Evalue les verrous de gain (profit target et anti-giveback).

        Cette methode prend en compte le profit deja realise sur la journee (ferme)
        ainsi que les gains/pertes latents (ouverts) en cours.

        Args:
            open_positions_pnl (Decimal): PnL latent total des positions ouvertes.

        Returns:
            dict[str, Any]: Statut des verrous de gain.
        """
        combined_pnl = self._daily_pnl + open_positions_pnl
        
        # Mise a jour du sommet de profit combiné de la journee
        if combined_pnl > self._daily_pnl_peak:
            self._daily_pnl_peak = combined_pnl

        balance = self._account_balance if self._account_balance > 0 else Decimal("100000")
        pnl_percent = (combined_pnl / balance * 100).quantize(Decimal("0.01"))
        peak_percent = (self._daily_pnl_peak / balance * 100).quantize(Decimal("0.01"))

        # 1. Controle du Verrou de Profit Target Lock-in
        if not self._daily_profit_locked and pnl_percent >= self.max_daily_profit:
            self._daily_profit_locked = True
            logger.warning(
                "TARGET PROFIT JOURNALIER ATTEINT (+%s%% >= +%s%%). Verrouillage active.",
                pnl_percent,
                self.max_daily_profit,
            )
            return {
                "lock_triggered": True,
                "reason": "daily_profit_target_reached",
                "combined_pnl_percent": pnl_percent,
                "peak_percent": peak_percent,
            }

        # 2. Controle de la Protection Anti-Giveback
        if not self._daily_giveback_locked and peak_percent >= self.giveback_activation:
            giveback_drop = peak_percent - pnl_percent
            if giveback_drop >= self.giveback_tolerance:
                self._daily_giveback_locked = True
                logger.warning(
                    "PROTECTION ANTI-GIVEBACK DECLENCHEE (Pic: +%s%% | Courant: +%s%% | Drop: %s%% >= tolerance %s%%). Verrouillage active.",
                    peak_percent,
                    pnl_percent,
                    giveback_drop,
                    self.giveback_tolerance,
                )
                return {
                    "lock_triggered": True,
                    "reason": "daily_giveback_triggered",
                    "combined_pnl_percent": pnl_percent,
                    "peak_percent": peak_percent,
                    "giveback_drop_percent": giveback_drop,
                }

        return {
            "lock_triggered": self._daily_profit_locked or self._daily_giveback_locked,
            "reason": "locked" if (self._daily_profit_locked or self._daily_giveback_locked) else "none",
            "combined_pnl_percent": pnl_percent,
            "peak_percent": peak_percent,
        }

    def _activate_anti_tilt(self) -> None:
        """Active le mode anti-tilt."""
        self._anti_tilt_until = datetime.now() + timedelta(hours=self.anti_tilt_hours)
        logger.warning("ANTI-TILT active jusqu'a %s", self._anti_tilt_until)

    def update_positions_count(self, count: int) -> None:
        """
        Met a jour le nombre de positions ouvertes.

        Args:
            count (int): Nombre courant de positions.
        """
        self._open_positions_count = count
        self._total_positions_count = count
        self._hold_positions_count = 0
        self._ignored_positions_count = 0

    def get_counted_open_positions(self) -> int:
        """
        Retourne le nombre de positions qui consomment encore du risque.

        Returns:
            int: Nombre de positions comptees contre le plafond de risque.
        """
        return self._open_positions_count

    def get_total_open_positions(self) -> int:
        """
        Retourne le nombre total de positions ouvertes.

        Returns:
            int: Nombre total de positions visibles sur le compte.
        """
        return self._total_positions_count

    def get_hold_positions_count(self) -> int:
        """
        Retourne le nombre de runners proteges exclus du plafond de risque.

        Returns:
            int: Nombre de positions HOLD.
        """
        return self._hold_positions_count

    def get_ignored_positions_count(self) -> int:
        """
        Retourne le nombre de positions ouvertes ignorees du plafond EVA.

        Returns:
            int: Nombre de positions sans commentaire ignorees.
        """
        return self._ignored_positions_count

    def update_account_balance(self, balance: Decimal) -> None:
        """
        Met a jour le solde de reference du compte.

        Args:
            balance (Decimal): Solde ou equity de reference.
        """
        self._account_balance = balance

    def calculate_lot_size(
        self,
        balance: Decimal,
        risk_percent: Decimal,
        sl_distance: Decimal,
        symbol: str = "EURUSD",
        sizing_hint: dict[str, Decimal] | None = None,
    ) -> float:
        """
        Calcule la taille de lot optimale.

        Args:
            balance (Decimal): Solde disponible.
            risk_percent (Decimal): Risque cible en pourcentage.
            sl_distance (Decimal): Distance au stop loss.
            symbol (str): Symbole traite.
            sizing_hint (dict[str, Decimal] | None): Metadonnees MT5 utiles
                au sizing (`tick_size`, `tick_value`, `volume_min`,
                `volume_step`, `volume_max`). Si absentes, un repli
                heuristique est applique.

        Returns:
            float: Taille de lot arrondie et securisee.
        """
        if sl_distance <= 0 or balance <= 0 or risk_percent <= 0:
            return 0.0

        risk_amount = balance * (risk_percent / Decimal("100"))
        symbol_upper = symbol.upper()
        asset_class = self._symbol_asset_classes.get(symbol_upper)
        sizing_hint = sizing_hint or {}
        tick_size = Decimal(str(sizing_hint.get("tick_size", Decimal("0")) or Decimal("0")))
        tick_value = Decimal(str(sizing_hint.get("tick_value", Decimal("0")) or Decimal("0")))
        volume_min = Decimal(str(sizing_hint.get("volume_min", Decimal("0.01")) or Decimal("0.01")))
        volume_step = Decimal(str(sizing_hint.get("volume_step", Decimal("0.01")) or Decimal("0.01")))
        volume_max = Decimal(str(sizing_hint.get("volume_max", Decimal("100.0")) or Decimal("100.0")))

        loss_per_lot: Decimal
        if tick_size > 0 and tick_value > 0:
            ticks_to_stop = sl_distance / tick_size
            loss_per_lot = ticks_to_stop * tick_value
        else:
            # Repli heuristique si le broker ne fournit pas les economics
            # du symbole via l'API Python MT5.
            if asset_class == "crypto" or self._is_crypto_symbol(symbol_upper):
                point_value = Decimal("1")
            elif "XAU" in symbol_upper:
                point_value = Decimal("100")
            elif any(idx in symbol_upper for idx in ["US30", "US100", "GER40", "CASH"]):
                point_value = Decimal("1")
            else:
                point_value = Decimal("100000")
            loss_per_lot = sl_distance * point_value

        try:
            raw_lots = risk_amount / loss_per_lot
            if raw_lots <= 0:
                return 0.0

            if volume_step <= 0:
                volume_step = Decimal("0.01")
            if volume_max < volume_min:
                volume_max = volume_min

            precision = max(0, -volume_step.normalize().as_tuple().exponent)
            quantum = Decimal("1").scaleb(-precision)

            if raw_lots >= volume_min:
                steps = ((raw_lots - volume_min) / volume_step).to_integral_value(
                    rounding=ROUND_FLOOR
                )
                normalized_lots = volume_min + (steps * volume_step)
            else:
                # On conserve la valeur brute arrondie pour que le garde-fou
                # "minimum broker" puisse journaliser un veto explicite.
                normalized_lots = raw_lots.quantize(quantum)

            normalized_lots = min(max(normalized_lots, Decimal("0")), volume_max)
            return float(normalized_lots)
        except ZeroDivisionError:
            return 0.0

    async def get_current_status(self) -> RiskStatus:
        """
        Retourne l'etat courant du module de risque.

        Returns:
            RiskStatus: Etat consolide du validateur.
        """
        trading_allowed = (
            not self._is_anti_tilt_active()
            and not self._daily_profit_locked
            and not self._daily_giveback_locked
            and self._get_daily_drawdown_percent() < self.max_daily_drawdown
            and self._get_total_drawdown_percent() < self.max_total_drawdown
            and self._open_positions_count < self.max_open_positions
        )
        return RiskStatus(
            account_id=uuid4(),
            daily_drawdown_percent=self._get_daily_drawdown_percent(),
            total_drawdown_percent=self._get_total_drawdown_percent(),
            open_positions_count=self._open_positions_count,
            total_positions_count=self._total_positions_count,
            hold_positions_count=self._hold_positions_count,
            ignored_positions_count=self._ignored_positions_count,
            anti_tilt_active=self._is_anti_tilt_active(),
            anti_tilt_expires_at=self._anti_tilt_until,
            trading_allowed=trading_allowed,
        )


@lru_cache
def get_risk_validator() -> RiskValidator:
    """
    Construit l'instance singleton du validateur de risques.

    Returns:
        RiskValidator: Instance configuree depuis les settings.
    """
    settings = get_settings()
    effective_max_open_positions = resolve_effective_max_open_positions(settings)
    return RiskValidator(
        max_risk_per_trade=Decimal(str(settings.risk_max_single_trade_percent)),
        max_daily_drawdown=Decimal(str(settings.risk_max_daily_drawdown_percent)),
        max_total_drawdown=Decimal(str(settings.risk_max_total_drawdown_percent)),
        max_open_positions=effective_max_open_positions,
        anti_tilt_losses=settings.risk_anti_tilt_losses,
        anti_tilt_hours=settings.risk_anti_tilt_duration_hours,
        max_daily_profit=Decimal(str(settings.risk_max_daily_profit_percent)),
        giveback_activation=Decimal(str(settings.risk_daily_giveback_activation_percent)),
        giveback_tolerance=Decimal(str(settings.risk_daily_giveback_tolerance_percent)),
    )
