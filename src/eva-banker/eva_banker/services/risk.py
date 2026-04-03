"""
Service de validation des risques pour The Banker.

Ce module applique les garde-fous de la Constitution (Loi 2) et centralise
les regles de session de marche utilisees par l'automate de trading.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from shared import RiskStatus, TradeOrder, calculate_var, get_settings

logger = logging.getLogger(__name__)


class RiskValidator:
    """
    Valide les ordres selon les limites de risque du projet.

    Le validateur controle notamment:
    - le risque maximal par trade ;
    - les drawdowns journalier et total ;
    - le gouverneur de drawdown intraday sur l'equity ;
    - l'anti-tilt ;
    - la disponibilite de la session de marche.
    """

    def __init__(
        self,
        max_risk_per_trade: Decimal = Decimal("1.0"),
        max_daily_drawdown: Decimal = Decimal("4.0"),
        max_total_drawdown: Decimal = Decimal("8.0"),
        max_open_positions: int = 3,
        anti_tilt_losses: int = 3,
        anti_tilt_hours: int = 12,
        anti_tilt_min_loss_amount: Decimal = Decimal("5"),
        anti_tilt_min_cumulative_loss_amount: Decimal = Decimal("20"),
        anti_tilt_reset_streak_on_new_day: bool = True,
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
            anti_tilt_min_loss_amount (Decimal): Perte minimale pour compter dans la serie.
            anti_tilt_min_cumulative_loss_amount (Decimal): Montant cumule minimal avant pause.
            anti_tilt_reset_streak_on_new_day (bool): Reinitialise la serie de pertes a l'ouverture.
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.max_open_positions = max_open_positions
        self.anti_tilt_losses = anti_tilt_losses
        self.anti_tilt_hours = anti_tilt_hours
        self.settings = get_settings()
        self.anti_tilt_min_loss_amount = anti_tilt_min_loss_amount
        self.anti_tilt_min_cumulative_loss_amount = anti_tilt_min_cumulative_loss_amount
        self.anti_tilt_reset_streak_on_new_day = anti_tilt_reset_streak_on_new_day
        self.block_new_entries_drawdown = Decimal(str(self.settings.risk_max_daily_drawdown_percent))
        self.flatten_progressive_drawdown = Decimal(
            str(getattr(self.settings, "risk_flatten_progressive_drawdown_percent", 2.25))
        )
        self.flatten_immediate_drawdown = Decimal(
            str(getattr(self.settings, "risk_flatten_immediate_drawdown_percent", 2.50))
        )

        self._consecutive_losses = 0
        self._consecutive_loss_amount = Decimal("0")
        self._anti_tilt_until: datetime | None = None
        self._daily_pnl = Decimal("0")
        self._daily_pnl_date = self._get_market_now().date()
        self._total_pnl = Decimal("0")
        self._open_positions_count = 0
        self._account_balance = Decimal("100000")
        self._account_equity = Decimal("100000")
        self._day_open_balance = Decimal("0")
        self._day_open_equity = Decimal("0")
        self._kill_switch_state = "normal"
        self._kill_switch_reason: str | None = None
        self._flatten_started_at: datetime | None = None
        self._flatten_last_action_at: datetime | None = None
        self._flatten_action_count = 0
        self._session_log_states: dict[str, str] = {}
        self._symbol_asset_classes: dict[str, str] = {}

        logger.info(
            "RiskValidator initialise: max_risk=%s%%, max_dd_daily=%s%%, max_dd_total=%s%%, anti_tilt=%s pertes/%sh, perte_min=%s USD, cumul_min=%s USD",
            max_risk_per_trade,
            max_daily_drawdown,
            max_total_drawdown,
            self.anti_tilt_losses,
            self.anti_tilt_hours,
            self.anti_tilt_min_loss_amount,
            self.anti_tilt_min_cumulative_loss_amount,
        )

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
        self._roll_daily_window_if_needed()

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

        if self.should_block_new_entries():
            result["allowed"] = False
            result["reason"] = self._kill_switch_reason or (
                f"Drawdown journalier limite atteinte ({self.block_new_entries_drawdown}%)"
            )
            result["checks"].append(("daily_drawdown", False, result["reason"]))
            return result
        result["checks"].append(
            (
                "daily_drawdown",
                True,
                (
                    "DD journalier realise "
                    f"{self._get_daily_realized_drawdown_percent():.2f}% | "
                    f"DD equity {self._get_daily_equity_drawdown_percent():.2f}%"
                ),
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
            }
            current_price = mock_prices.get(order.symbol.upper(), Decimal("100"))

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
        asset_class = self._symbol_asset_classes.get(symbol_upper)

        if asset_class == "crypto" or self._is_crypto_symbol(symbol_upper):
            return Decimal("1")
        if "XAU" in symbol_upper:
            return Decimal("100")
        if "US30" in symbol_upper or "GER40" in symbol_upper or "CASH" in symbol_upper:
            return Decimal("1")
        if "JPY" in symbol_upper:
            return Decimal("1000")
        if "USD" in symbol_upper or "EUR" in symbol_upper:
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
        asset_class = self._symbol_asset_classes.get(symbol_upper)
        if asset_class == "crypto" or self._is_crypto_symbol(symbol_upper):
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
        return max(
            self._get_daily_realized_drawdown_percent(),
            self._get_daily_equity_drawdown_percent(),
        )

    def _get_daily_realized_drawdown_percent(self) -> Decimal:
        """
        Retourne le drawdown realise journalier en pourcentage.

        Returns:
            Decimal: Drawdown realise par rapport au solde d'ouverture.
        """
        self._roll_daily_window_if_needed()
        reference_balance = self._day_open_balance or self._account_balance
        if reference_balance <= 0 or self._daily_pnl >= 0:
            return Decimal("0")
        return (abs(self._daily_pnl) / reference_balance * 100).quantize(
            Decimal("0.01")
        )

    def _get_daily_equity_drawdown_percent(self) -> Decimal:
        """
        Retourne le drawdown intraday calcule sur l'equity.

        Returns:
            Decimal: Drawdown d'equity par rapport a l'ouverture de jour.
        """
        self._roll_daily_window_if_needed()
        reference_equity = self._day_open_equity or self._account_equity
        if reference_equity <= 0 or self._account_equity >= reference_equity:
            return Decimal("0")
        return ((reference_equity - self._account_equity) / reference_equity * 100).quantize(
            Decimal("0.01")
        )

    def _get_total_drawdown_percent(self) -> Decimal:
        """
        Retourne le drawdown total en pourcentage.

        Returns:
            Decimal: Drawdown total.
        """
        if self._account_balance <= 0 or self._total_pnl >= 0:
            return Decimal("0")
        return (abs(self._total_pnl) / self._account_balance * 100).quantize(
            Decimal("0.01")
        )

    def record_trade_result(self, profit: Decimal) -> None:
        """
        Enregistre le resultat d'un trade.

        Args:
            profit (Decimal): Profit ou perte du trade.
        """
        self._roll_daily_window_if_needed()
        self._daily_pnl += profit
        self._total_pnl += profit

        if profit < 0:
            loss_amount = abs(profit)
            if loss_amount >= self.anti_tilt_min_loss_amount:
                self._consecutive_losses += 1
                self._consecutive_loss_amount += loss_amount
                if (
                    self._consecutive_losses >= self.anti_tilt_losses
                    and self._consecutive_loss_amount >= self.anti_tilt_min_cumulative_loss_amount
                ):
                    self._activate_anti_tilt()
            else:
                logger.info(
                    "Perte %s ignoree pour l'anti-tilt car inferieure au seuil minimal %s.",
                    loss_amount.quantize(Decimal("0.01")),
                    self.anti_tilt_min_loss_amount.quantize(Decimal("0.01")),
                )
        else:
            self._consecutive_losses = 0
            self._consecutive_loss_amount = Decimal("0")
        self._refresh_governor_state()

    def _activate_anti_tilt(self) -> None:
        """Active le mode anti-tilt."""
        self._anti_tilt_until = datetime.now() + timedelta(hours=self.anti_tilt_hours)
        logger.warning(
            "ANTI-TILT active jusqu'a %s apres %s pertes significatives pour %s USD cumules.",
            self._anti_tilt_until,
            self._consecutive_losses,
            self._consecutive_loss_amount.quantize(Decimal("0.01")),
        )

    def update_positions_count(self, count: int) -> None:
        """
        Met a jour le nombre de positions ouvertes.

        Args:
            count (int): Nombre courant de positions.
        """
        self._open_positions_count = count

    def update_account_balance(self, balance: Decimal, equity: Decimal | None = None) -> None:
        """
        Met a jour l'etat financier de reference du compte.

        Args:
            balance (Decimal): Solde de reference.
            equity (Decimal | None): Equity courante si disponible.
        """
        self._roll_daily_window_if_needed()
        self._account_balance = balance
        self._account_equity = equity if equity is not None else balance
        self._ensure_day_open_snapshot()
        self._refresh_governor_state()

    def should_block_new_entries(self) -> bool:
        """
        Indique si les nouvelles entrees doivent etre bloquees.

        Returns:
            bool: ``True`` si le drawdown journalier impose un stop.
        """
        self._refresh_governor_state()
        return self._kill_switch_state in {
            "blocked_new_entries",
            "flatten_progressive",
            "flatten_immediate",
        }

    def should_flatten_progressive(self) -> bool:
        """
        Indique si un flatten progressif doit etre applique.

        Returns:
            bool: ``True`` si le drawdown d'equity impose un flatten progressif.
        """
        self._refresh_governor_state()
        return self._kill_switch_state == "flatten_progressive"

    def should_flatten_immediate(self) -> bool:
        """
        Indique si un flatten immediat doit etre applique.

        Returns:
            bool: ``True`` si le drawdown d'equity impose un flatten total.
        """
        self._refresh_governor_state()
        return self._kill_switch_state == "flatten_immediate"

    def can_run_progressive_flatten(self, cooldown_seconds: int = 120) -> bool:
        """
        Indique si un nouveau cycle de flatten progressif peut partir.

        Args:
            cooldown_seconds (int): Cooldown minimal entre deux clotures.

        Returns:
            bool: ``True`` si une nouvelle cloture peut etre tentee.
        """
        if not self.should_flatten_progressive():
            return False
        if self._flatten_last_action_at is None:
            return True
        return (datetime.now() - self._flatten_last_action_at).total_seconds() >= max(
            30,
            cooldown_seconds,
        )

    def note_flatten_action(self, mode: str) -> None:
        """
        Enregistre une action de flatten pour l'observabilite.

        Args:
            mode (str): Mode applique (`progressive` ou `immediate`).
        """
        now = datetime.now()
        if self._flatten_started_at is None:
            self._flatten_started_at = now
        self._flatten_last_action_at = now
        self._flatten_action_count += 1
        logger.warning("Flatten %s execute a %s.", mode, now.isoformat())

    def reset_day_session(self) -> None:
        """
        Reinitialise explicitement la session de risque courante.

        Un reset manuel est utilise par l'operateur pour reouvrir une session
        de trading propre apres revue. Il doit donc lever aussi l'anti-tilt et
        oublier la serie de pertes consecutive ayant conduit a la pause.
        """
        self._daily_pnl = Decimal("0")
        self._daily_pnl_date = self._get_market_now().date()
        self._day_open_balance = self._account_balance
        self._day_open_equity = self._account_equity
        self._consecutive_losses = 0
        self._consecutive_loss_amount = Decimal("0")
        self._anti_tilt_until = None
        self._kill_switch_state = "normal"
        self._kill_switch_reason = None
        self._flatten_started_at = None
        self._flatten_last_action_at = None
        self._flatten_action_count = 0
        logger.warning(
            "Session de risque reinitialisee manuellement; anti-tilt et pertes consecutives effaces."
        )

    async def save_state(self) -> None:
        """
        Persiste l'etat de risque pour survivre aux redemarrages du banker.

        Le coupe-circuit journalier perd toute sa valeur si un redemarrage
        efface les pertes realisees de la session. Cette persistance garde la
        memoire du PnL journalier, du PnL total et de l'anti-tilt.
        """
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            await redis.cache_set(
                "banker:risk:state",
                {
                    "daily_pnl": str(self._daily_pnl),
                    "daily_pnl_date": self._daily_pnl_date.isoformat(),
                    "total_pnl": str(self._total_pnl),
                    "day_open_balance": str(self._day_open_balance),
                    "day_open_equity": str(self._day_open_equity),
                    "account_balance": str(self._account_balance),
                    "account_equity": str(self._account_equity),
                    "consecutive_losses": self._consecutive_losses,
                    "consecutive_loss_amount": str(self._consecutive_loss_amount),
                    "anti_tilt_until": (
                        self._anti_tilt_until.isoformat()
                        if self._anti_tilt_until
                        else None
                    ),
                    "kill_switch_state": self._kill_switch_state,
                    "kill_switch_reason": self._kill_switch_reason,
                    "flatten_started_at": (
                        self._flatten_started_at.isoformat()
                        if self._flatten_started_at
                        else None
                    ),
                    "flatten_last_action_at": (
                        self._flatten_last_action_at.isoformat()
                        if self._flatten_last_action_at
                        else None
                    ),
                    "flatten_action_count": self._flatten_action_count,
                },
                ttl_seconds=7 * 86400,
            )
        except Exception:
            pass

    async def load_state(self) -> None:
        """
        Recharge l'etat de risque persiste si disponible.

        Returns:
            None: L'etat interne est mis a jour en place.
        """
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            state = await redis.cache_get("banker:risk:state")
            if not state:
                return

            self._daily_pnl = Decimal(str(state.get("daily_pnl", "0")))
            raw_daily_date = state.get("daily_pnl_date")
            if raw_daily_date:
                self._daily_pnl_date = datetime.fromisoformat(raw_daily_date).date()
            self._total_pnl = Decimal(str(state.get("total_pnl", "0")))
            self._day_open_balance = Decimal(str(state.get("day_open_balance", "0")))
            self._day_open_equity = Decimal(str(state.get("day_open_equity", "0")))
            self._account_balance = Decimal(str(state.get("account_balance", str(self._account_balance))))
            self._account_equity = Decimal(str(state.get("account_equity", str(self._account_equity))))
            self._consecutive_losses = int(state.get("consecutive_losses", 0))
            self._consecutive_loss_amount = Decimal(
                str(state.get("consecutive_loss_amount", str(self._consecutive_loss_amount)))
            )

            raw_anti_tilt_until = state.get("anti_tilt_until")
            self._anti_tilt_until = (
                datetime.fromisoformat(raw_anti_tilt_until)
                if raw_anti_tilt_until
                else None
            )
            self._kill_switch_state = str(state.get("kill_switch_state", "normal") or "normal")
            self._kill_switch_reason = str(state.get("kill_switch_reason") or "").strip() or None
            raw_flatten_started_at = state.get("flatten_started_at")
            self._flatten_started_at = (
                datetime.fromisoformat(raw_flatten_started_at)
                if raw_flatten_started_at
                else None
            )
            raw_flatten_last_action_at = state.get("flatten_last_action_at")
            self._flatten_last_action_at = (
                datetime.fromisoformat(raw_flatten_last_action_at)
                if raw_flatten_last_action_at
                else None
            )
            self._flatten_action_count = int(state.get("flatten_action_count", 0) or 0)
            self._roll_daily_window_if_needed()
            self._refresh_governor_state()
            logger.info("Etat de risque recharge depuis Redis.")
        except Exception:
            pass

    def calculate_lot_size(
        self,
        balance: Decimal,
        risk_percent: Decimal,
        sl_distance: Decimal,
        symbol: str = "EURUSD",
    ) -> float:
        """
        Calcule la taille de lot optimale.

        Args:
            balance (Decimal): Solde disponible.
            risk_percent (Decimal): Risque cible en pourcentage.
            sl_distance (Decimal): Distance au stop loss.
            symbol (str): Symbole traite.

        Returns:
            float: Taille de lot arrondie et securisee.
        """
        if sl_distance <= 0 or balance <= 0 or risk_percent <= 0:
            return 0.0

        risk_amount = balance * (risk_percent / Decimal("100"))
        symbol_upper = symbol.upper()
        asset_class = self._symbol_asset_classes.get(symbol_upper)

        if asset_class == "crypto" or self._is_crypto_symbol(symbol_upper):
            point_value = Decimal("1")
        elif "XAU" in symbol_upper:
            point_value = Decimal("100")
        elif any(idx in symbol_upper for idx in ["US30", "US100", "GER40", "CASH"]):
            point_value = Decimal("1")
        else:
            point_value = Decimal("100000")

        try:
            raw_lots = risk_amount / (sl_distance * point_value)
            if raw_lots <= 0:
                return 0.0

            lot_size = float(max(raw_lots, Decimal("0")).quantize(Decimal("0.01")))
            return max(0.0, lot_size)
        except ZeroDivisionError:
            return 0.0

    async def get_current_status(self) -> RiskStatus:
        """
        Retourne l'etat courant du module de risque.

        Returns:
            RiskStatus: Etat consolide du validateur.
        """
        self._roll_daily_window_if_needed()
        self._refresh_governor_state()
        trading_allowed = (
            not self._is_anti_tilt_active()
            and not self.should_block_new_entries()
            and self._get_total_drawdown_percent() < self.max_total_drawdown
            and self._open_positions_count < self.max_open_positions
        )
        return RiskStatus(
            account_id=uuid4(),
            daily_drawdown_percent=self._get_daily_drawdown_percent(),
            total_drawdown_percent=self._get_total_drawdown_percent(),
            daily_realized_pnl=self._daily_pnl.quantize(Decimal("0.01")),
            day_open_balance=self._day_open_balance.quantize(Decimal("0.01")),
            day_open_equity=self._day_open_equity.quantize(Decimal("0.01")),
            daily_realized_drawdown_percent=self._get_daily_realized_drawdown_percent(),
            daily_equity_drawdown_percent=self._get_daily_equity_drawdown_percent(),
            open_positions_count=self._open_positions_count,
            anti_tilt_active=self._is_anti_tilt_active(),
            anti_tilt_expires_at=self._anti_tilt_until,
            trading_allowed=trading_allowed,
            kill_switch_state=self._kill_switch_state,
            kill_switch_reason=self._kill_switch_reason,
            flatten_state=self._build_flatten_state(),
            thresholds=self._build_thresholds_payload(),
        )

    def _roll_daily_window_if_needed(self) -> None:
        """
        Reinitialise le PnL journalier quand une nouvelle journee commence.

        Le drawdown journalier doit suivre la session locale courante et non
        cumuler les pertes des jours precedents. Sans cette remise a zero, le
        coupe-circuit devient soit trop permissif apres redemarrage, soit trop
        bloquant apres plusieurs jours.
        """
        market_date = self._get_market_now().date()
        if market_date == self._daily_pnl_date:
            return

        logger.info(
            "Nouvelle journee de risque detectee: remise a zero du PnL journalier (%s -> %s).",
            self._daily_pnl_date,
            market_date,
        )
        self._daily_pnl = Decimal("0")
        self._daily_pnl_date = market_date
        self._day_open_balance = self._account_balance
        self._day_open_equity = self._account_equity
        self._kill_switch_state = "normal"
        self._kill_switch_reason = None
        self._flatten_started_at = None
        self._flatten_last_action_at = None
        self._flatten_action_count = 0
        if self.anti_tilt_reset_streak_on_new_day:
            self._consecutive_losses = 0
            self._consecutive_loss_amount = Decimal("0")

    def _ensure_day_open_snapshot(self) -> None:
        """
        Initialise le snapshot d'ouverture de jour si necessaire.
        """
        if self._day_open_balance <= 0:
            self._day_open_balance = self._account_balance
        if self._day_open_equity <= 0:
            self._day_open_equity = self._account_equity

    def _refresh_governor_state(self) -> None:
        """
        Recalcule l'etat du gouverneur de drawdown intraday.
        """
        realized_dd = self._get_daily_realized_drawdown_percent()
        equity_dd = self._get_daily_equity_drawdown_percent()
        previous_state = self._kill_switch_state

        if equity_dd >= self.flatten_immediate_drawdown:
            self._kill_switch_state = "flatten_immediate"
            self._kill_switch_reason = "drawdown_equity_immediat"
        elif equity_dd >= self.flatten_progressive_drawdown:
            self._kill_switch_state = "flatten_progressive"
            self._kill_switch_reason = "drawdown_equity_progressif"
        elif (
            realized_dd >= self.block_new_entries_drawdown
            or equity_dd >= self.block_new_entries_drawdown
        ):
            self._kill_switch_state = "blocked_new_entries"
            self._kill_switch_reason = (
                "drawdown_realise_journalier"
                if realized_dd >= self.block_new_entries_drawdown
                else "drawdown_equity_journalier"
            )
        else:
            self._kill_switch_state = "normal"
            self._kill_switch_reason = None
            self._flatten_started_at = None
            self._flatten_last_action_at = None
            self._flatten_action_count = 0

        if self._kill_switch_state != "normal" and self._flatten_started_at is None:
            self._flatten_started_at = datetime.now()

        if previous_state != self._kill_switch_state:
            logger.warning(
                "Etat du gouverneur de risque mis a jour: %s -> %s (%s).",
                previous_state,
                self._kill_switch_state,
                self._kill_switch_reason,
            )

    def _build_flatten_state(self) -> dict[str, Any]:
        """
        Construit la vue publique du flatten intraday.

        Returns:
            dict[str, Any]: Etat publie dans les endpoints banker.
        """
        return {
            "active": self._kill_switch_state in {"flatten_progressive", "flatten_immediate"},
            "mode": self._kill_switch_state if self._kill_switch_state.startswith("flatten_") else None,
            "started_at": self._flatten_started_at.isoformat() if self._flatten_started_at else None,
            "last_action_at": (
                self._flatten_last_action_at.isoformat()
                if self._flatten_last_action_at
                else None
            ),
            "action_count": self._flatten_action_count,
            "cooldown_seconds": 120,
            "resume_below_percent": float(self.block_new_entries_drawdown),
        }

    def _build_thresholds_payload(self) -> dict[str, Any]:
        """
        Retourne les seuils actifs du gouverneur de risque.

        Returns:
            dict[str, Any]: Seuils actifs en pourcentage.
        """
        return {
            "block_new_entries_drawdown_percent": float(self.block_new_entries_drawdown),
            "flatten_progressive_drawdown_percent": float(self.flatten_progressive_drawdown),
            "flatten_immediate_drawdown_percent": float(self.flatten_immediate_drawdown),
            "max_total_drawdown_percent": float(self.max_total_drawdown),
            "anti_tilt_losses": int(self.anti_tilt_losses),
            "anti_tilt_hours": int(self.anti_tilt_hours),
            "anti_tilt_min_loss_amount_usd": float(self.anti_tilt_min_loss_amount),
            "anti_tilt_min_cumulative_loss_amount_usd": float(
                self.anti_tilt_min_cumulative_loss_amount
            ),
            "anti_tilt_reset_streak_on_new_day": bool(self.anti_tilt_reset_streak_on_new_day),
            "anti_tilt_current_consecutive_losses": int(self._consecutive_losses),
            "anti_tilt_current_cumulative_loss_amount_usd": float(self._consecutive_loss_amount),
        }


@lru_cache
def get_risk_validator() -> RiskValidator:
    """
    Construit l'instance singleton du validateur de risques.

    Returns:
        RiskValidator: Instance configuree depuis les settings.
    """
    settings = get_settings()
    return RiskValidator(
        max_risk_per_trade=Decimal(str(settings.risk_max_single_trade_percent)),
        max_daily_drawdown=Decimal(str(settings.risk_max_daily_drawdown_percent)),
        max_total_drawdown=Decimal(str(settings.risk_max_total_drawdown_percent)),
        max_open_positions=settings.risk_max_open_positions,
        anti_tilt_losses=settings.risk_anti_tilt_losses,
        anti_tilt_hours=settings.risk_anti_tilt_duration_hours,
        anti_tilt_min_loss_amount=Decimal(str(settings.risk_anti_tilt_min_loss_amount_usd)),
        anti_tilt_min_cumulative_loss_amount=Decimal(
            str(settings.risk_anti_tilt_min_cumulative_loss_amount_usd)
        ),
        anti_tilt_reset_streak_on_new_day=settings.risk_anti_tilt_reset_streak_on_new_day,
    )
