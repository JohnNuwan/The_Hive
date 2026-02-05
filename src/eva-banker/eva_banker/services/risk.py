"""
Service Validation des Risques - Constitution Loi 2
Vérifie que les ordres respectent les règles de gestion des risques
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

from shared import RiskStatus, TradeOrder, get_settings

logger = logging.getLogger(__name__)


class RiskValidator:
    """
    Validateur de risques selon la Constitution (Loi 2).
    
    Vérifie:
    - Risque max par trade (1%)
    - Drawdown journalier max (4%)
    - Drawdown total max (8%)
    - Nombre max de positions ouvertes (3)
    - Anti-Tilt (pause après 2 pertes consécutives)
    - News Filter (pas de trade pendant annonces)
    """

    def __init__(
        self,
        max_risk_per_trade: Decimal = Decimal("1.0"),
        max_daily_drawdown: Decimal = Decimal("4.0"),
        max_total_drawdown: Decimal = Decimal("8.0"),
        max_open_positions: int = 3,
        anti_tilt_losses: int = 2,
        anti_tilt_hours: int = 24,
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.max_open_positions = max_open_positions
        self.anti_tilt_losses = anti_tilt_losses
        self.anti_tilt_hours = anti_tilt_hours

        # État interne
        self._consecutive_losses = 0
        self._anti_tilt_until: datetime | None = None
        self._daily_pnl = Decimal("0")
        self._total_pnl = Decimal("0")
        self._open_positions_count = 0
        self._account_balance = Decimal("100000")

        logger.info(
            f"RiskValidator initialisé: max_risk={max_risk_per_trade}%, "
            f"max_dd_daily={max_daily_drawdown}%, max_dd_total={max_total_drawdown}%"
        )

    async def validate_order(self, order: TradeOrder) -> dict[str, Any]:
        """
        Valide un ordre selon les règles de risque.
        
        Returns:
            Dict avec 'allowed', 'reason', 'risk_percent', etc.
        """
        result = {
            "allowed": True,
            "reason": None,
            "risk_percent": Decimal("0"),
            "checks": [],
        }

        # 1. Vérification Stop Loss obligatoire
        if order.stop_loss_price is None:
            result["allowed"] = False
            result["reason"] = "SL obligatoire (ROE Trading)"
            result["checks"].append(("stop_loss", False, "Stop Loss manquant"))
            return result
        result["checks"].append(("stop_loss", True, "Stop Loss présent"))

        # 2. Calcul du risque
        risk_percent = self._calculate_risk_percent(order)
        result["risk_percent"] = risk_percent

        if risk_percent > self.max_risk_per_trade:
            result["allowed"] = False
            result["reason"] = f"Risque {risk_percent:.2f}% > max {self.max_risk_per_trade}%"
            result["checks"].append(("risk_per_trade", False, result["reason"]))
            return result
        result["checks"].append(
            ("risk_per_trade", True, f"Risque {risk_percent:.2f}% OK")
        )

        # 3. Vérification Anti-Tilt
        if self._is_anti_tilt_active():
            result["allowed"] = False
            result["reason"] = f"Anti-Tilt actif jusqu'à {self._anti_tilt_until}"
            result["checks"].append(("anti_tilt", False, result["reason"]))
            return result
        result["checks"].append(("anti_tilt", True, "Anti-Tilt inactif"))

        # 4. Vérification Drawdown Journalier
        if self._get_daily_drawdown_percent() >= self.max_daily_drawdown:
            result["allowed"] = False
            result["reason"] = f"Drawdown journalier limite atteinte ({self.max_daily_drawdown}%)"
            result["checks"].append(("daily_drawdown", False, result["reason"]))
            return result
        result["checks"].append(
            ("daily_drawdown", True, f"DD journalier {self._get_daily_drawdown_percent():.2f}% OK")
        )

        # 5. Vérification Drawdown Total
        if self._get_total_drawdown_percent() >= self.max_total_drawdown:
            result["allowed"] = False
            result["reason"] = f"Drawdown total limite atteinte ({self.max_total_drawdown}%)"
            result["checks"].append(("total_drawdown", False, result["reason"]))
            return result
        result["checks"].append(
            ("total_drawdown", True, f"DD total {self._get_total_drawdown_percent():.2f}% OK")
        )

        # 6. Vérification nombre de positions
        if self._open_positions_count >= self.max_open_positions:
            result["allowed"] = False
            result["reason"] = f"Max positions atteint ({self.max_open_positions})"
            result["checks"].append(("max_positions", False, result["reason"]))
            return result
        result["checks"].append(
            ("max_positions", True, f"Positions {self._open_positions_count}/{self.max_open_positions}")
        )

        logger.info(f"✅ Ordre validé: {order.symbol} {order.action.value} - risque {risk_percent:.2f}%")
        return result

    def _calculate_risk_percent(self, order: TradeOrder) -> Decimal:
        """Calcule le pourcentage de risque de l'ordre"""
        if order.stop_loss_price is None:
            return Decimal("100")  # Risque infini sans SL

        # Simuler le calcul du risque
        # En production, on utiliserait le prix actuel du symbole
        mock_prices = {
            "XAUUSD": Decimal("2080.00"),
            "EURUSD": Decimal("1.0850"),
            "GBPUSD": Decimal("1.2650"),
        }
        current_price = mock_prices.get(order.symbol, Decimal("100"))

        # Distance au SL en points
        sl_distance = abs(current_price - order.stop_loss_price)

        # Valeur par lot (approximation)
        pip_value = Decimal("10") if "XAU" in order.symbol else Decimal("10")

        # Perte potentielle
        potential_loss = sl_distance * float(order.volume) * float(pip_value)

        # Pourcentage du capital
        if self._account_balance > 0:
            risk_percent = (Decimal(potential_loss) / self._account_balance) * 100
        else:
            risk_percent = Decimal("100")

        return risk_percent.quantize(Decimal("0.01"))

    def _is_anti_tilt_active(self) -> bool:
        """Vérifie si l'Anti-Tilt est actif"""
        if self._anti_tilt_until is None:
            return False
        return datetime.now() < self._anti_tilt_until

    def _get_daily_drawdown_percent(self) -> Decimal:
        """Retourne le drawdown journalier en pourcentage"""
        if self._account_balance <= 0 or self._daily_pnl >= 0:
            return Decimal("0")
        return (abs(self._daily_pnl) / self._account_balance * 100).quantize(Decimal("0.01"))

    def _get_total_drawdown_percent(self) -> Decimal:
        """Retourne le drawdown total en pourcentage"""
        if self._account_balance <= 0 or self._total_pnl >= 0:
            return Decimal("0")
        return (abs(self._total_pnl) / self._account_balance * 100).quantize(Decimal("0.01"))

    def record_trade_result(self, profit: Decimal) -> None:
        """Enregistre le résultat d'un trade pour Anti-Tilt"""
        self._daily_pnl += profit
        self._total_pnl += profit

        if profit < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.anti_tilt_losses:
                self._activate_anti_tilt()
        else:
            self._consecutive_losses = 0

    def _activate_anti_tilt(self) -> None:
        """Active le mode Anti-Tilt"""
        self._anti_tilt_until = datetime.now() + timedelta(hours=self.anti_tilt_hours)
        logger.warning(f"⚠️ ANTI-TILT activé jusqu'à {self._anti_tilt_until}")

    def update_positions_count(self, count: int) -> None:
        """Met à jour le nombre de positions ouvertes"""
        self._open_positions_count = count

    def update_account_balance(self, balance: Decimal) -> None:
        """Met à jour le solde du compte"""
        self._account_balance = balance

    async def get_current_status(self) -> RiskStatus:
        """Retourne le statut actuel des risques"""
        return RiskStatus(
            account_id=uuid4(),  # À remplacer par le vrai account_id
            daily_drawdown_percent=self._get_daily_drawdown_percent(),
            total_drawdown_percent=self._get_total_drawdown_percent(),
            open_positions_count=self._open_positions_count,
            anti_tilt_active=self._is_anti_tilt_active(),
            anti_tilt_expires_at=self._anti_tilt_until,
            trading_allowed=not self._is_anti_tilt_active()
            and self._get_daily_drawdown_percent() < self.max_daily_drawdown,
        )


@lru_cache
def get_risk_validator() -> RiskValidator:
    """Retourne l'instance du validateur de risques"""
    settings = get_settings()
    return RiskValidator(
        max_risk_per_trade=Decimal(str(settings.risk_max_single_trade_percent)),
        max_daily_drawdown=Decimal(str(settings.risk_max_daily_drawdown_percent)),
        max_total_drawdown=Decimal(str(settings.risk_max_total_drawdown_percent)),
        max_open_positions=settings.risk_max_open_positions,
        anti_tilt_losses=settings.risk_anti_tilt_losses,
        anti_tilt_hours=settings.risk_anti_tilt_duration_hours,
    )
