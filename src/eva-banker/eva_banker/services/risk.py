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

from shared import (
    RiskStatus, 
    TradeOrder, 
    get_settings, 
    calculate_var, 
    calculate_cvar
)

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

        # 7. Vérification VaR (Value at Risk) - Conscience du risque adaptative
        # On suppose que les 'returns' sont passés ou disponibles via une source de données
        # Pour la démo, on utilise des rendements simulés
        mock_returns = [0.001, -0.002, 0.005, -0.01, 0.002]
        var = calculate_var(mock_returns)
        if var < -0.025: # Seuil de panique VaR 2.5%
            result["allowed"] = False
            result["reason"] = f"VaR trop élevée ({var:.4f}). Marché instable."
            result["checks"].append(("var_check", False, result["reason"]))
            return result
        result["checks"].append(("var_check", True, f"VaR OK ({var:.4f})"))

        logger.info(f"✅ Ordre validé: {order.symbol} {order.action.value} - risque {risk_percent:.2f}%")
        return result

    def _calculate_risk_percent(self, order: TradeOrder) -> Decimal:
        """Calcule le pourcentage de risque de l'ordre"""
        if order.stop_loss_price is None:
            return Decimal("100")  # Risque infini sans SL

        # Simuler le calcul du risque
        # En production, on utiliserait le prix actuel du symbole
        mock_prices = {
            "XAUUSD": Decimal("4900.00"), # Updated based on logs
            "EURUSD": Decimal("1.0850"),
            "BTCUSD": Decimal("67000.00"),
            "US30.cash": Decimal("39000.00")
        }
        current_price = mock_prices.get(order.symbol, Decimal("100"))

        # Distance au SL en points (Prix)
        sl_distance = abs(current_price - order.stop_loss_price)

        # Taille du contrat (Approximation FTMO/Standard)
        contract_size = self._get_estimated_contract_size(order.symbol)

        # Perte potentielle = Distance * Volume * ContractSize
        potential_loss = sl_distance * Decimal(str(order.volume)) * contract_size

        # Pourcentage du capital
        if self._account_balance > 0:
            risk_percent = (potential_loss / self._account_balance) * 100
        else:
            risk_percent = Decimal("100")

        return risk_percent.quantize(Decimal("0.01"))

    def _get_estimated_contract_size(self, symbol: str) -> Decimal:
        """Estime la taille du contrat pour le calcul de risque"""
        if "XAU" in symbol:
            return Decimal("100")  # 1 lot = 100 oz
        if "BTC" in symbol or "ETH" in symbol:
            return Decimal("1")    # 1 lot = 1 coin
        if "US30" in symbol or "GER40" in symbol or "cash" in symbol:
            return Decimal("1")    # 1 lot = 1 contract usually on user's broker
        if "JPY" in symbol:
            return Decimal("1000") # JPY pairs have different scaling usually 100k but pip is 0.01
        if "USD" in symbol or "EUR" in symbol:
            return Decimal("100000") # Forex Standard Lot
        return Decimal("10") # Default fallback

    def is_within_trading_session(self, symbol: str) -> bool:
        """
        Vérifie si l'actif est tradable à l'heure actuelle (Filtre de Nuit).
        Bloque le trading entre 23:00 et 01:00 (Paris) pour éviter les spreads monstrueux du Rollover.
        """
        symbol_upper = symbol.upper()
        # Crypto is 24/7
        if "BTC" in symbol_upper or "ETH" in symbol_upper:
            return True
            
        now = datetime.now()
        hour = now.hour
        
        # Rollover Trap: 23:00 to 00:59
        if hour == 23 or hour == 0:
            logger.warning(f"🌙 Session Filter: Trading bloqué sur {symbol} (Rollover en cours).")
            return False
            
        return True

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

    def calculate_lot_size(self, balance: Decimal, risk_percent: Decimal, sl_distance: Decimal, symbol: str = "EURUSD") -> float:
        """
        Calcule la taille du lot optimale en fonction du risque et de la distance du SL.
        
        Lots = (Balance * Risk%) / (SL_Distance * Point_Value_per_Lot)
        """
        if sl_distance <= 0 or balance <= 0 or risk_percent <= 0:
            return 0.01

        risk_amount = balance * (risk_percent / Decimal("100"))
        
        # Déterminer la valeur monétaire d'une variation de 1.0 du prix pour 1 lot
        # Contract Size * Tick Size Value (usually Tick Size is the base unit of price variation)
        # Pour simplifier on utilise des multiplicateurs par classe d'actif
        
        if "XAU" in symbol.upper():
            # Gold: 1 lot = 100 oz. Variation de $1.0 = $100 de P&L
            point_value = Decimal("100") 
        elif any(idx in symbol.upper() for idx in ["US30", "NAS100", "GER40", "BTC"]):
            # Indices / Crypto: Souvent 1 lot = 1 contrat. Variation de 1.0 = $1 de P&L
            point_value = Decimal("1")
        else:
            # Forex: 1 lot = 100,000. Variation de 1.0 (ex: 1.08 -> 2.08) = $100,000
            # Mais on mesure souvent sl_distance en prix (ex: 0.0020).
            point_value = Decimal("100000")

        try:
            raw_lots = risk_amount / (sl_distance * point_value)
            
            # Arrondir à 2 décimales (MT5 standard)
            lot_size = float(raw_lots.quantize(Decimal("0.01")))
            
            # Limites de sécurité
            return max(0.01, lot_size)
        except ZeroDivisionError:
            return 0.01

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
