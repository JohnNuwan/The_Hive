"""Environnement de trading MuZero pour THE HIVE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Actions
HOLD = 0
BUY = 1
SELL = 2
SPLIT = 3
CLOSE = 4

ACTION_NAMES = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]


@dataclass
class SymbolSpec:
    """Décrit les paramètres de trading simulés d'un symbole.

    Attributes:
        pip_size (float): Taille de pip indicative du symbole.
        trade_size (float): Notionnel engagé par ordre dans l'environnement.
        initial_balance (float): Capital initial utilisé pour l'épisode.
    """

    pip_size: float
    trade_size: float
    initial_balance: float

    @classmethod
    def for_symbol(cls, symbol: str) -> "SymbolSpec":
        """Construit une spécification prudente et cohérente cross-asset.

        Le moteur offline travaille désormais en notionnel, pas en volume brut.
        Cela évite qu'un actif peu cher ou très cher rende les métriques
        d'Arena incohérentes d'un univers à l'autre.

        Args:
            symbol (str): Symbole de marché.

        Returns:
            SymbolSpec: Paramètres simulés adaptés au symbole.
        """
        normalized = symbol.upper()
        initial_balance = 10_000.0
        trade_notional = 1_000.0
        pip_size = 0.0001

        if "JPY" in normalized and len(normalized) == 6:
            pip_size = 0.01
        elif any(index_name in normalized for index_name in ["US30", "GER40", "US100", "US500", "NAS", "SPX"]):
            pip_size = 1.0
            trade_notional = 1_200.0
        elif any(metal in normalized for metal in ["XAU", "XAG"]):
            pip_size = 0.01
        elif any(
            token in normalized
            for token in [
                "BTC",
                "ETH",
                "BNB",
                "ADA",
                "DOGE",
                "XRP",
                "SOL",
                "DOT",
                "LTC",
                "UNI",
                "AVAX",
                "LINK",
                "AAVE",
                "MATIC",
                "ALGO",
            ]
        ):
            pip_size = 0.01 if any(token in normalized for token in ["BTC", "ETH"]) else 0.0001
            trade_notional = 800.0
        elif len(normalized) <= 5:
            pip_size = 0.01
            trade_notional = 900.0

        return cls(
            pip_size=pip_size,
            trade_size=trade_notional,
            initial_balance=initial_balance,
        )


class TradingEnvironment:
    """Environnement de trading compatible Gymnasium pour MuZero.

    L'observation contient les colonnes de marché brutes suivies de six
    caractéristiques supplémentaires décrivant l'état de position.
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        symbol: str = "XAUUSD",
        config=None,
        max_steps: int = 1000,
    ) -> None:
        """Initialise l'environnement de trading.

        Args:
            data (Optional[np.ndarray]): Matrice OHLCV enrichie, avec au moins
                les colonnes open/high/low/close aux quatre premières positions.
            symbol (str): Symbole évalué.
            config (Any | None): Configuration MuZero utilisée pour les bonus
                et pénalités de récompense.
            max_steps (int): Nombre maximum de pas par épisode.
        """
        self.symbol = symbol
        self.spec = SymbolSpec.for_symbol(symbol)
        self.max_steps_per_episode = max_steps
        self.commission_rate = 0.00005

        if config:
            self.quality_mult = config.quality_trade_bonus
            self.final_growth_bonus = config.final_growth_bonus
            self.final_growth_threshold = config.final_growth_threshold
            self.dd_penalty_rate = config.drawdown_time_penalty_rate
            self.max_dd_penalty = config.max_drawdown_penalty
            self.loss_mult = config.loss_penalty_multiplier
            self.slbe_bonus = config.slbe_activation_bonus
        else:
            self.quality_mult = 10.0
            self.final_growth_bonus = 50.0
            self.final_growth_threshold = 0.10
            self.dd_penalty_rate = 0.2
            self.max_dd_penalty = 10.0
            self.loss_mult = 2.0
            self.slbe_bonus = 6.0

        self.data = data if data is not None else self._generate_synthetic_data()
        self.base_feature_count = self.data.shape[1]
        self.observation_dim = self.base_feature_count + 6
        self._reset_state()

    def _generate_synthetic_data(self, n_steps: int = 5000) -> np.ndarray:
        """Génère des données synthétiques pour les tests locaux.

        Args:
            n_steps (int): Longueur de série souhaitée.

        Returns:
            np.ndarray: Données OHLCV synthétiques.
        """
        np.random.seed(42)
        price = 2650.0
        rows = []
        for _ in range(n_steps):
            change = np.random.randn() * 2.0
            open_price = price
            high_price = price + abs(np.random.randn() * 3.0)
            low_price = price - abs(np.random.randn() * 3.0)
            close_price = price + change
            volume = np.random.randint(100, 10_000)
            ema_200 = price + np.random.randn() * 5.0
            rows.append([open_price, high_price, low_price, close_price, volume, ema_200])
            price = close_price
        return np.array(rows, dtype=np.float32)

    def _reset_state(self) -> None:
        """Réinitialise l'état interne de l'épisode."""
        self.current_step = min(100, max(0, len(self.data) - 2))
        self.start_step = self.current_step
        self.balance = self.spec.initial_balance
        self.peak_equity = self.spec.initial_balance
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.slbe_active = False
        self.slbe_price = 0.0
        self.steps_in_drawdown = 0
        self.steps_since_last_trade = 0
        self.total_trades = 0
        self.total_profitable = 0
        self.gross_profit_pct = 0.0
        self.gross_loss_pct = 0.0
        self.net_realized_pnl_pct = 0.0
        self.split_count = 0
        self.secured_count = 0
        self.equity_curve = [self.spec.initial_balance]

    def _record_closed_trade(self, realized_pnl: float) -> float:
        """Enregistre un trade clôturé dans les métriques d'épisode.

        Args:
            realized_pnl (float): Profit ou perte réalisé sur la clôture.

        Returns:
            float: Profit ou perte réalisé en pourcentage du capital initial.
        """
        realized_pct = (realized_pnl / self.spec.initial_balance) * 100.0
        self.balance += realized_pnl
        self.total_trades += 1
        self.net_realized_pnl_pct += realized_pct

        if realized_pnl > 0:
            self.total_profitable += 1
            self.gross_profit_pct += realized_pct
        elif realized_pnl < 0:
            self.gross_loss_pct += abs(realized_pct)

        return realized_pct

    @staticmethod
    def _price_return(entry_price: float, exit_price: float, direction: float) -> float:
        """Calcule le rendement signé d'une position.

        Args:
            entry_price (float): Prix moyen d'entrée.
            exit_price (float): Prix de sortie.
            direction (float): Sens de position, positif pour long, négatif pour short.

        Returns:
            float: Rendement signé de la transaction.
        """
        base_price = max(entry_price, 1e-8)
        if direction >= 0:
            return (exit_price - entry_price) / base_price
        return (entry_price - exit_price) / base_price

    def _realize_position(self, price: float, close_size: float | None = None) -> tuple[float, float]:
        """Réalise tout ou partie d'une position ouverte.

        Args:
            price (float): Prix de clôture.
            close_size (float | None): Notionnel à fermer. ``None`` ferme tout.

        Returns:
            tuple[float, float]: PnL réalisé et rendement relatif de la portion fermée.
        """
        if self.position_size == 0:
            return 0.0, 0.0

        current_notional = abs(self.position_size)
        realized_notional = current_notional if close_size is None else min(current_notional, close_size)
        direction = 1.0 if self.position_size > 0 else -1.0
        trade_ret = self._price_return(self.avg_entry_price, price, direction)
        pnl = trade_ret * realized_notional
        commission = realized_notional * self.commission_rate
        realized_trade = pnl - commission
        self._record_closed_trade(realized_trade)

        if close_size is None or realized_notional >= current_notional:
            self.position_size = 0.0
            self.avg_entry_price = 0.0
            self.slbe_active = False
            self.slbe_price = 0.0
        elif self.position_size > 0:
            self.position_size -= realized_notional
        else:
            self.position_size += realized_notional

        return realized_trade, trade_ret

    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement pour un nouvel épisode.

        Args:
            seed (Any | None): Graine Gymnasium, ignorée ici.
            options (Any | None): Options Gymnasium, ignorées ici.

        Returns:
            tuple[np.ndarray, dict]: Observation initiale et méta-informations.
        """
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Construit le vecteur d'observation courant.

        Returns:
            np.ndarray: Observation enrichie avec l'état de position.
        """
        base = self.data[self.current_step].copy()

        pos_state = 0.0
        if self.position_size > 0:
            pos_state = 1.0
        elif self.position_size < 0:
            pos_state = -1.0

        pnl_pct = 0.0
        if self.position_size != 0:
            price = self.data[self.current_step, 3]
            pnl_pct = self._price_return(self.avg_entry_price, price, self.position_size)

        slbe_state = 1.0 if self.slbe_active else 0.0
        hour_feat = (self.current_step % 24) / 23.0
        day_feat = ((self.current_step // 24) % 5) / 4.0
        high_price = self.data[self.current_step, 1]
        low_price = self.data[self.current_step, 2]
        close_price = self.data[self.current_step, 3]
        vol = min((high_price - low_price) / max(close_price, 1e-8) * 100.0, 1.0) if close_price > 0 else 0.0

        extra = np.array([pos_state, pnl_pct, slbe_state, hour_feat, day_feat, vol], dtype=np.float32)
        return np.concatenate([base, extra])

    def step(self, action: int):
        """Exécute un pas de trading.

        Args:
            action (int): Action discrète MuZero.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Sortie Gymnasium.
        """
        price = self.data[self.current_step, 3]
        ema_200 = self.data[self.current_step, 5] if self.data.shape[1] > 5 else price
        reward = 0.0
        done = False
        realized_pnl = 0.0

        trade_notional = self.spec.trade_size
        max_position = (2 + self.secured_count) * trade_notional

        if self.slbe_active and self.position_size != 0:
            hit = False
            if self.position_size > 0 and price <= self.slbe_price:
                hit = True
            elif self.position_size < 0 and price >= self.slbe_price:
                hit = True
            if hit:
                realized_trade, _ = self._realize_position(price)
                realized_pnl += realized_trade
                reward += 1.0

        if not self.slbe_active and self.position_size != 0:
            unr = self._price_return(self.avg_entry_price, price, self.position_size)
            if unr >= 0.005:
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                self.secured_count += 1
                reward += self.slbe_bonus

        if self.position_size == 0 and action in [SPLIT, CLOSE]:
            action = HOLD

        if action == BUY and price < ema_200:
            action = HOLD
        elif action == SELL and price > ema_200:
            action = HOLD

        if action == BUY:
            if self.position_size < 0:
                realized_trade, _ = self._realize_position(price)
                realized_pnl += realized_trade

            if self.position_size == 0:
                self.balance -= trade_notional * self.commission_rate
                self.position_size = trade_notional
                self.avg_entry_price = price
                self.split_count = 0
            elif self.position_size < max_position:
                curr_pnl = self._price_return(self.avg_entry_price, price, 1.0)
                if curr_pnl > 0.001:
                    self.balance -= trade_notional * self.commission_rate
                    total_value = (self.position_size * self.avg_entry_price) + (trade_notional * price)
                    self.position_size += trade_notional
                    self.avg_entry_price = total_value / self.position_size
                    reward += 0.1
                else:
                    reward -= 0.1

        elif action == SELL:
            if self.position_size > 0:
                realized_trade, _ = self._realize_position(price)
                realized_pnl += realized_trade

            if self.position_size == 0:
                self.balance -= trade_notional * self.commission_rate
                self.position_size = -trade_notional
                self.avg_entry_price = price
                self.split_count = 0
            elif self.position_size > -max_position:
                curr_pnl = self._price_return(self.avg_entry_price, price, -1.0)
                if curr_pnl > 0.001:
                    self.balance -= trade_notional * self.commission_rate
                    total_value = (abs(self.position_size) * self.avg_entry_price) + (trade_notional * price)
                    self.position_size -= trade_notional
                    self.avg_entry_price = total_value / abs(self.position_size)
                    reward += 0.1
                else:
                    reward -= 0.1

        elif action == SPLIT and abs(self.position_size) > 0:
            realized_trade, trade_ret = self._realize_position(price, abs(self.position_size) * 0.5)
            realized_pnl += realized_trade

            if self.split_count < 3:
                if trade_ret > 0.01:
                    reward += self.quality_mult
                    self.split_count += 1
                elif realized_trade > 0:
                    reward += 1.0
                    self.split_count += 1

            if self.position_size != 0 and not self.slbe_active:
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                reward += 2.0

        elif action == CLOSE and abs(self.position_size) > 0:
            realized_trade, trade_ret = self._realize_position(price)
            realized_pnl += realized_trade

            if trade_ret > 0.02:
                reward += self.quality_mult * 1.5
            elif trade_ret > 0.01:
                reward += self.quality_mult
            elif realized_trade > 0:
                reward += 1.0
            else:
                reward += (realized_trade / self.spec.initial_balance) * 100.0

        unrealized = 0.0
        if self.position_size != 0:
            unrealized = self._price_return(self.avg_entry_price, price, self.position_size) * abs(self.position_size)

        equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        if unrealized < 0:
            self.steps_in_drawdown += 1
            reward -= self.dd_penalty_rate * (self.steps_in_drawdown / 20)
        else:
            self.steps_in_drawdown = 0

        self.steps_since_last_trade += 1
        if action in [BUY, SELL, SPLIT, CLOSE]:
            self.steps_since_last_trade = 0
        elif self.steps_since_last_trade > 100:
            reward -= 1.0

        if self.position_size != 0:
            reward += (unrealized / self.spec.initial_balance) * 100.0 * 0.02

        if drawdown > 0.05:
            reward -= self.max_dd_penalty

        if realized_pnl > 0:
            reward += (realized_pnl / self.spec.initial_balance) * 100.0
        elif realized_pnl < 0:
            reward += (realized_pnl / self.spec.initial_balance) * 100.0 * self.loss_mult

        self.current_step += 1
        if self.current_step - self.start_step >= self.max_steps_per_episode:
            done = True
        if self.current_step >= len(self.data) - 1:
            done = True

        if done and self.position_size != 0:
            final_realized, _ = self._realize_position(price)
            realized_pnl += final_realized
            if final_realized > 0:
                reward += (final_realized / self.spec.initial_balance) * 100.0
            elif final_realized < 0:
                reward += (final_realized / self.spec.initial_balance) * 100.0 * self.loss_mult
            equity = self.balance
            self.peak_equity = max(self.peak_equity, equity)
            drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        if done:
            final_growth = (equity - self.spec.initial_balance) / self.spec.initial_balance
            if final_growth >= self.final_growth_threshold:
                reward += self.final_growth_bonus
                logger.info(
                    "Bonus de croissance finale active: %.2f%% -> +%.2f",
                    final_growth * 100.0,
                    self.final_growth_bonus,
                )

        self.equity_curve.append(equity)
        info = {
            "balance": self.balance,
            "equity": equity,
            "step": self.current_step,
            "realized_pnl": realized_pnl,
            "drawdown": drawdown,
            "slbe_active": self.slbe_active,
            "total_trades": self.total_trades,
            "win_rate": self.total_profitable / max(self.total_trades, 1),
        }

        obs = self._get_observation()
        return obs, reward, done, False, info

    def get_summary(self) -> dict:
        """Retourne les métriques consolidées de l'épisode.

        Returns:
            dict: Résumé de performance de l'épisode.
        """
        equity = self.equity_curve[-1] if self.equity_curve else self.spec.initial_balance
        return {
            "symbol": self.symbol,
            "total_trades": self.total_trades,
            "profitable_trades": self.total_profitable,
            "win_rate": self.total_profitable / max(self.total_trades, 1),
            "final_equity": equity,
            "return_pct": (equity - self.spec.initial_balance) / self.spec.initial_balance * 100.0,
            "gross_profit_pct": self.gross_profit_pct,
            "gross_loss_pct": self.gross_loss_pct,
            "net_realized_pct": self.net_realized_pnl_pct,
            "steps": self.current_step - self.start_step,
        }
