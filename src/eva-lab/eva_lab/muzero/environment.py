"""Environnement de trading MuZero pour THE HIVE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from eva_lab.training_utils import (
    resolve_feature_profile,
    resolve_model_family,
    resolve_position_mechanics_profile,
)

logger = logging.getLogger(__name__)

# Actions
HOLD = 0
BUY = 1
SELL = 2
SPLIT = 3
CLOSE = 4

ACTION_NAMES = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]
MARKET_COL_EMA_200 = 5
MARKET_COL_VWAP = 8
MARKET_COL_OBV = 9
MARKET_COL_MOMENTUM = 10
MARKET_COL_ADX = 15
MARKET_COL_ATR = 22
MARKET_COL_BB_PCT = 23


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

        self.horizon = str(getattr(config, "horizon", "intraday") or "intraday").lower()
        configured_family = getattr(config, "model_family", None)
        self.family = resolve_model_family(symbol=symbol, family=configured_family)
        self.feature_profile = resolve_feature_profile(self.horizon, self.family)
        self.position_mechanics_profile = resolve_position_mechanics_profile(
            self.horizon,
            self.family,
        )
        self.mechanics_profile_version = str(
            self.position_mechanics_profile.get("profile_version") or "v1"
        )
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
        self.action_counts = {name: 0 for name in ACTION_NAMES}
        self.long_entries = 0
        self.short_entries = 0
        self.ema200_blocked_buy = 0
        self.ema200_blocked_sell = 0
        self.entry_blocked_vwap = 0
        self.entry_blocked_adx = 0
        self.entry_blocked_obv = 0
        self.actions_above_vwap = 0
        self.actions_below_vwap = 0
        self.actions_low_adx = 0
        self.obv_divergent_actions = 0
        self.hold_in_trend_count = 0
        self.hold_in_range_count = 0
        self.hold_under_trend_penalty_count = 0
        self.hold_streak_total = 0
        self.hold_streak_count = 0
        self._current_hold_streak = 0
        self.pyramids_opened = 0
        self.pyramids_rejected = 0
        self.pyramid_profitable_count = 0
        self.pyramid_loss_count = 0
        self.position_pyramids = 0
        self.split_executed = 0
        self.split_profitable_count = 0
        self.slbe_triggered = 0
        self.slbe_hit = 0
        self.slbe_profitable_exits = 0
        self.position_had_slbe = False
        self.close_winner_count = 0
        self.close_loser_count = 0
        self.tp_like_exit_count = 0
        self.split_rejected = 0
        self.split_rejected_no_value = 0
        self.inactive_episode_penalties = 0
        self.insufficient_entry_penalties = 0
        self.directional_imbalance_penalties = 0
        self.realized_close_bonus_count = 0
        self.realized_split_bonus_count = 0
        self.slbe_exit_bonus_count = 0

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

    def _flush_hold_streak(self) -> None:
        """Consolide une serie de HOLD avant changement de regime."""
        if self._current_hold_streak <= 0:
            return
        self.hold_streak_total += self._current_hold_streak
        self.hold_streak_count += 1
        self._current_hold_streak = 0

    def _get_market_context(self, step_index: int | None = None) -> dict[str, float]:
        """Construit un contexte de marche minimal depuis les features existantes.

        Args:
            step_index (int | None): Index de barre a lire. Repli sur le pas courant.

        Returns:
            dict[str, float]: Contexte synthetique pour les filtres et l'audit.
        """
        index = self.current_step if step_index is None else max(0, min(step_index, len(self.data) - 1))
        row = self.data[index]
        previous_row = self.data[max(0, index - 1)]
        close_price = float(row[3])
        ema_200 = float(row[MARKET_COL_EMA_200]) if self.base_feature_count > MARKET_COL_EMA_200 else close_price
        vwap = float(row[MARKET_COL_VWAP]) if self.base_feature_count > MARKET_COL_VWAP else close_price
        obv = float(row[MARKET_COL_OBV]) if self.base_feature_count > MARKET_COL_OBV else 0.0
        previous_obv = float(previous_row[MARKET_COL_OBV]) if self.base_feature_count > MARKET_COL_OBV else obv
        adx = float(row[MARKET_COL_ADX]) if self.base_feature_count > MARKET_COL_ADX else 0.0
        atr = float(row[MARKET_COL_ATR]) if self.base_feature_count > MARKET_COL_ATR else 0.0
        bb_pct = float(row[MARKET_COL_BB_PCT]) if self.base_feature_count > MARKET_COL_BB_PCT else 0.5
        momentum = float(row[MARKET_COL_MOMENTUM]) if self.base_feature_count > MARKET_COL_MOMENTUM else 0.0

        return {
            "close": close_price,
            "ema_200": ema_200,
            "vwap": vwap,
            "price_vs_vwap": (close_price - vwap) / max(abs(vwap), 1e-8),
            "obv": obv,
            "obv_slope": obv - previous_obv,
            "obv_divergence": float(np.sign(momentum) != np.sign(obv - previous_obv)),
            "adx": adx,
            "atr": atr,
            "atr_pct": atr / max(abs(close_price), 1e-8),
            "bb_width_proxy": abs(bb_pct - 0.5) * 2.0,
            "momentum": momentum,
        }

    def _record_action_context(self, action: int, context: dict[str, float]) -> None:
        """Journalise les signaux de contexte dominants pour l'action courante.

        Args:
            action (int): Action finale executee par l'environnement.
            context (dict[str, float]): Contexte de marche courant.
        """
        if context["price_vs_vwap"] >= 0:
            self.actions_above_vwap += 1
        else:
            self.actions_below_vwap += 1

        min_adx = float(
            (self.position_mechanics_profile.get("entry_filter") or {}).get("min_adx", 0.0)
        )
        trend_adx = float(
            (self.position_mechanics_profile.get("entry_filter") or {}).get("trend_adx", min_adx)
        )
        if context["adx"] < min_adx:
            self.actions_low_adx += 1
        if action == HOLD:
            if context["adx"] >= trend_adx:
                self.hold_in_trend_count += 1
            else:
                self.hold_in_range_count += 1
        if action == BUY and context["obv_slope"] <= 0:
            self.obv_divergent_actions += 1
        elif action == SELL and context["obv_slope"] >= 0:
            self.obv_divergent_actions += 1

    def _apply_entry_filter(self, action: int, context: dict[str, float]) -> tuple[int, str | None]:
        """Filtre une entree directionnelle selon le profil horizon/famille.

        Args:
            action (int): Action demandee par le modele.
            context (dict[str, float]): Contexte courant du marche.

        Returns:
            tuple[int, str | None]: Action finale et raison du veto eventuel.
        """
        if action not in (BUY, SELL):
            return action, None

        entry_filter = dict(self.position_mechanics_profile.get("entry_filter") or {})
        ema_mode = str(entry_filter.get("ema_mode", "strict")).lower()
        require_vwap_alignment = bool(entry_filter.get("require_vwap_alignment", False))
        require_obv_confirmation = bool(entry_filter.get("require_obv_confirmation", False))
        allow_trend_fallback = bool(entry_filter.get("allow_trend_fallback", False))
        min_adx = float(entry_filter.get("min_adx", 0.0) or 0.0)
        trend_adx = float(entry_filter.get("trend_adx", min_adx) or min_adx)
        fallback_direction_ok = (
            allow_trend_fallback
            and context["adx"] >= max(min_adx * 0.6, trend_adx - 4.0, 0.0)
            and abs(context["momentum"]) >= max(context["atr_pct"], 1e-5)
        )

        if context["adx"] < min_adx and not fallback_direction_ok:
            self.entry_blocked_adx += 1
            return HOLD, "adx"

        if action == BUY:
            if ema_mode == "strict" and context["close"] < context["ema_200"]:
                self.ema200_blocked_buy += 1
                return HOLD, "ema200"
            if ema_mode == "moderate" and context["close"] < context["ema_200"] and context["price_vs_vwap"] < 0:
                self.ema200_blocked_buy += 1
                return HOLD, "ema200"
            if require_vwap_alignment and context["price_vs_vwap"] < 0 and not fallback_direction_ok:
                self.entry_blocked_vwap += 1
                return HOLD, "vwap"
            if require_obv_confirmation and context["obv_slope"] <= 0 and not fallback_direction_ok:
                self.entry_blocked_obv += 1
                return HOLD, "obv"
            return BUY, None

        if ema_mode == "strict" and context["close"] > context["ema_200"]:
            self.ema200_blocked_sell += 1
            return HOLD, "ema200"
        if ema_mode == "moderate" and context["close"] > context["ema_200"] and context["price_vs_vwap"] > 0:
            self.ema200_blocked_sell += 1
            return HOLD, "ema200"
        if require_vwap_alignment and context["price_vs_vwap"] > 0 and not fallback_direction_ok:
            self.entry_blocked_vwap += 1
            return HOLD, "vwap"
        if require_obv_confirmation and context["obv_slope"] >= 0 and not fallback_direction_ok:
            self.entry_blocked_obv += 1
            return HOLD, "obv"
        return SELL, None

    def _get_unrealized_return(self, price: float) -> float:
        """Calcule le rendement latent courant de la position.

        Args:
            price (float): Prix de marche courant.

        Returns:
            float: Rendement signe de la position ouverte.
        """
        if self.position_size == 0:
            return 0.0
        return self._price_return(self.avg_entry_price, price, self.position_size)

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

        full_close = close_size is None or realized_notional >= current_notional
        if full_close:
            if self.position_pyramids > 0:
                if realized_trade > 0:
                    self.pyramid_profitable_count += 1
                elif realized_trade < 0:
                    self.pyramid_loss_count += 1
            if self.slbe_active:
                self.slbe_hit += 1
                if realized_trade > 0:
                    self.slbe_profitable_exits += 1
            self.position_size = 0.0
            self.avg_entry_price = 0.0
            self.slbe_active = False
            self.slbe_price = 0.0
            self.position_pyramids = 0
            self.position_had_slbe = False
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
        context = self._get_market_context()
        price = context["close"]
        reward = 0.0
        done = False
        realized_pnl = 0.0

        trade_notional = self.spec.trade_size
        pyramiding_policy = dict(self.position_mechanics_profile.get("pyramiding_policy") or {})
        split_policy = dict(self.position_mechanics_profile.get("split_policy") or {})
        slbe_policy = dict(self.position_mechanics_profile.get("slbe_policy") or {})
        close_policy = dict(self.position_mechanics_profile.get("close_policy") or {})
        hold_policy = dict(self.position_mechanics_profile.get("hold_policy") or {})
        activity_policy = dict(self.position_mechanics_profile.get("activity_policy") or {})
        directional_policy = dict(self.position_mechanics_profile.get("directional_policy") or {})
        reward_policy = dict(self.position_mechanics_profile.get("reward_policy") or {})
        realized_reward_multiplier = float(
            reward_policy.get("realized_pnl_multiplier", 1.0) or 1.0
        )
        close_realized_multiplier = float(
            reward_policy.get("close_realized_multiplier", realized_reward_multiplier) or realized_reward_multiplier
        )
        split_realized_multiplier = float(
            reward_policy.get("split_realized_multiplier", realized_reward_multiplier) or realized_reward_multiplier
        )
        hold_drag_multiplier = float(reward_policy.get("hold_drag_multiplier", 0.0) or 0.0)
        pyramid_reject_penalty = float(reward_policy.get("pyramid_failure_penalty", 0.1) or 0.1)
        pyramid_negative_exit_penalty = float(
            reward_policy.get("pyramid_negative_exit_penalty", 0.0) or 0.0
        )
        max_position = (1 + int(pyramiding_policy.get("max_additions", 1))) * trade_notional

        if self.slbe_active and self.position_size != 0:
            hit = False
            if self.position_size > 0 and price <= self.slbe_price:
                hit = True
            elif self.position_size < 0 and price >= self.slbe_price:
                hit = True
            if hit:
                realized_trade, _ = self._realize_position(price)
                realized_pnl += realized_trade
                reward += 1.0 + float(slbe_policy.get("exit_bonus", 0.0) or 0.0)
                if realized_trade > 0:
                    self.slbe_exit_bonus_count += 1

        if not self.slbe_active and self.position_size != 0:
            unr = self._get_unrealized_return(price)
            activation_return = float(slbe_policy.get("activation_return", 0.005) or 0.005)
            if unr >= activation_return:
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                self.position_had_slbe = True
                self.slbe_triggered += 1
                self.secured_count += 1
                reward += float(slbe_policy.get("bonus", self.slbe_bonus) or self.slbe_bonus)

        requested_action = action

        if self.position_size == 0 and action in [SPLIT, CLOSE]:
            action = HOLD

        action, veto_reason = self._apply_entry_filter(action, context)

        final_action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"ACT_{action}"
        self.action_counts[final_action_name] = self.action_counts.get(final_action_name, 0) + 1
        self._record_action_context(action, context)

        if action == BUY:
            self._flush_hold_streak()
            if self.position_size < 0:
                had_pyramids = self.position_pyramids > 0
                realized_trade, _ = self._realize_position(price)
                realized_pnl += realized_trade
                if realized_trade < 0 and had_pyramids and pyramid_negative_exit_penalty > 0:
                    reward -= pyramid_negative_exit_penalty

            if self.position_size == 0:
                self.balance -= trade_notional * self.commission_rate
                self.position_size = trade_notional
                self.avg_entry_price = price
                self.split_count = 0
                self.position_pyramids = 0
                self.position_had_slbe = False
                self.long_entries += 1
            elif self.position_size < max_position:
                curr_pnl = self._price_return(self.avg_entry_price, price, 1.0)
                min_profit_to_add = float(pyramiding_policy.get("min_profit_to_add", 0.001) or 0.001)
                max_additions = int(pyramiding_policy.get("max_additions", 1) or 1)
                if curr_pnl > min_profit_to_add and self.position_pyramids < max_additions:
                    self.balance -= trade_notional * self.commission_rate
                    total_value = (self.position_size * self.avg_entry_price) + (trade_notional * price)
                    self.position_size += trade_notional
                    self.avg_entry_price = total_value / self.position_size
                    self.position_pyramids += 1
                    self.pyramids_opened += 1
                    reward += float(pyramiding_policy.get("reward_bonus", 0.1) or 0.1)
                else:
                    self.pyramids_rejected += 1
                    reward -= pyramid_reject_penalty

        elif action == SELL:
            self._flush_hold_streak()
            if self.position_size > 0:
                had_pyramids = self.position_pyramids > 0
                realized_trade, _ = self._realize_position(price)
                realized_pnl += realized_trade
                if realized_trade < 0 and had_pyramids and pyramid_negative_exit_penalty > 0:
                    reward -= pyramid_negative_exit_penalty

            if self.position_size == 0:
                self.balance -= trade_notional * self.commission_rate
                self.position_size = -trade_notional
                self.avg_entry_price = price
                self.split_count = 0
                self.position_pyramids = 0
                self.position_had_slbe = False
                self.short_entries += 1
            elif self.position_size > -max_position:
                curr_pnl = self._price_return(self.avg_entry_price, price, -1.0)
                min_profit_to_add = float(pyramiding_policy.get("min_profit_to_add", 0.001) or 0.001)
                max_additions = int(pyramiding_policy.get("max_additions", 1) or 1)
                if curr_pnl > min_profit_to_add and self.position_pyramids < max_additions:
                    self.balance -= trade_notional * self.commission_rate
                    total_value = (abs(self.position_size) * self.avg_entry_price) + (trade_notional * price)
                    self.position_size -= trade_notional
                    self.avg_entry_price = total_value / abs(self.position_size)
                    self.position_pyramids += 1
                    self.pyramids_opened += 1
                    reward += float(pyramiding_policy.get("reward_bonus", 0.1) or 0.1)
                else:
                    self.pyramids_rejected += 1
                    reward -= pyramid_reject_penalty

        elif action == SPLIT and abs(self.position_size) > 0:
            self._flush_hold_streak()
            max_splits = int(split_policy.get("max_splits", 3) or 3)
            min_trade_return = float(split_policy.get("min_trade_return", 0.01) or 0.01)
            min_realized_pct = float(split_policy.get("min_realized_pct", 0.0) or 0.0)
            current_trade_return = self._get_unrealized_return(price)
            if self.split_count < max_splits and current_trade_return >= min_trade_return:
                realized_trade, trade_ret = self._realize_position(price, abs(self.position_size) * 0.5)
                realized_pnl += realized_trade
                realized_pct = (realized_trade / self.spec.initial_balance) * 100.0
                self.split_count += 1
                self.split_executed += 1
                if realized_pct >= min_realized_pct:
                    reward += self.quality_mult + (realized_pct * split_realized_multiplier)
                    self.split_profitable_count += 1
                    self.realized_split_bonus_count += 1
                elif realized_trade > 0:
                    reward += max(0.25, realized_pct * split_realized_multiplier)
                else:
                    self.split_rejected_no_value += 1
                    reward -= float(split_policy.get("failure_penalty", 0.25) or 0.25)
            else:
                self.split_rejected += 1
                self.split_rejected_no_value += 1
                reward -= float(split_policy.get("failure_penalty", 0.25) or 0.25)

            if (
                self.position_size != 0
                and not self.slbe_active
                and bool(split_policy.get("slbe_after_split", True))
            ):
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                self.position_had_slbe = True
                self.slbe_triggered += 1
                reward += 2.0

        elif action == CLOSE and abs(self.position_size) > 0:
            self._flush_hold_streak()
            had_pyramids = self.position_pyramids > 0
            realized_trade, trade_ret = self._realize_position(price)
            realized_pnl += realized_trade

            strong_winner_threshold = float(close_policy.get("strong_winner_threshold", 0.02) or 0.02)
            winner_threshold = float(close_policy.get("winner_threshold", 0.01) or 0.01)
            tp_like_threshold = float(close_policy.get("tp_like_threshold", winner_threshold) or winner_threshold)
            realized_pct = (realized_trade / self.spec.initial_balance) * 100.0
            if trade_ret > strong_winner_threshold:
                reward += self.quality_mult * 1.5 + (realized_pct * close_realized_multiplier)
                self.realized_close_bonus_count += 1
            elif trade_ret > winner_threshold:
                reward += self.quality_mult + (realized_pct * close_realized_multiplier)
                self.realized_close_bonus_count += 1
            elif realized_trade > 0:
                reward += max(0.25, realized_pct * close_realized_multiplier)
            else:
                reward += realized_pct
            if realized_trade > 0:
                self.close_winner_count += 1
                if trade_ret >= tp_like_threshold:
                    self.tp_like_exit_count += 1
            elif realized_trade < 0:
                self.close_loser_count += 1
                if had_pyramids and pyramid_negative_exit_penalty > 0:
                    reward -= pyramid_negative_exit_penalty

        else:
            self._current_hold_streak += 1
            stale_after = int(hold_policy.get("stale_penalty_after_steps", 100) or 100)
            stale_penalty = float(hold_policy.get("stale_penalty", 1.0) or 1.0)
            trend_penalty = float(hold_policy.get("trend_penalty", 0.0) or 0.0)
            range_penalty = float(hold_policy.get("range_penalty", 0.0) or 0.0)
            trend_adx = float(
                (self.position_mechanics_profile.get("entry_filter") or {}).get(
                    "trend_adx",
                    (self.position_mechanics_profile.get("entry_filter") or {}).get("min_adx", 20.0),
                )
                or 20.0
            )
            if self.steps_since_last_trade > stale_after:
                reward -= stale_penalty
            if context["adx"] >= trend_adx:
                reward -= trend_penalty
                self.hold_under_trend_penalty_count += 1
                if self.position_size != 0:
                    profitable_return = max(0.0, self._get_unrealized_return(price))
                    if profitable_return > 0:
                        reward -= profitable_return * 100.0 * hold_drag_multiplier
            else:
                reward -= range_penalty

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

        if self.position_size != 0:
            reward += (unrealized / self.spec.initial_balance) * 100.0 * 0.02

        if drawdown > 0.05:
            reward -= self.max_dd_penalty

        if realized_pnl > 0:
            reward += (realized_pnl / self.spec.initial_balance) * 100.0 * realized_reward_multiplier
        elif realized_pnl < 0:
            reward += (realized_pnl / self.spec.initial_balance) * 100.0 * self.loss_mult
            if self.position_pyramids > 0 and pyramid_negative_exit_penalty > 0:
                reward -= pyramid_negative_exit_penalty

        self.current_step += 1
        if self.current_step - self.start_step >= self.max_steps_per_episode:
            done = True
        if self.current_step >= len(self.data) - 1:
            done = True

        if done and self.position_size != 0:
            had_pyramids = self.position_pyramids > 0
            final_realized, _ = self._realize_position(price)
            realized_pnl += final_realized
            if final_realized > 0:
                reward += (final_realized / self.spec.initial_balance) * 100.0
            elif final_realized < 0:
                reward += (final_realized / self.spec.initial_balance) * 100.0 * self.loss_mult
                if had_pyramids and pyramid_negative_exit_penalty > 0:
                    reward -= pyramid_negative_exit_penalty
            equity = self.balance
            self.peak_equity = max(self.peak_equity, equity)
            drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        if done:
            self._flush_hold_streak()
            final_growth = (equity - self.spec.initial_balance) / self.spec.initial_balance
            if final_growth >= self.final_growth_threshold:
                reward += self.final_growth_bonus
                logger.info(
                    "Bonus de croissance finale active: %.2f%% -> +%.2f",
                    final_growth * 100.0,
                    self.final_growth_bonus,
                )
            total_entries = self.long_entries + self.short_entries
            min_entries = int(activity_policy.get("min_entries", 0) or 0)
            if total_entries <= 0:
                reward -= float(activity_policy.get("inactive_episode_penalty", 0.0) or 0.0)
                self.inactive_episode_penalties += 1
            elif min_entries > 0 and total_entries < min_entries:
                reward -= float(activity_policy.get("insufficient_entries_penalty", 0.0) or 0.0)
                self.insufficient_entry_penalties += 1

            min_entry_share = float(directional_policy.get("min_entry_share", 0.0) or 0.0)
            max_imbalance = float(directional_policy.get("max_imbalance", 1.0) or 1.0)
            imbalance_penalty = float(directional_policy.get("imbalance_penalty", 0.0) or 0.0)
            if total_entries > 0:
                long_share = self.long_entries / total_entries
                short_share = self.short_entries / total_entries
                imbalance = abs(long_share - short_share)
                if long_share < min_entry_share or short_share < min_entry_share or imbalance > max_imbalance:
                    reward -= imbalance_penalty
                    self.directional_imbalance_penalties += 1

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
            "requested_action": ACTION_NAMES[requested_action] if 0 <= requested_action < len(ACTION_NAMES) else f"ACT_{requested_action}",
            "final_action": final_action_name,
            "veto_reason": veto_reason,
            "family": self.family,
            "feature_profile": self.feature_profile.get("profile_name"),
        }

        obs = self._get_observation()
        return obs, reward, done, False, info

    def get_summary(self) -> dict:
        """Retourne les métriques consolidées de l'épisode.

        Returns:
            dict: Résumé de performance de l'épisode.
        """
        self._flush_hold_streak()
        equity = self.equity_curve[-1] if self.equity_curve else self.spec.initial_balance
        split_efficiency = (
            self.split_profitable_count / self.split_executed
            if self.split_executed > 0
            else 0.0
        )
        pyramid_efficiency = (
            self.pyramid_profitable_count / self.pyramids_opened
            if self.pyramids_opened > 0
            else 0.0
        )
        slbe_capture_rate = (
            self.slbe_profitable_exits / self.slbe_triggered
            if self.slbe_triggered > 0
            else 0.0
        )
        hold_drag_score = (
            self.hold_under_trend_penalty_count / max(self.current_step - self.start_step, 1)
        )
        meaningful_exit_count = (
            self.close_winner_count
            + self.close_loser_count
            + self.slbe_profitable_exits
            + self.tp_like_exit_count
        )
        close_quality_score = (
            self.close_winner_count / max(self.close_winner_count + self.close_loser_count, 1)
            if (self.close_winner_count + self.close_loser_count) > 0
            else 0.0
        )
        hold_streak_mean = (
            self.hold_streak_total / self.hold_streak_count
            if self.hold_streak_count > 0
            else 0.0
        )
        mechanics_metrics = {
            "split_efficiency": split_efficiency,
            "pyramid_efficiency": pyramid_efficiency,
            "slbe_capture_rate": slbe_capture_rate,
            "hold_drag_score": hold_drag_score,
            "hold_drag_score_normalized": hold_drag_score,
            "close_quality_score": close_quality_score,
            "mechanics_profile_version": self.mechanics_profile_version,
            "pyramids_opened": self.pyramids_opened,
            "pyramids_rejected": self.pyramids_rejected,
            "pyramid_profitable_count": self.pyramid_profitable_count,
            "slbe_triggered": self.slbe_triggered,
            "slbe_hit": self.slbe_hit,
            "slbe_profitable_exits": self.slbe_profitable_exits,
            "split_executed": self.split_executed,
            "split_profitable_count": self.split_profitable_count,
            "split_rejected": self.split_rejected,
            "split_rejected_no_value": self.split_rejected_no_value,
            "close_winner_count": self.close_winner_count,
            "close_loser_count": self.close_loser_count,
            "meaningful_exit_count": meaningful_exit_count,
            "hold_streak_mean": hold_streak_mean,
            "hold_under_trend_penalty_count": self.hold_under_trend_penalty_count,
            "tp_like_exit_count": self.tp_like_exit_count,
            "inactive_episode_penalties": self.inactive_episode_penalties,
            "insufficient_entry_penalties": self.insufficient_entry_penalties,
            "directional_imbalance_penalties": self.directional_imbalance_penalties,
            "realized_close_bonus_count": self.realized_close_bonus_count,
            "realized_split_bonus_count": self.realized_split_bonus_count,
            "slbe_exit_bonus_count": self.slbe_exit_bonus_count,
            "entry_blocked_vwap": self.entry_blocked_vwap,
            "entry_blocked_adx": self.entry_blocked_adx,
            "entry_blocked_obv": self.entry_blocked_obv,
            "actions_above_vwap": self.actions_above_vwap,
            "actions_below_vwap": self.actions_below_vwap,
            "actions_low_adx": self.actions_low_adx,
            "obv_divergent_actions": self.obv_divergent_actions,
            "hold_in_trend_count": self.hold_in_trend_count,
            "hold_in_range_count": self.hold_in_range_count,
        }
        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "family": self.family,
            "feature_profile": self.feature_profile.get("profile_name"),
            "position_profile": self.position_mechanics_profile.get("profile_name"),
            "mechanics_profile_version": self.mechanics_profile_version,
            "total_trades": self.total_trades,
            "profitable_trades": self.total_profitable,
            "win_rate": self.total_profitable / max(self.total_trades, 1),
            "final_equity": equity,
            "return_pct": (equity - self.spec.initial_balance) / self.spec.initial_balance * 100.0,
            "gross_profit_pct": self.gross_profit_pct,
            "gross_loss_pct": self.gross_loss_pct,
            "net_realized_pct": self.net_realized_pnl_pct,
            "steps": self.current_step - self.start_step,
            "action_counts": dict(self.action_counts),
            "buy_actions": int(self.action_counts.get("BUY", 0)),
            "sell_actions": int(self.action_counts.get("SELL", 0)),
            "hold_actions": int(self.action_counts.get("HOLD", 0)),
            "split_actions": int(self.action_counts.get("SPLIT", 0)),
            "close_actions": int(self.action_counts.get("CLOSE", 0)),
            "long_entries": self.long_entries,
            "short_entries": self.short_entries,
            "ema200_blocked_buy": self.ema200_blocked_buy,
            "ema200_blocked_sell": self.ema200_blocked_sell,
            "entry_blocked_vwap": self.entry_blocked_vwap,
            "entry_blocked_adx": self.entry_blocked_adx,
            "entry_blocked_obv": self.entry_blocked_obv,
            "actions_above_vwap": self.actions_above_vwap,
            "actions_below_vwap": self.actions_below_vwap,
            "actions_low_adx": self.actions_low_adx,
            "obv_divergent_actions": self.obv_divergent_actions,
            "hold_in_trend_count": self.hold_in_trend_count,
            "hold_in_range_count": self.hold_in_range_count,
            "metrics_by_position_mechanics": mechanics_metrics,
        }
