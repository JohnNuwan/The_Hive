"""Environnement de trading MuZero pour THE HIVE."""

from __future__ import annotations

import logging
import math
from datetime import datetime
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
OBS_EXTRA_FEATURE_COUNT = 6
OBS_POSITION_STATE_INDEX = -OBS_EXTRA_FEATURE_COUNT - 3


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

    L'observation contient les colonnes de marchÃ© brutes suivies de six
    caractÃ©ristiques supplÃ©mentaires dÃ©crivant l'Ã©tat de position.
    """

    @staticmethod
    def resolve_runtime_family(symbol: str, configured_family: str | None = None) -> str:
        """Resout la famille runtime a utiliser pour un symbole.

        Un run multi-univers peut exposer ``model_family="mixed"`` au niveau
        global. Dans ce cas, retomber sur un profil unique `fx` pour tous les
        symboles fausse le masque racine, surtout autour de `EMA200`. On
        conserve donc la famille explicite seulement si elle est specifique,
        sinon on revient a la famille reelle du symbole.

        Args:
            symbol (str): Symbole de marche courant.
            configured_family (str | None): Famille globale du run si fournie.

        Returns:
            str: Famille runtime retenue pour les profils du symbole.
        """
        normalized_family = resolve_model_family(family=configured_family)
        if normalized_family and normalized_family != "mixed":
            return normalized_family
        return resolve_model_family(symbol=symbol)

    @classmethod
    def build_runtime_entry_filter(
        cls,
        *,
        horizon: str,
        symbol: str,
        configured_family: str | None = None,
        training_mode: bool,
        training_progress_step: int,
        curriculum_soft_end_step: int,
        curriculum_end_step: int,
    ) -> dict[str, float | bool | str]:
        """Construit le filtre runtime coherent avec un symbole.

        Args:
            horizon (str): Horizon strategique du run.
            symbol (str): Symbole de marche courant.
            configured_family (str | None): Famille globale du run si fournie.
            training_mode (bool): Active le curriculum d'apprentissage.
            training_progress_step (int): Etape d'optimisation courante.
            curriculum_soft_end_step (int): Fin de la phase la plus permissive.
            curriculum_end_step (int): Fin du curriculum complet.

        Returns:
            dict[str, float | bool | str]: Filtre d'entree actif pour le
                symbole courant.
        """
        runtime_family = cls.resolve_runtime_family(
            symbol=symbol,
            configured_family=configured_family,
        )
        runtime_profile = resolve_position_mechanics_profile(horizon, runtime_family)
        return cls.resolve_active_entry_filter(
            dict(runtime_profile.get("entry_filter") or {}),
            training_mode=training_mode,
            training_progress_step=training_progress_step,
            horizon=horizon,
            curriculum_soft_end_step=curriculum_soft_end_step,
            curriculum_end_step=curriculum_end_step,
        )
    """Environnement de trading compatible Gymnasium pour MuZero.

    L'observation contient les colonnes de marché brutes suivies de six
    caractéristiques supplémentaires décrivant l'état de position.
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        day_labels: Optional[np.ndarray | list[str]] = None,
        symbol: str = "XAUUSD",
        config=None,
        max_steps: int = 1000,
        training_mode: bool = False,
        training_progress_step: int | None = None,
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
        self.config = config
        self.spec = SymbolSpec.for_symbol(symbol)
        self.max_steps_per_episode = max_steps
        self.commission_rate = 0.00005
        self.training_mode = bool(training_mode)
        self.training_progress_step = int(training_progress_step or 0)
        self.randomize_episode_start = bool(
            getattr(config, "randomize_episode_start", True)
        ) if config is not None else True
        self.episode_warmup_bars = max(
            0,
            int(getattr(config, "episode_warmup_bars", 100) or 0),
        ) if config is not None else 100

        if config:
            self.quality_mult = config.quality_trade_bonus
            self.final_growth_bonus = config.final_growth_bonus
            self.final_growth_threshold = config.final_growth_threshold
            self.dd_penalty_rate = config.drawdown_time_penalty_rate
            self.max_dd_penalty = config.max_drawdown_penalty
            self.loss_mult = config.loss_penalty_multiplier
            self.slbe_bonus = config.slbe_activation_bonus
            self.daily_stretch_target_pct = float(
                getattr(config, "daily_stretch_target_pct", 10.0) or 10.0
            )
            self.daily_stretch_max_drawdown_pct = float(
                getattr(config, "daily_stretch_max_drawdown_pct", 3.5) or 3.5
            )
            configured_daily_bonus = float(
                getattr(config, "daily_stretch_reward_bonus", 4.0) or 0.0
            )
        else:
            self.quality_mult = 10.0
            self.final_growth_bonus = 50.0
            self.final_growth_threshold = 0.10
            self.dd_penalty_rate = 0.2
            self.max_dd_penalty = 10.0
            self.loss_mult = 2.0
            self.slbe_bonus = 6.0
            self.daily_stretch_target_pct = 10.0
            self.daily_stretch_max_drawdown_pct = 3.5
            configured_daily_bonus = 4.0
        self.daily_stretch_reward_bonus = max(
            0.0,
            min(configured_daily_bonus, max(1.0, self.final_growth_bonus * 0.10)),
        )

        self.horizon = str(getattr(config, "horizon", "intraday") or "intraday").lower()
        self.primary_timeframe = str(getattr(config, "primary_timeframe", "H1") or "H1").upper()
        configured_family = getattr(config, "model_family", None)
        self.family = self.resolve_runtime_family(
            symbol=symbol,
            configured_family=configured_family,
        )
        self.feature_profile = resolve_feature_profile(self.horizon, self.family)
        self.position_mechanics_profile = resolve_position_mechanics_profile(
            self.horizon,
            self.family,
        )
        self.mechanics_profile_version = str(
            self.position_mechanics_profile.get("profile_version") or "v1"
        )
        self.data = data if data is not None else self._generate_synthetic_data()
        self.day_labels = self._normalize_day_labels(day_labels, len(self.data))
        self.base_feature_count = self.data.shape[1]
        self.observation_dim = self.base_feature_count + 6
        self._regime_start_indices = self._build_regime_start_indices()
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

    def _build_default_day_labels(self, length: int) -> np.ndarray:
        """Construit des jours synthetiques si aucune etiquette n'est fournie.

        Args:
            length (int): Nombre total de barres a couvrir.

        Returns:
            np.ndarray: Tableau d'etiquettes journalieres synthetiques.
        """
        bars_per_day_by_timeframe = {
            "M1": 1440,
            "M5": 288,
            "M15": 96,
            "H1": 24,
            "D1": 1,
            "W1": 1,
        }
        bars_per_day = max(1, int(bars_per_day_by_timeframe.get(self.primary_timeframe, 24)))
        return np.asarray(
            [f"synthetic_day_{index // bars_per_day:04d}" for index in range(length)],
            dtype=object,
        )

    def _normalize_day_labels(
        self,
        day_labels: Optional[np.ndarray | list[str]],
        expected_length: int,
    ) -> np.ndarray:
        """Normalise les jours pour rester aligns sur la matrice MuZero.

        Args:
            day_labels (Optional[np.ndarray | list[str]]): Etiquettes brutes.
            expected_length (int): Longueur attendue pour la matrice.

        Returns:
            np.ndarray: Etiquettes journalieres normalisees.

        Raises:
            ValueError: Si la longueur des jours ne correspond pas aux donnees.
        """
        if expected_length <= 0:
            return np.asarray([], dtype=object)
        if day_labels is None:
            return self._build_default_day_labels(expected_length)

        normalized = np.asarray(day_labels, dtype=object).reshape(-1)
        if normalized.size != expected_length:
            raise ValueError(
                "Etiquettes journalieres MuZero incompatibles avec la matrice de marche."
            )
        return np.asarray(
            [str(label or f"day_{index:04d}") for index, label in enumerate(normalized)],
            dtype=object,
        )

    def _classify_regime(self, step_index: int) -> str:
        """Classe un index de marche dans un regime simple bull/bear/range.

        Args:
            step_index (int): Index de barre cible.

        Returns:
            str: Regime estime (`bull`, `bear` ou `range`).
        """
        if len(self.data) <= 0:
            return "range"

        index = max(0, min(step_index, len(self.data) - 1))
        close_price = float(self.data[index, 3])
        ema_200 = (
            float(self.data[index, MARKET_COL_EMA_200])
            if self.base_feature_count > MARKET_COL_EMA_200
            else close_price
        )
        slope_anchor = max(0, index - 12)
        previous_ema = (
            float(self.data[slope_anchor, MARKET_COL_EMA_200])
            if self.base_feature_count > MARKET_COL_EMA_200
            else ema_200
        )
        return_anchor = max(0, index - 48)
        previous_close = float(self.data[return_anchor, 3])
        return_48bars = (
            (close_price - previous_close) / max(abs(previous_close), 1e-8)
            if return_anchor != index
            else 0.0
        )
        ema_slope = ema_200 - previous_ema

        if close_price > ema_200 and ema_slope > 0.0 and return_48bars > 0.003:
            return "bull"
        if close_price < ema_200 and ema_slope < 0.0 and return_48bars < -0.003:
            return "bear"
        return "range"

    def _build_regime_start_indices(self) -> dict[str, list[int]]:
        """Construit les points de depart valides regroupes par regime.

        Returns:
            dict[str, list[int]]: Index de depart candidats par regime.
        """
        max_available_index = max(0, len(self.data) - 2)
        min_start_index = min(self.episode_warmup_bars, max_available_index)
        max_start_index = max(
            min_start_index,
            max_available_index - max(0, int(self.max_steps_per_episode)),
        )
        buckets = {"bull": [], "bear": [], "range": []}
        for index in range(min_start_index, max_start_index + 1):
            buckets[self._classify_regime(index)].append(index)
        return buckets

    def _get_temporal_features(self, step_index: int) -> tuple[float, float]:
        """Construit des features temporelles a partir des jours reels.

        Args:
            step_index (int): Index de barre courant.

        Returns:
            tuple[float, float]: Progression intra-jour et cycle hebdomadaire
                normalises dans ``[0, 1]``.
        """
        if len(self.day_labels) <= 0:
            fallback_hour = (step_index % 24) / 23.0
            fallback_day = ((step_index // 24) % 5) / 4.0
            return float(fallback_hour), float(fallback_day)

        index = max(0, min(step_index, len(self.day_labels) - 1))
        current_label = str(self.day_labels[index])

        start_index = index
        while start_index > 0 and str(self.day_labels[start_index - 1]) == current_label:
            start_index -= 1

        end_index = index
        last_index = len(self.day_labels) - 1
        while end_index < last_index and str(self.day_labels[end_index + 1]) == current_label:
            end_index += 1

        intraday_progress = (index - start_index) / max(end_index - start_index, 1)
        try:
            weekday_feature = datetime.fromisoformat(current_label).weekday() / 6.0
        except ValueError:
            day_ordinal = 0
            previous_label = str(self.day_labels[0])
            for raw_label in self.day_labels[1 : index + 1]:
                normalized_label = str(raw_label)
                if normalized_label != previous_label:
                    day_ordinal += 1
                    previous_label = normalized_label
            weekday_feature = (day_ordinal % 5) / 4.0

        return float(intraday_progress), float(weekday_feature)

    def _finalize_active_day(self) -> float:
        """Consolide la journee active et renvoie le bonus stretch eventuel.

        Returns:
            float: Bonus journalier a ajouter au reward.
        """
        start_equity = max(self._active_day_start_equity, 1e-8)
        day_return_pct = ((self._active_day_last_equity - start_equity) / start_equity) * 100.0
        day_drawdown_pct = float(self._active_day_max_drawdown_pct)

        self.daily_net_return_pct_by_day[self._active_day_label] = float(day_return_pct)
        self.daily_drawdown_pct_by_day[self._active_day_label] = float(day_drawdown_pct)
        self.best_day_net_return_pct = max(self.best_day_net_return_pct, float(day_return_pct))
        self.daily_max_drawdown_pct = max(self.daily_max_drawdown_pct, float(day_drawdown_pct))
        if day_return_pct > 0.0:
            self.positive_days += 1

        stretch_eligible = (
            day_return_pct >= self.daily_stretch_target_pct
            and day_drawdown_pct <= self.daily_stretch_max_drawdown_pct
        )
        if stretch_eligible:
            self.days_above_10pct += 1
            logger.info(
                "Bonus stretch journalier active sur %s: retour=%.2f%% | drawdown=%.2f%% | bonus=%.2f",
                self._active_day_label,
                day_return_pct,
                day_drawdown_pct,
                self.daily_stretch_reward_bonus,
            )
            return self.daily_stretch_reward_bonus
        return 0.0

    def _update_daily_tracking(self, day_label: str, equity: float, *, finalize: bool) -> float:
        """Met a jour le suivi journalier a partir de l'equite courante.

        Args:
            day_label (str): Jour reel du pas courant.
            equity (float): Equite courante apres execution.
            finalize (bool): Finalise la journee active si l'episode se termine.

        Returns:
            float: Bonus stretch journalier a ajouter au reward.
        """
        reward_bonus = 0.0
        normalized_label = str(day_label or self._active_day_label)
        if normalized_label != self._active_day_label:
            reward_bonus += self._finalize_active_day()
            carry_equity = self._active_day_last_equity
            self._active_day_label = normalized_label
            self._active_day_start_equity = carry_equity
            self._active_day_peak_equity = carry_equity
            self._active_day_trough_equity = carry_equity
            self._active_day_max_drawdown_pct = 0.0
            self._active_day_last_equity = carry_equity

        self._active_day_peak_equity = max(self._active_day_peak_equity, equity)
        self._active_day_trough_equity = min(self._active_day_trough_equity, equity)
        if self._active_day_peak_equity > 0.0:
            current_drawdown_pct = (
                (self._active_day_peak_equity - equity) / self._active_day_peak_equity
            ) * 100.0
            self._active_day_max_drawdown_pct = max(
                self._active_day_max_drawdown_pct,
                current_drawdown_pct,
            )
        self._active_day_last_equity = equity

        if finalize:
            reward_bonus += self._finalize_active_day()
        return reward_bonus

    def _reset_state(self) -> None:
        """Réinitialise l'état interne de l'épisode."""
        self.current_step = self._resolve_episode_start_index()
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
        self.nemesis_trap_losses = 0
        self.nemesis_recent_losses = 0
        self.nemesis_quarantine_active = 0.0
        self.nemesis_trap_rate = 0.0
        self.gross_profit_pct = 0.0
        self.gross_loss_pct = 0.0
        self.net_realized_pnl_pct = 0.0
        self.split_count = 0
        self.secured_count = 0
        self.equity_curve = [self.spec.initial_balance]
        self.action_counts = {name: 0 for name in ACTION_NAMES}
        self.long_entries = 0
        self.short_entries = 0
        self.requested_buy_actions = 0
        self.requested_sell_actions = 0
        self.blocked_buy_entries = 0
        self.blocked_sell_entries = 0
        self.blocked_buy_vwap = 0
        self.blocked_sell_vwap = 0
        self.blocked_buy_adx = 0
        self.blocked_sell_adx = 0
        self.blocked_buy_obv = 0
        self.blocked_sell_obv = 0
        self.blocked_buy_directional = 0
        self.blocked_sell_directional = 0
        self.entry_veto_to_hold = 0
        self.root_mask_directional_candidates_total = 0
        self.root_mask_blocked_buy_total = 0
        self.root_mask_blocked_sell_total = 0
        self.root_mask_blocked_buy_ema200 = 0
        self.root_mask_blocked_sell_ema200 = 0
        self.root_mask_blocked_buy_vwap = 0
        self.root_mask_blocked_sell_vwap = 0
        self.root_mask_blocked_buy_adx = 0
        self.root_mask_blocked_sell_adx = 0
        self.root_mask_blocked_buy_obv = 0
        self.root_mask_blocked_sell_obv = 0
        self.root_mask_blocked_buy_directional = 0
        self.root_mask_blocked_sell_directional = 0
        self.ema200_blocked_buy = 0
        self.ema200_blocked_sell = 0
        self.soft_entry_penalty_count = 0
        self.soft_entry_penalty_total = 0.0
        self.soft_entry_bonus_count = 0
        self.soft_entry_bonus_total = 0.0
        self.soft_penalty_ema200_count = 0
        self.soft_penalty_vwap_count = 0
        self.soft_penalty_adx_count = 0
        self.soft_penalty_obv_count = 0
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
        self.hold_drag_opportunity_count = 0
        self.hold_drag_penalized_count = 0
        self.hold_streak_total = 0
        self.hold_streak_count = 0
        self._current_hold_streak = 0
        self._current_hold_drag_streak = 0
        self.pyramids_opened = 0
        self.pyramids_rejected = 0
        self.pyramid_profitable_count = 0
        self.pyramid_loss_count = 0
        self.pyramid_good_add_count = 0
        self.pyramid_bad_add_count = 0
        self.pyramid_profitable_exit_count = 0
        self.pyramid_opportunity_count = 0
        self.position_pyramids = 0
        self.split_executed = 0
        self.split_profitable_count = 0
        self.split_runner_profitable_count = 0
        self.split_runner_failed_count = 0
        self.split_early_count = 0
        self.split_decorative_count = 0
        self.split_trade_value_delta = 0.0
        self.split_improved_total_trade_count = 0
        self.split_opportunity_count = 0
        self.split_tp_zone_opportunity_count = 0
        self.split_monetization_window_count = 0
        self.split_monetization_capture_count = 0
        self.split_missed_window_count = 0
        self.slbe_triggered = 0
        self.slbe_hit = 0
        self.slbe_profitable_exits = 0
        self.slbe_lock_profit_count = 0
        self.position_had_slbe = False
        self.slbe_profit_locked = False
        self.close_winner_count = 0
        self.close_loser_count = 0
        self.net_realized_long_pct = 0.0
        self.net_realized_short_pct = 0.0
        self.tp_like_exit_count = 0
        self.tp_like_missed_count = 0
        self.defensive_close_count = 0
        self.early_close_noise_count = 0
        self.hard_stop_exit_count = 0
        self.soft_tp_hit_count = 0
        self.full_tp_hit_count = 0
        self.time_stop_trigger_count = 0
        self.runner_extension_count = 0
        self.runner_extension_opportunity_count = 0
        self.runner_extension_capture_count = 0
        self.runner_profit_hold_window_count = 0
        self.runner_profit_hold_capture_count = 0
        self.runner_hold_after_soft_tp_count = 0
        self.runner_viable_window_count = 0
        self.runner_missed_extension_count = 0
        self.runner_viable_but_closed_count = 0
        self.early_full_close_after_soft_tp_count = 0
        self.runner_managed_exit_count = 0
        self.runner_exit_profitable_count = 0
        self.runner_forced_stop_count = 0
        self.runner_retained_profit_pct = 0.0
        self.runner_giveback_pct = 0.0
        self.runner_retained_profit_score_total = 0.0
        self.profit_peak_reached_count = 0
        self.profit_peak_giveback_ratio_total = 0.0
        self.profit_peak_giveback_ratio_observations = 0
        self.forced_stop_near_miss_count = 0
        self.split_rejected = 0
        self.split_rejected_no_value = 0
        self.inactive_episode_penalties = 0
        self.insufficient_entry_penalties = 0
        self.directional_imbalance_penalties = 0
        self.realized_close_bonus_count = 0
        self.realized_split_bonus_count = 0
        self.slbe_exit_bonus_count = 0
        self.hard_stop_price = 0.0
        self.soft_tp_price = 0.0
        self.full_tp_price = 0.0
        self.time_stop_steps = 0
        self.position_entry_step = 0
        self.position_peak_return = 0.0
        self.peak_profit_age_steps = 0
        self.runner_extension_active = False
        self.runner_active = False
        self.runner_entry_price = 0.0
        self.runner_target_price = 0.0
        self.runner_protected = False
        self.runner_origin_split_step = -1
        self.runner_peak_profit_pct = 0.0
        self.runner_entry_profit_pct = 0.0
        self._last_runner_retained_profit_pct = 0.0
        self._last_runner_giveback_pct = 0.0
        self._last_runner_retention_ratio = 0.0
        self.pyramid_total_trade_improvement_pct = 0.0
        self.pyramid_failed_to_improve_count = 0
        self.pyramid_add_opportunity_count = 0
        self.pyramid_add_capture_count = 0
        self.pyramid_monetization_window_count = 0
        self.pyramid_monetization_capture_count = 0
        self.pyramid_missed_add_count = 0
        self._pyramid_baseline_active = False
        self._pyramid_baseline_return = 0.0
        self.soft_tp_hit_active = False
        self.full_tp_hit_active = False
        self.time_stop_recorded_current_trade = False
        self._split_window_wait_steps = 0
        self._runner_window_wait_steps = 0
        self._pyramid_window_wait_steps = 0
        active_index = min(self.current_step, max(len(self.day_labels) - 1, 0))
        self.daily_net_return_pct_by_day: dict[str, float] = {}
        self.daily_drawdown_pct_by_day: dict[str, float] = {}
        self.best_day_net_return_pct = float("-inf")
        self.days_above_10pct = 0
        self.positive_days = 0
        self.daily_max_drawdown_pct = 0.0
        self._rebalance_bonus_paid_buy = False
        self._rebalance_bonus_paid_sell = False
        self._episode_regime = self._classify_regime(self.current_step)
        self._active_day_label = str(self.day_labels[active_index]) if len(self.day_labels) else "day_0000"
        self._active_day_start_equity = self.spec.initial_balance
        self._active_day_peak_equity = self.spec.initial_balance
        self._active_day_trough_equity = self.spec.initial_balance
        self._active_day_max_drawdown_pct = 0.0
        self._active_day_last_equity = self.spec.initial_balance

    def _resolve_episode_start_index(self) -> int:
        """Choisit un index de depart valide dans la fenetre historique.

        Returns:
            int: Index de depart de l'episode.
        """
        if len(self.data) <= 0:
            return 0

        max_available_index = max(0, len(self.data) - 2)
        min_start_index = min(self.episode_warmup_bars, max_available_index)
        max_start_index = max(
            min_start_index,
            max_available_index - max(0, int(self.max_steps_per_episode)),
        )

        if not self.randomize_episode_start or max_start_index <= min_start_index:
            return int(min_start_index)
        if self.training_mode:
            complete_regime_matrix = all(
                len(self._regime_start_indices.get(regime, [])) > 0
                for regime in ("bull", "bear", "range")
            )
            if complete_regime_matrix:
                target_regime = np.random.choice(["bull", "bear", "range"])
                bucket = list(self._regime_start_indices.get(str(target_regime), []))
                if bucket:
                    return int(np.random.choice(bucket))

        return int(np.random.randint(min_start_index, max_start_index + 1))

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
            "ema_gap_pct": (close_price - ema_200) / max(abs(ema_200), 1e-8),
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

        entry_filter = self._get_active_entry_filter()
        min_adx = float(entry_filter.get("min_adx", 0.0) or 0.0)
        trend_adx = float(entry_filter.get("trend_adx", min_adx) or min_adx)
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

    @staticmethod
    def resolve_active_entry_filter(
        entry_filter: dict[str, float | bool | str],
        *,
        training_mode: bool,
        training_progress_step: int,
        horizon: str,
        curriculum_soft_end_step: int,
        curriculum_end_step: int,
    ) -> dict[str, float | bool | str]:
        """Retourne le filtre d'entree actif apres application du curriculum.

        Args:
            entry_filter (dict[str, float | bool | str]): Filtre de base du profil.
            training_mode (bool): Indique si l'environnement est en apprentissage.
            training_progress_step (int): Etape d'optimisation courante.
            horizon (str): Horizon strategique du run.
            curriculum_soft_end_step (int): Fin de la phase la plus permissive.
            curriculum_end_step (int): Fin du curriculum.

        Returns:
            dict[str, float | bool | str]: Filtre actif applique au pas courant.
        """
        resolved_filter = dict(entry_filter or {})
        if (
            not training_mode
            or curriculum_end_step <= 0
            or training_progress_step >= curriculum_end_step
        ):
            return resolved_filter

        normalized_horizon = str(horizon or "").strip().lower()
        exploration_filter = dict(resolved_filter)
        if normalized_horizon == "scalp":
            soft_end_step = max(
                1,
                min(int(curriculum_soft_end_step or 0) or curriculum_end_step, curriculum_end_step),
            )
            exploration_filter["allow_trend_fallback"] = True
            exploration_filter["require_vwap_alignment"] = False
            exploration_filter["require_obv_confirmation"] = False
            if training_progress_step < soft_end_step:
                relax_delta = 5.0
                min_floor = 7.0
                trend_floor = 10.0
            else:
                relax_delta = 3.0
                min_floor = 8.0
                trend_floor = 12.0
        else:
            exploration_filter["allow_trend_fallback"] = True
            exploration_filter["require_vwap_alignment"] = False
            exploration_filter["require_obv_confirmation"] = False
            relax_delta = 2.0
            min_floor = 8.0
            trend_floor = 12.0

        exploration_filter["min_adx"] = max(
            min_floor,
            float(resolved_filter.get("min_adx", 0.0) or 0.0) - relax_delta,
        )
        exploration_filter["trend_adx"] = max(
            trend_floor,
            float(
                resolved_filter.get(
                    "trend_adx",
                    exploration_filter["min_adx"],
                ) or exploration_filter["min_adx"]
            ) - relax_delta,
        )
        return exploration_filter

    def _get_active_entry_filter(self) -> dict[str, float | bool | str]:
        """Retourne le filtre d'entree courant, avec curriculum si necessaire.

        Returns:
            dict[str, float | bool | str]: Filtre d'entree actif.
        """
        return self.resolve_active_entry_filter(
            dict(self.position_mechanics_profile.get("entry_filter") or {}),
            training_mode=self.training_mode,
            training_progress_step=self.training_progress_step,
            horizon=str(getattr(self.config, "horizon", "") or ""),
            curriculum_soft_end_step=int(
                getattr(self.config, "directional_curriculum_soft_end_step", 8000) or 8000
            ),
            curriculum_end_step=int(
                getattr(self.config, "directional_curriculum_end_step", 15000) or 15000
            ),
        )

    @staticmethod
    def _classify_directional_bias(long_entries: int, short_entries: int) -> str:
        """Retourne une etiquette simple du biais directionnel de l'episode.

        Args:
            long_entries (int): Nombre d'entrees longues.
            short_entries (int): Nombre d'entrees courtes.

        Returns:
            str: `inactive`, `buy_heavy`, `sell_heavy` ou `balanced`.
        """
        total_entries = long_entries + short_entries
        if total_entries <= 0:
            return "inactive"
        long_share = long_entries / total_entries
        short_share = short_entries / total_entries
        if long_share <= 0.20 and short_share >= 0.80:
            return "sell_heavy"
        if short_share <= 0.20 and long_share >= 0.80:
            return "buy_heavy"
        return "balanced"

    def _should_hard_veto_directional_entry(
        self,
        action: int,
        directional_policy: dict[str, float],
    ) -> bool:
        """Determine si une nouvelle entree doit etre bloquee par direction.

        Args:
            action (int): Action directionnelle candidate.
            directional_policy (dict[str, float]): Politique directionnelle active.

        Returns:
            bool: `True` si l'entree doit etre convertie en `HOLD`.
        """
        return self._would_hard_veto_directional_entry(
            action,
            directional_policy,
            long_entries=self.long_entries,
            short_entries=self.short_entries,
        )

    @classmethod
    def _would_hard_veto_directional_entry(
        cls,
        action: int,
        directional_policy: dict[str, float],
        *,
        long_entries: int,
        short_entries: int,
    ) -> bool:
        """Determine si une entree depasserait le plafond directionnel dur.

        Args:
            action (int): Action directionnelle candidate.
            directional_policy (dict[str, float]): Politique directionnelle active.
            long_entries (int): Nombre d'entrees longues deja executees.
            short_entries (int): Nombre d'entrees courtes deja executees.

        Returns:
            bool: `True` si l'entree doit etre vetoee.
        """
        resolved_policy = cls._resolve_directional_policy_terms(directional_policy)
        hard_veto_after_entries = int(resolved_policy["hard_veto_after_entries"])
        hard_veto_max_share = resolved_policy["hard_veto_max_share"]
        total_entries = int(long_entries) + int(short_entries)
        if hard_veto_after_entries <= 0 or total_entries < hard_veto_after_entries:
            return False

        projected_long_entries = int(long_entries) + (1 if action == BUY else 0)
        projected_short_entries = int(short_entries) + (1 if action == SELL else 0)
        projected_total_entries = projected_long_entries + projected_short_entries
        if projected_total_entries <= 0:
            return False

        projected_share = (
            projected_long_entries / projected_total_entries
            if action == BUY
            else projected_short_entries / projected_total_entries
        )
        return projected_share > hard_veto_max_share

    @staticmethod
    def _is_directional_entry_for_state(action: int, position_size: float) -> bool:
        """Retourne vrai si l'action ouvre ou retourne une exposition directionnelle.

        Args:
            action (int): Action candidate.
            position_size (float): Exposition courante.

        Returns:
            bool: `True` si l'action doit subir le veto directionnel dur.
        """
        return action in (BUY, SELL) and (
            (action == BUY and position_size <= 0.0)
            or (action == SELL and position_size >= 0.0)
        )

    @staticmethod
    def _compute_fallback_direction_ok(
        context: dict[str, float],
        entry_filter: dict[str, float | bool | str],
    ) -> bool:
        """Calcule le mode de repli `trend_fallback` pour un contexte de marche.

        Args:
            context (dict[str, float]): Contexte courant du marche.
            entry_filter (dict[str, float | bool | str]): Filtre actif.

        Returns:
            bool: `True` si le fallback autorise l'entree malgre un veto doux.
        """
        min_adx = float(entry_filter.get("min_adx", 0.0) or 0.0)
        trend_adx = float(entry_filter.get("trend_adx", min_adx) or min_adx)
        allow_trend_fallback = bool(entry_filter.get("allow_trend_fallback", False))
        return (
            allow_trend_fallback
            and context["adx"] >= max(min_adx * 0.6, trend_adx - 4.0, 0.0)
            and abs(context["momentum"]) >= max(context["atr_pct"], 1e-5)
        )

    @classmethod
    def _evaluate_entry_veto_reason(
        cls,
        action: int,
        context: dict[str, float],
        entry_filter: dict[str, float | bool | str],
        *,
        position_size: float = 0.0,
        long_entries: int = 0,
        short_entries: int = 0,
        directional_policy: dict[str, float] | None = None,
        include_directional_hard_veto: bool = True,
    ) -> str | None:
        """Retourne la premiere raison de veto applicable a une entree.

        Args:
            action (int): Action candidate.
            context (dict[str, float]): Contexte courant du marche.
            entry_filter (dict[str, float | bool | str]): Filtre actif.
            position_size (float): Taille de position courante.
            long_entries (int): Entrees longues deja executees.
            short_entries (int): Entrees courtes deja executees.
            directional_policy (dict[str, float] | None): Politique directionnelle active.
            include_directional_hard_veto (bool): Active le veto directionnel dur.

        Returns:
            str | None: Raison du veto, sinon `None`.
        """
        if action not in (BUY, SELL):
            return None

        ema_mode = str(entry_filter.get("ema_mode", "strict")).lower()
        require_vwap_alignment = bool(entry_filter.get("require_vwap_alignment", False))
        require_obv_confirmation = bool(entry_filter.get("require_obv_confirmation", False))
        min_adx = float(entry_filter.get("min_adx", 0.0) or 0.0)
        fallback_direction_ok = cls._compute_fallback_direction_ok(context, entry_filter)

        if context["adx"] < min_adx and not fallback_direction_ok:
            return "adx"

        atr_pct = max(float(context.get("atr_pct", 0.0) or 0.0), 1e-5)
        ema_gap_pct = float(context.get("ema_gap_pct", 0.0) or 0.0)
        momentum = float(context.get("momentum", 0.0) or 0.0)
        ema_gap_floor = max(atr_pct * 0.35, 0.0002)
        momentum_floor = max(atr_pct, 1e-5)

        if action == BUY:
            if ema_mode == "strict" and context["close"] < context["ema_200"]:
                return "ema200"
            if (
                ema_mode == "moderate"
                and context["close"] < context["ema_200"]
                and context["price_vs_vwap"] < 0
                and ema_gap_pct <= -ema_gap_floor
                and momentum <= -momentum_floor
                and not fallback_direction_ok
            ):
                return "ema200"
            if require_vwap_alignment and context["price_vs_vwap"] < 0 and not fallback_direction_ok:
                return "vwap"
            if require_obv_confirmation and context["obv_slope"] <= 0 and not fallback_direction_ok:
                return "obv"
        else:
            if ema_mode == "strict" and context["close"] > context["ema_200"]:
                return "ema200"
            if (
                ema_mode == "moderate"
                and context["close"] > context["ema_200"]
                and context["price_vs_vwap"] > 0
                and ema_gap_pct >= ema_gap_floor
                and momentum >= momentum_floor
                and not fallback_direction_ok
            ):
                return "ema200"
            if require_vwap_alignment and context["price_vs_vwap"] > 0 and not fallback_direction_ok:
                return "vwap"
            if require_obv_confirmation and context["obv_slope"] >= 0 and not fallback_direction_ok:
                return "obv"

        if (
            include_directional_hard_veto
            and directional_policy
            and cls._is_directional_entry_for_state(action, position_size)
            and cls._would_hard_veto_directional_entry(
                action,
                directional_policy,
                long_entries=long_entries,
                short_entries=short_entries,
            )
        ):
            return "directional"
        return None

    def _register_root_mask_block(self, action: int, reason: str) -> None:
        """Incremente les compteurs du masque racine pour la raison donnee.

        Args:
            action (int): Action retiree du masque.
            reason (str): Raison du retrait.
        """
        if action == BUY:
            self.root_mask_blocked_buy_total += 1
            if reason == "ema200":
                self.root_mask_blocked_buy_ema200 += 1
            elif reason == "vwap":
                self.root_mask_blocked_buy_vwap += 1
            elif reason == "adx":
                self.root_mask_blocked_buy_adx += 1
            elif reason == "obv":
                self.root_mask_blocked_buy_obv += 1
            elif reason == "directional":
                self.root_mask_blocked_buy_directional += 1
        elif action == SELL:
            self.root_mask_blocked_sell_total += 1
            if reason == "ema200":
                self.root_mask_blocked_sell_ema200 += 1
            elif reason == "vwap":
                self.root_mask_blocked_sell_vwap += 1
            elif reason == "adx":
                self.root_mask_blocked_sell_adx += 1
            elif reason == "obv":
                self.root_mask_blocked_sell_obv += 1
            elif reason == "directional":
                self.root_mask_blocked_sell_directional += 1

    def _should_soften_training_root_veto(
        self,
        *,
        action: int,
        reason: str,
        context: dict[str, float],
        entry_filter: dict[str, float | bool | str],
    ) -> bool:
        """Determine si un veto racine peut devenir une penalite douce en train.

        Args:
            action (int): Action directionnelle evaluee.
            reason (str): Raison du veto courant.
            context (dict[str, float]): Contexte de marche courant.
            entry_filter (dict[str, float | bool | str]): Filtre d'entree actif.

        Returns:
            bool: ``True`` si le veto doit etre converti en signal doux pour
                l'entrainement, ``False`` sinon.
        """
        if not self.training_mode or action not in (BUY, SELL):
            return False

        if reason == "directional":
            return bool(
                getattr(self.config, "training_root_mask_soften_directional", False)
            )

        if reason == "vwap":
            return bool(getattr(self.config, "training_root_mask_soften_vwap", True))

        if reason == "adx":
            if not bool(getattr(self.config, "training_root_mask_soften_adx", True)):
                return False
            min_adx = float(entry_filter.get("min_adx", 0.0) or 0.0)
            adx = float(context.get("adx", 0.0) or 0.0)
            if min_adx <= 0.0:
                return True
            extreme_ratio = float(
                getattr(self.config, "training_root_mask_adx_extreme_ratio", 0.75) or 0.75
            )
            return adx >= (min_adx * extreme_ratio)

        if reason == "ema200":
            if not bool(getattr(self.config, "training_root_mask_soften_ema200", True)):
                return False
            ema_mode = str(entry_filter.get("ema_mode", "strict") or "strict").strip().lower()
            if ema_mode == "strict":
                return False
            atr_pct = max(float(context.get("atr_pct", 0.0) or 0.0), 1e-5)
            ema_gap_pct = abs(float(context.get("ema_gap_pct", 0.0) or 0.0))
            price_vs_vwap = float(context.get("price_vs_vwap", 0.0) or 0.0)
            momentum = float(context.get("momentum", 0.0) or 0.0)
            direction_sign = 1.0 if action == BUY else -1.0
            vwap_floor = max(atr_pct * 0.10, 0.00005)
            momentum_floor = max(atr_pct * 0.35, 1e-5)
            severe_gap = ema_gap_pct >= max(atr_pct * 1.5, 0.001)
            vwap_support = (price_vs_vwap * direction_sign) >= -vwap_floor
            momentum_support = (momentum * direction_sign) >= -momentum_floor
            return vwap_support and momentum_support and not severe_gap

        return False

    def _compute_rebalance_bonus(
        self,
        action: int,
        directional_policy: dict[str, float],
    ) -> float:
        """Calcule le bonus de reequilibrage lors d'une entree minoritaire.

        Args:
            action (int): Action finale d'entree (`BUY` ou `SELL`).
            directional_policy (dict[str, float]): Politique directionnelle active.

        Returns:
            float: Bonus a ajouter a la recompense.
        """
        resolved_policy = self._resolve_directional_policy_terms(directional_policy)
        total_entries = self.long_entries + self.short_entries
        if total_entries <= 0:
            return 0.0

        target_is_buy = action == BUY
        paid_flag = self._rebalance_bonus_paid_buy if target_is_buy else self._rebalance_bonus_paid_sell
        if paid_flag:
            return 0.0

        target_entries = self.long_entries if target_is_buy else self.short_entries
        opposite_entries = self.short_entries if target_is_buy else self.long_entries
        imbalance = abs(self.long_entries - self.short_entries) / max(total_entries, 1)
        missing_side_entry = target_entries == 0 and opposite_entries > 0
        minority_reentry = (
            target_entries < opposite_entries
            and imbalance > resolved_policy["max_directional_imbalance"]
        )
        if not (missing_side_entry or minority_reentry):
            return 0.0

        bonus = float(resolved_policy["rebalance_bonus"])
        if bonus <= 0.0:
            return 0.0

        if target_is_buy:
            self._rebalance_bonus_paid_buy = True
        else:
            self._rebalance_bonus_paid_sell = True
        return bonus

    def _resolve_soft_reward_phase_scales(self) -> tuple[float, float]:
        """Retourne les multiplicateurs de shaping doux pour la phase courante.

        Returns:
            tuple[float, float]: Couple ``(penalty_scale, bonus_scale)``.
        """
        early_end_step = max(
            0,
            int(getattr(self.config, "soft_reward_early_end_step", 4000) or 4000),
        )
        mid_end_step = max(
            early_end_step,
            int(getattr(self.config, "soft_reward_mid_end_step", 10000) or 10000),
        )
        current_step = max(0, int(self.training_progress_step or 0))
        if current_step < early_end_step:
            return (
                float(getattr(self.config, "soft_reward_penalty_scale_early", 0.45) or 0.45),
                float(getattr(self.config, "soft_reward_bonus_scale_early", 1.15) or 1.15),
            )
        if current_step < mid_end_step:
            return (
                float(getattr(self.config, "soft_reward_penalty_scale_mid", 0.65) or 0.65),
                float(getattr(self.config, "soft_reward_bonus_scale_mid", 1.0) or 1.0),
            )
        return (
            float(getattr(self.config, "soft_reward_penalty_scale_late", 0.85) or 0.85),
            float(getattr(self.config, "soft_reward_bonus_scale_late", 0.95) or 0.95),
        )

    def _resolve_exit_reward_phase_scale(self) -> float:
        """Retourne l'intensite du shaping V5 pour la phase courante.

        Returns:
            float: Multiplicateur global applique aux ajustements de sortie
                implicites (`hard_stop`, `split`, `pyramid`, `SLBE`, `hold_drag`,
                `time_stop`). Les signaux de PnL reel restent inchanges.
        """
        early_end_step = max(
            0,
            int(
                getattr(
                    self.config,
                    "exit_reward_early_end_step",
                    getattr(self.config, "soft_reward_early_end_step", 4000),
                ) or 4000
            ),
        )
        mid_end_step = max(
            early_end_step,
            int(
                getattr(
                    self.config,
                    "exit_reward_mid_end_step",
                    getattr(self.config, "soft_reward_mid_end_step", 10000),
                ) or 10000
            ),
        )
        current_step = max(0, int(self.training_progress_step or 0))
        if current_step < early_end_step:
            return float(getattr(self.config, "exit_reward_scale_early", 0.35) or 0.35)
        if current_step < mid_end_step:
            return float(getattr(self.config, "exit_reward_scale_mid", 0.55) or 0.55)
        return float(getattr(self.config, "exit_reward_scale_late", 1.0) or 1.0)

    @staticmethod
    def _policy_float(policy: dict[str, float], key: str, default: float) -> float:
        """Lit un flottant de profil sans ecraser un zero explicite.

        Args:
            policy (dict[str, float]): Bloc de configuration mecanique.
            key (str): Cle a lire.
            default (float): Valeur de secours si la cle est absente.

        Returns:
            float: Valeur resolue en conservant ``0`` comme valeur legitime.
        """
        value = policy.get(key, default)
        if value is None:
            return float(default)
        return float(value)

    @staticmethod
    def _policy_int(policy: dict[str, float], key: str, default: int) -> int:
        """Lit un entier de profil sans ecraser un zero explicite.

        Args:
            policy (dict[str, float]): Bloc de configuration mecanique.
            key (str): Cle a lire.
            default (int): Valeur de secours si la cle est absente.

        Returns:
            int: Valeur resolue en conservant ``0`` comme valeur legitime.
        """
        value = policy.get(key, default)
        if value is None:
            return int(default)
        return int(value)

    def _reset_exit_plan_state(self) -> None:
        """Reinitialise le plan de sortie implicite de la position courante."""
        self.hard_stop_price = 0.0
        self.soft_tp_price = 0.0
        self.full_tp_price = 0.0
        self.time_stop_steps = 0
        self.position_entry_step = int(self.current_step)
        self.position_peak_return = 0.0
        self.peak_profit_age_steps = 0
        self.runner_extension_active = False
        self.runner_active = False
        self.runner_entry_price = 0.0
        self.runner_target_price = 0.0
        self.runner_protected = False
        self.runner_origin_split_step = -1
        self.runner_peak_profit_pct = 0.0
        self.runner_entry_profit_pct = 0.0
        self._last_runner_retained_profit_pct = 0.0
        self._last_runner_giveback_pct = 0.0
        self._last_runner_retention_ratio = 0.0
        self.soft_tp_hit_active = False
        self.full_tp_hit_active = False
        self.time_stop_recorded_current_trade = False
        self._split_window_wait_steps = 0
        self._runner_window_wait_steps = 0
        self._pyramid_window_wait_steps = 0

    def _configure_exit_plan(
        self,
        *,
        price: float,
        context: dict[str, float],
        hold_policy: dict[str, float],
        close_policy: dict[str, float],
        slbe_policy: dict[str, float],
        exit_plan_policy: dict[str, float],
        reset_progress: bool = True,
    ) -> None:
        """Initialise ou recalcule le plan de sortie implicite.

        Args:
            price (float): Prix courant de reference.
            context (dict[str, float]): Contexte de marche du pas.
            hold_policy (dict[str, float]): Regles HOLD actives.
            close_policy (dict[str, float]): Regles CLOSE actives.
            slbe_policy (dict[str, float]): Regles SLBE actives.
            exit_plan_policy (dict[str, float]): Parametres implicites de TP/SL.
            reset_progress (bool): Reinitialise les marqueurs de progression du
                trade si ``True``.
        """
        if self.position_size == 0:
            self._reset_exit_plan_state()
            return

        reference_price = max(float(price or 0.0), float(self.avg_entry_price or 0.0), 1e-8)
        atr_pct = max(float(context.get("atr_pct", 0.0) or 0.0), 1e-5)
        activation_return = self._policy_float(slbe_policy, "activation_return", 0.005)
        tp_like_threshold = self._policy_float(
            close_policy,
            "tp_like_threshold",
            self._policy_float(close_policy, "winner_threshold", 0.01),
        )
        strong_winner_threshold = self._policy_float(
            close_policy,
            "strong_winner_threshold",
            max(tp_like_threshold, 0.01),
        )
        hard_stop_return = max(
            activation_return * 0.75,
            atr_pct * self._policy_float(exit_plan_policy, "hard_stop_atr_mult", 0.90),
        )
        soft_tp_return = max(
            tp_like_threshold,
            atr_pct * self._policy_float(exit_plan_policy, "soft_tp_atr_mult", 1.35),
        )
        full_tp_return = max(
            strong_winner_threshold,
            atr_pct * self._policy_float(exit_plan_policy, "full_tp_atr_mult", 2.10),
        )
        fallback_time_stop = max(
            1,
            int(
                math.ceil(
                    float(hold_policy.get("stale_penalty_after_steps", 48) or 48) * 0.5
                )
            ),
        )
        resolved_time_stop_steps = max(
            1,
            self._policy_int(exit_plan_policy, "time_stop_steps", fallback_time_stop),
        )
        if self.position_size > 0:
            self.hard_stop_price = reference_price * (1.0 - hard_stop_return)
            self.soft_tp_price = reference_price * (1.0 + soft_tp_return)
            self.full_tp_price = reference_price * (1.0 + full_tp_return)
        else:
            self.hard_stop_price = reference_price * (1.0 + hard_stop_return)
            self.soft_tp_price = reference_price * (1.0 - soft_tp_return)
            self.full_tp_price = reference_price * (1.0 - full_tp_return)
        self.time_stop_steps = resolved_time_stop_steps
        if reset_progress:
            self.position_entry_step = int(self.current_step)
            self.position_peak_return = max(0.0, self._get_unrealized_return(price))
            self.peak_profit_age_steps = 0
            self.runner_extension_active = False
            self.soft_tp_hit_active = False
            self.full_tp_hit_active = False
            self.time_stop_recorded_current_trade = False
            self._split_window_wait_steps = 0
            self._runner_window_wait_steps = 0
            self._pyramid_window_wait_steps = 0

    def _mark_exit_plan_hits(
        self,
        *,
        price: float,
    ) -> None:
        """Met a jour les jalons de progression du plan de sortie implicite.

        Args:
            price (float): Prix courant observe.
        """
        if self.position_size == 0:
            return
        if self.slbe_active and self.slbe_price > 0.0:
            if self.position_size > 0:
                self.hard_stop_price = max(self.hard_stop_price, self.slbe_price)
            else:
                self.hard_stop_price = min(self.hard_stop_price, self.slbe_price)
        if (
            self.soft_tp_price > 0.0
            and not self.soft_tp_hit_active
            and (
                (self.position_size > 0 and price >= self.soft_tp_price)
                or (self.position_size < 0 and price <= self.soft_tp_price)
            )
        ):
            self.soft_tp_hit_active = True
            self.soft_tp_hit_count += 1
        if (
            self.full_tp_price > 0.0
            and not self.full_tp_hit_active
            and (
                (self.position_size > 0 and price >= self.full_tp_price)
                or (self.position_size < 0 and price <= self.full_tp_price)
            )
        ):
            self.full_tp_hit_active = True
            self.full_tp_hit_count += 1
        if (
            self.position_size != 0
            and self.time_stop_steps > 0
            and not self.time_stop_recorded_current_trade
            and self._position_age_steps() >= self.time_stop_steps
        ):
            self.time_stop_recorded_current_trade = True
            self.time_stop_trigger_count += 1

    def _position_age_steps(self) -> int:
        """Retourne l'age courant de la position en nombre de pas."""
        if self.position_size == 0:
            return 0
        return max(0, int(self.current_step) - int(self.position_entry_step))

    def _is_near_hard_stop(self, price: float) -> bool:
        """Indique si le prix courant est proche du stop implicite courant.

        Args:
            price (float): Prix courant.

        Returns:
            bool: ``True`` si la position est a faible distance du stop.
        """
        if self.position_size == 0 or self.hard_stop_price <= 0.0:
            return False
        total_buffer = abs(float(self.avg_entry_price) - float(self.hard_stop_price))
        if total_buffer <= 0.0:
            return False
        remaining_buffer = abs(float(price) - float(self.hard_stop_price))
        return remaining_buffer <= max(total_buffer * 0.25, abs(float(self.avg_entry_price)) * 0.0002)

    def _hard_stop_triggered(self, price: float) -> bool:
        """Detecte si le stop implicite doit fermer la position.

        Args:
            price (float): Prix courant observe.

        Returns:
            bool: ``True`` si le stop implicite est touche.
        """
        if self.position_size == 0 or self.hard_stop_price <= 0.0:
            return False
        if self.position_size > 0:
            return price <= self.hard_stop_price
        return price >= self.hard_stop_price

    def _activate_runner_extension(
        self,
        *,
        price: float,
        context: dict[str, float],
        exit_plan_policy: dict[str, float],
    ) -> None:
        """Active l'extension du runner apres un split utile.

        Args:
            price (float): Prix courant du split.
            context (dict[str, float]): Contexte de marche courant.
            exit_plan_policy (dict[str, float]): Parametres implicites de TP/SL.
        """
        if self.position_size == 0:
            return
        atr_pct = max(float(context.get("atr_pct", 0.0) or 0.0), 1e-5)
        extension_return = max(
            atr_pct * self._policy_float(exit_plan_policy, "runner_extension_atr_mult", 0.80),
            atr_pct,
        )
        if self.position_size > 0:
            self.full_tp_price = max(self.full_tp_price, float(price) * (1.0 + extension_return))
            if self.slbe_active and self.slbe_price > 0.0:
                self.hard_stop_price = max(self.hard_stop_price, self.slbe_price)
        else:
            self.full_tp_price = min(self.full_tp_price, float(price) * (1.0 - extension_return))
            if self.slbe_active and self.slbe_price > 0.0:
                self.hard_stop_price = min(self.hard_stop_price, self.slbe_price)
        if not self.runner_extension_active:
            self.runner_extension_count += 1
        self.runner_extension_active = True
        self.runner_active = True
        self.runner_protected = bool(self.slbe_active)
        self.runner_target_price = float(self.full_tp_price or price)

    def _build_position_management_snapshot(
        self,
        *,
        price: float,
        context: dict[str, float],
        entry_filter: dict[str, float | bool | str],
        hold_policy: dict[str, float],
        split_policy: dict[str, float],
        pyramiding_policy: dict[str, float],
        close_policy: dict[str, float],
        exit_plan_policy: dict[str, float],
        trade_notional: float,
    ) -> dict[str, float | bool]:
        """Construit les signaux utiles au pilotage de position.

        Args:
            price (float): Prix courant.
            context (dict[str, float]): Contexte de marche du pas courant.
            entry_filter (dict[str, float | bool | str]): Filtre d'entree actif.
            hold_policy (dict[str, float]): Regles de gestion HOLD.
            split_policy (dict[str, float]): Regles de split.
            pyramiding_policy (dict[str, float]): Regles de pyramiding.
            close_policy (dict[str, float]): Regles de cloture.
            exit_plan_policy (dict[str, float]): Parametres implicites de TP/SL.
            trade_notional (float): Taille notionnelle d'un ordre unitaire.

        Returns:
            dict[str, float | bool]: Instantane des opportunites et signaux
                de gestion de position.
        """
        if self.position_size == 0:
            return {
                "trade_ret": 0.0,
                "peak_trade_return": 0.0,
                "tp_like_threshold": float(
                    close_policy.get(
                        "tp_like_threshold",
                        close_policy.get("winner_threshold", 0.0),
                    )
                    or 0.0
                ),
                "winner_threshold": float(close_policy.get("winner_threshold", 0.0) or 0.0),
                "split_opportunity": False,
                "split_tp_zone_opportunity": False,
                "pyramid_opportunity": False,
                "pyramid_add_opportunity": False,
                "hold_drag_opportunity": False,
                "reversal_context": False,
                "offensive_reversal_context": False,
                "clear_reversal_context": False,
                "strong_trend_support": False,
                "moderate_trend_support": False,
                "continuation_support": False,
                "offensive_continuation_support": False,
                "soft_tp_zone_active": False,
                "soft_tp_hit": False,
                "full_tp_hit": False,
                "time_stop_expired": False,
                "time_stop_grace_expired": False,
                "near_hard_stop": False,
                "runner_extension_active": False,
                "runner_extension_opportunity": False,
                "split_monetization_window": False,
                "runner_viable_window": False,
                "runner_profit_hold_window": False,
                "runner_hold_after_soft_tp": False,
                "pyramid_monetization_window": False,
                "profit_peak_reached": False,
                "position_peak_giveback_ratio": 0.0,
                "offensive_profit_floor": 0.0,
            }

        direction_sign = 1.0 if self.position_size > 0 else -1.0
        trade_ret = self._get_unrealized_return(price)
        winner_threshold = float(close_policy.get("winner_threshold", 0.01) or 0.01)
        tp_like_threshold = float(close_policy.get("tp_like_threshold", winner_threshold) or winner_threshold)
        trend_adx = float(
            entry_filter.get(
                "trend_adx",
                entry_filter.get("min_adx", 20.0),
            )
            or 20.0
        )
        min_profit_to_add = float(pyramiding_policy.get("min_profit_to_add", 0.001) or 0.001)
        max_additions = int(pyramiding_policy.get("max_additions", 1) or 1)
        max_splits = int(split_policy.get("max_splits", 3) or 3)
        min_trade_return = float(split_policy.get("min_trade_return", 0.01) or 0.01)
        drag_profit_floor = self._policy_float(hold_policy, "drag_profit_floor", 0.0040)
        recovery_grace_steps = self._policy_int(exit_plan_policy, "recovery_grace_steps", 0)

        atr_pct = max(float(context.get("atr_pct", 0.0) or 0.0), 1e-5)
        momentum = float(context.get("momentum", 0.0) or 0.0)
        price_vs_vwap = float(context.get("price_vs_vwap", 0.0) or 0.0)
        adx = float(context.get("adx", 0.0) or 0.0)
        momentum_floor = max(atr_pct * 0.35, 1e-5)
        vwap_reversal_floor = max(atr_pct * 0.10, 0.00005)
        vwap_continuation_floor = max(atr_pct * 0.06, 0.00003)
        vwap_clear_reversal_floor = max(atr_pct * 0.18, 0.00008)
        continuation_adx_floor = max(trend_adx * 0.70, 12.0)
        favorable_momentum = (
            (direction_sign > 0 and momentum >= momentum_floor)
            or (direction_sign < 0 and momentum <= -momentum_floor)
        )
        reversal_momentum = (
            (direction_sign > 0 and momentum <= -momentum_floor)
            or (direction_sign < 0 and momentum >= momentum_floor)
        )
        reversal_vwap = (
            (direction_sign > 0 and price_vs_vwap <= -vwap_reversal_floor)
            or (direction_sign < 0 and price_vs_vwap >= vwap_reversal_floor)
        )
        vwap_continuation = (
            (direction_sign > 0 and price_vs_vwap >= -vwap_continuation_floor)
            or (direction_sign < 0 and price_vs_vwap <= vwap_continuation_floor)
        )
        clear_reversal_vwap = (
            (direction_sign > 0 and price_vs_vwap <= -vwap_clear_reversal_floor)
            or (direction_sign < 0 and price_vs_vwap >= vwap_clear_reversal_floor)
        )
        adx_rollover = adx < trend_adx
        reversal_context = reversal_momentum or reversal_vwap or adx_rollover
        offensive_reversal_context = reversal_momentum or clear_reversal_vwap
        strong_trend_support = adx >= trend_adx and favorable_momentum
        moderate_trend_support = (
            adx >= continuation_adx_floor
            and (
                favorable_momentum
                or (
                    trade_ret >= winner_threshold
                    and vwap_continuation
                )
            )
        )
        soft_tp_near_active = trade_ret >= (tp_like_threshold * 0.85)
        soft_tp_zone_active = (
            bool(self.soft_tp_hit_active)
            or bool(self.full_tp_hit_active)
            or soft_tp_near_active
        )
        continuation_retest_support = (
            trade_ret >= max(min_trade_return, tp_like_threshold * 0.65)
            and vwap_continuation
            and adx >= max(continuation_adx_floor * 0.55, 8.0)
        )
        continuation_support = (
            strong_trend_support
            or moderate_trend_support
            or continuation_retest_support
        )
        offensive_continuation_support = (
            continuation_support
            or (
                soft_tp_zone_active
                and vwap_continuation
                and adx >= max(continuation_adx_floor * 0.45, 7.0)
            )
        )
        clear_reversal_context = (
            reversal_momentum
            or clear_reversal_vwap
            or adx < max(continuation_adx_floor * 0.60, 7.0)
        )
        offensive_profit_floor = max(
            min_profit_to_add,
            min(
                max(min_trade_return * 0.40, min_profit_to_add),
                tp_like_threshold * 0.40,
                winner_threshold * 0.30,
            ),
        )
        near_hard_stop = self._is_near_hard_stop(price)
        current_positive_return = max(0.0, trade_ret)
        peak_trade_return = max(float(self.position_peak_return or 0.0), current_positive_return)
        monetization_return_floor = max(
            min_trade_return * 0.60,
            tp_like_threshold * 0.20,
            min_profit_to_add * 0.80,
            1e-6,
        )
        runner_hold_floor = max(
            min_trade_return * 0.20,
            tp_like_threshold * 0.10,
            1e-6,
        )
        pyramid_window_floor = max(
            min_profit_to_add * 0.75,
            tp_like_threshold * 0.15,
            1e-6,
        )
        position_peak_giveback_ratio = (
            max(0.0, peak_trade_return - current_positive_return) / max(peak_trade_return, 1e-6)
            if peak_trade_return > 0.0
            else 0.0
        )
        profit_peak_reached = peak_trade_return >= monetization_return_floor
        qualifying_profit = trade_ret >= max(winner_threshold, drag_profit_floor)
        has_management_trigger = (
            trade_ret >= tp_like_threshold
            or self.slbe_active
            or self.position_pyramids > 0
        )

        split_opportunity = (
            abs(self.position_size) >= trade_notional * 0.75
            and self.split_count < max_splits
            and trade_ret >= min_trade_return
        )
        # Les fenetres offensives doivent s'ouvrir sur la trajectoire du trade.
        # Le giveback reste juge par la reward et non par un veto dur.
        split_monetization_window = (
            split_opportunity
            and trade_ret >= monetization_return_floor
            and peak_trade_return >= current_positive_return
        )
        split_tp_zone_opportunity = split_monetization_window
        pyramid_opportunity = (
            trade_ret >= min_profit_to_add
            and self.position_pyramids < max_additions
        )
        pyramid_monetization_window = (
            pyramid_opportunity
            and trade_ret >= pyramid_window_floor
            and current_positive_return > 0.0
        )
        pyramid_add_opportunity = (
            pyramid_monetization_window
        )
        runner_peak_floor = max(
            min_trade_return * 0.25,
            tp_like_threshold * 0.20,
            1e-6,
        )
        runner_context_live = bool(self.runner_active or self.runner_extension_active)
        runner_viable_window = (
            peak_trade_return >= runner_peak_floor
            and current_positive_return >= max(peak_trade_return * 0.25, runner_hold_floor, 1e-6)
        )
        runner_profit_hold_window = (
            runner_context_live
            and peak_trade_return >= runner_peak_floor
            and current_positive_return >= max(peak_trade_return * 0.35, 1e-6)
        )
        runner_hold_after_soft_tp = runner_profit_hold_window and soft_tp_zone_active
        runner_extension_context_ready = (
            self.runner_active
            or self.runner_extension_active
            or (
                self.split_count > 0
                and (
                    self.slbe_active
                    or self.runner_protected
                )
            )
        )
        runner_extension_opportunity = runner_extension_context_ready and runner_profit_hold_window
        hold_drag_opportunity = qualifying_profit and has_management_trigger and reversal_context
        position_age_steps = self._position_age_steps()
        time_stop_expired = self.time_stop_steps > 0 and position_age_steps >= self.time_stop_steps
        time_stop_grace_expired = (
            self.time_stop_steps > 0
            and position_age_steps >= (self.time_stop_steps + max(recovery_grace_steps, 0))
        )

        return {
            "trade_ret": trade_ret,
            "peak_trade_return": peak_trade_return,
            "tp_like_threshold": tp_like_threshold,
            "winner_threshold": winner_threshold,
            "split_opportunity": split_opportunity,
            "split_tp_zone_opportunity": split_tp_zone_opportunity,
            "pyramid_opportunity": pyramid_opportunity,
            "pyramid_add_opportunity": pyramid_add_opportunity,
            "hold_drag_opportunity": hold_drag_opportunity,
            "reversal_context": reversal_context,
            "offensive_reversal_context": offensive_reversal_context,
            "clear_reversal_context": clear_reversal_context,
            "strong_trend_support": strong_trend_support,
            "moderate_trend_support": moderate_trend_support,
            "continuation_support": continuation_support,
            "offensive_continuation_support": offensive_continuation_support,
            "soft_tp_zone_active": soft_tp_zone_active,
            "soft_tp_hit": bool(self.soft_tp_hit_active),
            "full_tp_hit": bool(self.full_tp_hit_active),
            "time_stop_expired": time_stop_expired,
            "time_stop_grace_expired": time_stop_grace_expired,
            "near_hard_stop": near_hard_stop,
            "runner_extension_active": bool(self.runner_extension_active),
            "runner_extension_opportunity": runner_extension_opportunity,
            "split_monetization_window": split_monetization_window,
            "runner_viable_window": runner_viable_window,
            "runner_profit_hold_window": runner_profit_hold_window,
            "runner_hold_after_soft_tp": runner_hold_after_soft_tp,
            "pyramid_monetization_window": pyramid_monetization_window,
            "profit_peak_reached": profit_peak_reached,
            "position_peak_giveback_ratio": position_peak_giveback_ratio,
            "offensive_profit_floor": offensive_profit_floor,
        }

    def _register_position_management_opportunities(
        self,
        snapshot: dict[str, float | bool],
    ) -> None:
        """Cumule les opportunites de gestion observees sur le pas courant.

        Args:
            snapshot (dict[str, float | bool]): Instantane des signaux de
                gestion construit avant l'action.
        """
        if bool(snapshot.get("hold_drag_opportunity", False)):
            self.hold_drag_opportunity_count += 1
        if bool(snapshot.get("split_opportunity", False)):
            self.split_opportunity_count += 1
        if bool(snapshot.get("split_tp_zone_opportunity", False)):
            self.split_tp_zone_opportunity_count += 1
        if bool(snapshot.get("split_monetization_window", False)):
            self.split_monetization_window_count += 1
        if bool(snapshot.get("pyramid_opportunity", False)):
            self.pyramid_opportunity_count += 1
        if bool(snapshot.get("pyramid_add_opportunity", False)):
            self.pyramid_add_opportunity_count += 1
        if bool(snapshot.get("pyramid_monetization_window", False)):
            self.pyramid_monetization_window_count += 1
        if bool(snapshot.get("runner_extension_opportunity", False)):
            self.runner_extension_opportunity_count += 1
        if bool(snapshot.get("runner_viable_window", False)):
            self.runner_viable_window_count += 1
        if bool(snapshot.get("runner_profit_hold_window", False)):
            self.runner_profit_hold_window_count += 1
        if bool(snapshot.get("runner_hold_after_soft_tp", False)):
            self.runner_hold_after_soft_tp_count += 1
        if bool(snapshot.get("profit_peak_reached", False)):
            self.profit_peak_reached_count += 1
            self.profit_peak_giveback_ratio_total += float(
                snapshot.get("position_peak_giveback_ratio", 0.0) or 0.0
            )
            self.profit_peak_giveback_ratio_observations += 1

    def _update_offensive_window_wait_steps(
        self,
        *,
        active: bool,
        captured: bool,
        current_wait_steps: int,
    ) -> tuple[int, bool]:
        """Met a jour l'attente avant de penaliser une fenetre offensive ratee.

        Args:
            active (bool): Indique si la fenetre offensive est ouverte au pas courant.
            captured (bool): Indique si une action utile a ete executee.
            current_wait_steps (int): Nombre de pas deja attendus sans action utile.

        Returns:
            tuple[int, bool]: Nouveau compteur d'attente et drapeau de fermeture
                ratee si la fenetre vient de se refermer apres au moins trois pas.
        """
        if captured:
            return 0, False
        if active:
            return current_wait_steps + 1, False
        if current_wait_steps >= 3:
            return 0, True
        return 0, False

    def _mark_runner_after_split(
        self,
        *,
        price: float,
        runner_protected: bool,
        context: dict[str, float],
        exit_plan_policy: dict[str, float],
    ) -> None:
        """Initialise l'etat interne du runner apres un split.

        Args:
            price (float): Prix courant du split.
            runner_protected (bool): Indique si le runner est deja protege.
            context (dict[str, float]): Contexte de marche du pas courant.
            exit_plan_policy (dict[str, float]): Parametres du plan de sortie.
        """
        if self.position_size == 0:
            return

        self.runner_active = True
        self.runner_entry_price = float(price)
        self.runner_origin_split_step = int(self.current_step)
        self.runner_protected = bool(runner_protected)
        self.runner_target_price = float(self.full_tp_price or price)
        self.runner_entry_profit_pct = max(0.0, self._get_unrealized_pnl_pct(price))
        self.runner_peak_profit_pct = self.runner_entry_profit_pct
        if bool(self.runner_extension_active):
            self.runner_target_price = float(self.full_tp_price or self.runner_target_price)
        elif bool(context.get("momentum", 0.0) or 0.0) and float(context.get("adx", 0.0) or 0.0) > 0.0:
            self.runner_target_price = float(self.full_tp_price or self.runner_target_price)

        if self.runner_target_price <= 0.0:
            self.runner_target_price = float(price)

    def _capture_runner_exit_context(self) -> dict[str, float | bool | int | None]:
        """Capture l'etat du runner avant une cloture potentielle.

        Returns:
            dict[str, float | bool | int | None]: Contexte minimum utile
                pour evaluer la monétisation offensive du runner apres
                la realise de position.
        """
        steps_since_split: int | None = None
        if self.runner_active and self.runner_origin_split_step >= 0:
            steps_since_split = max(0, int(self.current_step) - int(self.runner_origin_split_step))
        return {
            "active": bool(self.runner_active),
            "protected": bool(self.runner_protected),
            "steps_since_split": steps_since_split,
            "peak_profit_pct": float(self.runner_peak_profit_pct),
            "entry_profit_pct": float(self.runner_entry_profit_pct),
        }

    def _register_runner_exit_outcome(
        self,
        *,
        realized_trade: float,
        forced_exit: bool,
        runner_was_active: bool | None = None,
        runner_peak_profit_pct: float | None = None,
        runner_entry_profit_pct: float | None = None,
    ) -> float:
        """Enregistre le resultat final du runner apres un split.

        Args:
            realized_trade (float): PnL realise sur la cloture finale.
            forced_exit (bool): Indique si la sortie a ete subie.
            runner_was_active (bool | None): Etat runner capture avant la
                cloture, pour ne pas perdre l'information apres reset.
            runner_peak_profit_pct (float | None): Pic de profit enregistre
                avant la cloture du runner.
            runner_entry_profit_pct (float | None): Profit disponible lors du
                split initial, deja normalise en pourcentage du capital.

        Returns:
            float: Delta de valeur apporte par le runner sur le trade complet,
                exprime en pourcentage du capital initial.
        """
        if runner_was_active is None:
            runner_was_active = bool(self.runner_active)
        if not runner_was_active:
            return 0.0

        peak_profit_pct = max(
            0.0,
            float(
                self.runner_peak_profit_pct
                if runner_peak_profit_pct is None
                else runner_peak_profit_pct
            ),
        )
        entry_profit_pct = max(
            0.0,
            float(
                self.runner_entry_profit_pct
                if runner_entry_profit_pct is None
                else runner_entry_profit_pct
            ),
        )
        realized_pct = (realized_trade / self.spec.initial_balance) * 100.0
        retained_profit_pct = max(0.0, realized_pct)
        if peak_profit_pct <= 0.0 and retained_profit_pct > 0.0:
            peak_profit_pct = retained_profit_pct
        giveback_pct = max(0.0, peak_profit_pct - retained_profit_pct)
        retention_ratio = (
            retained_profit_pct / peak_profit_pct
            if peak_profit_pct > 1e-8
            else 0.0
        )

        self.runner_managed_exit_count += 1
        self.split_trade_value_delta += realized_pct
        self._last_runner_retained_profit_pct = retained_profit_pct
        self._last_runner_giveback_pct = giveback_pct
        self._last_runner_retention_ratio = retention_ratio

        if realized_trade > 0.0:
            self.runner_exit_profitable_count += 1
            self.split_runner_profitable_count += 1
            self.split_improved_total_trade_count += 1
            self.runner_retained_profit_pct += retained_profit_pct
        else:
            self.split_runner_failed_count += 1
            if forced_exit:
                self.runner_forced_stop_count += 1
        if giveback_pct > 0.0:
            self.runner_giveback_pct += giveback_pct
        elif realized_pct < 0.0 and peak_profit_pct <= 0.0 and entry_profit_pct <= 0.0:
            self.runner_giveback_pct += abs(realized_pct)
        elif entry_profit_pct > 0.0 and retained_profit_pct <= 0.0:
            self.runner_giveback_pct += entry_profit_pct
        return realized_pct

    def _register_pyramid_exit_outcome(
        self,
        *,
        had_pyramids: bool,
        realized_trade: float,
        trade_ret: float,
        close_policy: dict[str, float],
        baseline_trade_return: float | None = None,
    ) -> float:
        """Enregistre la qualite de sortie d'un trade pyramide.

        Args:
            had_pyramids (bool): Indique si le trade etait pyramide.
            realized_trade (float): PnL final realise.
            trade_ret (float): Rendement global du trade.
            close_policy (dict[str, float]): Regles CLOSE utilisees.
            baseline_trade_return (float | None): Rendement latent present avant
                le premier ajout pyramidal du trade.

        Returns:
            float: Amelioration totale du trade attribuable au pyramiding,
                exprimee en points de rendement.
        """
        if not had_pyramids:
            return 0.0

        baseline_return = (
            float(baseline_trade_return)
            if baseline_trade_return is not None
            else float(self._pyramid_baseline_return if self._pyramid_baseline_active else 0.0)
        )
        trade_improvement_pct = max(0.0, (float(trade_ret) - baseline_return) * 100.0)
        self.pyramid_total_trade_improvement_pct += trade_improvement_pct
        if trade_improvement_pct <= 0.0:
            self.pyramid_failed_to_improve_count += 1
        if realized_trade > 0.0 and trade_ret > 0.0 and trade_improvement_pct > 0.0:
            self.pyramid_profitable_exit_count += 1
        self._pyramid_baseline_active = False
        self._pyramid_baseline_return = 0.0
        return trade_improvement_pct

    def _compute_hold_drag_penalty(
        self,
        profitable_return: float,
        hold_policy: dict[str, float],
        hold_drag_multiplier: float,
    ) -> float:
        """Calcule une penalite HOLD bornee sur une opportunite de gestion.

        Args:
            profitable_return (float): Rendement latent positif du trade.
            hold_policy (dict[str, float]): Regles HOLD actives.
            hold_drag_multiplier (float): Multiplicateur de penalite issu du
                profil de reward.

        Returns:
            float: Penalite a soustraire a la reward.
        """
        raw_penalty = max(0.0, profitable_return) * 100.0 * max(hold_drag_multiplier, 0.0)
        penalty_cap = self._policy_float(hold_policy, "drag_penalty_cap", 1.25)
        return min(raw_penalty, max(penalty_cap, 0.0))

    def _compute_close_management_reward(
        self,
        *,
        realized_trade: float,
        trade_ret: float,
        snapshot: dict[str, float | bool],
        close_policy: dict[str, float],
        pyramiding_policy: dict[str, float],
        slbe_policy: dict[str, float],
        reward_terms: dict[str, float],
        close_realized_multiplier: float,
        had_pyramids: bool,
        pyramid_count_before_close: int,
        had_locked_profit: bool,
        runner_active_before_close: bool,
        runner_protected_before_close: bool,
        runner_steps_since_split: int | None,
        split_trade_value_delta_pct: float,
        runner_retained_profit_pct: float,
        runner_giveback_pct: float,
        runner_retention_ratio: float,
        runner_giveback_ratio: float,
        pyramid_trade_improvement_pct: float,
        phase_scale: float,
    ) -> float:
        """Calcule la reward de pilotage sur une sortie choisie par l'agent.

        Args:
            realized_trade (float): PnL realise de la sortie.
            trade_ret (float): Rendement du trade clos.
            snapshot (dict[str, float | bool]): Instantane de gestion construit
                avant l'action.
            close_policy (dict[str, float]): Regles CLOSE.
            pyramiding_policy (dict[str, float]): Regles de pyramiding.
            slbe_policy (dict[str, float]): Regles SLBE.
            reward_terms (dict[str, float]): Parametres de reward resolves.
            close_realized_multiplier (float): Multiplicateur de reward sur
                les sorties gagnantes.
            had_pyramids (bool): Indique si la position avait ete pyramidée.
            pyramid_count_before_close (int): Nombre d'ajouts avant la sortie.
            had_locked_profit (bool): Indique si le SLBE etait deja en phase
                de verrouillage de profit.
            runner_active_before_close (bool): Indique si un runner issu d'un
                split etait encore vivant.
            runner_protected_before_close (bool): Indique si le runner etait
                protege au moment de la sortie.
            runner_steps_since_split (int | None): Age du runner depuis le
                split initial.
            split_trade_value_delta_pct (float): Valeur additionnelle apportee
                par le runner sur le trade complet.
            runner_retained_profit_pct (float): Profit effectivement retenu
                par le runner au moment de la sortie.
            runner_giveback_pct (float): Profit rendu au marche apres le pic
                du runner.
            runner_retention_ratio (float): Part du pic de profit finalement
                conservee par le runner.
            runner_giveback_ratio (float): Part du pic de profit rendue au
                marche par le runner.
            pyramid_trade_improvement_pct (float): Gain de sortie attribuable
                au pyramiding sur le trade complet.
            phase_scale (float): Intensite du shaping V5 pour la phase
                d'entrainement courante.

        Returns:
            float: Ajustement de reward a appliquer.
        """
        phase_scale = max(0.0, float(phase_scale or 0.0))
        core_reward = 0.0
        management_adjustment = 0.0
        strong_winner_threshold = float(close_policy.get("strong_winner_threshold", 0.02) or 0.02)
        winner_threshold = float(close_policy.get("winner_threshold", 0.01) or 0.01)
        tp_like_threshold = float(close_policy.get("tp_like_threshold", winner_threshold) or winner_threshold)
        reversal_close_bonus = self._policy_float(close_policy, "reversal_close_bonus", 0.35)
        early_profit_close_penalty = self._policy_float(
            close_policy,
            "early_profit_close_penalty",
            0.20,
        )
        split_decorative_penalty = float(reward_terms.get("split_decorative_penalty", 0.15) or 0.15)
        pyramid_exit_capture_bonus = float(reward_terms.get("pyramid_exit_capture_bonus", 0.35) or 0.35)
        pyramid_trade_completion_bonus = float(
            reward_terms.get("pyramid_trade_completion_bonus", 0.30) or 0.30
        )
        pyramid_stagnant_exit_penalty = float(
            reward_terms.get("pyramid_stagnant_exit_penalty", 0.25) or 0.25
        )
        runner_protected_exit_bonus = float(
            reward_terms.get("runner_protected_exit_bonus", 0.30) or 0.30
        )
        runner_viable_but_closed_penalty = float(
            reward_terms.get("runner_viable_but_closed_penalty", 0.18) or 0.18
        )
        early_full_close_after_soft_tp_penalty = float(
            reward_terms.get("early_full_close_after_soft_tp_penalty", 0.22) or 0.22
        )
        runner_retained_profit_bonus = float(
            reward_terms.get("runner_retained_profit_bonus", 0.25) or 0.25
        )
        runner_trade_completion_bonus = float(
            reward_terms.get("runner_trade_completion_bonus", 0.40) or 0.40
        )
        runner_giveback_penalty = float(
            reward_terms.get("runner_giveback_penalty", 0.45) or 0.45
        )
        runner_giveback_ratio_penalty = float(
            reward_terms.get("runner_giveback_ratio_penalty", 0.35) or 0.35
        )
        split_zone_capture_bonus = float(
            reward_terms.get("split_zone_capture_bonus", 0.18) or 0.18
        )
        pyramid_add_capture_bonus = float(
            reward_terms.get("pyramid_add_capture_bonus", 0.18) or 0.18
        )
        realized_pct = (realized_trade / self.spec.initial_balance) * 100.0
        defensive_close = trade_ret > 0.0 and bool(snapshot.get("reversal_context", False))
        offensive_close_zone = (
            realized_trade > 0.0
            and trade_ret > 0.0
            and not defensive_close
            and (
                bool(snapshot.get("split_tp_zone_opportunity", False))
                or bool(snapshot.get("runner_extension_opportunity", False))
                or bool(snapshot.get("pyramid_add_opportunity", False))
            )
        )
        early_close_noise = (
            realized_trade > 0.0
            and trade_ret > 0.0
            and trade_ret < winner_threshold
            and not defensive_close
        )
        near_hard_stop = bool(snapshot.get("near_hard_stop", False))
        time_stop_expired = bool(snapshot.get("time_stop_expired", False))
        runner_viable_window = bool(snapshot.get("runner_viable_window", False))
        # Le nom historique est conserve pour la compatibilite des metriques,
        # mais la penalite vise maintenant toute fermeture totale d'un runner
        # encore viable, meme hors zone `soft_tp` stricte.
        early_full_close_after_soft_tp = (
            realized_trade > 0.0
            and runner_viable_window
            and not defensive_close
            and not near_hard_stop
            and not time_stop_expired
            and not runner_active_before_close
        )

        if trade_ret > strong_winner_threshold:
            core_reward += self.quality_mult * 1.5 + (realized_pct * close_realized_multiplier)
            self.realized_close_bonus_count += 1
        elif trade_ret > winner_threshold:
            core_reward += self.quality_mult + (realized_pct * close_realized_multiplier)
            self.realized_close_bonus_count += 1
        elif defensive_close:
            core_reward += max(realized_pct, -reversal_close_bonus * 0.5)
            management_adjustment += reversal_close_bonus
        elif realized_trade > 0:
            core_reward += max(0.25, realized_pct * close_realized_multiplier)
        else:
            core_reward += realized_pct

        if defensive_close:
            self.defensive_close_count += 1
            self.close_winner_count += 1
        elif realized_trade > 0:
            self.close_winner_count += 1
        elif realized_trade < 0:
            self.close_loser_count += 1

        if trade_ret >= tp_like_threshold and realized_trade > 0:
            self.tp_like_exit_count += 1

        if early_close_noise:
            self.early_close_noise_count += 1
            management_adjustment -= early_profit_close_penalty
        elif early_full_close_after_soft_tp:
            self.early_full_close_after_soft_tp_count += 1
            self.runner_viable_but_closed_count += 1
            management_adjustment -= early_full_close_after_soft_tp_penalty
            management_adjustment -= runner_viable_but_closed_penalty
        elif offensive_close_zone:
            management_adjustment -= early_profit_close_penalty * 0.35

        if near_hard_stop and bool(snapshot.get("reversal_context", False)):
            self.forced_stop_near_miss_count += 1
            management_adjustment += min(0.35, reversal_close_bonus)

        if time_stop_expired and realized_trade > 0:
            management_adjustment += 0.15

        if had_pyramids and realized_trade > 0:
            continuation_bonus = float(
                pyramiding_policy.get("strong_trend_reward_bonus", 0.20) or 0.20
            )
            if pyramid_trade_improvement_pct > 0.0:
                management_adjustment += continuation_bonus * max(1, min(pyramid_count_before_close, 2))
                if trade_ret >= winner_threshold:
                    management_adjustment += min(
                        pyramid_trade_completion_bonus,
                        max(0.10, pyramid_trade_improvement_pct * 0.25),
                    )
                    management_adjustment += min(
                        pyramid_exit_capture_bonus,
                        max(0.08, pyramid_trade_improvement_pct * 0.15),
                    )
                    management_adjustment += min(
                        pyramid_add_capture_bonus,
                        max(0.05, pyramid_trade_improvement_pct * 0.12),
                    )
            else:
                management_adjustment -= max(
                    float(reward_terms.get("pyramid_bad_add_penalty", 0.35) or 0.35) * 0.50,
                    pyramid_stagnant_exit_penalty,
                )

        if had_locked_profit and realized_trade > 0:
            management_adjustment += min(
                1.0,
                float(slbe_policy.get("exit_bonus", 0.0) or 0.0) * 0.35,
            )

        if runner_active_before_close:
            if (
                runner_steps_since_split is not None
                and runner_steps_since_split <= 1
                and not bool(snapshot.get("reversal_context", False))
            ):
                management_adjustment -= split_decorative_penalty
            elif split_trade_value_delta_pct > 0.0:
                management_adjustment += min(
                    float(reward_terms.get("split_runner_profit_bonus", 0.45) or 0.45),
                    max(0.10, split_trade_value_delta_pct * 0.25),
                )
                management_adjustment += min(
                    split_zone_capture_bonus,
                    max(0.05, split_trade_value_delta_pct * 0.15),
                )
                if runner_retained_profit_pct > 0.0:
                    management_adjustment += min(
                        runner_retained_profit_bonus,
                        max(0.05, runner_retained_profit_pct * 0.20),
                    )
                if runner_retention_ratio > 0.0:
                    management_adjustment += min(
                        runner_trade_completion_bonus,
                        max(0.05, runner_retention_ratio * runner_trade_completion_bonus),
                    )
                if runner_protected_before_close:
                    management_adjustment += runner_protected_exit_bonus
                if runner_giveback_pct > 0.0:
                    management_adjustment -= min(
                        runner_giveback_penalty,
                        max(0.05, runner_giveback_pct * 0.25),
                    )
                if runner_giveback_ratio > 0.0:
                    management_adjustment -= min(
                        runner_giveback_ratio_penalty,
                        max(0.05, runner_giveback_ratio * runner_giveback_ratio_penalty),
                    )
            elif split_trade_value_delta_pct < 0.0:
                management_adjustment -= max(
                    split_decorative_penalty * 0.5,
                    abs(split_trade_value_delta_pct) * 0.40,
                )
                if runner_giveback_pct > 0.0:
                    management_adjustment -= min(
                        runner_giveback_penalty,
                        max(0.05, runner_giveback_pct * 0.30),
                    )
                if runner_giveback_ratio > 0.0:
                    management_adjustment -= min(
                        runner_giveback_ratio_penalty,
                        max(0.05, runner_giveback_ratio * runner_giveback_ratio_penalty),
                    )
            elif realized_trade <= 0.0 and not runner_protected_before_close:
                management_adjustment -= split_decorative_penalty * 0.5
                if runner_giveback_pct > 0.0:
                    management_adjustment -= min(
                        runner_giveback_penalty,
                        max(0.05, runner_giveback_pct * 0.25),
                    )
                if runner_giveback_ratio > 0.0:
                    management_adjustment -= min(
                        runner_giveback_ratio_penalty,
                        max(0.05, runner_giveback_ratio * runner_giveback_ratio_penalty),
                    )

        return core_reward + (management_adjustment * phase_scale)

    def _compute_soft_entry_quality_adjustment(
        self,
        action: int,
        context: dict[str, float],
        entry_filter: dict[str, float | bool | str],
        reward_terms: dict[str, float],
    ) -> float:
        """Calcule un ajustement doux de reward pour la qualite d'entree.

        L'objectif est de redonner du signal a la policy sans revenir aux
        veto durs qui bloquaient artificiellement l'exploration. Les setups
        contre-tendance ou de faible qualite restent possibles, mais deviennent
        moins attractifs qu'une entree propre et alignee.

        Args:
            action (int): Action directionnelle executee.
            context (dict[str, float]): Contexte de marche du pas courant.
            entry_filter (dict[str, float | bool | str]): Filtre actif.
            reward_terms (dict[str, float]): Parametres de reward resolves.

        Returns:
            float: Ajustement de reward positif ou negatif.
        """
        if action not in (BUY, SELL):
            return 0.0

        direction_sign = 1.0 if action == BUY else -1.0
        fallback_direction_ok = self._compute_fallback_direction_ok(context, entry_filter)
        min_adx = float(entry_filter.get("min_adx", 0.0) or 0.0)
        trend_adx = float(entry_filter.get("trend_adx", min_adx) or min_adx)
        atr_pct = max(float(context.get("atr_pct", 0.0) or 0.0), 1e-5)
        ema_gap_pct = float(context.get("ema_gap_pct", 0.0) or 0.0)
        price_vs_vwap = float(context.get("price_vs_vwap", 0.0) or 0.0)
        momentum = float(context.get("momentum", 0.0) or 0.0)
        obv_slope = float(context.get("obv_slope", 0.0) or 0.0)
        adx = float(context.get("adx", 0.0) or 0.0)

        against_ema = (direction_sign > 0 and ema_gap_pct < 0.0) or (
            direction_sign < 0 and ema_gap_pct > 0.0
        )
        against_vwap = (direction_sign > 0 and price_vs_vwap < 0.0) or (
            direction_sign < 0 and price_vs_vwap > 0.0
        )
        against_obv = (direction_sign > 0 and obv_slope <= 0.0) or (
            direction_sign < 0 and obv_slope >= 0.0
        )
        against_momentum = (direction_sign > 0 and momentum < 0.0) or (
            direction_sign < 0 and momentum > 0.0
        )

        ema_gap_floor = max(atr_pct * 0.20, 0.0001)
        vwap_gap_floor = max(atr_pct * 0.12, 0.00005)
        momentum_floor = max(atr_pct * 0.50, 1e-5)
        phase_penalty_scale, phase_bonus_scale = self._resolve_soft_reward_phase_scales()
        penalty_scale = (0.5 if fallback_direction_ok else 1.0) * phase_penalty_scale
        bonus_scale = phase_bonus_scale
        reward_adjustment = 0.0
        strong_countertrend = abs(ema_gap_pct) >= ema_gap_floor and (
            against_momentum or against_vwap
        )
        net_vwap_contradiction = abs(price_vs_vwap) >= vwap_gap_floor and (
            (against_ema and abs(ema_gap_pct) >= ema_gap_floor * 0.5)
            or abs(momentum) >= momentum_floor * 0.8
        )
        adx_structure_required = abs(momentum) >= momentum_floor or abs(ema_gap_pct) >= ema_gap_floor * 0.5

        if against_ema and strong_countertrend:
            ema_severity = min(1.0, abs(ema_gap_pct) / max(ema_gap_floor, 1e-6))
            momentum_severity = (
                min(1.0, abs(momentum) / max(momentum_floor, 1e-6))
                if against_momentum
                else 0.0
            )
            ema_penalty = reward_terms["soft_countertrend_ema_penalty"] * penalty_scale * (
                0.55 + (0.45 * max(ema_severity, momentum_severity))
            )
            reward_adjustment -= ema_penalty
            self.soft_penalty_ema200_count += 1

        if against_vwap and net_vwap_contradiction:
            vwap_severity = min(1.0, abs(price_vs_vwap) / max(vwap_gap_floor, 1e-6))
            vwap_penalty = reward_terms["soft_countertrend_vwap_penalty"] * penalty_scale * (
                0.55 + (0.45 * vwap_severity)
            )
            reward_adjustment -= vwap_penalty
            self.soft_penalty_vwap_count += 1

        if min_adx > 0.0 and adx < min_adx and not fallback_direction_ok and adx_structure_required:
            adx_deficit = min(1.0, (min_adx - adx) / max(min_adx, 1e-6))
            adx_penalty = reward_terms["soft_low_adx_penalty"] * penalty_scale * (0.8 + adx_deficit)
            reward_adjustment -= adx_penalty
            self.soft_penalty_adx_count += 1

        if (
            self.horizon != "scalp"
            and against_obv
            and reward_terms["soft_obv_divergence_penalty"] > 0.0
        ):
            obv_penalty = reward_terms["soft_obv_divergence_penalty"] * penalty_scale
            reward_adjustment -= obv_penalty
            self.soft_penalty_obv_count += 1

        aligned_with_ema = not against_ema
        aligned_with_vwap = not against_vwap
        aligned_with_obv = not against_obv
        strong_trend_context = adx >= max(trend_adx, min_adx)
        supportive_momentum = (
            (direction_sign > 0 and momentum >= momentum_floor)
            or (direction_sign < 0 and momentum <= -momentum_floor)
        )

        if aligned_with_ema and aligned_with_vwap:
            reward_adjustment += reward_terms["soft_trend_alignment_bonus"] * bonus_scale
            if strong_trend_context and supportive_momentum and aligned_with_obv:
                reward_adjustment += reward_terms["soft_strong_alignment_bonus"] * bonus_scale

        if reward_adjustment < 0.0:
            self.soft_entry_penalty_count += 1
            self.soft_entry_penalty_total += abs(reward_adjustment)
        elif reward_adjustment > 0.0:
            self.soft_entry_bonus_count += 1
            self.soft_entry_bonus_total += reward_adjustment

        return reward_adjustment

    def _apply_entry_filter(self, action: int, context: dict[str, float]) -> tuple[int, str | None]:
        """Filtre une entree directionnelle selon le profil horizon/famille.

        Args:
            action (int): Action demandee par le modele.
            context (dict[str, float]): Contexte courant du marche.

        Returns:
            tuple[int, str | None]: Action finale. Un veto convertit
                l'entree en ``HOLD`` et fournit la raison du veto.
        """
        veto_reason = self._evaluate_entry_veto_reason(
            action,
            context,
            self._get_active_entry_filter(),
            position_size=self.position_size,
            long_entries=self.long_entries,
            short_entries=self.short_entries,
            directional_policy=dict(self.position_mechanics_profile.get("directional_policy") or {}),
            include_directional_hard_veto=True,
        )
        if veto_reason is None:
            return action, None
        if self._should_soften_training_root_veto(
            action=action,
            reason=veto_reason,
            context=context,
            entry_filter=self._get_active_entry_filter(),
        ):
            return action, None

        if veto_reason == "adx":
            self.entry_blocked_adx += 1
            if action == BUY:
                self.blocked_buy_adx += 1
            else:
                self.blocked_sell_adx += 1
        elif veto_reason == "ema200":
            if action == BUY:
                self.ema200_blocked_buy += 1
            else:
                self.ema200_blocked_sell += 1
        elif veto_reason == "vwap":
            self.entry_blocked_vwap += 1
            if action == BUY:
                self.blocked_buy_vwap += 1
            else:
                self.blocked_sell_vwap += 1
        elif veto_reason == "obv":
            self.entry_blocked_obv += 1
            if action == BUY:
                self.blocked_buy_obv += 1
            else:
                self.blocked_sell_obv += 1
        elif veto_reason == "directional":
            if action == BUY:
                self.blocked_buy_directional += 1
            else:
                self.blocked_sell_directional += 1
        return HOLD, veto_reason

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

    def _get_unrealized_pnl_pct(self, price: float) -> float:
        """Calcule la valeur latente de la position en pourcentage du capital.

        Args:
            price (float): Prix de marche courant.

        Returns:
            float: PnL latent normalise en pourcentage du capital initial.
        """
        if self.position_size == 0:
            return 0.0
        trade_ret = self._get_unrealized_return(price)
        unrealized_trade = (trade_ret * abs(self.position_size)) - (
            abs(self.position_size) * self.commission_rate
        )
        return (unrealized_trade / self.spec.initial_balance) * 100.0

    def _update_runner_progress(self, price: float) -> None:
        """Met a jour le pic de profit atteint par le runner actif.

        Args:
            price (float): Prix de marche courant.
        """
        if not self.runner_active or self.position_size == 0:
            return
        current_profit_pct = max(0.0, self._get_unrealized_pnl_pct(price))
        self.runner_peak_profit_pct = max(self.runner_peak_profit_pct, current_profit_pct)

    def _update_position_peak_progress(self, price: float) -> None:
        """Met a jour le pic latent de profit de la position courante.

        Args:
            price (float): Prix de marche courant.
        """
        if self.position_size == 0:
            self.position_peak_return = 0.0
            self.peak_profit_age_steps = 0
            return

        current_trade_return = max(0.0, self._get_unrealized_return(price))
        if current_trade_return > (self.position_peak_return + 1e-8):
            self.position_peak_return = current_trade_return
            self.peak_profit_age_steps = 0
        elif self.position_peak_return > 0.0:
            self.peak_profit_age_steps += 1

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
        realized_pct = self._record_closed_trade(realized_trade)
        if direction > 0:
            self.net_realized_long_pct += realized_pct
        else:
            self.net_realized_short_pct += realized_pct

        full_close = close_size is None or realized_notional >= current_notional
        if full_close:
            if realized_trade < 0:
                self.nemesis_recent_losses += 1
                if str(self._episode_regime or "").strip().lower() == "range":
                    self.nemesis_trap_losses += 1
                self.nemesis_trap_rate = self.nemesis_trap_losses / max(self.nemesis_recent_losses, 1)
                self.nemesis_quarantine_active = 1.0 if self.nemesis_recent_losses >= 3 else 0.0

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
            self.slbe_profit_locked = False
            self.position_pyramids = 0
            self.position_had_slbe = False
            self._reset_exit_plan_state()
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
        hour_feat, day_feat = self._get_temporal_features(self.current_step)
        high_price = self.data[self.current_step, 1]
        low_price = self.data[self.current_step, 2]
        close_price = self.data[self.current_step, 3]
        vol = min((high_price - low_price) / max(close_price, 1e-8) * 100.0, 1.0) if close_price > 0 else 0.0

        extra = np.array([pos_state, pnl_pct, slbe_state, hour_feat, day_feat, vol], dtype=np.float32)
        nemesis_feats = np.array([
            self.nemesis_trap_rate,
            float(self.nemesis_recent_losses),
            self.nemesis_quarantine_active
        ], dtype=np.float32)
        return np.concatenate([base, extra, nemesis_feats])

    def get_legal_root_actions(self) -> list[int]:
        """Retourne les actions legales a la racine pour l'etat courant.

        Returns:
            list[int]: Actions structurellement possibles a la racine.
        """
        legal_actions = [HOLD, BUY, SELL]
        if self.position_size != 0:
            legal_actions.extend([SPLIT, CLOSE])
        return legal_actions

    def get_root_policy_actions(self) -> list[int]:
        """Retourne le masque racine structurel et metier pour le self-play.

        Returns:
            list[int]: Actions autorisees a la racine apres application
                des veto d'entree et du veto directionnel dur.
        """
        legal_actions = [HOLD]
        structural_actions = self.get_legal_root_actions()
        context = self._get_market_context()
        entry_filter = self._get_active_entry_filter()
        directional_policy = dict(self.position_mechanics_profile.get("directional_policy") or {})

        for action in structural_actions:
            if action not in (BUY, SELL):
                if action not in legal_actions:
                    legal_actions.append(action)
                continue
            self.root_mask_directional_candidates_total += 1
            veto_reason = self._evaluate_entry_veto_reason(
                action,
                context,
                entry_filter,
                position_size=self.position_size,
                long_entries=self.long_entries,
                short_entries=self.short_entries,
                directional_policy=directional_policy,
                include_directional_hard_veto=True,
            )
            if veto_reason is None:
                legal_actions.append(action)
                continue
            if self._should_soften_training_root_veto(
                action=action,
                reason=veto_reason,
                context=context,
                entry_filter=entry_filter,
            ):
                legal_actions.append(action)
                continue
            self._register_root_mask_block(action, veto_reason)

        return legal_actions

    @staticmethod
    def _build_observation_context(observation: np.ndarray) -> dict[str, float]:
        """Reconstruit un contexte de marche minimal depuis une observation MuZero.

        Args:
            observation (np.ndarray): Observation complete de forme ``[32]``.

        Returns:
            dict[str, float]: Contexte compatible avec les veto d'entree.

        Raises:
            ValueError: Si l'observation est trop courte.
        """
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation_array.size < 24:
            raise ValueError("Observation MuZero invalide: contexte de marche incomplet.")

        close_price = float(observation_array[3])
        ema_200 = float(observation_array[MARKET_COL_EMA_200]) if observation_array.size > MARKET_COL_EMA_200 else close_price
        vwap = float(observation_array[MARKET_COL_VWAP]) if observation_array.size > MARKET_COL_VWAP else close_price
        obv = float(observation_array[MARKET_COL_OBV]) if observation_array.size > MARKET_COL_OBV else 0.0
        momentum = float(observation_array[MARKET_COL_MOMENTUM]) if observation_array.size > MARKET_COL_MOMENTUM else 0.0
        adx = float(observation_array[MARKET_COL_ADX]) if observation_array.size > MARKET_COL_ADX else 0.0
        atr = float(observation_array[MARKET_COL_ATR]) if observation_array.size > MARKET_COL_ATR else 0.0
        obv_slope_proxy = momentum if abs(momentum) > 1e-8 else float(np.sign(obv))
        return {
            "close": close_price,
            "ema_200": ema_200,
            "ema_gap_pct": (close_price - ema_200) / max(abs(ema_200), 1e-8),
            "vwap": vwap,
            "price_vs_vwap": (close_price - vwap) / max(abs(vwap), 1e-8),
            "obv": obv,
            "obv_slope": obv_slope_proxy,
            "obv_divergence": float(np.sign(momentum) != np.sign(obv_slope_proxy)),
            "adx": adx,
            "atr": atr,
            "atr_pct": atr / max(abs(close_price), 1e-8),
            "bb_width_proxy": 0.0,
            "momentum": momentum,
        }

    @staticmethod
    def infer_legal_root_actions_from_observation(observation: np.ndarray) -> list[int]:
        """Reconstruit les actions legales racine a partir d'une observation.

        Args:
            observation (np.ndarray): Observation MuZero complete.

        Returns:
            list[int]: Actions autorisees au noeud racine.

        Raises:
            ValueError: Si l'observation ne contient pas l'etat de position.
        """
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation_array.size < OBS_EXTRA_FEATURE_COUNT:
            raise ValueError(
                "Observation MuZero invalide: etat de position indisponible."
            )

        legal_actions = [HOLD, BUY, SELL]
        if abs(float(observation_array[OBS_POSITION_STATE_INDEX])) > 1e-6:
            legal_actions.extend([SPLIT, CLOSE])
        return legal_actions

    @classmethod
    def infer_root_policy_actions_from_observation(
        cls,
        observation: np.ndarray,
        entry_filter: dict[str, float | bool | str] | None = None,
    ) -> list[int]:
        """Reconstruit un masque racine metier depuis une observation seule.

        Cette variante est utilisee en reanalyse et en inference live. Elle
        reconstruit les veto `EMA/VWAP/ADX/OBV` a partir du vecteur
        d'observation mais n'essaie pas de reproduire le veto directionnel
        dur, car l'historique `long_entries/short_entries` n'est pas present.

        Args:
            observation (np.ndarray): Observation MuZero complete.
            entry_filter (dict[str, float | bool | str] | None): Filtre actif
                a appliquer. Sans valeur, on conserve uniquement la legalite
                structurelle.

        Returns:
            list[int]: Actions autorisees a la racine.
        """
        legal_actions = cls.infer_legal_root_actions_from_observation(observation)
        if entry_filter is None:
            return legal_actions

        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        position_state = 0.0
        if observation_array.size >= OBS_EXTRA_FEATURE_COUNT:
            position_state = float(observation_array[OBS_POSITION_STATE_INDEX])
        context = cls._build_observation_context(observation_array)
        filtered_actions = [HOLD]
        for action in legal_actions:
            if action not in (BUY, SELL):
                if action not in filtered_actions:
                    filtered_actions.append(action)
                continue
            veto_reason = cls._evaluate_entry_veto_reason(
                action,
                context,
                dict(entry_filter or {}),
                position_size=position_state,
                long_entries=0,
                short_entries=0,
                directional_policy=None,
                include_directional_hard_veto=False,
            )
            if veto_reason is None:
                filtered_actions.append(action)
        return filtered_actions

    @staticmethod
    def _resolve_reward_policy_terms(reward_policy: dict[str, float]) -> dict[str, float]:
        """Normalise les cles V1 et V2 de la politique de recompense.

        Args:
            reward_policy (dict[str, float]): Politique brute issue du
                profil de mecanique.

        Returns:
            dict[str, float]: Parametres de recompense resolves.
        """
        realized_reward_multiplier = float(
            reward_policy.get(
                "realized_reward_multiplier",
                reward_policy.get("realized_pnl_multiplier", 1.0),
            )
            or 1.0
        )
        return {
            "realized_reward_multiplier": realized_reward_multiplier,
            "close_realized_multiplier": float(
                reward_policy.get(
                    "close_realized_bonus_multiplier",
                    reward_policy.get("close_realized_multiplier", realized_reward_multiplier),
                )
                or realized_reward_multiplier
            ),
            "split_realized_multiplier": float(
                reward_policy.get(
                    "split_realized_bonus_multiplier",
                    reward_policy.get("split_realized_multiplier", realized_reward_multiplier),
                )
                or realized_reward_multiplier
            ),
            "hold_drag_multiplier": float(
                reward_policy.get(
                    "hold_drag_penalty_multiplier",
                    reward_policy.get("hold_drag_multiplier", 0.0),
                )
                or 0.0
            ),
            "pyramid_reject_penalty": float(
                reward_policy.get("pyramid_failure_penalty", 0.1) or 0.1
            ),
            "pyramid_negative_exit_penalty": float(
                reward_policy.get("pyramid_negative_exit_penalty", 0.0) or 0.0
            ),
            "split_runner_profit_bonus": float(
                reward_policy.get("split_runner_profit_bonus", 0.45) or 0.45
            ),
            "split_early_zone_penalty": float(
                reward_policy.get("split_early_zone_penalty", 0.25) or 0.25
            ),
            "split_decorative_penalty": float(
                reward_policy.get("split_decorative_penalty", 0.15) or 0.15
            ),
            "pyramid_exit_capture_bonus": float(
                reward_policy.get("pyramid_exit_capture_bonus", 0.35) or 0.35
            ),
            "pyramid_bad_add_penalty": float(
                reward_policy.get("pyramid_bad_add_penalty", 0.35) or 0.35
            ),
            "runner_protected_exit_bonus": float(
                reward_policy.get("runner_protected_exit_bonus", 0.30) or 0.30
            ),
            "runner_hold_capture_bonus": float(
                reward_policy.get("runner_hold_capture_bonus", 0.10) or 0.10
            ),
            "split_zone_capture_bonus": float(
                reward_policy.get("split_zone_capture_bonus", 0.18) or 0.18
            ),
            "split_window_activation_bonus": float(
                reward_policy.get("split_window_activation_bonus", 0.14) or 0.14
            ),
            "runner_extension_capture_bonus": float(
                reward_policy.get("runner_extension_capture_bonus", 0.22) or 0.22
            ),
            "runner_split_activation_bonus": float(
                reward_policy.get("runner_split_activation_bonus", 0.18) or 0.18
            ),
            "runner_missed_extension_penalty": float(
                reward_policy.get("runner_missed_extension_penalty", 0.10) or 0.10
            ),
            "runner_trade_completion_bonus": float(
                reward_policy.get("runner_trade_completion_bonus", 0.40) or 0.40
            ),
            "runner_giveback_penalty": float(
                reward_policy.get("runner_giveback_penalty", 0.45) or 0.45
            ),
            "runner_giveback_ratio_penalty": float(
                reward_policy.get("runner_giveback_ratio_penalty", 0.35) or 0.35
            ),
            "runner_giveback_soft_penalty": float(
                reward_policy.get("runner_giveback_soft_penalty", 0.10) or 0.10
            ),
            "runner_giveback_hard_penalty": float(
                reward_policy.get("runner_giveback_hard_penalty", 0.22) or 0.22
            ),
            "runner_retained_profit_bonus": float(
                reward_policy.get("runner_retained_profit_bonus", 0.25) or 0.25
            ),
            "runner_hold_after_soft_tp_bonus": float(
                reward_policy.get("runner_hold_after_soft_tp_bonus", 0.12) or 0.12
            ),
            "runner_viable_but_closed_penalty": float(
                reward_policy.get("runner_viable_but_closed_penalty", 0.18) or 0.18
            ),
            "early_full_close_after_soft_tp_penalty": float(
                reward_policy.get("early_full_close_after_soft_tp_penalty", 0.22) or 0.22
            ),
            "pyramid_trade_completion_bonus": float(
                reward_policy.get("pyramid_trade_completion_bonus", 0.30) or 0.30
            ),
            "pyramid_stagnant_exit_penalty": float(
                reward_policy.get("pyramid_stagnant_exit_penalty", 0.25) or 0.25
            ),
            "pyramid_hold_capture_bonus": float(
                reward_policy.get("pyramid_hold_capture_bonus", 0.10) or 0.10
            ),
            "pyramid_window_activation_bonus": float(
                reward_policy.get("pyramid_window_activation_bonus", 0.12) or 0.12
            ),
            "pyramid_add_capture_bonus": float(
                reward_policy.get("pyramid_add_capture_bonus", 0.18) or 0.18
            ),
            "pyramid_missed_add_penalty": float(
                reward_policy.get("pyramid_missed_add_penalty", 0.12) or 0.12
            ),
            "missed_window_penalty": float(
                reward_policy.get("missed_window_penalty", 0.05) or 0.05
            ),
            "soft_countertrend_ema_penalty": float(
                reward_policy.get("soft_countertrend_ema_penalty", 0.0) or 0.0
            ),
            "soft_countertrend_vwap_penalty": float(
                reward_policy.get("soft_countertrend_vwap_penalty", 0.0) or 0.0
            ),
            "soft_low_adx_penalty": float(
                reward_policy.get("soft_low_adx_penalty", 0.0) or 0.0
            ),
            "soft_obv_divergence_penalty": float(
                reward_policy.get("soft_obv_divergence_penalty", 0.0) or 0.0
            ),
            "soft_trend_alignment_bonus": float(
                reward_policy.get("soft_trend_alignment_bonus", 0.0) or 0.0
            ),
            "soft_strong_alignment_bonus": float(
                reward_policy.get("soft_strong_alignment_bonus", 0.0) or 0.0
            ),
        }

    @staticmethod
    def _resolve_directional_policy_terms(directional_policy: dict[str, float]) -> dict[str, float]:
        """Normalise les cles de pilotage directionnel.

        Args:
            directional_policy (dict[str, float]): Politique brute issue du
                profil de mecanique.

        Returns:
            dict[str, float]: Parametres directionnels resolves.
        """
        return {
            "min_entry_share": float(directional_policy.get("min_entry_share", 0.0) or 0.0),
            "max_directional_imbalance": float(
                directional_policy.get(
                    "max_directional_imbalance",
                    directional_policy.get("max_imbalance", 1.0),
                )
                or 1.0
            ),
            "imbalance_penalty": float(directional_policy.get("imbalance_penalty", 0.0) or 0.0),
            "entry_penalty_scale": float(directional_policy.get("entry_penalty_scale", 0.35) or 0.35),
            "hard_veto_after_entries": float(directional_policy.get("hard_veto_after_entries", 4) or 4),
            "hard_veto_max_share": float(directional_policy.get("hard_veto_max_share", 0.80) or 0.80),
            "final_max_directional_imbalance": float(
                directional_policy.get(
                    "final_max_directional_imbalance",
                    directional_policy.get("max_directional_imbalance", 1.0),
                )
                or 1.0
            ),
            "rebalance_bonus": float(directional_policy.get("rebalance_bonus", 0.0) or 0.0),
        }

    def _compute_directional_entry_feedback(self, directional_policy: dict[str, float]) -> float:
        """Applique une penalite precoce quand l'episode derive trop d'un cote.

        Args:
            directional_policy (dict[str, float]): Politique directionnelle.

        Returns:
            float: Ajustement de reward negatif si le desequilibre devient
                excessif.
        """
        resolved_policy = self._resolve_directional_policy_terms(directional_policy)
        total_entries = self.long_entries + self.short_entries
        min_entry_share = resolved_policy["min_entry_share"]
        if total_entries < 4 or min_entry_share <= 0.0:
            return 0.0

        long_share = self.long_entries / total_entries
        short_share = self.short_entries / total_entries
        imbalance = abs(long_share - short_share)
        max_imbalance = resolved_policy["max_directional_imbalance"]
        if (
            long_share >= min_entry_share
            and short_share >= min_entry_share
            and imbalance <= max_imbalance
        ):
            return 0.0

        severity = max(
            0.0,
            min_entry_share - min(long_share, short_share),
            imbalance - max_imbalance,
        )
        penalty = (
            resolved_policy["imbalance_penalty"]
            * resolved_policy["entry_penalty_scale"]
            * (1.0 + min(1.0, severity * 5.0))
        )
        self.directional_imbalance_penalties += 1
        return -penalty

    def step(self, action: int):
        """Exécute un pas de trading.

        Args:
            action (int): Action discrète MuZero.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Sortie Gymnasium.
        """
        context = self._get_market_context()
        # Les ordres partent sur l'ouverture de la barre suivante pour eviter
        # un biais d'anticipation dans la simulation offline.
        next_step = min(self.current_step + 1, len(self.data) - 1)
        price = float(self.data[next_step, 0])
        reward = 0.0
        done = False
        realized_pnl = 0.0

        trade_notional = self.spec.trade_size
        pyramiding_policy = dict(self.position_mechanics_profile.get("pyramiding_policy") or {})
        split_policy = dict(self.position_mechanics_profile.get("split_policy") or {})
        slbe_policy = dict(self.position_mechanics_profile.get("slbe_policy") or {})
        close_policy = dict(self.position_mechanics_profile.get("close_policy") or {})
        exit_plan_policy = dict(self.position_mechanics_profile.get("exit_plan_policy") or {})
        hold_policy = dict(self.position_mechanics_profile.get("hold_policy") or {})
        activity_policy = dict(self.position_mechanics_profile.get("activity_policy") or {})
        directional_policy = dict(self.position_mechanics_profile.get("directional_policy") or {})
        reward_policy = dict(self.position_mechanics_profile.get("reward_policy") or {})
        reward_terms = self._resolve_reward_policy_terms(reward_policy)
        exit_reward_phase_scale = self._resolve_exit_reward_phase_scale()
        realized_reward_multiplier = reward_terms["realized_reward_multiplier"]
        close_realized_multiplier = reward_terms["close_realized_multiplier"]
        split_realized_multiplier = reward_terms["split_realized_multiplier"]
        hold_drag_multiplier = reward_terms["hold_drag_multiplier"]
        pyramid_reject_penalty = reward_terms["pyramid_reject_penalty"]
        pyramid_negative_exit_penalty = reward_terms["pyramid_negative_exit_penalty"]
        max_position = (1 + int(pyramiding_policy.get("max_additions", 1))) * trade_notional
        active_entry_filter = self._get_active_entry_filter()

        if not self.slbe_active and self.position_size != 0:
            unr = self._get_unrealized_return(price)
            activation_return = float(slbe_policy.get("activation_return", 0.005) or 0.005)
            if unr >= activation_return:
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                self.position_had_slbe = True
                self.slbe_profit_locked = False
                if self.runner_active:
                    self.runner_protected = True
                self.slbe_triggered += 1
                self.secured_count += 1
                reward += (
                    float(slbe_policy.get("bonus", self.slbe_bonus) or self.slbe_bonus)
                    * exit_reward_phase_scale
                )

        if self.slbe_active and self.position_size != 0:
            lock_profit_return = self._policy_float(slbe_policy, "lock_profit_return", 0.0075)
            lock_profit_buffer = self._policy_float(slbe_policy, "lock_profit_buffer", 0.0010)
            unr = self._get_unrealized_return(price)
            if not self.slbe_profit_locked and unr >= lock_profit_return:
                if self.position_size > 0:
                    self.slbe_price = max(
                        self.slbe_price,
                        self.avg_entry_price * (1.0 + lock_profit_buffer),
                    )
                else:
                    self.slbe_price = min(
                        self.slbe_price,
                        self.avg_entry_price * (1.0 - lock_profit_buffer),
                    )
                self.slbe_profit_locked = True
                if self.runner_active:
                    self.runner_protected = True
                self.slbe_lock_profit_count += 1

            hit = False
            if self.position_size > 0 and price <= self.slbe_price:
                hit = True
            elif self.position_size < 0 and price >= self.slbe_price:
                hit = True
            if hit:
                had_locked_profit = self.slbe_profit_locked
                runner_exit_context = self._capture_runner_exit_context()
                had_runner = bool(runner_exit_context.get("active"))
                realized_trade, _ = self._realize_position(price)
                realized_pnl += realized_trade
                _ = self._register_runner_exit_outcome(
                    realized_trade=realized_trade,
                    forced_exit=had_runner,
                    runner_was_active=had_runner,
                    runner_peak_profit_pct=float(runner_exit_context.get("peak_profit_pct") or 0.0),
                    runner_entry_profit_pct=float(runner_exit_context.get("entry_profit_pct") or 0.0),
                )
                reward += 1.0 + float(slbe_policy.get("exit_bonus", 0.0) or 0.0)
                if realized_trade > 0:
                    self.slbe_exit_bonus_count += 1
                    if had_locked_profit:
                        reward += (
                            min(
                                1.0,
                                float(slbe_policy.get("exit_bonus", 0.0) or 0.0) * 0.35,
                            )
                            * exit_reward_phase_scale
                        )

        if self.position_size != 0:
            self._update_position_peak_progress(price)
            self._update_runner_progress(price)
            self._mark_exit_plan_hits(price=price)

        forced_hard_stop = False
        if self.position_size != 0 and self._hard_stop_triggered(price):
            forced_hard_stop = True
            stop_price = float(self.hard_stop_price or price)
            runner_exit_context = self._capture_runner_exit_context()
            had_runner = bool(runner_exit_context.get("active"))
            realized_trade, trade_ret = self._realize_position(stop_price)
            realized_pnl += realized_trade
            self._register_runner_exit_outcome(
                realized_trade=realized_trade,
                forced_exit=had_runner,
                runner_was_active=had_runner,
                runner_peak_profit_pct=float(runner_exit_context.get("peak_profit_pct") or 0.0),
                runner_entry_profit_pct=float(runner_exit_context.get("entry_profit_pct") or 0.0),
            )
            self.hard_stop_exit_count += 1
            self.close_loser_count += 1
            reward += (
                min(
                    -0.10,
                    (realized_trade / self.spec.initial_balance) * 100.0 - 0.10,
                )
                * exit_reward_phase_scale
            )

        management_snapshot = self._build_position_management_snapshot(
            price=price,
            context=context,
            entry_filter=active_entry_filter,
            hold_policy=hold_policy,
            split_policy=split_policy,
            pyramiding_policy=pyramiding_policy,
            close_policy=close_policy,
            exit_plan_policy=exit_plan_policy,
            trade_notional=trade_notional,
        )
        self._register_position_management_opportunities(management_snapshot)
        position_before_action = float(self.position_size)
        split_tp_zone_opportunity = bool(
            management_snapshot.get("split_tp_zone_opportunity", False)
        )
        split_monetization_window = bool(
            management_snapshot.get("split_monetization_window", False)
        )
        runner_viable_window = bool(
            management_snapshot.get("runner_viable_window", False)
        )
        runner_extension_opportunity = bool(
            management_snapshot.get("runner_extension_opportunity", False)
        )
        runner_profit_hold_window = bool(
            management_snapshot.get("runner_profit_hold_window", False)
        )
        runner_hold_after_soft_tp = bool(
            management_snapshot.get("runner_hold_after_soft_tp", False)
        )
        pyramid_add_opportunity = bool(
            management_snapshot.get("pyramid_add_opportunity", False)
        )
        pyramid_monetization_window = bool(
            management_snapshot.get("pyramid_monetization_window", False)
        )
        if bool(management_snapshot.get("near_hard_stop", False)) and (
            split_monetization_window
            or runner_profit_hold_window
            or pyramid_monetization_window
        ):
            # Une fenetre offensive peut rester exploitable pres du stop, mais
            # elle doit couter un peu plus cher pour pousser le modele a
            # monetiser vite.
            reward -= 0.03 * exit_reward_phase_scale
        runner_extension_captured = False
        pyramid_add_captured = False

        requested_action = HOLD if forced_hard_stop else action
        if forced_hard_stop:
            action = HOLD
        if requested_action == BUY:
            self.requested_buy_actions += 1
        elif requested_action == SELL:
            self.requested_sell_actions += 1

        if self.position_size == 0 and action in [SPLIT, CLOSE]:
            action = HOLD

        action, veto_reason = self._apply_entry_filter(action, context)
        if veto_reason:
            # Une entree vetoee devient un HOLD reel. La penalite reste
            # moderee pour ne pas figer artificiellement une direction.
            self.entry_veto_to_hold += 1
            if requested_action == BUY:
                self.blocked_buy_entries += 1
            elif requested_action == SELL:
                self.blocked_sell_entries += 1
            reward -= 1.0

        final_action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"ACT_{action}"
        self.action_counts[final_action_name] = self.action_counts.get(final_action_name, 0) + 1
        self._record_action_context(action, context)

        if action == BUY:
            self._flush_hold_streak()
            self._current_hold_drag_streak = 0
            if self.position_size < 0:
                had_pyramids = self.position_pyramids > 0
                pyramid_count_before_close = self.position_pyramids
                pyramid_baseline_return = (
                    float(self._pyramid_baseline_return)
                    if self._pyramid_baseline_active
                    else None
                )
                had_locked_profit = self.slbe_profit_locked
                runner_exit_context = self._capture_runner_exit_context()
                runner_active_before_close = bool(runner_exit_context.get("active"))
                runner_protected_before_close = bool(runner_exit_context.get("protected"))
                runner_steps_since_split = runner_exit_context.get("steps_since_split")
                realized_trade, trade_ret = self._realize_position(price)
                realized_pnl += realized_trade
                split_trade_value_delta_pct = self._register_runner_exit_outcome(
                    realized_trade=realized_trade,
                    forced_exit=False,
                    runner_was_active=runner_active_before_close,
                    runner_peak_profit_pct=float(runner_exit_context.get("peak_profit_pct") or 0.0),
                    runner_entry_profit_pct=float(runner_exit_context.get("entry_profit_pct") or 0.0),
                )
                pyramid_trade_improvement_pct = self._register_pyramid_exit_outcome(
                    had_pyramids=had_pyramids,
                    realized_trade=realized_trade,
                    trade_ret=trade_ret,
                    close_policy=close_policy,
                    baseline_trade_return=pyramid_baseline_return,
                )
                reward += self._compute_close_management_reward(
                    realized_trade=realized_trade,
                    trade_ret=trade_ret,
                    snapshot=management_snapshot,
                    close_policy=close_policy,
                    pyramiding_policy=pyramiding_policy,
                    slbe_policy=slbe_policy,
                    reward_terms=reward_terms,
                    close_realized_multiplier=close_realized_multiplier,
                    had_pyramids=had_pyramids,
                    pyramid_count_before_close=pyramid_count_before_close,
                    had_locked_profit=had_locked_profit,
                    runner_active_before_close=runner_active_before_close,
                    runner_protected_before_close=runner_protected_before_close,
                    runner_steps_since_split=runner_steps_since_split,
                    split_trade_value_delta_pct=split_trade_value_delta_pct,
                    runner_retained_profit_pct=self._last_runner_retained_profit_pct,
                    runner_giveback_pct=self._last_runner_giveback_pct,
                    runner_retention_ratio=self._last_runner_retention_ratio,
                    runner_giveback_ratio=max(0.0, 1.0 - self._last_runner_retention_ratio),
                    pyramid_trade_improvement_pct=pyramid_trade_improvement_pct,
                    phase_scale=exit_reward_phase_scale,
                )
                if realized_trade < 0 and had_pyramids and pyramid_negative_exit_penalty > 0:
                    reward -= pyramid_negative_exit_penalty * exit_reward_phase_scale

            if self.position_size == 0:
                self.balance -= trade_notional * self.commission_rate
                self.position_size = trade_notional
                self.avg_entry_price = price
                self.split_count = 0
                self.position_pyramids = 0
                self.position_had_slbe = False
                self.slbe_profit_locked = False
                self._configure_exit_plan(
                    price=price,
                    context=context,
                    hold_policy=hold_policy,
                    close_policy=close_policy,
                    slbe_policy=slbe_policy,
                    exit_plan_policy=exit_plan_policy,
                    reset_progress=True,
                )
                reward += self._compute_soft_entry_quality_adjustment(
                    BUY,
                    context,
                    active_entry_filter,
                    reward_terms,
                )
                reward += self._compute_rebalance_bonus(BUY, directional_policy)
                self.long_entries += 1
                reward += self._compute_directional_entry_feedback(directional_policy)
            elif self.position_size < max_position:
                curr_pnl = self._price_return(self.avg_entry_price, price, 1.0)
                min_profit_to_add = float(pyramiding_policy.get("min_profit_to_add", 0.001) or 0.001)
                max_additions = int(pyramiding_policy.get("max_additions", 1) or 1)
                if curr_pnl > min_profit_to_add and self.position_pyramids < max_additions:
                    self.balance -= trade_notional * self.commission_rate
                    total_value = (self.position_size * self.avg_entry_price) + (trade_notional * price)
                    self.position_size += trade_notional
                    self.avg_entry_price = total_value / self.position_size
                    if not self._pyramid_baseline_active:
                        self._pyramid_baseline_active = True
                        self._pyramid_baseline_return = float(curr_pnl)
                    self.position_pyramids += 1
                    self.pyramids_opened += 1
                    if pyramid_monetization_window or bool(
                        management_snapshot.get("offensive_continuation_support", False)
                    ):
                        self.pyramid_good_add_count += 1
                    else:
                        self.pyramid_bad_add_count += 1
                    self._configure_exit_plan(
                        price=price,
                        context=context,
                        hold_policy=hold_policy,
                        close_policy=close_policy,
                        slbe_policy=slbe_policy,
                        exit_plan_policy=exit_plan_policy,
                        reset_progress=True,
                    )
                    reward += 0.35 * self._compute_soft_entry_quality_adjustment(
                        BUY,
                        context,
                        active_entry_filter,
                        reward_terms,
                    )
                    reward += (
                        float(pyramiding_policy.get("reward_bonus", 0.1) or 0.1)
                        * 0.25
                        * exit_reward_phase_scale
                    )
                    if bool(management_snapshot.get("strong_trend_support", False)):
                        reward += (
                            self._policy_float(
                                pyramiding_policy,
                                "strong_trend_reward_bonus",
                                0.20,
                            )
                            * 0.25
                            * exit_reward_phase_scale
                        )
                    elif bool(management_snapshot.get("offensive_continuation_support", False)):
                        reward += (
                            self._policy_float(
                                pyramiding_policy,
                                "strong_trend_reward_bonus",
                                0.20,
                            )
                            * 0.12
                            * exit_reward_phase_scale
                        )
                    if pyramid_add_opportunity:
                        self.pyramid_add_capture_count += 1
                        if pyramid_monetization_window:
                            self.pyramid_monetization_capture_count += 1
                        pyramid_add_captured = True
                        reward += reward_terms["pyramid_window_activation_bonus"] * exit_reward_phase_scale
                        reward += reward_terms["pyramid_add_capture_bonus"] * exit_reward_phase_scale
                    else:
                        reward -= reward_terms["pyramid_bad_add_penalty"] * exit_reward_phase_scale
                else:
                    self.pyramids_rejected += 1
                    reward -= pyramid_reject_penalty * exit_reward_phase_scale

        elif action == SELL:
            self._flush_hold_streak()
            self._current_hold_drag_streak = 0
            if self.position_size > 0:
                had_pyramids = self.position_pyramids > 0
                pyramid_count_before_close = self.position_pyramids
                pyramid_baseline_return = (
                    float(self._pyramid_baseline_return)
                    if self._pyramid_baseline_active
                    else None
                )
                had_locked_profit = self.slbe_profit_locked
                runner_exit_context = self._capture_runner_exit_context()
                runner_active_before_close = bool(runner_exit_context.get("active"))
                runner_protected_before_close = bool(runner_exit_context.get("protected"))
                runner_steps_since_split = runner_exit_context.get("steps_since_split")
                realized_trade, trade_ret = self._realize_position(price)
                realized_pnl += realized_trade
                split_trade_value_delta_pct = self._register_runner_exit_outcome(
                    realized_trade=realized_trade,
                    forced_exit=False,
                    runner_was_active=runner_active_before_close,
                    runner_peak_profit_pct=float(runner_exit_context.get("peak_profit_pct") or 0.0),
                    runner_entry_profit_pct=float(runner_exit_context.get("entry_profit_pct") or 0.0),
                )
                pyramid_trade_improvement_pct = self._register_pyramid_exit_outcome(
                    had_pyramids=had_pyramids,
                    realized_trade=realized_trade,
                    trade_ret=trade_ret,
                    close_policy=close_policy,
                    baseline_trade_return=pyramid_baseline_return,
                )
                reward += self._compute_close_management_reward(
                    realized_trade=realized_trade,
                    trade_ret=trade_ret,
                    snapshot=management_snapshot,
                    close_policy=close_policy,
                    pyramiding_policy=pyramiding_policy,
                    slbe_policy=slbe_policy,
                    reward_terms=reward_terms,
                    close_realized_multiplier=close_realized_multiplier,
                    had_pyramids=had_pyramids,
                    pyramid_count_before_close=pyramid_count_before_close,
                    had_locked_profit=had_locked_profit,
                    runner_active_before_close=runner_active_before_close,
                    runner_protected_before_close=runner_protected_before_close,
                    runner_steps_since_split=runner_steps_since_split,
                    split_trade_value_delta_pct=split_trade_value_delta_pct,
                    runner_retained_profit_pct=self._last_runner_retained_profit_pct,
                    runner_giveback_pct=self._last_runner_giveback_pct,
                    runner_retention_ratio=self._last_runner_retention_ratio,
                    runner_giveback_ratio=max(0.0, 1.0 - self._last_runner_retention_ratio),
                    pyramid_trade_improvement_pct=pyramid_trade_improvement_pct,
                    phase_scale=exit_reward_phase_scale,
                )
                if realized_trade < 0 and had_pyramids and pyramid_negative_exit_penalty > 0:
                    reward -= pyramid_negative_exit_penalty * exit_reward_phase_scale

            if self.position_size == 0:
                self.balance -= trade_notional * self.commission_rate
                self.position_size = -trade_notional
                self.avg_entry_price = price
                self.split_count = 0
                self.position_pyramids = 0
                self.position_had_slbe = False
                self.slbe_profit_locked = False
                self._configure_exit_plan(
                    price=price,
                    context=context,
                    hold_policy=hold_policy,
                    close_policy=close_policy,
                    slbe_policy=slbe_policy,
                    exit_plan_policy=exit_plan_policy,
                    reset_progress=True,
                )
                reward += self._compute_soft_entry_quality_adjustment(
                    SELL,
                    context,
                    active_entry_filter,
                    reward_terms,
                )
                reward += self._compute_rebalance_bonus(SELL, directional_policy)
                self.short_entries += 1
                reward += self._compute_directional_entry_feedback(directional_policy)
            elif self.position_size > -max_position:
                curr_pnl = self._price_return(self.avg_entry_price, price, -1.0)
                min_profit_to_add = float(pyramiding_policy.get("min_profit_to_add", 0.001) or 0.001)
                max_additions = int(pyramiding_policy.get("max_additions", 1) or 1)
                if curr_pnl > min_profit_to_add and self.position_pyramids < max_additions:
                    self.balance -= trade_notional * self.commission_rate
                    total_value = (abs(self.position_size) * self.avg_entry_price) + (trade_notional * price)
                    self.position_size -= trade_notional
                    self.avg_entry_price = total_value / abs(self.position_size)
                    if not self._pyramid_baseline_active:
                        self._pyramid_baseline_active = True
                        self._pyramid_baseline_return = float(curr_pnl)
                    self.position_pyramids += 1
                    self.pyramids_opened += 1
                    if pyramid_monetization_window or bool(
                        management_snapshot.get("offensive_continuation_support", False)
                    ):
                        self.pyramid_good_add_count += 1
                    else:
                        self.pyramid_bad_add_count += 1
                    self._configure_exit_plan(
                        price=price,
                        context=context,
                        hold_policy=hold_policy,
                        close_policy=close_policy,
                        slbe_policy=slbe_policy,
                        exit_plan_policy=exit_plan_policy,
                        reset_progress=True,
                    )
                    reward += 0.35 * self._compute_soft_entry_quality_adjustment(
                        SELL,
                        context,
                        active_entry_filter,
                        reward_terms,
                    )
                    reward += (
                        float(pyramiding_policy.get("reward_bonus", 0.1) or 0.1)
                        * 0.25
                        * exit_reward_phase_scale
                    )
                    if bool(management_snapshot.get("strong_trend_support", False)):
                        reward += (
                            self._policy_float(
                                pyramiding_policy,
                                "strong_trend_reward_bonus",
                                0.20,
                            )
                            * 0.25
                            * exit_reward_phase_scale
                        )
                    elif bool(management_snapshot.get("offensive_continuation_support", False)):
                        reward += (
                            self._policy_float(
                                pyramiding_policy,
                                "strong_trend_reward_bonus",
                                0.20,
                            )
                            * 0.12
                            * exit_reward_phase_scale
                        )
                    if pyramid_add_opportunity:
                        self.pyramid_add_capture_count += 1
                        if pyramid_monetization_window:
                            self.pyramid_monetization_capture_count += 1
                        pyramid_add_captured = True
                        reward += reward_terms["pyramid_window_activation_bonus"] * exit_reward_phase_scale
                        reward += reward_terms["pyramid_add_capture_bonus"] * exit_reward_phase_scale
                    else:
                        reward -= reward_terms["pyramid_bad_add_penalty"] * exit_reward_phase_scale
                else:
                    self.pyramids_rejected += 1
                    reward -= pyramid_reject_penalty * exit_reward_phase_scale

        elif action == SPLIT and abs(self.position_size) > 0:
            self._flush_hold_streak()
            self._current_hold_drag_streak = 0
            max_splits = int(split_policy.get("max_splits", 3) or 3)
            min_trade_return = float(split_policy.get("min_trade_return", 0.01) or 0.01)
            min_realized_pct = float(split_policy.get("min_realized_pct", 0.0) or 0.0)
            soft_partial_value_floor = self._policy_float(
                split_policy,
                "soft_partial_value_floor",
                0.010,
            )
            current_trade_return = self._get_unrealized_return(price)
            runner_protected = bool(self.slbe_active)
            split_performed = False
            split_realized_trade = 0.0
            split_realized_pct = 0.0
            split_useful = False
            split_neutral = False
            split_destructive = False
            split_early = False
            in_tp_zone = bool(
                management_snapshot.get("soft_tp_hit", False)
                or management_snapshot.get("full_tp_hit", False)
                or current_trade_return >= float(management_snapshot.get("tp_like_threshold", 0.0) or 0.0)
            )
            if self.split_count < max_splits and current_trade_return >= min_trade_return:
                realized_trade, trade_ret = self._realize_position(price, abs(self.position_size) * 0.5)
                realized_pnl += realized_trade
                split_realized_trade = realized_trade
                split_realized_pct = (realized_trade / self.spec.initial_balance) * 100.0
                self.split_count += 1
                self.split_executed += 1
                split_performed = True
                if split_realized_pct >= min_realized_pct and in_tp_zone and realized_trade > 0.0:
                    reward += (
                        self.quality_mult * 0.35 * exit_reward_phase_scale
                    ) + (split_realized_pct * split_realized_multiplier * 0.35)
                    self.split_profitable_count += 1
                    self.realized_split_bonus_count += 1
                    split_useful = True
                    if split_monetization_window:
                        self.split_monetization_capture_count += 1
                        reward += reward_terms["split_window_activation_bonus"] * exit_reward_phase_scale
                    if split_tp_zone_opportunity or split_monetization_window:
                        reward += reward_terms["split_zone_capture_bonus"] * exit_reward_phase_scale
                    if runner_viable_window:
                        reward += (
                            reward_terms["runner_split_activation_bonus"] * exit_reward_phase_scale
                        )
                elif realized_trade > 0:
                    if split_realized_pct >= soft_partial_value_floor:
                        reward += (
                            max(0.05, split_realized_pct * split_realized_multiplier * 0.35)
                            * exit_reward_phase_scale
                        )
                        split_neutral = True
                    else:
                        reward += 0.02 * exit_reward_phase_scale
                        split_neutral = True
                    if split_monetization_window:
                        self.split_monetization_capture_count += 1
                        reward += reward_terms["split_window_activation_bonus"] * exit_reward_phase_scale
                else:
                    self.split_rejected_no_value += 1
                    split_destructive = True
                    reward -= (
                        float(split_policy.get("failure_penalty", 0.25) or 0.25)
                        * exit_reward_phase_scale
                    )
            else:
                self.split_rejected += 1
                self.split_rejected_no_value += 1
                split_destructive = True
                reward -= (
                    float(split_policy.get("failure_penalty", 0.25) or 0.25)
                    * exit_reward_phase_scale
                )

            if (
                split_performed
                and self.position_size != 0
                and not self.slbe_active
                and bool(split_policy.get("slbe_after_split", True))
            ):
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                self.position_had_slbe = True
                self.slbe_profit_locked = False
                self.slbe_triggered += 1
                reward += 0.50 * exit_reward_phase_scale
                runner_protected = True
                self.runner_protected = True

            if (
                split_realized_trade > 0.0
                and runner_protected
                and in_tp_zone
                and bool(split_policy.get("slbe_after_split", True))
            ):
                reward += (
                    self._policy_float(split_policy, "post_split_slbe_bonus", 0.60)
                    * exit_reward_phase_scale
                )
                if bool(
                    management_snapshot.get("soft_tp_hit", False)
                    or management_snapshot.get("full_tp_hit", False)
                ):
                    self._activate_runner_extension(
                        price=price,
                        context=context,
                        exit_plan_policy=exit_plan_policy,
                    )
                    reward += 0.20 * exit_reward_phase_scale

            if split_performed and self.position_size != 0:
                self._mark_runner_after_split(
                    price=price,
                    runner_protected=runner_protected,
                    context=context,
                    exit_plan_policy=exit_plan_policy,
                )

            if split_performed:
                split_early = not in_tp_zone
                if split_neutral:
                    self.split_decorative_count += 1
                    reward -= reward_terms["split_decorative_penalty"] * exit_reward_phase_scale
                    if split_early:
                        self.split_early_count += 1
                        reward -= reward_terms["split_early_zone_penalty"] * exit_reward_phase_scale
                else:
                    split_destructive = True

                if split_destructive:
                    self.split_runner_failed_count += 1
                    if split_early:
                        self.split_early_count += 1
                        reward -= reward_terms["split_early_zone_penalty"] * exit_reward_phase_scale
                    else:
                        self.split_decorative_count += 1
                        reward -= reward_terms["split_decorative_penalty"] * exit_reward_phase_scale

        elif action == CLOSE and abs(self.position_size) > 0:
            self._flush_hold_streak()
            self._current_hold_drag_streak = 0
            had_pyramids = self.position_pyramids > 0
            pyramid_count_before_close = self.position_pyramids
            pyramid_baseline_return = (
                float(self._pyramid_baseline_return)
                if self._pyramid_baseline_active
                else None
            )
            had_locked_profit = self.slbe_profit_locked
            runner_exit_context = self._capture_runner_exit_context()
            runner_active_before_close = bool(runner_exit_context.get("active"))
            runner_protected_before_close = bool(runner_exit_context.get("protected"))
            runner_steps_since_split = runner_exit_context.get("steps_since_split")
            realized_trade, trade_ret = self._realize_position(price)
            realized_pnl += realized_trade
            split_trade_value_delta_pct = self._register_runner_exit_outcome(
                realized_trade=realized_trade,
                forced_exit=False,
                runner_was_active=runner_active_before_close,
                runner_peak_profit_pct=float(runner_exit_context.get("peak_profit_pct") or 0.0),
                runner_entry_profit_pct=float(runner_exit_context.get("entry_profit_pct") or 0.0),
            )
            pyramid_trade_improvement_pct = self._register_pyramid_exit_outcome(
                had_pyramids=had_pyramids,
                realized_trade=realized_trade,
                trade_ret=trade_ret,
                close_policy=close_policy,
                baseline_trade_return=pyramid_baseline_return,
            )
            reward += self._compute_close_management_reward(
                realized_trade=realized_trade,
                trade_ret=trade_ret,
                snapshot=management_snapshot,
                close_policy=close_policy,
                pyramiding_policy=pyramiding_policy,
                slbe_policy=slbe_policy,
                reward_terms=reward_terms,
                close_realized_multiplier=close_realized_multiplier,
                had_pyramids=had_pyramids,
                pyramid_count_before_close=pyramid_count_before_close,
                had_locked_profit=had_locked_profit,
                runner_active_before_close=runner_active_before_close,
                runner_protected_before_close=runner_protected_before_close,
                runner_steps_since_split=runner_steps_since_split,
                split_trade_value_delta_pct=split_trade_value_delta_pct,
                runner_retained_profit_pct=self._last_runner_retained_profit_pct,
                runner_giveback_pct=self._last_runner_giveback_pct,
                runner_retention_ratio=self._last_runner_retention_ratio,
                runner_giveback_ratio=max(0.0, 1.0 - self._last_runner_retention_ratio),
                pyramid_trade_improvement_pct=pyramid_trade_improvement_pct,
                phase_scale=exit_reward_phase_scale,
            )
            if realized_trade < 0 and had_pyramids and pyramid_negative_exit_penalty > 0:
                reward -= pyramid_negative_exit_penalty * exit_reward_phase_scale

        else:
            self._current_hold_streak += 1
            stale_after = int(hold_policy.get("stale_penalty_after_steps", 100) or 100)
            stale_penalty = float(hold_policy.get("stale_penalty", 1.0) or 1.0)
            trend_penalty = float(hold_policy.get("trend_penalty", 0.0) or 0.0)
            range_penalty = float(hold_policy.get("range_penalty", 0.0) or 0.0)
            trend_adx = float(
                active_entry_filter.get(
                    "trend_adx",
                    active_entry_filter.get("min_adx", 20.0),
                )
                or 20.0
            )
            if self.steps_since_last_trade > stale_after:
                reward -= stale_penalty
            if context["adx"] >= trend_adx:
                reward -= trend_penalty
            else:
                reward -= range_penalty
            if runner_profit_hold_window:
                trade_ret = max(0.0, float(management_snapshot.get("trade_ret", 0.0) or 0.0))
                peak_trade_return = max(
                    trade_ret,
                    float(management_snapshot.get("peak_trade_return", 0.0) or 0.0),
                )
                peak_giveback_ratio = max(
                    0.0,
                    float(management_snapshot.get("position_peak_giveback_ratio", 0.0) or 0.0),
                )
                retained_ratio = (
                    trade_ret / max(peak_trade_return, 1e-6)
                    if peak_trade_return > 0.0
                    else 0.0
                )
                self.runner_profit_hold_capture_count += 1
                self.runner_retained_profit_score_total += retained_ratio
                reward += min(
                    reward_terms["runner_hold_capture_bonus"],
                    max(0.03, retained_ratio * reward_terms["runner_hold_capture_bonus"]),
                ) * exit_reward_phase_scale
                if runner_hold_after_soft_tp:
                    reward += min(
                        reward_terms["runner_hold_after_soft_tp_bonus"],
                        max(0.03, retained_ratio * reward_terms["runner_hold_after_soft_tp_bonus"]),
                    ) * exit_reward_phase_scale
                if peak_giveback_ratio > 0.75:
                    reward -= reward_terms["runner_giveback_hard_penalty"] * exit_reward_phase_scale
                elif peak_giveback_ratio > 0.55:
                    reward -= reward_terms["runner_giveback_soft_penalty"] * exit_reward_phase_scale
            if runner_extension_opportunity:
                self.runner_extension_capture_count += 1
                runner_extension_captured = True
                reward += reward_terms["runner_extension_capture_bonus"] * exit_reward_phase_scale
            if (
                self.position_pyramids > 0
                and pyramid_monetization_window
            ):
                trade_ret = max(0.0, float(management_snapshot.get("trade_ret", 0.0) or 0.0))
                offensive_profit_floor = max(
                    1e-6,
                    float(management_snapshot.get("offensive_profit_floor", 0.0) or 0.0),
                )
                if trade_ret >= offensive_profit_floor:
                    reward += min(
                        reward_terms["pyramid_hold_capture_bonus"],
                        max(0.02, (trade_ret - offensive_profit_floor) * 8.0),
                    ) * exit_reward_phase_scale
            if (
                bool(management_snapshot.get("soft_tp_hit", False))
                and not bool(management_snapshot.get("offensive_continuation_support", False))
                and not runner_viable_window
                and not runner_profit_hold_window
            ):
                reward -= 0.20 * exit_reward_phase_scale
                if bool(management_snapshot.get("reversal_context", False)):
                    self.tp_like_missed_count += 1
            if bool(management_snapshot.get("full_tp_hit", False)) and not bool(
                management_snapshot.get("offensive_continuation_support", False)
            ) and not runner_profit_hold_window:
                reward -= 0.30 * exit_reward_phase_scale
            if bool(management_snapshot.get("profit_peak_reached", False)):
                peak_giveback_ratio = max(
                    0.0,
                    float(management_snapshot.get("position_peak_giveback_ratio", 0.0) or 0.0),
                )
                if peak_giveback_ratio > 0.75:
                    reward -= (
                        reward_terms["runner_giveback_hard_penalty"]
                        * min(1.0, 0.50 + ((peak_giveback_ratio - 0.75) / 0.25))
                        * exit_reward_phase_scale
                    )
                elif peak_giveback_ratio > 0.55:
                    reward -= reward_terms["runner_giveback_soft_penalty"] * exit_reward_phase_scale
            if bool(management_snapshot.get("time_stop_expired", False)) and (
                not bool(management_snapshot.get("offensive_continuation_support", False))
                or bool(management_snapshot.get("time_stop_grace_expired", False))
            ):
                reward -= 0.25 * exit_reward_phase_scale
            if (
                self.runner_active
                and bool(management_snapshot.get("reversal_context", False))
                and not bool(self.runner_protected)
            ):
                reward -= reward_terms["split_decorative_penalty"] * 0.5 * exit_reward_phase_scale
            if bool(management_snapshot.get("hold_drag_opportunity", False)):
                drag_grace_steps = self._policy_int(hold_policy, "drag_grace_steps", 3)
                self._current_hold_drag_streak += 1
                profitable_return = max(
                    0.0,
                    float(management_snapshot.get("trade_ret", 0.0) or 0.0),
                )
                if profitable_return >= self._policy_float(
                    hold_policy,
                    "drag_profit_floor",
                    0.0040,
                ):
                    if self._current_hold_drag_streak > drag_grace_steps:
                        reward -= (
                            self._compute_hold_drag_penalty(
                                profitable_return,
                                hold_policy,
                                hold_drag_multiplier,
                            )
                            * exit_reward_phase_scale
                        )
                        self.hold_drag_penalized_count += 1
                        self.hold_under_trend_penalty_count += 1
                    if profitable_return >= float(
                        management_snapshot.get("tp_like_threshold", 0.0) or 0.0
                    ):
                        self.tp_like_missed_count += 1
                else:
                    self._current_hold_drag_streak = 0
            else:
                self._current_hold_drag_streak = 0

        split_window_captured = bool(
            split_monetization_window and action == SPLIT and split_performed and split_realized_trade > 0.0
        )
        runner_window_captured = bool(
            (runner_profit_hold_window and action == HOLD)
            or (
                runner_viable_window
                and action == SPLIT
                and split_performed
                and split_realized_trade > 0.0
            )
        )
        pyramid_window_captured = bool(
            pyramid_monetization_window
            and (
                (position_before_action > 0.0 and action == BUY)
                or (position_before_action < 0.0 and action == SELL)
            )
            and pyramid_add_captured
        )

        self._split_window_wait_steps, split_window_missed = self._update_offensive_window_wait_steps(
            active=split_monetization_window,
            captured=split_window_captured,
            current_wait_steps=self._split_window_wait_steps,
        )
        self._runner_window_wait_steps, runner_window_missed = self._update_offensive_window_wait_steps(
            active=(runner_profit_hold_window or runner_viable_window),
            captured=runner_window_captured,
            current_wait_steps=self._runner_window_wait_steps,
        )
        self._pyramid_window_wait_steps, pyramid_window_missed = self._update_offensive_window_wait_steps(
            active=pyramid_monetization_window,
            captured=pyramid_window_captured,
            current_wait_steps=self._pyramid_window_wait_steps,
        )

        if split_window_missed:
            self.split_missed_window_count += 1
            reward -= reward_terms["missed_window_penalty"] * exit_reward_phase_scale
        if runner_window_missed:
            self.runner_missed_extension_count += 1
            reward -= reward_terms["missed_window_penalty"] * exit_reward_phase_scale
        if pyramid_window_missed:
            self.pyramid_missed_add_count += 1
            reward -= reward_terms["missed_window_penalty"] * exit_reward_phase_scale

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
            runner_exit_context = self._capture_runner_exit_context()
            had_runner = bool(runner_exit_context.get("active"))
            pyramid_baseline_return = (
                float(self._pyramid_baseline_return)
                if self._pyramid_baseline_active
                else None
            )
            final_realized, final_trade_ret = self._realize_position(price)
            realized_pnl += final_realized
            _ = self._register_runner_exit_outcome(
                realized_trade=final_realized,
                forced_exit=had_runner,
                runner_was_active=had_runner,
                runner_peak_profit_pct=float(runner_exit_context.get("peak_profit_pct") or 0.0),
                runner_entry_profit_pct=float(runner_exit_context.get("entry_profit_pct") or 0.0),
            )
            _ = self._register_pyramid_exit_outcome(
                had_pyramids=had_pyramids,
                realized_trade=final_realized,
                trade_ret=final_trade_ret,
                close_policy=close_policy,
                baseline_trade_return=pyramid_baseline_return,
            )
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

            resolved_directional_policy = self._resolve_directional_policy_terms(directional_policy)
            min_entry_share = resolved_directional_policy["min_entry_share"]
            max_imbalance = resolved_directional_policy["final_max_directional_imbalance"]
            imbalance_penalty = resolved_directional_policy["imbalance_penalty"]
            if total_entries > 0:
                long_share = self.long_entries / total_entries
                short_share = self.short_entries / total_entries
                imbalance = abs(long_share - short_share)
                if long_share < min_entry_share or short_share < min_entry_share or imbalance > max_imbalance:
                    reward -= imbalance_penalty
                    self.directional_imbalance_penalties += 1

        executed_day_index = min(next_step, max(len(self.day_labels) - 1, 0))
        reward += self._update_daily_tracking(
            str(self.day_labels[executed_day_index]) if len(self.day_labels) else self._active_day_label,
            equity,
            finalize=done,
        )
        self.equity_curve.append(equity)
        root_mask_block_total = self.root_mask_blocked_buy_total + self.root_mask_blocked_sell_total
        root_mask_ema200_share = (
            (self.root_mask_blocked_buy_ema200 + self.root_mask_blocked_sell_ema200)
            / max(root_mask_block_total, 1)
        )
        root_mask_vwap_share = (
            (self.root_mask_blocked_buy_vwap + self.root_mask_blocked_sell_vwap)
            / max(root_mask_block_total, 1)
        )
        root_mask_adx_share = (
            (self.root_mask_blocked_buy_adx + self.root_mask_blocked_sell_adx)
            / max(root_mask_block_total, 1)
        )
        root_mask_directional_share = (
            (self.root_mask_blocked_buy_directional + self.root_mask_blocked_sell_directional)
            / max(root_mask_block_total, 1)
        )
        info = {
            "balance": self.balance,
            "equity": equity,
            "step": self.current_step,
            "realized_pnl": realized_pnl,
            "drawdown": drawdown,
            "daily_best_return_pct": (
                self.best_day_net_return_pct if self.best_day_net_return_pct != float("-inf") else 0.0
            ),
            "days_above_10pct": self.days_above_10pct,
            "slbe_active": self.slbe_active,
            "total_trades": self.total_trades,
            "win_rate": self.total_profitable / max(self.total_trades, 1),
            "requested_action": ACTION_NAMES[requested_action] if 0 <= requested_action < len(ACTION_NAMES) else f"ACT_{requested_action}",
            "final_action": final_action_name,
            "veto_reason": veto_reason,
            "root_mask_rate": (
                root_mask_block_total
                / max(self.root_mask_directional_candidates_total, 1)
            ),
            "root_mask_ema200_share": root_mask_ema200_share,
            "root_mask_vwap_share": root_mask_vwap_share,
            "root_mask_adx_share": root_mask_adx_share,
            "root_mask_directional_share": root_mask_directional_share,
            "post_veto_to_hold_rate": (
                self.entry_veto_to_hold
                / max(self.requested_buy_actions + self.requested_sell_actions, 1)
            ),
            "soft_entry_penalty_rate": (
                self.soft_entry_penalty_count
                / max(self.long_entries + self.short_entries + self.pyramids_opened, 1)
            ),
            "soft_entry_bonus_rate": (
                self.soft_entry_bonus_count
                / max(self.long_entries + self.short_entries + self.pyramids_opened, 1)
            ),
            "soft_penalty_net": self.soft_entry_penalty_total - self.soft_entry_bonus_total,
            "soft_penalty_to_bonus_ratio": (
                self.soft_entry_penalty_total / max(self.soft_entry_bonus_total, 1e-6)
                if (self.soft_entry_penalty_total > 0.0 or self.soft_entry_bonus_total > 0.0)
                else 0.0
            ),
            "soft_penalty_ema_rate": (
                self.soft_penalty_ema200_count
                / max(self.long_entries + self.short_entries + self.pyramids_opened, 1)
            ),
            "soft_penalty_vwap_rate": (
                self.soft_penalty_vwap_count
                / max(self.long_entries + self.short_entries + self.pyramids_opened, 1)
            ),
            "soft_penalty_adx_rate": (
                self.soft_penalty_adx_count
                / max(self.long_entries + self.short_entries + self.pyramids_opened, 1)
            ),
            "soft_penalty_obv_rate": (
                self.soft_penalty_obv_count
                / max(self.long_entries + self.short_entries + self.pyramids_opened, 1)
            ),
            "hold_drag_score": (
                self.hold_drag_penalized_count / max(self.hold_drag_opportunity_count, 1)
            ),
            "split_efficiency": (
                self.split_profitable_count / max(self.split_executed, 1)
                if self.split_executed > 0
                else 0.0
            ),
            "split_runner_capture_rate": (
                self.split_runner_profitable_count / max(self.split_executed, 1)
                if self.split_executed > 0
                else 0.0
            ),
            "split_zone_capture_rate": (
                self.split_profitable_count / max(self.split_tp_zone_opportunity_count, 1)
                if self.split_tp_zone_opportunity_count > 0
                else 0.0
            ),
            "split_monetization_capture_rate": (
                self.split_monetization_capture_count / max(self.split_monetization_window_count, 1)
                if self.split_monetization_window_count > 0
                else 0.0
            ),
            "pyramid_efficiency": (
                self.pyramid_profitable_count / max(self.pyramids_opened, 1)
                if self.pyramids_opened > 0
                else 0.0
            ),
            "pyramid_entry_quality_score": (
                self.pyramid_good_add_count / max(self.pyramids_opened, 1)
                if self.pyramids_opened > 0
                else 0.0
            ),
            "pyramid_exit_capture_rate": (
                self.pyramid_profitable_exit_count / max(self.pyramids_opened, 1)
                if self.pyramids_opened > 0
                else 0.0
            ),
            "pyramid_add_capture_rate": (
                self.pyramid_add_capture_count / max(self.pyramid_add_opportunity_count, 1)
                if self.pyramid_add_opportunity_count > 0
                else 0.0
            ),
            "pyramid_monetization_capture_rate": (
                self.pyramid_monetization_capture_count / max(self.pyramid_monetization_window_count, 1)
                if self.pyramid_monetization_window_count > 0
                else 0.0
            ),
            "slbe_capture_rate": (
                self.slbe_profitable_exits / max(self.slbe_triggered, 1)
                if self.slbe_triggered > 0
                else 0.0
            ),
            "close_quality_score": (
                self.close_winner_count / max(self.close_winner_count + self.close_loser_count, 1)
                if (self.close_winner_count + self.close_loser_count) > 0
                else 0.0
            ),
            "hold_drag_opportunity_count": self.hold_drag_opportunity_count,
            "hold_drag_penalized_count": self.hold_drag_penalized_count,
            "split_opportunity_count": self.split_opportunity_count,
            "split_tp_zone_opportunity_count": self.split_tp_zone_opportunity_count,
            "split_runner_profitable_count": self.split_runner_profitable_count,
            "split_runner_failed_count": self.split_runner_failed_count,
            "split_early_count": self.split_early_count,
            "split_decorative_count": self.split_decorative_count,
            "split_trade_value_delta": self.split_trade_value_delta,
            "split_improved_total_trade_count": self.split_improved_total_trade_count,
            "split_missed_window_count": self.split_missed_window_count,
            "pyramid_opportunity_count": self.pyramid_opportunity_count,
            "pyramid_add_opportunity_count": self.pyramid_add_opportunity_count,
            "pyramid_add_capture_count": self.pyramid_add_capture_count,
            "pyramid_missed_add_count": self.pyramid_missed_add_count,
            "pyramid_good_add_count": self.pyramid_good_add_count,
            "pyramid_bad_add_count": self.pyramid_bad_add_count,
            "pyramid_profitable_exit_count": self.pyramid_profitable_exit_count,
            "pyramid_total_trade_improvement_pct": self.pyramid_total_trade_improvement_pct,
            "pyramid_failed_to_improve_count": self.pyramid_failed_to_improve_count,
            "slbe_lock_profit_count": self.slbe_lock_profit_count,
            "tp_like_missed_count": self.tp_like_missed_count,
            "defensive_close_count": self.defensive_close_count,
            "early_close_noise_count": self.early_close_noise_count,
            "hard_stop_exit_count": self.hard_stop_exit_count,
            "soft_tp_hit_count": self.soft_tp_hit_count,
            "full_tp_hit_count": self.full_tp_hit_count,
            "time_stop_trigger_count": self.time_stop_trigger_count,
            "runner_extension_count": self.runner_extension_count,
            "runner_extension_opportunity_count": self.runner_extension_opportunity_count,
            "runner_extension_capture_rate": (
                self.runner_extension_capture_count / max(self.runner_extension_opportunity_count, 1)
                if self.runner_extension_opportunity_count > 0
                else 0.0
            ),
            "runner_viable_window_count": self.runner_viable_window_count,
            "runner_profit_hold_capture_rate": (
                self.runner_profit_hold_capture_count / max(self.runner_profit_hold_window_count, 1)
                if self.runner_profit_hold_window_count > 0
                else 0.0
            ),
            "runner_missed_extension_count": self.runner_missed_extension_count,
            "runner_hold_after_soft_tp_count": self.runner_hold_after_soft_tp_count,
            "runner_viable_but_closed_count": self.runner_viable_but_closed_count,
            "early_full_close_after_soft_tp_count": self.early_full_close_after_soft_tp_count,
            "runner_managed_exit_count": self.runner_managed_exit_count,
            "runner_exit_profitable_count": self.runner_exit_profitable_count,
            "runner_forced_stop_count": self.runner_forced_stop_count,
            "runner_retained_profit_pct": self.runner_retained_profit_pct,
            "runner_retained_profit_score": (
                self.runner_retained_profit_score_total / max(self.runner_profit_hold_capture_count, 1)
                if self.runner_profit_hold_capture_count > 0
                else 0.0
            ),
            "runner_giveback_pct": self.runner_giveback_pct,
            "runner_giveback_ratio": (
                self.runner_giveback_pct
                / max(self.runner_retained_profit_pct + self.runner_giveback_pct, 1e-6)
                if (self.runner_retained_profit_pct + self.runner_giveback_pct) > 0.0
                else 0.0
            ),
            "profit_peak_giveback_ratio": (
                self.profit_peak_giveback_ratio_total
                / max(self.profit_peak_giveback_ratio_observations, 1)
                if self.profit_peak_giveback_ratio_observations > 0
                else 0.0
            ),
            "forced_stop_near_miss_count": self.forced_stop_near_miss_count,
            "profit_peak_reached_count": self.profit_peak_reached_count,
            "split_monetization_window_count": self.split_monetization_window_count,
            "runner_profit_hold_window_count": self.runner_profit_hold_window_count,
            "pyramid_monetization_window_count": self.pyramid_monetization_window_count,
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
        split_runner_capture_rate = (
            self.split_runner_profitable_count / self.split_executed
            if self.split_executed > 0
            else 0.0
        )
        split_zone_capture_rate = (
            self.split_profitable_count / self.split_tp_zone_opportunity_count
            if self.split_tp_zone_opportunity_count > 0
            else 0.0
        )
        split_monetization_capture_rate = (
            self.split_monetization_capture_count / self.split_monetization_window_count
            if self.split_monetization_window_count > 0
            else 0.0
        )
        pyramid_efficiency = (
            self.pyramid_profitable_count / self.pyramids_opened
            if self.pyramids_opened > 0
            else 0.0
        )
        pyramid_entry_quality_score = (
            self.pyramid_good_add_count / self.pyramids_opened
            if self.pyramids_opened > 0
            else 0.0
        )
        pyramid_exit_capture_rate = (
            self.pyramid_profitable_exit_count / self.pyramids_opened
            if self.pyramids_opened > 0
            else 0.0
        )
        pyramid_add_capture_rate = (
            self.pyramid_add_capture_count / self.pyramid_add_opportunity_count
            if self.pyramid_add_opportunity_count > 0
            else 0.0
        )
        pyramid_monetization_capture_rate = (
            self.pyramid_monetization_capture_count / self.pyramid_monetization_window_count
            if self.pyramid_monetization_window_count > 0
            else 0.0
        )
        slbe_capture_rate = (
            self.slbe_profitable_exits / self.slbe_triggered
            if self.slbe_triggered > 0
            else 0.0
        )
        hold_drag_score = (
            self.hold_drag_penalized_count / max(self.hold_drag_opportunity_count, 1)
        )
        close_quality_score = (
            self.close_winner_count / max(self.close_winner_count + self.close_loser_count, 1)
            if (self.close_winner_count + self.close_loser_count) > 0
            else 0.0
        )
        runner_extension_capture_rate = (
            self.runner_extension_capture_count / self.runner_extension_opportunity_count
            if self.runner_extension_opportunity_count > 0
            else 0.0
        )
        runner_profit_hold_capture_rate = (
            self.runner_profit_hold_capture_count / self.runner_profit_hold_window_count
            if self.runner_profit_hold_window_count > 0
            else 0.0
        )
        runner_retained_profit_score = (
            self.runner_retained_profit_score_total / max(self.runner_profit_hold_capture_count, 1)
            if self.runner_profit_hold_capture_count > 0
            else 0.0
        )
        runner_giveback_ratio = (
            self.runner_giveback_pct / max(self.runner_retained_profit_pct + self.runner_giveback_pct, 1e-6)
            if (self.runner_retained_profit_pct + self.runner_giveback_pct) > 0.0
            else 0.0
        )
        profit_peak_giveback_ratio = (
            self.profit_peak_giveback_ratio_total / max(self.profit_peak_giveback_ratio_observations, 1)
            if self.profit_peak_giveback_ratio_observations > 0
            else 0.0
        )
        hold_streak_mean = (
            self.hold_streak_total / self.hold_streak_count
            if self.hold_streak_count > 0
            else 0.0
        )
        root_mask_block_total = self.root_mask_blocked_buy_total + self.root_mask_blocked_sell_total
        root_mask_ema200_share = (
            (self.root_mask_blocked_buy_ema200 + self.root_mask_blocked_sell_ema200)
            / max(root_mask_block_total, 1)
        )
        root_mask_vwap_share = (
            (self.root_mask_blocked_buy_vwap + self.root_mask_blocked_sell_vwap)
            / max(root_mask_block_total, 1)
        )
        root_mask_adx_share = (
            (self.root_mask_blocked_buy_adx + self.root_mask_blocked_sell_adx)
            / max(root_mask_block_total, 1)
        )
        root_mask_directional_share = (
            (self.root_mask_blocked_buy_directional + self.root_mask_blocked_sell_directional)
            / max(root_mask_block_total, 1)
        )
        mechanics_metrics = {
            "split_efficiency": split_efficiency,
            "split_runner_capture_rate": split_runner_capture_rate,
            "split_zone_capture_rate": split_zone_capture_rate,
            "split_monetization_capture_rate": split_monetization_capture_rate,
            "pyramid_efficiency": pyramid_efficiency,
            "pyramid_entry_quality_score": pyramid_entry_quality_score,
            "pyramid_exit_capture_rate": pyramid_exit_capture_rate,
            "pyramid_add_capture_rate": pyramid_add_capture_rate,
            "pyramid_monetization_capture_rate": pyramid_monetization_capture_rate,
            "slbe_capture_rate": slbe_capture_rate,
            "hold_drag_score": hold_drag_score,
            "close_quality_score": close_quality_score,
            "root_mask_ema200_share": root_mask_ema200_share,
            "root_mask_vwap_share": root_mask_vwap_share,
            "root_mask_adx_share": root_mask_adx_share,
            "root_mask_directional_share": root_mask_directional_share,
            "mechanics_profile_version": self.mechanics_profile_version,
            "pyramids_opened": self.pyramids_opened,
            "pyramids_rejected": self.pyramids_rejected,
            "pyramid_profitable_count": self.pyramid_profitable_count,
            "pyramid_good_add_count": self.pyramid_good_add_count,
            "pyramid_bad_add_count": self.pyramid_bad_add_count,
            "pyramid_profitable_exit_count": self.pyramid_profitable_exit_count,
            "pyramid_total_trade_improvement_pct": self.pyramid_total_trade_improvement_pct,
            "pyramid_failed_to_improve_count": self.pyramid_failed_to_improve_count,
            "pyramid_loss_count": self.pyramid_loss_count,
            "pyramid_opportunity_count": self.pyramid_opportunity_count,
            "pyramid_add_opportunity_count": self.pyramid_add_opportunity_count,
            "pyramid_monetization_window_count": self.pyramid_monetization_window_count,
            "pyramid_add_capture_count": self.pyramid_add_capture_count,
            "pyramid_missed_add_count": self.pyramid_missed_add_count,
            "slbe_triggered": self.slbe_triggered,
            "slbe_hit": self.slbe_hit,
            "slbe_profitable_exits": self.slbe_profitable_exits,
            "slbe_lock_profit_count": self.slbe_lock_profit_count,
            "split_executed": self.split_executed,
            "split_profitable_count": self.split_profitable_count,
            "split_runner_profitable_count": self.split_runner_profitable_count,
            "split_runner_failed_count": self.split_runner_failed_count,
            "split_early_count": self.split_early_count,
            "split_decorative_count": self.split_decorative_count,
            "split_trade_value_delta": self.split_trade_value_delta,
            "split_improved_total_trade_count": self.split_improved_total_trade_count,
            "split_missed_window_count": self.split_missed_window_count,
            "split_opportunity_count": self.split_opportunity_count,
            "split_tp_zone_opportunity_count": self.split_tp_zone_opportunity_count,
            "split_monetization_window_count": self.split_monetization_window_count,
            "split_rejected": self.split_rejected,
            "split_rejected_no_value": self.split_rejected_no_value,
            "hard_stop_exit_count": self.hard_stop_exit_count,
            "soft_tp_hit_count": self.soft_tp_hit_count,
            "full_tp_hit_count": self.full_tp_hit_count,
            "time_stop_trigger_count": self.time_stop_trigger_count,
            "runner_extension_count": self.runner_extension_count,
            "runner_extension_opportunity_count": self.runner_extension_opportunity_count,
            "runner_extension_capture_rate": runner_extension_capture_rate,
            "runner_viable_window_count": self.runner_viable_window_count,
            "runner_profit_hold_window_count": self.runner_profit_hold_window_count,
            "runner_profit_hold_capture_rate": runner_profit_hold_capture_rate,
            "runner_missed_extension_count": self.runner_missed_extension_count,
            "runner_hold_after_soft_tp_count": self.runner_hold_after_soft_tp_count,
            "runner_viable_but_closed_count": self.runner_viable_but_closed_count,
            "early_full_close_after_soft_tp_count": self.early_full_close_after_soft_tp_count,
            "runner_managed_exit_count": self.runner_managed_exit_count,
            "runner_exit_profitable_count": self.runner_exit_profitable_count,
            "runner_forced_stop_count": self.runner_forced_stop_count,
            "runner_retained_profit_pct": self.runner_retained_profit_pct,
            "runner_retained_profit_score": runner_retained_profit_score,
            "runner_giveback_pct": self.runner_giveback_pct,
            "runner_giveback_ratio": runner_giveback_ratio,
            "profit_peak_reached_count": self.profit_peak_reached_count,
            "profit_peak_giveback_ratio": profit_peak_giveback_ratio,
            "forced_stop_near_miss_count": self.forced_stop_near_miss_count,
            "close_winner_count": self.close_winner_count,
            "close_loser_count": self.close_loser_count,
            "defensive_close_count": self.defensive_close_count,
            "early_close_noise_count": self.early_close_noise_count,
            "hold_streak_mean": hold_streak_mean,
            "hold_under_trend_penalty_count": self.hold_under_trend_penalty_count,
            "hold_drag_opportunity_count": self.hold_drag_opportunity_count,
            "hold_drag_penalized_count": self.hold_drag_penalized_count,
            "tp_like_exit_count": self.tp_like_exit_count,
            "tp_like_missed_count": self.tp_like_missed_count,
            "inactive_episode_penalties": self.inactive_episode_penalties,
            "insufficient_entry_penalties": self.insufficient_entry_penalties,
            "directional_imbalance_penalties": self.directional_imbalance_penalties,
            "soft_entry_penalty_count": self.soft_entry_penalty_count,
            "soft_entry_penalty_total": self.soft_entry_penalty_total,
            "soft_entry_bonus_count": self.soft_entry_bonus_count,
            "soft_entry_bonus_total": self.soft_entry_bonus_total,
            "soft_penalty_ema200_count": self.soft_penalty_ema200_count,
            "soft_penalty_vwap_count": self.soft_penalty_vwap_count,
            "soft_penalty_adx_count": self.soft_penalty_adx_count,
            "soft_penalty_obv_count": self.soft_penalty_obv_count,
            "realized_close_bonus_count": self.realized_close_bonus_count,
            "realized_split_bonus_count": self.realized_split_bonus_count,
            "slbe_exit_bonus_count": self.slbe_exit_bonus_count,
            "requested_buy_actions": self.requested_buy_actions,
            "requested_sell_actions": self.requested_sell_actions,
            "root_mask_directional_candidates_total": self.root_mask_directional_candidates_total,
            "root_mask_blocked_buy_total": self.root_mask_blocked_buy_total,
            "root_mask_blocked_sell_total": self.root_mask_blocked_sell_total,
            "root_mask_blocked_buy_ema200": self.root_mask_blocked_buy_ema200,
            "root_mask_blocked_sell_ema200": self.root_mask_blocked_sell_ema200,
            "root_mask_blocked_buy_vwap": self.root_mask_blocked_buy_vwap,
            "root_mask_blocked_sell_vwap": self.root_mask_blocked_sell_vwap,
            "root_mask_blocked_buy_adx": self.root_mask_blocked_buy_adx,
            "root_mask_blocked_sell_adx": self.root_mask_blocked_sell_adx,
            "root_mask_blocked_buy_obv": self.root_mask_blocked_buy_obv,
            "root_mask_blocked_sell_obv": self.root_mask_blocked_sell_obv,
            "root_mask_blocked_buy_directional": self.root_mask_blocked_buy_directional,
            "root_mask_blocked_sell_directional": self.root_mask_blocked_sell_directional,
            "blocked_buy_entries": self.blocked_buy_entries,
            "blocked_sell_entries": self.blocked_sell_entries,
            "blocked_buy_vwap": self.blocked_buy_vwap,
            "blocked_sell_vwap": self.blocked_sell_vwap,
            "blocked_buy_adx": self.blocked_buy_adx,
            "blocked_sell_adx": self.blocked_sell_adx,
            "blocked_buy_obv": self.blocked_buy_obv,
            "blocked_sell_obv": self.blocked_sell_obv,
            "blocked_buy_directional": self.blocked_buy_directional,
            "blocked_sell_directional": self.blocked_sell_directional,
            "entry_veto_to_hold": self.entry_veto_to_hold,
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
        positive_day_rate = (
            self.positive_days / max(len(self.daily_net_return_pct_by_day), 1)
        ) * 100.0
        directional_entries = self.long_entries + self.short_entries
        directional_imbalance = (
            abs(self.long_entries - self.short_entries) / directional_entries
            if directional_entries > 0
            else 1.0
        )
        directional_bias = self._classify_directional_bias(
            self.long_entries,
            self.short_entries,
        )
        balanced_episode = (
            directional_entries > 0
            and self.long_entries > 0
            and self.short_entries > 0
            and directional_bias == "balanced"
        )
        executed_long_entry_share = self.long_entries / max(directional_entries, 1)
        executed_short_entry_share = self.short_entries / max(directional_entries, 1)
        veto_to_hold_rate = self.entry_veto_to_hold / max(
            self.requested_buy_actions + self.requested_sell_actions,
            1,
        )
        soft_event_total = max(directional_entries + self.pyramids_opened, 1)
        soft_penalty_to_bonus_ratio = (
            self.soft_entry_penalty_total / max(self.soft_entry_bonus_total, 1e-6)
            if (self.soft_entry_penalty_total > 0.0 or self.soft_entry_bonus_total > 0.0)
            else 0.0
        )
        root_mask_rate = (
            root_mask_block_total
            / max(self.root_mask_directional_candidates_total, 1)
        )
        episode_return_pct = (
            (equity - self.spec.initial_balance) / self.spec.initial_balance * 100.0
        )
        hard_stop_exit = self.hard_stop_exit_count > 0
        bad_runner_exit = self.runner_forced_stop_count > 0 or (
            self.runner_managed_exit_count > 0
            and self.runner_giveback_pct > max(self.runner_retained_profit_pct, 0.0)
        )
        bad_pyramid_exit = (
            self.pyramids_opened > 0 and self.pyramid_profitable_exit_count <= 0
        )
        bad_split = self.split_executed > 0 and self.split_runner_profitable_count <= 0
        range_entry_loss = (
            str(self._episode_regime or "").strip().lower() == "range"
            and episode_return_pct < 0.0
        )
        liquidity_trap_loss = range_entry_loss and (
            hard_stop_exit or close_quality_score < 0.35
        )
        if liquidity_trap_loss:
            nemesis_type = "LIQUIDITY_TRAP"
        elif bad_runner_exit:
            nemesis_type = "BAD_RUNNER_EXIT"
        elif bad_pyramid_exit:
            nemesis_type = "BAD_PYRAMID_EXIT"
        elif hard_stop_exit:
            nemesis_type = "HARD_STOP_EXIT"
        elif range_entry_loss:
            nemesis_type = "RANGE_ENTRY_LOSS"
        elif bad_split:
            nemesis_type = "BAD_SPLIT"
        else:
            nemesis_type = "NONE"
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
            "return_pct": episode_return_pct,
            "daily_net_return_pct_by_day": dict(self.daily_net_return_pct_by_day),
            "best_day_net_return_pct": (
                self.best_day_net_return_pct if self.best_day_net_return_pct != float("-inf") else 0.0
            ),
            "days_above_10pct": self.days_above_10pct,
            "positive_day_rate": positive_day_rate,
            "daily_max_drawdown_pct": self.daily_max_drawdown_pct,
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
            "requested_buy_actions": self.requested_buy_actions,
            "requested_sell_actions": self.requested_sell_actions,
            "root_mask_directional_candidates_total": self.root_mask_directional_candidates_total,
            "root_mask_blocked_buy_total": self.root_mask_blocked_buy_total,
            "root_mask_blocked_sell_total": self.root_mask_blocked_sell_total,
            "root_mask_blocked_buy_ema200": self.root_mask_blocked_buy_ema200,
            "root_mask_blocked_sell_ema200": self.root_mask_blocked_sell_ema200,
            "root_mask_blocked_buy_vwap": self.root_mask_blocked_buy_vwap,
            "root_mask_blocked_sell_vwap": self.root_mask_blocked_sell_vwap,
            "root_mask_blocked_buy_adx": self.root_mask_blocked_buy_adx,
            "root_mask_blocked_sell_adx": self.root_mask_blocked_sell_adx,
            "root_mask_blocked_buy_obv": self.root_mask_blocked_buy_obv,
            "root_mask_blocked_sell_obv": self.root_mask_blocked_sell_obv,
            "root_mask_blocked_buy_directional": self.root_mask_blocked_buy_directional,
            "root_mask_blocked_sell_directional": self.root_mask_blocked_sell_directional,
            "root_mask_rate": root_mask_rate,
            "root_mask_ema200_share": root_mask_ema200_share,
            "root_mask_vwap_share": root_mask_vwap_share,
            "root_mask_adx_share": root_mask_adx_share,
            "root_mask_directional_share": root_mask_directional_share,
            "long_entries": self.long_entries,
            "short_entries": self.short_entries,
            "long_present": self.long_entries > 0,
            "short_present": self.short_entries > 0,
            "balanced_episode": balanced_episode,
            "executed_long_entry_share": executed_long_entry_share,
            "executed_short_entry_share": executed_short_entry_share,
            "directional_imbalance": directional_imbalance,
            "directional_bias": directional_bias,
            "blocked_buy_entries": self.blocked_buy_entries,
            "blocked_sell_entries": self.blocked_sell_entries,
            "blocked_buy_vwap": self.blocked_buy_vwap,
            "blocked_sell_vwap": self.blocked_sell_vwap,
            "blocked_buy_adx": self.blocked_buy_adx,
            "blocked_sell_adx": self.blocked_sell_adx,
            "blocked_buy_obv": self.blocked_buy_obv,
            "blocked_sell_obv": self.blocked_sell_obv,
            "blocked_buy_directional": self.blocked_buy_directional,
            "blocked_sell_directional": self.blocked_sell_directional,
            "entry_veto_to_hold": self.entry_veto_to_hold,
            "veto_to_hold_rate": veto_to_hold_rate,
            "post_veto_to_hold_rate": veto_to_hold_rate,
            "soft_entry_penalty_count": self.soft_entry_penalty_count,
            "soft_entry_penalty_total": self.soft_entry_penalty_total,
            "soft_entry_bonus_count": self.soft_entry_bonus_count,
            "soft_entry_bonus_total": self.soft_entry_bonus_total,
            "soft_entry_penalty_rate": self.soft_entry_penalty_count / soft_event_total,
            "soft_entry_bonus_rate": self.soft_entry_bonus_count / soft_event_total,
            "soft_penalty_net": self.soft_entry_penalty_total - self.soft_entry_bonus_total,
            "soft_penalty_to_bonus_ratio": soft_penalty_to_bonus_ratio,
            "soft_penalty_ema200_count": self.soft_penalty_ema200_count,
            "soft_penalty_vwap_count": self.soft_penalty_vwap_count,
            "soft_penalty_adx_count": self.soft_penalty_adx_count,
            "soft_penalty_obv_count": self.soft_penalty_obv_count,
            "soft_penalty_ema_rate": self.soft_penalty_ema200_count / soft_event_total,
            "soft_penalty_vwap_rate": self.soft_penalty_vwap_count / soft_event_total,
            "soft_penalty_adx_rate": self.soft_penalty_adx_count / soft_event_total,
            "soft_penalty_obv_rate": self.soft_penalty_obv_count / soft_event_total,
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
            "hold_drag_opportunity_count": self.hold_drag_opportunity_count,
            "hold_drag_penalized_count": self.hold_drag_penalized_count,
            "hold_drag_score": hold_drag_score,
            "split_opportunity_count": self.split_opportunity_count,
            "split_efficiency": split_efficiency,
            "split_runner_capture_rate": split_runner_capture_rate,
            "split_runner_profitable_count": self.split_runner_profitable_count,
            "split_runner_failed_count": self.split_runner_failed_count,
            "split_early_count": self.split_early_count,
            "split_decorative_count": self.split_decorative_count,
            "split_trade_value_delta": self.split_trade_value_delta,
            "split_improved_total_trade_count": self.split_improved_total_trade_count,
            "pyramid_opportunity_count": self.pyramid_opportunity_count,
            "pyramid_efficiency": pyramid_efficiency,
            "pyramid_entry_quality_score": pyramid_entry_quality_score,
            "pyramid_exit_capture_rate": pyramid_exit_capture_rate,
            "pyramid_good_add_count": self.pyramid_good_add_count,
            "pyramid_bad_add_count": self.pyramid_bad_add_count,
            "pyramid_profitable_exit_count": self.pyramid_profitable_exit_count,
            "pyramid_total_trade_improvement_pct": self.pyramid_total_trade_improvement_pct,
            "pyramid_failed_to_improve_count": self.pyramid_failed_to_improve_count,
            "slbe_lock_profit_count": self.slbe_lock_profit_count,
            "slbe_capture_rate": slbe_capture_rate,
            "close_quality_score": close_quality_score,
            "tp_like_missed_count": self.tp_like_missed_count,
            "defensive_close_count": self.defensive_close_count,
            "early_close_noise_count": self.early_close_noise_count,
            "hard_stop_exit_count": self.hard_stop_exit_count,
            "soft_tp_hit_count": self.soft_tp_hit_count,
            "full_tp_hit_count": self.full_tp_hit_count,
            "time_stop_trigger_count": self.time_stop_trigger_count,
            "runner_extension_count": self.runner_extension_count,
            "runner_managed_exit_count": self.runner_managed_exit_count,
            "runner_exit_profitable_count": self.runner_exit_profitable_count,
            "runner_forced_stop_count": self.runner_forced_stop_count,
            "runner_viable_window_count": self.runner_viable_window_count,
            "runner_hold_after_soft_tp_count": self.runner_hold_after_soft_tp_count,
            "runner_viable_but_closed_count": self.runner_viable_but_closed_count,
            "early_full_close_after_soft_tp_count": self.early_full_close_after_soft_tp_count,
            "runner_retained_profit_pct": self.runner_retained_profit_pct,
            "runner_retained_profit_score": runner_retained_profit_score,
            "runner_giveback_pct": self.runner_giveback_pct,
            "forced_stop_near_miss_count": self.forced_stop_near_miss_count,
            "net_return_long_pct": self.net_realized_long_pct,
            "net_return_short_pct": self.net_realized_short_pct,
            "episode_regime": self._episode_regime,
            "nemesis_type": nemesis_type,
            "liquidity_trap_loss": liquidity_trap_loss,
            "range_entry_loss": range_entry_loss,
            "bad_split": bad_split,
            "bad_runner_exit": bad_runner_exit,
            "bad_pyramid_exit": bad_pyramid_exit,
            "hard_stop_exit": hard_stop_exit,
            "metrics_by_position_mechanics": mechanics_metrics,
        }
