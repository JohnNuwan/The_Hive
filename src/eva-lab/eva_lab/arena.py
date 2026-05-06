"""Arena Darwinienne pour l'evaluation des modeles MuZero sur historique reel."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.environment import BUY, SELL, TradingEnvironment
from eva_lab.muzero.dreamer_agent import JAXDreamerAgent
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.training_utils import (
    build_muzero_market_context,
    infer_family_from_symbols,
    get_horizon_history_bars,
    get_horizon_timeframe,
    load_history_frame,
    resolve_feature_profile,
    resolve_family_training_symbols,
    resolve_training_symbols,
)
from eva_lab.training_status import set_arena_progress

logger = logging.getLogger(__name__)

POSITION_MECHANICS_KEYS = (
    "split_efficiency",
    "split_runner_capture_rate",
    "pyramid_efficiency",
    "pyramid_entry_quality_score",
    "pyramid_exit_capture_rate",
    "slbe_capture_rate",
    "hold_drag_score",
    "close_quality_score",
    "split_profitable_count",
    "split_runner_profitable_count",
    "split_runner_failed_count",
    "split_early_count",
    "split_decorative_count",
    "pyramid_profitable_count",
    "pyramid_good_add_count",
    "pyramid_bad_add_count",
    "pyramid_profitable_exit_count",
    "slbe_profitable_exits",
    "pyramids_opened",
    "pyramids_rejected",
    "pyramid_opportunity_count",
    "slbe_triggered",
    "slbe_hit",
    "slbe_lock_profit_count",
    "split_executed",
    "split_opportunity_count",
    "close_winner_count",
    "close_loser_count",
    "defensive_close_count",
    "early_close_noise_count",
    "hold_streak_mean",
    "hold_under_trend_penalty_count",
    "hold_drag_opportunity_count",
    "hold_drag_penalized_count",
    "tp_like_exit_count",
    "tp_like_missed_count",
    "hard_stop_exit_count",
    "soft_tp_hit_count",
    "full_tp_hit_count",
    "time_stop_trigger_count",
    "runner_extension_count",
    "runner_managed_exit_count",
    "runner_exit_profitable_count",
    "runner_forced_stop_count",
    "forced_stop_near_miss_count",
    "entry_blocked_vwap",
    "entry_blocked_adx",
    "entry_blocked_obv",
    "actions_above_vwap",
    "actions_below_vwap",
    "actions_low_adx",
    "obv_divergent_actions",
    "hold_in_trend_count",
    "hold_in_range_count",
    "split_rejected",
    "split_rejected_no_value",
    "inactive_episode_penalties",
    "insufficient_entry_penalties",
    "directional_imbalance_penalties",
    "realized_close_bonus_count",
    "realized_split_bonus_count",
    "slbe_exit_bonus_count",
    "requested_buy_actions",
    "requested_sell_actions",
    "blocked_buy_entries",
    "blocked_sell_entries",
    "blocked_buy_vwap",
    "blocked_sell_vwap",
    "blocked_buy_adx",
    "blocked_sell_adx",
    "blocked_buy_obv",
    "blocked_sell_obv",
    "blocked_buy_directional",
    "blocked_sell_directional",
    "entry_veto_to_hold",
)

POSITION_MECHANICS_RATIO_KEYS = {
    "split_efficiency",
    "split_runner_capture_rate",
    "pyramid_efficiency",
    "pyramid_entry_quality_score",
    "pyramid_exit_capture_rate",
    "slbe_capture_rate",
    "hold_drag_score",
    "close_quality_score",
    "hold_streak_mean",
}

POSITION_MECHANICS_COUNTER_KEYS = tuple(
    key for key in POSITION_MECHANICS_KEYS if key not in POSITION_MECHANICS_RATIO_KEYS
)


class Arena:
    """Execute des combats entre challenger et champion sur des historiques reels."""

    def __init__(
        self,
        weights_dir: str = "data/muzero/weights",
        data_dir: str = "data/history",
    ) -> None:
        """Initialise l'Arena et ses emplacements de donnees.

        Args:
            weights_dir (str): Dossier des checkpoints MuZero.
            data_dir (str): Dossier des historiques CSV.
        """
        self.history: list[dict[str, Any]] = []
        self.weights_dir = Path(weights_dir)
        self.data_dir = data_dir

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        """Lit une variable d'environnement entiere avec repli.

        Args:
            name (str): Nom de la variable d'environnement.
            default (int): Valeur de repli.

        Returns:
            int: Valeur normalisee.
        """
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except ValueError:
            logger.warning("Variable %s invalide (%s). Repli sur %s.", name, raw_value, default)
            return default

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        """Lit une variable d'environnement flottante avec repli.

        Args:
            name (str): Nom de la variable d'environnement.
            default (float): Valeur de repli.

        Returns:
            float: Valeur normalisee.
        """
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return float(raw_value)
        except ValueError:
            logger.warning("Variable %s invalide (%s). Repli sur %s.", name, raw_value, default)
            return default

    @staticmethod
    def _read_best_day_metric(value: Any) -> float:
        """Lit une metrique de meilleur jour sans perdre les zeros valides.

        Args:
            value (Any): Valeur brute a normaliser.

        Returns:
            float: Valeur flottante exploitable, ou ``-inf`` si absente.
        """
        if value is None:
            return float("-inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        """Retourne un jeu de metriques neutres si l'evaluation echoue.

        Returns:
            dict[str, Any]: Metriques nulles et tailles d'echantillons a zero.
        """
        return {
            "profit_factor": 0.0,
            "return_pct": 0.0,
            "net_realized_pct": 0.0,
            "daily_net_return_pct_by_day": {},
            "best_day_net_return_pct": 0.0,
            "days_above_10pct": 0,
            "positive_day_rate": 0.0,
            "daily_max_drawdown_pct": 100.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "evaluation_games": 0,
            "evaluation_symbols": 0,
            "max_drawdown_pct": 100.0,
            "expectancy_pct": 0.0,
            "positive_episode_rate": 0.0,
            "buy_actions": 0,
            "sell_actions": 0,
            "hold_actions": 0,
            "split_actions": 0,
            "close_actions": 0,
            "long_entries": 0,
            "short_entries": 0,
            "ema200_blocked_buy": 0,
            "ema200_blocked_sell": 0,
            "buy_share": 0.0,
            "sell_share": 0.0,
            "long_entry_share": 0.0,
            "short_entry_share": 0.0,
            "executed_long_entry_share": 0.0,
            "executed_short_entry_share": 0.0,
            "directional_imbalance": 1.0,
            "directional_bias": "inactive",
            "balanced_episode_rate": 0.0,
            "episodes_long_present_rate": 0.0,
            "episodes_short_present_rate": 0.0,
            "veto_to_hold_rate": 0.0,
            "net_return_long_pct": 0.0,
            "net_return_short_pct": 0.0,
            "blocked_buy_total": 0.0,
            "blocked_sell_total": 0.0,
            "directional_collapse": True,
            "directional_by_symbol": {},
            "metrics_by_symbol": {},
            "metrics_by_position_mechanics": {},
            "split_runner_capture_rate": 0.0,
            "pyramid_entry_quality_score": 0.0,
            "pyramid_exit_capture_rate": 0.0,
            "family": "mixed",
            "feature_profile": None,
            "mechanics_profile_version": None,
            "dataset_coverage": {},
        }

    @staticmethod
    def _inverse_action(action: int) -> int:
        """Inverse uniquement la direction d'une action MuZero.

        Args:
            action (int): Action brute proposee par le modele.

        Returns:
            int: Action eventuellement inversee.
        """
        if int(action) == BUY:
            return SELL
        if int(action) == SELL:
            return BUY
        return int(action)

    @staticmethod
    def _compute_inverse_edge(
        challenger_metrics: dict[str, Any],
        inverse_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Calcule l'edge du challenger face a son miroir directionnel.

        Args:
            challenger_metrics (dict[str, Any]): Metriques du challenger normal.
            inverse_metrics (dict[str, Any]): Metriques du challenger inverse.

        Returns:
            dict[str, Any]: Resume global et par symbole de l'edge relatif.
        """
        challenger_by_symbol = dict(challenger_metrics.get("metrics_by_symbol") or {})
        inverse_by_symbol = dict(inverse_metrics.get("metrics_by_symbol") or {})
        profitable_symbols = 0
        edge_by_symbol: dict[str, dict[str, float | bool]] = {}
        for symbol in sorted(set(challenger_by_symbol) | set(inverse_by_symbol)):
            challenger_symbol = dict(challenger_by_symbol.get(symbol) or {})
            inverse_symbol = dict(inverse_by_symbol.get(symbol) or {})
            challenger_pf = float(challenger_symbol.get("profit_factor", 0.0) or 0.0)
            inverse_pf = float(inverse_symbol.get("profit_factor", 0.0) or 0.0)
            challenger_return = float(challenger_symbol.get("return_pct", 0.0) or 0.0)
            inverse_return = float(inverse_symbol.get("return_pct", 0.0) or 0.0)
            beats_inverse = challenger_pf > inverse_pf or challenger_return > inverse_return
            profitable_symbols += int(beats_inverse)
            edge_by_symbol[symbol] = {
                "profit_factor_delta": challenger_pf - inverse_pf,
                "return_pct_delta": challenger_return - inverse_return,
                "beats_inverse": beats_inverse,
            }
        return {
            "edge_vs_inverse_pf": float(challenger_metrics.get("profit_factor", 0.0) or 0.0)
            - float(inverse_metrics.get("profit_factor", 0.0) or 0.0),
            "edge_vs_inverse_return_pct": float(challenger_metrics.get("return_pct", 0.0) or 0.0)
            - float(inverse_metrics.get("return_pct", 0.0) or 0.0),
            "edge_vs_inverse_expectancy_pct": float(
                challenger_metrics.get("expectancy_pct", 0.0) or 0.0
            )
            - float(inverse_metrics.get("expectancy_pct", 0.0) or 0.0),
            "edge_vs_inverse_by_symbol": edge_by_symbol,
            "edge_vs_inverse_profitable_symbols": profitable_symbols,
        }

    @staticmethod
    def _compute_max_drawdown_pct(equity_curve: list[float] | tuple[float, ...]) -> float:
        """Calcule le drawdown maximal d'une courbe d'equite.

        Args:
            equity_curve (list[float] | tuple[float, ...]): Courbe d'equite brute.

        Returns:
            float: Drawdown maximal en pourcentage.
        """
        if not equity_curve:
            return 0.0

        peak = float(equity_curve[0])
        max_drawdown = 0.0
        for equity in equity_curve:
            equity_value = float(equity)
            peak = max(peak, equity_value)
            if peak <= 0:
                continue
            drawdown = (peak - equity_value) / peak * 100.0
            drawdown = max(0.0, min(drawdown, 100.0))
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    @staticmethod
    def _score_metrics(metrics: dict[str, float]) -> float:
        """Produit un score unique robuste a partir des metriques d'Arena.

        Args:
            metrics (dict[str, float]): Metriques agregees du modele.

        Returns:
            float: Score comparatif utilise pour le duel champion/challenger.
        """
        profit_factor = min(max(float(metrics.get("profit_factor", 0.0)), 0.0), 5.0)
        stretch_target_pct = Arena._read_float_env("MUZERO_DAILY_STRETCH_TARGET_PCT", 10.0)
        stretch_score_bonus = Arena._read_float_env("ARENA_DAILY_STRETCH_SCORE_BONUS", 4.0)
        best_day_net_return_pct = float(metrics.get("best_day_net_return_pct", 0.0) or 0.0)
        days_above_10pct = max(float(metrics.get("days_above_10pct", 0.0) or 0.0), 0.0)
        stretch_bonus = 0.0
        if best_day_net_return_pct >= stretch_target_pct and days_above_10pct > 0.0:
            stretch_bonus += stretch_score_bonus
            stretch_bonus += min(days_above_10pct, 3.0) * (stretch_score_bonus * 0.25)
        directional_bias = str(metrics.get("directional_bias") or "inactive").strip().lower()
        directional_penalty = 0.0
        if directional_bias != "balanced":
            directional_penalty -= 8.0
        raw_directional_imbalance = metrics.get("directional_imbalance", 1.0)
        try:
            directional_imbalance = float(raw_directional_imbalance)
        except (TypeError, ValueError):
            directional_imbalance = 1.0
        directional_penalty -= 12.0 * max(0.0, directional_imbalance - 0.25)
        if int(metrics.get("long_entries", 0) or 0) <= 0 or int(metrics.get("short_entries", 0) or 0) <= 0:
            directional_penalty -= 6.0
        directional_by_symbol = dict(metrics.get("directional_by_symbol") or {})
        heavy_symbols = sum(
            1
            for symbol_metrics in directional_by_symbol.values()
            if str((symbol_metrics or {}).get("directional_bias") or "inactive").strip().lower()
            in {"buy_heavy", "sell_heavy"}
        )
        directional_penalty -= min(12.0, heavy_symbols * 3.0)
        mechanics_metrics = dict(metrics.get("metrics_by_position_mechanics") or {})
        split_executed = int(mechanics_metrics.get("split_executed", metrics.get("split_executed", 0)) or 0)
        pyramids_opened = int(mechanics_metrics.get("pyramids_opened", metrics.get("pyramids_opened", 0)) or 0)
        slbe_triggered = int(mechanics_metrics.get("slbe_triggered", metrics.get("slbe_triggered", 0)) or 0)
        mechanics_score = float(
            mechanics_metrics.get("close_quality_score", metrics.get("close_quality_score", 0.0)) or 0.0
        ) * 8.0
        if split_executed >= 3:
            mechanics_score += float(
                mechanics_metrics.get("split_efficiency", metrics.get("split_efficiency", 0.0)) or 0.0
            ) * 4.0
            mechanics_score += float(
                mechanics_metrics.get(
                    "split_runner_capture_rate",
                    metrics.get("split_runner_capture_rate", 0.0),
                )
                or 0.0
            ) * 5.0
        if pyramids_opened >= 3:
            mechanics_score += float(
                mechanics_metrics.get(
                    "pyramid_efficiency",
                    metrics.get("pyramid_efficiency", 0.0),
                )
                or 0.0
            ) * 4.0
            mechanics_score += float(
                mechanics_metrics.get(
                    "pyramid_entry_quality_score",
                    metrics.get("pyramid_entry_quality_score", 0.0),
                )
                or 0.0
            ) * 2.5
            mechanics_score += float(
                mechanics_metrics.get(
                    "pyramid_exit_capture_rate",
                    metrics.get("pyramid_exit_capture_rate", 0.0),
                )
                or 0.0
            ) * 5.0
        if slbe_triggered >= 3:
            mechanics_score += float(
                mechanics_metrics.get("slbe_capture_rate", metrics.get("slbe_capture_rate", 0.0)) or 0.0
            ) * 5.0
        mechanics_score -= float(
            mechanics_metrics.get("hold_drag_score", metrics.get("hold_drag_score", 0.0)) or 0.0
        ) * 6.0
        return (
            float(metrics.get("return_pct", 0.0)) * 8.0
            + max(0.0, profit_factor - 1.0) * 18.0
            + float(metrics.get("expectancy_pct", 0.0)) * 120.0
            + float(metrics.get("win_rate", 0.0)) * 0.08
            + float(metrics.get("positive_episode_rate", 0.0)) * 0.06
            - float(metrics.get("max_drawdown_pct", 0.0)) * 1.5
            + stretch_bonus
            + directional_penalty
            + mechanics_score
        )

    @staticmethod
    def _classify_directional_bias(long_entries: int, short_entries: int) -> str:
        """Retourne une etiquette lisible du biais directionnel observe.

        Args:
            long_entries (int): Nombre d'entrees longues.
            short_entries (int): Nombre d'entrees courtes.

        Returns:
            str: Etiquette compacte de biais directionnel.
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

    @staticmethod
    def _init_symbol_metrics() -> dict[str, float]:
        """Retourne l'accumulateur brut de metriques pour un symbole."""

        return {
            "evaluation_games": 0,
            "return_sum": 0.0,
            "net_realized_pct": 0.0,
            "daily_net_return_pct_by_day": {},
            "best_day_net_return_pct": float("-inf"),
            "days_above_10pct": 0,
            "positive_days": 0,
            "total_days": 0,
            "daily_max_drawdown_pct": 0.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "max_drawdown_pct": 0.0,
            "positive_episodes": 0,
            "buy_actions": 0,
            "sell_actions": 0,
            "hold_actions": 0,
            "split_actions": 0,
            "close_actions": 0,
            "long_entries": 0,
            "short_entries": 0,
            "long_present_games": 0,
            "short_present_games": 0,
            "balanced_games": 0,
            "entry_veto_to_hold": 0,
            "requested_buy_actions": 0,
            "requested_sell_actions": 0,
            "blocked_buy_entries": 0,
            "blocked_sell_entries": 0,
            "blocked_buy_vwap": 0,
            "blocked_sell_vwap": 0,
            "blocked_buy_adx": 0,
            "blocked_sell_adx": 0,
            "blocked_buy_obv": 0,
            "blocked_sell_obv": 0,
            "blocked_buy_directional": 0,
            "blocked_sell_directional": 0,
            "net_return_long_pct": 0.0,
            "net_return_short_pct": 0.0,
            "ema200_blocked_buy": 0,
            "ema200_blocked_sell": 0,
            "metrics_by_position_mechanics": {},
            "family": "mixed",
            "feature_profile": None,
            "mechanics_profile_version": None,
            "dataset_coverage": {},
            "hard_stop_exit_count": 0,
            "soft_tp_hit_count": 0,
            "full_tp_hit_count": 0,
            "time_stop_trigger_count": 0,
            "runner_extension_count": 0,
            "runner_exit_profitable_count": 0,
            "forced_stop_near_miss_count": 0,
        }

    @staticmethod
    def _compute_position_mechanics_metrics(state: dict[str, Any]) -> dict[str, Any]:
        """Consolide les metriques de mecanique de position.

        Args:
            state (dict[str, Any]): Accumulateur brut global ou par symbole.

        Returns:
            dict[str, Any]: Metriques consolidees de split, pyramiding, SLBE et hold.
        """
        pyramids_opened = int(state.get("pyramids_opened", 0) or 0)
        pyramids_rejected = int(state.get("pyramids_rejected", 0) or 0)
        split_executed = int(state.get("split_executed", 0) or 0)
        split_profitable_count = int(state.get("split_profitable_count", 0) or 0)
        split_runner_profitable_count = int(state.get("split_runner_profitable_count", 0) or 0)
        pyramid_profitable_count = int(state.get("pyramid_profitable_count", 0) or 0)
        pyramid_good_add_count = int(state.get("pyramid_good_add_count", 0) or 0)
        pyramid_profitable_exit_count = int(state.get("pyramid_profitable_exit_count", 0) or 0)
        slbe_triggered = int(state.get("slbe_triggered", 0) or 0)
        slbe_profitable_exits = int(state.get("slbe_profitable_exits", 0) or 0)
        close_winner_count = int(state.get("close_winner_count", 0) or 0)
        close_loser_count = int(state.get("close_loser_count", 0) or 0)
        hold_under_trend_penalty_count = int(state.get("hold_under_trend_penalty_count", 0) or 0)
        hold_drag_opportunity_count = int(state.get("hold_drag_opportunity_count", 0) or 0)
        hold_drag_penalized_count = int(state.get("hold_drag_penalized_count", 0) or 0)
        evaluation_games = int(state.get("evaluation_games", 0) or 0)

        return {
            "split_efficiency": (
                split_profitable_count / split_executed if split_executed > 0 else 0.0
            ),
            "split_runner_capture_rate": (
                split_runner_profitable_count / split_executed if split_executed > 0 else 0.0
            ),
            "pyramid_efficiency": (
                pyramid_profitable_count / pyramids_opened if pyramids_opened > 0 else 0.0
            ),
            "pyramid_entry_quality_score": (
                pyramid_good_add_count / pyramids_opened if pyramids_opened > 0 else 0.0
            ),
            "pyramid_exit_capture_rate": (
                pyramid_profitable_exit_count / pyramids_opened if pyramids_opened > 0 else 0.0
            ),
            "slbe_capture_rate": (
                slbe_profitable_exits / slbe_triggered if slbe_triggered > 0 else 0.0
            ),
            "hold_drag_score": (
                hold_drag_penalized_count / max(hold_drag_opportunity_count, 1)
            ),
            "close_quality_score": (
                close_winner_count / max(close_winner_count + close_loser_count, 1)
                if (close_winner_count + close_loser_count) > 0
                else 0.0
            ),
            "pyramids_opened": pyramids_opened,
            "pyramids_rejected": pyramids_rejected,
            "pyramid_profitable_count": pyramid_profitable_count,
            "pyramid_opportunity_count": int(state.get("pyramid_opportunity_count", 0) or 0),
            "slbe_triggered": slbe_triggered,
            "slbe_hit": int(state.get("slbe_hit", 0) or 0),
            "slbe_profitable_exits": slbe_profitable_exits,
            "slbe_lock_profit_count": int(state.get("slbe_lock_profit_count", 0) or 0),
            "split_executed": split_executed,
            "split_profitable_count": split_profitable_count,
            "split_runner_profitable_count": split_runner_profitable_count,
            "split_runner_failed_count": int(state.get("split_runner_failed_count", 0) or 0),
            "split_early_count": int(state.get("split_early_count", 0) or 0),
            "split_decorative_count": int(state.get("split_decorative_count", 0) or 0),
            "split_opportunity_count": int(state.get("split_opportunity_count", 0) or 0),
            "pyramid_good_add_count": pyramid_good_add_count,
            "pyramid_bad_add_count": int(state.get("pyramid_bad_add_count", 0) or 0),
            "pyramid_profitable_exit_count": pyramid_profitable_exit_count,
            "close_winner_count": close_winner_count,
            "close_loser_count": close_loser_count,
            "defensive_close_count": int(state.get("defensive_close_count", 0) or 0),
            "early_close_noise_count": int(state.get("early_close_noise_count", 0) or 0),
            "hard_stop_exit_count": int(state.get("hard_stop_exit_count", 0) or 0),
            "soft_tp_hit_count": int(state.get("soft_tp_hit_count", 0) or 0),
            "full_tp_hit_count": int(state.get("full_tp_hit_count", 0) or 0),
            "time_stop_trigger_count": int(state.get("time_stop_trigger_count", 0) or 0),
            "runner_extension_count": int(state.get("runner_extension_count", 0) or 0),
            "runner_managed_exit_count": int(state.get("runner_managed_exit_count", 0) or 0),
            "runner_exit_profitable_count": int(state.get("runner_exit_profitable_count", 0) or 0),
            "runner_forced_stop_count": int(state.get("runner_forced_stop_count", 0) or 0),
            "forced_stop_near_miss_count": int(state.get("forced_stop_near_miss_count", 0) or 0),
            "hold_streak_mean": float(state.get("hold_streak_mean_sum", 0.0) or 0.0) / max(evaluation_games, 1),
            "hold_under_trend_penalty_count": hold_under_trend_penalty_count,
            "hold_drag_opportunity_count": hold_drag_opportunity_count,
            "hold_drag_penalized_count": hold_drag_penalized_count,
            "tp_like_exit_count": int(state.get("tp_like_exit_count", 0) or 0),
            "tp_like_missed_count": int(state.get("tp_like_missed_count", 0) or 0),
            "entry_blocked_vwap": int(state.get("entry_blocked_vwap", 0) or 0),
            "entry_blocked_adx": int(state.get("entry_blocked_adx", 0) or 0),
            "entry_blocked_obv": int(state.get("entry_blocked_obv", 0) or 0),
            "actions_above_vwap": int(state.get("actions_above_vwap", 0) or 0),
            "actions_below_vwap": int(state.get("actions_below_vwap", 0) or 0),
            "actions_low_adx": int(state.get("actions_low_adx", 0) or 0),
            "obv_divergent_actions": int(state.get("obv_divergent_actions", 0) or 0),
            "hold_in_trend_count": int(state.get("hold_in_trend_count", 0) or 0),
            "hold_in_range_count": int(state.get("hold_in_range_count", 0) or 0),
            "split_rejected": int(state.get("split_rejected", 0) or 0),
            "split_rejected_no_value": int(state.get("split_rejected_no_value", 0) or 0),
            "inactive_episode_penalties": int(state.get("inactive_episode_penalties", 0) or 0),
            "insufficient_entry_penalties": int(state.get("insufficient_entry_penalties", 0) or 0),
            "directional_imbalance_penalties": int(state.get("directional_imbalance_penalties", 0) or 0),
            "realized_close_bonus_count": int(state.get("realized_close_bonus_count", 0) or 0),
            "realized_split_bonus_count": int(state.get("realized_split_bonus_count", 0) or 0),
            "slbe_exit_bonus_count": int(state.get("slbe_exit_bonus_count", 0) or 0),
            "requested_buy_actions": int(state.get("requested_buy_actions", 0) or 0),
            "requested_sell_actions": int(state.get("requested_sell_actions", 0) or 0),
            "blocked_buy_entries": int(state.get("blocked_buy_entries", 0) or 0),
            "blocked_sell_entries": int(state.get("blocked_sell_entries", 0) or 0),
            "blocked_buy_vwap": int(state.get("blocked_buy_vwap", 0) or 0),
            "blocked_sell_vwap": int(state.get("blocked_sell_vwap", 0) or 0),
            "blocked_buy_adx": int(state.get("blocked_buy_adx", 0) or 0),
            "blocked_sell_adx": int(state.get("blocked_sell_adx", 0) or 0),
            "blocked_buy_obv": int(state.get("blocked_buy_obv", 0) or 0),
            "blocked_sell_obv": int(state.get("blocked_sell_obv", 0) or 0),
            "blocked_buy_directional": int(state.get("blocked_buy_directional", 0) or 0),
            "blocked_sell_directional": int(state.get("blocked_sell_directional", 0) or 0),
            "entry_veto_to_hold": int(state.get("entry_veto_to_hold", 0) or 0),
            "mechanics_profile_version": str(state.get("mechanics_profile_version") or "") or None,
        }

    def _finalize_metrics_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Consolide un etat brut d'evaluation en metriques lisibles.

        Args:
            state (dict[str, Any]): Etat brut d'evaluation.

        Returns:
            dict[str, Any]: Metriques consolidees.
        """
        evaluation_games = int(state.get("evaluation_games", 0) or 0)
        if evaluation_games <= 0:
            return self._empty_metrics()

        total_return = float(state.get("total_return", 0.0) or 0.0)
        gross_profit = float(state.get("gross_profit", 0.0) or 0.0)
        gross_loss = float(state.get("gross_loss", 0.0) or 0.0)
        daily_net_return_pct_by_day = dict(state.get("daily_net_return_pct_by_day") or {})
        best_day_net_return_pct = self._read_best_day_metric(state.get("best_day_net_return_pct"))
        days_above_10pct = int(state.get("days_above_10pct", 0) or 0)
        positive_days = int(state.get("positive_days", 0) or 0)
        total_days = int(state.get("total_days", 0) or 0)
        daily_max_drawdown_pct = float(state.get("daily_max_drawdown_pct", 0.0) or 0.0)
        total_trades = int(state.get("total_trades", 0) or 0)
        total_profitable = int(state.get("total_profitable", 0) or 0)
        evaluated_symbols = int(state.get("evaluated_symbols", 0) or 0)
        positive_episodes = int(state.get("positive_episodes", 0) or 0)
        worst_drawdown = float(state.get("worst_drawdown", 0.0) or 0.0)
        total_net_realized_pct = float(state.get("total_net_realized_pct", 0.0) or 0.0)
        buy_actions = int(state.get("buy_actions", 0) or 0)
        sell_actions = int(state.get("sell_actions", 0) or 0)
        hold_actions = int(state.get("hold_actions", 0) or 0)
        split_actions = int(state.get("split_actions", 0) or 0)
        close_actions = int(state.get("close_actions", 0) or 0)
        long_entries = int(state.get("long_entries", 0) or 0)
        short_entries = int(state.get("short_entries", 0) or 0)
        balanced_games = int(state.get("balanced_games", 0) or 0)
        long_present_games = int(state.get("long_present_games", 0) or 0)
        short_present_games = int(state.get("short_present_games", 0) or 0)
        requested_buy_actions = int(state.get("requested_buy_actions", 0) or 0)
        requested_sell_actions = int(state.get("requested_sell_actions", 0) or 0)
        blocked_buy_entries = int(state.get("blocked_buy_entries", 0) or 0)
        blocked_sell_entries = int(state.get("blocked_sell_entries", 0) or 0)
        net_return_long_pct = float(state.get("net_return_long_pct", 0.0) or 0.0)
        net_return_short_pct = float(state.get("net_return_short_pct", 0.0) or 0.0)
        ema200_blocked_buy = int(state.get("ema200_blocked_buy", 0) or 0)
        ema200_blocked_sell = int(state.get("ema200_blocked_sell", 0) or 0)
        metrics_by_symbol = dict(state.get("metrics_by_symbol") or {})
        family = str(
            state.get("family")
            or infer_family_from_symbols(list(metrics_by_symbol.keys()))
            or "mixed"
        )
        feature_profile = str(
            state.get("feature_profile")
            or resolve_feature_profile(str(state.get("horizon") or "intraday"), family).get("profile_name")
            or "default"
        )
        mechanics_profile_version = str(state.get("mechanics_profile_version") or "").strip() or None
        dataset_coverage = dict(state.get("dataset_coverage") or {})

        average_return = total_return / evaluation_games
        win_rate = (total_profitable / max(total_trades, 1)) * 100.0
        expectancy_pct = total_net_realized_pct / max(total_trades, 1)
        positive_episode_rate = (positive_episodes / evaluation_games) * 100.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else (gross_profit if gross_profit > 0 else 0.0)
        directional_actions = buy_actions + sell_actions
        directional_entries = long_entries + short_entries
        buy_share = buy_actions / max(directional_actions, 1)
        sell_share = sell_actions / max(directional_actions, 1)
        long_entry_share = long_entries / max(directional_entries, 1)
        short_entry_share = short_entries / max(directional_entries, 1)
        executed_long_entry_share = long_entry_share
        executed_short_entry_share = short_entry_share
        directional_imbalance = (
            abs(long_entries - short_entries) / directional_entries
            if directional_entries > 0
            else 1.0
        )
        directional_bias = self._classify_directional_bias(long_entries, short_entries)
        veto_to_hold_rate = float(state.get("entry_veto_to_hold", 0.0) or 0.0) / max(
            requested_buy_actions + requested_sell_actions,
            1,
        )
        finalized_symbol_metrics: dict[str, dict[str, float]] = {}
        directional_by_symbol: dict[str, dict[str, float]] = {}
        global_position_mechanics = self._compute_position_mechanics_metrics(state)

        for symbol, raw_metrics in metrics_by_symbol.items():
            symbol_games = int(raw_metrics.get("evaluation_games", 0) or 0)
            symbol_trades = int(raw_metrics.get("total_trades", 0) or 0)
            symbol_profitable = int(raw_metrics.get("profitable_trades", 0) or 0)
            symbol_gross_profit = float(raw_metrics.get("gross_profit_pct", 0.0) or 0.0)
            symbol_gross_loss = float(raw_metrics.get("gross_loss_pct", 0.0) or 0.0)
            symbol_buy_actions = int(raw_metrics.get("buy_actions", 0) or 0)
            symbol_sell_actions = int(raw_metrics.get("sell_actions", 0) or 0)
            symbol_long_entries = int(raw_metrics.get("long_entries", 0) or 0)
            symbol_short_entries = int(raw_metrics.get("short_entries", 0) or 0)
            symbol_balanced_games = int(raw_metrics.get("balanced_games", 0) or 0)
            symbol_long_present_games = int(raw_metrics.get("long_present_games", 0) or 0)
            symbol_short_present_games = int(raw_metrics.get("short_present_games", 0) or 0)
            symbol_requested_buy_actions = int(raw_metrics.get("requested_buy_actions", 0) or 0)
            symbol_requested_sell_actions = int(raw_metrics.get("requested_sell_actions", 0) or 0)
            symbol_blocked_buy_entries = int(raw_metrics.get("blocked_buy_entries", 0) or 0)
            symbol_blocked_sell_entries = int(raw_metrics.get("blocked_sell_entries", 0) or 0)
            symbol_directional_actions = symbol_buy_actions + symbol_sell_actions
            symbol_directional_entries = symbol_long_entries + symbol_short_entries
            symbol_return_pct = float(raw_metrics.get("return_sum", 0.0) or 0.0) / max(symbol_games, 1)
            symbol_net_realized_pct = float(raw_metrics.get("net_realized_pct", 0.0) or 0.0)
            symbol_profit_factor = (
                symbol_gross_profit / symbol_gross_loss
                if symbol_gross_loss > 1e-8
                else (symbol_gross_profit if symbol_gross_profit > 0 else 0.0)
            )
            symbol_metrics_view = {
                "evaluation_games": symbol_games,
                "return_pct": symbol_return_pct,
                "net_realized_pct": symbol_net_realized_pct,
                "daily_net_return_pct_by_day": dict(raw_metrics.get("daily_net_return_pct_by_day") or {}),
                "best_day_net_return_pct": (
                    self._read_best_day_metric(raw_metrics.get("best_day_net_return_pct"))
                    if int(raw_metrics.get("total_days", 0) or 0) > 0
                    else 0.0
                ),
                "days_above_10pct": int(raw_metrics.get("days_above_10pct", 0) or 0),
                "positive_day_rate": (
                    int(raw_metrics.get("positive_days", 0) or 0)
                    / max(int(raw_metrics.get("total_days", 0) or 0), 1)
                ) * 100.0,
                "daily_max_drawdown_pct": float(raw_metrics.get("daily_max_drawdown_pct", 0.0) or 0.0),
                "gross_profit_pct": symbol_gross_profit,
                "gross_loss_pct": symbol_gross_loss,
                "profit_factor": symbol_profit_factor,
                "win_rate": (symbol_profitable / max(symbol_trades, 1)) * 100.0,
                "total_trades": symbol_trades,
                "profitable_trades": symbol_profitable,
                "max_drawdown_pct": float(raw_metrics.get("max_drawdown_pct", 0.0) or 0.0),
                "expectancy_pct": symbol_net_realized_pct / max(symbol_trades, 1),
                "positive_episode_rate": (
                    int(raw_metrics.get("positive_episodes", 0) or 0) / max(symbol_games, 1)
                ) * 100.0,
                "buy_actions": symbol_buy_actions,
                "sell_actions": symbol_sell_actions,
                "hold_actions": int(raw_metrics.get("hold_actions", 0) or 0),
                "split_actions": int(raw_metrics.get("split_actions", 0) or 0),
                "close_actions": int(raw_metrics.get("close_actions", 0) or 0),
                "long_entries": symbol_long_entries,
                "short_entries": symbol_short_entries,
                "ema200_blocked_buy": int(raw_metrics.get("ema200_blocked_buy", 0) or 0),
                "ema200_blocked_sell": int(raw_metrics.get("ema200_blocked_sell", 0) or 0),
                "buy_share": symbol_buy_actions / max(symbol_directional_actions, 1),
                "sell_share": symbol_sell_actions / max(symbol_directional_actions, 1),
                "long_entry_share": symbol_long_entries / max(symbol_directional_entries, 1),
                "short_entry_share": symbol_short_entries / max(symbol_directional_entries, 1),
                "executed_long_entry_share": symbol_long_entries / max(symbol_directional_entries, 1),
                "executed_short_entry_share": symbol_short_entries / max(symbol_directional_entries, 1),
                "directional_imbalance": (
                    abs(symbol_long_entries - symbol_short_entries) / symbol_directional_entries
                    if symbol_directional_entries > 0
                    else 1.0
                ),
                "balanced_episode_rate": (symbol_balanced_games / max(symbol_games, 1)) * 100.0,
                "episodes_long_present_rate": (symbol_long_present_games / max(symbol_games, 1)) * 100.0,
                "episodes_short_present_rate": (symbol_short_present_games / max(symbol_games, 1)) * 100.0,
                "veto_to_hold_rate": float(raw_metrics.get("entry_veto_to_hold", 0.0) or 0.0)
                / max(symbol_requested_buy_actions + symbol_requested_sell_actions, 1),
                "net_return_long_pct": float(raw_metrics.get("net_return_long_pct", 0.0) or 0.0),
                "net_return_short_pct": float(raw_metrics.get("net_return_short_pct", 0.0) or 0.0),
                "blocked_buy_total": symbol_blocked_buy_entries,
                "blocked_sell_total": symbol_blocked_sell_entries,
                "directional_collapse": bool(symbol_long_entries <= 0 or symbol_short_entries <= 0),
                "family": str(raw_metrics.get("family") or family),
                "feature_profile": str(raw_metrics.get("feature_profile") or feature_profile),
                "mechanics_profile_version": str(
                    raw_metrics.get("mechanics_profile_version") or mechanics_profile_version or ""
                ) or None,
            }
            symbol_metrics_view["directional_bias"] = self._classify_directional_bias(
                symbol_long_entries,
                symbol_short_entries,
            )
            symbol_metrics_view["metrics_by_position_mechanics"] = self._compute_position_mechanics_metrics(
                raw_metrics,
            )
            symbol_metrics_view["score"] = self._score_metrics(symbol_metrics_view)
            finalized_symbol_metrics[symbol] = symbol_metrics_view
            directional_by_symbol[symbol] = {
                "buy_actions": symbol_metrics_view["buy_actions"],
                "sell_actions": symbol_metrics_view["sell_actions"],
                "hold_actions": symbol_metrics_view["hold_actions"],
                "long_entries": symbol_metrics_view["long_entries"],
                "short_entries": symbol_metrics_view["short_entries"],
                "balanced_episode_rate": symbol_metrics_view["balanced_episode_rate"],
                "ema200_blocked_buy": symbol_metrics_view["ema200_blocked_buy"],
                "ema200_blocked_sell": symbol_metrics_view["ema200_blocked_sell"],
                "return_pct": symbol_metrics_view["return_pct"],
                "profit_factor": symbol_metrics_view["profit_factor"],
                "directional_imbalance": symbol_metrics_view["directional_imbalance"],
                "directional_bias": symbol_metrics_view["directional_bias"],
            }

        return {
            "profit_factor": profit_factor,
            "return_pct": average_return,
            "net_realized_pct": total_net_realized_pct,
            "daily_net_return_pct_by_day": daily_net_return_pct_by_day,
            "best_day_net_return_pct": best_day_net_return_pct if total_days > 0 else 0.0,
            "days_above_10pct": days_above_10pct,
            "positive_day_rate": (positive_days / max(total_days, 1)) * 100.0,
            "daily_max_drawdown_pct": daily_max_drawdown_pct,
            "gross_profit_pct": gross_profit,
            "gross_loss_pct": gross_loss,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "profitable_trades": total_profitable,
            "evaluation_games": evaluation_games,
            "evaluation_symbols": evaluated_symbols,
            "max_drawdown_pct": worst_drawdown,
            "expectancy_pct": expectancy_pct,
            "positive_episode_rate": positive_episode_rate,
            "buy_actions": buy_actions,
            "sell_actions": sell_actions,
            "hold_actions": hold_actions,
            "split_actions": split_actions,
            "close_actions": close_actions,
            "long_entries": long_entries,
            "short_entries": short_entries,
            "ema200_blocked_buy": ema200_blocked_buy,
            "ema200_blocked_sell": ema200_blocked_sell,
            "buy_share": buy_share,
            "sell_share": sell_share,
            "long_entry_share": long_entry_share,
            "short_entry_share": short_entry_share,
            "executed_long_entry_share": executed_long_entry_share,
            "executed_short_entry_share": executed_short_entry_share,
            "directional_imbalance": directional_imbalance,
            "directional_bias": directional_bias,
            "balanced_episode_rate": (balanced_games / evaluation_games) * 100.0,
            "episodes_long_present_rate": (long_present_games / evaluation_games) * 100.0,
            "episodes_short_present_rate": (short_present_games / evaluation_games) * 100.0,
            "veto_to_hold_rate": veto_to_hold_rate,
            "net_return_long_pct": net_return_long_pct,
            "net_return_short_pct": net_return_short_pct,
            "blocked_buy_total": blocked_buy_entries,
            "blocked_sell_total": blocked_sell_entries,
            "directional_collapse": bool(long_entries <= 0 or short_entries <= 0),
            "directional_by_symbol": directional_by_symbol,
            "metrics_by_symbol": finalized_symbol_metrics,
            "metrics_by_position_mechanics": global_position_mechanics,
            "family": family,
            "feature_profile": feature_profile,
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": str(state.get("dataset_id") or "") or None,
            "dataset_source": str(state.get("dataset_source") or "") or None,
            "dataset_coverage": dataset_coverage,
        }

    def _resolve_model_path(
        self,
        model_id: str,
        horizon: str,
        *,
        engine: str = "muzero",
    ) -> Path | None:
        """Retourne le chemin d'un modele, avec fallback sur les champions historiques.

        Args:
            model_id (str): Identifiant du modele.
            horizon (str): Horizon strategique evalue.
            engine (str): Moteur cible (`muzero` ou `dreamer`).

        Returns:
            Path | None: Chemin vers les poids du modele si disponible.
        """
        normalized_engine = str(engine or "muzero").strip().lower()
        direct_path = self.weights_dir / f"{model_id}.pkl"
        if direct_path.exists():
            return direct_path

        raw_candidate = Path(str(model_id))
        if raw_candidate.exists():
            return raw_candidate

        candidates = []
        if normalized_engine == "dreamer":
            if model_id in {"gen_000_baseline", "dreamer_champion"}:
                candidates.extend(
                    [
                        self.weights_dir / f"dreamer_champion_{horizon}.pkl",
                        self.weights_dir / "dreamer_champion.pkl",
                    ]
                )
        else:
            if model_id == "gen_000_baseline":
                candidates.extend(
                    [
                        self.weights_dir / f"muzero_champion_{horizon}.pkl",
                        self.weights_dir / "muzero_champion.pkl",
                        self.weights_dir / "muzero_global_latest.pkl",
                    ]
                )
            elif model_id == "muzero_champion":
                candidates.extend(
                    [
                        self.weights_dir / f"muzero_champion_{horizon}.pkl",
                        self.weights_dir / "muzero_champion.pkl",
                    ]
                )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_market_data(self, symbol: str, horizon: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Charge la matrice de marche et ses jours reels pour l'evaluation.

        Args:
            symbol (str): Symbole a evaluer.
            horizon (str): Horizon strategique cible.

        Returns:
            tuple[np.ndarray, np.ndarray] | None: Matrice MuZero et etiquettes
                journalieres alignees, ou ``None`` si absentes.
        """
        timeframe = get_horizon_timeframe(horizon)
        frame = load_history_frame(symbol, timeframe, self.data_dir)
        if frame is None:
            logger.warning("Arena: historique absent pour %s sur %s.", symbol, timeframe)
            return None

        history_bars = self._read_int_env(
            "ARENA_HISTORY_BARS",
            get_horizon_history_bars(horizon, env_prefix="ARENA_HISTORY", fallback=6000),
        )
        market_data, day_labels = build_muzero_market_context(frame.tail(history_bars))
        if market_data.shape[0] < 240:
            logger.warning("Arena: historique insuffisant pour %s sur %s.", symbol, timeframe)
            return None
        return market_data, day_labels

    def _build_eval_segments(
        self,
        market_data: np.ndarray,
        config: MuZeroConfigV3,
        games_per_symbol: int,
    ) -> list[tuple[int, int, np.ndarray]]:
        """Decoupe un historique en plusieurs fenetres d'evaluation.

        Args:
            market_data (np.ndarray): Matrice de marche complete.
            config (MuZeroConfigV3): Configuration MuZero courante.
            games_per_symbol (int): Nombre de fenetres a evaluer par symbole.

        Returns:
            list[tuple[int, int, np.ndarray]]: Liste de segments chronologiques
                avec bornes ``start/end`` explicites.
        """
        min_segment_bars = max(240, config.max_moves + 120)
        target_segment_bars = max(
            min_segment_bars,
            self._read_int_env("ARENA_SEGMENT_BARS", 1400),
        )
        if market_data.shape[0] < min_segment_bars:
            return []

        if market_data.shape[0] <= target_segment_bars:
            start_index = max(0, market_data.shape[0] - target_segment_bars)
            return [(start_index, market_data.shape[0], market_data[-target_segment_bars:])]

        endpoint_values = np.linspace(
            min_segment_bars,
            market_data.shape[0],
            num=max(1, games_per_symbol),
            dtype=int,
        )
        segments: list[tuple[int, int, np.ndarray]] = []
        for end_index in sorted({int(value) for value in endpoint_values if int(value) >= min_segment_bars}):
            start_index = max(0, end_index - target_segment_bars)
            segment = market_data[start_index:end_index]
            if segment.shape[0] >= min_segment_bars:
                segments.append((start_index, end_index, segment))
        return segments

    def _load_agent(
        self,
        *,
        engine: str,
        horizon: str,
        symbols: list[str],
        weights_path: Path,
    ) -> object | None:
        """Charge l'agent d'inference adapte au moteur demande.

        Args:
            engine (str): Moteur cible (`muzero` ou `dreamer`).
            horizon (str): Horizon strategique evalue.
            symbols (list[str]): Univers utilise pour l'evaluation.
            weights_path (Path): Checkpoint a charger.

        Returns:
            object | None: Agent charge ou ``None`` si le chargement echoue.
        """
        config = MuZeroConfigV3(
            horizon=horizon,
            primary_timeframe=get_horizon_timeframe(horizon),
            symbols=symbols,
        )
        normalized_engine = str(engine or "muzero").strip().lower()
        try:
            if normalized_engine == "dreamer":
                agent = JAXDreamerAgent(config)
            else:
                agent = JAXMuZeroAgent(config)
            agent.load(str(weights_path))
            return agent
        except Exception as exc:
            logger.error(
                "Chargement %s impossible pour %s: %s",
                normalized_engine,
                weights_path,
                exc,
            )
            return None

    def _evaluate_model(
        self,
        weights_path: Path,
        symbols: list[str],
        horizon: str,
        *,
        engine: str = "muzero",
        role: str | None = None,
        eval_games_per_symbol: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        action_transform: Callable[[int], int] | None = None,
    ) -> dict[str, Any]:
        """Evalue un modele sur plusieurs fenetres historiques reellement distinctes.

        Args:
            weights_path (Path): Checkpoint du modele a tester.
            symbols (list[str]): Univers historique retenu.
            horizon (str): Horizon strategique evalue.
            role (str | None): Role logique (`challenger` ou `champion`).
            eval_games_per_symbol (int | None): Nombre de segments d'evaluation
                par symbole. Si absent, la variable d'environnement Arena est
                utilisee.
            progress_callback (Callable[[dict[str, Any]], None] | None): Rappel
                optionnel appele a chaque symbole complete.
            action_transform (Callable[[int], int] | None): Transformation
                optionnelle appliquee aux actions avant execution.

        Returns:
            dict[str, Any]: Metriques consolidees de robustesse.
        """
        if not weights_path.exists():
            logger.warning("Poids absents pour l'evaluation: %s", weights_path)
            return self._empty_metrics()

        config = MuZeroConfigV3(
            horizon=horizon,
            primary_timeframe=get_horizon_timeframe(horizon),
            symbols=symbols,
        )
        agent = self._load_agent(
            engine=engine,
            horizon=horizon,
            symbols=symbols,
            weights_path=weights_path,
        )
        if agent is None:
            return self._empty_metrics()

        effective_games_per_symbol = (
            max(1, int(eval_games_per_symbol))
            if eval_games_per_symbol is not None
            else max(1, self._read_int_env("ARENA_GAMES_PER_SYMBOL", 6))
        )
        state: dict[str, Any] = {
            "total_return": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "daily_net_return_pct_by_day": {},
            "best_day_net_return_pct": float("-inf"),
            "days_above_10pct": 0,
            "positive_days": 0,
            "total_days": 0,
            "daily_max_drawdown_pct": 0.0,
            "total_trades": 0,
            "total_profitable": 0,
            "evaluated_symbols": 0,
            "evaluation_games": 0,
            "positive_episodes": 0,
            "worst_drawdown": 0.0,
            "total_net_realized_pct": 0.0,
            "buy_actions": 0,
            "sell_actions": 0,
            "hold_actions": 0,
            "split_actions": 0,
            "close_actions": 0,
            "long_entries": 0,
            "short_entries": 0,
            "balanced_games": 0,
            "long_present_games": 0,
            "short_present_games": 0,
            "requested_buy_actions": 0,
            "requested_sell_actions": 0,
            "blocked_buy_entries": 0,
            "blocked_sell_entries": 0,
            "net_return_long_pct": 0.0,
            "net_return_short_pct": 0.0,
            "ema200_blocked_buy": 0,
            "ema200_blocked_sell": 0,
            "hold_streak_mean_sum": 0.0,
            "family": str(getattr(config, "model_family", None) or infer_family_from_symbols(symbols)),
            "feature_profile": str((getattr(config, "feature_profile", {}) or {}).get("profile_name") or ""),
            "mechanics_profile_version": str(getattr(config, "mechanics_profile_version", "") or ""),
            "dataset_id": str(getattr(config, "dataset_id", "") or ""),
            "dataset_source": str(getattr(config, "dataset_source", "") or ""),
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "horizon": horizon,
            "metrics_by_symbol": {},
        }
        for key in POSITION_MECHANICS_COUNTER_KEYS:
            state[key] = 0

        for symbol_index, symbol in enumerate(symbols, start=1):
            market_payload = self._load_market_data(symbol, horizon)
            if market_payload is None:
                continue
            market_data, day_labels = market_payload

            segments = self._build_eval_segments(market_data, config, effective_games_per_symbol)
            if not segments:
                continue

            state["evaluated_symbols"] += 1
            for segment_index, (segment_start, _segment_end, segment) in enumerate(segments, start=1):
                # Le segment doit conserver ses jours reels pour calculer le stress journalier.
                segment_day_labels = day_labels[segment_start : segment_start + segment.shape[0]]
                max_steps = min(config.max_moves, segment.shape[0] - 101)
                env = TradingEnvironment(
                    data=segment,
                    day_labels=segment_day_labels,
                    symbol=symbol,
                    config=config,
                    max_steps=max_steps,
                )
                agent.play_game(
                    env,
                    exploration=False,
                    action_transform=action_transform,
                )
                summary = env.get_summary()

                episode_return = float(summary.get("return_pct", 0.0))
                episode_net_realized = float(summary.get("net_realized_pct", episode_return))
                episode_gross_profit = float(summary.get("gross_profit_pct", 0.0))
                episode_gross_loss = float(summary.get("gross_loss_pct", 0.0))
                episode_trades = int(summary.get("total_trades", 0) or 0)
                episode_profitable = int(summary.get("profitable_trades", 0) or 0)
                episode_drawdown = self._compute_max_drawdown_pct(getattr(env, "equity_curve", []))

                state["evaluation_games"] += 1
                state["total_return"] += episode_return
                state["total_net_realized_pct"] += episode_net_realized
                state["total_trades"] += episode_trades
                state["total_profitable"] += episode_profitable
                state["worst_drawdown"] = max(float(state.get("worst_drawdown", 0.0)), episode_drawdown)

                state["gross_profit"] += max(0.0, episode_gross_profit)
                state["gross_loss"] += max(0.0, episode_gross_loss)
                episode_daily_returns = dict(summary.get("daily_net_return_pct_by_day") or {})
                state["best_day_net_return_pct"] = max(
                    self._read_best_day_metric(state.get("best_day_net_return_pct")),
                    self._read_best_day_metric(summary.get("best_day_net_return_pct")),
                )
                state["days_above_10pct"] += int(summary.get("days_above_10pct", 0) or 0)
                state["positive_days"] += sum(
                    1 for value in episode_daily_returns.values() if float(value) > 0.0
                )
                state["total_days"] += len(episode_daily_returns)
                state["daily_max_drawdown_pct"] = max(
                    float(state.get("daily_max_drawdown_pct", 0.0) or 0.0),
                    float(summary.get("daily_max_drawdown_pct", 0.0) or 0.0),
                )
                for day_label, day_return_pct in episode_daily_returns.items():
                    state["daily_net_return_pct_by_day"][
                        f"{symbol}::g{segment_index:02d}::{day_label}"
                    ] = float(day_return_pct)

                episode_buy_actions = int(summary.get("buy_actions", 0) or 0)
                episode_sell_actions = int(summary.get("sell_actions", 0) or 0)
                episode_hold_actions = int(summary.get("hold_actions", 0) or 0)
                episode_split_actions = int(summary.get("split_actions", 0) or 0)
                episode_close_actions = int(summary.get("close_actions", 0) or 0)
                episode_long_entries = int(summary.get("long_entries", 0) or 0)
                episode_short_entries = int(summary.get("short_entries", 0) or 0)
                episode_balanced = bool(summary.get("balanced_episode", False))
                episode_long_present = bool(summary.get("long_present", False))
                episode_short_present = bool(summary.get("short_present", False))
                episode_requested_buy_actions = int(summary.get("requested_buy_actions", 0) or 0)
                episode_requested_sell_actions = int(summary.get("requested_sell_actions", 0) or 0)
                episode_blocked_buy_entries = int(summary.get("blocked_buy_entries", 0) or 0)
                episode_blocked_sell_entries = int(summary.get("blocked_sell_entries", 0) or 0)
                episode_net_return_long_pct = float(summary.get("net_return_long_pct", 0.0) or 0.0)
                episode_net_return_short_pct = float(summary.get("net_return_short_pct", 0.0) or 0.0)
                episode_blocked_buy = int(summary.get("ema200_blocked_buy", 0) or 0)
                episode_blocked_sell = int(summary.get("ema200_blocked_sell", 0) or 0)

                state["buy_actions"] += episode_buy_actions
                state["sell_actions"] += episode_sell_actions
                state["hold_actions"] += episode_hold_actions
                state["split_actions"] += episode_split_actions
                state["close_actions"] += episode_close_actions
                state["long_entries"] += episode_long_entries
                state["short_entries"] += episode_short_entries
                state["balanced_games"] += int(episode_balanced)
                state["long_present_games"] += int(episode_long_present)
                state["short_present_games"] += int(episode_short_present)
                state["requested_buy_actions"] += episode_requested_buy_actions
                state["requested_sell_actions"] += episode_requested_sell_actions
                state["blocked_buy_entries"] += episode_blocked_buy_entries
                state["blocked_sell_entries"] += episode_blocked_sell_entries
                state["net_return_long_pct"] += episode_net_return_long_pct
                state["net_return_short_pct"] += episode_net_return_short_pct
                state["ema200_blocked_buy"] += episode_blocked_buy
                state["ema200_blocked_sell"] += episode_blocked_sell
                mechanics_metrics = dict(summary.get("metrics_by_position_mechanics") or {})
                state["hold_streak_mean_sum"] += float(mechanics_metrics.get("hold_streak_mean", 0.0) or 0.0)
                for key in POSITION_MECHANICS_COUNTER_KEYS:
                    state[key] += int(mechanics_metrics.get(key, 0) or 0)

                symbol_metrics = state["metrics_by_symbol"].setdefault(symbol, self._init_symbol_metrics())
                symbol_metrics["evaluation_games"] += 1
                symbol_metrics["return_sum"] += episode_return
                symbol_metrics["net_realized_pct"] += episode_net_realized
                symbol_metrics["best_day_net_return_pct"] = max(
                    self._read_best_day_metric(symbol_metrics.get("best_day_net_return_pct")),
                    self._read_best_day_metric(summary.get("best_day_net_return_pct")),
                )
                symbol_metrics["days_above_10pct"] += int(summary.get("days_above_10pct", 0) or 0)
                symbol_metrics["positive_days"] += sum(
                    1 for value in episode_daily_returns.values() if float(value) > 0.0
                )
                symbol_metrics["total_days"] += len(episode_daily_returns)
                symbol_metrics["daily_max_drawdown_pct"] = max(
                    float(symbol_metrics.get("daily_max_drawdown_pct", 0.0) or 0.0),
                    float(summary.get("daily_max_drawdown_pct", 0.0) or 0.0),
                )
                for day_label, day_return_pct in episode_daily_returns.items():
                    symbol_metrics["daily_net_return_pct_by_day"][
                        f"g{segment_index:02d}::{day_label}"
                    ] = float(day_return_pct)
                symbol_metrics["gross_profit_pct"] += max(0.0, episode_gross_profit)
                symbol_metrics["gross_loss_pct"] += max(0.0, episode_gross_loss)
                symbol_metrics["total_trades"] += episode_trades
                symbol_metrics["profitable_trades"] += episode_profitable
                symbol_metrics["max_drawdown_pct"] = max(
                    float(symbol_metrics.get("max_drawdown_pct", 0.0)),
                    episode_drawdown,
                )
                symbol_metrics["buy_actions"] += episode_buy_actions
                symbol_metrics["sell_actions"] += episode_sell_actions
                symbol_metrics["hold_actions"] += episode_hold_actions
                symbol_metrics["split_actions"] += episode_split_actions
                symbol_metrics["close_actions"] += episode_close_actions
                symbol_metrics["long_entries"] += episode_long_entries
                symbol_metrics["short_entries"] += episode_short_entries
                symbol_metrics["balanced_games"] += int(episode_balanced)
                symbol_metrics["long_present_games"] += int(episode_long_present)
                symbol_metrics["short_present_games"] += int(episode_short_present)
                symbol_metrics["requested_buy_actions"] += episode_requested_buy_actions
                symbol_metrics["requested_sell_actions"] += episode_requested_sell_actions
                symbol_metrics["blocked_buy_entries"] += episode_blocked_buy_entries
                symbol_metrics["blocked_sell_entries"] += episode_blocked_sell_entries
                symbol_metrics["net_return_long_pct"] += episode_net_return_long_pct
                symbol_metrics["net_return_short_pct"] += episode_net_return_short_pct
                symbol_metrics["ema200_blocked_buy"] += episode_blocked_buy
                symbol_metrics["ema200_blocked_sell"] += episode_blocked_sell
                symbol_metrics["family"] = str(summary.get("family") or state.get("family") or "mixed")
                symbol_metrics["feature_profile"] = str(
                    summary.get("feature_profile") or state.get("feature_profile") or ""
                )
                symbol_metrics["mechanics_profile_version"] = str(
                    summary.get("mechanics_profile_version") or state.get("mechanics_profile_version") or ""
                )
                symbol_metrics["hold_streak_mean_sum"] = float(
                    symbol_metrics.get("hold_streak_mean_sum", 0.0) or 0.0
                ) + float(mechanics_metrics.get("hold_streak_mean", 0.0) or 0.0)
                for key in POSITION_MECHANICS_COUNTER_KEYS:
                    symbol_metrics[key] = int(symbol_metrics.get(key, 0) or 0) + int(
                        mechanics_metrics.get(key, 0) or 0
                    )

                if episode_return > 0:
                    state["positive_episodes"] += 1
                    symbol_metrics["positive_episodes"] += 1

            if progress_callback and int(state.get("evaluation_games", 0) or 0) > 0:
                partial_metrics = self._finalize_metrics_from_state(state)
                progress_callback(
                    {
                        "role": role,
                        "symbol": symbol,
                        "symbol_index": symbol_index,
                        "symbol_total": len(symbols),
                        "metrics": partial_metrics,
                        "symbol_metrics": dict(partial_metrics.get("metrics_by_symbol", {}).get(symbol) or {}),
                        "mechanics_profile_version": partial_metrics.get("mechanics_profile_version"),
                        "dataset_coverage": dict(partial_metrics.get("dataset_coverage") or {}),
                    }
                )

        if int(state.get("evaluation_games", 0) or 0) == 0:
            return self._empty_metrics()
        return self._finalize_metrics_from_state(state)

    def evaluate_candidate(
        self,
        challenger_id: str,
        *,
        horizon: str = "intraday",
        engine: str = "muzero",
        eval_symbols: list[str] | None = None,
        games_per_symbol: int = 6,
    ) -> dict[str, Any]:
        """Evalue un challenger seul sur un univers et un budget explicites.

        Cette methode sert aux prechecks Gold intermediaires. Elle ne compare
        pas le challenger au champion live et ne remplace pas l'Arena finale.

        Args:
            challenger_id (str): Identifiant du challenger a tester.
            horizon (str): Horizon strategique evalue.
            engine (str): Moteur du challenger (`muzero` ou `dreamer`).
            eval_symbols (list[str] | None): Univers explicite de symboles.
            games_per_symbol (int): Nombre de segments d'evaluation par symbole.

        Returns:
            dict[str, Any]: Rapport compact de pre-evaluation du challenger.

        Raises:
            FileNotFoundError: Si le checkpoint challenger est introuvable.
        """

        normalized_horizon = str(horizon or "intraday").lower()
        timeframe = get_horizon_timeframe(normalized_horizon)
        symbols = [str(symbol).strip() for symbol in list(eval_symbols or []) if str(symbol).strip()]
        if not symbols:
            symbols = resolve_training_symbols(
                data_dir=self.data_dir,
                required_timeframes={timeframe},
                max_symbols=self._read_int_env("ARENA_MAX_SYMBOLS", 12),
                override_env_names=[
                    f"ARENA_SYMBOLS_{normalized_horizon.upper()}",
                    "ARENA_SYMBOLS",
                    f"MUZERO_SYMBOLS_{normalized_horizon.upper()}",
                    "MUZERO_SYMBOLS",
                ],
            )
        if not symbols:
            symbols = ["XAUUSD"]

        normalized_engine = str(engine or "muzero").strip().lower()
        challenger_path = self._resolve_model_path(
            challenger_id,
            normalized_horizon,
            engine=normalized_engine,
        )
        if challenger_path is None:
            raise FileNotFoundError(f"Modele challenger introuvable: {challenger_id}")

        challenger_metrics = self._evaluate_model(
            challenger_path,
            symbols,
            normalized_horizon,
            engine=normalized_engine,
            role="challenger",
            eval_games_per_symbol=max(1, int(games_per_symbol)),
        )
        return {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": (
                "DREAMER_GOLD_PRECHECK"
                if normalized_engine == "dreamer"
                else "MUZERO_GOLD_PRECHECK"
            ),
            "engine": normalized_engine,
            "horizon": normalized_horizon,
            "timeframe": timeframe,
            "eval_symbols": symbols,
            "games_per_symbol": max(1, int(games_per_symbol)),
            "challenger": {
                "id": challenger_id,
                "path": str(challenger_path),
                "score": round(self._score_metrics(challenger_metrics), 4),
                "metrics": challenger_metrics,
            },
        }

    def run_family_probes(
        self,
        challenger_id: str,
        *,
        horizon: str,
        engine: str = "muzero",
        weights_path: Path | None = None,
    ) -> dict[str, Any]:
        """Lance des probes rapides par famille pour un challenger mixte.

        Args:
            challenger_id (str): Identifiant logique du challenger.
            horizon (str): Horizon strategique evalue.
            engine (str): Moteur a evaluer.
            weights_path (Path | None): Chemin explicite du checkpoint. Si
                absent, il est resolu depuis l'identifiant.

        Returns:
            dict[str, Any]: Rapports par famille et verdict consolide.
        """
        normalized_engine = str(engine or "muzero").strip().lower()
        normalized_horizon = str(horizon or "scalp").strip().lower()
        challenger_path = weights_path or self._resolve_model_path(
            challenger_id,
            normalized_horizon,
            engine=normalized_engine,
        )
        if challenger_path is None:
            raise FileNotFoundError(f"Modele challenger introuvable: {challenger_id}")

        min_required_by_family = {
            "fx": 3,
            "indices": 3,
            "metals": 2,
            "crypto": 2,
        }
        games_per_symbol = max(
            1,
            self._read_int_env("MUZERO_ARENA_FAMILY_PROBE_GAMES_PER_SYMBOL", 2),
        )
        max_symbols = self._read_int_env("ARENA_MAX_SYMBOLS", 12)
        max_inverse_pf_gap = self._read_float_env(
            "MUZERO_ARENA_FAMILY_PROBE_MAX_INVERSE_PF_GAP",
            0.10,
        )
        reports: dict[str, Any] = {}
        ready_families = 0
        positive_families = 0
        inverse_beaten_families: list[str] = []

        for family in ("fx", "indices", "metals", "crypto"):
            probe_symbols = resolve_family_training_symbols(
                horizon=normalized_horizon,
                family=family,
                data_dir=self.data_dir,
                max_symbols=max_symbols,
            )
            min_required = min_required_by_family[family]
            if len(probe_symbols) < min_required:
                reports[family] = {
                    "status": "not_ready",
                    "family": family,
                    "required_symbols": min_required,
                    "available_symbols": len(probe_symbols),
                    "symbols": probe_symbols,
                }
                continue

            ready_families += 1
            challenger_metrics = self._evaluate_model(
                challenger_path,
                probe_symbols,
                normalized_horizon,
                engine=normalized_engine,
                role=f"family_{family}",
                eval_games_per_symbol=games_per_symbol,
            )
            inverse_metrics = (
                self._evaluate_model(
                    challenger_path,
                    probe_symbols,
                    normalized_horizon,
                    engine=normalized_engine,
                    role=f"inverse_family_{family}",
                    eval_games_per_symbol=games_per_symbol,
                    action_transform=self._inverse_action,
                )
                if normalized_engine == "muzero"
                else self._empty_metrics()
            )
            edge = self._compute_inverse_edge(challenger_metrics, inverse_metrics)
            challenger_pf = float(challenger_metrics.get("profit_factor", 0.0) or 0.0)
            challenger_return = float(challenger_metrics.get("return_pct", 0.0) or 0.0)
            inverse_pf = float(inverse_metrics.get("profit_factor", 0.0) or 0.0)
            family_positive = challenger_pf >= 1.0 or challenger_return > 0.0
            if family_positive:
                positive_families += 1
            inverse_beats = (
                challenger_pf <= (inverse_pf - max_inverse_pf_gap)
                and challenger_return <= 0.0
            )
            if inverse_beats:
                inverse_beaten_families.append(family)
            reports[family] = {
                "status": "ready",
                "family": family,
                "symbols": probe_symbols,
                "games_per_symbol": games_per_symbol,
                "challenger": {
                    "score": round(self._score_metrics(challenger_metrics), 4),
                    "metrics": challenger_metrics,
                },
                "inverse_metrics": inverse_metrics,
                "positive": family_positive,
                "inverse_beats_challenger": inverse_beats,
                **edge,
            }

        min_ready_families = max(
            1,
            self._read_int_env("MUZERO_ARENA_FAMILY_PROBE_MIN_READY_FAMILIES", 3),
        )
        min_positive_families = max(
            1,
            self._read_int_env("MUZERO_ARENA_FAMILY_PROBE_MIN_POSITIVE_FAMILIES", 2),
        )
        summary = {
            "ready_families": ready_families,
            "positive_families": positive_families,
            "required_ready_families": min_ready_families,
            "required_positive_families": min_positive_families,
            "inverse_beaten_families": inverse_beaten_families,
            "allowed": (
                ready_families >= min_ready_families
                and positive_families >= min_positive_families
                and not inverse_beaten_families
            ),
        }
        if ready_families < min_ready_families:
            summary["reason"] = "family_probe_not_ready"
        elif positive_families < min_positive_families:
            summary["reason"] = "family_probe_insufficient_positive_families"
        elif inverse_beaten_families:
            summary["reason"] = "family_probe_inverse_beats_challenger"
        else:
            summary["reason"] = "family_probe_passed"

        return {
            "family_probe_reports": reports,
            "family_probe_summary": summary,
        }

    def battle(
        self,
        challenger_id: str,
        champion_id: str = "gen_000_baseline",
        horizon: str = "intraday",
        engine: str = "muzero",
    ) -> dict[str, Any]:
        """Compare deux generations MuZero et retourne le verdict ADN.

        Args:
            challenger_id (str): Identifiant du challenger.
            champion_id (str): Identifiant du champion courant.
            horizon (str): Horizon strategique evalue.
            engine (str): Moteur du duel (`muzero` ou `dreamer`).

        Returns:
            dict[str, Any]: Rapport complet de duel et de validation.
        """
        normalized_engine = str(engine or "muzero").strip().lower()
        horizon = horizon.lower()
        timeframe = get_horizon_timeframe(horizon)
        eval_symbols = resolve_training_symbols(
            data_dir=self.data_dir,
            required_timeframes={timeframe},
            max_symbols=self._read_int_env("ARENA_MAX_SYMBOLS", 12),
            override_env_names=[
                f"ARENA_SYMBOLS_{horizon.upper()}",
                "ARENA_SYMBOLS",
                f"MUZERO_SYMBOLS_{horizon.upper()}",
                "MUZERO_SYMBOLS",
            ],
        )
        if not eval_symbols:
            eval_symbols = ["XAUUSD", "EURUSD", "BTCUSD", "GBPUSD", "USDJPY"]

        challenger_path = self._resolve_model_path(
            challenger_id,
            horizon,
            engine=normalized_engine,
        )
        champion_path = self._resolve_model_path(
            champion_id,
            horizon,
            engine=normalized_engine,
        )
        if challenger_path is None:
            raise FileNotFoundError(f"Modele challenger introuvable: {challenger_id}")

        logger.info(
            "Arena %s/%s: challenger=%s | champion=%s | symboles=%s",
            normalized_engine,
            horizon,
            challenger_path,
            champion_path,
            eval_symbols,
        )

        arena_progress: dict[str, Any] = {
            "status": "running",
            "engine": normalized_engine,
            "horizon": horizon,
            "timeframe": timeframe,
            "family": infer_family_from_symbols(eval_symbols),
            "feature_profile": resolve_feature_profile(
                horizon,
                infer_family_from_symbols(eval_symbols),
            ).get("profile_name"),
            "mechanics_profile_version": None,
            "dataset_coverage": {},
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "eval_symbols": eval_symbols,
            "symbol_total": len(eval_symbols),
            "current_role": "challenger",
            "current_symbol": None,
            "symbol_index": 0,
            "challenger": {
                "id": challenger_id,
                "path": str(challenger_path),
                "score": 0.0,
                "metrics": {},
            },
            "champion": {
                "id": champion_id,
                "path": str(champion_path) if champion_path is not None else None,
                "score": 0.0,
                "metrics": {},
            },
            "inverse_challenger": {
                "id": f"{challenger_id}::inverse",
                "path": str(challenger_path),
                "score": 0.0,
                "metrics": {},
            },
            "symbols": {},
        }

        def publish_progress(payload: dict[str, Any]) -> None:
            """Publie l'avancement partiel d'une evaluation Arena."""

            role_name = str(payload.get("role") or "").lower()
            if role_name not in {"challenger", "champion", "inverse_challenger"}:
                return

            partial_metrics = dict(payload.get("metrics") or {})
            symbol = str(payload.get("symbol") or "")
            symbol_metrics = dict(payload.get("symbol_metrics") or {})
            symbol_index = int(payload.get("symbol_index") or 0)
            symbol_total = int(payload.get("symbol_total") or len(eval_symbols))
            score = round(self._score_metrics(partial_metrics), 4) if partial_metrics else 0.0

            arena_progress["updated_at"] = datetime.now().isoformat()
            arena_progress["current_role"] = role_name
            arena_progress["current_symbol"] = symbol or None
            arena_progress["symbol_index"] = symbol_index
            arena_progress["symbol_total"] = symbol_total
            arena_progress["family"] = partial_metrics.get("family") or arena_progress.get("family")
            arena_progress["feature_profile"] = (
                partial_metrics.get("feature_profile") or arena_progress.get("feature_profile")
            )
            arena_progress["mechanics_profile_version"] = (
                partial_metrics.get("mechanics_profile_version")
                or arena_progress.get("mechanics_profile_version")
            )
            arena_progress["dataset_coverage"] = dict(
                partial_metrics.get("dataset_coverage")
                or arena_progress.get("dataset_coverage")
                or {}
            )
            arena_progress[f"{role_name}_score"] = score
            arena_progress[role_name] = {
                "id": (
                    challenger_id
                    if role_name == "challenger"
                    else (
                        f"{challenger_id}::inverse"
                        if role_name == "inverse_challenger"
                        else champion_id
                    )
                ),
                "path": (
                    str(challenger_path)
                    if role_name in {"challenger", "inverse_challenger"}
                    else (str(champion_path) if champion_path is not None else None)
                ),
                "score": score,
                "metrics": partial_metrics,
            }

            if symbol:
                symbol_entry = dict(arena_progress["symbols"].get(symbol) or {"symbol": symbol, "order": symbol_index})
                symbol_entry["order"] = symbol_index
                if symbol_metrics:
                    symbol_entry[role_name] = {
                        "score": round(self._score_metrics(symbol_metrics), 4),
                        **symbol_metrics,
                    }
                arena_progress["symbols"][symbol] = symbol_entry

            set_arena_progress(arena_progress)

        set_arena_progress(arena_progress)

        challenger_metrics = self._evaluate_model(
            challenger_path,
            eval_symbols,
            horizon,
            engine=normalized_engine,
            role="challenger",
            progress_callback=publish_progress,
        )
        champion_metrics = (
            self._evaluate_model(
                champion_path,
                eval_symbols,
                horizon,
                engine=normalized_engine,
                role="champion",
                progress_callback=publish_progress,
            )
            if champion_path is not None
            else self._empty_metrics()
        )
        inverse_metrics = (
            self._evaluate_model(
                challenger_path,
                eval_symbols,
                horizon,
                engine=normalized_engine,
                role="inverse_challenger",
                progress_callback=publish_progress,
                action_transform=self._inverse_action,
            )
            if normalized_engine == "muzero"
            else self._empty_metrics()
        )

        challenger_score = self._score_metrics(challenger_metrics)
        champion_score = self._score_metrics(champion_metrics)
        inverse_score = self._score_metrics(inverse_metrics)
        min_score_edge = self._read_float_env("ARENA_MIN_SCORE_EDGE", 0.5)
        min_games = self._read_int_env("ARENA_MIN_GAMES", 12)
        min_symbols = min(
            len(eval_symbols),
            self._read_int_env("ARENA_MIN_SYMBOLS", 3),
        )
        try:
            challenger_directional_imbalance = float(challenger_metrics.get("directional_imbalance", 1.0))
        except (TypeError, ValueError):
            challenger_directional_imbalance = 1.0
        challenger_directional_ok = (
            int(challenger_metrics.get("long_entries", 0) or 0) > 0
            and int(challenger_metrics.get("short_entries", 0) or 0) > 0
            and challenger_directional_imbalance <= 0.75
            and not (
                str(challenger_metrics.get("directional_bias") or "inactive").strip().lower() != "balanced"
                and float(challenger_metrics.get("profit_factor", 0.0) or 0.0) < 1.0
            )
        )
        sample_size_ok = (
            int(challenger_metrics.get("evaluation_games", 0)) >= min_games
            and int(challenger_metrics.get("evaluation_symbols", 0)) >= min_symbols
        )
        inverse_edge = self._compute_inverse_edge(challenger_metrics, inverse_metrics)
        inverse_ok = True
        inverse_checks = {
            "profit_factor": float(challenger_metrics.get("profit_factor", 0.0) or 0.0)
            > float(inverse_metrics.get("profit_factor", 0.0) or 0.0),
            "return_pct": float(challenger_metrics.get("return_pct", 0.0) or 0.0)
            > float(inverse_metrics.get("return_pct", 0.0) or 0.0),
            "expectancy_pct": float(challenger_metrics.get("expectancy_pct", 0.0) or 0.0)
            > float(inverse_metrics.get("expectancy_pct", 0.0) or 0.0),
            "profitable_symbols": int(inverse_edge.get("edge_vs_inverse_profitable_symbols", 0))
            >= self._read_int_env("MUZERO_ARENA_INVERSE_MIN_PROFITABLE_SYMBOLS", 5),
        }
        if normalized_engine == "muzero":
            inverse_ok = all(inverse_checks.values())

        is_bootstrap = champion_path is None
        is_victory = (
            sample_size_ok
            and challenger_directional_ok
            and inverse_ok
            and (is_bootstrap or challenger_score > (champion_score + min_score_edge))
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "combat_type": (
                "DREAMER_HISTORICAL_ARENA"
                if normalized_engine == "dreamer"
                else "MUZERO_HISTORICAL_ARENA"
            ),
            "engine": normalized_engine,
            "horizon": horizon,
            "timeframe": timeframe,
            "family": challenger_metrics.get("family") or arena_progress.get("family"),
            "feature_profile": challenger_metrics.get("feature_profile") or arena_progress.get("feature_profile"),
            "mechanics_profile_version": challenger_metrics.get("mechanics_profile_version")
            or arena_progress.get("mechanics_profile_version"),
            "dataset_id": challenger_metrics.get("dataset_id"),
            "dataset_source": challenger_metrics.get("dataset_source"),
            "dataset_coverage": dict(challenger_metrics.get("dataset_coverage") or {}),
            "eval_symbols": eval_symbols,
            "validation": {
                "sample_size_ok": sample_size_ok,
                "directional_ok": challenger_directional_ok,
                "inverse_ok": inverse_ok,
                "inverse_checks": inverse_checks,
                "min_games": min_games,
                "min_symbols": min_symbols,
                "score_edge_required": min_score_edge,
            },
            "challenger": {
                "id": challenger_id,
                "path": str(challenger_path),
                "score": round(challenger_score, 4),
                "metrics": challenger_metrics,
            },
            "champion": {
                "id": champion_id,
                "path": str(champion_path) if champion_path is not None else None,
                "score": round(champion_score, 4),
                "metrics": champion_metrics,
            },
            "inverse_metrics": inverse_metrics,
            "inverse_challenger": {
                "id": f"{challenger_id}::inverse",
                "path": str(challenger_path),
                "score": round(inverse_score, 4),
                "metrics": inverse_metrics,
            },
            "reverse_candidate_only": bool(not inverse_ok),
            "outcome": "VICTORY" if is_victory else "DEFEAT",
            **inverse_edge,
            "action_required": (
                "BOOTSTRAP_CHAMPION"
                if is_bootstrap and is_victory
                else (
                    "HOT_SWAP_DEPLOY"
                    if is_victory
                    else (
                        "EXTEND_VALIDATION"
                        if not sample_size_ok
                        else (
                            "INVERSE_BEATS_CHALLENGER"
                            if not inverse_ok
                            else (
                                "REJECT_DIRECTIONAL_COLLAPSE"
                                if not challenger_directional_ok
                                else "KEEP_CURRENT"
                            )
                        )
                    )
                )
            ),
        }
        arena_progress["status"] = "completed"
        arena_progress["updated_at"] = datetime.now().isoformat()
        arena_progress["current_role"] = None
        arena_progress["current_symbol"] = None
        arena_progress["symbol_index"] = len(eval_symbols)
        arena_progress["symbol_total"] = len(eval_symbols)
        arena_progress["outcome"] = report["outcome"]
        arena_progress["action_required"] = report["action_required"]
        arena_progress["validation"] = report["validation"]
        arena_progress["mechanics_profile_version"] = report.get("mechanics_profile_version")
        arena_progress["dataset_coverage"] = dict(report.get("dataset_coverage") or {})
        arena_progress["challenger_score"] = round(challenger_score, 4)
        arena_progress["champion_score"] = round(champion_score, 4)
        arena_progress["inverse_challenger_score"] = round(inverse_score, 4)
        arena_progress["challenger"] = {
            "id": challenger_id,
            "path": str(challenger_path),
            "score": round(challenger_score, 4),
            "metrics": challenger_metrics,
        }
        arena_progress["champion"] = {
            "id": champion_id,
            "path": str(champion_path) if champion_path is not None else None,
            "score": round(champion_score, 4),
            "metrics": champion_metrics,
        }
        arena_progress["inverse_challenger"] = {
            "id": f"{challenger_id}::inverse",
            "path": str(challenger_path),
            "score": round(inverse_score, 4),
            "metrics": inverse_metrics,
        }

        final_symbols: dict[str, dict[str, Any]] = {}
        for order, symbol in enumerate(eval_symbols, start=1):
            symbol_entry: dict[str, Any] = {"symbol": symbol, "order": order}
            challenger_symbol = dict(challenger_metrics.get("metrics_by_symbol", {}).get(symbol) or {})
            champion_symbol = dict(champion_metrics.get("metrics_by_symbol", {}).get(symbol) or {})
            inverse_symbol = dict(inverse_metrics.get("metrics_by_symbol", {}).get(symbol) or {})
            if challenger_symbol:
                symbol_entry["challenger"] = {
                    "score": round(self._score_metrics(challenger_symbol), 4),
                    **challenger_symbol,
                }
            if champion_symbol:
                symbol_entry["champion"] = {
                    "score": round(self._score_metrics(champion_symbol), 4),
                    **champion_symbol,
                }
            if inverse_symbol:
                symbol_entry["inverse_challenger"] = {
                    "score": round(self._score_metrics(inverse_symbol), 4),
                    **inverse_symbol,
                }
            final_symbols[symbol] = symbol_entry
        arena_progress["symbols"] = final_symbols
        set_arena_progress(arena_progress)
        self.history.append(report)
        return report


if __name__ == "__main__":
    arena = Arena()
    print(arena.battle("muzero_global_latest", "gen_000_baseline", horizon="intraday"))
