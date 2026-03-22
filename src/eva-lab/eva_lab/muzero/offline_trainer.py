"""Entraine DreamerV3 hors ligne a partir des historiques de marche."""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.gold_cpu_prep import load_dreamer_replay_cache
from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.dreamer_networks import make_dreamer_networks
from eva_lab.muzero.dreamer_trainer import DreamerTrainerJAX
from eva_lab.muzero.replay_buffer import GameHistory, PrioritizedReplayBuffer
from eva_lab.shadow_dataset import load_shadow_games
from eva_lab.timescale_store import record_arena_result, record_training_dataset
from eva_lab.training_status import (
    append_training_log,
    load_training_status,
    mark_step_running,
    write_terminal_summary,
)
from eva_lab.training_utils import load_history_frame
from shared.indicators import IndicatorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfflineTrainer:
    """Charge les historiques et orchestre le pre-entrainement DreamerV3."""

    def __init__(self, data_dir: str = "data/history") -> None:
        """Initialise DreamerV3 avec le contexte V4 par famille et horizon.

        Args:
            data_dir (str): Dossier contenant les historiques CSV de secours.
        """
        self.data_dir = data_dir
        hidden_dims = [
            int(value.strip())
            for value in os.getenv("DREAMER_NETWORK_HIDDEN_DIMS", "256,256").split(",")
            if value.strip()
        ]
        self.sequence_length = int(os.getenv("DREAMER_SEQUENCE_LENGTH", "64"))
        self.sequence_stride = int(
            os.getenv("DREAMER_SEQUENCE_STRIDE", str(max(16, self.sequence_length // 2)))
        )
        self.replay_capacity = int(os.getenv("DREAMER_REPLAY_MAX_GAMES", "2500"))
        self.shadow_data_dirs = [
            item.strip()
            for item in os.getenv("DREAMER_SHADOW_DATA_DIRS", "data/shadow_learning").split(os.pathsep)
            if item.strip()
        ]
        self.horizon = str(
            os.getenv("DREAMER_HORIZON")
            or os.getenv("MUZERO_HORIZON")
            or os.getenv("DREAMER_DEFAULT_HORIZON")
            or "scalp"
        ).strip().lower()
        self.config = MuZeroConfigV3(
            horizon=self.horizon,
            model_family=os.getenv("MUZERO_MODEL_FAMILY", "") or None,
            batch_size=int(os.getenv("DREAMER_BATCH_SIZE", "8")),
            hidden_state_size=int(os.getenv("DREAMER_HIDDEN_STATE_SIZE", "128")),
            num_unroll_steps=int(os.getenv("DREAMER_NUM_UNROLL_STEPS", "3")),
            network_hidden_dims=hidden_dims or [256, 256],
        )
        self.config.dreamer_sequence_length = self.sequence_length
        self.config.dreamer_max_start_states = int(os.getenv("DREAMER_MAX_START_STATES", "256"))
        self.transformed = make_dreamer_networks(self.config)
        self.trainer = DreamerTrainerJAX(self.config, self.transformed)
        self.replay_buffer = PrioritizedReplayBuffer(max_games=self.replay_capacity)
        self.promoter = ChampionPromoter()
        self.weights_dir = Path(self.config.weights_path)
        self.results_dir = Path(self.config.results_path)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        sample_obs = np.zeros((1, *self.config.observation_shape))
        self.params, _ = self.trainer.init_params(sample_obs)
        self.trainer.params["wm"] = self.params

        self.family = str(getattr(self.config, "model_family", "") or "mixed")
        self.dataset_descriptor = dict(getattr(self.config, "dataset_descriptor", {}) or {})
        self.dataset_id = str(self.dataset_descriptor.get("dataset_id") or "")
        self.dataset_source = str(getattr(self.config, "dataset_source", "csv") or "csv")
        self.dataset_coverage = dict(getattr(self.config, "dataset_coverage", {}) or {})
        self.feature_profile = dict(getattr(self.config, "feature_profile", {}) or {})
        self.feature_profile_name = str(self.feature_profile.get("profile_name") or "dreamer_default")
        self.mechanics_profile = dict(getattr(self.config, "position_mechanics_profile", {}) or {})
        self.mechanics_profile_version = str(
            getattr(self.config, "mechanics_profile_version", "")
            or self.mechanics_profile.get("profile_version")
            or "v1"
        )
        self.ga_status = str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None
        self.ga_generation = (
            int(os.getenv("TRAINING_GA_GENERATION", "0"))
            if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
            else None
        )
        self.ga_trial = str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None
        self.trial_mode = str(os.getenv("TRAINING_TRIAL_MODE", "")).strip() or self.ga_status or "full"
        self.trial_cost_profile = (
            str(os.getenv("TRAINING_TRIAL_COST_PROFILE", "")).strip()
            or ("proxy" if self.trial_mode == "proxy_ga" else "full")
        )
        self.gate_profile = str(os.getenv("TRAINING_GATE_PROFILE", "")).strip() or "standard"
        self.focus_symbols = [
            str(symbol).strip()
            for symbol in list(getattr(self.config, "symbols", []) or [])
            if str(symbol).strip()
        ]
        self.replay_cache_key = (
            f"dreamer:{self.horizon}:{self.family}:{self.mechanics_profile_version}:{self.feature_profile_name}"
        )
        self.replay_cache_status = "cold_start"
        self.action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.total_steps = 0
        self.total_shadow_games = 0
        self.last_training_metrics: dict[str, Any] = {}

    def _load_prepared_replay_cache(self) -> bool:
        """Charge un replay Dreamer deja prepare par un job CPU.

        Returns:
            bool: ``True`` si un cache exploitable a ete charge.
        """
        payload = load_dreamer_replay_cache(
            horizon=self.horizon,
            family=self.family,
            symbols=list(getattr(self.config, "symbols", []) or []),
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
        )
        if not payload:
            return False

        games = [
            game
            for game in list(payload.get("games") or [])
            if len(game) >= self.sequence_length
        ]
        if not games:
            logger.warning("Cache Dreamer detecte mais inutilisable pour la sequence courante.")
            return False

        for game in games:
            self.replay_buffer.save_game(game)

        counts = dict(payload.get("action_counts") or {})
        self.action_counts = {
            "BUY": int(counts.get("BUY", 0) or 0),
            "SELL": int(counts.get("SELL", 0) or 0),
            "HOLD": int(counts.get("HOLD", 0) or 0),
        }
        self.total_steps = int(payload.get("total_steps") or sum(len(game) for game in games))
        self.total_shadow_games = int(payload.get("shadow_games") or 0)
        self.replay_cache_status = "cpu_prepared"

        cache_path = str(payload.get("cache_path") or "")
        logger.info(
            "Replay Dreamer charge depuis le cache CPU %s (%s episodes).",
            cache_path,
            len(games),
        )
        append_training_log(
            f"Dreamer: replay CPU charge depuis {Path(cache_path).name}.",
            source="dreamer",
        )
        return True

    def _build_status_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Construit le socle de statut commun pour DreamerV3.

        Args:
            **overrides (Any): Surcharges specifiques a l'etape courante.

        Returns:
            dict[str, Any]: Arguments compatibles avec ``mark_step_running``.
        """
        payload: dict[str, Any] = {
            "engine": "dreamer",
            "horizon": self.horizon,
            "family": self.family,
            "dataset_id": self.dataset_id or None,
            "dataset_source": self.dataset_source or None,
            "feature_profile": self.feature_profile_name or None,
            "mechanics_profile_version": self.mechanics_profile_version or None,
            "focus_symbols": self.focus_symbols,
            "gate_profile": self.gate_profile,
            "ga_status": self.ga_status,
            "ga_generation": self.ga_generation,
            "ga_trial": self.ga_trial,
            "trial_mode": self.trial_mode,
            "trial_cost_profile": self.trial_cost_profile,
            "replay_cache_status": self.replay_cache_status,
            "replay_cache_key": self.replay_cache_key,
            "replay_cache_entries": self.replay_buffer.size,
            "replay_cache_source": "local_disk",
            "shadow_buffer_size": self.replay_buffer.size,
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "world_model_steps": int(self.last_training_metrics.get("world_model_steps", 0) or 0),
            "dataset_coverage": self.dataset_coverage,
        }
        payload.update(overrides)
        return payload

    def _iter_history_frames(self) -> Iterable[tuple[str, pd.DataFrame]]:
        """Retourne les historiques compatibles pour la famille courante.

        Yields:
            Iterable[tuple[str, pd.DataFrame]]: Couple symbole/historique exploitable.
        """
        timeframe = str(self.config.primary_timeframe or "M5").upper()
        for symbol in list(getattr(self.config, "symbols", []) or []):
            frame = load_history_frame(symbol, timeframe, data_dir=self.data_dir)
            if frame is None or frame.empty:
                logger.warning("Historique Dreamer indisponible pour %s sur %s.", symbol, timeframe)
                continue
            yield symbol, frame.copy()

    @staticmethod
    def _compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs utiles au pre-entrainement Dreamer.

        Args:
            frame (pd.DataFrame): Historique brut OHLCV.

        Returns:
            pd.DataFrame: Historique enrichi des indicateurs utilises.
        """
        enriched = frame.copy()
        enriched["rsi"] = IndicatorFactory.rsi(enriched["close"], 14)
        macd_res = IndicatorFactory.macd(enriched["close"])
        enriched["macd"] = macd_res["macd"]
        enriched["macd_signal"] = macd_res["signal"]
        enriched["macd_hist"] = macd_res["histogram"]
        enriched["vwap"] = IndicatorFactory.vwap(
            enriched["high"],
            enriched["low"],
            enriched["close"],
            enriched["tick_volume"],
        )
        enriched["obv"] = IndicatorFactory.obv(enriched["close"], enriched["tick_volume"])
        enriched["momentum"] = IndicatorFactory.momentum(enriched["close"])
        enriched["trix"] = IndicatorFactory.trix(enriched["close"])
        stoch_res = IndicatorFactory.stochastic(enriched["high"], enriched["low"], enriched["close"])
        enriched["stoch_k"] = stoch_res["percent_k"]
        enriched["stoch_d"] = stoch_res["percent_d"]
        enriched["cci"] = IndicatorFactory.cci(enriched["high"], enriched["low"], enriched["close"])
        adx_res = IndicatorFactory.adx(enriched["high"], enriched["low"], enriched["close"])
        enriched["adx"] = adx_res["adx"]
        enriched["adx_plus_di"] = adx_res["plus_di"]
        enriched["adx_minus_di"] = adx_res["minus_di"]
        ichi_res = IndicatorFactory.ichimoku(enriched["high"], enriched["low"], enriched["close"])
        enriched["ichi_tenkan"] = ichi_res["tenkan_sen"]
        enriched["ichi_kijun"] = ichi_res["kijun_sen"]
        enriched["ichi_senkou_a"] = ichi_res["senkou_span_a"]
        enriched["ichi_senkou_b"] = ichi_res["senkou_span_b"]
        return enriched.bfill().fillna(0.0)

    def load_and_process_data(self) -> None:
        """Charge les historiques, calcule les indicateurs et construit les episodes."""
        symbols = list(getattr(self.config, "symbols", []) or [])
        logger.info("Dreamer V4 charge %s symboles sur %s.", len(symbols), self.horizon)
        append_training_log(
            f"Dreamer {self.horizon}: chargement de {len(symbols)} symboles pour la famille {self.family}.",
            source="dreamer",
        )
        mark_step_running(
            "dreamer_offline",
            phase="chargement_donnees",
            symbol_total=len(symbols),
            **self._build_status_kwargs(),
        )

        if self._load_prepared_replay_cache():
            logger.info(
                "Dreamer V4 reutilise un replay CPU prepare (%s episodes).",
                self.replay_buffer.size,
            )
            return

        total_steps = 0
        for symbol_index, (symbol, frame) in enumerate(self._iter_history_frames(), start=1):
            logger.info("Traitement Dreamer de %s (%s/%s).", symbol, symbol_index, len(symbols))
            append_training_log(
                f"Dreamer {self.horizon}: preparation de {symbol} ({symbol_index}/{len(symbols)}).",
                source="dreamer",
            )
            mark_step_running(
                "dreamer_offline",
                phase="collecte",
                symbol=symbol,
                symbol_index=symbol_index,
                symbol_total=len(symbols),
                **self._build_status_kwargs(),
            )

            try:
                df = self._compute_indicators(frame)
            except Exception as exc:
                logger.error("Calcul des indicateurs impossible pour %s: %s", symbol, exc)
                continue

            segment_length = self.sequence_length
            closes_seg = df["close"].values
            for start_idx in range(0, len(df) - segment_length, self.sequence_stride):
                end_idx = start_idx + segment_length
                if end_idx > len(df):
                    break

                seg_closes = closes_seg[start_idx:end_idx]
                game = GameHistory()
                actions = np.random.choice([0, 1, 2], size=segment_length, p=[0.35, 0.325, 0.325])
                initial_balance = 10000.0
                balance = initial_balance
                peak_balance = initial_balance
                position = 0
                entry_price = 0.0

                for index_in_segment in range(segment_length):
                    idx = start_idx + index_in_segment
                    price = seg_closes[index_in_segment]
                    obs_vec = np.zeros(self.config.observation_shape)
                    obs_vec[0] = price / 3000.0
                    obs_vec[1] = df["rsi"].values[idx] / 100.0
                    features_list = [
                        df["rsi"].values[idx],
                        df["macd_hist"].values[idx],
                        df["macd_signal"].values[idx],
                        df["vwap"].values[idx],
                        df["obv"].values[idx] / 10000.0,
                        df["momentum"].values[idx],
                        df["trix"].values[idx],
                        df["stoch_k"].values[idx],
                        df["stoch_d"].values[idx],
                        df["cci"].values[idx],
                        df["adx"].values[idx],
                        df["adx_plus_di"].values[idx],
                        df["adx_minus_di"].values[idx],
                        df["ichi_tenkan"].values[idx],
                        df["ichi_kijun"].values[idx],
                        df["ichi_senkou_a"].values[idx],
                        df["ichi_senkou_b"].values[idx],
                    ]
                    for feature_index, feature_value in enumerate(features_list):
                        if feature_index + 2 < self.config.observation_shape[0]:
                            obs_vec[feature_index + 2] = feature_value

                    action_val = int(actions[index_in_segment])
                    reward = 0.0
                    if action_val == 1:
                        self.action_counts["BUY"] += 1
                    elif action_val == 2:
                        self.action_counts["SELL"] += 1
                    else:
                        self.action_counts["HOLD"] += 1

                    if index_in_segment < segment_length - 1:
                        next_price = seg_closes[index_in_segment + 1]
                        ret = (next_price - price) / max(price, 1e-9) * 100
                        if action_val == 1:
                            reward = ret - 0.02
                            if position == 0:
                                position = 1
                                entry_price = price
                        elif action_val == 2:
                            reward = -ret - 0.02
                            if position == 0:
                                position = -1
                                entry_price = price
                        elif action_val == 0 and position != 0:
                            trade_pnl = (
                                (price - entry_price) / max(entry_price, 1e-9) * 100
                                if position == 1
                                else (entry_price - price) / max(entry_price, 1e-9) * 100
                            )
                            balance += balance * trade_pnl / 100
                            position = 0

                        peak_balance = max(peak_balance, balance)
                        drawdown_pct = (peak_balance - balance) / max(peak_balance, 1e-9) * 100
                        if drawdown_pct >= 4.0:
                            reward -= 15.0

                    action_one_hot = np.zeros(self.config.action_space_size)
                    action_one_hot[action_val] = 1.0
                    game.store(obs_vec, action_one_hot, reward, [1 / 3] * 3, 0.0)

                self.replay_buffer.save_game(game)
                total_steps += segment_length

        shadow_games = load_shadow_games(
            self.shadow_data_dirs,
            observation_size=self.config.observation_shape[0],
            action_space_size=self.config.action_space_size,
        )
        for game in shadow_games:
            if len(game) < self.sequence_length:
                logger.info(
                    "Episode shadow ignore car trop court pour Dreamer: %s < %s.",
                    len(game),
                    self.sequence_length,
                )
                continue
            self.replay_buffer.save_game(game)
            total_steps += len(game)
        self.total_shadow_games = len(shadow_games)
        self.total_steps = total_steps
        self.replay_cache_status = "warm" if self.replay_buffer.size > 0 else "empty"

        logger.info(
            "Episodes hors-ligne charges: %s (%s pas de temps).",
            self.replay_buffer.size,
            total_steps,
        )
        append_training_log(
            f"Dreamer: {self.replay_buffer.size} episodes charges pour {total_steps} pas de temps.",
            source="dreamer",
        )

    def train_loop(self, epochs: int = 5000) -> dict[str, Any]:
        """Execute la boucle d'optimisation Dreamer.

        Args:
            epochs (int): Nombre d'epochs a executer.

        Returns:
            dict[str, Any]: Resume simple d'optimisation.

        Raises:
            RuntimeError: Si le buffer Dreamer est vide.
        """
        logger.info("Demarrage de l'entrainement hors-ligne DreamerV3...")
        append_training_log(
            f"Dreamer: demarrage de l'entrainement sur {epochs} epochs.",
            source="dreamer",
        )
        if self.replay_buffer.size == 0:
            raise RuntimeError("Le buffer Dreamer est vide. Aucun historique n'a ete charge.")

        avg_loss = 0.0
        total_updates = 0
        for epoch in range(epochs):
            mark_step_running(
                "dreamer_offline",
                phase="optimisation",
                epoch_current=epoch + 1,
                epoch_total=epochs,
                **self._build_status_kwargs(world_model_steps=total_updates),
            )
            loss_sum = 0.0
            steps = 0
            effective_batch_size = min(self.config.batch_size, self.replay_buffer.size)
            updates_per_epoch = max(1, self.replay_buffer.size // max(effective_batch_size, 1))
            for _ in range(updates_per_epoch):
                samples = self.replay_buffer.sample(effective_batch_size)
                games = [
                    game
                    for game, _, _ in samples
                    if len(game) >= self.sequence_length
                ]
                if not games:
                    raise RuntimeError(
                        "Aucune sequence Dreamer exploitable n'a ete echantillonnee pour le batch courant."
                    )
                batch = self.trainer.prepare_batch(games)
                metrics = self.trainer.train_step(batch)
                loss_sum += float(metrics["loss_total"])
                steps += 1
                total_updates += 1
            avg_loss = loss_sum / steps if steps > 0 else 0.0
            logger.info("Epoch %s/%s - Loss: %.4f", epoch + 1, epochs, avg_loss)
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                append_training_log(
                    f"Dreamer: epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}",
                    source="dreamer",
                )

        self.last_training_metrics = {
            "epochs": epochs,
            "avg_loss": avg_loss,
            "world_model_steps": total_updates,
            "buffer_size": self.replay_buffer.size,
        }
        return dict(self.last_training_metrics)

    def save_checkpoint(self, path: str) -> str:
        """Sauvegarde un checkpoint local Dreamer.

        Args:
            path (str): Prefixe du fichier de sortie sans extension.

        Returns:
            str: Chemin final du checkpoint serialise.
        """
        target = Path(path).with_suffix(".pkl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as file_obj:
            pickle.dump(self.trainer.params, file_obj)
        logger.info("Checkpoint Dreamer sauvegarde dans %s", target)
        return str(target)

    def _build_activity_metrics(self) -> dict[str, Any]:
        """Construit des metriques minimales de direction et d'activite.

        Returns:
            dict[str, Any]: Metriques minimales exploitables pour le resume terminal.
        """
        total_actions = sum(self.action_counts.values())
        long_entries = float(self.action_counts["BUY"])
        short_entries = float(self.action_counts["SELL"])
        long_share = long_entries / total_actions if total_actions else 0.0
        short_share = short_entries / total_actions if total_actions else 0.0
        directional_imbalance = abs(long_share - short_share)
        if total_actions == 0:
            directional_bias = "inactive"
        elif long_share >= 0.8:
            directional_bias = "buy_heavy"
        elif short_share >= 0.8:
            directional_bias = "sell_heavy"
        else:
            directional_bias = "balanced"

        return {
            "family": self.family,
            "feature_profile": self.feature_profile_name,
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "mechanics_profile_version": self.mechanics_profile_version,
            "dataset_coverage": self.dataset_coverage,
            "evaluation_games": 0.0,
            "evaluation_symbols": float(len(getattr(self.config, "symbols", []) or [])),
            "win_rate": 0.0,
            "return_pct": 0.0,
            "net_realized_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0.0,
            "expectancy_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "positive_episode_rate": 0.0,
            "long_entries": long_entries,
            "short_entries": short_entries,
            "long_entry_share": round(long_share, 6),
            "short_entry_share": round(short_share, 6),
            "directional_imbalance": round(directional_imbalance, 6),
            "directional_bias": directional_bias,
            "metrics_by_position_mechanics": {
                "split_efficiency": 0.0,
                "pyramid_efficiency": 0.0,
                "slbe_capture_rate": 0.0,
                "hold_drag_score": 1.0,
                "close_quality_score": 0.0,
                "split_executed": 0.0,
                "pyramids_opened": 0.0,
                "slbe_triggered": 0.0,
                "close_winner_count": 0.0,
                "close_loser_count": 0.0,
            },
        }

    def _build_terminal_summary(
        self,
        *,
        terminal_status: str,
        training_metrics: dict[str, Any] | None = None,
        failed_step: str | None = None,
        failure_mode: str | None = None,
        reason: str | None = None,
        latest_checkpoint: str | None = None,
        challenger_id: str | None = None,
        challenger_checkpoint: str | None = None,
        battle_report: dict[str, Any] | None = None,
        promotion_gate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construit un resume terminal Dreamer lie au ``run_id`` courant.

        Args:
            terminal_status (str): Statut terminal du run Dreamer.
            training_metrics (dict[str, Any] | None): Resume d'optimisation si disponible.
            failed_step (str | None): Etape terminale en erreur ou blocage.
            failure_mode (str | None): Cause principale normalisee.
            reason (str | None): Message humain de la terminaison.
            latest_checkpoint (str | None): Dernier checkpoint courant.
            challenger_id (str | None): Identifiant du candidat genere.
            challenger_checkpoint (str | None): Checkpoint du candidat.
            battle_report (dict[str, Any] | None): Rapport Arena si disponible.
            promotion_gate (dict[str, Any] | None): Verdict de promotion si disponible.

        Returns:
            dict[str, Any]: Resume terminal complet et autoporteur.
        """

        run_status = load_training_status()
        run_id = str(run_status.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("Resume terminal Dreamer impossible sans run_id actif.")

        challenger_metrics = dict(
            ((battle_report or {}).get("challenger") or {}).get("metrics") or self._build_activity_metrics()
        )
        metrics_by_symbol = dict(challenger_metrics.get("metrics_by_symbol") or {})
        mechanics = dict(challenger_metrics.get("metrics_by_position_mechanics") or {})
        latest_verdict_status = (
            str((promotion_gate or {}).get("status") or "").strip()
            or ("completed" if terminal_status == "completed" else terminal_status)
        )
        latest_verdict_reason = (
            str((promotion_gate or {}).get("reason") or "").strip()
            or reason
            or ("dreamer_validation_pending" if terminal_status == "completed" else None)
        )

        return {
            "run_id": run_id,
            "sequence_id": str(run_status.get("sequence_id") or "").strip() or None,
            "sequence_profile": str(run_status.get("sequence_profile") or "").strip() or None,
            "window_id": str(run_status.get("window_id") or "").strip() or None,
            "trial_id": str(run_status.get("trial_id") or "").strip() or self.ga_trial,
            "engine": "dreamer",
            "horizon": self.horizon,
            "family": self.family,
            "feature_profile": self.feature_profile_name,
            "mechanics_profile_version": self.mechanics_profile_version,
            "focus_symbols": self.focus_symbols,
            "gate_profile": self.gate_profile,
            "ga_status": self.ga_status,
            "ga_generation": self.ga_generation,
            "ga_trial": self.ga_trial,
            "trial_mode": self.trial_mode,
            "trial_cost_profile": self.trial_cost_profile,
            "dataset_id": self.dataset_id or None,
            "dataset_source": self.dataset_source or None,
            "dataset_coverage": self.dataset_coverage,
            "terminal_status": terminal_status,
            "failed_step": failed_step,
            "failure_mode": failure_mode,
            "reason": reason,
            "arena_outcome": (battle_report or {}).get("outcome"),
            "promotion_gate": promotion_gate,
            "metrics": challenger_metrics,
            "metrics_by_symbol": metrics_by_symbol,
            "metrics_by_position_mechanics": mechanics,
            "artifact_state": {
                "latest_checkpoint_present": bool(latest_checkpoint),
                "candidate_checkpoint_present": bool(challenger_checkpoint),
                "battle_report_present": battle_report is not None,
                "promotion_gate_present": bool(promotion_gate),
            },
            "latest_checkpoint": latest_checkpoint,
            "latest_candidate": challenger_id,
            "challenger_path": challenger_checkpoint,
            "latest_verdict": {
                "status": latest_verdict_status,
                "reason": latest_verdict_reason,
                "failure_mode": failure_mode,
            },
            "training_metrics": dict(training_metrics or {}),
        }

    @staticmethod
    def _classify_terminal_failure(exc: Exception) -> tuple[str, str, str]:
        """Mappe une exception Dreamer vers un statut terminal normalise.

        Args:
            exc (Exception): Exception terminale observee.

        Returns:
            tuple[str, str, str]: ``(terminal_status, failure_mode, failed_step)``.
        """

        message = str(exc or "").strip().lower()
        if (
            "buffer dreamer est vide" in message
            or "aucune sequence dreamer exploitable" in message
            or "aucun historique" in message
        ):
            return "blocked", "insufficient_sample", "dreamer_offline"
        if "homogene" in message or isinstance(exc, ValueError):
            return "error", "batch_invalid", "prepare_batch"
        return "error", "dreamer_pipeline_error", "dreamer_offline"

    def _write_failed_summary(
        self,
        *,
        exc: Exception,
        training_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ecrit un resume terminal Dreamer en cas de blocage ou d'erreur.

        Args:
            exc (Exception): Exception terminale capturee.
            training_metrics (dict[str, Any] | None): Metriques partielles si disponibles.

        Returns:
            dict[str, Any]: Resume terminal persiste.
        """

        terminal_status, failure_mode, failed_step = self._classify_terminal_failure(exc)
        summary = self._build_terminal_summary(
            terminal_status=terminal_status,
            training_metrics=training_metrics or self.last_training_metrics,
            failed_step=failed_step,
            failure_mode=failure_mode,
            reason=str(exc),
        )
        path = write_terminal_summary(summary)
        append_training_log(
            f"Dreamer: resume terminal {terminal_status} ecrit dans {path.name}.",
            level="WARNING" if terminal_status == "blocked" else "ERROR",
            source="dreamer",
        )
        return summary

    def finalize_run(self, training_metrics: dict[str, Any]) -> dict[str, Any]:
        """Ecrit les artefacts V4 d'un candidat Dreamer.

        Args:
            training_metrics (dict[str, Any]): Resume d'optimisation courant.

        Returns:
            dict[str, Any]: Rapport final ecrit sur disque.
        """
        latest_prefix = self.weights_dir / f"dreamer_{self.horizon}_{self.family}_latest"
        latest_checkpoint = self.save_checkpoint(str(latest_prefix))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        challenger_id = f"gen_dreamer_{self.horizon}_{timestamp}"
        challenger_checkpoint = self.save_checkpoint(str(self.weights_dir / challenger_id))
        promotion_result = {
            "status": "skipped",
            "reason": "dreamer_validation_pending",
            "engine": "dreamer",
            "horizon": self.horizon,
            "source_path": challenger_checkpoint,
            "champion_paths": [],
            "promotion_gate": None,
        }
        report_payload = {
            "engine": "dreamer",
            "horizon": self.horizon,
            "family": self.family,
            "feature_profile": self.feature_profile_name,
            "mechanics_profile_version": self.mechanics_profile_version,
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "focus_symbols": self.focus_symbols,
            "gate_profile": self.gate_profile,
            "dataset_descriptor": self.dataset_descriptor,
            "dataset_coverage": self.dataset_coverage,
            "latest_checkpoint": latest_checkpoint,
            "challenger_path": challenger_checkpoint,
            "shadow_buffer_size": self.replay_buffer.size,
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "world_model_steps": training_metrics.get("world_model_steps", 0),
            "ga_status": self.ga_status,
            "ga_generation": self.ga_generation,
            "ga_trial": self.ga_trial,
            "battle_report": None,
            "promotion": promotion_result,
            "training_metrics": training_metrics,
        }
        report_path = self.promoter.get_arena_report_path(self.horizon, engine="dreamer")
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_payload = {
            "status": "candidate_only",
            "promoted_at": None,
            "engine": "dreamer",
            "horizon": self.horizon,
            "family": self.family,
            "feature_profile": self.feature_profile_name,
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "mechanics_profile_version": self.mechanics_profile_version,
            "dataset_coverage": self.dataset_coverage,
            "focus_symbols": self.focus_symbols,
            "gate_profile": self.gate_profile,
            "selection_policy": "champion_only",
            "engine_label": self.promoter.get_engine_label("dreamer", variant="blocked"),
            "challenger_id": challenger_id,
            "source_path": challenger_checkpoint,
            "latest_checkpoint": latest_checkpoint,
            "champion_path": None,
            "battle_report": None,
            "training_metrics": training_metrics,
            "promotion_gate": None,
        }
        manifest_path = self.promoter.get_manifest_path(self.horizon, engine="dreamer")
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        record_arena_result(
            report_payload,
            metadata={
                "engine": "dreamer",
                "run_trigger": str(os.getenv("TRAINING_RUN_TRIGGER", "manual") or "manual"),
                "ga_status": self.ga_status,
                "ga_generation": self.ga_generation,
                "ga_trial": self.ga_trial,
            },
        )
        terminal_summary = self._build_terminal_summary(
            terminal_status="completed",
            training_metrics=training_metrics,
            latest_checkpoint=latest_checkpoint,
            challenger_id=challenger_id,
            challenger_checkpoint=challenger_checkpoint,
            reason="dreamer_validation_pending",
        )
        terminal_summary_path = write_terminal_summary(terminal_summary)
        append_training_log(
            f"Dreamer: resume terminal complet ecrit dans {terminal_summary_path.name}.",
            source="dreamer",
        )
        report_payload["terminal_summary_path"] = str(terminal_summary_path)
        return report_payload


def main() -> dict[str, Any]:
    """Execute le pipeline DreamerV3 V4 complet.

    Returns:
        dict[str, Any]: Rapport final du candidat Dreamer genere.
    """
    epochs = int(os.getenv("DREAMER_EPOCHS", "5000"))
    trainer = OfflineTrainer()
    record_training_dataset(
        dict(trainer.dataset_descriptor),
        metadata={
            "engine": "dreamer",
            "run_trigger": str(os.getenv("TRAINING_RUN_TRIGGER", "manual") or "manual"),
            "ga_status": trainer.ga_status,
            "ga_generation": trainer.ga_generation,
            "ga_trial": trainer.ga_trial,
        },
    )
    try:
        trainer.load_and_process_data()
        metrics = trainer.train_loop(epochs=epochs)
        return trainer.finalize_run(metrics)
    except Exception as exc:
        summary = trainer._write_failed_summary(exc=exc)
        if summary.get("terminal_status") == "blocked":
            logger.warning("DreamerV3 bloque proprement: %s", exc)
            return summary
        logger.exception("DreamerV3 en erreur: %s", exc)
        raise


if __name__ == "__main__":
    summary = main()
    logger.info("DreamerV3 termine: %s", summary.get("challenger_path"))
