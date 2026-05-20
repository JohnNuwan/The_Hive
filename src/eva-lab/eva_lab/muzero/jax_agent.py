"""Agent MuZero JAX pour le self-play, l'entrainement et la persistence."""

from __future__ import annotations

import faulthandler
import logging
import os
import re
import signal
import threading
from datetime import datetime
from time import perf_counter
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from eva_lab.muzero.checkpoint_utils import (
    MuZeroCheckpointCompatibilityError,
    build_muzero_expected_context,
    inspect_muzero_checkpoint,
    save_muzero_checkpoint,
)
from eva_lab.muzero.environment import TradingEnvironment
from eva_lab.muzero.jax_mcts import JAXMuZeroMCTS
from eva_lab.muzero.jax_networks import make_muzero_networks
from eva_lab.muzero.jax_trainer import MuZeroTrainerJAX, TrainingBatch
from eva_lab.muzero.replay_buffer import GameHistory, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class CollectionStepTimeoutError(TimeoutError):
    """Signale qu'un pas de collecte MuZero a depasse le budget autorise."""


class PreparedTrainingStep(NamedTuple):
    """Lot d'entrainement deja prepare et transfere vers le device cible."""

    batch: TrainingBatch
    tree_indices: tuple[int, ...]
    batch_prepare_ms: float
    device_put_ms: float


class JAXMuZeroAgent:
    """Pilote les reseaux MuZero, le buffer de replay et le self-play."""

    def __init__(self, config):
        """Initialise les reseaux, l'optimiseur et les hooks JIT."""
        self.config = config
        self.transformed = make_muzero_networks(config)
        self.initial_apply, self.recurrent_apply = self.transformed.apply
        self.trainer = MuZeroTrainerJAX(config, self.transformed)

        dummy_obs = jnp.zeros((1, *config.observation_shape))
        self.params, self.opt_state = self.trainer.init_params(dummy_obs)

        self.replay_buffer = PrioritizedReplayBuffer(
            max_games=config.window_size // config.max_moves
        )
        self.training_step_count = 0
        self.last_reanalyze_positions_count = 0
        self.last_reanalyze_num_simulations = int(
            getattr(config, "reanalyze_num_simulations", config.num_simulations) or config.num_simulations
        )

        self._jit_init = jax.jit(self._initial_inference)
        self._jit_rec = jax.jit(self._recurrent_inference)
        self._expected_checkpoint_context = build_muzero_expected_context(
            config=config,
            expected_params=self.params,
        )

        logger.info(
            "[JAXMuZeroAgent] Agent operationnel. Etat latent=%s.",
            config.hidden_state_size,
        )

    @staticmethod
    def _extract_training_step_from_checkpoint_path(path: str) -> int | None:
        """Extrait l'etape numerique a partir du nom de checkpoint.

        Args:
            path (str): Chemin du checkpoint charge.

        Returns:
            int | None: Etape detectee si le suffixe ``_ckpt_<n>`` est present.
        """
        match = re.search(r"_ckpt_(\d+)", str(path or ""))
        if match is None:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _run_collection_step_with_timeout(
        self,
        step_callback: Callable[[], tuple[object, float, bool, float, float, int, np.ndarray]],
        *,
        symbol: str,
        step_index: int,
        timeout_seconds: float,
    ) -> tuple[object, float, bool, float, float, int, np.ndarray]:
        """Execute un pas de collecte avec un timeout interruptible.

        Args:
            step_callback (Callable[[], tuple[object, float, bool, float, float, int, np.ndarray]]):
                Fonction qui execute le pas complet et retourne
                ``(next_obs, reward, done, value, mcts_elapsed_seconds, action, policy)``.
            symbol (str): Symbole de l'episode courant.
            step_index (int): Index humain du pas courant.
            timeout_seconds (float): Budget maximal autorise pour le pas.

        Returns:
            tuple[object, float, bool, float, float, int, np.ndarray]: Resultat du pas.

        Raises:
            CollectionStepTimeoutError: Si le pas depasse le budget autorise.
        """
        effective_timeout = max(0.0, float(timeout_seconds or 0.0))
        if (
            effective_timeout <= 0.0
            or os.name == "nt"
            or not hasattr(signal, "setitimer")
            or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()
        ):
            return step_callback()

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

        def _handle_timeout(_signum, _frame):
            raise CollectionStepTimeoutError(
                f"Pas MuZero depasse sur {symbol} au step {step_index}."
            )

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, effective_timeout)
        faulthandler.dump_traceback_later(effective_timeout, repeat=False)
        try:
            return step_callback()
        finally:
            faulthandler.cancel_dump_traceback_later()
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            signal.signal(signal.SIGALRM, previous_handler)

    def _initial_inference(self, params, observation):
        """Execute l'inference initiale MuZero sur une observation brute."""
        return self.initial_apply(params, None, observation)

    def _recurrent_inference(self, params, hidden_state, action_onehot):
        """Execute l'inference recurrente MuZero sur un etat latent."""
        return self.recurrent_apply(params, None, hidden_state, action_onehot)

    def play_game(
        self,
        env,
        exploration: bool = True,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        action_transform: Callable[[int], int] | None = None,
    ) -> GameHistory:
        """Joue une partie complete dans l'environnement et alimente le replay buffer.

        Args:
            env (TradingEnvironment): Environnement de self-play courant.
            exploration (bool): Active le bruit d'exploration a la racine.
            progress_callback (Callable[[dict[str, object]], None] | None):
                Callback appele periodiquement pour exposer un heartbeat
                intra-partie pendant la collecte.
            action_transform (Callable[[int], int] | None): Transformation
                optionnelle appliquee a l'action choisie avant execution.

        Returns:
            GameHistory: Episode collecte, eventuellement tronque si un
                garde-fou de temps coupe la partie.
        """
        game = GameHistory()
        obs, _ = env.reset()
        done = False
        steps = 0
        episode_started_at = perf_counter()
        last_heartbeat_at = episode_started_at
        heartbeat_every_steps = max(
            1,
            int(getattr(self.config, "collection_heartbeat_every_steps", 25) or 25),
        )
        heartbeat_every_seconds = max(
            1.0,
            float(getattr(self.config, "collection_heartbeat_every_seconds", 30.0) or 30.0),
        )
        max_episode_seconds = max(
            0.0,
            float(
                getattr(
                    env,
                    "collection_max_episode_seconds",
                    getattr(self.config, "collection_max_episode_seconds", 0.0),
                )
                or 0.0
            ),
        )
        max_step_seconds = max(
            0.0,
            float(getattr(self.config, "collection_max_step_seconds", 0.0) or 0.0),
        )
        collection_num_simulations = int(
            getattr(
                env,
                "collection_num_simulations",
                getattr(self.config, "collection_num_simulations", self.config.num_simulations),
            )
            or getattr(self.config, "collection_num_simulations", self.config.num_simulations)
        )
        max_episode_moves = int(
            getattr(
                env,
                "max_steps_per_episode",
                getattr(self.config, "collection_max_moves", self.config.max_moves),
            )
            or getattr(self.config, "collection_max_moves", self.config.max_moves)
        )
        mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))

        while not done and steps < max_episode_moves:
            steps += 1
            step_started_at = perf_counter()
            current_observation = obs

            def _execute_collection_step() -> tuple[object, float, bool, float, float]:
                obs_jax = jnp.array(current_observation).reshape(1, -1)
                hidden_state, root_logits, root_value_logits = self._jit_init(self.params, obs_jax)
                root_legal_actions = env.get_root_policy_actions()

                mcts_started_at = perf_counter()
                root = mcts.run(
                    hidden_state,
                    root_logits,
                    root_value_logits,
                    root_legal_actions=root_legal_actions,
                    add_exploration_noise=exploration,
                    num_simulations=collection_num_simulations,
                )
                mcts_elapsed_seconds = perf_counter() - mcts_started_at

                action = self._select_action(root, exploration)
                if action_transform is not None:
                    action = int(action_transform(int(action)))
                policy = self._get_policy_distribution(root)
                value = float(root.value)
                next_obs, reward, done_flag, _, _ = env.step(action)
                return next_obs, reward, done_flag, value, mcts_elapsed_seconds, action, policy

            try:
                (
                    next_obs,
                    reward,
                    done,
                    value,
                    mcts_elapsed_seconds,
                    action,
                    policy,
                ) = self._run_collection_step_with_timeout(
                    _execute_collection_step,
                    symbol=str(getattr(env, "symbol", "unknown")),
                    step_index=steps,
                    timeout_seconds=max_step_seconds,
                )
            except CollectionStepTimeoutError as exc:
                logger.exception(
                    "Collecte MuZero interrompue sur %s au step %s: %s",
                    getattr(env, "symbol", "unknown"),
                    steps,
                    exc,
                )
                game.metadata["stopped_reason"] = "depassement_temps_pas_interruptible"
                game.metadata["stopped_step"] = int(steps)
                game.metadata["step_elapsed_seconds"] = float(perf_counter() - step_started_at)
                break

            game.store(current_observation, action, reward, policy, value)
            obs = next_obs

            now = perf_counter()
            episode_elapsed_seconds = now - episode_started_at
            step_elapsed_seconds = now - step_started_at
            should_emit_heartbeat = (
                done
                or steps == 1
                or steps % heartbeat_every_steps == 0
                or (now - last_heartbeat_at) >= heartbeat_every_seconds
            )
            if should_emit_heartbeat:
                heartbeat_payload = {
                    "symbol": str(getattr(env, "symbol", "unknown")),
                    "steps": int(steps),
                    "max_moves": int(getattr(env, "max_steps_per_episode", self.config.max_moves)),
                    "elapsed_seconds": float(episode_elapsed_seconds),
                    "step_elapsed_seconds": float(step_elapsed_seconds),
                    "mcts_elapsed_seconds": float(mcts_elapsed_seconds),
                    "done": bool(done),
                }
                logger.info(
                    "[collecte:%s] heartbeat partie | step=%s/%s | episode=%.1fs | step=%.3fs | mcts=%.3fs | sims=%s | done=%s",
                    heartbeat_payload["symbol"],
                    heartbeat_payload["steps"],
                    heartbeat_payload["max_moves"],
                    heartbeat_payload["elapsed_seconds"],
                    heartbeat_payload["step_elapsed_seconds"],
                    heartbeat_payload["mcts_elapsed_seconds"],
                    collection_num_simulations,
                    heartbeat_payload["done"],
                )
                if progress_callback is not None:
                    progress_callback(heartbeat_payload)
                last_heartbeat_at = now

            if max_step_seconds > 0.0 and step_elapsed_seconds > max_step_seconds:
                logger.warning(
                    "Collecte MuZero interrompue: pas trop long sur %s (step=%s, %.3fs > %.3fs).",
                    getattr(env, "symbol", "unknown"),
                    steps,
                    step_elapsed_seconds,
                    max_step_seconds,
                )
                game.metadata["stopped_reason"] = "depassement_temps_pas"
                game.metadata["stopped_step"] = int(steps)
                game.metadata["step_elapsed_seconds"] = float(step_elapsed_seconds)
                break

            if max_episode_seconds > 0.0 and episode_elapsed_seconds > max_episode_seconds:
                logger.warning(
                    "Collecte MuZero interrompue: episode trop long sur %s (step=%s, %.1fs > %.1fs).",
                    getattr(env, "symbol", "unknown"),
                    steps,
                    episode_elapsed_seconds,
                    max_episode_seconds,
                )
                game.metadata["stopped_reason"] = "depassement_temps_episode"
                game.metadata["stopped_step"] = int(steps)
                game.metadata["episode_elapsed_seconds"] = float(episode_elapsed_seconds)
                break

        game.metadata["total_steps"] = int(steps)
        try:
            episode_summary = dict(env.get_summary() or {})
        except Exception as exc:
            logger.warning(
                "Resume d'episode MuZero indisponible pour %s: %s",
                getattr(env, "symbol", "unknown"),
                exc,
            )
            episode_summary = {}

        self._copy_episode_summary_to_game_metadata(game, episode_summary)
        if len(game) > 0:
            self.replay_buffer.save_game(game)
        else:
            logger.warning(
                "Episode MuZero ignore car aucune transition n'a ete collectee pour %s.",
                getattr(env, "symbol", "unknown"),
            )
        return game

    @staticmethod
    def _copy_episode_summary_to_game_metadata(
        game: GameHistory,
        episode_summary: dict[str, object],
    ) -> None:
        """Copie les metriques d'episode utiles vers le replay.

        Les agrégations de replay et le ``training_status`` se basent sur
        ``game.metadata``. Toute métrique offensive oubliée ici devient
        invisible ensuite, même si l'environnement la calcule correctement.

        Args:
            game (GameHistory): Episode MuZero a enrichir.
            episode_summary (dict[str, object]): Resume complet de l'episode.
        """
        metadata_fields = (
            "symbol",
            "return_pct",
            "net_realized_pct",
            "total_trades",
            "buy_actions",
            "sell_actions",
            "hold_actions",
            "split_actions",
            "close_actions",
            "long_entries",
            "short_entries",
            "long_present",
            "short_present",
            "balanced_episode",
            "executed_long_entry_share",
            "executed_short_entry_share",
            "directional_imbalance",
            "directional_bias",
            "entry_veto_to_hold",
            "requested_buy_actions",
            "requested_sell_actions",
            "root_mask_directional_candidates_total",
            "root_mask_blocked_buy_total",
            "root_mask_blocked_sell_total",
            "root_mask_blocked_buy_ema200",
            "root_mask_blocked_sell_ema200",
            "root_mask_blocked_buy_vwap",
            "root_mask_blocked_sell_vwap",
            "root_mask_blocked_buy_adx",
            "root_mask_blocked_sell_adx",
            "root_mask_blocked_buy_obv",
            "root_mask_blocked_sell_obv",
            "root_mask_blocked_buy_directional",
            "root_mask_blocked_sell_directional",
            "root_mask_rate",
            "root_mask_ema200_share",
            "root_mask_vwap_share",
            "root_mask_adx_share",
            "root_mask_directional_share",
            "soft_entry_penalty_count",
            "soft_entry_penalty_total",
            "soft_entry_bonus_count",
            "soft_entry_bonus_total",
            "soft_entry_penalty_rate",
            "soft_entry_bonus_rate",
            "soft_penalty_net",
            "soft_penalty_to_bonus_ratio",
            "soft_penalty_ema200_count",
            "soft_penalty_vwap_count",
            "soft_penalty_adx_count",
            "soft_penalty_obv_count",
            "soft_penalty_ema_rate",
            "soft_penalty_vwap_rate",
            "soft_penalty_adx_rate",
            "soft_penalty_obv_rate",
            "hold_drag_opportunity_count",
            "hold_drag_penalized_count",
            "hold_drag_score",
            "split_opportunity_count",
            "split_executed",
            "split_profitable_count",
            "split_efficiency",
            "split_trade_value_delta",
            "split_improved_total_trade_count",
            "pyramid_opportunity_count",
            "pyramids_opened",
            "pyramid_profitable_count",
            "pyramid_efficiency",
            "pyramid_total_trade_improvement_pct",
            "pyramid_failed_to_improve_count",
            "slbe_triggered",
            "slbe_profitable_exits",
            "slbe_lock_profit_count",
            "slbe_capture_rate",
            "close_winner_count",
            "close_loser_count",
            "close_quality_score",
            "tp_like_exit_count",
            "tp_like_missed_count",
            "defensive_close_count",
            "early_close_noise_count",
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
            "net_return_long_pct",
            "net_return_short_pct",
            "post_veto_to_hold_rate",
            "episode_regime",
            "nemesis_type",
            "liquidity_trap_loss",
            "range_entry_loss",
            "bad_split",
            "bad_runner_exit",
            "bad_pyramid_exit",
            "hard_stop_exit",
            "runner_retained_profit_pct",
            "runner_retained_profit_score",
            "runner_giveback_pct",
            "runner_viable_window_count",
            "runner_hold_after_soft_tp_count",
            "runner_viable_but_closed_count",
            "early_full_close_after_soft_tp_count",
        )
        for field_name in metadata_fields:
            if field_name in episode_summary:
                game.metadata[field_name] = episode_summary[field_name]
        mechanics_summary = dict(episode_summary.get("metrics_by_position_mechanics") or {})
        for field_name, field_value in mechanics_summary.items():
            if isinstance(field_value, (dict, list, tuple, set)):
                continue
            game.metadata[field_name] = field_value

    def prepare_training_step(self) -> PreparedTrainingStep | None:
        """Prepare un lot d'entrainement complet sans mettre a jour les poids.

        Returns:
            PreparedTrainingStep | None: Lot transfere sur device ou
                ``None`` si le replay buffer est encore trop petit.
        """
        if self.replay_buffer.size < self.config.batch_size // 10:
            return None

        samples = self.replay_buffer.sample(self.config.batch_size)
        host_batch, batch_prepare_ms = self.trainer.prepare_batch_host(samples)
        batch, device_put_ms = self.trainer.device_put_batch(host_batch)
        tree_indices = tuple(
            int(sample[2])
            for sample in samples
            if len(sample) >= 3
        )
        return PreparedTrainingStep(
            batch=batch,
            tree_indices=tree_indices,
            batch_prepare_ms=batch_prepare_ms,
            device_put_ms=device_put_ms,
        )

    def train_step(
        self,
        prepared_step: PreparedTrainingStep | None = None,
    ):
        """Execute une mise a jour MuZero a partir du replay buffer."""
        if prepared_step is None:
            prepared_step = self.prepare_training_step()
        if prepared_step is None:
            return None

        update_started_at = perf_counter()
        self.params, self.opt_state, metrics = self.trainer.update_fn(
            self.params,
            self.opt_state,
            prepared_step.batch,
        )
        jax.block_until_ready(metrics["loss_total"])
        update_ms = (perf_counter() - update_started_at) * 1000.0
        metrics_payload = dict(metrics)
        metrics_payload["batch_prepare_ms"] = prepared_step.batch_prepare_ms
        metrics_payload["device_put_ms"] = prepared_step.device_put_ms
        metrics_payload["update_ms"] = update_ms
        platform_token = str(os.getenv("JAX_PLATFORMS", "auto")).strip() or "auto"
        cuda_token = str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip() or "none"
        metrics_payload["gpu_target_mode"] = f"{platform_token}:{cuda_token}"
        metrics_payload.update(self.replay_buffer.diversity_stats())
        priority_errors = np.asarray(
            metrics_payload.pop("priority_errors", []),
            dtype=np.float32,
        ).reshape(-1)
        if priority_errors.size > 0:
            self.replay_buffer.update_priorities(
                list(prepared_step.tree_indices),
                priority_errors.tolist(),
            )

        self.training_step_count += 1
        return self._sanitize_metrics(metrics_payload)

    def _build_root_policy_entry_filter(
        self,
        *,
        training_mode: bool,
        symbol: str | None = None,
    ) -> dict[str, float | bool | str]:
        """Construit le filtre racine cohérent avec le curriculum courant.

        Args:
            training_mode (bool): Active le curriculum d'apprentissage.
            symbol (str | None): Symbole courant si disponible.

        Returns:
            dict[str, float | bool | str]: Filtre racine observation-only.
        """
        horizon = str(getattr(self.config, "horizon", "") or "")
        curriculum_soft_end_step = int(
            getattr(self.config, "directional_curriculum_soft_end_step", 8000) or 8000
        )
        curriculum_end_step = int(
            getattr(self.config, "directional_curriculum_end_step", 15000) or 15000
        )
        if symbol:
            return TradingEnvironment.build_runtime_entry_filter(
                horizon=horizon,
                symbol=symbol,
                configured_family=getattr(self.config, "model_family", None),
                training_mode=training_mode,
                training_progress_step=int(self.training_step_count),
                curriculum_soft_end_step=curriculum_soft_end_step,
                curriculum_end_step=curriculum_end_step,
            )
        return TradingEnvironment.resolve_active_entry_filter(
            dict(getattr(self.config, "position_mechanics_profile", {}).get("entry_filter") or {}),
            training_mode=training_mode,
            training_progress_step=int(self.training_step_count),
            horizon=horizon,
            curriculum_soft_end_step=curriculum_soft_end_step,
            curriculum_end_step=curriculum_end_step,
        )

    @staticmethod
    def _select_reanalyze_indices(total_observations: int, max_positions: int) -> list[int]:
        """Selectionne un sous-ensemble stable d'observations a reanalyser.

        Args:
            total_observations (int): Nombre d'observations disponibles.
            max_positions (int): Budget maximal de positions a revisiter.

        Returns:
            list[int]: Indices tries sans doublon.
        """
        if total_observations <= 0:
            return []
        if max_positions <= 0 or total_observations <= max_positions:
            return list(range(total_observations))

        even_count = max(1, max_positions // 2)
        tail_count = max(1, max_positions - even_count)
        indices: list[int] = []
        seen: set[int] = set()

        for raw_index in np.linspace(0, total_observations - 1, num=even_count):
            index = int(round(float(raw_index)))
            if index not in seen:
                seen.add(index)
                indices.append(index)

        tail_start = max(0, total_observations - tail_count)
        for index in range(tail_start, total_observations):
            if index not in seen:
                seen.add(index)
                indices.append(index)

        indices.sort()
        if len(indices) > max_positions:
            indices = indices[-max_positions:]
        return indices

    def reanalyze_game(
        self,
        game: GameHistory,
        *,
        max_positions: int | None = None,
        num_simulations: int | None = None,
    ) -> int:
        """Recalcule une partie sans retraiter toutes les positions.

        Args:
            game (GameHistory): Episode a revisiter.
            max_positions (int | None): Nombre maximal de positions a
                reevaluer dans l'episode.
            num_simulations (int | None): Budget MCTS dedie a la reanalyse.

        Returns:
            int: Nombre de positions effectivement reanalysees.
        """
        total_observations = len(game.observations)
        selected_indices = self._select_reanalyze_indices(
            total_observations,
            total_observations if max_positions is None else int(max_positions),
        )
        if not selected_indices:
            return 0

        target_simulations = max(
            0,
            int(
                getattr(self.config, "reanalyze_num_simulations", self.config.num_simulations)
                if num_simulations is None
                else num_simulations
            ),
        )
        new_policies = list(game.policies[:total_observations])
        while len(new_policies) < total_observations:
            new_policies.append(
                np.full(
                    self.config.action_space_size,
                    1.0 / float(self.config.action_space_size),
                    dtype=np.float32,
                )
            )
        new_values = list(game.values[:total_observations])
        while len(new_values) < total_observations:
            new_values.append(0.0)

        symbol = str((game.metadata or {}).get("symbol") or "")
        entry_filter = self._build_root_policy_entry_filter(
            training_mode=True,
            symbol=symbol,
        )
        selected_observations = np.asarray(
            [np.asarray(game.observations[index], dtype=np.float32) for index in selected_indices],
            dtype=np.float32,
        ).reshape((len(selected_indices), *self.config.observation_shape))
        batched_hidden_states, batched_root_logits, batched_root_value_logits = self._jit_init(
            self.params,
            jnp.asarray(selected_observations),
        )
        mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
        for batch_index, observation_index in enumerate(selected_indices):
            obs = selected_observations[batch_index]
            hidden_state = jnp.expand_dims(batched_hidden_states[batch_index], axis=0)
            root_logits = jnp.expand_dims(batched_root_logits[batch_index], axis=0)
            root_value_logits = jnp.expand_dims(batched_root_value_logits[batch_index], axis=0)
            root_legal_actions = TradingEnvironment.infer_root_policy_actions_from_observation(
                obs,
                entry_filter=entry_filter,
            )
            root = mcts.run(
                hidden_state,
                root_logits,
                root_value_logits,
                root_legal_actions=root_legal_actions,
                add_exploration_noise=False,
                num_simulations=target_simulations,
            )
            new_policies[observation_index] = self._get_policy_distribution(root)
            new_values[observation_index] = float(root.value)

        game.policies = new_policies
        game.values = new_values
        self.last_reanalyze_num_simulations = target_simulations
        return len(selected_indices)

    def reanalyze_recent_games(self, limit: int) -> int:
        """Reanalyse les episodes les plus recents du replay buffer.

        Args:
            limit (int): Nombre maximal d'episodes a recalculer.

        Returns:
            int: Nombre d'episodes effectivement reanalyses.
        """
        self.last_reanalyze_positions_count = 0
        reanalyzed = 0
        max_positions = int(
            getattr(self.config, "reanalyze_max_positions_per_game", 0) or 0
        )
        num_simulations = int(
            getattr(self.config, "reanalyze_num_simulations", self.config.num_simulations)
            or self.config.num_simulations
        )
        for game in self.replay_buffer.recent_games(limit):
            reanalyzed_positions = self.reanalyze_game(
                game,
                max_positions=max_positions,
                num_simulations=num_simulations,
            )
            if reanalyzed_positions > 0:
                self.last_reanalyze_positions_count += reanalyzed_positions
                reanalyzed += 1
        return reanalyzed

    def _select_action(self, root, exploration: bool) -> int:
        """Choisit une action a partir des visites MCTS."""
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        actions = [item[0] for item in visit_counts]
        counts = np.asarray([item[1] for item in visit_counts], dtype=np.float64)
        if counts.size == 0:
            return 0
        if counts.sum() <= 0.0:
            counts = np.asarray(
                [root.children[action].prior for action in actions],
                dtype=np.float64,
            )
        if exploration:
            temperature = max(
                float(self.config.visit_softmax_temperature(self.training_step_count)),
                1e-3,
            )
            probs = np.power(np.maximum(counts, 1e-8), 1.0 / temperature)
            total = float(probs.sum())
            if total <= 0.0 or not np.isfinite(total):
                probs = np.full(len(actions), 1.0 / float(len(actions)), dtype=np.float64)
            else:
                probs = probs / total
            return int(np.random.choice(actions, p=probs))
        return actions[int(np.argmax(counts))]

    def _get_policy_distribution(self, root) -> np.ndarray:
        """Construit une distribution de politique a partir des visites MCTS."""
        policy = np.zeros(self.config.action_space_size)
        for action, child in root.children.items():
            policy[action] = child.visit_count
        total = policy.sum()
        if total > 0:
            policy /= total
        elif root.children:
            for action, child in root.children.items():
                policy[action] = child.prior
            total = policy.sum()
            if total > 0:
                policy /= total
        return policy

    def _sanitize_metrics(self, metrics: dict[str, object]) -> dict[str, object]:
        """Convertit les sorties JAX en types Python simples.

        Args:
            metrics (dict[str, object]): Metriques brutes renvoyees par JAX.

        Returns:
            dict[str, object]: Metriques serialisables.
        """
        sanitized: dict[str, object] = {}
        for key, value in dict(metrics or {}).items():
            array = np.asarray(value)
            if array.dtype.kind in {"U", "S"}:
                if array.ndim == 0:
                    sanitized[str(key)] = str(array.item())
                else:
                    sanitized[str(key)] = [str(item) for item in array.tolist()]
                continue
            if array.dtype.kind == "O":
                if array.ndim == 0:
                    scalar = array.item()
                    sanitized[str(key)] = scalar if isinstance(scalar, (str, bool, int, float)) else str(scalar)
                else:
                    serialized_values: list[object] = []
                    for item in array.tolist():
                        serialized_values.append(item if isinstance(item, (str, bool, int, float)) else str(item))
                    sanitized[str(key)] = serialized_values
                continue
            if array.ndim == 0:
                sanitized[str(key)] = float(array)
            else:
                sanitized[str(key)] = array.tolist()
        return sanitized

    def process_observation(self, observation: dict) -> np.ndarray:
        """Convertit une observation live en vecteur compatible MuZero.

        Args:
            observation (dict): Charge utile live du banker.

        Returns:
            np.ndarray: Vecteur ``[32]`` aligne avec l'observation MuZero.
        """
        candle = observation.get("latest_candle", {}) or {}
        indicators = observation.get("indicators", {}) or {}
        price = float(observation.get("price", candle.get("close", 0.0)) or 0.0)
        close_price = float(candle.get("close", price) or price)
        open_price = float(candle.get("open", close_price) or close_price)
        high_price = float(candle.get("high", close_price) or close_price)
        low_price = float(candle.get("low", close_price) or close_price)
        volume = float(candle.get("tick_volume", candle.get("volume", 0.0)) or 0.0)
        spread = float(candle.get("spread", 0.0) or 0.0)

        obs_vec = np.zeros(self.config.observation_shape[0], dtype=np.float32)
        obs_vec[:26] = np.array(
            [
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                float(indicators.get("EMA_200", close_price) or close_price),
                float(indicators.get("RSI", 50.0) or 50.0),
                float(indicators.get("MACD_Hist", 0.0) or 0.0),
                float(indicators.get("VWAP", close_price) or close_price),
                float(indicators.get("OBV", 0.0) or 0.0),
                float(indicators.get("Momentum", 0.0) or 0.0),
                float(indicators.get("TRIX", 0.0) or 0.0),
                float(indicators.get("Stoch_K", 50.0) or 50.0),
                float(indicators.get("Stoch_D", 50.0) or 50.0),
                float(indicators.get("CCI", 0.0) or 0.0),
                float(indicators.get("ADX", 0.0) or 0.0),
                float(indicators.get("ADX_Plus_DI", 0.0) or 0.0),
                float(indicators.get("ADX_Minus_DI", 0.0) or 0.0),
                float(indicators.get("Ichi_Tenkan", close_price) or close_price),
                float(indicators.get("Ichi_Kijun", close_price) or close_price),
                float(indicators.get("Ichi_Senkou_A", close_price) or close_price),
                float(indicators.get("Ichi_Senkou_B", close_price) or close_price),
                float(indicators.get("ATR", 0.0) or 0.0),
                float(indicators.get("BB_Pct", 0.5) or 0.5),
                spread / max(close_price, 1e-8),
                float(indicators.get("Return_1", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )

        timestamp_raw = observation.get("timestamp")
        if timestamp_raw:
            try:
                current_time = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
            except ValueError:
                current_time = datetime.utcnow()
        else:
            current_time = datetime.utcnow()

        volatility = min((high_price - low_price) / max(close_price, 1e-8) * 100.0, 1.0) if close_price > 0 else 0.0
        position_state = float(observation.get("position_state", 0.0) or 0.0)
        unrealized_return = float(observation.get("unrealized_return", 0.0) or 0.0)
        slbe_state = float(observation.get("slbe_state", 0.0) or 0.0)
        obs_vec[26:] = np.array(
            [
                position_state,
                unrealized_return,
                slbe_state,
                current_time.hour / 23.0 if 23 else 0.0,
                current_time.weekday() / 6.0 if 6 else 0.0,
                volatility,
            ],
            dtype=np.float32,
        )
        return obs_vec

    def infer_action(self, observation: dict | np.ndarray) -> dict[str, object]:
        """Execute une inference greedy MuZero a partir d'une observation live.

        Args:
            observation (dict | np.ndarray): Observation brute banker ou vecteur deja prepare.

        Returns:
            dict[str, object]: Action, politique, valeur et confiance.
        """
        if isinstance(observation, dict):
            obs_vec = self.process_observation(observation)
            observation_symbol = str(
                observation.get("symbol")
                or observation.get("instrument")
                or observation.get("ticker")
                or ""
            )
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)
            observation_symbol = ""

        obs_jax = jnp.array(obs_vec).reshape(1, -1)
        hidden_state, root_logits, root_value_logits = self._jit_init(self.params, obs_jax)
        root_legal_actions = TradingEnvironment.infer_root_policy_actions_from_observation(
            obs_vec,
            entry_filter=self._build_root_policy_entry_filter(
                training_mode=False,
                symbol=observation_symbol,
            ),
        )
        mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
        root = mcts.run(
            hidden_state,
            root_logits,
            root_value_logits,
            root_legal_actions=root_legal_actions,
            add_exploration_noise=False,
        )

        action = self._select_action(root, exploration=False)
        policy = self._get_policy_distribution(root)
        action_names = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]

        return {
            "action": action,
            "action_name": action_names[action] if action < len(action_names) else f"ACT_{action}",
            "policy": policy.tolist(),
            "value": float(root.value),
            "confidence": float(policy[action]) if action < len(policy) else 0.0,
            "simulations": self.config.num_simulations,
        }

    def save(
        self,
        path: str,
        *,
        artifact_kind: str = "checkpoint",
        lineage: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Sauvegarde les poids MuZero dans le schema structure v2.

        Args:
            path (str): Chemin cible du checkpoint.
            artifact_kind (str): Nature de l'artefact (`checkpoint`, `latest`,
                `challenger`, `champion`, etc.).
            lineage (dict[str, object] | None): Metadonnees de filiation du
                checkpoint.

        Returns:
            dict[str, object]: Payload structure serialise.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = save_muzero_checkpoint(
            path,
            config=self.config,
            params=self.params,
            opt_state=self.opt_state,
            training_step_count=int(self.training_step_count),
            artifact_kind=artifact_kind,
            lineage=dict(lineage or {}),
        )
        logger.info("[JAXMuZeroAgent] Checkpoint sauvegarde: %s", path)
        return payload

    def inspect_checkpoint(self, path: str) -> dict[str, object]:
        """Retourne le rapport de compatibilite d'un checkpoint MuZero.

        Args:
            path (str): Chemin du checkpoint a inspecter.

        Returns:
            dict[str, object]: Rapport detaille de compatibilite.
        """
        _payload, compatibility = inspect_muzero_checkpoint(
            path,
            expected_context=self._expected_checkpoint_context,
        )
        return compatibility

    def load(self, path: str) -> dict[str, object]:
        """Recharge les poids et l'etat de l'optimiseur depuis un checkpoint.

        Args:
            path (str): Chemin du checkpoint a charger.

        Returns:
            dict[str, object]: Rapport de compatibilite du checkpoint charge.

        Raises:
            MuZeroCheckpointCompatibilityError: Si le checkpoint ne respecte
                pas l'architecture attendue.
        """
        payload, compatibility = inspect_muzero_checkpoint(
            path,
            expected_context=self._expected_checkpoint_context,
        )
        if not compatibility.get("allowed", False):
            raise MuZeroCheckpointCompatibilityError(
                str(compatibility.get("reason") or "Checkpoint MuZero incompatible.")
            )
        checkpoint_payload = dict(payload or {})
        self.params = checkpoint_payload["params"]
        self.opt_state = checkpoint_payload.get("opt_state", self.opt_state)
        raw_training_step_count = checkpoint_payload.get("training_step_count")
        if raw_training_step_count is None:
            raw_training_step_count = self._extract_training_step_from_checkpoint_path(path)
        try:
            self.training_step_count = max(0, int(raw_training_step_count or 0))
        except (TypeError, ValueError):
            self.training_step_count = 0
        logger.info("[JAXMuZeroAgent] Checkpoint charge: %s", path)
        return compatibility
