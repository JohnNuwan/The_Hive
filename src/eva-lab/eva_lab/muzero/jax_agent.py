"""Agent MuZero JAX pour le self-play, l'entrainement et la persistence."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from time import perf_counter
from typing import NamedTuple

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

    def _initial_inference(self, params, observation):
        """Execute l'inference initiale MuZero sur une observation brute."""
        return self.initial_apply(params, None, observation)

    def _recurrent_inference(self, params, hidden_state, action_onehot):
        """Execute l'inference recurrente MuZero sur un etat latent."""
        return self.recurrent_apply(params, None, hidden_state, action_onehot)

    def play_game(self, env, exploration: bool = True) -> GameHistory:
        """Joue une partie complete dans l'environnement et alimente le replay buffer."""
        game = GameHistory()
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < self.config.max_moves:
            steps += 1
            obs_jax = jnp.array(obs).reshape(1, -1)
            hidden_state, root_logits, root_value_logits = self._jit_init(self.params, obs_jax)
            root_legal_actions = env.get_legal_root_actions()

            mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
            root = mcts.run(
                hidden_state,
                root_logits,
                root_value_logits,
                root_legal_actions=root_legal_actions,
                add_exploration_noise=exploration,
            )

            action = self._select_action(root, exploration)
            policy = self._get_policy_distribution(root)
            value = float(root.value)

            next_obs, reward, done, _, _ = env.step(action)
            game.store(obs, action, reward, policy, value)
            obs = next_obs

        self.replay_buffer.save_game(game)
        return game

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

        for observation_index in selected_indices:
            obs = game.observations[observation_index]
            obs_jax = jnp.array(obs).reshape(1, -1)
            hidden_state, root_logits, root_value_logits = self._jit_init(self.params, obs_jax)
            root_legal_actions = TradingEnvironment.infer_legal_root_actions_from_observation(obs)
            mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
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
        obs_vec[26:] = np.array(
            [
                0.0,
                0.0,
                0.0,
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
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)

        obs_jax = jnp.array(obs_vec).reshape(1, -1)
        hidden_state, root_logits, root_value_logits = self._jit_init(self.params, obs_jax)
        root_legal_actions = TradingEnvironment.infer_legal_root_actions_from_observation(obs_vec)
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
        logger.info("[JAXMuZeroAgent] Checkpoint charge: %s", path)
        return compatibility
