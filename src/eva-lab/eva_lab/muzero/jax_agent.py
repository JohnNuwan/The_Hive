"""Agent MuZero JAX pour le self-play, l'entrainement et la persistence."""

from __future__ import annotations

import logging
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from eva_lab.muzero.jax_mcts import JAXMuZeroMCTS
from eva_lab.muzero.jax_networks import make_muzero_networks
from eva_lab.muzero.jax_trainer import MuZeroTrainerJAX
from eva_lab.muzero.replay_buffer import GameHistory, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


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

        self._jit_init = jax.jit(self._initial_inference)
        self._jit_rec = jax.jit(self._recurrent_inference)

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

    def initial_inference_batch(
        self,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute une inference initiale batchee materialisee cote Python.

        Args:
            observations (np.ndarray): Observations plates a projeter.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Etats latents, logits
            et valeurs sous forme Numpy.
        """

        batch = np.asarray(observations, dtype=np.float32).reshape(len(observations), -1)
        hidden_state, logits, value = self._jit_init(self.params, jnp.asarray(batch))
        return (
            np.asarray(jax.device_get(hidden_state), dtype=np.float32),
            np.asarray(jax.device_get(logits), dtype=np.float32),
            np.asarray(jax.device_get(value), dtype=np.float32),
        )

    def recurrent_inference_batch(
        self,
        hidden_states: np.ndarray,
        action_onehots: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute une inference recurrente batchee materialisee cote Python.

        Args:
            hidden_states (np.ndarray): Etats latents a propager.
            action_onehots (np.ndarray): Actions one-hot associees.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Etats
            latents suivants, recompenses, logits et valeurs.
        """

        hidden_batch = np.asarray(hidden_states, dtype=np.float32).reshape(len(hidden_states), -1)
        action_batch = np.asarray(action_onehots, dtype=np.float32).reshape(len(action_onehots), -1)
        next_state, reward, logits, value = self._jit_rec(
            self.params,
            jnp.asarray(hidden_batch),
            jnp.asarray(action_batch),
        )
        return (
            np.asarray(jax.device_get(next_state), dtype=np.float32),
            np.asarray(jax.device_get(reward), dtype=np.float32),
            np.asarray(jax.device_get(logits), dtype=np.float32),
            np.asarray(jax.device_get(value), dtype=np.float32),
        )

    def play_game(
        self,
        env,
        exploration: bool = True,
        collection_mode: str = "mcts",
        max_wall_time_seconds: float | None = None,
    ) -> GameHistory:
        """Joue une partie complete et alimente le replay buffer.

        Args:
            env (Any): Environnement de trading MuZero.
            exploration (bool): Active la stochasticite de collecte.
            collection_mode (str): Mode de collecte ``mcts`` ou ``policy_only``.
            max_wall_time_seconds (float | None): Garde-fou de duree
                maximale pour un episode de collecte.

        Returns:
            GameHistory: Partie collecte, potentiellement tronquee si un
                garde-fou de duree s'active.

        Raises:
            ValueError: Si le mode de collecte demande est inconnu.
        """
        game = GameHistory()
        obs, _ = env.reset()
        done = False
        steps = 0
        started_at = time.perf_counter()

        while not done and steps < self.config.max_moves:
            if (
                max_wall_time_seconds is not None
                and max_wall_time_seconds > 0.0
                and (time.perf_counter() - started_at) >= max_wall_time_seconds
            ):
                logger.warning(
                    "[JAXMuZeroAgent] Collecte interrompue apres %.1fs au pas %s (mode=%s).",
                    max_wall_time_seconds,
                    steps,
                    collection_mode,
                )
                break
            steps += 1
            obs_jax = jnp.array(obs).reshape(1, -1)
            hidden_state, logits, value_tensor = self._jit_init(self.params, obs_jax)

            if collection_mode == "mcts":
                mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
                root = mcts.run(hidden_state, add_exploration_noise=exploration)
                action = self._select_action(root, exploration)
                policy = self._get_policy_distribution(root)
                value = float(root.value)
            elif collection_mode == "policy_only":
                policy = self._policy_from_logits(logits)
                action = self._select_action_from_policy(policy, exploration)
                value = self._tensor_to_scalar(value_tensor)
            else:
                raise ValueError(f"Mode de collecte MuZero inconnu: {collection_mode}")

            next_obs, reward, done, _, _ = env.step(action)
            game.store(obs, action, reward, policy, value)
            obs = next_obs

        if len(game) > 0:
            self.replay_buffer.save_game(game)
        return game

    def _policy_from_logits(self, logits: Any) -> np.ndarray:
        """Convertit des logits reseau en distribution de politique stable.

        Args:
            logits (Any): Sortie brute du reseau de politique.

        Returns:
            np.ndarray: Distribution normalisee sur l'espace d'actions.
        """
        logits_np = np.asarray(jax.device_get(logits), dtype=np.float32).reshape(-1)
        if logits_np.size != self.config.action_space_size:
            logits_np = np.resize(logits_np, self.config.action_space_size)
        logits_np = logits_np - np.max(logits_np)
        probs = np.exp(logits_np).astype(np.float32)
        total = float(np.sum(probs))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(
                self.config.action_space_size,
                1.0 / max(self.config.action_space_size, 1),
                dtype=np.float32,
            )
        return (probs / total).astype(np.float32)

    def _select_action_from_policy(self, policy: np.ndarray, exploration: bool) -> int:
        """Selectionne une action directement depuis la politique reseau.

        Args:
            policy (np.ndarray): Distribution de politique normalisee.
            exploration (bool): Active l'echantillonnage stochastique.

        Returns:
            int: Action discrete choisie.
        """
        if exploration:
            return int(np.random.choice(np.arange(len(policy)), p=policy))
        return int(np.argmax(policy))

    def _tensor_to_scalar(self, value_tensor: Any) -> float:
        """Materialise un scalaire JAX/Numpy en ``float`` Python.

        Args:
            value_tensor (Any): Tenseur source.

        Returns:
            float: Valeur scalaire materialisee.
        """
        value_np = np.asarray(jax.device_get(value_tensor), dtype=np.float32).reshape(-1)
        if value_np.size <= 0:
            return 0.0
        return float(value_np[0])

    def train_step(
        self,
        trace_hook: Callable[[str], None] | None = None,
    ) -> dict[str, Any] | None:
        """Execute une mise a jour MuZero de facon synchrone et tracable.

        Args:
            trace_hook (Callable[[str], None] | None): Callback optionnel
                appele a chaque sous-phase critique.

        Returns:
            dict[str, Any] | None: Metriques Python materialisees et durees
                par sous-phase, ou ``None`` si le buffer est trop faible.
        """
        if self.replay_buffer.size < self.config.batch_size // 10:
            return None

        phase_durations_ms: dict[str, float] = {}

        if trace_hook is not None:
            trace_hook("replay_sample_start")
        sample_started_at = time.perf_counter()
        samples = self.replay_buffer.sample(self.config.batch_size)
        phase_durations_ms["sample"] = round((time.perf_counter() - sample_started_at) * 1000.0, 3)
        if trace_hook is not None:
            trace_hook("replay_sample_done")

        if trace_hook is not None:
            trace_hook("prepare_batch_start")
        prepare_started_at = time.perf_counter()
        batch = self.trainer.prepare_batch(samples)
        phase_durations_ms["prepare_batch"] = round(
            (time.perf_counter() - prepare_started_at) * 1000.0,
            3,
        )
        if trace_hook is not None:
            trace_hook("prepare_batch_done")

        if trace_hook is not None:
            trace_hook("update_fn_start")
        update_started_at = time.perf_counter()
        self.params, self.opt_state, metrics = self.trainer.update_fn(
            self.params,
            self.opt_state,
            batch,
        )
        metrics = jax.tree_util.tree_map(jax.block_until_ready, metrics)
        phase_durations_ms["update_fn"] = round((time.perf_counter() - update_started_at) * 1000.0, 3)
        if trace_hook is not None:
            trace_hook("update_fn_done")

        if trace_hook is not None:
            trace_hook("metrics_materialize_start")
        materialize_started_at = time.perf_counter()
        materialized_metrics = self._materialize_metrics(metrics)
        phase_durations_ms["materialize_metrics"] = round(
            (time.perf_counter() - materialize_started_at) * 1000.0,
            3,
        )
        if trace_hook is not None:
            trace_hook("metrics_materialize_done")
        return {
            "metrics": materialized_metrics,
            "phase_durations_ms": phase_durations_ms,
            "buffer_size": self.replay_buffer.size,
        }

    def reanalyze_game(self, game: GameHistory) -> None:
        """Recalcule politiques et valeurs d'une partie avec le reseau courant."""
        new_policies = []
        new_values = []

        for obs in game.observations:
            obs_jax = jnp.array(obs).reshape(1, -1)
            hidden_state, _, _ = self._jit_init(self.params, obs_jax)
            mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
            root = mcts.run(hidden_state, add_exploration_noise=False)
            new_policies.append(self._get_policy_distribution(root))
            new_values.append(float(root.value))

        game.policies = new_policies
        game.values = new_values

    def _select_action(self, root, exploration: bool) -> int:
        """Choisit une action a partir des visites MCTS."""
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        actions = [item[0] for item in visit_counts]
        counts = [item[1] for item in visit_counts]
        if exploration:
            probs = np.array(counts, dtype=float)
            probs /= probs.sum()
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
        return policy

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
        hidden_state, _, _ = self._jit_init(self.params, obs_jax)
        mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
        root = mcts.run(hidden_state, add_exploration_noise=False)

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

    def _materialize_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Convertit les metriques JAX en scalaires Python synchrones."""

        materialized = jax.device_get(metrics)
        payload: dict[str, Any] = {}
        for key, value in dict(materialized).items():
            array_value = np.asarray(value)
            if array_value.ndim == 0:
                payload[key] = float(array_value)
            else:
                payload[key] = array_value.tolist()
        return payload

    def _infer_checkpoint_step(self, path: str) -> int | None:
        """Infere le numero de step a partir du nom du checkpoint."""

        stem = Path(path).stem
        for pattern in (r"_ckpt_(\d+)$", r"_gold_precheck_(\d+)$"):
            match = re.search(pattern, stem)
            if match:
                return int(match.group(1))
        return None

    def save(
        self,
        path: str,
        *,
        step: int | None = None,
        run_id: str | None = None,
        trial_id: str | None = None,
        gate_profile: str | None = None,
        focus_symbols: list[str] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Sauvegarde les poids, l'optimiseur et les metadonnees de reprise.

        Args:
            path (str): Chemin de destination du checkpoint.
            step (int | None): Etape exacte associee au snapshot.
            run_id (str | None): Identifiant du run courant.
            trial_id (str | None): Identifiant logique du trial.
            gate_profile (str | None): Gate de promotion associee.
            focus_symbols (list[str] | None): Univers reduit du run.
            created_at (str | None): Horodatage ISO de creation.

        Returns:
            dict[str, Any]: Metadonnees persistées avec le checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        metadata = {
            "step": step if step is not None else self._infer_checkpoint_step(path),
            "run_id": run_id,
            "trial_id": trial_id,
            "gate_profile": gate_profile,
            "focus_symbols": [str(item).strip() for item in focus_symbols or [] if str(item).strip()],
            "created_at": created_at or datetime.now().isoformat(),
        }
        with open(path, "wb") as file_obj:
            pickle.dump(
                {
                    "params": self.params,
                    "opt_state": self.opt_state,
                    "metadata": metadata,
                },
                file_obj,
            )
        logger.info("[JAXMuZeroAgent] Checkpoint sauvegarde: %s", path)
        return metadata

    def load(self, path: str) -> dict[str, Any]:
        """Recharge les poids et l'etat de l'optimiseur depuis un checkpoint.

        Args:
            path (str): Chemin du checkpoint a relire.

        Returns:
            dict[str, Any]: Metadonnees de reprise reconstruites.
        """
        with open(path, "rb") as file_obj:
            data = pickle.load(file_obj)
        self.params = data["params"]
        self.opt_state = data["opt_state"]
        metadata = dict(data.get("metadata") or {})
        metadata.setdefault("step", self._infer_checkpoint_step(path))
        metadata.setdefault("focus_symbols", [])
        metadata.setdefault("gate_profile", None)
        metadata.setdefault("trial_id", None)
        metadata.setdefault("run_id", None)
        logger.info("[JAXMuZeroAgent] Checkpoint charge: %s", path)
        return metadata
