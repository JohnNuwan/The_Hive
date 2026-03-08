"""Agent MuZero JAX pour le self-play, l'entrainement et la persistence."""

from __future__ import annotations

import logging
import os
import pickle

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

    def play_game(self, env, exploration: bool = True) -> GameHistory:
        """Joue une partie complete dans l'environnement et alimente le replay buffer."""
        game = GameHistory()
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < self.config.max_moves:
            steps += 1
            obs_jax = jnp.array(obs).reshape(1, -1)
            hidden_state, _, _ = self._jit_init(self.params, obs_jax)

            mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
            root = mcts.run(hidden_state, add_exploration_noise=exploration)

            action = self._select_action(root, exploration)
            policy = self._get_policy_distribution(root)
            value = float(root.value)

            next_obs, reward, done, _, _ = env.step(action)
            game.store(obs, action, reward, policy, value)
            obs = next_obs

        self.replay_buffer.save_game(game)
        return game

    def train_step(self):
        """Execute une mise a jour MuZero a partir du replay buffer."""
        if self.replay_buffer.size < self.config.batch_size // 10:
            return None

        samples = self.replay_buffer.sample(self.config.batch_size)
        batch = self.trainer.prepare_batch(samples)
        self.params, self.opt_state, metrics = self.trainer.update_fn(
            self.params,
            self.opt_state,
            batch,
        )
        return metrics

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

    def save(self, path: str) -> None:
        """Sauvegarde les poids et l'etat de l'optimiseur."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file_obj:
            pickle.dump({"params": self.params, "opt_state": self.opt_state}, file_obj)
        logger.info("[JAXMuZeroAgent] Checkpoint sauvegarde: %s", path)

    def load(self, path: str) -> None:
        """Recharge les poids et l'etat de l'optimiseur depuis un checkpoint."""
        with open(path, "rb") as file_obj:
            data = pickle.load(file_obj)
        self.params = data["params"]
        self.opt_state = data["opt_state"]
        logger.info("[JAXMuZeroAgent] Checkpoint charge: %s", path)
