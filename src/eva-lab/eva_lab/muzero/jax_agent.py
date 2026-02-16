"""
MuZero Agent in JAX — THE HIVE EVA Lab

This agent orchestrates the self-play, inference, and training using
the JAX-based engine.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
from typing import Optional

from eva_lab.muzero.jax_networks import make_muzero_networks
from eva_lab.muzero.jax_mcts import JAXMuZeroMCTS
from eva_lab.muzero.jax_trainer import MuZeroTrainerJAX
from eva_lab.muzero.replay_buffer import PrioritizedReplayBuffer, GameHistory

logger = logging.getLogger(__name__)

class JAXMuZeroAgent:
    def __init__(self, config):
        self.config = config
        
        # 1. Networks & Trainer
        self.transformed = make_muzero_networks(config)
        self.trainer = MuZeroTrainerJAX(config, self.transformed)
        
        # Initialize params
        dummy_obs = jnp.zeros((1, *config.observation_shape))
        self.params, self.opt_state = self.trainer.init_params(dummy_obs)
        
        # 2. Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            max_games=config.window_size // config.max_moves
        )
        
        # 3. JIT hooks
        self._jit_init = jax.jit(lambda p, o: self.transformed.apply(p, None, o, method=0))
        self._jit_rec = jax.jit(lambda p, s, a: self.transformed.apply(p, None, s, a, method=1))

        logger.info(f"[JAXMuZeroAgent] Operational (JAX/Cuda). Strategy: {config.hidden_state_size} dims.")

    # ── Self-Play ─────────────────────────────────────────────

    def play_game(self, env, exploration: bool = True) -> GameHistory:
        game = GameHistory()
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < self.config.max_moves:
            steps += 1
            obs_jax = jnp.array(obs).reshape(1, -1)
            
            # initial inference to get hidden state
            h, logits, v = self._jit_init(self.params, None, obs_jax)
            
            # MCTS
            mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
            root = mcts.run(h, add_exploration_noise=exploration)
            
            action = self._select_action(root, exploration)
            policy = self._get_policy_distribution(root)
            value = float(root.value)
            
            # Step env
            next_obs, reward, done, _, _ = env.step(action)
            game.store(obs, action, reward, policy, value)
            obs = next_obs
            
        self.replay_buffer.save_game(game)
        return game

    # ── Training ──────────────────────────────────────────────

    def train_step(self):
        if self.replay_buffer.size < self.config.batch_size // 10: # Minimum seed
            return None
            
        # Sample batch
        samples = self.replay_buffer.sample(self.config.batch_size)
        batch = self.trainer.prepare_batch(samples)
        
        # Update
        self.params, self.opt_state, metrics = self.trainer.update_fn(
            self.params, self.opt_state, batch
        )
        
        return metrics

    # ── Re-analyze (Dreamer logic) ───────────────────────────
    
    def reanalyze_game(self, game: GameHistory):
        """
        Re-run MCTS on old games with current network to update policies/values.
        This provides 'fresh' targets for the same experiences.
        """
        new_policies = []
        new_values = []
        
        for obs in game.observations:
            obs_jax = jnp.array(obs).reshape(1, -1)
            h, _, _ = self._jit_init(self.params, None, obs_jax)
            mcts = JAXMuZeroMCTS(self.config, self.params, (self._jit_init, self._jit_rec))
            root = mcts.run(h, add_exploration_noise=False)
            new_policies.append(self._get_policy_distribution(root))
            new_values.append(float(root.value))
            
        game.policies = new_policies
        game.values = new_values

    # ── Helpers ──────────────────────────────────────────────

    def _select_action(self, root, exploration: bool) -> int:
        visit_counts = [(a, c.visit_count) for a, c in root.children.items()]
        actions = [x[0] for x in visit_counts]
        counts = [x[1] for x in visit_counts]
        if exploration:
            probs = np.array(counts, dtype=float)
            probs /= probs.sum()
            return int(np.random.choice(actions, p=probs))
        return actions[np.argmax(counts)]

    def _get_policy_distribution(self, root) -> np.ndarray:
        policy = np.zeros(self.config.action_space_size)
        for action, child in root.children.items():
            policy[action] = child.visit_count
        total = policy.sum()
        if total > 0: policy /= total
        return policy

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"params": self.params, "opt_state": self.opt_state}, f)
        logger.info(f"[JAXMuZeroAgent] Checkpoint saved: {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.params = data["params"]
            self.opt_state = data["opt_state"]
        logger.info(f"[JAXMuZeroAgent] Checkpoint loaded: {path}")
