"""
MuZero Agent — THE HIVE Self-Play & Inference Engine

Ported from Muzero_Pro_Trader/MuZero/agents/muzero_agent.py.

Provides:
  - Self-play episodes (training mode with replay buffer)
  - Inference-only action selection (live/shadow mode)
  - Model save/load for checkpoint management
  - Integration with Shadow Learning buffer
"""

import os
import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from eva_lab.muzero.networks import MuZeroNet
from eva_lab.muzero.mcts import MuZeroMCTS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Replay Buffer (lightweight, for self-play games)
# ═══════════════════════════════════════════════════════════════

@dataclass
class GameHistory:
    """One episode's worth of (obs, action, reward, policy, value)."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def store(self, obs, action, reward, policy, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.dones.append(done)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    def __len__(self) -> int:
        return len(self.observations)


class ReplayBuffer:
    """Sliding-window replay buffer for MuZero self-play games."""

    def __init__(self, max_games: int = 1000):
        self.max_games = max_games
        self.games: List[GameHistory] = []
        self.total_games_saved = 0

    def save_game(self, game: GameHistory):
        if len(self.games) >= self.max_games:
            self.games.pop(0)
        self.games.append(game)
        self.total_games_saved += 1

    @property
    def size(self) -> int:
        return len(self.games)


# ═══════════════════════════════════════════════════════════════
#  MuZero Agent
# ═══════════════════════════════════════════════════════════════

class MuZeroAgent:
    """
    MuZero Agent for The Hive.

    Modes:
      - Training:  self-play loop → replay buffer → gradient updates
      - Inference:  MCTS action selection without training
      - Shadow:     Record observations for future DreamerGate training
    """

    def __init__(self, config, device: str = None):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = MuZeroNet(config).to(self.device)
        self.replay_buffer = ReplayBuffer(max_games=config.window_size // config.max_moves)

        logger.info(
            f"[MuZero:Agent] Ready on {self.device} "
            f"(buffer capacity: {self.replay_buffer.max_games} games)"
        )

    # ── Self-Play ─────────────────────────────────────────────

    def play_game(self, env, exploration: bool = True) -> GameHistory:
        """
        Play one full episode, storing (obs, action, reward, policy, value).

        Returns the GameHistory (also saved to replay buffer).
        """
        game = GameHistory()
        obs, _ = env.reset()
        done = False
        steps = 0

        with torch.no_grad():
            while not done and steps < self.config.max_moves:
                steps += 1
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

                # Representation → hidden state
                root_state = self.network.representation(obs_tensor)

                # MCTS → visit counts → action
                mcts = MuZeroMCTS(self.config, self.network)
                root = mcts.run(root_state, add_exploration_noise=exploration)

                action = self._select_action(root, exploration)
                policy = self._get_policy_distribution(root)
                value = root.value

                # Step environment
                next_obs, reward, done, _, _ = env.step(action)
                game.store(obs, action, reward, policy, value, done)
                obs = next_obs

        self.replay_buffer.save_game(game)
        logger.info(
            f"[MuZero:Agent] Game #{self.replay_buffer.total_games_saved}: "
            f"{steps} steps, reward={game.total_reward:.1f}"
        )
        return game

    # ── Inference Only ────────────────────────────────────────

    def infer_action(self, observation: np.ndarray) -> dict:
        """
        Select the best action given an observation (no training).

        Returns dict with action, policy, value, confidence.
        """
        with torch.no_grad():
            obs_tensor = (
                torch.tensor(observation, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            root_state = self.network.representation(obs_tensor)
            mcts = MuZeroMCTS(self.config, self.network)
            root = mcts.run(root_state, add_exploration_noise=False)

            action = self._select_action(root, exploration=False)
            policy = self._get_policy_distribution(root)

            ACTION_NAMES = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]
            return {
                "action": action,
                "action_name": ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"ACT_{action}",
                "policy": policy.tolist(),
                "value": root.value,
                "confidence": float(policy[action]),
                "simulations": self.config.num_simulations,
            }

    # ── Action Selection ──────────────────────────────────────

    def _select_action(self, root, exploration: bool = True) -> int:
        """Select action from MCTS visit counts."""
        visit_counts = [(a, c.visit_count) for a, c in root.children.items()]
        actions = [x[0] for x in visit_counts]
        counts = [x[1] for x in visit_counts]

        if exploration:
            # Proportional sampling (temperature=1)
            probs = np.array(counts, dtype=float)
            probs /= probs.sum()
            return int(np.random.choice(actions, p=probs))
        else:
            # Greedy (temperature→0)
            return actions[np.argmax(counts)]

    def _get_policy_distribution(self, root) -> np.ndarray:
        """Extract normalized policy from visit counts."""
        policy = np.zeros(self.config.action_space_size)
        for action, child in root.children.items():
            policy[action] = child.visit_count
        total = policy.sum()
        if total > 0:
            policy /= total
        return policy

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str = None):
        """Save network weights to disk."""
        path = path or os.path.join(self.config.weights_path, "muzero_latest.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)
        logger.info(f"[MuZero:Agent] Saved weights → {path}")

    def load(self, path: str):
        """Load network weights from disk."""
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        logger.info(f"[MuZero:Agent] Loaded weights ← {path}")

    # ── Stats ─────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return agent status for API endpoints."""
        return {
            "device": str(self.device),
            "total_games": self.replay_buffer.total_games_saved,
            "buffer_size": self.replay_buffer.size,
            "network_params": sum(p.numel() for p in self.network.parameters()),
            "config": self.config.to_dict(),
        }
