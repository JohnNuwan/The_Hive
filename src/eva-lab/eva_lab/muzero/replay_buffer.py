"""
Prioritized Replay Buffer for MuZero — THE HIVE EVA Lab

Uses a SumTree structure to sample transitions with probability proportional 
to their error/priority.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GameHistory:
    """One episode's worth of data."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    priorities: list = field(default_factory=list)
    
    def store(self, obs, action, reward, policy, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        # Initial priority is max or 1.0
        self.priorities.append(1.0)

    def __len__(self):
        return len(self.observations)

class SumTree:
    """Binary tree where each node is the sum of its children."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayBuffer:
    """Buffer storing GameHistory objects with priority sampling."""
    def __init__(self, max_games: int, alpha: float = 0.6):
        self.max_games = max_games
        self.alpha = alpha  # Prioritization exponent
        self.tree = SumTree(max_games)

    def save_game(self, game: GameHistory):
        # Priority based on avg priority or max priority of moves in game
        p = np.max(game.priorities) if game.priorities else 1.0
        self.tree.add(p**self.alpha, game)

    def sample(self, batch_size: int) -> List[Tuple[GameHistory, int, float]]:
        """Sample (game, start_idx, weight) batch."""
        batch = []
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, game) = self.tree.get(s)
            
            # Pick a random start index in the game
            start_idx = random.randint(0, len(game) - 1)
            batch.append((game, start_idx, idx))
            
        return batch

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            p = (error + 1e-5) ** self.alpha
            self.tree.update(idx, p)

    @property
    def size(self):
        return self.tree.n_entries
