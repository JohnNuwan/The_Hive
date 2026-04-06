"""
Buffer de replay priorise pour MuZero.

Le buffer s'appuie sur une SumTree afin d'echantillonner les transitions
avec une probabilite proportionnelle a leur priorite.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Any, List, Tuple


class ReplayValidationError(ValueError):
    """Represente une incoherence exploitable dans le replay MuZero."""

    def __init__(self, code: str, message: str):
        """Construit une erreur de validation avec code stable.

        Args:
            code (str): Code court exploitable par l'orchestrateur.
            message (str): Message explicatif en francais.
        """
        self.code = str(code or "REPLAY_SAMPLE_INVALID").strip() or "REPLAY_SAMPLE_INVALID"
        super().__init__(f"{self.code}: {message}")

@dataclass
class GameHistory:
    """Stocke l'ensemble des transitions d'un episode."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    priorities: list = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def store(self, obs, action, reward, policy, value, priority: float = 1.0):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        normalized_priority = float(priority) if priority is not None else 1.0
        if not np.isfinite(normalized_priority) or normalized_priority <= 0.0:
            normalized_priority = 1.0
        self.priorities.append(normalized_priority)

    def __len__(self):
        return len(self.observations)

class SumTree:
    """Arbre binaire dont chaque noeud vaut la somme de ses enfants."""
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
    """Buffer de replay stockant des episodes avec echantillonnage priorise."""
    def __init__(self, max_games: int, alpha: float = 0.6):
        self.max_games = max_games
        self.alpha = alpha  # Exposant de prioritisation.
        self.tree = SumTree(max_games)

    def save_game(self, game: GameHistory):
        # On conserve la priorite maximale de l'episode pour favoriser
        # les trajectoires les plus instructives dans le replay.
        p = np.max(game.priorities) if game.priorities else 1.0
        self.tree.add(p**self.alpha, game)

    def sample(self, batch_size: int) -> List[Tuple[GameHistory, int, float]]:
        """Echantillonne un lot ``(episode, start_idx, weight)``."""
        if batch_size <= 0:
            raise ReplayValidationError(
                "REPLAY_SAMPLE_INVALID",
                "La taille de lot MuZero doit etre strictement positive.",
            )
        if self.tree.n_entries <= 0:
            raise ReplayValidationError(
                "REPLAY_SAMPLE_INVALID",
                "Le replay MuZero est vide au moment de l'echantillonnage.",
            )
        total_priority = float(self.tree.total())
        if not np.isfinite(total_priority) or total_priority <= 0.0:
            raise ReplayValidationError(
                "REPLAY_SAMPLE_INVALID",
                "La priorite totale du replay MuZero est invalide.",
            )
        batch = []
        segment = total_priority / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, game) = self.tree.get(s)
            if game is None:
                raise ReplayValidationError(
                    "REPLAY_SAMPLE_INVALID",
                    "Un episode MuZero absent a ete renvoye par le replay buffer.",
                )
            if len(game) <= 0:
                raise ReplayValidationError(
                    "REPLAY_SAMPLE_INVALID",
                    "Un episode MuZero vide a ete renvoye par le replay buffer.",
                )
            
            # Un point d'entree aleatoire evite de toujours sur-entrainer
            # les memes prefixes d'episodes.
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
