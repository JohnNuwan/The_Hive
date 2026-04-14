"""Buffer de replay priorise pour MuZero."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class GameHistory:
    """Stocke l'historique complet d'un episode MuZero."""

    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    priorities: list = field(default_factory=list)

    def store(self, obs, action, reward, policy, value) -> None:
        """Ajoute une transition dans l'historique courant.

        Args:
            obs (object): Observation racine.
            action (int): Action executee.
            reward (float): Recompense immediate.
            policy (object): Distribution de politique cible.
            value (float): Valeur estimee a la racine.
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        # Une nouvelle transition doit rester echantillonnable immediatement.
        self.priorities.append(1.0)

    def __len__(self) -> int:
        """Retourne le nombre de transitions memorisees."""

        return len(self.observations)


class SumTree:
    """Maintient la somme des priorites pour un echantillonnage rapide."""

    def __init__(self, capacity: int):
        """Initialise l'arbre de somme.

        Args:
            capacity (int): Nombre maximal d'elements stockes.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propage une variation de priorite jusqu'a la racine."""

        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrouve la feuille correspondant a une somme cumulative."""

        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Retourne la somme totale des priorites."""

        return float(self.tree[0])

    def add(self, priority: float, data: object) -> None:
        """Insere un element avec sa priorite courante.

        Args:
            priority (float): Priorite deja normalisee.
            data (object): Charge utile a stocker.
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        """Met a jour la priorite d'une feuille.

        Args:
            idx (int): Index feuille dans l'arbre.
            priority (float): Nouvelle priorite.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Recupere la feuille correspondant a une somme cumulative.

        Args:
            s (float): Somme cumulative cible.

        Returns:
            tuple[int, float, object]: Index feuille, priorite et charge utile.
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """Stocke des episodes MuZero avec echantillonnage par priorite."""

    def __init__(self, max_games: int, alpha: float = 0.6):
        """Initialise le buffer priorise.

        Args:
            max_games (int): Nombre maximal d'episodes conserves.
            alpha (float): Exposant de priorisation.
        """
        self.max_games = max_games
        self.alpha = alpha
        self.tree = SumTree(max_games)

    def save_game(self, game: GameHistory) -> None:
        """Persiste un episode complet dans l'arbre de priorites.

        Args:
            game (GameHistory): Episode a memoriser.
        """
        priority = np.max(game.priorities) if game.priorities else 1.0
        self.tree.add(float(priority**self.alpha), game)

    def sample(
        self,
        batch_size: int,
        num_unroll_steps: int = 5,
    ) -> List[Tuple[GameHistory, int, float]]:
        """Echantillonne un lot ``(game, start_idx, tree_idx)``.

        Args:
            batch_size (int): Taille du lot.
            num_unroll_steps (int): Longueur d'unroll necessaire.

        Returns:
            List[Tuple[GameHistory, int, float]]: Episodes et indices de
                depart associes.
        """
        batch = []
        segment = self.tree.total() / batch_size

        for index in range(batch_size):
            left = segment * index
            right = segment * (index + 1)
            sample_value = random.uniform(left, right)
            tree_idx, _priority, game = self.tree.get(sample_value)

            max_idx = max(0, len(game) - num_unroll_steps - 1)
            start_idx = random.randint(0, max_idx)
            batch.append((game, start_idx, tree_idx))

        return batch

    def update_priorities(self, indices, errors) -> None:
        """Met a jour les priorites d'episodes deja echantillonnes.

        Args:
            indices (list[int]): Indices feuilles dans la SumTree.
            errors (list[float]): Erreurs scalaires utilisees comme priorites.
        """
        for idx, error in zip(indices, errors):
            priority = (float(error) + 1e-5) ** self.alpha
            self.tree.update(int(idx), priority)

    def recent_games(self, limit: int) -> List[GameHistory]:
        """Retourne les episodes les plus recents encore presents.

        Args:
            limit (int): Nombre maximum d'episodes a renvoyer.

        Returns:
            List[GameHistory]: Episodes du plus recent au plus ancien.
        """
        if limit <= 0 or self.tree.n_entries <= 0:
            return []

        games: List[GameHistory] = []
        cursor = (self.tree.write - 1) % self.tree.capacity
        scanned = 0
        while scanned < self.tree.n_entries and len(games) < limit:
            game = self.tree.data[cursor]
            if game is not None:
                games.append(game)
            scanned += 1
            cursor = (cursor - 1) % self.tree.capacity
        return games

    @property
    def size(self) -> int:
        """Retourne le nombre d'episodes actuellement en memoire."""

        return self.tree.n_entries
