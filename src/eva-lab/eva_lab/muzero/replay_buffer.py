"""Buffer de replay priorise pour MuZero avec quotas de diversite."""

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
    metadata: dict[str, object] = field(default_factory=dict)

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

    @staticmethod
    def _metadata_bool(game: GameHistory, key: str) -> bool:
        """Retourne un booleen de metadata avec repli robuste."""

        return bool((game.metadata or {}).get(key, False))

    @staticmethod
    def _metadata_float(game: GameHistory, key: str) -> float:
        """Retourne un flottant de metadata avec repli robuste."""

        try:
            return float((game.metadata or {}).get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _iter_entries(self) -> list[tuple[int, float, GameHistory]]:
        """Retourne les episodes actuellement presents avec leurs priorites."""

        entries: list[tuple[int, float, GameHistory]] = []
        for data_idx in range(self.tree.n_entries):
            game = self.tree.data[data_idx]
            if game is None:
                continue
            tree_idx = data_idx + self.tree.capacity - 1
            entries.append((tree_idx, float(self.tree.tree[tree_idx]), game))
        return entries

    def _sample_prioritized_segments(
        self,
        batch_size: int,
        num_unroll_steps: int,
    ) -> List[Tuple[GameHistory, int, float]]:
        """Conserve le comportement historique si aucun quota n'est applicable."""

        batch = []
        segment = self.tree.total() / max(batch_size, 1)

        for index in range(batch_size):
            left = segment * index
            right = segment * (index + 1)
            sample_value = random.uniform(left, right)
            tree_idx, _priority, game = self.tree.get(sample_value)
            if game is None:
                continue
            max_idx = max(0, len(game) - num_unroll_steps - 1)
            start_idx = random.randint(0, max_idx)
            batch.append((game, start_idx, tree_idx))

        return batch

    @staticmethod
    def _weighted_pick_without_replacement(
        candidates: list[tuple[int, float, GameHistory]],
        count: int,
    ) -> list[tuple[int, float, GameHistory]]:
        """Selectionne des episodes sans remise en respectant les priorites."""

        if count <= 0 or not candidates:
            return []

        pool = list(candidates)
        selected: list[tuple[int, float, GameHistory]] = []
        while pool and len(selected) < count:
            weights = [max(priority, 1e-6) for _, priority, _ in pool]
            picked_index = random.choices(range(len(pool)), weights=weights, k=1)[0]
            selected.append(pool.pop(picked_index))
        return selected

    @staticmethod
    def _one_sided_bucket(game: GameHistory) -> str | None:
        """Retourne le bucket unidirectionnel d'un episode si applicable."""

        long_present = bool((game.metadata or {}).get("long_present", False))
        short_present = bool((game.metadata or {}).get("short_present", False))
        if long_present and not short_present:
            return "buy_only"
        if short_present and not long_present:
            return "sell_only"
        return None

    def save_game(self, game: GameHistory) -> None:
        """Persiste un episode complet dans l'arbre de priorites.

        Args:
            game (GameHistory): Episode a memoriser.
        """
        if len(game) <= 0:
            return
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
        entries = self._iter_entries()
        if not entries:
            return []

        metadata_ready = any(
            isinstance(game.metadata, dict) and "balanced_episode" in game.metadata
            for _, _, game in entries
        )
        if not metadata_ready:
            return self._sample_prioritized_segments(batch_size, num_unroll_steps)

        balanced_target = int(batch_size * 0.40)
        long_target = int(batch_size * 0.30)
        short_target = int(batch_size * 0.30)
        one_sided_cap = max(1, int(batch_size * 0.35))

        selected_tree_indices: set[int] = set()
        selected_entries: list[tuple[int, float, GameHistory]] = []
        one_sided_counts = {"buy_only": 0, "sell_only": 0}

        def add_from_bucket(
            candidates: list[tuple[int, float, GameHistory]],
            target_count: int,
        ) -> None:
            if target_count <= 0:
                return
            filtered = [
                item
                for item in candidates
                if item[0] not in selected_tree_indices
            ]
            for tree_idx, priority, game in self._weighted_pick_without_replacement(filtered, target_count):
                bucket = self._one_sided_bucket(game)
                if bucket and one_sided_counts[bucket] >= one_sided_cap:
                    continue
                selected_tree_indices.add(tree_idx)
                selected_entries.append((tree_idx, priority, game))
                if bucket:
                    one_sided_counts[bucket] += 1

        balanced_entries = [
            item
            for item in entries
            if self._metadata_bool(item[2], "balanced_episode")
        ]
        long_entries = [
            item
            for item in entries
            if self._metadata_bool(item[2], "long_present")
        ]
        short_entries = [
            item
            for item in entries
            if self._metadata_bool(item[2], "short_present")
        ]

        add_from_bucket(balanced_entries, balanced_target)
        add_from_bucket(long_entries, long_target)
        add_from_bucket(short_entries, short_target)

        remainder_target = max(0, batch_size - len(selected_entries))
        if remainder_target > 0:
            remainder_candidates = [
                item
                for item in entries
                if item[0] not in selected_tree_indices
            ]
            for tree_idx, priority, game in self._weighted_pick_without_replacement(
                remainder_candidates,
                len(remainder_candidates),
            ):
                if len(selected_entries) >= batch_size:
                    break
                bucket = self._one_sided_bucket(game)
                if bucket and one_sided_counts[bucket] >= one_sided_cap:
                    continue
                selected_tree_indices.add(tree_idx)
                selected_entries.append((tree_idx, priority, game))
                if bucket:
                    one_sided_counts[bucket] += 1

        batch: list[tuple[GameHistory, int, float]] = []
        for tree_idx, _priority, game in selected_entries:
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

    def diversity_stats(self) -> dict[str, float]:
        """Consolide les metriques de diversite du replay buffer.

        Returns:
            dict[str, float]: Resume global du buffer courant.
        """
        entries = self._iter_entries()
        total_games = len(entries)
        if total_games <= 0:
            return {
                "balanced_episode_rate": 0.0,
                "episodes_long_present_rate": 0.0,
                "episodes_short_present_rate": 0.0,
                "executed_long_entry_share": 0.0,
                "executed_short_entry_share": 0.0,
                "directional_imbalance": 1.0,
                "root_mask_rate": 0.0,
                "veto_to_hold_rate": 0.0,
                "post_veto_to_hold_rate": 0.0,
                "net_return_long_pct": 0.0,
                "net_return_short_pct": 0.0,
                "root_mask_blocked_buy_total": 0.0,
                "root_mask_blocked_sell_total": 0.0,
                "root_mask_blocked_buy_ema200": 0.0,
                "root_mask_blocked_sell_ema200": 0.0,
                "root_mask_blocked_buy_vwap": 0.0,
                "root_mask_blocked_sell_vwap": 0.0,
                "root_mask_blocked_buy_adx": 0.0,
                "root_mask_blocked_sell_adx": 0.0,
                "root_mask_blocked_buy_obv": 0.0,
                "root_mask_blocked_sell_obv": 0.0,
                "root_mask_blocked_buy_directional": 0.0,
                "root_mask_blocked_sell_directional": 0.0,
                "blocked_buy_total": 0.0,
                "blocked_sell_total": 0.0,
                "soft_entry_penalty_rate": 0.0,
                "soft_entry_bonus_rate": 0.0,
                "soft_entry_penalty_total": 0.0,
                "soft_entry_bonus_total": 0.0,
                "soft_penalty_net": 0.0,
                "soft_penalty_to_bonus_ratio": 0.0,
                "soft_penalty_ema200_count": 0.0,
                "soft_penalty_vwap_count": 0.0,
                "soft_penalty_adx_count": 0.0,
                "soft_penalty_obv_count": 0.0,
                "soft_penalty_ema_rate": 0.0,
                "soft_penalty_vwap_rate": 0.0,
                "soft_penalty_adx_rate": 0.0,
                "soft_penalty_obv_rate": 0.0,
                "directional_collapse": True,
                "long_entry_share": 0.0,
                "short_entry_share": 0.0,
            }

        balanced_games = 0
        long_present_games = 0
        short_present_games = 0
        total_long_entries = 0.0
        total_short_entries = 0.0
        root_mask_directional_candidates_total = 0.0
        root_mask_blocked_buy_total = 0.0
        root_mask_blocked_sell_total = 0.0
        root_mask_blocked_buy_ema200 = 0.0
        root_mask_blocked_sell_ema200 = 0.0
        root_mask_blocked_buy_vwap = 0.0
        root_mask_blocked_sell_vwap = 0.0
        root_mask_blocked_buy_adx = 0.0
        root_mask_blocked_sell_adx = 0.0
        root_mask_blocked_buy_obv = 0.0
        root_mask_blocked_sell_obv = 0.0
        root_mask_blocked_buy_directional = 0.0
        root_mask_blocked_sell_directional = 0.0
        blocked_buy_total = 0.0
        blocked_sell_total = 0.0
        entry_veto_total = 0.0
        requested_directional_total = 0.0
        soft_entry_penalty_count = 0.0
        soft_entry_bonus_count = 0.0
        soft_entry_penalty_total = 0.0
        soft_entry_bonus_total = 0.0
        soft_penalty_ema200_count = 0.0
        soft_penalty_vwap_count = 0.0
        soft_penalty_adx_count = 0.0
        soft_penalty_obv_count = 0.0
        long_return_sum = 0.0
        short_return_sum = 0.0
        long_return_games = 0
        short_return_games = 0

        for _, _priority, game in entries:
            metadata = dict(game.metadata or {})
            if bool(metadata.get("balanced_episode", False)):
                balanced_games += 1
            if bool(metadata.get("long_present", False)):
                long_present_games += 1
                long_return_sum += self._metadata_float(game, "net_return_long_pct")
                long_return_games += 1
            if bool(metadata.get("short_present", False)):
                short_present_games += 1
                short_return_sum += self._metadata_float(game, "net_return_short_pct")
                short_return_games += 1

            total_long_entries += self._metadata_float(game, "long_entries")
            total_short_entries += self._metadata_float(game, "short_entries")
            root_mask_directional_candidates_total += self._metadata_float(
                game,
                "root_mask_directional_candidates_total",
            )
            root_mask_blocked_buy_total += self._metadata_float(game, "root_mask_blocked_buy_total")
            root_mask_blocked_sell_total += self._metadata_float(game, "root_mask_blocked_sell_total")
            root_mask_blocked_buy_ema200 += self._metadata_float(game, "root_mask_blocked_buy_ema200")
            root_mask_blocked_sell_ema200 += self._metadata_float(game, "root_mask_blocked_sell_ema200")
            root_mask_blocked_buy_vwap += self._metadata_float(game, "root_mask_blocked_buy_vwap")
            root_mask_blocked_sell_vwap += self._metadata_float(game, "root_mask_blocked_sell_vwap")
            root_mask_blocked_buy_adx += self._metadata_float(game, "root_mask_blocked_buy_adx")
            root_mask_blocked_sell_adx += self._metadata_float(game, "root_mask_blocked_sell_adx")
            root_mask_blocked_buy_obv += self._metadata_float(game, "root_mask_blocked_buy_obv")
            root_mask_blocked_sell_obv += self._metadata_float(game, "root_mask_blocked_sell_obv")
            root_mask_blocked_buy_directional += self._metadata_float(game, "root_mask_blocked_buy_directional")
            root_mask_blocked_sell_directional += self._metadata_float(game, "root_mask_blocked_sell_directional")
            blocked_buy_total += self._metadata_float(game, "blocked_buy_entries")
            blocked_sell_total += self._metadata_float(game, "blocked_sell_entries")
            entry_veto_total += self._metadata_float(game, "entry_veto_to_hold")
            requested_directional_total += self._metadata_float(game, "requested_buy_actions")
            requested_directional_total += self._metadata_float(game, "requested_sell_actions")
            soft_entry_penalty_count += self._metadata_float(game, "soft_entry_penalty_count")
            soft_entry_bonus_count += self._metadata_float(game, "soft_entry_bonus_count")
            soft_entry_penalty_total += self._metadata_float(game, "soft_entry_penalty_total")
            soft_entry_bonus_total += self._metadata_float(game, "soft_entry_bonus_total")
            soft_penalty_ema200_count += self._metadata_float(game, "soft_penalty_ema200_count")
            soft_penalty_vwap_count += self._metadata_float(game, "soft_penalty_vwap_count")
            soft_penalty_adx_count += self._metadata_float(game, "soft_penalty_adx_count")
            soft_penalty_obv_count += self._metadata_float(game, "soft_penalty_obv_count")

        directional_entries = total_long_entries + total_short_entries
        executed_long_entry_share = total_long_entries / max(directional_entries, 1.0)
        executed_short_entry_share = total_short_entries / max(directional_entries, 1.0)
        directional_imbalance = (
            abs(total_long_entries - total_short_entries) / directional_entries
            if directional_entries > 0.0
            else 1.0
        )
        return {
            "balanced_episode_rate": (balanced_games / total_games) * 100.0,
            "episodes_long_present_rate": (long_present_games / total_games) * 100.0,
            "episodes_short_present_rate": (short_present_games / total_games) * 100.0,
            "executed_long_entry_share": executed_long_entry_share,
            "executed_short_entry_share": executed_short_entry_share,
            "directional_imbalance": directional_imbalance,
            "root_mask_rate": (
                (root_mask_blocked_buy_total + root_mask_blocked_sell_total)
                / max(root_mask_directional_candidates_total, 1.0)
            ),
            "veto_to_hold_rate": entry_veto_total / max(requested_directional_total, 1.0),
            "post_veto_to_hold_rate": entry_veto_total / max(requested_directional_total, 1.0),
            "net_return_long_pct": long_return_sum / max(long_return_games, 1),
            "net_return_short_pct": short_return_sum / max(short_return_games, 1),
            "root_mask_blocked_buy_total": root_mask_blocked_buy_total,
            "root_mask_blocked_sell_total": root_mask_blocked_sell_total,
            "root_mask_blocked_buy_ema200": root_mask_blocked_buy_ema200,
            "root_mask_blocked_sell_ema200": root_mask_blocked_sell_ema200,
            "root_mask_blocked_buy_vwap": root_mask_blocked_buy_vwap,
            "root_mask_blocked_sell_vwap": root_mask_blocked_sell_vwap,
            "root_mask_blocked_buy_adx": root_mask_blocked_buy_adx,
            "root_mask_blocked_sell_adx": root_mask_blocked_sell_adx,
            "root_mask_blocked_buy_obv": root_mask_blocked_buy_obv,
            "root_mask_blocked_sell_obv": root_mask_blocked_sell_obv,
            "root_mask_blocked_buy_directional": root_mask_blocked_buy_directional,
            "root_mask_blocked_sell_directional": root_mask_blocked_sell_directional,
            "blocked_buy_total": blocked_buy_total,
            "blocked_sell_total": blocked_sell_total,
            "soft_entry_penalty_rate": soft_entry_penalty_count / max(directional_entries, 1.0),
            "soft_entry_bonus_rate": soft_entry_bonus_count / max(directional_entries, 1.0),
            "soft_entry_penalty_total": soft_entry_penalty_total,
            "soft_entry_bonus_total": soft_entry_bonus_total,
            "soft_penalty_net": soft_entry_penalty_total - soft_entry_bonus_total,
            "soft_penalty_to_bonus_ratio": (
                soft_entry_penalty_total / max(soft_entry_bonus_total, 1e-6)
                if (soft_entry_penalty_total > 0.0 or soft_entry_bonus_total > 0.0)
                else 0.0
            ),
            "soft_penalty_ema200_count": soft_penalty_ema200_count,
            "soft_penalty_vwap_count": soft_penalty_vwap_count,
            "soft_penalty_adx_count": soft_penalty_adx_count,
            "soft_penalty_obv_count": soft_penalty_obv_count,
            "soft_penalty_ema_rate": soft_penalty_ema200_count / max(directional_entries, 1.0),
            "soft_penalty_vwap_rate": soft_penalty_vwap_count / max(directional_entries, 1.0),
            "soft_penalty_adx_rate": soft_penalty_adx_count / max(directional_entries, 1.0),
            "soft_penalty_obv_rate": soft_penalty_obv_count / max(directional_entries, 1.0),
            "directional_collapse": bool(total_long_entries <= 0.0 or total_short_entries <= 0.0),
            "long_entry_share": executed_long_entry_share,
            "short_entry_share": executed_short_entry_share,
        }

    @property
    def size(self) -> int:
        """Retourne le nombre d'episodes actuellement en memoire."""

        return self.tree.n_entries
