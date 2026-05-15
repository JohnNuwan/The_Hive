"""Buffer de replay priorise pour MuZero avec quotas de diversite."""

from __future__ import annotations

import math
import os
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

    @staticmethod
    def _metadata_str(game: GameHistory, key: str) -> str:
        """Retourne une chaine de metadata avec repli robuste."""

        return str((game.metadata or {}).get(key, "") or "").strip()

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

    @staticmethod
    def _hard_negative_type(game: GameHistory) -> str | None:
        """Retourne le type d'echec prioritaire d'un episode si applicable.

        Args:
            game (GameHistory): Episode MuZero analyse.

        Returns:
            str | None: Type d'echec retenu, ou ``None`` si l'episode n'est
                pas considere comme hard-negative.
        """
        metadata = dict(game.metadata or {})
        nemesis_type = str(metadata.get("nemesis_type") or "").strip().upper()
        if nemesis_type and nemesis_type != "NONE":
            return nemesis_type
        if bool(metadata.get("liquidity_trap_loss", False)):
            return "LIQUIDITY_TRAP"
        if bool(metadata.get("bad_runner_exit", False)):
            return "BAD_RUNNER_EXIT"
        if bool(metadata.get("bad_pyramid_exit", False)):
            return "BAD_PYRAMID_EXIT"
        if bool(metadata.get("range_entry_loss", False)):
            return "RANGE_ENTRY_LOSS"
        if bool(metadata.get("hard_stop_exit", False)):
            return "HARD_STOP_EXIT"
        if bool(metadata.get("bad_split", False)):
            return "BAD_SPLIT"
        return None

    @classmethod
    def _is_hard_negative(cls, game: GameHistory) -> bool:
        """Indique si un episode fait partie du replay correctif prioritaire.

        Args:
            game (GameHistory): Episode MuZero analyse.

        Returns:
            bool: ``True`` si l'episode porte un tag d'echec cible.
        """
        return cls._hard_negative_type(game) is not None

    @classmethod
    def _offensive_curriculum_multiplier(cls, game: GameHistory) -> float:
        """Calcule le boost de replay pour les episodes offensifs utiles.

        Args:
            game (GameHistory): Episode MuZero analyse.

        Returns:
            float: Multiplicateur borne applique a la priorite.
        """
        if str(os.getenv("MUZERO_REPLAY_OFFENSIVE_CURRICULUM", "1")).strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return 1.0

        metadata = dict(game.metadata or {})
        boost = 1.0
        boost += min(cls._metadata_float(game, "soft_tp_hit_count"), 3.0) * 0.06
        boost += min(cls._metadata_float(game, "full_tp_hit_count"), 2.0) * 0.08
        boost += min(cls._metadata_float(game, "split_monetization_capture_count"), 3.0) * 0.12
        boost += min(cls._metadata_float(game, "runner_profit_hold_capture_count"), 3.0) * 0.14
        boost += min(cls._metadata_float(game, "runner_extension_capture_count"), 3.0) * 0.10
        boost += min(cls._metadata_float(game, "pyramid_monetization_capture_count"), 3.0) * 0.10
        boost += min(cls._metadata_float(game, "pyramid_add_capture_count"), 3.0) * 0.08
        boost += max(0.0, cls._metadata_float(game, "close_quality_score")) * 0.20
        boost += max(0.0, cls._metadata_float(game, "return_pct")) * 0.02
        if float(metadata.get("profit_factor", 0.0) or 0.0) > 1.0:
            boost += min(float(metadata.get("profit_factor", 1.0) or 1.0) - 1.0, 2.0) * 0.12
        return max(1.0, min(boost, 2.50))

    def save_game(self, game: GameHistory) -> None:
        """Persiste un episode complet dans l'arbre de priorites.

        Args:
            game (GameHistory): Episode a memoriser.
        """
        if len(game) <= 0:
            return
        priority = np.max(game.priorities) if game.priorities else 1.0
        priority *= self._offensive_curriculum_multiplier(game)
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
        hard_negative_ratio = min(
            0.50,
            max(0.0, float(os.getenv("MUZERO_REPLAY_HARD_NEGATIVE_RATIO", "0.20"))),
        )
        hard_negative_type_cap_ratio = min(
            hard_negative_ratio,
            max(0.0, float(os.getenv("MUZERO_REPLAY_HARD_NEGATIVE_TYPE_CAP", "0.08"))),
        )
        hard_negative_target = max(
            0,
            min(batch_size, int(math.ceil(batch_size * hard_negative_ratio))),
        )
        hard_negative_type_cap = max(
            1,
            int(math.ceil(batch_size * hard_negative_type_cap_ratio)),
        )
        symbols = sorted(
            {
                self._metadata_str(game, "symbol")
                for _, _, game in entries
                if self._metadata_str(game, "symbol")
            }
        )
        symbol_cap = (
            batch_size
            if len(symbols) < 4
            else max(1, int(math.ceil(batch_size * 0.25)))
        )

        selected_tree_indices: set[int] = set()
        selected_entries: list[tuple[int, float, GameHistory]] = []
        one_sided_counts = {"buy_only": 0, "sell_only": 0}
        symbol_counts: dict[str, int] = {}
        hard_negative_counts: dict[str, int] = {}

        def can_accept(
            game: GameHistory,
            *,
            enforce_symbol_cap: bool = True,
            enforce_hard_negative_cap: bool = True,
        ) -> bool:
            bucket = self._one_sided_bucket(game)
            if bucket and one_sided_counts[bucket] >= one_sided_cap:
                return False
            symbol = self._metadata_str(game, "symbol")
            if enforce_symbol_cap and symbol and len(symbols) >= 4:
                if symbol_counts.get(symbol, 0) >= symbol_cap:
                    return False
            hard_negative_type = self._hard_negative_type(game)
            if (
                enforce_hard_negative_cap
                and hard_negative_type
                and hard_negative_counts.get(hard_negative_type, 0) >= hard_negative_type_cap
            ):
                return False
            return True

        def register_entry(tree_idx: int, priority: float, game: GameHistory) -> bool:
            if tree_idx in selected_tree_indices:
                return False
            if not can_accept(game):
                return False
            selected_tree_indices.add(tree_idx)
            selected_entries.append((tree_idx, priority, game))
            bucket = self._one_sided_bucket(game)
            if bucket:
                one_sided_counts[bucket] += 1
            symbol = self._metadata_str(game, "symbol")
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            hard_negative_type = self._hard_negative_type(game)
            if hard_negative_type:
                hard_negative_counts[hard_negative_type] = (
                    hard_negative_counts.get(hard_negative_type, 0) + 1
                )
            return True

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
                register_entry(tree_idx, priority, game)

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

        if symbols and batch_size > 0:
            symbol_buckets: dict[str, list[tuple[int, float, GameHistory]]] = {}
            for entry in entries:
                symbol = self._metadata_str(entry[2], "symbol")
                if not symbol:
                    continue
                symbol_buckets.setdefault(symbol, []).append(entry)
            for symbol in symbols:
                if len(selected_entries) >= batch_size:
                    break
                bucket_entries = list(symbol_buckets.get(symbol) or [])
                if not bucket_entries:
                    continue
                for tree_idx, priority, game in self._weighted_pick_without_replacement(bucket_entries, 1):
                    register_entry(tree_idx, priority, game)

        hard_negative_selected = 0
        hard_negative_candidates = [
            item
            for item in entries
            if item[0] not in selected_tree_indices and self._is_hard_negative(item[2])
        ]
        for tree_idx, priority, game in self._weighted_pick_without_replacement(
            hard_negative_candidates,
            len(hard_negative_candidates),
        ):
            if len(selected_entries) >= batch_size or hard_negative_selected >= hard_negative_target:
                break
            if register_entry(tree_idx, priority, game):
                hard_negative_selected += 1

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
                register_entry(tree_idx, priority, game)

        if len(selected_entries) < batch_size:
            fallback_candidates = [
                item
                for item in entries
                if item[0] not in selected_tree_indices
            ]
            for tree_idx, priority, game in self._weighted_pick_without_replacement(
                fallback_candidates,
                len(fallback_candidates),
            ):
                if len(selected_entries) >= batch_size:
                    break
                if not can_accept(game, enforce_symbol_cap=False, enforce_hard_negative_cap=False):
                    continue
                selected_tree_indices.add(tree_idx)
                selected_entries.append((tree_idx, priority, game))
                bucket = self._one_sided_bucket(game)
                if bucket:
                    one_sided_counts[bucket] += 1
                symbol = self._metadata_str(game, "symbol")
                if symbol:
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                hard_negative_type = self._hard_negative_type(game)
                if hard_negative_type:
                    hard_negative_counts[hard_negative_type] = (
                        hard_negative_counts.get(hard_negative_type, 0) + 1
                    )

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
                "root_mask_ema200_share": 0.0,
                "root_mask_vwap_share": 0.0,
                "root_mask_adx_share": 0.0,
                "root_mask_directional_share": 0.0,
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
                "hold_drag_opportunity_count": 0.0,
                "hold_drag_penalized_count": 0.0,
                "hold_drag_score": 0.0,
                "split_opportunity_count": 0.0,
                "split_executed": 0.0,
                "split_profitable_count": 0.0,
                "split_efficiency": 0.0,
                "pyramid_opportunity_count": 0.0,
                "pyramids_opened": 0.0,
                "pyramid_profitable_count": 0.0,
                "pyramid_efficiency": 0.0,
                "slbe_triggered": 0.0,
                "slbe_profitable_exits": 0.0,
                "slbe_lock_profit_count": 0.0,
                "slbe_capture_rate": 0.0,
                "close_winner_count": 0.0,
                "close_loser_count": 0.0,
                "close_quality_score": 0.0,
                "tp_like_exit_count": 0.0,
                "tp_like_missed_count": 0.0,
                "defensive_close_count": 0.0,
                "early_close_noise_count": 0.0,
                "hard_stop_exit_count": 0.0,
                "soft_tp_hit_count": 0.0,
                "full_tp_hit_count": 0.0,
                "time_stop_trigger_count": 0.0,
                "runner_extension_count": 0.0,
                "runner_managed_exit_count": 0.0,
                "runner_exit_profitable_count": 0.0,
                "runner_forced_stop_count": 0.0,
                "forced_stop_near_miss_count": 0.0,
                "split_runner_capture_rate": 0.0,
                "split_zone_capture_rate": 0.0,
                "split_monetization_capture_rate": 0.0,
                "split_runner_profitable_count": 0.0,
                "split_runner_failed_count": 0.0,
                "split_early_count": 0.0,
                "split_decorative_count": 0.0,
                "split_trade_value_delta": 0.0,
                "split_improved_total_trade_count": 0.0,
                "split_tp_zone_opportunity_count": 0.0,
                "split_monetization_window_count": 0.0,
                "pyramid_entry_quality_score": 0.0,
                "pyramid_exit_capture_rate": 0.0,
                "pyramid_add_capture_rate": 0.0,
                "pyramid_monetization_capture_rate": 0.0,
                "pyramid_good_add_count": 0.0,
                "pyramid_bad_add_count": 0.0,
                "pyramid_profitable_exit_count": 0.0,
                "pyramid_total_trade_improvement_pct": 0.0,
                "pyramid_failed_to_improve_count": 0.0,
                "pyramid_add_opportunity_count": 0.0,
                "pyramid_monetization_window_count": 0.0,
                "pyramid_missed_add_count": 0.0,
                "runner_extension_capture_rate": 0.0,
                "runner_extension_opportunity_count": 0.0,
                "runner_profit_hold_capture_rate": 0.0,
                "runner_profit_hold_window_count": 0.0,
                "runner_viable_window_count": 0.0,
                "runner_hold_after_soft_tp_count": 0.0,
                "runner_viable_but_closed_count": 0.0,
                "early_full_close_after_soft_tp_count": 0.0,
                "runner_missed_extension_count": 0.0,
                "runner_retained_profit_pct": 0.0,
                "runner_retained_profit_score": 0.0,
                "runner_giveback_pct": 0.0,
                "runner_giveback_ratio": 0.0,
                "profit_peak_reached_count": 0.0,
                "profit_peak_giveback_ratio": 0.0,
                "close_quality_by_symbol": {},
                "close_events_by_symbol": {},
                "split_efficiency_by_symbol": {},
                "split_runner_capture_rate_by_symbol": {},
                "pyramid_efficiency_by_symbol": {},
                "pyramid_entry_quality_score_by_symbol": {},
                "pyramid_exit_capture_rate_by_symbol": {},
                "slbe_capture_rate_by_symbol": {},
                "hold_drag_score_by_symbol": {},
                "root_mask_share_by_symbol": {},
                "root_mask_ema200_share_by_symbol": {},
                "root_mask_vwap_share_by_symbol": {},
                "root_mask_adx_share_by_symbol": {},
                "root_mask_directional_share_by_symbol": {},
                "hard_negative_mix": {},
                "nemesis_mix": {},
                "liquidity_trap_share": 0.0,
                "bad_runner_share": 0.0,
                "bad_pyramid_share": 0.0,
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
        hold_drag_opportunity_count = 0.0
        hold_drag_penalized_count = 0.0
        split_opportunity_count = 0.0
        split_executed = 0.0
        split_profitable_count = 0.0
        split_runner_profitable_count = 0.0
        split_runner_failed_count = 0.0
        split_early_count = 0.0
        split_decorative_count = 0.0
        split_trade_value_delta_total = 0.0
        split_improved_total_trade_count = 0.0
        split_tp_zone_opportunity_count = 0.0
        split_monetization_window_count = 0.0
        split_monetization_capture_count = 0.0
        pyramid_opportunity_count = 0.0
        pyramid_add_opportunity_count = 0.0
        pyramid_add_capture_count = 0.0
        pyramid_monetization_window_count = 0.0
        pyramid_monetization_capture_count = 0.0
        pyramid_missed_add_count = 0.0
        pyramids_opened = 0.0
        pyramid_profitable_count = 0.0
        pyramid_good_add_count = 0.0
        pyramid_bad_add_count = 0.0
        pyramid_profitable_exit_count = 0.0
        pyramid_total_trade_improvement_total = 0.0
        pyramid_failed_to_improve_count = 0.0
        slbe_triggered = 0.0
        slbe_profitable_exits = 0.0
        slbe_lock_profit_count = 0.0
        close_winner_count = 0.0
        close_loser_count = 0.0
        tp_like_exit_count = 0.0
        tp_like_missed_count = 0.0
        defensive_close_count = 0.0
        early_close_noise_count = 0.0
        hard_stop_exit_count = 0.0
        soft_tp_hit_count = 0.0
        full_tp_hit_count = 0.0
        time_stop_trigger_count = 0.0
        runner_extension_count = 0.0
        runner_extension_opportunity_count = 0.0
        runner_extension_capture_count = 0.0
        runner_profit_hold_window_count = 0.0
        runner_profit_hold_capture_count = 0.0
        runner_viable_window_count = 0.0
        runner_hold_after_soft_tp_count = 0.0
        runner_viable_but_closed_count = 0.0
        early_full_close_after_soft_tp_count = 0.0
        runner_missed_extension_count = 0.0
        runner_managed_exit_count = 0.0
        runner_exit_profitable_count = 0.0
        runner_forced_stop_count = 0.0
        runner_retained_profit_total = 0.0
        runner_retained_profit_score_total = 0.0
        runner_giveback_total = 0.0
        profit_peak_reached_count = 0.0
        profit_peak_giveback_ratio_total = 0.0
        profit_peak_giveback_ratio_observations = 0.0
        forced_stop_near_miss_count = 0.0
        long_return_sum = 0.0
        short_return_sum = 0.0
        long_return_games = 0
        short_return_games = 0
        hard_negative_total = 0
        hard_negative_counts: dict[str, int] = {}
        nemesis_counts: dict[str, int] = {}
        symbol_counters: dict[str, dict[str, float]] = {}

        for _, _priority, game in entries:
            metadata = dict(game.metadata or {})
            symbol = self._metadata_str(game, "symbol") or "unknown"
            hard_negative_type = self._hard_negative_type(game)
            if hard_negative_type:
                hard_negative_total += 1
                hard_negative_counts[hard_negative_type] = (
                    hard_negative_counts.get(hard_negative_type, 0) + 1
                )
            nemesis_type = str(metadata.get("nemesis_type") or "").strip().upper()
            if nemesis_type and nemesis_type != "NONE":
                nemesis_counts[nemesis_type] = nemesis_counts.get(nemesis_type, 0) + 1
            symbol_counter = symbol_counters.setdefault(
                symbol,
                {
                    "close_winner_count": 0.0,
                    "close_loser_count": 0.0,
                    "root_mask_directional_candidates_total": 0.0,
                    "root_mask_blocked_buy_total": 0.0,
                    "root_mask_blocked_sell_total": 0.0,
                    "root_mask_blocked_buy_ema200": 0.0,
                    "root_mask_blocked_sell_ema200": 0.0,
                    "root_mask_blocked_buy_vwap": 0.0,
                    "root_mask_blocked_sell_vwap": 0.0,
                    "root_mask_blocked_buy_adx": 0.0,
                    "root_mask_blocked_sell_adx": 0.0,
                    "root_mask_blocked_buy_directional": 0.0,
                    "root_mask_blocked_sell_directional": 0.0,
                    "split_executed": 0.0,
                    "split_profitable_count": 0.0,
                    "split_runner_profitable_count": 0.0,
                    "split_trade_value_delta": 0.0,
                    "pyramids_opened": 0.0,
                    "pyramid_profitable_count": 0.0,
                    "pyramid_good_add_count": 0.0,
                    "pyramid_profitable_exit_count": 0.0,
                    "pyramid_total_trade_improvement_pct": 0.0,
                    "slbe_triggered": 0.0,
                    "slbe_profitable_exits": 0.0,
                    "hold_drag_opportunity_count": 0.0,
                    "hold_drag_penalized_count": 0.0,
                },
            )
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
            hold_drag_opportunity_count += self._metadata_float(game, "hold_drag_opportunity_count")
            hold_drag_penalized_count += self._metadata_float(game, "hold_drag_penalized_count")
            split_opportunity_count += self._metadata_float(game, "split_opportunity_count")
            split_executed += self._metadata_float(game, "split_executed")
            split_profitable_count += self._metadata_float(game, "split_profitable_count")
            split_runner_profitable_count += self._metadata_float(game, "split_runner_profitable_count")
            split_runner_failed_count += self._metadata_float(game, "split_runner_failed_count")
            split_early_count += self._metadata_float(game, "split_early_count")
            split_decorative_count += self._metadata_float(game, "split_decorative_count")
            split_trade_value_delta_total += self._metadata_float(game, "split_trade_value_delta")
            split_improved_total_trade_count += self._metadata_float(
                game,
                "split_improved_total_trade_count",
            )
            split_tp_zone_opportunity_count += self._metadata_float(
                game,
                "split_tp_zone_opportunity_count",
            )
            split_monetization_window_count += self._metadata_float(
                game,
                "split_monetization_window_count",
            )
            split_monetization_capture_count += self._metadata_float(
                game,
                "split_monetization_capture_rate",
            ) * self._metadata_float(game, "split_monetization_window_count")
            pyramid_opportunity_count += self._metadata_float(game, "pyramid_opportunity_count")
            pyramid_add_opportunity_count += self._metadata_float(
                game,
                "pyramid_add_opportunity_count",
            )
            pyramid_add_capture_count += self._metadata_float(game, "pyramid_add_capture_count")
            pyramid_monetization_window_count += self._metadata_float(
                game,
                "pyramid_monetization_window_count",
            )
            pyramid_monetization_capture_count += self._metadata_float(
                game,
                "pyramid_monetization_capture_rate",
            ) * self._metadata_float(game, "pyramid_monetization_window_count")
            pyramid_missed_add_count += self._metadata_float(game, "pyramid_missed_add_count")
            pyramids_opened += self._metadata_float(game, "pyramids_opened")
            pyramid_profitable_count += self._metadata_float(game, "pyramid_profitable_count")
            pyramid_good_add_count += self._metadata_float(game, "pyramid_good_add_count")
            pyramid_bad_add_count += self._metadata_float(game, "pyramid_bad_add_count")
            pyramid_profitable_exit_count += self._metadata_float(game, "pyramid_profitable_exit_count")
            pyramid_total_trade_improvement_total += self._metadata_float(
                game,
                "pyramid_total_trade_improvement_pct",
            )
            pyramid_failed_to_improve_count += self._metadata_float(
                game,
                "pyramid_failed_to_improve_count",
            )
            slbe_triggered += self._metadata_float(game, "slbe_triggered")
            slbe_profitable_exits += self._metadata_float(game, "slbe_profitable_exits")
            slbe_lock_profit_count += self._metadata_float(game, "slbe_lock_profit_count")
            close_winner_count += self._metadata_float(game, "close_winner_count")
            close_loser_count += self._metadata_float(game, "close_loser_count")
            tp_like_exit_count += self._metadata_float(game, "tp_like_exit_count")
            tp_like_missed_count += self._metadata_float(game, "tp_like_missed_count")
            defensive_close_count += self._metadata_float(game, "defensive_close_count")
            early_close_noise_count += self._metadata_float(game, "early_close_noise_count")
            hard_stop_exit_count += self._metadata_float(game, "hard_stop_exit_count")
            soft_tp_hit_count += self._metadata_float(game, "soft_tp_hit_count")
            full_tp_hit_count += self._metadata_float(game, "full_tp_hit_count")
            time_stop_trigger_count += self._metadata_float(game, "time_stop_trigger_count")
            runner_extension_count += self._metadata_float(game, "runner_extension_count")
            runner_extension_opportunity_count += self._metadata_float(
                game,
                "runner_extension_opportunity_count",
            )
            runner_extension_capture_count += self._metadata_float(
                game,
                "runner_extension_capture_rate",
            ) * self._metadata_float(game, "runner_extension_opportunity_count")
            runner_profit_hold_window_count += self._metadata_float(
                game,
                "runner_profit_hold_window_count",
            )
            runner_profit_hold_capture_count += self._metadata_float(
                game,
                "runner_profit_hold_capture_rate",
            ) * self._metadata_float(game, "runner_profit_hold_window_count")
            runner_viable_window_count += self._metadata_float(game, "runner_viable_window_count")
            runner_hold_after_soft_tp_count += self._metadata_float(
                game,
                "runner_hold_after_soft_tp_count",
            )
            runner_viable_but_closed_count += self._metadata_float(
                game,
                "runner_viable_but_closed_count",
            )
            early_full_close_after_soft_tp_count += self._metadata_float(
                game,
                "early_full_close_after_soft_tp_count",
            )
            runner_missed_extension_count += self._metadata_float(game, "runner_missed_extension_count")
            runner_managed_exit_count += self._metadata_float(game, "runner_managed_exit_count")
            runner_exit_profitable_count += self._metadata_float(game, "runner_exit_profitable_count")
            runner_forced_stop_count += self._metadata_float(game, "runner_forced_stop_count")
            runner_retained_profit_total += self._metadata_float(game, "runner_retained_profit_pct")
            runner_retained_profit_score_total += self._metadata_float(
                game,
                "runner_retained_profit_score",
            ) * (
                self._metadata_float(game, "runner_profit_hold_capture_rate")
                * self._metadata_float(game, "runner_profit_hold_window_count")
            )
            runner_giveback_total += self._metadata_float(game, "runner_giveback_pct")
            profit_peak_reached_count += self._metadata_float(game, "profit_peak_reached_count")
            profit_peak_giveback_ratio_total += self._metadata_float(
                game,
                "profit_peak_giveback_ratio",
            ) * self._metadata_float(game, "profit_peak_reached_count")
            profit_peak_giveback_ratio_observations += self._metadata_float(
                game,
                "profit_peak_reached_count",
            )
            forced_stop_near_miss_count += self._metadata_float(game, "forced_stop_near_miss_count")

            symbol_counter["root_mask_directional_candidates_total"] += self._metadata_float(
                game,
                "root_mask_directional_candidates_total",
            )
            symbol_counter["root_mask_blocked_buy_total"] += self._metadata_float(game, "root_mask_blocked_buy_total")
            symbol_counter["root_mask_blocked_sell_total"] += self._metadata_float(game, "root_mask_blocked_sell_total")
            symbol_counter["root_mask_blocked_buy_ema200"] += self._metadata_float(
                game,
                "root_mask_blocked_buy_ema200",
            )
            symbol_counter["root_mask_blocked_sell_ema200"] += self._metadata_float(
                game,
                "root_mask_blocked_sell_ema200",
            )
            symbol_counter["root_mask_blocked_buy_vwap"] += self._metadata_float(game, "root_mask_blocked_buy_vwap")
            symbol_counter["root_mask_blocked_sell_vwap"] += self._metadata_float(game, "root_mask_blocked_sell_vwap")
            symbol_counter["root_mask_blocked_buy_adx"] += self._metadata_float(game, "root_mask_blocked_buy_adx")
            symbol_counter["root_mask_blocked_sell_adx"] += self._metadata_float(game, "root_mask_blocked_sell_adx")
            symbol_counter["root_mask_blocked_buy_directional"] += self._metadata_float(
                game,
                "root_mask_blocked_buy_directional",
            )
            symbol_counter["root_mask_blocked_sell_directional"] += self._metadata_float(
                game,
                "root_mask_blocked_sell_directional",
            )
            symbol_counter["close_winner_count"] += self._metadata_float(game, "close_winner_count")
            symbol_counter["close_loser_count"] += self._metadata_float(game, "close_loser_count")
            symbol_counter["split_executed"] += self._metadata_float(game, "split_executed")
            symbol_counter["split_profitable_count"] += self._metadata_float(game, "split_profitable_count")
            symbol_counter["split_runner_profitable_count"] += self._metadata_float(game, "split_runner_profitable_count")
            symbol_counter["split_trade_value_delta"] += self._metadata_float(game, "split_trade_value_delta")
            symbol_counter["pyramids_opened"] += self._metadata_float(game, "pyramids_opened")
            symbol_counter["pyramid_profitable_count"] += self._metadata_float(game, "pyramid_profitable_count")
            symbol_counter["pyramid_good_add_count"] += self._metadata_float(game, "pyramid_good_add_count")
            symbol_counter["pyramid_profitable_exit_count"] += self._metadata_float(game, "pyramid_profitable_exit_count")
            symbol_counter["pyramid_total_trade_improvement_pct"] += self._metadata_float(
                game,
                "pyramid_total_trade_improvement_pct",
            )
            symbol_counter["slbe_triggered"] += self._metadata_float(game, "slbe_triggered")
            symbol_counter["slbe_profitable_exits"] += self._metadata_float(game, "slbe_profitable_exits")
            symbol_counter["hold_drag_opportunity_count"] += self._metadata_float(game, "hold_drag_opportunity_count")
            symbol_counter["hold_drag_penalized_count"] += self._metadata_float(game, "hold_drag_penalized_count")

        directional_entries = total_long_entries + total_short_entries
        executed_long_entry_share = total_long_entries / max(directional_entries, 1.0)
        executed_short_entry_share = total_short_entries / max(directional_entries, 1.0)
        directional_imbalance = (
            abs(total_long_entries - total_short_entries) / directional_entries
            if directional_entries > 0.0
            else 1.0
        )
        close_quality_by_symbol = {
            symbol: (
                counters["close_winner_count"]
                / max(counters["close_winner_count"] + counters["close_loser_count"], 1.0)
                if (counters["close_winner_count"] + counters["close_loser_count"]) > 0.0
                else 0.0
            )
            for symbol, counters in symbol_counters.items()
        }
        close_events_by_symbol = {
            symbol: int(counters["close_winner_count"] + counters["close_loser_count"])
            for symbol, counters in symbol_counters.items()
        }
        split_efficiency_by_symbol = {
            symbol: (
                counters["split_profitable_count"] / max(counters["split_executed"], 1.0)
                if counters["split_executed"] > 0.0
                else 0.0
            )
            for symbol, counters in symbol_counters.items()
        }
        split_runner_capture_rate_by_symbol = {
            symbol: (
                counters["split_runner_profitable_count"] / max(counters["split_executed"], 1.0)
                if counters["split_executed"] > 0.0
                else 0.0
            )
            for symbol, counters in symbol_counters.items()
        }
        pyramid_efficiency_by_symbol = {
            symbol: (
                counters["pyramid_profitable_count"] / max(counters["pyramids_opened"], 1.0)
                if counters["pyramids_opened"] > 0.0
                else 0.0
            )
            for symbol, counters in symbol_counters.items()
        }
        pyramid_entry_quality_score_by_symbol = {
            symbol: (
                counters["pyramid_good_add_count"] / max(counters["pyramids_opened"], 1.0)
                if counters["pyramids_opened"] > 0.0
                else 0.0
            )
            for symbol, counters in symbol_counters.items()
        }
        pyramid_exit_capture_rate_by_symbol = {
            symbol: (
                counters["pyramid_profitable_exit_count"] / max(counters["pyramids_opened"], 1.0)
                if counters["pyramids_opened"] > 0.0
                else 0.0
            )
            for symbol, counters in symbol_counters.items()
        }
        slbe_capture_rate_by_symbol = {
            symbol: (
                counters["slbe_profitable_exits"] / max(counters["slbe_triggered"], 1.0)
                if counters["slbe_triggered"] > 0.0
                else 0.0
            )
            for symbol, counters in symbol_counters.items()
        }
        hold_drag_score_by_symbol = {
            symbol: (
                counters["hold_drag_penalized_count"] / max(counters["hold_drag_opportunity_count"], 1.0)
            )
            for symbol, counters in symbol_counters.items()
        }
        root_mask_share_by_symbol = {
            symbol: (
                (counters["root_mask_blocked_buy_total"] + counters["root_mask_blocked_sell_total"])
                / max(counters["root_mask_directional_candidates_total"], 1.0)
            )
            for symbol, counters in symbol_counters.items()
        }
        root_mask_ema200_share_by_symbol = {
            symbol: (
                (counters["root_mask_blocked_buy_ema200"] + counters["root_mask_blocked_sell_ema200"])
                / max(counters["root_mask_blocked_buy_total"] + counters["root_mask_blocked_sell_total"], 1.0)
            )
            for symbol, counters in symbol_counters.items()
        }
        root_mask_vwap_share_by_symbol = {
            symbol: (
                (counters["root_mask_blocked_buy_vwap"] + counters["root_mask_blocked_sell_vwap"])
                / max(counters["root_mask_blocked_buy_total"] + counters["root_mask_blocked_sell_total"], 1.0)
            )
            for symbol, counters in symbol_counters.items()
        }
        root_mask_adx_share_by_symbol = {
            symbol: (
                (counters["root_mask_blocked_buy_adx"] + counters["root_mask_blocked_sell_adx"])
                / max(counters["root_mask_blocked_buy_total"] + counters["root_mask_blocked_sell_total"], 1.0)
            )
            for symbol, counters in symbol_counters.items()
        }
        root_mask_directional_share_by_symbol = {
            symbol: (
                (counters["root_mask_blocked_buy_directional"] + counters["root_mask_blocked_sell_directional"])
                / max(counters["root_mask_blocked_buy_total"] + counters["root_mask_blocked_sell_total"], 1.0)
            )
            for symbol, counters in symbol_counters.items()
        }
        hard_negative_mix = {
            hard_negative_type: count / max(hard_negative_total, 1)
            for hard_negative_type, count in sorted(hard_negative_counts.items())
        }
        nemesis_mix = {
            nemesis_type: count / max(total_games, 1)
            for nemesis_type, count in sorted(nemesis_counts.items())
        }
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
            "root_mask_ema200_share": (
                (root_mask_blocked_buy_ema200 + root_mask_blocked_sell_ema200)
                / max(root_mask_blocked_buy_total + root_mask_blocked_sell_total, 1.0)
            ),
            "root_mask_vwap_share": (
                (root_mask_blocked_buy_vwap + root_mask_blocked_sell_vwap)
                / max(root_mask_blocked_buy_total + root_mask_blocked_sell_total, 1.0)
            ),
            "root_mask_adx_share": (
                (root_mask_blocked_buy_adx + root_mask_blocked_sell_adx)
                / max(root_mask_blocked_buy_total + root_mask_blocked_sell_total, 1.0)
            ),
            "root_mask_directional_share": (
                (root_mask_blocked_buy_directional + root_mask_blocked_sell_directional)
                / max(root_mask_blocked_buy_total + root_mask_blocked_sell_total, 1.0)
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
            "hold_drag_opportunity_count": hold_drag_opportunity_count,
            "hold_drag_penalized_count": hold_drag_penalized_count,
            "hold_drag_score": (
                hold_drag_penalized_count / max(hold_drag_opportunity_count, 1.0)
            ),
            "split_opportunity_count": split_opportunity_count,
            "split_executed": split_executed,
            "split_profitable_count": split_profitable_count,
            "split_efficiency": (
                split_profitable_count / max(split_executed, 1.0)
                if split_executed > 0.0
                else 0.0
            ),
            "split_runner_profitable_count": split_runner_profitable_count,
            "split_runner_failed_count": split_runner_failed_count,
            "split_early_count": split_early_count,
            "split_decorative_count": split_decorative_count,
            "split_trade_value_delta": (
                split_trade_value_delta_total / max(split_executed, 1.0)
                if split_executed > 0.0
                else 0.0
            ),
            "split_improved_total_trade_count": split_improved_total_trade_count,
            "split_tp_zone_opportunity_count": split_tp_zone_opportunity_count,
            "split_monetization_window_count": split_monetization_window_count,
            "split_zone_capture_rate": (
                split_profitable_count / max(split_tp_zone_opportunity_count, 1.0)
                if split_tp_zone_opportunity_count > 0.0
                else 0.0
            ),
            "split_monetization_capture_rate": (
                split_monetization_capture_count / max(split_monetization_window_count, 1.0)
                if split_monetization_window_count > 0.0
                else 0.0
            ),
            "split_runner_capture_rate": (
                split_runner_profitable_count / max(split_executed, 1.0)
                if split_executed > 0.0
                else 0.0
            ),
            "pyramid_opportunity_count": pyramid_opportunity_count,
            "pyramid_add_opportunity_count": pyramid_add_opportunity_count,
            "pyramid_monetization_window_count": pyramid_monetization_window_count,
            "pyramids_opened": pyramids_opened,
            "pyramid_profitable_count": pyramid_profitable_count,
            "pyramid_efficiency": (
                pyramid_profitable_count / max(pyramids_opened, 1.0)
                if pyramids_opened > 0.0
                else 0.0
            ),
            "pyramid_good_add_count": pyramid_good_add_count,
            "pyramid_bad_add_count": pyramid_bad_add_count,
            "pyramid_profitable_exit_count": pyramid_profitable_exit_count,
            "pyramid_total_trade_improvement_pct": (
                pyramid_total_trade_improvement_total / max(pyramids_opened, 1.0)
                if pyramids_opened > 0.0
                else 0.0
            ),
            "pyramid_failed_to_improve_count": pyramid_failed_to_improve_count,
            "pyramid_entry_quality_score": (
                pyramid_good_add_count / max(pyramids_opened, 1.0)
                if pyramids_opened > 0.0
                else 0.0
            ),
            "pyramid_exit_capture_rate": (
                pyramid_profitable_exit_count / max(pyramids_opened, 1.0)
                if pyramids_opened > 0.0
                else 0.0
            ),
            "pyramid_add_capture_rate": (
                pyramid_add_capture_count / max(pyramid_add_opportunity_count, 1.0)
                if pyramid_add_opportunity_count > 0.0
                else 0.0
            ),
            "pyramid_monetization_capture_rate": (
                pyramid_monetization_capture_count / max(pyramid_monetization_window_count, 1.0)
                if pyramid_monetization_window_count > 0.0
                else 0.0
            ),
            "pyramid_missed_add_count": pyramid_missed_add_count,
            "slbe_triggered": slbe_triggered,
            "slbe_profitable_exits": slbe_profitable_exits,
            "slbe_lock_profit_count": slbe_lock_profit_count,
            "slbe_capture_rate": (
                slbe_profitable_exits / max(slbe_triggered, 1.0)
                if slbe_triggered > 0.0
                else 0.0
            ),
            "close_winner_count": close_winner_count,
            "close_loser_count": close_loser_count,
            "close_quality_score": (
                close_winner_count / max(close_winner_count + close_loser_count, 1.0)
                if (close_winner_count + close_loser_count) > 0.0
                else 0.0
            ),
            "tp_like_exit_count": tp_like_exit_count,
            "tp_like_missed_count": tp_like_missed_count,
            "defensive_close_count": defensive_close_count,
            "early_close_noise_count": early_close_noise_count,
            "hard_stop_exit_count": hard_stop_exit_count,
            "soft_tp_hit_count": soft_tp_hit_count,
            "full_tp_hit_count": full_tp_hit_count,
            "time_stop_trigger_count": time_stop_trigger_count,
            "runner_extension_count": runner_extension_count,
            "runner_extension_opportunity_count": runner_extension_opportunity_count,
            "runner_extension_capture_rate": (
                runner_extension_capture_count / max(runner_extension_opportunity_count, 1.0)
                if runner_extension_opportunity_count > 0.0
                else 0.0
            ),
            "runner_profit_hold_window_count": runner_profit_hold_window_count,
            "runner_viable_window_count": runner_viable_window_count,
            "runner_profit_hold_capture_rate": (
                runner_profit_hold_capture_count / max(runner_profit_hold_window_count, 1.0)
                if runner_profit_hold_window_count > 0.0
                else 0.0
            ),
            "runner_missed_extension_count": runner_missed_extension_count,
            "runner_hold_after_soft_tp_count": runner_hold_after_soft_tp_count,
            "runner_viable_but_closed_count": runner_viable_but_closed_count,
            "early_full_close_after_soft_tp_count": early_full_close_after_soft_tp_count,
            "runner_managed_exit_count": runner_managed_exit_count,
            "runner_exit_profitable_count": runner_exit_profitable_count,
            "runner_forced_stop_count": runner_forced_stop_count,
            "runner_retained_profit_pct": (
                runner_retained_profit_total / max(split_runner_profitable_count, 1.0)
                if split_runner_profitable_count > 0.0
                else 0.0
            ),
            "runner_retained_profit_score": (
                runner_retained_profit_score_total / max(runner_profit_hold_capture_count, 1.0)
                if runner_profit_hold_capture_count > 0.0
                else 0.0
            ),
            "runner_giveback_pct": (
                runner_giveback_total / max(split_runner_failed_count, 1.0)
                if split_runner_failed_count > 0.0
                else 0.0
            ),
            "runner_giveback_ratio": (
                runner_giveback_total / max(runner_retained_profit_total + runner_giveback_total, 1e-6)
                if (runner_retained_profit_total + runner_giveback_total) > 0.0
                else 0.0
            ),
            "profit_peak_reached_count": profit_peak_reached_count,
            "profit_peak_giveback_ratio": (
                profit_peak_giveback_ratio_total / max(profit_peak_giveback_ratio_observations, 1.0)
                if profit_peak_giveback_ratio_observations > 0.0
                else 0.0
            ),
            "forced_stop_near_miss_count": forced_stop_near_miss_count,
            "close_quality_by_symbol": close_quality_by_symbol,
            "close_events_by_symbol": close_events_by_symbol,
            "split_efficiency_by_symbol": split_efficiency_by_symbol,
            "split_runner_capture_rate_by_symbol": split_runner_capture_rate_by_symbol,
            "pyramid_efficiency_by_symbol": pyramid_efficiency_by_symbol,
            "pyramid_entry_quality_score_by_symbol": pyramid_entry_quality_score_by_symbol,
            "pyramid_exit_capture_rate_by_symbol": pyramid_exit_capture_rate_by_symbol,
            "slbe_capture_rate_by_symbol": slbe_capture_rate_by_symbol,
            "hold_drag_score_by_symbol": hold_drag_score_by_symbol,
            "root_mask_share_by_symbol": root_mask_share_by_symbol,
            "root_mask_ema200_share_by_symbol": root_mask_ema200_share_by_symbol,
            "root_mask_vwap_share_by_symbol": root_mask_vwap_share_by_symbol,
            "root_mask_adx_share_by_symbol": root_mask_adx_share_by_symbol,
            "root_mask_directional_share_by_symbol": root_mask_directional_share_by_symbol,
            "hard_negative_mix": hard_negative_mix,
            "nemesis_mix": nemesis_mix,
            "liquidity_trap_share": hard_negative_counts.get("LIQUIDITY_TRAP", 0) / max(total_games, 1),
            "bad_runner_share": hard_negative_counts.get("BAD_RUNNER_EXIT", 0) / max(total_games, 1),
            "bad_pyramid_share": hard_negative_counts.get("BAD_PYRAMID_EXIT", 0) / max(total_games, 1),
            "directional_collapse": bool(total_long_entries <= 0.0 or total_short_entries <= 0.0),
            "long_entry_share": executed_long_entry_share,
            "short_entry_share": executed_short_entry_share,
        }

    @property
    def size(self) -> int:
        """Retourne le nombre d'episodes actuellement en memoire."""

        return self.tree.n_entries
