"""Buffer de ligue pour l'apprentissage fictif (Fictitious Play) contre d'anciens champions.

Ce module gère la persistance sur disque et le chargement en mémoire des trajectoires
historiques générées par des champions et checkpoints de référence.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from eva_lab.muzero.replay_buffer import GameHistory

logger = logging.getLogger(__name__)


class LeagueBuffer:
    """Stocke, gère et échantillonne des trajectoires historiques de champions passés."""

    def __init__(
        self,
        league_dir: str = "data/muzero/league",
        max_champions: int = 10,
        capacity_per_champion: int = 50,
    ) -> None:
        """Initialise le buffer de ligue.

        Args:
            league_dir (str): Répertoire de stockage des trajectoires sur disque.
            max_champions (int): Nombre maximum de champions conservés en mémoire.
            capacity_per_champion (int): Nombre maximum de parties stockées par champion.
        """
        self.league_dir = league_dir
        self.max_champions = max_champions
        self.capacity_per_champion = capacity_per_champion

        # Dictionnaire en mémoire : {champion_id: [GameHistory, ...]}
        self.champions_pool: Dict[str, List[GameHistory]] = {}
        
        # Création du dossier racine de la ligue si inexistant
        os.makedirs(self.league_dir, exist_ok=True)
        
        # Chargement initial depuis le disque
        self.load_all_games()

    def load_all_games(self) -> None:
        """Parcourt le répertoire disque de la ligue et charge toutes les parties en mémoire."""
        if not os.path.exists(self.league_dir):
            return

        loaded_champions_count = 0
        loaded_games_count = 0

        # Exploration des sous-dossiers par champion
        for entry in os.scandir(self.league_dir):
            if entry.is_dir():
                champion_id = entry.name
                self.champions_pool[champion_id] = []
                
                # Chargement des fichiers .pkl pour ce champion
                for file_entry in os.scandir(entry.path):
                    if file_entry.is_file() and file_entry.name.endswith(".pkl"):
                        try:
                            with open(file_entry.path, "rb") as f:
                                game = pickle.load(f)
                            if isinstance(game, GameHistory) and len(game) > 0:
                                self.champions_pool[champion_id].append(game)
                                loaded_games_count += 1
                        except Exception as exc:
                            logger.warning(
                                "[LeagueBuffer] Impossible de lire la trajectoire %s: %s",
                                file_entry.path,
                                exc,
                            )
                
                # Tri des dossiers champions et nettoyage FIFO s'il y a trop de parties
                if len(self.champions_pool[champion_id]) > self.capacity_per_champion:
                    self.champions_pool[champion_id] = self.champions_pool[champion_id][-self.capacity_per_champion:]
                
                loaded_champions_count += 1

        # Limite le nombre de champions en mémoire s'il y a dépassement
        if len(self.champions_pool) > self.max_champions:
            overflow_keys = list(self.champions_pool.keys())[:-self.max_champions]
            for key in overflow_keys:
                del self.champions_pool[key]

        logger.info(
            "[LeagueBuffer] Chargé %d champions et %d parties historiques depuis %s.",
            loaded_champions_count,
            loaded_games_count,
            self.league_dir,
        )

    def save_game(self, game: GameHistory, champion_id: str) -> None:
        """Enregistre une partie historique sur disque et met à jour le pool mémoire.

        Args:
            game (GameHistory): Épisode complet à stocker.
            champion_id (str): Identifiant unique du champion ayant produit la trajectoire.
        """
        if len(game) <= 0:
            return

        champion_dir = os.path.join(self.league_dir, champion_id)
        os.makedirs(champion_dir, exist_ok=True)

        # Génération d'un nom de fichier unique basé sur le timestamp ou hash
        game_hash = abs(hash(str(game.observations[0]) + str(len(game)))) % 10000000
        filename = f"game_{game_hash}.pkl"
        filepath = os.path.join(champion_dir, filename)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(game, f)
        except Exception as exc:
            logger.error(
                "[LeagueBuffer] Échec de la sérialisation de la trajectoire pour %s: %s",
                champion_id,
                exc,
            )
            return

        # Ajout dans le dictionnaire mémoire
        if champion_id not in self.champions_pool:
            # Gestion de la capacité max des champions
            if len(self.champions_pool) >= self.max_champions:
                oldest_champ = list(self.champions_pool.keys())[0]
                del self.champions_pool[oldest_champ]
                # Le nettoyage physique des dossiers optionnel pour éviter les suppressions massives
                logger.info("[LeagueBuffer] Capacité maximale de champions atteinte. Écartement de %s de la mémoire.", oldest_champ)
            self.champions_pool[champion_id] = []

        self.champions_pool[champion_id].append(game)

        # Respect de la capacité par champion (FIFO)
        if len(self.champions_pool[champion_id]) > self.capacity_per_champion:
            self.champions_pool[champion_id].pop(0)

    def sample(
        self,
        batch_size: int,
        num_unroll_steps: int = 5,
    ) -> List[Tuple[GameHistory, int, float]]:
        """Échantillonne un lot de trajectoires de ligue.

        Cette méthode imite la signature de replay_buffer.sample mais utilise
        un index SumTree fictif de `-1` pour que update_priorities l'ignore.

        Args:
            batch_size (int): Taille du lot désiré.
            num_unroll_steps (int): Longueur d'unroll requis.

        Returns:
            List[Tuple[GameHistory, int, float]]: Liste de triplets (game, start_idx, tree_idx).
        """
        if not self.champions_pool:
            return []

        # Regroupement de toutes les trajectoires disponibles
        all_games: List[GameHistory] = []
        for games in self.champions_pool.values():
            all_games.extend(games)

        if not all_games:
            return []

        batch: List[Tuple[GameHistory, int, float]] = []
        for _ in range(batch_size):
            game = random.choice(all_games)
            max_idx = max(0, len(game) - num_unroll_steps - 1)
            start_idx = random.randint(0, max_idx)
            # -1 est l'index fictif pour marquer les batchs venant de la ligue
            batch.append((game, start_idx, -1.0))

        return batch

    def __len__(self) -> int:
        """Retourne le nombre total de parties stockées dans la ligue en mémoire."""
        return sum(len(games) for games in self.champions_pool.values())
