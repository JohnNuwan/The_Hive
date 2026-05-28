#!/usr/bin/env python3
"""THE HIVE — Pont de Rétroaction AlphaEvolve (Live Bridging).

Ce script extrait la meilleure variante génomique issue de la dernière
campagne d'optimisation hors-ligne d'AlphaEvolve et l'applique de manière
directe sur les configurations actives du Banker master local (.env et
.env.banker.master.local) afin de fermer la boucle d'auto-amélioration.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apply_alphaevolve_best")

# Définition des chemins absolus
WORKDIR = Path(__file__).resolve().parents[1]
CAMPAIGNS_DIR = WORKDIR / "data" / "alphaevolve" / "campaigns"
ENV_BANKER_PATH = WORKDIR / ".env.banker.master.local"
ENV_PATH = WORKDIR / ".env"

# Dictionnaire de correspondance entre les gènes d'AlphaEvolve et les variables d'environnement
PARAM_MAPPING = {
    "split_window_activation_bonus": "MUZERO_SPLIT_WINDOW_ACTIVATION_BONUS",
    "runner_window_hold_bonus": "MUZERO_RUNNER_HOLD_CAPTURE_BONUS",
    "pyramid_window_activation_bonus": "MUZERO_PYRAMID_WINDOW_ACTIVATION_BONUS",
    "missed_window_penalty": "MUZERO_MISSED_WINDOW_PENALTY",
    "giveback_soft_penalty": "MUZERO_RUNNER_GIVEBACK_SOFT_PENALTY",
    "giveback_hard_penalty": "MUZERO_RUNNER_GIVEBACK_HARD_PENALTY",
    "muzero_collection_num_simulations_xauusd": "MUZERO_COLLECTION_NUM_SIMULATIONS_XAUUSD",
    "muzero_collection_max_moves_xauusd": "MUZERO_COLLECTION_MAX_MOVES_XAUUSD",
    "muzero_collection_max_episode_seconds_xauusd": "MUZERO_COLLECTION_MAX_EPISODE_SECONDS_XAUUSD",
}


def load_latest_campaign() -> dict[str, Any] | None:
    """Recherche et charge le fichier JSON de la dernière campagne AlphaEvolve.

    Returns:
        dict[str, Any] | None: Charge utile JSON de la campagne, ou None si introuvable.
    """
    if not CAMPAIGNS_DIR.exists():
        logger.warning("Le répertoire des campagnes AlphaEvolve n'existe pas : %s", CAMPAIGNS_DIR)
        return None
    
    files = sorted(CAMPAIGNS_DIR.glob("alphaevolve_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        logger.warning("Aucune campagne AlphaEvolve trouvée.")
        return None
    
    latest_file = files[0]
    logger.info("Chargement de la dernière campagne AlphaEvolve : %s...", latest_file.name)
    try:
        return json.loads(latest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Échec de la lecture de la campagne %s : %s", latest_file.name, exc)
        return None


def find_best_variant(campaign: dict[str, Any]) -> dict[str, Any] | None:
    """Identifie la meilleure variante d'optimisation dans une campagne donnée.

    Cette fonction compare les scores Arena, Nemesis et Proxy de chaque variante
    et sélectionne celle ayant obtenu la performance maximale.

    Args:
        campaign (dict[str, Any]): Le dictionnaire de la campagne AlphaEvolve.

    Returns:
        dict[str, Any] | None: La meilleure variante sous forme de dictionnaire, ou None.
    """
    variants = campaign.get("variants", [])
    if not variants:
        logger.warning("Aucune variante trouvée dans la campagne.")
        return None
    
    best_variant = None
    best_score = -999999.0
    
    for variant in variants:
        # Résolution progressive des scores dans l'ordre de priorité
        score = -999999.0
        for key in ["score_arena", "score_nemesis", "score_proxy"]:
            val = variant.get(key)
            if val is not None:
                score = float(val)
                break
        
        # En cas d'égalité ou de premier élément
        if best_variant is None or score > best_score:
            best_variant = variant
            best_score = score if score != -999999.0 else 0.0
            
    if best_variant:
        logger.info("Variante optimale retenue : %s (score: %s)", best_variant.get('variant_id'), best_score)
    return best_variant


def update_env_file(file_path: Path, params: dict[str, float]) -> bool:
    """Met à jour ou ajoute les variables d'environnement dans un fichier .env donné.

    Args:
        file_path (Path): Chemin absolu du fichier .env à modifier.
        params (dict[str, float]): Dictionnaire des gènes et valeurs à appliquer.

    Returns:
        bool: True si l'écriture a réussi, False sinon.
    """
    if not file_path.exists():
        logger.warning("Fichier introuvable : %s", file_path)
        return False
        
    lines = file_path.read_text(encoding="utf-8").splitlines()
    updated_keys = set()
    new_lines = []
    
    # Construction de la table de variables d'environnement cibles
    env_params = {}
    for raw_name, value in params.items():
        env_name = PARAM_MAPPING.get(raw_name)
        if env_name:
            env_params[env_name] = str(value)
            
    # Remplacement en ligne des valeurs existantes
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
            
        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in env_params:
            new_lines.append(f'{key}="{env_params[key]}"')
            updated_keys.add(key)
            logger.info("Mise à jour dans %s : %s = %s", file_path.name, key, env_params[key])
        else:
            new_lines.append(line)
            
    # Ajout des nouvelles clés qui n'existaient pas
    for key, val in env_params.items():
        if key not in updated_keys:
            new_lines.append(f'{key}="{val}"')
            logger.info("Ajout dans %s : %s = %s", file_path.name, key, val)
            
    try:
        file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        logger.error("Échec de l'écriture sur %s : %s", file_path.name, exc)
        return False


def main() -> None:
    """Routine principale d'application du pont de rétroaction AlphaEvolve."""
    campaign = load_latest_campaign()
    if not campaign:
        logger.error("Aucune campagne chargée. Annulation de la rétroaction.")
        return
        
    best_variant = find_best_variant(campaign)
    if not best_variant:
        logger.error("Aucune variante résolue. Annulation de la rétroaction.")
        return
        
    params = best_variant.get("params", {})
    if not params:
        logger.error("La variante retenue ne contient pas de paramètres.")
        return
        
    logger.info("Application des paramètres de la variante %s :", best_variant.get('variant_id'))
    for k, v in params.items():
        logger.info("  %s = %s", k, v)
        
    # Application aux fichiers de prod active
    update_env_file(ENV_BANKER_PATH, params)
    update_env_file(ENV_PATH, params)
    logger.info("Pont de rétroaction AlphaEvolve appliqué avec succès !")


if __name__ == "__main__":
    main()
