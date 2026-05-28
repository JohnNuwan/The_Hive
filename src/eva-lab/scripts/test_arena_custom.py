"""Script de test unitaire et de validation pour l'Arena Darwinienne MuZero/JAX.

Ce script permet d'évaluer de manière isolée et supervisée les performances
d'un Challenger MuZero par rapport à son Champion courant sous des conditions
de marché réelles (horizon Scalp) et d'interroger Hermes Coordinator pour les stress-tests.
"""

from __future__ import annotations

import logging
from eva_lab.arena import Arena

# Configuration du logging standard pour l'environnement The Hive
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eva_lab.arena_test")


def main() -> None:
    """Exécute le combat de ligue Arena entre le Challenger et le Champion.

    Cette routine :
    1. Initialise l'instance Arena avec les chemins de données par défaut.
    2. Lance la méthode battle sur l'univers de symboles configuré (scalp).
    3. Extrait et affiche le score comparatif du Challenger, du Champion,
       des baselines directionnelles (Always Long/Short) et le rapport de stress d'Hermes.
    """
    logger.info("Démarrage du combat de ligue Arena pour l'horizon Scalp...")
    
    # Initialisation de l'arène darwinienne
    arena = Arena()
    
    # Exécution de la ligue de combat complète
    report = arena.battle(
        challenger_id="muzero_scalp_latest",
        champion_id="muzero_scalp_ckpt_1000",
        horizon="scalp"
    )
    
    logger.info("Combat d'évaluation complété avec succès !")
    
    # Affichage structuré du diagnostic de ligue
    print("=================== RESULTATS DE L'ARENA ===================")
    print(f"Verdict Global           : {report.get('outcome')}")
    print(f"Action Recommandée       : {report.get('action_required')}")
    print(f"Score du Challenger      : {report.get('challenger', {}).get('score')}")
    print(f"Score du Champion Live   : {report.get('champion', {}).get('score')}")
    print(f"Baseline Always Long     : {report.get('always_long_baseline', {}).get('score')}")
    print(f"Baseline Always Short    : {report.get('always_short_baseline', {}).get('score')}")
    print("------------------------------------------------------------")
    print("Scénario de Stress Macro-économique (Hermes Coordinator) :")
    print(report.get("hermes_stress_scenario"))
    print("============================================================")


if __name__ == "__main__":
    main()
