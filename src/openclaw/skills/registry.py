"""
OpenClaw Skill Registry
Part of Sovereign Stack V3.0

Système central d'enregistrement et de découverte des compétences (Skills).
Chaque Skill est une fonction Python décorée avec @skill qui s'auto-enregistre
dans le SKILL_REGISTRY global au moment de l'import.

Usage :
    from openclaw.skills.registry import skill, get_available_skills

    @skill("my_tool", "Description de mon outil")
    def my_tool(arg: str) -> str:
        return f"Résultat: {arg}"

    # Récupérer la liste des skills disponibles
    skills = get_available_skills()
"""

import logging
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)

# Registre global des skills (nom -> fonction callable)
SKILL_REGISTRY: Dict[str, Callable] = {}


def skill(name: str, description: str):
    """Décorateur pour enregistrer une fonction comme un Skill OpenClaw.

    Ajoute les métadonnées (_is_skill, _skill_name, _skill_description)
    à la fonction et l'inscrit dans le SKILL_REGISTRY global.

    Args:
        name: Nom unique du skill (ex: "fs_read", "git_status").
        description: Description courte pour le catalogue d'outils.

    Returns:
        Le décorateur qui enregistre la fonction.

    Example:
        @skill("fs_read", "Lit le contenu d'un fichier")
        def fs_read(path: str) -> str:
            ...
    """

    def decorator(func: Callable):
        func._is_skill = True
        func._skill_name = name
        func._skill_description = description
        SKILL_REGISTRY[name] = func
        logger.debug(f"Skill registered: {name}")
        return func

    return decorator


def get_available_skills() -> List[Dict[str, str]]:
    """Retourne la liste des skills disponibles avec nom et description.

    Returns:
        Liste de dictionnaires {"name": ..., "description": ...}.
    """
    return [
        {"name": name, "description": func._skill_description}
        for name, func in SKILL_REGISTRY.items()
    ]


def get_skill(name: str) -> Callable:
    """Récupère un skill par son nom.

    Args:
        name: Nom du skill à récupérer.

    Returns:
        La fonction callable du skill.

    Raises:
        KeyError: Si le skill n'est pas trouvé dans le registre.
    """
    if name not in SKILL_REGISTRY:
        raise KeyError(
            f"Skill '{name}' not found. "
            f"Available: {list(SKILL_REGISTRY.keys())}"
        )
    return SKILL_REGISTRY[name]


def load_all_skills():
    """Charge tous les modules de skills pour qu'ils s'enregistrent via @skill.

    Cette fonction importe dynamiquement chaque module de skills,
    déclenchant l'exécution des décorateurs @skill au niveau module.
    """
    from . import basic, public_apis, git_ops

    logger.info(
        f"All skills loaded. Registry contains {len(SKILL_REGISTRY)} skills: "
        f"{list(SKILL_REGISTRY.keys())}"
    )
