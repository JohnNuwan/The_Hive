"""
OpenClaw Skill Registry
Part of Sovereign Stack V3.0

Système d'enregistrement et de découverte des compétences (Skills).
"""

import inspect
from typing import Callable, Dict, Any, List

# Registre global des skills
SKILL_REGISTRY: Dict[str, Callable] = {}

def skill(name: str, description: str):
    """
    Décorateur pour enregistrer une fonction comme un Skill OpenClaw.
    """
    def decorator(func: Callable):
        func._is_skill = True
        func._skill_name = name
        func._skill_description = description
        SKILL_REGISTRY[name] = func
        return func
    return decorator

def get_available_skills() -> List[Dict[str, str]]:
    """Retourne la liste des skills disponibles (nom + description)"""
    return [
        {"name": name, "description": func._skill_description}
        for name, func in SKILL_REGISTRY.items()
    ]

# Placeholder pour l'intégration future de Public APIs
def load_all_skills():
    """
    Charge tous les modules de skills pour qu'ils s'enregistrent via @skill.
    """
    from . import basic, public_apis, git_ops
