"""Package Banker de THE HIVE.

Le package expose l'application FastAPI de facon paresseuse pour ne pas forcer
les dependances completes du Banker lors du lancement des outils legers comme
l'agent follower.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Charge les exports lourds uniquement a la demande.

    Args:
        name (str): Nom d'attribut demande.

    Returns:
        Any: Objet exporte par le package.

    Raises:
        AttributeError: Si l'attribut n'est pas supporte.
    """

    if name == "app":
        from eva_banker.main import app

        return app
    raise AttributeError(f"Attribut inconnu pour eva_banker: {name}")

__all__ = ["app"]
