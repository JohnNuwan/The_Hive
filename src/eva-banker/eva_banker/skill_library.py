"""
Skill Library — Bibliothèque de Compétences de Trading.

Implémente le répertoire des « Skills » (comportements pré-appris)
utilisées dans l'architecture hiérarchique SPlaTES du Banker.

Le Manager (niveau haut) sélectionne un Skill en fonction du
contexte marché, puis le Worker (niveau bas) l'exécute.

Skills disponibles :
    - SCALPING : Prise de profit rapide sur micros-mouvements (< 5 min).
    - HEDGING : Protection des positions par ouverture opposée (1h).
    - ACCUMULATION : Achat progressif sur zone de support (24h).
    - BREAKOUT_PLAY : Entrée sur cassure de range (variable).
"""

from enum import Enum
from typing import List

from pydantic import BaseModel


class SkilledBehavior(str, Enum):
    """
    Énumération des comportements de trading disponibles.

    Chaque comportement correspond à une stratégie d'exécution
    spécifique avec son propre horizon temporel et profil de risque.
    """
    SCALPING = "SCALPING"
    HEDGING = "HEDGING"
    ACCUMULATION = "ACCUMULATION"
    BREAKOUT_PLAY = "BREAKOUT_PLAY"


class Skill(BaseModel):
    """
    Représente une compétence de trading unitaire.

    Attributes:
        name: Le type de comportement (SkilledBehavior).
        horizon_seconds: Durée typique d'un trade avec cette skill (en secondes).
        risk_profile: Profil de risque (Aggressive, Conservative, Neutral).
        description: Description humaine de la stratégie.
    """
    name: SkilledBehavior
    horizon_seconds: int
    risk_profile: str
    description: str


class SkillLibrary:
    """
    Bibliothèque de compétences pré-apprises (Skills).

    Utilisée par le Manager (Niveau Haut) pour orchestrer
    l'Exécutant (Niveau Bas) dans l'architecture SPlaTES.

    Le catalogue est extensible : en production, de nouvelles skills
    peuvent être ajoutées par le module GeneticUpdater du Lab.
    """

    def __init__(self):
        self.skills: dict[SkilledBehavior, Skill] = {
            SkilledBehavior.SCALPING: Skill(
                name=SkilledBehavior.SCALPING,
                horizon_seconds=300,
                risk_profile="Aggressive",
                description="Prise de profit rapide sur micros-mouvements.",
            ),
            SkilledBehavior.HEDGING: Skill(
                name=SkilledBehavior.HEDGING,
                horizon_seconds=3600,
                risk_profile="Conservative",
                description="Protection des positions par ouverture opposée.",
            ),
            SkilledBehavior.ACCUMULATION: Skill(
                name=SkilledBehavior.ACCUMULATION,
                horizon_seconds=86400,
                risk_profile="Neutral",
                description="Achat progressif sur zone de support long terme.",
            ),
        }

    def get_skill(self, behavior: SkilledBehavior) -> Skill | None:
        """
        Récupère une skill par son type de comportement.

        Args:
            behavior: Le type de comportement recherché.

        Returns:
            Skill | None: La skill correspondante ou None si absente.
        """
        return self.skills.get(behavior)

    def list_available_skills(self) -> List[SkilledBehavior]:
        """
        Liste tous les comportements disponibles dans la bibliothèque.

        Returns:
            List[SkilledBehavior]: Liste des skills enregistrées.
        """
        return list(self.skills.keys())
