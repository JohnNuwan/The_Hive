from enum import Enum
from pydantic import BaseModel
from typing import List

class SkilledBehavior(str, Enum):
    SCALPING = "SCALPING"
    HEDGING = "HEDGING"
    ACCUMULATION = "ACCUMULATION"
    BREAKOUT_PLAY = "BREAKOUT_PLAY"

class Skill(BaseModel):
    name: SkilledBehavior
    horizon_seconds: int
    risk_profile: str
    description: str

class SkillLibrary:
    """
    Bibliothèque de compétences pré-apprises (Skills).
    Utilisée par le Manager (Niveau Haut) pour orchestrer l'Exécutant (Niveau Bas).
    """
    def __init__(self):
        self.skills = {
            SkilledBehavior.SCALPING: Skill(
                name=SkilledBehavior.SCALPING,
                horizon_seconds=300,
                risk_profile="Aggressive",
                description="Prise de profit rapide sur micros-mouvements."
            ),
            SkilledBehavior.HEDGING: Skill(
                name=SkilledBehavior.HEDGING,
                horizon_seconds=3600,
                risk_profile="Conservative",
                description="Protection des positions par ouverture opposée."
            ),
            SkilledBehavior.ACCUMULATION: Skill(
                name=SkilledBehavior.ACCUMULATION,
                horizon_seconds=86400,
                risk_profile="Neutral",
                description="Achat progressif sur zone de support long terme."
            )
        }

    def get_skill(self, behavior: SkilledBehavior) -> Skill:
        return self.skills.get(behavior)

    def list_available_skills(self) -> List[SkilledBehavior]:
        return list(self.skills.keys())
