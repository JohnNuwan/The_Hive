"""
Rules of Engagement (ROE) - THE HIVE
Définition formelle des limites opérationnelles par module.
"""

from typing import Dict, List
from pydantic import BaseModel

class ROERule(BaseModel):
    id: str
    description: str
    is_active: bool

class ModuleROE(BaseModel):
    module_name: str
    rules: List[ROERule]

# --- DÉFINITION DES ROE ---

TRADING_ROE = ModuleROE(
    module_name="Banker",
    rules=[
        ROERule(id="NO_REVENGE_TRADING", description="Pas de trade immédiat après une perte > 1%", is_active=True),
        ROERule(id="MAX_ACCOUNT_EXPOSURE", description="Exposition totale max 5% du capital", is_active=True),
        ROERule(id="NEWS_TRADING_HALT", description="Pas de trade 15min avant/après News High Impact", is_active=True)
    ]
)

SHADOW_ROE = ModuleROE(
    module_name="Shadow",
    rules=[
        ROERule(id="NO_ACTIVE_HACKING", description="Interdiction de hack actif hors de l'Arène", is_active=True),
        ROERule(id="DATA_MINIMIZATION", description="Collecte uniquement du strict nécessaire à l'enquête", is_active=True),
        ROERule(id="GREY_ZONE_ALERT", description="Alerte Admin avant toute recherche sur infra critique", is_active=True)
    ]
)

CORE_ROE = ModuleROE(
    module_name="Core",
    rules=[
        ROERule(id="ADMIN_DISSUASION", description="Alerte si un ordre Admin semble auto-destructeur", is_active=True),
        ROERule(id="HUMILITY_MODE", description="Reconnaître l'incertitude dans les prédictions financières", is_active=True)
    ]
)

def get_all_roe() -> List[ModuleROE]:
    return [TRADING_ROE, SHADOW_ROE, CORE_ROE]
