import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class GeneticUpdater:
    """
    Système de Mémoire Génétique et d'Automutation de l'IA.
    Archive l'ADN (générations, performances, configurations) de l'Agent.
    """
    
    def __init__(self, registry_path: str = "data/models_registry.json"):
        self.registry_path = registry_path
        self._ensure_registry()

    def _ensure_registry(self):
        """Vérifie que le registre ADN existe, sinon le crée."""
        dirname = os.path.dirname(self.registry_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        if not os.path.exists(self.registry_path):
            initial_state = {
                "current_champion": "gen_000_baseline",
                "generations": {
                    "gen_000_baseline": {
                        "timestamp": datetime.now().isoformat(),
                        "win_rate": 50.0,
                        "return_pct": 0.0,
                        "battles_won": 0
                    }
                }
            }
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(initial_state, f, indent=4)

    def get_champion(self) -> str:
        """Retourne l'ID du modèle actuellement en production (le Champion)."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("current_champion", "gen_000_baseline")

    def register_new_generation(self, gen_id: str, metrics: Dict[str, Any], is_champion: bool = False):
        """
        Enregistre l'ADN d'une nouvelle génération de modèle.
        Si is_champion est True, ce modèle remplace le modèle actuel.
        """
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        data["generations"][gen_id] = {
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        if is_champion:
            old_champion = data.get("current_champion")
            data["current_champion"] = gen_id
            logger.info(f"🧬 MUTATION ACCEPTÉE : Le Champion {old_champion} est détrôné par {gen_id} !")
            
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    def check_for_updates(self):
        """Compatibilité avec l'ancienne signature mockée."""
        return {
            "updates_found": 0,
            "type": "TRADING_STRATEGY",
            "action": "NO_ACTION",
            "safety_check": "PASSED"
        }
