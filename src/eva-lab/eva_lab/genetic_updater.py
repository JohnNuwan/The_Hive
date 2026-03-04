import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Timeframe horizons for multi-strategy tracking
HORIZONS = ["scalp", "intraday", "swing"]


class GeneticUpdater:
    """
    Système de Mémoire Génétique et d'Automutation de l'IA.
    Archive l'ADN (générations, performances, configurations) de l'Agent.
    
    ### MTF Architecture (Sprint 19)
    Chaque registre stocke maintenant 3 Champions distincts, un par horizon:
    - scalp:    Optimisé pour les positions courtes (M5 / +1H)
    - intraday: Optimisé pour les positions journalières (H1 / +1D)
    - swing:    Optimisé pour les positions longues (D1 / +1W)
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
                "version": "2.0-MTF",
                "description": "Universal Multi-Timeframe Genetic Registry",
                "champions": {
                    # One champion per trading horizon
                    "scalp": "gen_000_baseline",
                    "intraday": "gen_000_baseline",
                    "swing": "gen_000_baseline",
                },
                "generations": {
                    "gen_000_baseline": {
                        "timestamp": datetime.now().isoformat(),
                        "win_rate": {"scalp": 50.0, "intraday": 50.0, "swing": 50.0},
                        "return_pct": {"scalp": 0.0, "intraday": 0.0, "swing": 0.0},
                        "battles_won": {"scalp": 0, "intraday": 0, "swing": 0},
                        "horizon_accuracy": {"scalp": 0.33, "intraday": 0.33, "swing": 0.33}
                    }
                }
            }
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(initial_state, f, indent=4)

    def get_champion(self, horizon: str = "intraday") -> str:
        """Retourne l'ID du modèle actuellement Champion pour un horizon donné."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migrate legacy registry format if needed
        if "current_champion" in data:
            return data.get("current_champion", "gen_000_baseline")
        return data.get("champions", {}).get(horizon, "gen_000_baseline")
    
    def get_all_champions(self) -> Dict[str, str]:
        """Retourne tous les Champions par horizon."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "current_champion" in data:
            # Legacy support
            champ = data.get("current_champion", "gen_000_baseline")
            return {h: champ for h in HORIZONS}
        return data.get("champions", {h: "gen_000_baseline" for h in HORIZONS})

    def register_new_generation(
        self,
        gen_id: str,
        metrics: Dict[str, Any],
        is_champion: bool = False,
        horizon: Optional[str] = None
    ):
        """
        Enregistre l'ADN d'une nouvelle génération de modèle.
        
        Args:
            gen_id:       Identifiant unique de la génération (ex: gen_042_bullish)
            metrics:      Dictionnaire de métriques {win_rate, return_pct, battles_won, ...}
            is_champion:  Si True, ce modèle devient le Champion pour son horizon
            horizon:      Trading horizon ('scalp', 'intraday', 'swing'). None = tous les horizons.
        """
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Ensure the registry is in the new schema
        if "champions" not in data:
            data["champions"] = {h: data.get("current_champion", "gen_000_baseline") for h in HORIZONS}
            
        data["generations"][gen_id] = {
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        if is_champion:
            if horizon:
                old_champion = data["champions"].get(horizon)
                data["champions"][horizon] = gen_id
                logger.info(f"🧬 MTF MUTATION [{horizon.upper()}]: Champion {old_champion} → {gen_id} !")
            else:
                # Global champion for all horizons
                for h in HORIZONS:
                    data["champions"][h] = gen_id
                logger.info(f"🧬 MUTATION GLOBALE : Nouveau Champion universel = {gen_id} !")
            
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances par horizon."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        summary = {}
        champions = data.get("champions", {})
        
        for horizon, gen_id in champions.items():
            gen_data = data.get("generations", {}).get(gen_id, {})
            wr = gen_data.get("win_rate", {})
            rp = gen_data.get("return_pct", {})
            
            summary[horizon] = {
                "champion": gen_id,
                "win_rate": wr.get(horizon, wr) if isinstance(wr, dict) else wr,
                "return_pct": rp.get(horizon, rp) if isinstance(rp, dict) else rp,
            }
        return summary
            
    def check_for_updates(self):
        """Compatibilité avec l'ancienne signature mockée."""
        return {
            "updates_found": 0,
            "type": "TRADING_STRATEGY",
            "action": "NO_ACTION",
            "safety_check": "PASSED"
        }
