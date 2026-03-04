import logging
import os
from typing import Dict, Any
from datetime import datetime

# Importer les vrais composants de simulation de l'environnement Gym de THE HIVE
# Pour tester les modèles Muzero / DreamerV3
try:
    from eva_lab.muzero.config import MuZeroConfigV3
    from eva_lab.muzero.jax_agent import JAXMuZeroAgent
    from eva_lab.muzero.environment import TradingEnvironment
    HAS_MUZERO = True
except ImportError:
    HAS_MUZERO = False

logger = logging.getLogger(__name__)

class Arena:
    """
    THE ARENA (Le Colisée)
    ----------------------
    Environnement de simulation compétitive pour l'évolution Darwinienne.
    Vérifie si une nouvelle stratégie (V2 - Challenger) bat l'ancienne (V1 - Champion).
    Les modèles sont passés au banc d'essai sur un environnement reconstitué.
    """
    
    def __init__(self, weights_dir: str = "data/weights"):
        self.history = []
        self.weights_dir = weights_dir
        self.config = MuZeroConfigV3() if HAS_MUZERO else None

    def _evaluate_model(self, weights_path: str, symbols: list[str]) -> Dict[str, float]:
        """Évalue un modèle spécifique sur plusieurs actifs et retourne le score global."""
        if not HAS_MUZERO:
            logger.error("MuZero dependencies missing! Cannot run real Arena.")
            return {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}
            
        if not os.path.exists(weights_path):
            logger.warning(f"Weights file missing at {weights_path}, assuming zero score.")
            return {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}

        try:
            agent = JAXMuZeroAgent(self.config)
            agent.load(weights_path)
        except Exception as e:
            logger.error(f"Failed to load agent {weights_path}: {e}")
            return {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}

        total_return = 0.0
        total_win_rate = 0.0
        valid_symbols = 0

        for symbol in symbols:
            # 200 pas de simulation (ex: 200 heures)
            env = TradingEnvironment(symbol=symbol, config=self.config, max_steps=200)
            _ = agent.play_game(env, exploration=False)
            summary = env.get_summary()
            
            total_return += getattr(summary, 'get', lambda k, d=0: 0)("return_pct", summary.get("return_pct", 0.0) if isinstance(summary, dict) else getattr(summary, "return_pct", 0.0))
            total_win_rate += getattr(summary, 'get', lambda k, d=0: 0)("win_rate_pct", summary.get("win_rate_pct", 0.0) if isinstance(summary, dict) else getattr(summary, "win_rate_pct", 0.0))
            valid_symbols += 1

        if valid_symbols == 0:
            return {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}

        return {
            "profit_factor": 1.0 + (total_return / valid_symbols / 100.0), # Approximate mapping
            "return_pct": total_return / valid_symbols,
            "win_rate": total_win_rate / valid_symbols
        }

    def battle(self, challenger_id: str, champion_id: str = "muzero_global_champion") -> Dict[str, Any]:
        """
        Organise un combat (Backtest comparatif en isolation).
        """
        logger.info(f"⚔️ ARENA FIGHT: {challenger_id} (Challenger) vs {champion_id} (Champion)")
        
        # Récupération de l'intégralité des actifs (27+) pour un crash-test total sur la RTX 3090
        try:
            from shared.config import get_settings
            eval_symbols = get_settings().banker_symbols
        except ImportError:
            eval_symbols = ["EURUSD", "XAUUSD", "BTCUSD", "US30.cash"]
            logger.warning("Could not load full symbol list from settings, using fallback.")
        
        challenger_path = os.path.join(self.weights_dir, f"{challenger_id}.pkl")
        champion_path = os.path.join(self.weights_dir, f"{champion_id}.pkl")
        
        logger.info(f"🔍 Evaluating Challenger: {challenger_path}")
        challenger_metrics = self._evaluate_model(challenger_path, eval_symbols)
        
        logger.info(f"🛡️ Evaluating Champion: {champion_path}")
        champion_metrics = self._evaluate_model(champion_path, eval_symbols)
        
        challenger_score = challenger_metrics["return_pct"]
        champion_score = champion_metrics["return_pct"]
        
        logger.info(f"📊 RESULT | Challenger: {challenger_score:.2f}% | Champion: {champion_score:.2f}%")
        
        # Pour éviter de changer de modèle pour des gains infimes (friction)
        # on exige que le challenger batte le champion d'au moins 2% de rentabilité globale
        is_victory = challenger_score > (champion_score + 2.0)
        
        if is_victory:
            logger.info(f"👑 LE CHALLENGER L'EMPORTE ! Une nouvelle génération est née.")
        else:
            logger.info(f"🛡️ LE CHAMPION TIENT SA POSITION. Le challenger est rejeté.")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "combat_type": "DREAMER_V3_ISOLATED_EVAL",
            "eval_symbols": eval_symbols,
            "challenger": {
                "id": challenger_id,
                "score": round(challenger_score, 2),
                "metrics": challenger_metrics
            },
            "champion": {
                "id": champion_id,
                "score": round(champion_score, 2),
                "metrics": champion_metrics
            },
            "outcome": "VICTORY" if is_victory else "DEFEAT",
            "action_required": "HOT_SWAP_DEPLOY" if is_victory else "DELETE_CODE"
        }
        
        self.history.append(report)
        return report

# Exemple d'usage
if __name__ == "__main__":
    arena = Arena()
    print(arena.battle("muzero_global_latest", "muzero_global_champion"))
