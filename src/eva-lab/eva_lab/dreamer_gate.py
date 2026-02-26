"""
DreamerV3 Training Gate — Activation Conditionnelle
Part of Sovereign Stack V3.0 — Sprint 5

Ce module gère l'activation/désactivation conditionnelle de l'entraînement
DreamerV3 basé sur le Feature Flag `ENABLE_DREAMER_TRAINING`.

Si le flag est True (RTX 3090 disponible) :
    → Lance la boucle d'entraînement du World Model sur les données collectées
      par le Shadow Learning.

Si le flag est False (RTX 2060, config actuelle) :
    → Le World Model est chargé en mode inférence uniquement.
    → Le Shadow Learning collecte les données passivement.
    → L'agent utilise les boucles RLM (Sprint 4) comme remplacement léger.

Références :
    - CDcs v3.0 : "Feature Flag" et "Shadow Learning"
    - DreamerV3 (Hafner et al., 2023)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class DreamerGate:
    """Gestionnaire de l'activation conditionnelle de DreamerV3.

    Lit le Feature Flag et décide si l'entraînement du World Model
    doit être lancé ou si le système fonctionne en mode dégradé (RLM).

    Usage :
        gate = DreamerGate(enable_training=False)
        if gate.can_train():
            gate.start_training(shadow_data_dir)
        else:
            gate.run_inference_only(world_model)
    """

    def __init__(self, enable_training: bool = False):
        """Initialise le gate.

        Args:
            enable_training: Valeur du Feature Flag ENABLE_DREAMER_TRAINING.
        """
        self.enable_training = enable_training
        self._training_active = False
        self._inference_count = 0
        self._muzero_agent = None  # Lazy-loaded

        if enable_training:
            logger.info("🧬 [DreamerGate] TRAINING MODE — MuZero will train on shadow data")
        else:
            logger.info("💤 [DreamerGate] INFERENCE ONLY — MuZero dormant, Shadow Learning active")

    def _get_muzero_agent(self):
        """Lazy-load MuZero agent (avoids importing torch at module level)."""
        if self._muzero_agent is None:
            try:
                from eva_lab.muzero.config import MuZeroConfigV3
                from eva_lab.muzero.agent import MuZeroAgent
                config = MuZeroConfigV3()
                self._muzero_agent = MuZeroAgent(config)

                # Try to load pre-trained weights if available
                weights_path = os.path.join(config.weights_path, "muzero_latest.pt")
                if os.path.exists(weights_path):
                    self._muzero_agent.load(weights_path)
                    logger.info(f"[DreamerGate] Loaded MuZero weights from {weights_path}")
                else:
                    logger.info("[DreamerGate] MuZero initialized (no pre-trained weights)")
            except ImportError as e:
                logger.warning(f"[DreamerGate] MuZero not available: {e}")
                self._muzero_agent = None
        return self._muzero_agent

    def can_train(self) -> bool:
        """Vérifie si l'entraînement est autorisé.

        Returns:
            True si le Feature Flag est activé.
        """
        return self.enable_training

    def start_training(self, data_dir: str, world_model=None) -> dict:
        """Lance l'entraînement du World Model (MuZero V3.1)."""
        if not self.enable_training:
            return {
                "status": "blocked",
                "reason": "ENABLE_DREAMER_TRAINING=False",
                "advice": "Activez le flag quand la RTX 3090 sera disponible",
            }

        if self._training_active:
             return {"status": "already_running"}

        # 1. Initialize Agent & Trainer
        agent = self._get_muzero_agent()
        if not agent:
             return {"status": "error", "reason": "MuZero Agent failed to load"}
        
        from eva_lab.muzero.trainer import MuZeroTrainer
        self.trainer = MuZeroTrainer(agent)

        # 2. Load Data from Shadow Learning
        loaded_count = self._load_shadow_data(data_dir)
        if loaded_count == 0:
             return {"status": "no_data", "reason": "No valid .jsonl files found"}

        # 3. Start Background Loop
        self._training_active = True
        import asyncio
        self._training_task = asyncio.create_task(self._training_loop())

        logger.info(f"🏋️ [DreamerGate] Training STARTED on {loaded_count} games.")
        
        return {
            "status": "training_started",
            "games_loaded": loaded_count,
            "buffer_size": agent.replay_buffer.size,
            "device": str(agent.device),
        }

    def _load_shadow_data(self, data_dir: str) -> int:
        """Load JSONL files into ReplayBuffer."""
        import json
        from eva_lab.muzero.agent import GameHistory
        
        if not os.path.exists(data_dir):
            return 0
            
        data_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
        count = 0
        agent = self._get_muzero_agent()
        
        for fname in data_files:
            try:
                path = os.path.join(data_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    game = GameHistory()
                    # Shadow Learning saves transitions. 
                    # We need to reconstruct episodes or treat each line as a step?
                    # Shadow lines are discrete transitions.
                    # We can group them into a single "continuous" game per file?
                    # Or just one giant game? ReplayBuffer expects GameHistory objects.
                    # Let's assume one file = one chunk of history.
                    
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        
                        # Convert data to format
                        obs = agent.process_observation(data.get("observation", {}))
                        
                        # Action: stored as dict in shadow {"type": "BUY", ...}
                        # Agent needs int index.
                        # Mapping: HOLD=0, BUY=1, SELL=2, SPLIT=3, CLOSE=4
                        act_map = {"HOLD":0, "BUY":1, "SELL":2, "SPLIT":3, "CLOSE":4}
                        act_data = data.get("action", {})
                        if isinstance(act_data, dict):
                            act_str = act_data.get("type", "HOLD")
                        else:
                            act_str = str(act_data)
                        action = act_map.get(act_str, 0)
                        
                        reward = float(data.get("reward", 0.0))
                        done = data.get("done", False)
                        
                        # Policy/Value: we don't have them in shadow (unless recorded from inference)
                        # Use placeholders
                        policy = [0.2] * 5
                        value = 0.0
                        
                        game.store(obs, action, reward, policy, value, done)
                    
                    if len(game) > 0:
                        agent.replay_buffer.save_game(game)
                        count += 1
            except Exception as e:
                logger.error(f"Failed to load {fname}: {e}")
                
        return count

    async def _training_loop(self):
        """Active Learning Loop."""
        import asyncio
        logger.info("[DreamerGate] Training Loop Active 🔄")
        
        while self._training_active:
            try:
                metrics = self.trainer.train_step()
                
                # Check if we actually trained
                if metrics.get("status") == "waiting_for_data":
                    # Slow down if waiting for data
                    await asyncio.sleep(5.0)
                    continue

                if self.trainer.steps % 10 == 0:
                    logger.info(f"[Dreamer] Step {self.trainer.steps} | Loss: {metrics.get('loss_total', 0):.4f}")
                    
                if self.trainer.steps % 100 == 0:
                    self.trainer.agent.save()
                    
                await asyncio.sleep(0.01) # Yield to event loop
            except Exception as e:
                logger.error(f"[Dreamer] Training error: {e}")
                await asyncio.sleep(5)

    def run_inference(self, observation: dict) -> dict:
        """Exécute une inférence World Model.

        Si MuZero est disponible, utilise le réseau de prédiction MCTS.
        Sinon, fallback sur des heuristiques RSI simples.

        Args:
            observation: Observation courante (prix, indicateurs).

        Returns:
            Prédiction du World Model.
        """
        self._inference_count += 1

        # Try MuZero MCTS inference first
        agent = self._get_muzero_agent()
        if agent is not None:
            try:
                import numpy as np
                # Build a minimal observation vector for MuZero
                price = observation.get("price", 0.0)
                indicators = observation.get("indicators", {})
                rsi = indicators.get("RSI", 50.0)

                # Create a simplified obs vector (pad to 142 features)
                obs_vec = np.zeros(142, dtype=np.float32)
                obs_vec[0] = price / 3000.0  # Normalized price
                obs_vec[1] = rsi / 100.0     # Normalized RSI
                for i, (k, v) in enumerate(indicators.items()):
                    if i + 2 < 142:
                        obs_vec[i + 2] = float(v) if isinstance(v, (int, float)) else 0.0

                result = agent.infer_action(obs_vec)
                return {
                    "action": result["action"], # CRITICAL FIX: Pass int action to Brain
                    "prediction": result["action_name"],
                    "confidence": result["confidence"],
                    "policy": result["policy"],
                    "value": result["value"],
                    "price_input": price,
                    "engine": "MuZero V3.1 MCTS",
                    "simulations": result["simulations"],
                    "mode": "training" if self._training_active else "inference_only",
                    "inference_count": self._inference_count,
                }
            except Exception as e:
                logger.warning(f"[DreamerGate] MuZero inference failed, falling back: {e}")

        # Fallback: simple RSI-based heuristics
        price = observation.get("price", 0.0)
        rsi = observation.get("indicators", {}).get("RSI", 50.0)

        action_int = 0 # HOLD
        if rsi < 30:
            prediction = "BULLISH_REVERSAL"
            confidence = 0.75
            action_int = 1 # BUY
        elif rsi > 70:
            prediction = "BEARISH_REVERSAL"
            confidence = 0.75
            action_int = 2 # SELL
        else:
            prediction = "CONSOLIDATION"
            confidence = 0.50
            action_int = 0 # HOLD

        return {
            "action": action_int, # CRITICAL FIX
            "prediction": prediction,
            "confidence": confidence,
            "price_input": price,
            "rsi_input": rsi,
            "engine": "RSI Heuristic (fallback)",
            "mode": "training" if self._training_active else "inference_only",
            "inference_count": self._inference_count,
        }

    def get_status(self) -> dict:
        """Retourne le statut complet du gate.

        Returns:
            Dictionnaire avec l'état d'activation et les stats.
        """
        muzero_available = self._muzero_agent is not None
        return {
            "enable_training": self.enable_training,
            "training_active": self._training_active,
            "inference_count": self._inference_count,
            "mode": "FULL" if self.enable_training else "SHADOW_ONLY",
            "engine": "MuZero V3.1" if muzero_available else "RSI Heuristic",
            "muzero_loaded": muzero_available,
        }

