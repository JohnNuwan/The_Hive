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

        if enable_training:
            logger.info("🧬 [DreamerGate] TRAINING MODE — DreamerV3 will train on shadow data")
        else:
            logger.info("💤 [DreamerGate] INFERENCE ONLY — DreamerV3 dormant, Shadow Learning active")

    def can_train(self) -> bool:
        """Vérifie si l'entraînement est autorisé.

        Returns:
            True si le Feature Flag est activé.
        """
        return self.enable_training

    def start_training(self, data_dir: str, world_model=None) -> dict:
        """Lance (ou simule) l'entraînement du World Model.

        Si le flag est True, charge les données JSONL depuis data_dir
        et configure le World Model pour l'entraînement.

        Args:
            data_dir: Répertoire contenant les fichiers .jsonl du Shadow Learning.
            world_model: Instance du WorldModel (optionnel, pour injection).

        Returns:
            Status dict avec les informations de l'entraînement.
        """
        if not self.enable_training:
            return {
                "status": "blocked",
                "reason": "ENABLE_DREAMER_TRAINING=False",
                "advice": "Activez le flag quand la RTX 3090 sera disponible",
            }

        # Compter les données disponibles
        data_files = []
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]

        if not data_files:
            return {
                "status": "no_data",
                "reason": "Aucun fichier .jsonl trouvé dans le répertoire shadow",
                "data_dir": data_dir,
            }

        self._training_active = True
        logger.info(
            f"🏋️ [DreamerGate] Training started on {len(data_files)} files "
            f"from {data_dir}"
        )

        # NOTE: L'entraînement réel nécessite PyTorch et le WorldModel de
        # world_model.py. Pour l'instant, on retourne le statut.
        # L'implémentation complète sera activée avec la RTX 3090.
        return {
            "status": "training_ready",
            "data_files": len(data_files),
            "data_dir": data_dir,
            "model": "DreamerV3 (FSQ + GRU)",
            "gpu_required": "RTX 3090 (24GB VRAM)",
        }

    def run_inference(self, observation: dict) -> dict:
        """Exécute une inférence World Model légère.

        En mode inference-only, utilise le DreamerModel stub pour
        des prédictions rapides sans entraînement.

        Args:
            observation: Observation courante (prix, indicateurs).

        Returns:
            Prédiction du World Model.
        """
        self._inference_count += 1

        # En mode dégradé, retourner une prédiction basée sur les tendances
        price = observation.get("price", 0.0)
        rsi = observation.get("indicators", {}).get("RSI", 50.0)

        if rsi < 30:
            prediction = "BULLISH_REVERSAL"
            confidence = 0.75
        elif rsi > 70:
            prediction = "BEARISH_REVERSAL"
            confidence = 0.75
        else:
            prediction = "CONSOLIDATION"
            confidence = 0.50

        return {
            "prediction": prediction,
            "confidence": confidence,
            "price_input": price,
            "rsi_input": rsi,
            "mode": "training" if self._training_active else "inference_only",
            "inference_count": self._inference_count,
        }

    def get_status(self) -> dict:
        """Retourne le statut complet du gate.

        Returns:
            Dictionnaire avec l'état d'activation et les stats.
        """
        return {
            "enable_training": self.enable_training,
            "training_active": self._training_active,
            "inference_count": self._inference_count,
            "mode": "FULL" if self.enable_training else "SHADOW_ONLY",
        }
