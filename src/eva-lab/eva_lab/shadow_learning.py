"""
Shadow Learning — Collecteur de Données pour DreamerV3
Part of Sovereign Stack V3.0 — Sprint 5

Collecte passivement les transitions (observation, action, reward, next_obs)
pendant le fonctionnement normal de E.V.A. Ces données sont stockées
dans un buffer circulaire et flushées périodiquement sur disque au format
attendu par DreamerV3/MuZero.

Quand `ENABLE_DREAMER_TRAINING=True` sera activé (RTX 3090), ces données
pourront être immédiatement consommées pour l'entraînement du World Model.

Architecture :
    ┌──────────┐  transitions  ┌──────────────┐  flush  ┌─────────┐
    │  Banker  │──────────────→│ ShadowBuffer │────────→│ .jsonl  │
    │  Probes  │               │ (circulaire) │         │ (disque)│
    │  Trades  │               └──────────────┘         └─────────┘
    └──────────┘                      │
                                      ▼ (si ENABLE_DREAMER_TRAINING)
                              ┌──────────────┐
                              │  WorldModel  │
                              │  (Training)  │
                              └──────────────┘
"""

import logging
import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Transition:
    """Une transition (s, a, r, s') pour l'entraînement du World Model.

    Attributes:
        timestamp: Horodatage ISO de la transition.
        observation: Vecteur d'observation (prix, indicateurs, positions).
        action: Action prise (BUY, SELL, HOLD, paramètres).
        reward: Récompense reçue (P&L, drawdown, etc.).
        next_observation: Observation suivante.
        metadata: Données additionnelles (symbol, timeframe, etc.).
        done: True si l'épisode est terminé (session coupée, SL/TP hit).
    """

    timestamp: str = ""
    observation: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    next_observation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    done: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW BUFFER
# ═══════════════════════════════════════════════════════════════════════════════


class ShadowBuffer:
    """Buffer circulaire pour stocker les transitions.

    Stocke les transitions en mémoire dans un deque de taille fixe.
    Quand le buffer est plein, les transitions les plus anciennes sont
    écrasées (comportement FIFO circulaire).

    Usage :
        buffer = ShadowBuffer(max_size=10000)
        buffer.add(transition)
        buffer.flush_to_disk("/path/to/data/")
    """

    def __init__(self, max_size: int = 10000):
        """Initialise le buffer.

        Args:
            max_size: Nombre maximum de transitions en mémoire.
        """
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._total_added: int = 0
        self._total_flushed: int = 0

    def add(self, transition: Transition):
        """Ajoute une transition au buffer.

        Args:
            transition: La transition à stocker.
        """
        if not transition.timestamp:
            transition.timestamp = datetime.now().isoformat()
        self._buffer.append(transition)
        self._total_added += 1

    def add_raw(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_observation: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        done: bool = False,
    ):
        """Ajoute une transition à partir de ses composants bruts.

        Raccourci pratique pour ne pas avoir à construire un objet Transition.

        Args:
            observation: État courant.
            action: Action prise.
            reward: Récompense reçue.
            next_observation: État suivant.
            metadata: Données additionnelles optionnelles.
            done: Si l'épisode est terminé.
        """
        self.add(Transition(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            metadata=metadata or {},
            done=done,
        ))

    def flush_to_disk(self, output_dir: str) -> int:
        """Écrit toutes les transitions du buffer sur disque en format JSONL.

        Chaque flush crée un fichier horodaté. Le buffer est vidé après.

        Args:
            output_dir: Répertoire de sortie pour les fichiers .jsonl.

        Returns:
            Nombre de transitions écrites.
        """
        if not self._buffer:
            return 0

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shadow_{timestamp}.jsonl"
        filepath = os.path.join(output_dir, filename)

        count = 0
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for transition in self._buffer:
                    json.dump(asdict(transition), f, ensure_ascii=False)
                    f.write("\n")
                    count += 1

            self._total_flushed += count
            self._buffer.clear()
            logger.info(
                f"[ShadowLearning] Flushed {count} transitions → {filepath}"
            )
        except Exception as e:
            logger.error(f"[ShadowLearning] Flush failed: {e}")

        return count

    @property
    def size(self) -> int:
        """Nombre actuel de transitions dans le buffer."""
        return len(self._buffer)

    def get_stats(self) -> dict:
        """Retourne les statistiques du buffer.

        Returns:
            Dictionnaire avec taille, total ajouté, total flushé.
        """
        return {
            "buffer_size": self.size,
            "max_size": self.max_size,
            "total_added": self._total_added,
            "total_flushed": self._total_flushed,
            "utilization_pct": round(self.size / self.max_size * 100, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW LEARNING SERVICE
# ═══════════════════════════════════════════════════════════════════════════════


class ShadowLearningService:
    """Service de collecte passive pour le Shadow Learning.

    Ce service tourne en tâche de fond et :
        1. Collecte les transitions (trades, signaux, observations) via `record()`.
        2. Stocke dans un ShadowBuffer circulaire.
        3. Flush périodiquement sur disque au format JSONL.
        4. Quand `ENABLE_DREAMER_TRAINING=True`, alimente aussi le World Model.

    Usage :
        service = ShadowLearningService(data_dir="/app/data/shadow")
        service.record_trade(observation, action, reward, next_obs)
        await service.start_auto_flush(interval=300)
    """

    def __init__(
        self,
        data_dir: str = "data/shadow_learning",
        buffer_size: int = 10000,
        dreamer_enabled: bool = False,
    ):
        """Initialise le service Shadow Learning.

        Args:
            data_dir: Répertoire de stockage des données.
            buffer_size: Taille du buffer circulaire.
            dreamer_enabled: Si True, alimente aussi le World Model.
        """
        self.data_dir = data_dir
        self.buffer = ShadowBuffer(max_size=buffer_size)
        self.dreamer_enabled = dreamer_enabled
        self._flush_task: Optional[asyncio.Task] = None

        os.makedirs(data_dir, exist_ok=True)
        logger.info(
            f"[ShadowLearning] Initialized (buffer: {buffer_size}, "
            f"dreamer: {'ON' if dreamer_enabled else 'OFF'})"
        )

    def record_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        volume: float,
        pnl: float,
        indicators: Optional[Dict[str, float]] = None,
        done: bool = False,
    ):
        """Enregistre un trade comme transition pour le World Model.

        Convertit les données de trading en format (s, a, r, s').

        Args:
            symbol: Paire de trading (XAUUSD, EURUSD, etc.).
            action: Action (BUY, SELL, HOLD, CLOSE).
            price: Prix d'exécution.
            volume: Volume du trade.
            pnl: Profit/Perte résultant.
            indicators: Indicateurs techniques (RSI, MACD, etc.).
            done: Si la position est fermée.
        """
        observation = {
            "price": price,
            "indicators": indicators or {},
        }
        action_data = {
            "type": action,
            "volume": volume,
            "symbol": symbol,
        }
        self.buffer.add_raw(
            observation=observation,
            action=action_data,
            reward=pnl,
            next_observation=observation,  # sera mis à jour au tick suivant
            metadata={"symbol": symbol, "source": "banker"},
            done=done,
        )

    def record_signal(
        self,
        signal_type: str,
        confidence: float,
        context: Dict[str, Any],
    ):
        """Enregistre un signal (alerte, anomalie) comme observation.

        Args:
            signal_type: Type de signal (RSI_DIVERGENCE, NEWS_SPIKE, etc.).
            confidence: Confiance du signal (0.0 → 1.0).
            context: Données de contexte du signal.
        """
        self.buffer.add_raw(
            observation={"signal": signal_type, "confidence": confidence, **context},
            action={"type": "OBSERVE"},
            reward=0.0,
            next_observation={},
            metadata={"source": "signal", "type": signal_type},
        )

    def record_probe(
        self,
        service_name: str,
        status: str,
        latency_ms: float,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Enregistre une probe système comme observation d'environnement.

        Args:
            service_name: Nom du service (hive-core, hive-banker, etc.).
            status: État du service (running, unhealthy, exited).
            latency_ms: Latence observée en ms.
            metrics: Métriques additionnelles (CPU, RAM, etc.).
        """
        self.buffer.add_raw(
            observation={
                "service": service_name,
                "status": status,
                "latency_ms": latency_ms,
                **(metrics or {}),
            },
            action={"type": "MONITOR"},
            reward=-1.0 if status != "running" else 0.0,
            next_observation={},
            metadata={"source": "probe", "service": service_name},
        )

    async def start_auto_flush(self, interval_seconds: int = 300):
        """Lance le flush périodique automatique en tâche de fond.

        Args:
            interval_seconds: Intervalle entre les flushes (défaut: 5 min).
        """
        logger.info(
            f"[ShadowLearning] Auto-flush started (every {interval_seconds}s)"
        )
        while True:
            await asyncio.sleep(interval_seconds)
            count = self.buffer.flush_to_disk(self.data_dir)
            if count > 0:
                logger.info(f"[ShadowLearning] Auto-flush: {count} transitions saved")

    def manual_flush(self) -> int:
        """Force un flush immédiat du buffer.

        Returns:
            Nombre de transitions écrites.
        """
        return self.buffer.flush_to_disk(self.data_dir)

    def get_stats(self) -> dict:
        """Retourne les statistiques du Shadow Learning.

        Returns:
            Dictionnaire complet des stats.
        """
        return {
            **self.buffer.get_stats(),
            "data_dir": self.data_dir,
            "dreamer_enabled": self.dreamer_enabled,
        }

    def count_stored_files(self) -> int:
        """Compte le nombre de fichiers JSONL stockés sur disque.

        Returns:
            Nombre de fichiers .jsonl dans le répertoire de données.
        """
        try:
            return len([
                f for f in os.listdir(self.data_dir)
                if f.endswith(".jsonl")
            ])
        except FileNotFoundError:
            return 0
