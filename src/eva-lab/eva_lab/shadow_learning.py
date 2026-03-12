"""
Shadow Learning â€” Collecteur de DonnÃ©es pour DreamerV3
Part of Sovereign Stack V3.0 â€” Sprint 5

Collecte passivement les transitions (observation, action, reward, next_obs)
pendant le fonctionnement normal de E.V.A. Ces donnÃ©es sont stockÃ©es
dans un buffer circulaire et flushÃ©es pÃ©riodiquement sur disque au format
attendu par DreamerV3/MuZero.

Quand `ENABLE_DREAMER_TRAINING=True` sera activÃ© (RTX 3090), ces donnÃ©es
pourront Ãªtre immÃ©diatement consommÃ©es pour l'entraÃ®nement du World Model.

Architecture :
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  transitions  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  flush  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚  Banker  â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â†’â”‚ ShadowBuffer â”‚â”€â”€â”€â”€â”€â”€â”€â”€â†’â”‚ .jsonl  â”‚
    â”‚  Probes  â”‚               â”‚ (circulaire) â”‚         â”‚ (disque)â”‚
    â”‚  Trades  â”‚               â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                      â”‚
                                      â–¼ (si ENABLE_DREAMER_TRAINING)
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚  WorldModel  â”‚
                              â”‚  (Training)  â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DATA MODEL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@dataclass
class Transition:
    """Une transition (s, a, r, s') pour l'entraÃ®nement du World Model.

    Attributes:
        timestamp: Horodatage ISO de la transition.
        observation: Vecteur d'observation (prix, indicateurs, positions).
        action: Action prise (BUY, SELL, HOLD, paramÃ¨tres).
        reward: RÃ©compense reÃ§ue (P&L, drawdown, etc.).
        next_observation: Observation suivante.
        metadata: DonnÃ©es additionnelles (symbol, timeframe, etc.).
        done: True si l'Ã©pisode est terminÃ© (session coupÃ©e, SL/TP hit).
    """

    timestamp: str = ""
    observation: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    next_observation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    done: bool = False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SHADOW BUFFER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class ShadowBuffer:
    """Buffer circulaire pour stocker les transitions.

    Stocke les transitions en mÃ©moire dans un deque de taille fixe.
    Quand le buffer est plein, les transitions les plus anciennes sont
    Ã©crasÃ©es (comportement FIFO circulaire).

    Usage :
        buffer = ShadowBuffer(max_size=10000)
        buffer.add(transition)
        buffer.flush_to_disk("/path/to/data/")
    """

    def __init__(self, max_size: int = 10000):
        """Initialise le buffer.

        Args:
            max_size: Nombre maximum de transitions en mÃ©moire.
        """
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._total_added: int = 0
        self._total_flushed: int = 0

    def add(self, transition: Transition):
        """Ajoute une transition au buffer.

        Args:
            transition: La transition Ã  stocker.
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
        """Ajoute une transition Ã  partir de ses composants bruts.

        Raccourci pratique pour ne pas avoir Ã  construire un objet Transition.

        Args:
            observation: Ã‰tat courant.
            action: Action prise.
            reward: RÃ©compense reÃ§ue.
            next_observation: Ã‰tat suivant.
            metadata: DonnÃ©es additionnelles optionnelles.
            done: Si l'Ã©pisode est terminÃ©.
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
        """Ã‰crit toutes les transitions du buffer sur disque en format JSONL.

        Chaque flush crÃ©e un fichier horodatÃ©. Le buffer est vidÃ© aprÃ¨s.

        Args:
            output_dir: RÃ©pertoire de sortie pour les fichiers .jsonl.

        Returns:
            Nombre de transitions Ã©crites.
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
                f"[ShadowLearning] Flushed {count} transitions â†’ {filepath}"
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
            Dictionnaire avec taille, total ajoutÃ©, total flushÃ©.
        """
        return {
            "buffer_size": self.size,
            "max_size": self.max_size,
            "total_added": self._total_added,
            "total_flushed": self._total_flushed,
            "utilization_pct": round(self.size / self.max_size * 100, 1),
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SHADOW LEARNING SERVICE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class ShadowLearningService:
    """Service de collecte passive pour le Shadow Learning.

    Ce service tourne en tÃ¢che de fond et :
        1. Collecte les transitions (trades, signaux, observations) via `record()`.
        2. Stocke dans un ShadowBuffer circulaire.
        3. Flush pÃ©riodiquement sur disque au format JSONL.
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
            data_dir: RÃ©pertoire de stockage des donnÃ©es.
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
        observation: Optional[Dict[str, Any]] = None,
        next_observation: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        done: bool = False,
    ):
        """Enregistre un trade comme transition pour le World Model.

        Convertit les donnees de trading en format (s, a, r, s').

        Args:
            symbol: Paire de trading (XAUUSD, EURUSD, etc.).
            action: Action (BUY, SELL, HOLD, CLOSE).
            price: Prix d'execution.
            volume: Volume du trade.
            pnl: Profit/Perte resultant.
            indicators: Indicateurs techniques (RSI, MACD, etc.).
            observation: Observation complete optionnelle.
            next_observation: Observation suivante optionnelle.
            metadata: Metadonnees d'episode ou d'origine.
            timestamp: Horodatage ISO de la transition.
            done: Si la position est fermee.
        """
        current_observation = observation or {
            "price": price,
            "indicators": indicators or {},
        }
        future_observation = next_observation or current_observation
        action_data = {
            "type": action,
            "volume": volume,
            "symbol": symbol,
        }
        transition = Transition(
            timestamp=timestamp or "",
            observation=current_observation,
            action=action_data,
            reward=pnl,
            next_observation=future_observation,
            metadata={"symbol": symbol, "source": "banker", **(metadata or {})},
            done=done,
        )
        self.buffer.add(transition)

    def record_signal(
        self,
        signal_type: str,
        confidence: float,
        context: Dict[str, Any],
    ):
        """Enregistre un signal (alerte, anomalie) comme observation.

        Args:
            signal_type: Type de signal (RSI_DIVERGENCE, NEWS_SPIKE, etc.).
            confidence: Confiance du signal (0.0 â†’ 1.0).
            context: DonnÃ©es de contexte du signal.
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
        """Enregistre une probe systÃ¨me comme observation d'environnement.

        Args:
            service_name: Nom du service (hive-core, hive-banker, etc.).
            status: Ã‰tat du service (running, unhealthy, exited).
            latency_ms: Latence observÃ©e en ms.
            metrics: MÃ©triques additionnelles (CPU, RAM, etc.).
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
        """Lance le flush pÃ©riodique automatique en tÃ¢che de fond.

        Args:
            interval_seconds: Intervalle entre les flushes (dÃ©faut: 5 min).
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
        """Force un flush immÃ©diat du buffer.

        Returns:
            Nombre de transitions Ã©crites.
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
        """Compte le nombre de fichiers JSONL stockÃ©s sur disque.

        Returns:
            Nombre de fichiers .jsonl dans le rÃ©pertoire de donnÃ©es.
        """
        try:
            return len([
                f for f in os.listdir(self.data_dir)
                if f.endswith(".jsonl")
            ])
        except FileNotFoundError:
            return 0

