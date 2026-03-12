"""Porte d'acces Dreamer/MuZero pour l'inference live et le shadow training."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.shadow_dataset import load_shadow_games

logger = logging.getLogger(__name__)


class DreamerGate:
    """Orchestre l'inference live et le shadow training des modeles trading.

    Le chemin d'inference live privilegie les champions MuZero JAX generes par
    EVA Lab. Le chemin de shadow training conserve l'ancien pipeline PyTorch afin
    de ne pas casser la collecte historique deja en place.
    """

    def __init__(self, enable_training: bool = False):
        """Initialise le gate Dreamer.

        Args:
            enable_training (bool): Active ou non le chemin de shadow training.
        """
        self.enable_training = enable_training
        self._training_active = False
        self._inference_count = 0
        self._muzero_agent = None
        self._training_task: Optional[asyncio.Task] = None
        self._jax_inference_agents: dict[str, object] = {}
        self._jax_inference_meta: dict[str, dict[str, object]] = {}
        self._promoter = ChampionPromoter()

        if enable_training:
            logger.info("[DreamerGate] Mode shadow training actif.")
        else:
            logger.info("[DreamerGate] Mode inference uniquement actif.")

    def _get_muzero_agent(self):
        """Charge l'ancien agent MuZero pour le shadow training.

        Returns:
            object | None: Agent legacy charge, sinon ``None``.
        """
        if self._muzero_agent is None:
            try:
                from eva_lab.muzero.agent import MuZeroAgent
                from eva_lab.muzero.config import MuZeroConfigV3

                config = MuZeroConfigV3()
                self._muzero_agent = MuZeroAgent(config)

                weights_path = os.path.join(config.weights_path, "muzero_champion.pkl")
                if os.path.exists(weights_path):
                    self._muzero_agent.load(weights_path)
                    logger.info("[DreamerGate] Agent legacy charge depuis %s.", weights_path)
                else:
                    logger.info("[DreamerGate] Agent legacy initialise sans checkpoint champion.")
            except ImportError as exc:
                logger.warning("[DreamerGate] Agent legacy indisponible: %s", exc)
                self._muzero_agent = None
        return self._muzero_agent

    def _resolve_inference_checkpoint(self, horizon: str) -> tuple[Path | None, dict[str, object]]:
        """Retourne le checkpoint JAX autorise pour l'inference live.

        Args:
            horizon (str): Horizon cible.

        Returns:
            tuple[object | None, dict[str, object]]: Chemin retenu et metadonnees.
        """
        return self._promoter.resolve_live_checkpoint(horizon)

    def _get_muzero_inference_agent(self, horizon: str):
        """Charge a la demande un agent MuZero JAX pour l'inference live.

        Args:
            horizon (str): Horizon de prediction.

        Returns:
            object | None: Agent JAX charge, sinon ``None``.
        """
        horizon = (horizon or "intraday").lower()
        checkpoint_path, selection_meta = self._resolve_inference_checkpoint(horizon)
        checkpoint_mtime = checkpoint_path.stat().st_mtime if checkpoint_path else None
        meta = self._jax_inference_meta.get(horizon)

        if (
            meta
            and meta.get("path") == str(checkpoint_path)
            and meta.get("mtime") == checkpoint_mtime
            and meta.get("selection") == selection_meta.get("selection")
        ):
            return self._jax_inference_agents.get(horizon)

        if checkpoint_path is None:
            self._jax_inference_agents.pop(horizon, None)
            self._jax_inference_meta[horizon] = selection_meta
            logger.warning(
                "[DreamerGate] Aucun checkpoint live promu pour %s. Fallback heuristique actif.",
                horizon,
            )
            return None

        try:
            from eva_lab.muzero.config import MuZeroConfigV3
            from eva_lab.muzero.jax_agent import JAXMuZeroAgent

            config = MuZeroConfigV3(horizon=horizon)
            agent = JAXMuZeroAgent(config)
            agent.load(str(checkpoint_path))
            self._jax_inference_agents[horizon] = agent
            self._jax_inference_meta[horizon] = {
                "path": str(checkpoint_path),
                "mtime": checkpoint_mtime,
                **selection_meta,
            }
            logger.info(
                "[DreamerGate] Agent JAX %s charge depuis %s (%s).",
                horizon,
                checkpoint_path,
                selection_meta.get("selection", "unknown"),
            )
            return agent
        except Exception as exc:
            logger.warning("[DreamerGate] Chargement JAX impossible pour %s: %s", horizon, exc)
            self._jax_inference_agents.pop(horizon, None)
            self._jax_inference_meta.pop(horizon, None)
            return None

    def can_train(self) -> bool:
        """Indique si le shadow training est autorise.

        Returns:
            bool: ``True`` si le training est active.
        """
        return self.enable_training

    def start_training(self, data_dir: str, world_model=None) -> dict:
        """Lance l'ancien pipeline de shadow training.

        Args:
            data_dir (str): Dossier contenant les fichiers ``jsonl`` de shadow.
            world_model: Parametre legacy conserve pour compatibilite.

        Returns:
            dict: Statut de lancement.
        """
        if not self.enable_training:
            return {
                "status": "blocked",
                "reason": "ENABLE_DREAMER_TRAINING=False",
                "advice": "Activez le flag uniquement sur le serveur GPU.",
            }

        if self._training_active:
            return {"status": "already_running"}

        agent = self._get_muzero_agent()
        if not agent:
            return {"status": "error", "reason": "Chargement MuZero legacy impossible"}

        from eva_lab.muzero.trainer import MuZeroTrainer

        self.trainer = MuZeroTrainer(agent)
        loaded_count = self._load_shadow_data(data_dir)
        if loaded_count == 0:
            return {"status": "no_data", "reason": "Aucun fichier .jsonl exploitable"}

        self._training_active = True
        self._training_task = asyncio.create_task(self._training_loop())
        logger.info("[DreamerGate] Shadow training demarre sur %s episodes.", loaded_count)
        return {
            "status": "training_started",
            "games_loaded": loaded_count,
            "buffer_size": agent.replay_buffer.size,
            "device": str(agent.device),
        }

    def _load_shadow_data(self, data_dir: str) -> int:
        """Charge les donnees shadow dans le replay buffer legacy.

        Args:
            data_dir (str): Dossier des fichiers ``jsonl``.

        Returns:
            int: Nombre de fichiers charges.
        """
        count = 0
        agent = self._get_muzero_agent()
        if agent is None:
            return 0

        games = load_shadow_games(
            [data_dir],
            observation_size=agent.config.observation_shape[0],
            action_space_size=agent.config.action_space_size,
        )
        for game in games:
            agent.replay_buffer.save_game(game)
            count += 1
        return count

    async def _training_loop(self) -> None:
        """Execute la boucle legacy de shadow training."""
        logger.info("[DreamerGate] Boucle de shadow training active.")
        while self._training_active:
            try:
                metrics = self.trainer.train_step()
                if metrics.get("status") == "waiting_for_data":
                    await asyncio.sleep(5.0)
                    continue

                if self.trainer.steps % 10 == 0:
                    logger.info(
                        "[DreamerGate] Etape %s | loss=%.4f",
                        self.trainer.steps,
                        float(metrics.get("loss_total", 0.0)),
                    )
                if self.trainer.steps % 100 == 0:
                    self.trainer.agent.save()
                await asyncio.sleep(0.01)
            except Exception as exc:
                logger.error("[DreamerGate] Erreur shadow training: %s", exc)
                await asyncio.sleep(5.0)

    def run_inference(self, observation: dict) -> dict:
        """Execute une inference live a partir des champions JAX.

        Args:
            observation (dict): Observation live envoyee par le banker.

        Returns:
            dict: Action proposee et metadonnees d'inference.
        """
        self._inference_count += 1
        horizon = str(observation.get("horizon", os.getenv("DREAMER_DEFAULT_HORIZON", "intraday"))).lower()
        agent = self._get_muzero_inference_agent(horizon)
        checkpoint_meta = self._jax_inference_meta.get(horizon, {})
        if agent is not None:
            try:
                result = agent.infer_action(observation)
                checkpoint_path = checkpoint_meta.get("path")
                return {
                    "action": result["action"],
                    "prediction": result["action_name"],
                    "confidence": result["confidence"],
                    "policy": result["policy"],
                    "value": result["value"],
                    "price_input": observation.get("price", 0.0),
                    "engine": checkpoint_meta.get("engine_label", "MuZero JAX"),
                    "horizon": horizon,
                    "checkpoint": checkpoint_path,
                    "selection": checkpoint_meta.get("selection"),
                    "selection_policy": checkpoint_meta.get("policy"),
                    "manifest": checkpoint_meta.get("manifest"),
                    "simulations": result["simulations"],
                    "mode": "training" if self._training_active else "inference_only",
                    "inference_count": self._inference_count,
                }
            except Exception as exc:
                logger.warning("[DreamerGate] Inference JAX en echec, fallback heuristique: %s", exc)

        if checkpoint_meta.get("policy") == "champion_only":
            selection = str(checkpoint_meta.get("selection", "blocked_champion") or "blocked_champion")
            logger.warning(
                "[DreamerGate] Inference live bloquee sur %s: aucun champion valide (%s).",
                horizon,
                selection,
            )
            return {
                "action": 0,
                "prediction": "NO_CHAMPION_DEPLOYED",
                "confidence": 1.0,
                "price_input": float(observation.get("price", 0.0) or 0.0),
                "engine": "Champion bloque",
                "horizon": horizon,
                "checkpoint": checkpoint_meta.get("path"),
                "selection": selection,
                "selection_policy": checkpoint_meta.get("policy"),
                "manifest": checkpoint_meta.get("manifest"),
                "mode": "training" if self._training_active else "inference_only",
                "inference_count": self._inference_count,
                "reason": "Aucun champion positif n'est autorise en live.",
            }

        price = float(observation.get("price", 0.0) or 0.0)
        indicators = observation.get("indicators", {}) or {}
        rsi = float(indicators.get("RSI", 50.0) or 50.0)

        action_int = 0
        if rsi < 30:
            prediction = "BULLISH_REVERSAL"
            confidence = 0.75
            action_int = 1
        elif rsi > 70:
            prediction = "BEARISH_REVERSAL"
            confidence = 0.75
            action_int = 2
        else:
            prediction = "CONSOLIDATION"
            confidence = 0.50

        return {
            "action": action_int,
            "prediction": prediction,
            "confidence": confidence,
            "price_input": price,
            "rsi_input": rsi,
            "engine": "RSI Heuristic (fallback)",
            "horizon": horizon,
            "mode": "training" if self._training_active else "inference_only",
            "inference_count": self._inference_count,
        }

    def get_status(self) -> dict:
        """Retourne l'etat complet du gate.

        Returns:
            dict: Statut d'activation, agents charges et mode courant.
        """
        jax_agents = {
            horizon: {
                "path": meta.get("path"),
                "selection": meta.get("selection"),
                "policy": meta.get("policy"),
            }
            for horizon, meta in self._jax_inference_meta.items()
        }
        return {
            "enable_training": self.enable_training,
            "training_active": self._training_active,
            "inference_count": self._inference_count,
            "mode": "FULL" if self.enable_training else "SHADOW_ONLY",
            "engine": "MuZero JAX" if bool(jax_agents) else "RSI Heuristic",
            "muzero_loaded": bool(jax_agents),
            "live_selection_policy": self._promoter.get_live_selection_policy(),
            "jax_agents": jax_agents,
            "legacy_agent_loaded": self._muzero_agent is not None,
        }
