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
        self._ensemble_min_edge = float(os.getenv("ENSEMBLE_MIN_EDGE", "0.15"))
        self._ensemble_requires_double_validation = True

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

    def _resolve_inference_checkpoint(
        self,
        horizon: str,
        selection_policy_override: str | None = None,
        engine: str = "muzero",
    ) -> tuple[Path | None, dict[str, object]]:
        """Retourne le checkpoint JAX autorise pour l'inference live.

        Args:
            horizon (str): Horizon cible.
            selection_policy_override (str | None): Politique de selection a imposer.
            engine (str): Moteur de prediction cible.

        Returns:
            tuple[object | None, dict[str, object]]: Chemin retenu et metadonnees.
        """
        return self._promoter.resolve_live_checkpoint(
            horizon,
            selection_policy=selection_policy_override,
            engine=engine,
        )

    def _get_muzero_inference_agent(
        self,
        horizon: str,
        selection_policy_override: str | None = None,
    ):
        """Charge a la demande un agent MuZero JAX pour l'inference live.

        Args:
            horizon (str): Horizon de prediction.
            selection_policy_override (str | None): Politique de selection a imposer.

        Returns:
            object | None: Agent JAX charge, sinon ``None``.
        """
        horizon = (horizon or "intraday").lower()
        checkpoint_path, selection_meta = self._resolve_inference_checkpoint(
            horizon,
            selection_policy_override=selection_policy_override,
        )
        checkpoint_mtime = checkpoint_path.stat().st_mtime if checkpoint_path else None
        meta = self._jax_inference_meta.get(horizon)

        if (
            meta
            and meta.get("path") == str(checkpoint_path)
            and meta.get("mtime") == checkpoint_mtime
            and meta.get("selection") == selection_meta.get("selection")
            and meta.get("policy") == selection_meta.get("policy")
        ):
            return self._jax_inference_agents.get(horizon)

        if checkpoint_path is None:
            self._jax_inference_agents.pop(horizon, None)
            self._jax_inference_meta[horizon] = selection_meta
            logger.warning(
                "[DreamerGate] Aucun checkpoint live promu pour %s. Aucun agent JAX charge.",
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

    @staticmethod
    def _build_model_version(
        checkpoint_meta: dict[str, object],
        checkpoint_path: str | None,
    ) -> str | None:
        """Construit une version de modele lisible pour le live.

        Args:
            checkpoint_meta (dict[str, object]): Metadonnees de selection live.
            checkpoint_path (str | None): Checkpoint effectivement retenu.

        Returns:
            str | None: Version lisible du modele, sinon ``None``.
        """
        manifest = checkpoint_meta.get("manifest", {}) or {}
        version = str(manifest.get("challenger_id") or "").strip()
        if version:
            return version
        if checkpoint_path:
            return Path(str(checkpoint_path)).stem
        return None

    def _build_blocked_live_response(
        self,
        *,
        observation: dict,
        horizon: str,
        checkpoint_meta: dict[str, object],
        reason: str,
        prediction: str = "NO_CHAMPION_DEPLOYED",
        engine: str = "muzero",
        model_status: str = "blocked",
        service: str = "live_inference_cpu",
    ) -> dict[str, object]:
        """Construit une reponse live bloquee sans fallback heuristique.

        Args:
            observation (dict): Observation brute fournie au gate.
            horizon (str): Horizon live vise.
            checkpoint_meta (dict[str, object]): Metadonnees de selection live.
            reason (str): Raison detaillee du blocage.
            prediction (str): Etiquette de prediction a remonter.

        Returns:
            dict[str, object]: Charge utile stable pour le banker live.
        """
        normalized_engine = self._promoter.normalize_engine_name(engine)
        checkpoint_path = checkpoint_meta.get("path")
        model_version = self._build_model_version(checkpoint_meta, checkpoint_path)
        return {
            "action": 0,
            "prediction": prediction,
            "confidence": 1.0,
            "policy": [],
            "value": 0.0,
            "price_input": float(observation.get("price", 0.0) or 0.0),
            "engine": self._promoter.get_engine_label(normalized_engine, variant="champion"),
            "engine_name": normalized_engine,
            "horizon": horizon,
            "checkpoint": checkpoint_path,
            "selection": checkpoint_meta.get("selection"),
            "selection_policy": checkpoint_meta.get("policy"),
            "manifest": checkpoint_meta.get("manifest"),
            "simulations": 0,
            "mode": "training" if self._training_active else "inference_only",
            "inference_count": self._inference_count,
            "reason": reason,
            "service": service,
            "device": "cpu",
            "model_status": model_status,
            "model_version": model_version,
        }

    @staticmethod
    def _normalize_action_label(action_id: object, fallback: object = None) -> str:
        """Normalise une action numerique en libelle stable.

        Args:
            action_id (object): Identifiant brut du moteur.
            fallback (object): Libelle secondaire eventuel.

        Returns:
            str: Action normalisee (`BUY`, `SELL`, `HOLD`).
        """
        action_map = {
            0: "HOLD",
            1: "BUY",
            2: "SELL",
        }
        try:
            normalized_id = int(action_id)
        except (TypeError, ValueError):
            normalized_id = None
        if normalized_id in action_map:
            return action_map[normalized_id]
        candidate = str(fallback or "").strip().upper()
        return candidate or "HOLD"

    def _build_action_scores(self, result: dict[str, object]) -> dict[str, float]:
        """Transforme une prediction moteur en scores normalises.

        Args:
            result (dict[str, object]): Reponse d'un moteur live.

        Returns:
            dict[str, float]: Scores `BUY`, `SELL`, `HOLD` exploitables par l'ensemble.
        """
        confidence = max(0.0, min(float(result.get("confidence", 0.0) or 0.0), 1.0))
        action_label = self._normalize_action_label(result.get("action"), result.get("prediction"))
        scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        if action_label in {"BUY", "SELL"}:
            scores[action_label] = confidence
            scores["HOLD"] = max(0.0, 1.0 - confidence)
        else:
            scores["HOLD"] = max(confidence, 0.5)
        return scores

    def _run_live_inference_for_engine(
        self,
        observation: dict[str, object],
        *,
        engine: str,
        strict_live: bool = True,
    ) -> dict[str, object]:
        """Execute une inference live pour un moteur cible.

        Args:
            observation (dict[str, object]): Observation live brute.
            engine (str): Moteur cible (`muzero` ou `dreamer`).
            strict_live (bool): Indique si le chemin doit respecter les contraintes live CPU.

        Returns:
            dict[str, object]: Prediction brute ou blocage explicite.
        """
        normalized_engine = self._promoter.normalize_engine_name(engine)
        horizon = str(observation.get("horizon", "scalp") or "scalp").lower()
        selection_policy = ChampionPromoter.normalize_live_selection_policy(
            observation.get("selection_policy"),
            default="champion_only",
        )
        checkpoint_meta: dict[str, object] = {
            "selection": "none",
            "policy": selection_policy,
            "manifest": {},
            "path": None,
        }

        if strict_live and horizon != "scalp":
            return self._build_blocked_live_response(
                observation=observation,
                horizon=horizon,
                checkpoint_meta=checkpoint_meta,
                reason="Le live CPU unifie ne supporte que l'horizon scalp.",
                prediction="UNSUPPORTED_HORIZON",
                engine=normalized_engine,
                model_status="blocked",
                service="live_inference_cpu",
            )

        if strict_live and selection_policy != "champion_only":
            checkpoint_meta["selection"] = "blocked_policy"
            return self._build_blocked_live_response(
                observation=observation,
                horizon=horizon,
                checkpoint_meta=checkpoint_meta,
                reason="Le live CPU exige champion_only.",
                prediction="INVALID_SELECTION_POLICY",
                engine=normalized_engine,
                model_status="blocked",
                service="live_inference_cpu",
            )

        if normalized_engine == "muzero":
            agent = self._get_muzero_inference_agent(
                horizon,
                selection_policy_override="champion_only" if strict_live else selection_policy,
            )
            checkpoint_meta = self._jax_inference_meta.get(horizon, {})
            checkpoint_path = checkpoint_meta.get("path")
            model_version = self._build_model_version(checkpoint_meta, checkpoint_path)
            if agent is not None:
                try:
                    result = agent.infer_action(observation)
                    return {
                        "action": result["action"],
                        "prediction": result["action_name"],
                        "confidence": result["confidence"],
                        "policy": result["policy"],
                        "value": result["value"],
                        "price_input": observation.get("price", 0.0),
                        "engine": "MuZero JAX Live CPU",
                        "engine_name": "muzero",
                        "horizon": horizon,
                        "checkpoint": checkpoint_path,
                        "selection": checkpoint_meta.get("selection"),
                        "selection_policy": "champion_only" if strict_live else checkpoint_meta.get("policy"),
                        "manifest": checkpoint_meta.get("manifest"),
                        "simulations": result["simulations"],
                        "mode": "training" if self._training_active else "inference_only",
                        "inference_count": self._inference_count,
                        "service": "live_inference_cpu" if strict_live else "dreamer_predict",
                        "device": "cpu" if strict_live else "jax_default",
                        "model_status": "live",
                        "model_version": model_version,
                    }
                except Exception as exc:
                    logger.warning("[DreamerGate] Inference live MuZero impossible pour %s: %s", horizon, exc)
                    return self._build_blocked_live_response(
                        observation=observation,
                        horizon=horizon,
                        checkpoint_meta=checkpoint_meta,
                        reason=f"Inference MuZero impossible: {exc}",
                        prediction="INFERENCE_ERROR",
                        engine=normalized_engine,
                        model_status="error",
                        service="live_inference_cpu" if strict_live else "dreamer_predict",
                    )

            return self._build_blocked_live_response(
                observation=observation,
                horizon=horizon,
                checkpoint_meta=checkpoint_meta,
                reason="Aucun champion MuZero valide n'est autorise pour le live CPU.",
                engine=normalized_engine,
                model_status="blocked",
                service="live_inference_cpu" if strict_live else "dreamer_predict",
            )

        checkpoint_path, checkpoint_meta = self._resolve_inference_checkpoint(
            horizon,
            selection_policy_override="champion_only" if strict_live else selection_policy,
            engine="dreamer",
        )
        checkpoint_meta = dict(checkpoint_meta or {})
        checkpoint_meta["path"] = str(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint_path is None:
            return self._build_blocked_live_response(
                observation=observation,
                horizon=horizon,
                checkpoint_meta=checkpoint_meta,
                reason="Aucun champion DreamerV3 valide n'est disponible.",
                engine="dreamer",
                model_status="blocked",
                service="dreamer_predict",
            )

        return self._build_blocked_live_response(
            observation=observation,
            horizon=horizon,
            checkpoint_meta=checkpoint_meta,
            reason="Le runtime DreamerV3 live n'est pas encore active dans cette usine.",
            prediction="DREAMER_RUNTIME_UNAVAILABLE",
            engine="dreamer",
            model_status="unavailable",
            service="dreamer_predict",
        )

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
                    "service": "dreamer_predict",
                    "device": "jax_default",
                    "model_status": "live" if checkpoint_path else "fallback",
                    "model_version": self._build_model_version(checkpoint_meta, checkpoint_path),
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
                "policy": [],
                "value": 0.0,
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
                "service": "dreamer_predict",
                "device": "jax_default",
                "model_status": "blocked",
                "model_version": self._build_model_version(
                    checkpoint_meta,
                    checkpoint_meta.get("path"),
                ),
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
            "policy": [],
            "value": 0.0,
            "price_input": price,
            "rsi_input": rsi,
            "engine": "RSI Heuristic (fallback)",
            "horizon": horizon,
            "mode": "training" if self._training_active else "inference_only",
            "inference_count": self._inference_count,
            "service": "dreamer_predict",
            "device": "cpu",
            "model_status": "fallback",
            "model_version": None,
        }

    def run_live_inference(self, observation: dict) -> dict[str, object]:
        """Execute une inference live stricte pour le service CPU dedie.

        Ce chemin refuse tout horizon autre que ``scalp`` et interdit toute
        politique de selection autre que ``champion_only``.

        Args:
            observation (dict): Observation live envoyee par le banker.

        Returns:
            dict[str, object]: Prediction brute du champion live ou blocage propre.
        """
        self._inference_count += 1
        return self._run_live_inference_for_engine(observation, engine="muzero", strict_live=True)

    def run_ensemble_inference(self, observation: dict[str, object]) -> dict[str, object]:
        """Arbitre une prediction finale entre MuZero et DreamerV3.

        Args:
            observation (dict[str, object]): Observation live brute du banker.

        Returns:
            dict[str, object]: Sous-decisions, scores et decision finale.
        """
        self._inference_count += 1
        muzero_result = self._run_live_inference_for_engine(observation, engine="muzero", strict_live=True)
        dreamer_result = self._run_live_inference_for_engine(observation, engine="dreamer", strict_live=True)

        muzero_live = str(muzero_result.get("model_status") or "").lower() == "live"
        dreamer_live = str(dreamer_result.get("model_status") or "").lower() == "live"
        governance_mode = "ensemble_50_50"
        degraded_reason = None

        if not (muzero_live and dreamer_live):
            governance_mode = "degraded_muzero_only"
            degraded_reason = (
                str(dreamer_result.get("reason") or "dreamer_indisponible")
                if not dreamer_live
                else str(muzero_result.get("reason") or "muzero_indisponible")
            )
            final_result = dict(muzero_result)
            final_result["governance"] = {
                "mode": governance_mode,
                "degraded_fallback_reason": degraded_reason,
                "requires_double_validation": self._ensemble_requires_double_validation,
                "muzero": muzero_result,
                "dreamer": dreamer_result,
                "scores": {
                    "muzero": self._build_action_scores(muzero_result),
                    "dreamer": self._build_action_scores(dreamer_result),
                },
            }
            final_result["engine"] = "Ensemble 50/50"
            final_result["engine_name"] = "ensemble"
            final_result["selection"] = "degraded_muzero_only"
            final_result["selection_policy"] = "champion_only"
            final_result["ensemble_mode"] = governance_mode
            final_result["degraded_fallback_reason"] = degraded_reason
            final_result["model_status"] = "degraded"
            return final_result

        muzero_scores = self._build_action_scores(muzero_result)
        dreamer_scores = self._build_action_scores(dreamer_result)
        final_scores = {
            action: round((muzero_scores[action] + dreamer_scores[action]) / 2.0, 6)
            for action in ("BUY", "SELL", "HOLD")
        }
        best_action = max(final_scores, key=final_scores.get)
        sorted_scores = sorted(final_scores.values(), reverse=True)
        score_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else sorted_scores[0]
        disagreement = (
            self._normalize_action_label(muzero_result.get("action"), muzero_result.get("prediction"))
            != self._normalize_action_label(dreamer_result.get("action"), dreamer_result.get("prediction"))
        )
        if disagreement or score_gap < self._ensemble_min_edge:
            best_action = "HOLD"
        action_to_id = {"HOLD": 0, "BUY": 1, "SELL": 2}
        return {
            "action": action_to_id[best_action],
            "prediction": best_action,
            "confidence": final_scores.get(best_action, 0.0),
            "policy": [],
            "value": max(
                float(muzero_result.get("value", 0.0) or 0.0),
                float(dreamer_result.get("value", 0.0) or 0.0),
            ),
            "price_input": float(observation.get("price", 0.0) or 0.0),
            "engine": "Ensemble 50/50",
            "engine_name": "ensemble",
            "horizon": str(observation.get("horizon", "scalp") or "scalp").lower(),
            "checkpoint": None,
            "selection": "ensemble_50_50",
            "selection_policy": "champion_only",
            "manifest": None,
            "simulations": int(muzero_result.get("simulations", 0) or 0),
            "mode": "training" if self._training_active else "inference_only",
            "inference_count": self._inference_count,
            "service": "live_inference_cpu",
            "device": "cpu",
            "model_status": "ensemble",
            "model_version": None,
            "ensemble_mode": governance_mode,
            "degraded_fallback_reason": None,
            "governance": {
                "mode": governance_mode,
                "degraded_fallback_reason": None,
                "requires_double_validation": self._ensemble_requires_double_validation,
                "score_gap": round(score_gap, 6),
                "disagreement": disagreement,
                "muzero": muzero_result,
                "dreamer": dreamer_result,
                "scores": {
                    "muzero": muzero_scores,
                    "dreamer": dreamer_scores,
                    "final": final_scores,
                },
            },
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
            "live_cpu_supported_horizons": ["scalp"],
            "live_cpu_selection_policy": "champion_only",
            "ensemble_mode": "vote_50_50",
            "ensemble_min_edge": self._ensemble_min_edge,
            "double_validation_required": self._ensemble_requires_double_validation,
            "dreamer_pipeline": {
                "status": "shadow_orchestrated",
                "live_runtime": "degraded_until_promoted",
            },
        }
