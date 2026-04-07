"""Agent DreamerV3 JAX pour l'inference live et l'evaluation Arena."""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from eva_lab.muzero.dreamer_networks import make_dreamer_networks
from eva_lab.muzero.dreamer_trainer import DreamerTrainerJAX

logger = logging.getLogger(__name__)


class JAXDreamerAgent:
    """Expose l'inference greedy DreamerV3 pour le live et l'Arena."""

    def __init__(self, config) -> None:
        """Initialise le modele DreamerV3 et ses etats JAX.

        Args:
            config: Configuration compatible avec ``MuZeroConfigV3``.
        """
        self.config = config
        self.transformed = make_dreamer_networks(config)
        self.trainer = DreamerTrainerJAX(config, self.transformed)
        sample_obs = np.zeros((1, *config.observation_shape), dtype=np.float32)
        params, _ = self.trainer.init_params(sample_obs)
        self.params = params
        self.trainer.params["wm"] = params
        self._rng = jax.random.PRNGKey(42)

        logger.info(
            "[JAXDreamerAgent] Agent operationnel. Etat latent=%s.",
            config.hidden_state_size,
        )

    @staticmethod
    def _extract_params(payload: Any) -> tuple[Any, dict[str, Any] | None]:
        """Extrait les poids Dreamer depuis un checkpoint heterogene.

        Args:
            payload (Any): Charge utile deserializee depuis le checkpoint.

        Returns:
            tuple[Any, dict[str, Any] | None]: Parametres du modele et etat
            d'optimiseur optionnel.
        """
        if isinstance(payload, dict):
            if "params" in payload:
                params = payload.get("params")
                opt_states = payload.get("opt_states")
                return params, opt_states if isinstance(opt_states, dict) else None
            if "wm" in payload:
                return payload.get("wm"), None
        return payload, None

    @staticmethod
    def _normalize_action_label(action_id: int) -> str:
        """Traduit un identifiant d'action en libelle stable.

        Args:
            action_id (int): Identifiant numerique de l'action.

        Returns:
            str: Libelle de l'action.
        """
        action_names = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]
        if 0 <= int(action_id) < len(action_names):
            return action_names[int(action_id)]
        return f"ACT_{action_id}"

    def process_observation(self, observation: dict[str, Any]) -> np.ndarray:
        """Convertit une observation live en vecteur compatible Dreamer.

        Args:
            observation (dict[str, Any]): Charge utile live du banker.

        Returns:
            np.ndarray: Vecteur ``[32]`` compatible avec le modele.
        """
        candle = observation.get("latest_candle", {}) or {}
        indicators = observation.get("indicators", {}) or {}
        price = float(observation.get("price", candle.get("close", 0.0)) or 0.0)
        close_price = float(candle.get("close", price) or price)
        open_price = float(candle.get("open", close_price) or close_price)
        high_price = float(candle.get("high", close_price) or close_price)
        low_price = float(candle.get("low", close_price) or close_price)
        volume = float(candle.get("tick_volume", candle.get("volume", 0.0)) or 0.0)
        spread = float(candle.get("spread", 0.0) or 0.0)

        obs_vec = np.zeros(self.config.observation_shape[0], dtype=np.float32)
        obs_vec[:26] = np.array(
            [
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                float(indicators.get("EMA_200", close_price) or close_price),
                float(indicators.get("RSI", 50.0) or 50.0),
                float(indicators.get("MACD_Hist", 0.0) or 0.0),
                float(indicators.get("VWAP", close_price) or close_price),
                float(indicators.get("OBV", 0.0) or 0.0),
                float(indicators.get("Momentum", 0.0) or 0.0),
                float(indicators.get("TRIX", 0.0) or 0.0),
                float(indicators.get("Stoch_K", 50.0) or 50.0),
                float(indicators.get("Stoch_D", 50.0) or 50.0),
                float(indicators.get("CCI", 0.0) or 0.0),
                float(indicators.get("ADX", 0.0) or 0.0),
                float(indicators.get("ADX_Plus_DI", 0.0) or 0.0),
                float(indicators.get("ADX_Minus_DI", 0.0) or 0.0),
                float(indicators.get("Ichi_Tenkan", close_price) or close_price),
                float(indicators.get("Ichi_Kijun", close_price) or close_price),
                float(indicators.get("Ichi_Senkou_A", close_price) or close_price),
                float(indicators.get("Ichi_Senkou_B", close_price) or close_price),
                float(indicators.get("ATR", 0.0) or 0.0),
                float(indicators.get("BB_Pct", 0.5) or 0.5),
                spread / max(close_price, 1e-8),
                float(indicators.get("Return_1", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )

        timestamp_raw = observation.get("timestamp")
        if timestamp_raw:
            try:
                current_time = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
            except ValueError:
                current_time = datetime.utcnow()
        else:
            current_time = datetime.utcnow()

        volatility = 0.0
        if close_price > 0:
            volatility = min((high_price - low_price) / max(close_price, 1e-8) * 100.0, 1.0)
        obs_vec[26:] = np.array(
            [
                0.0,
                0.0,
                0.0,
                current_time.hour / 23.0 if 23 else 0.0,
                current_time.weekday() / 6.0 if 6 else 0.0,
                volatility,
            ],
            dtype=np.float32,
        )
        return obs_vec

    def _initial_state(self, batch_size: int = 1) -> jnp.ndarray:
        """Retourne l'etat latent initial du RSSM.

        Args:
            batch_size (int): Nombre d'instances a initialiser.

        Returns:
            jnp.ndarray: Etat latent plat.
        """
        self._rng, rng = jax.random.split(self._rng)
        return self.transformed.apply(self.params, rng, 2, batch_size)

    def _infer_step(
        self,
        observation: np.ndarray,
        *,
        prev_action: jnp.ndarray,
        prev_state: jnp.ndarray,
        exploration: bool,
    ) -> tuple[dict[str, object], jnp.ndarray, jnp.ndarray]:
        """Execute une inference Dreamer avec etat latent explicite.

        Args:
            observation (np.ndarray): Observation vectorisee.
            prev_action (jnp.ndarray): Action precedente en one-hot.
            prev_state (jnp.ndarray): Etat latent precedent.
            exploration (bool): Active ou non un echantillonnage stochastique.

        Returns:
            tuple[dict[str, object], jnp.ndarray, jnp.ndarray]: Prediction
            normalisee, nouvel etat latent et action one-hot courante.
        """
        obs_jax = jnp.array(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
        self._rng, rng_obs, rng_action = jax.random.split(self._rng, 3)
        _, posterior, _, _ = self.transformed.apply(
            self.params,
            rng_obs,
            0,
            obs_jax,
            prev_action,
            prev_state,
        )
        probs, value = self.transformed.apply(self.params, rng_obs, 3, posterior)
        probs_np = np.asarray(jax.device_get(probs))[0]
        value_np = float(np.asarray(jax.device_get(value)).reshape(-1)[0])
        probs_np = np.clip(probs_np, 0.0, None)
        total = float(probs_np.sum())
        if total <= 0.0:
            probs_np = np.zeros(self.config.action_space_size, dtype=np.float32)
            probs_np[0] = 1.0
        else:
            probs_np = probs_np / total

        if exploration:
            action = int(
                jax.device_get(
                    jax.random.categorical(rng_action, jnp.log(jnp.array(probs_np) + 1e-8))
                )
            )
        else:
            action = int(np.argmax(probs_np))
        action_one_hot = jax.nn.one_hot(
            jnp.array([action]),
            self.config.action_space_size,
            dtype=jnp.float32,
        )

        return (
            {
                "action": action,
                "action_name": self._normalize_action_label(action),
                "policy": probs_np.tolist(),
                "value": value_np,
                "confidence": float(probs_np[action]) if action < len(probs_np) else 0.0,
                "simulations": 0,
            },
            posterior,
            action_one_hot,
        )

    def infer_action(self, observation: dict[str, Any] | np.ndarray) -> dict[str, object]:
        """Execute une inference greedy Dreamer a partir d'une observation.

        Args:
            observation (dict[str, Any] | np.ndarray): Observation brute ou vecteur.

        Returns:
            dict[str, object]: Action, politique, confiance et valeur.
        """
        if isinstance(observation, dict):
            obs_vec = self.process_observation(observation)
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)
        initial_action = jnp.zeros((1, self.config.action_space_size), dtype=jnp.float32)
        state = self._initial_state(batch_size=1)
        result, _, _ = self._infer_step(
            obs_vec,
            prev_action=initial_action,
            prev_state=state,
            exploration=False,
        )
        return result

    def play_game(self, env, exploration: bool = False) -> dict[str, Any]:
        """Joue une evaluation Dreamer complete dans l'environnement.

        Args:
            env: Environnement de trading compatible avec MuZero.
            exploration (bool): Active ou non un echantillonnage stochastique.

        Returns:
            dict[str, Any]: Resume simple du deroulement.
        """
        observation, _ = env.reset()
        done = False
        steps = 0
        previous_action = jnp.zeros((1, self.config.action_space_size), dtype=jnp.float32)
        latent_state = self._initial_state(batch_size=1)
        last_result: dict[str, object] = {}

        while not done and steps < self.config.max_moves:
            steps += 1
            last_result, latent_state, previous_action = self._infer_step(
                np.asarray(observation, dtype=np.float32),
                prev_action=previous_action,
                prev_state=latent_state,
                exploration=exploration,
            )
            observation, _, done, _, _ = env.step(int(last_result["action"]))

        return {
            "steps": steps,
            "last_action": last_result.get("action_name"),
            "confidence": last_result.get("confidence"),
        }

    def save(self, path: str) -> None:
        """Sauvegarde les poids Dreamer dans un checkpoint structure.

        Args:
            path (str): Chemin cible du checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "params": self.params,
            "opt_states": dict(self.trainer.opt_states or {}),
            "engine": "dreamer",
            "horizon": str(getattr(self.config, "horizon", "scalp") or "scalp"),
            "symbols": list(getattr(self.config, "symbols", []) or []),
        }
        with open(path, "wb") as file_obj:
            pickle.dump(payload, file_obj)
        logger.info("[JAXDreamerAgent] Checkpoint sauvegarde: %s", path)

    def load(self, path: str) -> None:
        """Recharge les poids Dreamer depuis un checkpoint heterogene.

        Args:
            path (str): Chemin du checkpoint a charger.
        """
        with open(path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        params, opt_states = self._extract_params(payload)
        self.params = params
        self.trainer.params["wm"] = params
        if opt_states:
            self.trainer.opt_states.update(opt_states)
        logger.info("[JAXDreamerAgent] Checkpoint charge: %s", path)
