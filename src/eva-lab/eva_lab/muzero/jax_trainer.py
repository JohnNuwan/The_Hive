"""Trainer MuZero JAX/Optax avec loss hybride, unroll et mise a jour JIT."""

from __future__ import annotations

import logging
from typing import List, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from eva_lab.muzero.replay_buffer import ReplayValidationError

logger = logging.getLogger(__name__)


class TrainingBatch(NamedTuple):
    """Lot de donnees utilise pour une mise a jour MuZero."""

    observations: jnp.ndarray
    actions: jnp.ndarray
    target_values: jnp.ndarray
    target_rewards: jnp.ndarray
    target_policies: jnp.ndarray


class MuZeroTrainerJAX:
    """Encapsule l'initialisation, la loss et l'optimisation MuZero."""

    def __init__(self, config, transformed_nets):
        """Memorise la configuration et prepare l'optimiseur AdamW."""
        self.config = config
        self.transformed = transformed_nets
        self.initial_apply, self.recurrent_apply = self.transformed.apply
        self.optimizer = optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._rng = jax.random.PRNGKey(42)
        self._update_fn = self._build_update_fn()

    def init_params(self, sample_obs):
        """Initialise les poids et l'etat de l'optimiseur."""
        self._rng, init_key = jax.random.split(self._rng)
        params = self.transformed.init(init_key, sample_obs)
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def loss_fn(self, params, batch: TrainingBatch):
        """Calcule la loss hybride MuZero sur un lot."""
        hidden_state, logits, value = self.initial_apply(params, None, batch.observations)
        loss_val = jnp.mean(optax.l2_loss(value.squeeze(-1), batch.target_values[:, 0]))
        loss_pol = jnp.mean(optax.softmax_cross_entropy(logits, batch.target_policies[:, 0]))
        loss_rew = 0.0

        for step_idx in range(self.config.num_unroll_steps):
            action_onehot = jax.nn.one_hot(
                batch.actions[:, step_idx],
                self.config.action_space_size,
            )
            next_hidden_state, reward, logits, value = self.recurrent_apply(
                params,
                None,
                hidden_state,
                action_onehot,
            )
            hidden_state = next_hidden_state

            loss_rew += jnp.mean(optax.l2_loss(reward.squeeze(-1), batch.target_rewards[:, step_idx]))
            loss_val += jnp.mean(optax.l2_loss(value.squeeze(-1), batch.target_values[:, step_idx + 1]))
            loss_pol += jnp.mean(
                optax.softmax_cross_entropy(logits, batch.target_policies[:, step_idx + 1])
            )

        total_loss = loss_val + loss_rew + loss_pol
        return total_loss, {
            "loss_total": total_loss,
            "loss_val": loss_val,
            "loss_rew": loss_rew,
            "loss_pol": loss_pol,
        }

    @property
    def update_fn(self):
        """Expose une etape d'optimisation JITtee."""

        return self._update_fn

    def _build_update_fn(self):
        """Construit une unique fonction JITtee reutilisee a chaque step."""

        def step(params, opt_state, batch):
            (loss, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(params, batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, metrics

        return jax.jit(step)

    def prepare_batch(self, batch_list: List[tuple]) -> TrainingBatch:
        """Convertit une liste de jeux en tenseurs JAX pour l'entrainement.

        Le buffer priorise renvoie actuellement des tuples de la forme
        ``(game, start_idx, tree_idx)``. Le trainer ne consomme que
        ``game`` et ``start_idx``.
        """
        obs_list = []
        action_list = []
        target_v_list = []
        target_r_list = []
        target_p_list = []
        num_unroll = self.config.num_unroll_steps
        expected_observation_size = int(np.prod(self.config.observation_shape))

        for sample in batch_list:
            if len(sample) < 2:
                raise ReplayValidationError(
                    "REPLAY_SAMPLE_INVALID",
                    "Echantillon MuZero invalide: start_idx manquant.",
                )
            game = sample[0]
            start_idx = sample[1]
            if start_idx < 0 or start_idx >= len(game.observations):
                raise ReplayValidationError(
                    "REPLAY_SAMPLE_INVALID",
                    "start_idx MuZero hors bornes pour le jeu echantillonne.",
                )
            obs_list.append(
                self._normalize_observation(
                    game.observations[start_idx],
                    expected_observation_size,
                )
            )
            actions = []
            values = []
            rewards = []
            policies = []

            for step_idx in range(num_unroll + 1):
                sample_idx = start_idx + step_idx
                if sample_idx < len(game):
                    values.append(game.values[sample_idx])
                    policies.append(
                        self._normalize_policy(
                            game.policies[sample_idx],
                            self.config.action_space_size,
                        )
                    )
                    if step_idx < num_unroll:
                        actions.append(
                            self._normalize_action(
                                game.actions[sample_idx],
                                self.config.action_space_size,
                            )
                        )
                        rewards.append(game.rewards[sample_idx])
                else:
                    values.append(0.0)
                    policies.append(
                        np.ones(self.config.action_space_size) / self.config.action_space_size,
                    )
                    if step_idx < num_unroll:
                        actions.append(0)
                        rewards.append(0.0)

            action_list.append(actions)
            target_v_list.append(values)
            target_r_list.append(rewards)
            target_p_list.append(policies)

        return TrainingBatch(
            observations=jnp.array(obs_list),
            actions=jnp.array(action_list),
            target_values=jnp.array(target_v_list),
            target_rewards=jnp.array(target_r_list),
            target_policies=jnp.array(target_p_list),
        )

    @staticmethod
    def _normalize_observation(raw_observation: object, expected_size: int) -> np.ndarray:
        """Normalise une observation MuZero vers un vecteur dense stable.

        Args:
            raw_observation (object): Observation brute issue du replay.
            expected_size (int): Taille plate attendue par le reseau.

        Returns:
            np.ndarray: Vecteur ``float32`` de taille attendue.

        Raises:
            ReplayValidationError: Si l'observation est incompatible.
        """

        observation = np.asarray(raw_observation, dtype=np.float32).reshape(-1)
        if observation.size != expected_size:
            raise ReplayValidationError(
                "BATCH_SHAPE_MISMATCH",
                (
                    "Observation MuZero incompatible: "
                    f"taille attendue={expected_size}, recue={observation.size}."
                ),
            )
        return observation

    @staticmethod
    def _normalize_action(raw_action: object, action_space_size: int) -> int:
        """Normalise une action replay vers un indice discret.

        Args:
            raw_action (object): Action brute issue du replay.
            action_space_size (int): Taille de l'espace d'actions.

        Returns:
            int: Indice discret compatible avec MuZero.

        Raises:
            ReplayValidationError: Si l'action ne peut pas etre interpretee.
        """

        action_array = np.asarray(raw_action)
        if action_array.ndim == 0 or action_array.size == 1:
            action_index = int(action_array.reshape(-1)[0])
        elif action_array.ndim == 1 and action_array.size == action_space_size:
            action_index = int(np.argmax(action_array.astype(np.float32)))
        else:
            raise ReplayValidationError(
                "BATCH_SHAPE_MISMATCH",
                (
                    "Action MuZero incompatible avec l'espace discret: "
                    f"forme recue={tuple(action_array.shape)}."
                ),
            )
        if action_index < 0 or action_index >= action_space_size:
            raise ReplayValidationError(
                "BATCH_SHAPE_MISMATCH",
                (
                    "Indice d'action MuZero hors bornes: "
                    f"{action_index} pour action_space_size={action_space_size}."
                ),
            )
        return action_index

    @staticmethod
    def _normalize_policy(raw_policy: object, action_space_size: int) -> np.ndarray:
        """Normalise une politique replay vers un vecteur probabiliste stable.

        Args:
            raw_policy (object): Politique brute issue du replay.
            action_space_size (int): Taille attendue du vecteur.

        Returns:
            np.ndarray: Politique ``float32`` normalisee.

        Raises:
            ReplayValidationError: Si la politique ne peut pas etre interpretee.
        """

        policy = np.asarray(raw_policy, dtype=np.float32).reshape(-1)
        if policy.size != action_space_size:
            raise ReplayValidationError(
                "BATCH_SHAPE_MISMATCH",
                (
                    "Politique MuZero incompatible: "
                    f"taille attendue={action_space_size}, recue={policy.size}."
                ),
            )
        total = float(np.sum(policy))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(
                action_space_size,
                1.0 / max(action_space_size, 1),
                dtype=np.float32,
            )
        return (policy / total).astype(np.float32)
