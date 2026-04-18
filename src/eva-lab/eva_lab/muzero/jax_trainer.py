"""Trainer MuZero JAX/Optax avec loss hybride, unroll et mise a jour JIT."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import List, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)


class TrainingBatch(NamedTuple):
    """Lot de donnees utilise pour une mise a jour MuZero."""

    observations: jnp.ndarray
    actions: jnp.ndarray
    target_values: jnp.ndarray
    target_rewards: jnp.ndarray
    target_policies: jnp.ndarray


class HostTrainingBatch(NamedTuple):
    """Lot de donnees prepare sur l'hote avant transfert JAX."""

    observations: np.ndarray
    actions: np.ndarray
    target_values: np.ndarray
    target_rewards: np.ndarray
    target_policies: np.ndarray


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

    def init_params(self, sample_obs):
        """Initialise les poids et l'etat de l'optimiseur."""
        self._rng, init_key = jax.random.split(self._rng)
        params = self.transformed.init(init_key, sample_obs)
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def loss_fn(self, params, batch: TrainingBatch):
        """Calcule la loss hybride MuZero sur un lot."""
        from eva_lab.muzero.jax_networks import scalar_to_support, support_to_scalar

        hidden_state, logits, value_logits = self.initial_apply(params, None, batch.observations)
        root_target_policy = batch.target_policies[:, 0]
        root_target_value = batch.target_values[:, 0]
        predicted_root_value = support_to_scalar(value_logits, self.config.support_size)
        target_value_support = scalar_to_support(root_target_value, self.config.support_size)
        loss_val = jnp.mean(optax.softmax_cross_entropy(value_logits, target_value_support))

        loss_pol = jnp.mean(optax.softmax_cross_entropy(logits, root_target_policy))
        loss_rew = jnp.array(0.0, dtype=value_logits.dtype)

        for step_idx in range(self.config.num_unroll_steps):
            action_onehot = jax.nn.one_hot(
                batch.actions[:, step_idx],
                self.config.action_space_size,
            )
            next_hidden_state, reward_logits, logits, value_logits = self.recurrent_apply(
                params,
                None,
                hidden_state,
                action_onehot,
            )
            hidden_state = next_hidden_state

            target_reward_support = scalar_to_support(batch.target_rewards[:, step_idx], self.config.support_size)
            target_value_support = scalar_to_support(batch.target_values[:, step_idx + 1], self.config.support_size)
            
            loss_rew += jnp.mean(optax.softmax_cross_entropy(reward_logits, target_reward_support))
            loss_val += jnp.mean(optax.softmax_cross_entropy(value_logits, target_value_support))
            loss_pol += jnp.mean(
                optax.softmax_cross_entropy(logits, batch.target_policies[:, step_idx + 1])
            )

        total_loss = loss_val + loss_rew + loss_pol
        clipped_policy = jnp.clip(root_target_policy, 1e-8, 1.0)
        policy_entropy = -jnp.mean(jnp.sum(root_target_policy * jnp.log(clipped_policy), axis=-1))
        policy_top1_share = jnp.mean(jnp.max(root_target_policy, axis=-1))
        root_has_position = jnp.abs(batch.observations[:, -6]) > 1e-6
        root_legal_action_count = jnp.mean(jnp.where(root_has_position, 5.0, 3.0))
        invalid_root_action_masked_rate = jnp.mean(
            (self.config.action_space_size - jnp.where(root_has_position, 5.0, 3.0))
            / float(self.config.action_space_size)
        )
        priority_errors = jnp.abs(
            jnp.squeeze(predicted_root_value, axis=-1) - root_target_value
        )
        return total_loss, {
            "loss_total": total_loss,
            "loss_val": loss_val,
            "loss_rew": loss_rew,
            "loss_pol": loss_pol,
            "policy_entropy": policy_entropy,
            "policy_top1_share": policy_top1_share,
            "root_legal_action_count": root_legal_action_count,
            "invalid_root_action_masked_rate": invalid_root_action_masked_rate,
            "priority_errors": priority_errors,
        }

    @property
    def update_fn(self):
        """Expose une etape d'optimisation JITtee."""

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
        host_batch, _prepare_ms = self.prepare_batch_host(batch_list)
        batch, _device_put_ms = self.device_put_batch(host_batch)
        return batch

    def prepare_batch_host(self, batch_list: List[tuple]) -> tuple[HostTrainingBatch, float]:
        """Construit un lot contigu sur CPU avant envoi vers JAX.

        Args:
            batch_list (List[tuple]): Echantillons issus du replay buffer.

        Returns:
            tuple[HostTrainingBatch, float]: Lot hote contigu et duree de
                preparation en millisecondes.
        """
        batch_size = len(batch_list)
        num_unroll = self.config.num_unroll_steps
        action_dim = self.config.action_space_size
        observation_shape = tuple(self.config.observation_shape)
        uniform_policy = np.full(
            (action_dim,),
            1.0 / float(action_dim),
            dtype=np.float32,
        )
        started_at = perf_counter()

        observations = np.empty((batch_size, *observation_shape), dtype=np.float32)
        actions = np.empty((batch_size, num_unroll), dtype=np.int32)
        target_values = np.zeros((batch_size, num_unroll + 1), dtype=np.float32)
        target_rewards = np.zeros((batch_size, num_unroll), dtype=np.float32)
        target_policies = np.empty(
            (batch_size, num_unroll + 1, action_dim),
            dtype=np.float32,
        )

        for batch_index, sample in enumerate(batch_list):
            if len(sample) < 2:
                raise ValueError("Echantillon MuZero invalide: start_idx manquant.")
            game = sample[0]
            start_idx = sample[1]
            observations[batch_index] = np.asarray(
                game.observations[start_idx],
                dtype=np.float32,
            ).reshape(observation_shape)

            for step_idx in range(num_unroll + 1):
                sample_idx = start_idx + step_idx
                if sample_idx < len(game):
                    target_values[batch_index, step_idx] = float(game.values[sample_idx])
                    target_policies[batch_index, step_idx] = np.asarray(
                        game.policies[sample_idx],
                        dtype=np.float32,
                    )
                    if step_idx < num_unroll:
                        actions[batch_index, step_idx] = int(game.actions[sample_idx])
                        target_rewards[batch_index, step_idx] = float(game.rewards[sample_idx])
                else:
                    target_policies[batch_index, step_idx] = uniform_policy
                    if step_idx < num_unroll:
                        actions[batch_index, step_idx] = 0

        elapsed_ms = (perf_counter() - started_at) * 1000.0
        return (
            HostTrainingBatch(
                observations=np.ascontiguousarray(observations),
                actions=np.ascontiguousarray(actions),
                target_values=np.ascontiguousarray(target_values),
                target_rewards=np.ascontiguousarray(target_rewards),
                target_policies=np.ascontiguousarray(target_policies),
            ),
            elapsed_ms,
        )

    def device_put_batch(self, host_batch: HostTrainingBatch) -> tuple[TrainingBatch, float]:
        """Transfere un lot hote vers le device JAX cible.

        Args:
            host_batch (HostTrainingBatch): Lot contigu prepare sur CPU.

        Returns:
            tuple[TrainingBatch, float]: Lot JAX et duree de transfert
                vers le device en millisecondes.
        """
        started_at = perf_counter()
        batch = TrainingBatch(
            observations=jax.device_put(host_batch.observations),
            actions=jax.device_put(host_batch.actions),
            target_values=jax.device_put(host_batch.target_values),
            target_rewards=jax.device_put(host_batch.target_rewards),
            target_policies=jax.device_put(host_batch.target_policies),
        )
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        return batch, elapsed_ms
