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

    @staticmethod
    def _normalize_policy(policy: jnp.ndarray) -> jnp.ndarray:
        """Normalise une distribution de policy sans creer de NaN.

        Args:
            policy (jnp.ndarray): Distribution a normaliser.

        Returns:
            jnp.ndarray: Distribution normalisee.
        """
        denom = jnp.sum(policy, axis=-1, keepdims=True)
        safe_denom = jnp.where(denom > 1e-8, denom, 1.0)
        normalized = policy / safe_denom
        uniform = jnp.full_like(policy, 1.0 / float(policy.shape[-1]))
        return jnp.where(denom > 1e-8, normalized, uniform)

    def _build_root_legal_policy(self, root_has_position: jnp.ndarray) -> jnp.ndarray:
        """Construit une policy uniforme sur les actions legales a la racine.

        Args:
            root_has_position (jnp.ndarray): Indique si la racine porte deja une
                position ouverte.

        Returns:
            jnp.ndarray: Distribution uniforme sur les actions legales.
        """
        no_position_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        with_position_mask = jnp.ones((self.config.action_space_size,), dtype=jnp.float32)
        legal_mask = jnp.where(
            root_has_position[:, None],
            with_position_mask[None, :],
            no_position_mask[None, :],
        )
        return self._normalize_policy(legal_mask)

    def _build_uniform_policy_from_target(self, target_policy: jnp.ndarray) -> jnp.ndarray:
        """Infere une policy uniforme a partir des actions non nulles de la cible.

        Args:
            target_policy (jnp.ndarray): Distribution cible brute.

        Returns:
            jnp.ndarray: Distribution uniforme sur les actions reputees legales.
        """
        legal_mask = jnp.where(target_policy > 1e-8, 1.0, 0.0).astype(target_policy.dtype)
        return self._normalize_policy(legal_mask)

    def _smooth_target_policy(
        self,
        target_policy: jnp.ndarray,
        uniform_legal_policy: jnp.ndarray,
        alpha: float,
    ) -> jnp.ndarray:
        """Lisse la cible policy pour reduire la pression cross-entropy.

        Args:
            target_policy (jnp.ndarray): Distribution cible initiale.
            uniform_legal_policy (jnp.ndarray): Policy uniforme sur les actions
                legales.
            alpha (float): Poids du melange avec la policy uniforme.

        Returns:
            jnp.ndarray: Distribution cible lisse.
        """
        alpha = float(max(0.0, min(1.0, alpha)))
        target_policy = self._normalize_policy(target_policy)
        smoothing_temperature = float(
            getattr(self.config, "policy_target_smoothing_temperature", 1.40) or 1.40
        )
        if abs(smoothing_temperature - 1.0) > 1e-6:
            target_policy = self._normalize_policy(
                jnp.power(jnp.clip(target_policy, 1e-8, 1.0), 1.0 / smoothing_temperature)
            )
        if alpha <= 0.0:
            return target_policy
        return self._normalize_policy(
            ((1.0 - alpha) * target_policy) + (alpha * uniform_legal_policy)
        )

    @staticmethod
    def _compute_loss_pol_per_head(loss_pol, num_unroll_steps: int):
        """Normalise la loss policy par tete de prediction.

        Args:
            loss_pol: Somme brute des cross-entropies policy.
            num_unroll_steps (int): Nombre de tetes recurrentes apres la racine.

        Returns:
            Meme type que ``loss_pol``: Loss policy moyenne par tete.
        """
        policy_head_count = max(1.0, float(1 + int(num_unroll_steps)))
        return loss_pol / policy_head_count

    @staticmethod
    def _compute_weighted_loss_pol_per_head(
        root_loss_pol,
        unroll_loss_pol_sum,
        num_unroll_steps: int,
        root_weight: float,
        unroll_weight: float,
    ):
        """Calcule une moyenne par tete compatible avec les poids policy.

        Args:
            root_loss_pol: Loss policy brute de la racine.
            unroll_loss_pol_sum: Somme des losses policy recurrentes.
            num_unroll_steps (int): Nombre de tetes recurrentes.
            root_weight (float): Poids applique a la tete racine.
            unroll_weight (float): Poids applique a chaque tete recurrente.

        Returns:
            Meme type que ``root_loss_pol``: Loss policy moyenne ponderee par tete.
        """
        effective_head_count = max(
            1.0,
            float(max(root_weight, 0.0) + (max(unroll_weight, 0.0) * max(0, int(num_unroll_steps)))),
        )
        weighted_loss = (max(root_weight, 0.0) * root_loss_pol) + (
            max(unroll_weight, 0.0) * unroll_loss_pol_sum
        )
        return weighted_loss / effective_head_count

    def loss_fn(self, params, batch: TrainingBatch):
        """Calcule la loss hybride MuZero sur un lot."""
        from eva_lab.muzero.jax_networks import scalar_to_support, support_to_scalar

        hidden_state, logits, value_logits = self.initial_apply(params, None, batch.observations)
        root_target_policy = batch.target_policies[:, 0]
        root_has_position = jnp.abs(batch.observations[:, -6]) > 1e-6
        root_target_policy_smoothed = self._smooth_target_policy(
            root_target_policy,
            self._build_root_legal_policy(root_has_position),
            float(getattr(self.config, "policy_target_smoothing_alpha_root", 0.28) or 0.28),
        )
        root_target_value = batch.target_values[:, 0]
        predicted_root_value = support_to_scalar(value_logits, self.config.support_size)
        target_value_support = scalar_to_support(root_target_value, self.config.support_size)
        loss_val = jnp.mean(optax.softmax_cross_entropy(value_logits, target_value_support))

        root_loss_pol = jnp.mean(optax.softmax_cross_entropy(logits, root_target_policy_smoothed))
        unroll_actions = jnp.swapaxes(batch.actions, 0, 1)
        unroll_target_rewards = jnp.swapaxes(batch.target_rewards, 0, 1)
        unroll_target_values = jnp.swapaxes(batch.target_values[:, 1:], 0, 1)
        unroll_target_policies = jnp.swapaxes(batch.target_policies[:, 1:], 0, 1)
        unroll_alpha = float(
            getattr(self.config, "policy_target_smoothing_alpha_unroll", 0.18) or 0.18
        )

        def _scan_unroll_step(
            carry_hidden_state: jnp.ndarray,
            scan_inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            """Calcule une tete recurrente complete sans reboucler en Python."""
            action_ids, target_rewards, target_values, target_policies = scan_inputs
            action_onehot = jax.nn.one_hot(action_ids, self.config.action_space_size)
            next_hidden_state, reward_logits, step_logits, step_value_logits = self.recurrent_apply(
                params,
                None,
                carry_hidden_state,
                action_onehot,
            )
            target_reward_support = scalar_to_support(target_rewards, self.config.support_size)
            target_value_support = scalar_to_support(target_values, self.config.support_size)
            target_policy_smoothed = self._smooth_target_policy(
                target_policies,
                self._build_uniform_policy_from_target(target_policies),
                unroll_alpha,
            )
            step_loss_rew = jnp.mean(
                optax.softmax_cross_entropy(reward_logits, target_reward_support)
            )
            step_loss_val = jnp.mean(
                optax.softmax_cross_entropy(step_value_logits, target_value_support)
            )
            step_loss_pol = jnp.mean(
                optax.softmax_cross_entropy(step_logits, target_policy_smoothed)
            )
            return next_hidden_state, (step_loss_rew, step_loss_val, step_loss_pol)

        hidden_state, (scan_loss_rew, scan_loss_val, scan_loss_pol) = jax.lax.scan(
            _scan_unroll_step,
            hidden_state,
            (
                unroll_actions,
                unroll_target_rewards,
                unroll_target_values,
                unroll_target_policies,
            ),
        )
        loss_rew = jnp.sum(scan_loss_rew)
        loss_val += jnp.sum(scan_loss_val)
        unroll_loss_pol_sum = jnp.sum(scan_loss_pol)

        policy_loss_root_weight = float(
            getattr(self.config, "policy_loss_root_weight", 0.95) or 0.95
        )
        policy_loss_unroll_weight = float(
            getattr(self.config, "policy_loss_unroll_weight", 0.70) or 0.70
        )
        loss_pol_unroll_mean = jnp.where(
            self.config.num_unroll_steps > 0,
            unroll_loss_pol_sum / float(max(1, int(self.config.num_unroll_steps))),
            jnp.array(0.0, dtype=value_logits.dtype),
        )
        loss_pol = (
            (policy_loss_root_weight * root_loss_pol)
            + (policy_loss_unroll_weight * unroll_loss_pol_sum)
        )
        loss_pol_per_head = self._compute_weighted_loss_pol_per_head(
            root_loss_pol,
            unroll_loss_pol_sum,
            self.config.num_unroll_steps,
            policy_loss_root_weight,
            policy_loss_unroll_weight,
        )
        total_loss = loss_val + loss_rew + loss_pol
        clipped_policy = jnp.clip(root_target_policy, 1e-8, 1.0)
        policy_entropy = -jnp.mean(jnp.sum(root_target_policy * jnp.log(clipped_policy), axis=-1))
        policy_top1_share = jnp.mean(jnp.max(root_target_policy, axis=-1))
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
            "loss_pol_per_head": loss_pol_per_head,
            "loss_pol_root": root_loss_pol,
            "loss_pol_unroll_mean": loss_pol_unroll_mean,
            "policy_entropy": policy_entropy,
            "policy_top1_share": policy_top1_share,
            "root_legal_action_count": root_legal_action_count,
            "invalid_root_action_masked_rate": invalid_root_action_masked_rate,
            "priority_errors": priority_errors,
        }

    @property
    def update_fn(self):
        """Expose une etape d'optimisation JITtee."""
        if not hasattr(self, "_update_fn_cached"):
            def step(params, opt_state, batch):
                (loss, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(params, batch)
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, metrics
            self._update_fn_cached = jax.jit(step)
        return self._update_fn_cached

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
