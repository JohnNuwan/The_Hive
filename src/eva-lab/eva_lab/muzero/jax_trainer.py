"""
MuZero Trainer in JAX/Optax — THE HIVE EVA Lab

Features:
  - JIT-compiled loss & update functions
  - Optax-based Adam optimizer
  - BPTT unrolling for num_unroll_steps
"""

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import logging
import numpy as np
from typing import NamedTuple, List

logger = logging.getLogger(__name__)

class TrainingBatch(NamedTuple):
    observations: jnp.ndarray   # [B, Obs]
    actions: jnp.ndarray        # [B, K]
    target_values: jnp.ndarray  # [B, K+1]
    target_rewards: jnp.ndarray # [B, K]
    target_policies: jnp.ndarray# [B, K+1, ActionDim]

class MuZeroTrainerJAX:
    def __init__(self, config, transformed_nets):
        self.config = config
        self.transformed = transformed_nets

        # Optimizer: Adam with weight decay
        self.optimizer = optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self._rng = jax.random.PRNGKey(42)

    def init_params(self, sample_obs):
        self._rng, init_key = jax.random.split(self._rng)
        params = self.transformed.init(init_key, sample_obs)
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def loss_fn(self, params, batch: TrainingBatch):
        """MuZero Hybrid Loss Function."""
        
        # 1. Initial Step (Representation + Prediction)
        h, logits, v = self.transformed.apply(params, None, batch.observations, method=0)
        
        loss_val = jnp.mean(optax.l2_loss(v.squeeze(-1), batch.target_values[:, 0]))
        loss_pol = jnp.mean(optax.softmax_cross_entropy(logits, batch.target_policies[:, 0]))
        loss_rew = 0.0
        
        hidden_state = h
        num_unroll = self.config.num_unroll_steps
        
        # 2. Recurrent Steps (Dynamics + Prediction)
        for k in range(num_unroll):
            # Convert scalar action to one-hot for JAX
            action_onehot = jax.nn.one_hot(batch.actions[:, k], self.config.action_space_size)
            
            # Recurrent inference
            next_h, r, logits, v = self.transformed.apply(params, None, hidden_state, action_onehot, method=1)
            
            # Scale gradients by 0.5 for dynamics stability (similar to PyTorch register_hook)
            # In JAX we can use jax.lax.stop_gradient or scaling
            # Standard MuZero: scale loss or use scale_gradient
            # hidden_state = (next_h + 0.5 * (next_h - jax.lax.stop_gradient(next_h))) # Approximate
            hidden_state = next_h
            
            loss_rew += jnp.mean(optax.l2_loss(r.squeeze(-1), batch.target_rewards[:, k]))
            loss_val += jnp.mean(optax.l2_loss(v.squeeze(-1), batch.target_values[:, k+1]))
            loss_pol += jnp.mean(optax.softmax_cross_entropy(logits, batch.target_policies[:, k+1]))

        total_loss = loss_val + loss_rew + loss_pol
        return total_loss, {
            "loss_total": total_loss,
            "loss_val": loss_val,
            "loss_rew": loss_rew,
            "loss_pol": loss_pol
        }

    @property
    def update_fn(self):
        """JIT-compiled update step."""
        def step(params, opt_state, batch):
            (loss, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(params, batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, metrics
        
        return jax.jit(step)

    def prepare_batch(self, batch_list: List[tuple]) -> TrainingBatch:
        """Convert list of (GameHistory, start_idx) to JAX TrainingBatch."""
        obs_list = []
        action_list = []
        target_v_list = []
        target_r_list = []
        target_p_list = []
        
        num_unroll = self.config.num_unroll_steps
        
        for game, start_idx in batch_list:
            obs_list.append(game.observations[start_idx])
            
            actions = []
            vals = []
            rews = []
            pols = []
            
            for k in range(num_unroll + 1):
                idx = start_idx + k
                if idx < len(game):
                    vals.append(game.values[idx])
                    pols.append(game.policies[idx])
                    if k < num_unroll:
                        actions.append(game.actions[idx])
                        rews.append(game.rewards[idx])
                else:
                    vals.append(0.0)
                    pols.append(np.ones(self.config.action_space_size) / self.config.action_space_size)
                    if k < num_unroll:
                        actions.append(0)
                        rews.append(0.0)
            
            action_list.append(actions)
            target_v_list.append(vals)
            target_r_list.append(rews)
            target_p_list.append(pols)
            
        return TrainingBatch(
            observations=jnp.array(obs_list),
            actions=jnp.array(action_list),
            target_values=jnp.array(target_v_list),
            target_rewards=jnp.array(target_r_list),
            target_policies=jnp.array(target_p_list)
        )
