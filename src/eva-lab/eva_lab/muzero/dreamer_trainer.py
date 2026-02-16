"""
DreamerV3 Trainer in JAX/Optax — THE HIVE EVA Lab
Implements World Model training and Actor-Critic optimization in imagination.
"""

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import logging
from typing import NamedTuple, Dict, Tuple

from eva_lab.muzero.rssm import unpack_state

logger = logging.getLogger(__name__)

class WorldModelBatch(NamedTuple):
    observations: jnp.ndarray  # [B, T, Obs]
    actions: jnp.ndarray       # [B, T, Action]
    rewards: jnp.ndarray       # [B, T, 1]
    is_first: jnp.ndarray      # [B, T, 1]

class DreamerTrainerJAX:
    def __init__(self, config, transformed_model):
        self.config = config
        self.model = transformed_model
        
        # Optimizers
        self.wm_opt = optax.adam(config.learning_rate)
        self.ac_opt = optax.adam(config.learning_rate)
        
        self.params = {}
        self.opt_states = {}
        self._rng = jax.random.PRNGKey(42)

    def prepare_batch(self, samples) -> WorldModelBatch:
        """
        Transforme une liste de GameHistory en WorldModelBatch JAX.
        """
        # Pour Dreamer, on a besoin de séquences de longueur T (ex: 50)
        # On suppose que samples est une liste de segments [T, Obs/Act/Rew]
        obs = jnp.array([s.observations for s in samples]) # [B, T, Obs]
        actions = jnp.array([s.actions for s in samples]) # [B, T, Act]
        rewards = jnp.array([s.rewards for s in samples]).reshape(len(samples), -1, 1) # [B, T, 1]
        
        # is_first: 1.0 au début de chaque séquence
        is_first = jnp.zeros_like(rewards)
        is_first = is_first.at[:, 0, 0].set(1.0)
        
        return WorldModelBatch(obs, actions, rewards, is_first)

    def world_model_loss(self, wm_params, rng, batch: WorldModelBatch):
        """
        Perte du World Model : Reconstruction + KL Divergence.
        """
        batch_size = batch.observations.shape[0]
        # mode 2 = init_state. Signature: (params, rng, mode, batch_size)
        state = self.model.apply(wm_params, rng, 2, batch_size)
        
        # Transpose to [T, B, ...] for lax.scan
        obs_t = jnp.transpose(batch.observations, (1, 0, 2))
        act_t = jnp.transpose(batch.actions, (1, 0, 2))

        def scan_fn(prev_state, inputs):
            obs, action = inputs
            # mode 0 = observe. Signature: (params, rng, mode, obs, action, state)
            prior, posterior, rec_obs, pred_rew = self.model.apply(wm_params, rng, 0, obs, action, prev_state)
            return posterior, (prior, posterior, rec_obs, pred_rew)

        _, (priors, posteriors, rec_obss, pred_rews) = jax.lax.scan(
            scan_fn, state, (obs_t, act_t)
        )
        
        # Transpose back to [B, T, ...]
        rec_obss = jnp.transpose(rec_obss, (1, 0, 2))
        pred_rews = jnp.transpose(pred_rews, (1, 0, 2))
        priors = jnp.transpose(priors, (1, 0, 2))
        posteriors = jnp.transpose(posteriors, (1, 0, 2))

        # 1. Reconstruction Losses
        loss_obs = jnp.mean(optax.l2_loss(rec_obss, batch.observations))
        loss_rew = jnp.mean(optax.l2_loss(pred_rews, batch.rewards))
        
        # 2. KL Divergence
        deterministic_size = self.config.hidden_state_size * 8
        _, _, p_logits = unpack_state(priors, deterministic_size, 32, 32)
        _, _, q_logits = unpack_state(posteriors, deterministic_size, 32, 32)
        
        def kl_loss(p_logits, q_logits):
            p_probs = jax.nn.softmax(p_logits)
            q_probs = jax.nn.softmax(q_logits)
            return jnp.sum(p_probs * (jnp.log(p_probs + 1e-8) - jnp.log(q_probs + 1e-8)), axis=-1)

        loss_kl = jnp.mean(kl_loss(q_logits, p_logits))
        total_wm_loss = loss_obs + loss_rew + 0.1 * loss_kl
        
        return total_wm_loss, {
            "loss_total": total_wm_loss,
            "loss_obs": loss_obs,
            "loss_rew": loss_rew,
            "loss_kl": loss_kl
        }

    @property
    def update_wm_fn(self):
        def step(params, opt_state, rng, batch):
            (loss, metrics), grads = jax.value_and_grad(self.world_model_loss, has_aux=True)(
                params, rng, batch
            )
            updates, new_opt_state = self.wm_opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, metrics
        return jax.jit(step)

    def train_step(self, batch: WorldModelBatch):
        """
        A full DreamerV3 training step.
        """
        self._rng, rng_wm = jax.random.split(self._rng)
        
        # 1. Update World Model
        self.params["wm"], self.opt_states["wm"], wm_metrics = self.update_wm_fn(
            self.params["wm"], self.opt_states["wm"], rng_wm, batch
        )
        
        return wm_metrics

    def init_params(self, sample_obs):
        self._rng, rng_init = jax.random.split(self._rng)
        batch_size = sample_obs.shape[0]
        dummy_action = jnp.zeros((batch_size, self.config.action_space_size))
        dummy_state = jnp.zeros((batch_size, 2560))
        
        # Init through dispatcher (mode 0)
        params = self.model.init(rng_init, 0, sample_obs, dummy_action, dummy_state)
        
        self.params["wm"] = params
        self.opt_states["wm"] = self.wm_opt.init(params)
        
        return params, self.opt_states["wm"]
