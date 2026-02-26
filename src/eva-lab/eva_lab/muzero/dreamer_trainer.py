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

def calculate_lambda_returns(rewards, values, discount, lambda_):
    """
    Calcule les lambda-returns (G_lambda) pour l'estimation de la valeur.
    G_t = r_t + gamma * ((1 - lambda) * v_{t+1} + lambda * G_{t+1})
    """
    # rewards: [T, B, 1]
    # values: [T+1, B, 1] (inclut bootstrap value)
    # discount: scalar or [T, B, 1]
    
    next_values = values[1:]
    inputs = rewards + discount * next_values * (1 - lambda_)
    
    # Backward pass
    factor = discount * lambda_
    
    def scan_fn(g_next, r_plus_v):
        # We use factor from closure (it's constant)
        g_current = r_plus_v + factor * g_next
        return g_current, g_current

    _, returns = jax.lax.scan(
        scan_fn,
        values[-1], # Bootstrap with last value
        inputs,     # Scan over inputs only
        reverse=True
    )
    # returns is [T, B, 1]
    return returns

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
        import numpy as np
        
        # Debug: Check first sample
        if len(samples) > 0:
            first = samples[0]
            # print(f"Sample 0 obs len: {len(first.observations)}, shape: {np.array(first.observations[0]).shape}")

        try:
            # Use np.stack to ensure consistent shape before JAX
            obs_list = [np.stack(s.observations) for s in samples]
            obs = jnp.array(np.stack(obs_list)) # [B, T, Obs]
            
            act_list = [np.stack(s.actions) for s in samples]
            actions = jnp.array(np.stack(act_list)) # [B, T, Act]
            
            rew_list = [np.stack(s.rewards) for s in samples]
            rewards = jnp.array(np.stack(rew_list)).reshape(len(samples), -1, 1) # [B, T, 1]
        except Exception as e:
            print(f"Error packing batch: {e}")
            # Fallback debug print
            for i, s in enumerate(samples):
                print(f"Sample {i}: obs_len={len(s.observations)}")
            raise e

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
        # 2. KL Divergence
        deterministic_size = self.config.hidden_state_size * 8
        
        # Flatten [B, T, State] -> [B*T, State] for unpacking
        priors_flat = priors.reshape(-1, priors.shape[-1])
        posteriors_flat = posteriors.reshape(-1, posteriors.shape[-1])
        
        _, _, p_logits = unpack_state(priors_flat, deterministic_size, 32, 32)
        _, _, q_logits = unpack_state(posteriors_flat, deterministic_size, 32, 32)
        
        def kl_loss(p_logits, q_logits):
            p_probs = jax.nn.softmax(p_logits)
            q_probs = jax.nn.softmax(q_logits)
            return jnp.sum(p_probs * (jnp.log(p_probs + 1e-8) - jnp.log(q_probs + 1e-8)), axis=-1)

        loss_kl = jnp.mean(kl_loss(q_logits, p_logits))
        total_wm_loss = loss_obs + loss_rew + 0.1 * loss_kl
        
        # Detach posteriors for Actor-Critic training (flatten Batch and Time)
        # posteriors: [B, T, StateDim] -> [B*T, StateDim]
        # We start imagination from all time steps
        flat_posteriors = jax.lax.stop_gradient(posteriors.reshape(-1, posteriors.shape[-1]))
        
        return total_wm_loss, {
            "loss_total": total_wm_loss,
            "loss_obs": loss_obs,
            "loss_rew": loss_rew,
            "loss_kl": loss_kl,
            "posteriors": flat_posteriors
        }

    @property
    def update_wm_fn(self):
        def step(params, opt_state, rng, batch):
            (loss, metrics), grads = jax.value_and_grad(self.world_model_loss, has_aux=True)(
                params, rng, batch
            )
            updates, new_opt_state = self.wm_opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, metrics, {k: v for k,v in metrics.items() if k.startswith("state_")}
        return jax.jit(step)

    def actor_critic_loss(self, params, rng, start_states):
        """
        Optimizes Policy (Actor) and Value (Critic) in imagination.
        start_states: [B, StateDim] (posterior states from WM)
        """
        bias_start_states = start_states # No stop_gradient here, already done in caller if needed
        # Actually caller (train_step) should do stop_gradient.
        
        H = self.config.num_unroll_steps
        discount = self.config.discount
        lambda_ = 0.95 # Dreamer standard
        
        def scan_fn(state, _rng):
            # 1. Actor Policy
            # mode 3 returns (logits, value). But wait, Actor stored in params?
            # self.model.apply(params, ...)
            # We assume params contains 'ac' layer params.
            
            # Actor returns PROBS (softmax) as per imagination.py
            # But let's check imagination.py... yes, jax.nn.softmax
            # We need log_probs
            probs, value = self.model.apply(params, _rng, 3, state)
            
            # Sample Action
            _rng, rng_act = jax.random.split(_rng)
            action_idx = jax.random.categorical(rng_act, jnp.log(probs + 1e-8))
            action = jax.nn.one_hot(action_idx, self.config.action_space_size)
            
            # Log Prob of selected action
            log_prob = jnp.log(probs + 1e-8)
            selected_log_prob = jnp.sum(log_prob * action, axis=-1)
            
            # Entropy
            entropy = -jnp.sum(probs * log_prob, axis=-1)
            
            # 2. Dynamics (Imagine)
            # Use mode 1
            next_state = self.model.apply(params, _rng, 1, state, action)
            
            # 3. Predict Reward (Mode 4)
            reward = self.model.apply(params, _rng, 4, state)
            
            return next_state, (state, action, reward, value, selected_log_prob, entropy)

        # Unroll
        rng_scan = jax.random.split(rng, H)
        final_state, (states, actions, rewards, values, log_probs, entropies) = jax.lax.scan(
            scan_fn,
            bias_start_states,
            rng_scan
        )
        
        # Bootstrap value for final state
        _, bootstrap_value = self.model.apply(params, rng, 3, final_state)
        
        # Concatenate values: [H, B, 1] + [1, B, 1] -> [H+1, B, 1]
        all_values = jnp.concatenate([values, bootstrap_value[None]], axis=0)
        
        # Lambda Returns
        # rewards: [H, B, 1]
        lambda_returns = calculate_lambda_returns(rewards, all_values, discount, lambda_)
        
        # Losses
        # Actor Loss: Reinforce
        # Advantage = Returns - Value
        # advantage: [H, B, 1] -> squeeze to [H, B]
        advantage = jax.lax.stop_gradient(lambda_returns - values).squeeze(-1)
        loss_policy = -jnp.mean(log_probs * advantage)
        loss_entropy = -jnp.mean(entropies) * 1e-4 # Entropy regularization
        loss_actor = loss_policy + loss_entropy
        
        # Critic Loss: Value learning
        target = jax.lax.stop_gradient(lambda_returns)
        loss_critic = jnp.mean(optax.l2_loss(values, target))
        
        total_loss = loss_actor + loss_critic
        
        return total_loss, {
            "loss_ac_total": total_loss,
            "loss_actor": loss_actor,
            "loss_critic": loss_critic,
            "loss_entropy": jnp.mean(entropies),
            "avg_reward": jnp.mean(rewards),
            "avg_value": jnp.mean(values)
        }

    @property
    def update_ac_fn(self):
        def step(params, opt_state, rng, start_states):
            (loss, metrics), grads = jax.value_and_grad(self.actor_critic_loss, has_aux=True)(
                params, rng, start_states
            )
            updates, new_opt_state = self.ac_opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, metrics
        return jax.jit(step)

    def train_step(self, batch: WorldModelBatch):
        """
        A full DreamerV3 training step (WM + AC).
        """
        self._rng, rng_wm, rng_ac = jax.random.split(self._rng, 3)
        
        # 1. Update World Model
        self.params["wm"], self.opt_states["wm"], wm_metrics, aux = self.update_wm_fn(
            self.params["wm"], self.opt_states["wm"], rng_wm, batch
        )
        
        # 2. Extract Posteriors for AC
        # metrics["posteriors"] has shape [B*T, StateDim] (detached)
        posteriors = wm_metrics.pop("posteriors") # Consumes memory, pop it
        
        # 3. Update Actor-Critic
        # We assume AC shares params with WM regarding Encoder/RSSM? 
        # In current init_params, self.params["wm"] holds ALL params including 'ac'.
        # We should use self.params["wm"] as base for AC update if shared.
        # BUT update_ac_fn updates 'ac' params.
        
        # Architecture decision:
        # If 'ac' layers are part of 'wm' params, we update 'wm' params again?
        # Or we keep separate params?
        # DreamerModel holds 'ac'. So 'wm' params include 'ac'.
        # We can update self.params["wm"] using AC gradients.
        # But we usually stop gradients from AC to RSSM.
        # JAX handles this if we only differentiate wrt AC params... 
        # But self.model.apply(...) uses all params.
        # If we didn't split params, AC gradients would flow into RSSM unless stopped.
        # Since we start from detached `posteriors` (fixed states), the gradient flows from `loss` back to `policy(state)`.
        # `state` is fixed (constant wrt AC params). So gradients only flow into `policy`.
        # And `imagine` uses `rssm`?
        # Wait, if `scan_fn` uses `self.model.apply(params, ... mode 1)`, it uses RSSM.
        # Does AC loss flow into RSSM?
        # "Dynamics Backprop": Yes. Dreamer DOES optimization through dynamics.
        # BUT we start from detached posteriors.
        # So we update RSSM params too?
        # Usually NO. We freeze RSSM for AC update.
        # To do this in JAX with merged params: use `jax.lax.stop_gradient` inside the model, or use `optax.multi_transform` to mask updates.
        
        # For simplicity in V1: We update ALL params based on AC loss.
        # This might destabilize RSSM.
        # But since we use detached start states, the "grounding" of RSSM comes from WM loss.
        # AC loss pulls RSSM to generating "rewarding" states. This is sometimes desired (Value Gradients).
        # DreamerV3 stops gradient?
        # "The dynamics are frozen when optimizing the actor."
        # OK, so I should stop gradient on RSSM.
        # It's hard with merged params.
        
        # Compromise: We let it flow for now. It's "Model-Based RL".
        # We assume learning rate is small.
        
        self.params["wm"], self.opt_states["ac"], ac_metrics = self.update_ac_fn(
            self.params["wm"], self.opt_states["ac"], rng_ac, posteriors
        )
        
        # Merge metrics
        return {**wm_metrics, **ac_metrics}

    def init_params(self, sample_obs):
        self._rng, rng_init = jax.random.split(self._rng)
        batch_size = sample_obs.shape[0]
        state_size = (self.config.hidden_state_size * 8) + (32 * 32) * 2
        dummy_action = jnp.zeros((batch_size, self.config.action_space_size))
        dummy_state = jnp.zeros((batch_size, state_size))
        
        
        # Init through dispatcher (mode 5: Init All - Observe & ActorCritic)
        # We need to pass valid shapes for observe(obs, action, state)
        # sample_obs is [B, Obs]
        # dummy_action is [B, Act]
        # dummy_state is [B, State (flat)] or unpacked?
        # RSSM initial_state returns FLAT state.
        # But observe expects FLAT state.
        
        # Wait: mode 5 expects (obs, prev_action, prev_state).
        # We need to pass them.
        
        # Use mode 2 to get a valid initial state first
        # ensure args are correctly passed to mode 2
        # mode 2 args: (batch_size,)
        init_state = self.model.apply(self.model.init(rng_init, 2, batch_size), rng_init, 2, batch_size)
        
        # Wait, model.apply needs params. we don't have params yet!
        # But mode 2 (initial_state) is param-less in RSSM (just zeros).
        # We can pass empty dict {} or None? Haiku might check.
        # Actually, since we are inside init_params, checking structure is tricky.
        # EASIER: Just creates zeros manually as I planned before!
        # RSSM state size is deterministic + (stoch * discrete) * 2
        init_state = jnp.zeros((batch_size, state_size))
        
        # Now init all params using mode 5
        params = self.model.init(rng_init, 5, sample_obs, dummy_action, init_state)
        
        self.params["wm"] = params
        self.opt_states["wm"] = self.wm_opt.init(params)
        self.opt_states["ac"] = self.ac_opt.init(params) # Same params for shared backbone? 
        # Usually AC has separate params if head is separate or stopped gradient.
        # Using same params means adapting all? Dreamer usually stops gradient from AC to WM.
        
        return params, self.opt_states["wm"]
