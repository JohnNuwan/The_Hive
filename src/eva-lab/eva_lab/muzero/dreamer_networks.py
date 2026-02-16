"""
DreamerV3 Neural Networks in JAX/Haiku — THE HIVE EVA Lab
Integrates RSSM, Actor-Critic, and Decoders.
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Dict

from eva_lab.muzero.rssm import RSSMCell, Decoder
from eva_lab.muzero.imagination import ActorCritic


class Encoder(hk.Module):
    """Encodeur pour transformer l'observation en vecteur d'embedding."""
    def __init__(self, hidden_dims: list, name: Optional[str] = None):
        super().__init__(name=name)
        self._hidden_dims = hidden_dims

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for h_dim in self._hidden_dims:
            x = hk.Linear(h_dim)(x)
            x = jax.nn.relu(x)
        return x

class DreamerModel(hk.Module):
    """
    World Model complet DreamerV3.
    Contient l'encodeur, le RSSM, et les têtes de décodage (observation, reward, continue).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.network_hidden_dims)
        self.rssm = RSSMCell(
            deterministic_size=config.hidden_state_size * 8, # Plus large pour le World Model
            stochastic_size=32,
            discrete_classes=32
        )
        self.obs_decoder = Decoder(config.observation_shape)
        self.reward_decoder = Decoder((1,))
        # Tête Actor-Critic pour l'imagination
        self.ac = ActorCritic(config.action_space_size)

    def __call__(self, obs: jnp.ndarray, prev_action: jnp.ndarray, prev_state: jnp.ndarray):
        """Passage en mode observation (entraînement sur données réelles)."""
        embed = self.encoder(obs)
        prior, posterior = self.rssm.observe(prev_state, prev_action, embed)
        
        # Reconstruction (via posterior pour l'entraînement)
        rec_obs = self.obs_decoder(posterior)
        pred_reward = self.reward_decoder(posterior)
        
        return prior, posterior, rec_obs, pred_reward

def make_dreamer_networks(config):
    """Factory pour transformer les réseaux Dreamer via un dispatcher unique."""
    
    def dreamer_dispatch(mode: int, *args, **kwargs):
        model = DreamerModel(config)
        
        if mode == 0: # Observe / Training
            obs, prev_action, prev_state = args
            # Force trace other paths for parameter initialization if needed
            # (Though in JAX/Haiku transformation, they are usually found)
            return model(obs, prev_action, prev_state)
            
        elif mode == 1: # Imagine
            prev_state, prev_action = args
            return model.rssm.imagine(prev_state, prev_action)
            
        elif mode == 2: # Initial State
            batch_size = args[0]
            return model.rssm.initial_state(batch_size)
            
        elif mode == 3: # Actor-Critic heads (direct call)
            state = args[0]
            # Trace actor and critic
            actor_logits = model.ac.actor(state)
            value = model.ac.critic(state)
            return actor_logits, value
            
        elif mode == 4: # Decoders only (Reward & Observation)
            state = args[0]
            # Used during imagination to predict rewards
            pred_reward = model.reward_decoder(state)
            # Optional: rec_obs = model.obs_decoder(state)
            return pred_reward
            
        else:
            raise ValueError(f"Unknown Dreamer dispatch mode: {mode}")

    return hk.transform(dreamer_dispatch)
