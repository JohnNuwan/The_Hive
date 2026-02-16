"""
MuZero Neural Networks in JAX/Haiku — THE HIVE EVA Lab

Architecture:
  - Representation: Observation → Hidden State
  - Dynamics:       Hidden State + Action → Next State + Reward
  - Prediction:     Hidden State → Policy (logits) + Value
"""

import haiku as hk
import jax
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)

class RepresentationNetwork(hk.Module):
    """Encode raw observation into compact latent space."""
    def __init__(self, output_dim: int, hidden_dims: list):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for h_dim in self.hidden_dims:
            x = hk.Linear(h_dim)(x)
            x = jax.nn.relu(x)
        x = hk.Linear(self.output_dim)(x)
        # Normalize to [-1, 1] for stability in recurrent dynamics
        return jnp.tanh(x)

class DynamicsNetwork(hk.Module):
    """Predict next hidden state and reward from (state, action)."""
    def __init__(self, state_dim: int, hidden_dims: list):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims

    def __call__(self, hidden_state: jnp.ndarray, action_onehot: jnp.ndarray) -> tuple:
        x = jnp.concatenate([hidden_state, action_onehot], axis=-1)
        for h_dim in self.hidden_dims:
            x = hk.Linear(h_dim)(x)
            x = jax.nn.relu(x)
        
        # Next state head
        next_state = hk.Linear(self.state_dim)(x)
        next_state = jnp.tanh(next_state)
        
        # Reward head
        reward = hk.Linear(1)(x)
        
        return next_state, reward

class PredictionNetwork(hk.Module):
    """Predict policy distribution (logits) and value from hidden state."""
    def __init__(self, action_dim: int, hidden_dims: list):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

    def __call__(self, hidden_state: jnp.ndarray) -> tuple:
        x = hidden_state
        for h_dim in self.hidden_dims:
            x = hk.Linear(h_dim)(x)
            x = jax.nn.relu(x)
            
        # Policy head (logits)
        logits = hk.Linear(self.action_dim)(x)
        
        # Value head
        value = hk.Linear(1)(x)
        
        return logits, value

class MuZeroNet(hk.MultiTransformed):
    """Facade for the three MuZero sub-networks."""
    def __init__(self, config):
        def initial_inference(observation):
            rep = RepresentationNetwork(config.hidden_state_size, config.network_hidden_dims)
            pred = PredictionNetwork(config.action_space_size, config.network_hidden_dims)
            h = rep(observation)
            logits, v = pred(h)
            return h, logits, v

        def recurrent_inference(hidden_state, action_onehot):
            dyn = DynamicsNetwork(config.hidden_state_size, config.network_hidden_dims)
            pred = PredictionNetwork(config.action_space_size, config.network_hidden_dims)
            next_h, r = dyn(hidden_state, action_onehot)
            logits, v = pred(next_h)
            return next_h, r, logits, v

        super().__init__(
            initial_inference=initial_inference,
            recurrent_inference=recurrent_inference
        )

def make_muzero_networks(config):
    """Factory to create and transform MuZero networks."""
    
    def initial_inference_fn(observation, *args, **kwargs):
        rep = RepresentationNetwork(config.hidden_state_size, config.network_hidden_dims)
        pred = PredictionNetwork(config.action_space_size, config.network_hidden_dims)
        h = rep(observation)
        logits, v = pred(h)
        return h, logits, v

    def recurrent_inference_fn(hidden_state, action_onehot, *args, **kwargs):
        dyn = DynamicsNetwork(config.hidden_state_size, config.network_hidden_dims)
        pred = PredictionNetwork(config.action_space_size, config.network_hidden_dims)
        next_h, r = dyn(hidden_state, action_onehot)
        logits, v = pred(next_h)
        return next_h, r, logits, v

    # We use hk.multi_transform to handle multiple entry points with shared params
    transformed = hk.multi_transform(
        lambda: (initial_inference_fn, recurrent_inference_fn)
    )
    
    return transformed
