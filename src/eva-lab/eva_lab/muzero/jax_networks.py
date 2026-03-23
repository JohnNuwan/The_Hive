"""Reseaux MuZero JAX/Haiku avec representation, dynamique et prediction partagees."""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


class RepresentationNetwork(hk.Module):
    """Encode l'observation brute dans un etat latent compact."""

    def __init__(self, output_dim: int, hidden_dims: list[int]):
        """Memorise la taille de sortie et les couches cachees."""
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Projette l'observation dans l'espace latent MuZero."""
        hidden = observation
        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)
        hidden = hk.Linear(self.output_dim)(hidden)
        return jnp.tanh(hidden)


class DynamicsNetwork(hk.Module):
    """Prevoit le prochain etat latent et la recompense immediate."""

    def __init__(self, state_dim: int, hidden_dims: list[int]):
        """Memorise la taille de l'etat latent et les couches cachees."""
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims

    def __call__(self, hidden_state: jnp.ndarray, action_onehot: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Propage un etat latent avec une action one-hot."""
        hidden = jnp.concatenate([hidden_state, action_onehot], axis=-1)
        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)

        next_state = hk.Linear(self.state_dim)(hidden)
        reward = hk.Linear(1)(hidden)
        return jnp.tanh(next_state), reward


class PredictionNetwork(hk.Module):
    """Prevoit la politique et la valeur a partir d'un etat latent."""

    def __init__(self, action_dim: int, hidden_dims: list[int]):
        """Memorise la taille d'action et les couches cachees."""
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

    def __call__(self, hidden_state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Produit les logits de politique et la valeur scalaire."""
        hidden = hidden_state
        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)

        logits = hk.Linear(self.action_dim)(hidden)
        value = hk.Linear(1)(hidden)
        return logits, value


def make_muzero_networks(config):
    """Construit les deux points d'entree Haiku partages de MuZero."""

    def network_factory():
        representation = RepresentationNetwork(
            config.hidden_state_size,
            config.network_hidden_dims,
        )
        dynamics = DynamicsNetwork(
            config.hidden_state_size,
            config.network_hidden_dims,
        )
        prediction = PredictionNetwork(
            config.action_space_size,
            config.network_hidden_dims,
        )

        def initial_inference(observation: jnp.ndarray):
            hidden_state = representation(observation)
            logits, value = prediction(hidden_state)
            return hidden_state, logits, value

        def recurrent_inference(hidden_state: jnp.ndarray, action_onehot: jnp.ndarray):
            next_state, reward = dynamics(hidden_state, action_onehot)
            logits, value = prediction(next_state)
            return next_state, reward, logits, value

        def init_all(observation: jnp.ndarray):
            hidden_state, logits, value = initial_inference(observation)
            dummy_action = jnp.zeros((observation.shape[0], config.action_space_size), dtype=observation.dtype)
            recurrent_inference(hidden_state, dummy_action)
            return hidden_state, logits, value

        return init_all, (initial_inference, recurrent_inference)

    return hk.multi_transform(network_factory)
