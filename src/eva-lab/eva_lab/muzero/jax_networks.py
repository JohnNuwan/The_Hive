"""Reseaux MuZero JAX/Haiku avec representation, dynamique et prediction partagees."""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


def support_to_scalar(logits: jnp.ndarray, support_size: int) -> jnp.ndarray:
    """Transforme les logits categories en valeur scalaire par esperance (valeur * proba)."""
    probabilities = jax.nn.softmax(logits, axis=-1)
    support = jnp.arange(-support_size, support_size + 1, dtype=logits.dtype)
    return jnp.sum(probabilities * support, axis=-1, keepdims=True)


def scalar_to_support(scalar: jnp.ndarray, support_size: int) -> jnp.ndarray:
    """Transforme des cibles scalaires en distribution sur le support discret.

    Args:
        scalar (jnp.ndarray): Valeur scalaire par echantillon. Le tenseur peut
            etre fourni sous la forme ``(batch,)`` ou ``(batch, 1)``.
        support_size (int): Taille maximale du support discret positif.

    Returns:
        jnp.ndarray: Distribution cible sur ``2 * support_size + 1`` classes.

    Raises:
        ValueError: Si la derniere dimension n'est pas scalaire.
    """
    scalar = jnp.asarray(scalar)
    if scalar.ndim > 1 and scalar.shape[-1] != 1:
        raise ValueError(
            "Les cibles MuZero doivent etre de forme (batch,) ou (batch, 1)."
        )
    if scalar.ndim > 0 and scalar.shape[-1] == 1:
        scalar = scalar[..., 0]
    scalar = jnp.clip(scalar, -support_size, support_size)
    b = scalar + support_size
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.ceil(b).astype(jnp.int32)
    fraction_upper = b - lower
    fraction_lower = 1.0 - fraction_upper
    num_bins = 2 * support_size + 1
    lower_hot = jax.nn.one_hot(lower, num_bins) * jnp.expand_dims(fraction_lower, -1)
    upper_hot = jax.nn.one_hot(upper, num_bins) * jnp.expand_dims(fraction_upper, -1)
    return lower_hot + upper_hot


class RepresentationNetwork(hk.Module):
    """Encode l'observation brute dans un etat latent compact (optionnellement via JEPA)."""

    def __init__(
        self,
        output_dim: int,
        hidden_dims: list[int],
        use_jepa_encoder: bool = False,
        jepa_latent_size: int = 128,
    ):
        """Mémorise la taille de sortie, les couches cachées et les paramètres JEPA."""
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_jepa_encoder = use_jepa_encoder
        self.jepa_latent_size = jepa_latent_size

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Projette l'observation (éventuellement compressée par JEPA) dans l'espace latent MuZero."""
        if self.use_jepa_encoder:
            from eva_lab.muzero.jepa_encoder import JEPAEncoder
            # Instanciation de l'encodeur de contexte JEPA pré-entraîné
            jepa_encoder = JEPAEncoder(
                self.jepa_latent_size,
                [256, 256],
                name="context_encoder",
            )
            hidden = jepa_encoder(observation)
        else:
            hidden = observation

        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)
        hidden = hk.Linear(self.output_dim)(hidden)
        return jnp.tanh(hidden)


class DynamicsNetwork(hk.Module):
    """Prevoit le prochain etat latent et la recompense immediate."""

    def __init__(self, state_dim: int, hidden_dims: list[int], support_size: int):
        """Memorise la taille de l'etat latent, les couches cachees et le support."""
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.support_size = support_size

    def __call__(self, hidden_state: jnp.ndarray, action_onehot: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Propage un etat latent avec une action one-hot."""
        hidden = jnp.concatenate([hidden_state, action_onehot], axis=-1)
        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)

        next_state = hk.Linear(self.state_dim)(hidden)
        reward_logits = hk.Linear(2 * self.support_size + 1)(hidden)
        return jnp.tanh(next_state), reward_logits


class PredictionNetwork(hk.Module):
    """Prevoit la politique et la valeur a partir d'un etat latent."""

    def __init__(self, action_dim: int, hidden_dims: list[int], support_size: int):
        """Memorise la taille d'action, les couches cachees et le support."""
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.support_size = support_size

    def __call__(self, hidden_state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Produit les logits de politique et la valeur scalaire."""
        hidden = hidden_state
        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)

        logits = hk.Linear(self.action_dim)(hidden)
        value_logits = hk.Linear(2 * self.support_size + 1)(hidden)
        return logits, value_logits


def make_muzero_networks(config):
    """Construit les deux points d'entree Haiku partages de MuZero."""

    def network_factory():
        representation = RepresentationNetwork(
            config.hidden_state_size,
            config.network_hidden_dims,
            use_jepa_encoder=getattr(config, "use_jepa_encoder", False),
            jepa_latent_size=getattr(config, "jepa_latent_size", 128),
        )
        dynamics = DynamicsNetwork(
            config.hidden_state_size,
            config.network_hidden_dims,
            config.support_size,
        )
        prediction = PredictionNetwork(
            config.action_space_size,
            config.network_hidden_dims,
            config.support_size,
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
