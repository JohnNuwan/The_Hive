"""Réseaux Market-JEPA et calcul de perte VICReg en JAX/Haiku."""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


class JEPAEncoder(hk.Module):
    """Encode une observation de marché (et optionnellement son contexte GNN) dans l'espace latent JEPA."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], name: str | None = None):
        """Initialise la dimension latente et les couches cachées."""
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

    def __call__(self, observation: jnp.ndarray, gnn_embedding: jnp.ndarray | None = None) -> jnp.ndarray:
        """Projette l'observation (et son contexte GNN) dans l'espace latent stable JEPA."""
        if gnn_embedding is not None:
            hidden = jnp.concatenate([observation, gnn_embedding], axis=-1)
        else:
            hidden = observation

        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)

        latent = hk.Linear(self.latent_dim)(hidden)
        # Normalisation tanh pour stabiliser le rayon de l'espace latent
        return jnp.tanh(latent)


class JEPAPredictor(hk.Module):
    """Prédit la représentation latente future à partir de la représentation courante."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], name: str | None = None):
        """Initialise la dimension latente et les couches de prédiction cachées."""
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

    def __call__(self, latent_state: jnp.ndarray, context_regime: jnp.ndarray | None = None) -> jnp.ndarray:
        """Prédit le prochain état latent, potentiellement conditionné par un régime macro."""
        if context_regime is not None:
            hidden = jnp.concatenate([latent_state, context_regime], axis=-1)
        else:
            hidden = latent_state

        for hidden_dim in self.hidden_dims:
            hidden = hk.Linear(hidden_dim)(hidden)
            hidden = jax.nn.relu(hidden)

        predicted_latent = hk.Linear(self.latent_dim)(hidden)
        return jnp.tanh(predicted_latent)


def make_jepa_networks(latent_dim: int, hidden_dims: list[int]):
    """Construit les transformations Haiku pour les encodeurs et le prédicteur JEPA."""

    def forward(observation: jnp.ndarray, future_observation: jnp.ndarray, gnn_embedding: jnp.ndarray | None = None):
        # Encodeur de contexte
        context_encoder = JEPAEncoder(latent_dim, hidden_dims, name="context_encoder")
        # Encodeur cible (partage la même architecture, mais peut être mis à jour séparément ou par EMA)
        target_encoder = JEPAEncoder(latent_dim, hidden_dims, name="target_encoder")
        # Prédicteur
        predictor = JEPAPredictor(latent_dim, hidden_dims, name="predictor")

        z_x = context_encoder(observation, gnn_embedding)
        z_y = target_encoder(future_observation, gnn_embedding)
        z_y_hat = predictor(z_x)
        return z_x, z_y, z_y_hat

    return hk.transform(forward)


def compute_vicreg_loss(
    z_x: jnp.ndarray,
    z_y: jnp.ndarray,
    z_y_hat: jnp.ndarray,
    *,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Calcule la perte VICReg (Variance, Invariance, Covariance) pour empêcher l'effondrement.

    Args:
        z_x (jnp.ndarray): Représentation latente du présent (batch x latent_dim).
        z_y (jnp.ndarray): Représentation latente cible du futur (batch x latent_dim).
        z_y_hat (jnp.ndarray): Représentation prédite du futur (batch x latent_dim).
        sim_weight (float): Poids du terme d'invariance (similitude).
        var_weight (float): Poids du terme de variance.
        cov_weight (float): Poids du terme de covariance.
        gamma (float): Écart-type cible pour la régularisation de la variance (défaut: 1.0).
        epsilon (float): Terme de stabilité pour la racine carrée de la variance.

    Returns:
        tuple[jnp.ndarray, dict[str, jnp.ndarray]]: Perte globale et dictionnaire des sous-pertes.
    """
    batch_size, latent_dim = z_y.shape

    # 1. Terme d'Invariance (MSE entre la prédiction et la cible)
    loss_invariance = jnp.mean(jnp.square(z_y - z_y_hat))

    # 2. Terme de Variance (empêche l'effondrement sur un point constant)
    # Calcul de la variance empirique de chaque dimension sur le lot
    std_z_x = jnp.sqrt(jnp.var(z_x, axis=0) + epsilon)
    std_z_y = jnp.sqrt(jnp.var(z_y, axis=0) + epsilon)
    
    loss_var_z_x = jnp.mean(jax.nn.relu(gamma - std_z_x))
    loss_var_z_y = jnp.mean(jax.nn.relu(gamma - std_z_y))
    loss_variance = 0.5 * (loss_var_z_x + loss_var_z_y)

    # 3. Terme de Covariance (empêche l'effondrement dimensionnel et limite la redondance)
    # Centrage des représentations
    z_x_centered = z_x - jnp.mean(z_x, axis=0, keepdims=True)
    z_y_centered = z_y - jnp.mean(z_y, axis=0, keepdims=True)

    # Matrices de covariance
    cov_z_x = (z_x_centered.T @ z_x_centered) / (batch_size - 1)
    cov_z_y = (z_y_centered.T @ z_y_centered) / (batch_size - 1)

    # Masquage de la diagonale pour ne pénaliser que les covariances croisées
    mask = jnp.ones((latent_dim, latent_dim)) - jnp.eye(latent_dim)
    loss_cov_z_x = jnp.sum(jnp.square(cov_z_x * mask)) / latent_dim
    loss_cov_z_y = jnp.sum(jnp.square(cov_z_y * mask)) / latent_dim
    loss_covariance = 0.5 * (loss_cov_z_x + loss_cov_z_y)

    # Perte globale pondérée
    total_loss = (
        sim_weight * loss_invariance
        + var_weight * loss_variance
        + cov_weight * loss_covariance
    )

    metrics = {
        "loss_total": total_loss,
        "loss_invariance": loss_invariance,
        "loss_variance": loss_variance,
        "loss_covariance": loss_covariance,
        "mean_std_z_x": jnp.mean(std_z_x),
        "mean_std_z_y": jnp.mean(std_z_y),
    }

    return total_loss, metrics
