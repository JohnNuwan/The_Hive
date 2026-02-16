"""
RSSM (Recurrent State Space Model) — THE HIVE EVA Lab
Part of DreamerV3 implementation in JAX/Haiku.

References:
- DreamerV3 (Hafner et al., 2023)
- Mastering Diverse Domains through World Models
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import NamedTuple, Tuple, Optional

# RSSMState is now a single jnp.ndarray (concatenated deterministic, stochastic, and logit)
# Size: deterministic_size + (stochastic_size * discrete_classes) * 2

def pack_state(deterministic, stochastic, logit):
    """Concatène les composants de l'état en un seul vecteur plat."""
    # Ensure they are flat on the last dimension
    s_flat = stochastic.reshape(stochastic.shape[0], -1)
    l_flat = logit.reshape(logit.shape[0], -1)
    return jnp.concatenate([deterministic, s_flat, l_flat], axis=-1)

def unpack_state(state, d_size, s_size, c_classes):
    """Extrait les composants de l'état à partir du vecteur plat."""
    batch_size = state.shape[0]
    stoch_total = s_size * c_classes
    
    h = state[:, :d_size]
    z = state[:, d_size : d_size + stoch_total]
    logits = state[:, d_size + stoch_total :]
    
    # Reshape back to (batch, stoch, classes)
    z = z.reshape(batch_size, s_size, c_classes)
    logits = logits.reshape(batch_size, s_size, c_classes)
    
    return h, z, logits

class RSSMCell(hk.Module):
    """
    Cellule récurrente pour le RSSM (Recurrent State Space Model).
    Gère la transition déterministe et les distributions prior/posterior.
    """
    def __init__(
        self,
        deterministic_size: int = 512,
        stochastic_size: int = 32,
        discrete_classes: int = 32,
        hidden_size: int = 512,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._deterministic_size = deterministic_size
        self._stochastic_size = stochastic_size
        self._discrete_classes = discrete_classes
        self._hidden_size = hidden_size
        
        self._gru = hk.GRU(deterministic_size)
        
    def initial_state(self, batch_size: int) -> jnp.ndarray:
        h = jnp.zeros((batch_size, self._deterministic_size))
        z = jnp.zeros((batch_size, self._stochastic_size, self._discrete_classes))
        logits = jnp.zeros((batch_size, self._stochastic_size, self._discrete_classes))
        return pack_state(h, z, logits)

    def observe(self, prev_state: jnp.ndarray, prev_action: jnp.ndarray, embed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calcul du Posterior (z_t | h_t, x_t).
        Utilisé durant l'entraînement avec des observations réelles.
        """
        h_prev, z_prev, _ = unpack_state(prev_state, self._deterministic_size, self._stochastic_size, self._discrete_classes)
        
        # 2. Update deterministic state: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
        z_prev_flat = z_prev.reshape(z_prev.shape[0], -1)
        gru_input = jnp.concatenate([z_prev_flat, prev_action], axis=-1)
        h_t, _ = self._gru(gru_input, h_prev)
        
        # 3. Prior distribution: p(z_t | h_t)
        prior_logits = self._get_logits(h_t, name="prior")
        prior_stoch = self._sample_stochastic(prior_logits)
        
        # 4. Posterior distribution: q(z_t | h_t, e_t)
        posterior_input = jnp.concatenate([h_t, embed], axis=-1)
        posterior_logits = self._get_logits(posterior_input, name="posterior")
        posterior_stoch = self._sample_stochastic(posterior_logits)
        
        return pack_state(h_t, prior_stoch, prior_logits), pack_state(h_t, posterior_stoch, posterior_logits)

    def imagine(self, prev_state: jnp.ndarray, prev_action: jnp.ndarray) -> jnp.ndarray:
        """
        Calcul du Prior (z_t | h_t).
        Utilisé pour "rêver" sans observations réelles.
        """
        # 1. Unpack
        h_prev, z_prev, _ = unpack_state(prev_state, self._deterministic_size, self._stochastic_size, self._discrete_classes)
        
        # 2. Update deterministic state
        z_prev_flat = z_prev.reshape(z_prev.shape[0], -1)
        gru_input = jnp.concatenate([z_prev_flat, prev_action], axis=-1)
        h_t, _ = self._gru(gru_input, h_prev)
        
        # 3. Predict prior stochastic state
        logits = self._get_logits(h_t, name="prior")
        stoch = self._sample_stochastic(logits)
        
        return pack_state(h_t, stoch, logits)

    def _get_logits(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """MLP pour prédire les logits de la distribution stochastique."""
        x = hk.nets.MLP(
            [self._hidden_size, self._stochastic_size * self._discrete_classes],
            name=f"logits_{name}"
        )(x)
        # Reshape to (batch, stochastic_size, discrete_classes)
        return x.reshape(x.shape[0], self._stochastic_size, self._discrete_classes)

    def _sample_stochastic(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Échantillonnage via Straight-Through Gradient ou Reparameterization."""
        # Pour DreamerV3 discrete states, on utilise souvent soft-categorical + ST gradient
        # Simples probabilités ici
        probs = jax.nn.softmax(logits, axis=-1)
        # Random sample (discrète)
        # Note: Durant l'entraînement, on utilise jax.random.categorical ou la distrax Categorical
        # Pour simplifier dans la cellule, on retourne les probs ou un sample
        return probs

class Decoder(hk.Module):
    """Décodeur pour reconstruire l'observation (ou prédire reward/continue)."""
    def __init__(self, output_shape: tuple, name: Optional[str] = None):
        super().__init__(name=name)
        self._output_shape = output_shape

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        # State is flat, MLP can take it directly or we can unpack if we need specific components
        # For simplicity, MLP on the whole flat state is often better in Dreamer
        output_dim = 1
        for s in self._output_shape:
            output_dim *= s
            
        return hk.nets.MLP(
            [512, 512, output_dim],
            name="decoder_mlp"
        )(state).reshape(-1, *self._output_shape)
