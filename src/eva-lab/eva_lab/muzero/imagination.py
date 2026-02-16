"""
Imagination Engine — THE HIVE EVA Lab
Handles latent space unrolling for DreamerV3 and MuZero integration.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Callable, Tuple, Any

from eva_lab.muzero.rssm import RSSMCell

class Imagination(hk.Module):
    """
    Moteur d'imagination.
    Permet de projeter des trajectoires futures dans l'espace latent du RSSM.
    """
    def __init__(self, rssm_cell: RSSMCell, horizon: int = 15):
        super().__init__()
        self._rssm = rssm_cell
        self._horizon = horizon

    def unroll(
        self,
        start_state: jnp.ndarray,
        policy_fn: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Génère une trajectoire imaginaire à partir d'un état initial.
        
        Args:
            start_state: L'état de départ (souvent le posterior de l'observation t).
            policy_fn: Une fonction qui retourne une action (ou distribution) à partir d'un état.
            
        Returns:
            Une séquence d'états imaginaires et les actions choisies.
        """
        def body_fn(state, _):
            # 1. Select action from policy
            action = policy_fn(state)
            
            # 2. Predict next state via latent transition (Prior)
            next_state = self._rssm.imagine(state, action)
            
            return next_state, (next_state, action)

        # Unroll for H steps
        _, (states, actions) = jax.lax.scan(
            body_fn,
            start_state,
            jnp.arange(self._horizon)
        )
        
        return states, actions

class ActorCritic(hk.Module):
    """
    Modèles Actor et Critic fonctionnant sur les états du RSSM.
    """
    def __init__(self, action_size: int):
        super().__init__()
        self._action_size = action_size

    def actor(self, state: jnp.ndarray) -> jnp.ndarray:
        """Prédit la distribution d'actions (Policy) à partir de l'état plat."""
        logits = hk.nets.MLP([512, 512, self._action_size], name="actor_mlp")(state)
        return jax.nn.softmax(logits, axis=-1)

    def critic(self, state: jnp.ndarray) -> jnp.ndarray:
        """Prédit la valeur de l'état (Value) à partir de l'état plat."""
        return hk.nets.MLP([512, 512, 1], name="critic_mlp")(state)
