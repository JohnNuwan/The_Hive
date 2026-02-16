"""
Validation script for DreamerV3 Components — THE HIVE EVA Lab
Checks structural integrity of RSSM and Imagination unroll.
"""

import os
import sys
import jax
import jax.numpy as jnp
import haiku as hk

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src", "eva-lab"))

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.muzero.rssm import RSSMCell, pack_state
from eva_lab.muzero.dreamer_networks import DreamerModel, make_dreamer_networks
from eva_lab.muzero.imagination import Imagination

def test_rssm_structural():
    print("Testing RSSM Structural Integrity...")
    config = MuZeroConfigV3()
    batch_size = 4
    
    # Init JAX Agent (to verify its init)
    agent = JAXMuZeroAgent(config)
    print("  JAX MuZero Agent initialized.")
    
    # Initialize Dreamer networks
    transformed = make_dreamer_networks(config)
    rng = jax.random.PRNGKey(42)
    
    # Dummy data
    obs = jnp.zeros((batch_size, *config.observation_shape), dtype=jnp.float32)
    action = jnp.zeros((batch_size, config.action_space_size), dtype=jnp.float32)
    
    # Initialize params with mode 0 (observe)
    params = transformed.init(rng, 0, obs, action, jnp.zeros((batch_size, 2560)))
    print("  Dreamer Model parameters initialized successfully.")

    # Initialize state through mode 2
    state = transformed.apply(params, rng, 2, batch_size)

    # Run observer step (mode 0)
    prior, posterior, rec_obs, pred_rew = transformed.apply(params, rng, 0, obs, action, state)
    print(f"  Observer run successful. Posterior flat state shape: {posterior.shape}")
    print(f"  Reconstruction shape: {rec_obs.shape}")
    
    # Run imagination step (mode 1)
    imagined_state = transformed.apply(params, rng, 1, posterior, action)
    print(f"  Imagination step successful. Imagine flat state shape: {imagined_state.shape}")
    
    print("  RSSM Structural check: PASSED")

if __name__ == "__main__":
    try:
        test_rssm_structural()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {e}")
