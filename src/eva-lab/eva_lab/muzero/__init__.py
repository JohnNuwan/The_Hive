"""
MuZero & Dreamer Engine — THE HIVE (eva-lab)
JAX/Haiku Powered.
"""

# Only expose non-torch components by default
from eva_lab.muzero.config import MuZeroConfigV3

# JAX Components (Optional exposure)
# from eva_lab.muzero.jax_agent import JAXMuZeroAgent
# from eva_lab.muzero.rssm import RSSMCell

__all__ = [
    "MuZeroConfigV3",
]
