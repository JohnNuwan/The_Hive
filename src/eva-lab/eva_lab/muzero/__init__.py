"""
MuZero Trading Engine — THE HIVE (eva-lab)

Ported from Muzero_Pro_Trader V3.1 "Hunger Mode" repository.
Provides:
  - MuZeroNet: 3-network architecture (Representation + Dynamics + Prediction)
  - MuZeroMCTS: Monte Carlo Tree Search with UCB + MinMax normalization
  - MuZeroAgent: Self-play loop with replay buffer
  - MuZeroConfigV3: Hunger Mode reward shaping (142 features, 5 actions)
  - TradingEnvironment: Commission-aware trading env (SLBE, pyramiding)
"""

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.networks import MuZeroNet
from eva_lab.muzero.mcts import MuZeroMCTS
from eva_lab.muzero.agent import MuZeroAgent

__all__ = [
    "MuZeroConfigV3",
    "MuZeroNet",
    "MuZeroMCTS",
    "MuZeroAgent",
]
