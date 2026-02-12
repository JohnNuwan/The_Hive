"""
Multi-Armed Bandit (MAB) Solver for Dynamic Difficulty Adjustment (DDA).
Uses Thompson Sampling to balance Exploration (learning new strategies) and Exploitation (using best strategy).
"""

import numpy as np
from enum import Enum
from pydantic import BaseModel

class StrategyType(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class BanditArm(BaseModel):
    name: StrategyType
    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0   # Failures + 1
    locked: bool = False
    unlock_threshold_sp: int = 0

class MABState(BaseModel):
    arms: dict[str, BanditArm]
    current_arm: str = StrategyType.BALANCED.value
    total_pulls: int = 0

class MABSolver:
    """
    Thompson Sampling MAB Solver.
    Adjusts difficulty based on performance (Profit/Loss).
    """

    def __init__(self, state: MABState | None = None):
        if state:
            self.state = state
        else:
            self.state = MABState(arms={
                StrategyType.CONSERVATIVE.value: BanditArm(
                    name=StrategyType.CONSERVATIVE,
                    alpha=2.0, beta=1.0 # Bias slightly towards conservative initially
                ),
                StrategyType.BALANCED.value: BanditArm(
                    name=StrategyType.BALANCED,
                    alpha=1.0, beta=1.0
                ),
                StrategyType.AGGRESSIVE.value: BanditArm(
                    name=StrategyType.AGGRESSIVE,
                    locked=True,
                    unlock_threshold_sp=5000,
                    alpha=1.0, beta=1.0
                )
            })

    def select_arm(self, current_sp: float) -> str:
        """
        Selects the best strategy using Thompson Sampling.
        Only considers unlocked arms.
        """
        best_arm = None
        max_sample = -1.0

        for arm_name, arm in self.state.arms.items():
            # Check unlock condition
            if arm.locked:
                if current_sp >= arm.unlock_threshold_sp:
                    arm.locked = False
                else:
                    continue

            # Thompson Sampling: Sample from Beta(alpha, beta)
            sample = np.random.beta(arm.alpha, arm.beta)

            if sample > max_sample:
                max_sample = sample
                best_arm = arm_name

        self.state.current_arm = best_arm or StrategyType.CONSERVATIVE.value
        return self.state.current_arm

    def update(self, arm_name: str, reward: float):
        """
        Updates the Alpha/Beta parameters based on reward.
        Reward should be normalized [0, 1] ideally, or binary (Win/Loss).
        Here we treat Profit > 0 as Win (Reward=1) and Loss as 0.
        """
        if arm_name not in self.state.arms:
            return

        arm = self.state.arms[arm_name]
        self.state.total_pulls += 1

        # Simple Binary Reward for Thompson Sampling
        # Can be improved with Normal-Gamma for continuous reward
        if reward > 0:
            arm.alpha += 1
        else:
            arm.beta += 1

    def force_difficulty(self, strategy: StrategyType):
        """Overrides the bandit for safety (e.g., Drawdown too high)."""
        self.state.current_arm = strategy.value
