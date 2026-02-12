"""
Tests for MABSolver (Multi-Armed Bandit).
"""
import pytest
from eva_gamification.mab import MABSolver, StrategyType

def test_mab_initialization():
    mab = MABSolver()
    # Default state check
    assert mab.state.current_arm == StrategyType.BALANCED.value
    assert StrategyType.CONSERVATIVE.value in mab.state.arms
    assert StrategyType.AGGRESSIVE.value in mab.state.arms
    assert mab.state.arms[StrategyType.AGGRESSIVE.value].locked is True

def test_mab_selection_locked():
    mab = MABSolver()

    # Run selection many times with 0 SP -> Aggressive should never be picked
    for _ in range(50):
        arm = mab.select_arm(0)
        assert arm != StrategyType.AGGRESSIVE.value

def test_mab_unlocking():
    mab = MABSolver()

    # Run selection with sufficient SP to unlock Aggressive
    # It might not be picked immediately due to probability, but 'locked' status should change
    mab.select_arm(6000)
    assert mab.state.arms[StrategyType.AGGRESSIVE.value].locked is False

def test_mab_update():
    mab = MABSolver()
    arm_name = StrategyType.BALANCED.value
    initial_alpha = mab.state.arms[arm_name].alpha
    initial_beta = mab.state.arms[arm_name].beta

    # Simulate a Win
    mab.update(arm_name, 1.0)
    assert mab.state.arms[arm_name].alpha == initial_alpha + 1
    assert mab.state.arms[arm_name].beta == initial_beta

    # Simulate a Loss
    mab.update(arm_name, 0.0)
    assert mab.state.arms[arm_name].beta == initial_beta + 1
