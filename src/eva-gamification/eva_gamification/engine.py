"""
Gamification Engine - Core Logic for "EVA-Evolution".
Manages Sovereignty Points (SP), Tech Tree, and Dynamic Difficulty.
"""

import asyncio
import json
import logging
import subprocess
from typing import Optional

from shared.redis_client import RedisClient, get_redis_client
from eva_gamification.models import TradeResult, GameState, GamificationConfig
from eva_gamification.mab import MABSolver, StrategyType, MABState
from eva_gamification.compliance import ComplianceGovernor

logger = logging.getLogger(__name__)

class GamificationEngine:

    REDIS_KEY_STATE = "gamification:state"
    REDIS_KEY_MAB = "gamification:mab"

    def __init__(self):
        self.redis = get_redis_client()
        self.config = GamificationConfig()
        self.state = GameState()
        self.mab = MABSolver()
        self.governor = ComplianceGovernor(self.redis)
        self.logger = logger

    async def initialize(self):
        """Loads state from Redis."""
        await self.redis.connect()

        # Load Game State
        data = await self.redis.cache_get(self.REDIS_KEY_STATE)
        if data:
            self.state = GameState(**data)

        # Load MAB State
        mab_data = await self.redis.cache_get(self.REDIS_KEY_MAB)
        if mab_data:
            try:
                self.mab.state = MABState(**mab_data)
            except Exception as e:
                self.logger.warning(f"Failed to load MAB state: {e}. Using default.")

        self.logger.info(f"Gamification Engine Online. Level: {self.state.level}, SP: {self.state.sp}")

    async def process_trade_result(self, trade: TradeResult) -> dict:
        """
        Main loop:
        1. Calculate SP based on trade outcome.
        2. Update XP/Levels/Quest Progress.
        3. Update MAB (Bandit) with reward.
        4. Check Compliance for next trades.
        5. Persist State.
        """
        # 1. Calculate SP
        sp_earned = self._calculate_sp(trade)
        self.state.sp += sp_earned
        self.state.total_profit += trade.profit

        # 2. Update Quest (4000 EUR Target)
        self.state.quest_progress = min(self.state.total_profit / self.config.quest_target_amount, 1.0)
        if self.state.quest_progress >= 1.0 and "The Sight Awakening" not in self.state.unlocked_techs:
             # Check physical unlock
             if self._detect_axelera():
                 self.state.level = 1
                 self.state.unlocked_techs.append("The Sight Awakening")
                 self.state.current_quest = "Argus Panoptes"
                 self.logger.info("LEVEL UP! Axelera Cards Detected. Vision Unlocked.")

        # 3. Update MAB
        # Reward function for Bandit: Profit > 0 = 1, else 0
        reward = 1.0 if trade.profit > 0 else 0.0
        self.mab.update(self.mab.state.current_arm, reward)

        # Select next strategy
        next_strategy = self.mab.select_arm(self.state.sp)

        # Safety Override (DDA - Anxiété)
        # If huge drawdown on this trade, force Conservative
        if trade.max_drawdown_percent > 2.0:
             self.mab.force_difficulty(StrategyType.CONSERVATIVE)
             next_strategy = StrategyType.CONSERVATIVE.value
             self.logger.warning("DDA: Forced Conservative due to high drawdown (Anxiety Zone).")

        # 4. Compliance Check (for logs mostly, actual block happens at open)
        await self.governor.record_trade()

        # 5. Persist
        await self._save_state()

        return {
            "sp_earned": sp_earned,
            "total_sp": self.state.sp,
            "level": self.state.level,
            "next_strategy": next_strategy,
            "quest_progress": self.state.quest_progress
        }

    def _calculate_sp(self, trade: TradeResult) -> float:
        """
        SP = (Profit * alpha) + (Efficiency * beta) - (Risk * gamma)
        """
        base_reward = trade.profit * self.config.alpha_profit
        efficiency_bonus = trade.efficiency_score * self.config.beta_efficiency

        # Penalty is exponential if risk was high (drawdown)
        risk_penalty = 0.0
        if trade.max_drawdown_percent > 0:
            risk_penalty = (trade.max_drawdown_percent ** 2) * self.config.gamma_risk

        return base_reward + efficiency_bonus - risk_penalty

    def _detect_axelera(self) -> bool:
        """Checks for Axelera AI cards via lspci."""
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True)
            if "Axelera" in result.stdout or "AI accelerator" in result.stdout:
                return True
        except Exception:
            return False
        return False

    async def _save_state(self):
        """Saves current state to Redis."""
        await self.redis.cache_set(self.REDIS_KEY_STATE, self.state.model_dump())
        await self.redis.cache_set(self.REDIS_KEY_MAB, self.mab.state.model_dump())
