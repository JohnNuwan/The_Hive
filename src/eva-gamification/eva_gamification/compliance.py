"""
Compliance Governor - Ethics & Risk Management Module.
Ensures EVA does not game the system or take excessive risks for SP.
"""

import logging
import time
from typing import Optional
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

class ComplianceGovernor:
    """
    The Governor enforces hard-coded rules to prevent:
    1. Over-trading (Grinding/Farming SP).
    2. Excessive Drawdown (Gambling).
    """

    MAX_TRADES_PER_HOUR = 5
    MAX_DRAWDOWN_PERCENT = 4.0
    COOLDOWN_DURATION_HOURS = 4

    def __init__(self, redis_client):
        self.redis_wrapper = redis_client
        self.logger = logger

    async def _get_redis(self) -> Redis:
        """Helper to get the raw redis client."""
        await self.redis_wrapper.connect()
        return self.redis_wrapper._client

    async def can_trade(self, account_balance: float, current_equity: float) -> bool:
        """
        Checks if trading is allowed based on risk parameters.
        """
        # 1. Check Drawdown
        if account_balance <= 0:
            return False

        drawdown = ((account_balance - current_equity) / account_balance) * 100
        if drawdown >= self.MAX_DRAWDOWN_PERCENT:
            self.logger.warning(f"Compliance Block: Drawdown {drawdown:.2f}% exceeds {self.MAX_DRAWDOWN_PERCENT}% limit.")
            return False

        # 2. Check Cooldown (Anti-Grinding)
        redis = await self._get_redis()
        cooldown = await redis.get("gamification:governor:cooldown")
        if cooldown:
            remaining = float(cooldown) - time.time()
            if remaining > 0:
                self.logger.info(f"Compliance Block: System in Cooldown ({remaining:.0f}s left).")
                return False

        # 3. Check Frequency (Trades per hour)
        trades_last_hour = await redis.get("gamification:governor:trades_1h")
        if trades_last_hour:
            count = int(trades_last_hour)
            if count >= self.MAX_TRADES_PER_HOUR:
                self.logger.warning(f"Compliance Block: Too many trades ({count}/h). Activating Cooldown.")
                await self.activate_cooldown()
                return False

        return True

    async def record_trade(self):
        """Records a trade execution for frequency monitoring."""
        redis = await self._get_redis()
        key = "gamification:governor:trades_1h"

        # Atomic increment
        new_val = await redis.incr(key)

        # Set expiry on first increment
        if new_val == 1:
            await redis.expire(key, 3600)

    async def activate_cooldown(self):
        """Activates a temporary ban on trading."""
        redis = await self._get_redis()
        expiry = time.time() + (self.COOLDOWN_DURATION_HOURS * 3600)
        await redis.set("gamification:governor:cooldown", str(expiry), ex=self.COOLDOWN_DURATION_HOURS * 3600)
        self.logger.error(f"Governor activated {self.COOLDOWN_DURATION_HOURS}h cooldown due to excessive trading.")

    def log_decision(self, action: str, reason: str):
        """XAI Log for transparency."""
        self.logger.info(f"[GOVERNOR] {action}: {reason}")
