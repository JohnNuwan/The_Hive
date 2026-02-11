"""
MuZero Trading Environment — THE HIVE (eva-lab)

Ported from Muzero_Pro_Trader/MuZero/environment/commission_trinity_env_v3.py.

This is a standalone, simplified version of the CommissionTrinityEnvV3
that works without MT5 data dependencies — it can run from OHLCV DataFrames
or synthetic data for testing.

Features:
  - 5 discrete actions: Hold, Buy, Sell, Split, Close All
  - SLBE (Stop Loss Break Even) system
  - Dynamic position sizing and pyramiding
  - Commission modeling (5 bps)
  - Hunger Mode reward shaping (V3.1)
  - Time-based drawdown penalties
  - Multi-asset symbol support (11 instruments)
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Actions
HOLD = 0
BUY = 1
SELL = 2
SPLIT = 3
CLOSE = 4

ACTION_NAMES = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]


@dataclass
class SymbolSpec:
    """Per-symbol trading specifications."""
    pip_size: float
    trade_size: float
    initial_balance: float

    @classmethod
    def for_symbol(cls, symbol: str) -> "SymbolSpec":
        symbol = symbol.upper()
        if "JPY" in symbol:
            return cls(pip_size=0.01, trade_size=10_000.0, initial_balance=1_000_000.0)
        elif any(idx in symbol for idx in ["US30", "GER40", "US100", "US500"]):
            return cls(pip_size=1.0, trade_size=0.1, initial_balance=100_000.0)
        elif "BTC" in symbol:
            return cls(pip_size=1.0, trade_size=0.1, initial_balance=100_000.0)
        elif "XAU" in symbol:
            return cls(pip_size=0.01, trade_size=1.0, initial_balance=10_000.0)
        else:  # Forex majors (EURUSD, GBPUSD, etc.)
            return cls(pip_size=0.0001, trade_size=10_000.0, initial_balance=10_000.0)


class TradingEnvironment:
    """
    Gymnasium-compatible trading environment for MuZero.

    Accepts OHLCV data as a numpy array or can generate synthetic data for testing.
    Observation: (feature_count + 6) features
      - Base features from data columns
      - +6: position_state, pnl_pct, slbe_active, hour, day, volatility
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        symbol: str = "XAUUSD",
        config=None,
        max_steps: int = 1000,
    ):
        """
        Args:
            data: OHLCV+ array of shape (timesteps, features) with at least
                  columns [open, high, low, close] in the first 4 positions.
                  If None, generates synthetic data for testing.
            symbol: Trading instrument name.
            config: MuZeroConfigV3 instance (uses defaults if None).
        """
        self.symbol = symbol
        self.spec = SymbolSpec.for_symbol(symbol)
        self.max_steps_per_episode = max_steps
        self.commission_rate = 0.00005  # 5 bps

        # Reward config (from MuZeroConfigV3 or defaults)
        if config:
            self.quality_mult = config.quality_trade_bonus
            self.final_growth_bonus = config.final_growth_bonus
            self.final_growth_threshold = config.final_growth_threshold
            self.dd_penalty_rate = config.drawdown_time_penalty_rate
            self.max_dd_penalty = config.max_drawdown_penalty
            self.loss_mult = config.loss_penalty_multiplier
            self.slbe_bonus = config.slbe_activation_bonus
        else:
            self.quality_mult = 10.0
            self.final_growth_bonus = 50.0
            self.final_growth_threshold = 0.10
            self.dd_penalty_rate = 0.2
            self.max_dd_penalty = 10.0
            self.loss_mult = 2.0
            self.slbe_bonus = 6.0

        # Data
        if data is not None:
            self.data = data
        else:
            self.data = self._generate_synthetic_data()

        self.base_feature_count = self.data.shape[1]
        self.observation_dim = self.base_feature_count + 6  # +6 extra features

        # State (populated in reset)
        self._reset_state()

    def _generate_synthetic_data(self, n_steps: int = 5000) -> np.ndarray:
        """Generate synthetic OHLCV data for testing."""
        np.random.seed(42)
        price = 2650.0  # Gold-like
        data = []
        for i in range(n_steps):
            change = np.random.randn() * 2.0
            o = price
            h = price + abs(np.random.randn() * 3.0)
            l = price - abs(np.random.randn() * 3.0)
            c = price + change
            v = np.random.randint(100, 10000)
            # EMA-200 (simple approximation)
            ema = price + np.random.randn() * 5.0
            data.append([o, h, l, c, v, ema])
            price = c
        return np.array(data, dtype=np.float32)

    def _reset_state(self):
        """Reset all internal trading state."""
        self.current_step = 100  # Skip first 100 for indicators
        self.start_step = self.current_step
        self.balance = self.spec.initial_balance
        self.peak_equity = self.spec.initial_balance
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.slbe_active = False
        self.slbe_price = 0.0
        self.steps_in_drawdown = 0
        self.steps_since_last_trade = 0
        self.total_trades = 0
        self.total_profitable = 0
        self.split_count = 0
        self.secured_count = 0
        self.equity_curve = [self.spec.initial_balance]

    def reset(self, seed=None, options=None):
        """Reset environment for a new episode."""
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Build full observation vector."""
        base = self.data[self.current_step].copy()

        # Position state
        pos_state = 0.0
        if self.position_size > 0:
            pos_state = 1.0
        elif self.position_size < 0:
            pos_state = -1.0

        # Unrealized PnL %
        pnl_pct = 0.0
        if self.position_size != 0:
            price = self.data[self.current_step, 3]  # close
            if self.position_size > 0:
                pnl_pct = (price - self.avg_entry_price) / max(self.avg_entry_price, 1e-8)
            else:
                pnl_pct = (self.avg_entry_price - price) / max(self.avg_entry_price, 1e-8)

        slbe_state = 1.0 if self.slbe_active else 0.0

        # Time features (normalized)
        hour_feat = (self.current_step % 24) / 23.0
        day_feat = ((self.current_step // 24) % 5) / 4.0

        # Volatility (high-low range normalized)
        h, l, c = self.data[self.current_step, 1], self.data[self.current_step, 2], self.data[self.current_step, 3]
        vol = min((h - l) / max(c, 1e-8) * 100.0, 1.0) if c > 0 else 0.0

        extra = np.array([pos_state, pnl_pct, slbe_state, hour_feat, day_feat, vol], dtype=np.float32)
        return np.concatenate([base, extra])

    def step(self, action: int):
        """Execute one trading step."""
        price = self.data[self.current_step, 3]  # close
        ema_200 = self.data[self.current_step, 5] if self.data.shape[1] > 5 else price  # ema_200
        reward = 0.0
        done = False
        realized_pnl = 0.0

        trade_size = self.spec.trade_size
        MAX_POS = (2 + self.secured_count) * trade_size

        # ── SLBE Check ──
        if self.slbe_active and self.position_size != 0:
            hit = False
            if self.position_size > 0 and price <= self.slbe_price:
                hit = True
            elif self.position_size < 0 and price >= self.slbe_price:
                hit = True
            if hit:
                commission = abs(self.position_size) * price * self.commission_rate
                self.balance -= commission
                self.position_size = 0
                self.avg_entry_price = 0
                self.slbe_active = False
                self.slbe_price = 0
                reward += 1.0

        # ── Auto SLBE Activation (+0.5%) ──
        if not self.slbe_active and self.position_size != 0:
            if self.position_size > 0:
                unr = (price - self.avg_entry_price) / max(self.avg_entry_price, 1e-8)
            else:
                unr = (self.avg_entry_price - price) / max(self.avg_entry_price, 1e-8)
            if unr >= 0.005:
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                self.secured_count += 1
                reward += self.slbe_bonus

        # Force HOLD if no position for Split/Close
        if self.position_size == 0 and action in [SPLIT, CLOSE]:
            action = HOLD

        # Trend filter
        if action == BUY and price < ema_200:
            action = HOLD
        elif action == SELL and price > ema_200:
            action = HOLD

        # ── Execute Action ──
        if action == BUY:
            if self.position_size <= 0:
                cost = trade_size * price
                self.balance -= cost * self.commission_rate
                total_val = (self.position_size * self.avg_entry_price) + (trade_size * price)
                self.position_size += trade_size
                self.avg_entry_price = total_val / self.position_size if abs(self.position_size) > 1e-9 else 0
                self.split_count = 0
            elif self.position_size < MAX_POS:
                curr_pnl = (price - self.avg_entry_price) / max(self.avg_entry_price, 1e-8)
                if curr_pnl > 0.001:
                    self.balance -= trade_size * price * self.commission_rate
                    total_val = (self.position_size * self.avg_entry_price) + (trade_size * price)
                    self.position_size += trade_size
                    self.avg_entry_price = total_val / self.position_size
                    reward += 0.1
                else:
                    reward -= 0.1

        elif action == SELL:
            if self.position_size >= 0:
                cost = trade_size * price
                self.balance -= cost * self.commission_rate
                if self.position_size > 0:
                    self.position_size = 0
                total_val = (abs(self.position_size) * self.avg_entry_price) + (trade_size * price)
                self.position_size -= trade_size
                self.avg_entry_price = total_val / abs(self.position_size) if abs(self.position_size) > 1e-9 else 0
                self.split_count = 0
            elif self.position_size > -MAX_POS:
                curr_pnl = (self.avg_entry_price - price) / max(self.avg_entry_price, 1e-8)
                if curr_pnl > 0.001:
                    self.balance -= trade_size * price * self.commission_rate
                    total_val = (abs(self.position_size) * self.avg_entry_price) + (trade_size * price)
                    self.position_size -= trade_size
                    self.avg_entry_price = total_val / abs(self.position_size)
                    reward += 0.1
                else:
                    reward -= 0.1

        elif action == SPLIT and abs(self.position_size) > 0:
            close_amt = abs(self.position_size) * 0.5
            if self.position_size > 0:
                pnl = (price - self.avg_entry_price) * close_amt
                trade_ret = (price - self.avg_entry_price) / max(self.avg_entry_price, 1e-8)
                self.position_size -= close_amt
            else:
                pnl = (self.avg_entry_price - price) * close_amt
                trade_ret = (self.avg_entry_price - price) / max(self.avg_entry_price, 1e-8)
                self.position_size += close_amt

            commission = close_amt * price * self.commission_rate
            realized_pnl = pnl - commission
            self.balance += realized_pnl
            self.total_trades += 1

            if self.split_count < 3:
                if trade_ret > 0.01:
                    reward += self.quality_mult
                    self.total_profitable += 1
                    self.split_count += 1
                elif pnl > 0:
                    reward += 1.0
                    self.total_profitable += 1
                    self.split_count += 1

            if not self.slbe_active:
                self.slbe_active = True
                self.slbe_price = self.avg_entry_price
                reward += 2.0

        elif action == CLOSE and abs(self.position_size) > 0:
            if self.position_size > 0:
                pnl = (price - self.avg_entry_price) * abs(self.position_size)
                trade_ret = (price - self.avg_entry_price) / max(self.avg_entry_price, 1e-8)
            else:
                pnl = (self.avg_entry_price - price) * abs(self.position_size)
                trade_ret = (self.avg_entry_price - price) / max(self.avg_entry_price, 1e-8)

            commission = abs(self.position_size) * price * self.commission_rate
            realized_pnl = pnl - commission
            self.balance += realized_pnl
            self.position_size = 0
            self.avg_entry_price = 0
            self.slbe_active = False
            self.slbe_price = 0
            self.total_trades += 1

            if trade_ret > 0.02:
                reward += self.quality_mult * 1.5
                self.total_profitable += 1
            elif trade_ret > 0.01:
                reward += self.quality_mult
                self.total_profitable += 1
            elif pnl > 0:
                reward += 1.0
                self.total_profitable += 1
            else:
                reward += (pnl / self.spec.initial_balance) * 100.0

        # ── Metrics ──
        unrealized = 0.0
        if self.position_size != 0:
            if self.position_size > 0:
                unrealized = (price - self.avg_entry_price) * abs(self.position_size)
            else:
                unrealized = (self.avg_entry_price - price) * abs(self.position_size)

        equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Time-based drawdown penalty
        if unrealized < 0:
            self.steps_in_drawdown += 1
            reward -= self.dd_penalty_rate * (self.steps_in_drawdown / 20)
        else:
            self.steps_in_drawdown = 0

        # Inactivity penalty
        self.steps_since_last_trade += 1
        if action in [BUY, SELL, SPLIT, CLOSE]:
            self.steps_since_last_trade = 0
        elif self.steps_since_last_trade > 100:
            reward -= 1.0

        # Holding reward
        if self.position_size != 0:
            reward += (unrealized / self.spec.initial_balance) * 100.0 * 0.02

        # Max drawdown penalty
        if drawdown > 0.05:
            reward -= self.max_dd_penalty

        # Asymmetric PnL
        if realized_pnl > 0:
            reward += (realized_pnl / self.spec.initial_balance) * 100.0
        elif realized_pnl < 0:
            reward += (realized_pnl / self.spec.initial_balance) * 100.0 * self.loss_mult

        # ── Next Step ──
        self.current_step += 1
        if self.current_step - self.start_step >= self.max_steps_per_episode:
            done = True
        if self.current_step >= len(self.data) - 1:
            done = True

        # Final growth bonus
        if done:
            final_growth = (equity - self.spec.initial_balance) / self.spec.initial_balance
            if final_growth >= self.final_growth_threshold:
                reward += self.final_growth_bonus
                logger.info(f"🎯 FINAL GROWTH BONUS: {final_growth*100:.1f}% → +{self.final_growth_bonus}")

        self.equity_curve.append(equity)
        info = {
            "balance": self.balance,
            "equity": equity,
            "step": self.current_step,
            "realized_pnl": realized_pnl,
            "drawdown": drawdown,
            "slbe_active": self.slbe_active,
            "total_trades": self.total_trades,
            "win_rate": self.total_profitable / max(self.total_trades, 1),
        }

        obs = self._get_observation()
        return obs, reward, done, False, info

    def get_summary(self) -> dict:
        """Episode summary stats."""
        equity = self.equity_curve[-1] if self.equity_curve else self.spec.initial_balance
        return {
            "symbol": self.symbol,
            "total_trades": self.total_trades,
            "profitable_trades": self.total_profitable,
            "win_rate": self.total_profitable / max(self.total_trades, 1),
            "final_equity": equity,
            "return_pct": (equity - self.spec.initial_balance) / self.spec.initial_balance * 100,
            "steps": self.current_step - self.start_step,
        }
