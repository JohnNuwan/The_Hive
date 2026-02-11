"""
MuZero V3.1 Configuration — "Hunger Mode" (Pro Trader Edition)

Ported from Muzero_Pro_Trader → adapted for The Hive eva-lab.

Features:
  - 142-feature observation space (DeepTrinity 136 + pos/pnl/slbe + time/vol)
  - 5 discrete actions: Hold, Buy, Sell, Split, Close
  - 11 multi-asset symbols (Forex, Crypto, Indices)
  - Aggressive reward shaping (doubled bonuses, unchanged penalties)
  - MCTS with 150 simulations + Dirichlet noise
"""
import os
import logging

logger = logging.getLogger(__name__)


class MuZeroConfigV3:
    """Hunger Mode V3.1 — chase big wins, fear losses just as much."""

    def __init__(self, **overrides):
        # ═══ Network Architecture ═══
        self.observation_shape = (142,)  # DeepTrinity(136) + V3(3) + Time(3) = 142
        self.action_space_size = 5       # Hold, Buy, Sell, Split, Close
        self.hidden_state_size = 64
        self.network_hidden_dims = [256, 256]

        # ═══ Symbols (Multi-Asset) ═══
        self.symbols = [
            "EURUSD", "XAUUSD", "BTCUSD",
            "US30.cash", "US500.cash", "USDJPY",
            "GBPUSD", "USDCAD", "USDCHF",
            "GER40.cash", "US100.cash",
        ]

        # ═══ Training — Aggressive ═══
        self.batch_size = 128
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_steps = 30_000
        self.checkpoint_interval = 100
        self.num_unroll_steps = 5
        self.td_steps = 10

        # ═══ MCTS ═══
        self.num_simulations = 150
        self.discount = 0.99
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.50  # Electrochoc V3.1
        self.pb_c_base = 19_652
        self.pb_c_init = 1.25

        # ═══ Replay Buffer ═══
        self.window_size = 200_000
        self.max_moves = 500

        # ═══ Paths ═══
        self.results_path = os.path.join("data", "muzero", "results")
        self.weights_path = os.path.join("data", "muzero", "weights")

        # ═══ Reward Shaping — Hunger Mode V3.1 (DOUBLED) ═══
        self.quality_trade_bonus = 10.0      # +10 pts per 1% trade
        self.final_growth_bonus = 50.0       # +50 pts end-of-episode
        self.final_growth_threshold = 0.10   # 10% growth needed

        # SLBE Rewards (DOUBLED)
        self.slbe_activation_bonus = 6.0     # +6 for activating SLBE
        self.split_with_profit_bonus = 10.0  # +10 for smart SPLIT
        self.close_big_winner_bonus = 15.0   # +15 for CLOSE > +2%

        # Time Penalties (INCREASED)
        self.drawdown_time_penalty_rate = 0.2   # -0.2 per 20 steps
        self.max_drawdown_penalty = 10.0        # -10 for > 5% DD
        self.loss_penalty_multiplier = 2.0      # 2x asymmetric penalty

        # ═══ Apply overrides ═══
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"[MuZero:Config] Override {key} = {value}")
            else:
                logger.warning(f"[MuZero:Config] Unknown key: {key}")

    def visit_softmax_temperature(self, trained_steps: int) -> float:
        """Aggressive temperature decay for exploitation."""
        if trained_steps < 0.3 * self.training_steps:
            return 1.0
        elif trained_steps < 0.6 * self.training_steps:
            return 0.5
        else:
            return 0.1

    def to_dict(self) -> dict:
        """Serialize config for logging/archival."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
