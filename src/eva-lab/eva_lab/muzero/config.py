"""Configuration centralisee pour MuZero et Dreamer cote EVA Lab."""

from __future__ import annotations

import logging
import os

from eva_lab.training_utils import get_horizon_timeframe, resolve_training_symbols

logger = logging.getLogger(__name__)


class MuZeroConfigV3:
    """Configuration d'entrainement MuZero adaptee au trading reel."""

    def __init__(self, **overrides):
        self.observation_shape = (32,)
        self.action_space_size = 5
        self.hidden_state_size = 256
        self.network_hidden_dims = [512, 512, 512]

        self.horizon = os.getenv("MUZERO_HORIZON", "intraday").lower()
        self.primary_timeframe = get_horizon_timeframe(self.horizon)
        self.symbols = resolve_training_symbols(
            required_timeframes={self.primary_timeframe},
            max_symbols=int(os.getenv("MUZERO_MAX_SYMBOLS", "12")),
        )
        if not self.symbols:
            self.symbols = ["XAUUSD", "EURUSD", "BTCUSD"]
            logger.warning("Aucun historique compatible detecte. Fallback sur un univers minimal.")

        self.batch_size = int(os.getenv("MUZERO_BATCH_SIZE", "32"))
        self.learning_rate = float(os.getenv("MUZERO_LEARNING_RATE", "5e-5"))
        self.weight_decay = float(os.getenv("MUZERO_WEIGHT_DECAY", "1e-4"))
        self.momentum = 0.9
        self.training_steps = int(os.getenv("MUZERO_TRAINING_STEPS", "24000"))
        self.checkpoint_interval = int(os.getenv("MUZERO_CHECKPOINT_INTERVAL", "500"))
        self.num_unroll_steps = int(os.getenv("MUZERO_NUM_UNROLL_STEPS", "5"))
        self.td_steps = int(os.getenv("MUZERO_TD_STEPS", "10"))
        self.max_moves = int(os.getenv("MUZERO_MAX_MOVES", "300"))

        self.num_simulations = int(os.getenv("MUZERO_NUM_SIMULATIONS", "100"))
        self.discount = 0.99
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.50
        self.pb_c_base = 19_652
        self.pb_c_init = 1.25

        self.window_size = int(os.getenv("MUZERO_WINDOW_SIZE", "200000"))

        self.results_path = os.path.join("data", "muzero", "results")
        self.weights_path = os.path.join("data", "muzero", "weights")

        self.quality_trade_bonus = 10.0
        self.final_growth_bonus = 50.0
        self.final_growth_threshold = 0.10
        self.slbe_activation_bonus = 6.0
        self.split_with_profit_bonus = 10.0
        self.close_big_winner_bonus = 15.0
        self.drawdown_time_penalty_rate = 0.2
        self.max_drawdown_penalty = 10.0
        self.loss_penalty_multiplier = 2.0

        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info("[MuZeroConfig] Override %s=%s", key, value)
            else:
                logger.warning("[MuZeroConfig] Cle inconnue ignoree: %s", key)

    def visit_softmax_temperature(self, trained_steps: int) -> float:
        """Retourne la temperature d'exploration selon l'avancement du train."""
        if trained_steps < 0.3 * self.training_steps:
            return 1.0
        if trained_steps < 0.6 * self.training_steps:
            return 0.5
        return 0.1

    def to_dict(self) -> dict:
        """Serialize la configuration pour archivage et debug."""
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
