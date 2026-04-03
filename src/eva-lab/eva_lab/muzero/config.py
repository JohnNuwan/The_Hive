"""Configuration centralisee pour MuZero et Dreamer cote EVA Lab."""

from __future__ import annotations

import logging
import os

from eva_lab.training_utils import (
    build_dataset_coverage,
    build_dataset_descriptor,
    get_horizon_history_bars,
    get_horizon_timeframe,
    infer_family_from_symbols,
    resolve_symbol_overrides,
    resolve_family_training_symbols,
    resolve_feature_profile,
    resolve_position_mechanics_profile,
    resolve_training_symbols,
)

logger = logging.getLogger(__name__)


class MuZeroConfigV3:
    """Configuration d'entrainement MuZero adaptee au trading reel."""

    def __init__(self, **overrides):
        handled_override_keys = {
            "horizon",
            "primary_timeframe",
            "model_family",
            "symbols",
            "max_symbols",
            "dataset_source",
        }
        self.observation_shape = (32,)
        self.action_space_size = 5
        self.hidden_state_size = 256
        self.network_hidden_dims = [512, 512, 512]

        self.horizon = str(overrides.get("horizon") or os.getenv("MUZERO_HORIZON", "intraday")).lower()
        self.primary_timeframe = str(
            overrides.get("primary_timeframe") or get_horizon_timeframe(self.horizon)
        ).upper()
        self.model_family = str(
            overrides.get("model_family") or os.getenv("MUZERO_MODEL_FAMILY", "")
        ).strip().lower() or None
        horizon_env = self.horizon.upper()
        max_symbols = int(overrides.get("max_symbols") or os.getenv("MUZERO_MAX_SYMBOLS", "12"))
        explicit_symbols = overrides.get("symbols")
        explicit_env_symbols, explicit_env_source = resolve_symbol_overrides(
            [
                "TRAINING_FOCUS_SYMBOLS",
                f"MUZERO_SYMBOLS_{horizon_env}",
                "MUZERO_SYMBOLS",
                "TRAINING_SYMBOLS",
            ]
        )
        family_hint = self.model_family
        if explicit_symbols:
            self.symbols = [str(symbol).strip() for symbol in list(explicit_symbols) if str(symbol).strip()]
            family_hint = None
        elif explicit_env_symbols:
            # Un univers impose par l'orchestrateur nightly doit rester
            # prioritaire sur un profil famille historique (`fx`, `metals`, etc.).
            self.symbols = explicit_env_symbols[:max_symbols] if max_symbols > 0 else explicit_env_symbols
            family_hint = None
            if explicit_env_source:
                logger.info(
                    "Univers MuZero force via %s (%s symboles).",
                    explicit_env_source,
                    len(self.symbols),
                )
        elif self.model_family:
            self.symbols = resolve_family_training_symbols(
                horizon=self.horizon,
                family=self.model_family,
                max_symbols=max_symbols,
            )
        else:
            self.symbols = resolve_training_symbols(
                required_timeframes={self.primary_timeframe},
                max_symbols=max_symbols,
                override_env_names=[
                    f"MUZERO_SYMBOLS_{horizon_env}",
                    "MUZERO_SYMBOLS",
                ],
            )
        if not self.symbols:
            self.symbols = ["XAUUSD", "EURUSD", "BTCUSD"]
            logger.warning("Aucun historique compatible detecte. Fallback sur un univers minimal.")
        self.model_family = infer_family_from_symbols(self.symbols, family=family_hint)
        self.feature_profile = resolve_feature_profile(self.horizon, self.model_family)
        self.position_mechanics_profile = resolve_position_mechanics_profile(
            self.horizon,
            self.model_family,
        )
        self.mechanics_profile_version = str(
            self.position_mechanics_profile.get("profile_version") or "v1"
        )
        self.history_bars = get_horizon_history_bars(
            self.horizon,
            env_prefix="MUZERO_HISTORY",
            fallback=4000,
        )
        self.dataset_source = str(
            overrides.get("dataset_source") or os.getenv("MUZERO_DATASET_SOURCE", "auto")
        )
        self.dataset_coverage = build_dataset_coverage(
            symbols=self.symbols,
            timeframe=self.primary_timeframe,
        )
        effective_source = str(self.dataset_coverage.get("effective_source") or "").strip().lower()
        if self.dataset_source in {"", "auto"} and effective_source:
            self.dataset_source = effective_source
        self.dataset_descriptor = build_dataset_descriptor(
            horizon=self.horizon,
            family=self.model_family,
            timeframe=self.primary_timeframe,
            symbols=self.symbols,
            source=self.dataset_source,
            feature_profile=self.feature_profile,
            mechanics_profile=self.position_mechanics_profile,
            history_bars=self.history_bars,
            dataset_coverage=self.dataset_coverage,
        )
        self.dataset_id = self.dataset_descriptor["dataset_id"]

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
            if key in handled_override_keys:
                continue
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
