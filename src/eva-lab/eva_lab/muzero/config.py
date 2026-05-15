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
        self.support_size = int(overrides.get("support_size") or os.getenv("MUZERO_SUPPORT_SIZE", "100"))

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
        if explicit_symbols:
            self.symbols = [str(symbol).strip() for symbol in list(explicit_symbols) if str(symbol).strip()]
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
        self.model_family = infer_family_from_symbols(self.symbols, family=self.model_family)
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
        raw_collection_max_moves = str(
            os.getenv("MUZERO_COLLECTION_MAX_MOVES", "")
        ).strip()
        self.randomize_episode_start = str(
            os.getenv("MUZERO_RANDOMIZE_EPISODE_START", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.episode_warmup_bars = int(os.getenv("MUZERO_EPISODE_WARMUP_BARS", "100"))
        self.collection_heartbeat_every_steps = int(
            os.getenv("MUZERO_COLLECTION_HEARTBEAT_EVERY_STEPS", "25")
        )
        self.collection_heartbeat_every_seconds = float(
            os.getenv("MUZERO_COLLECTION_HEARTBEAT_EVERY_SECONDS", "30")
        )
        self.collection_max_episode_seconds = float(
            os.getenv("MUZERO_COLLECTION_MAX_EPISODE_SECONDS", "300")
        )
        self.collection_max_step_seconds = float(
            os.getenv("MUZERO_COLLECTION_MAX_STEP_SECONDS", "20")
        )
        self.collection_parallel_games = max(
            1,
            int(os.getenv("MUZERO_COLLECTION_PARALLEL_GAMES", "1")),
        )
        self.directional_curriculum_soft_end_step = int(
            os.getenv("MUZERO_DIRECTIONAL_CURRICULUM_SOFT_END_STEP", "8000")
        )
        self.directional_curriculum_end_step = int(
            os.getenv("MUZERO_DIRECTIONAL_CURRICULUM_END_STEP", "15000")
        )
        self.directional_collapse_check_step = int(
            os.getenv("MUZERO_DIRECTIONAL_COLLAPSE_CHECK_STEP", "4000")
        )
        self.directional_collapse_stop_step = int(
            os.getenv("MUZERO_DIRECTIONAL_COLLAPSE_STOP_STEP", "8000")
        )
        self.directional_collapse_max_imbalance = float(
            os.getenv("MUZERO_DIRECTIONAL_COLLAPSE_MAX_IMBALANCE", "0.80")
        )
        self.soft_reward_early_end_step = int(
            os.getenv("MUZERO_SOFT_REWARD_EARLY_END_STEP", "4000")
        )
        self.soft_reward_mid_end_step = int(
            os.getenv("MUZERO_SOFT_REWARD_MID_END_STEP", "10000")
        )
        self.soft_reward_penalty_scale_early = float(
            os.getenv("MUZERO_SOFT_REWARD_PENALTY_SCALE_EARLY", "0.45")
        )
        self.soft_reward_penalty_scale_mid = float(
            os.getenv("MUZERO_SOFT_REWARD_PENALTY_SCALE_MID", "0.65")
        )
        self.soft_reward_penalty_scale_late = float(
            os.getenv("MUZERO_SOFT_REWARD_PENALTY_SCALE_LATE", "0.85")
        )
        self.soft_reward_bonus_scale_early = float(
            os.getenv("MUZERO_SOFT_REWARD_BONUS_SCALE_EARLY", "1.15")
        )
        self.soft_reward_bonus_scale_mid = float(
            os.getenv("MUZERO_SOFT_REWARD_BONUS_SCALE_MID", "1.00")
        )
        self.soft_reward_bonus_scale_late = float(
            os.getenv("MUZERO_SOFT_REWARD_BONUS_SCALE_LATE", "0.95")
        )
        self.exit_reward_early_end_step = int(
            os.getenv("MUZERO_EXIT_REWARD_EARLY_END_STEP", "4000")
        )
        self.exit_reward_mid_end_step = int(
            os.getenv("MUZERO_EXIT_REWARD_MID_END_STEP", "10000")
        )
        self.exit_reward_scale_early = float(
            os.getenv("MUZERO_EXIT_REWARD_SCALE_EARLY", "0.35")
        )
        self.exit_reward_scale_mid = float(
            os.getenv("MUZERO_EXIT_REWARD_SCALE_MID", "0.55")
        )
        self.exit_reward_scale_late = float(
            os.getenv("MUZERO_EXIT_REWARD_SCALE_LATE", "1.00")
        )
        self.policy_precheck_step = int(
            os.getenv("MUZERO_POLICY_PRECHECK_STEP", "12000")
        )
        self.policy_precheck_max_loss_pol = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MAX_LOSS_POL", "5.8")
        )
        self.policy_precheck_max_loss_pol_per_head = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MAX_LOSS_POL_PER_HEAD", "1.12")
        )
        self.policy_precheck_min_top1_share = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_TOP1_SHARE", "0.75")
        )
        self.policy_precheck_max_policy_entropy = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MAX_POLICY_ENTROPY", "1.00")
        )
        self.policy_precheck_max_root_mask_rate = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MAX_ROOT_MASK_RATE", "0.02")
        )
        self.policy_precheck_max_post_veto_rate = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MAX_POST_VETO_RATE", "0.01")
        )
        self.policy_precheck_min_balanced_episode_rate = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_BALANCED_EPISODE_RATE", "0.85")
        )
        self.policy_precheck_min_long_entry_share = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_LONG_ENTRY_SHARE", "0.35")
        )
        self.policy_precheck_min_short_entry_share = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_SHORT_ENTRY_SHARE", "0.35")
        )
        self.policy_precheck_min_close_quality_score = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_CLOSE_QUALITY_SCORE", "0.40")
        )
        self.policy_precheck_min_split_efficiency = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_SPLIT_EFFICIENCY", "0.35")
        )
        self.policy_precheck_min_pyramid_efficiency = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_PYRAMID_EFFICIENCY", "0.35")
        )
        self.policy_precheck_min_slbe_capture_rate = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_SLBE_CAPTURE_RATE", "0.45")
        )
        self.policy_precheck_max_hold_drag_score = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MAX_HOLD_DRAG_SCORE", "0.10")
        )
        self.policy_precheck_min_good_close_symbols = int(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_GOOD_CLOSE_SYMBOLS", "5")
        )
        self.policy_precheck_min_symbol_close_quality_score = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_SYMBOL_CLOSE_QUALITY_SCORE", "0.25")
        )
        self.policy_precheck_min_symbol_close_events = int(
            os.getenv("MUZERO_POLICY_PRECHECK_MIN_SYMBOL_CLOSE_EVENTS", "6")
        )
        self.policy_screen_max_loss_pol = float(
            os.getenv("MUZERO_POLICY_SCREEN_MAX_LOSS_POL", "6.6")
        )
        self.policy_screen_max_loss_pol_per_head = float(
            os.getenv("MUZERO_POLICY_SCREEN_MAX_LOSS_POL_PER_HEAD", "1.20")
        )
        self.policy_screen_min_top1_share = float(
            os.getenv("MUZERO_POLICY_SCREEN_MIN_TOP1_SHARE", "0.60")
        )
        self.policy_screen_max_policy_entropy = float(
            os.getenv("MUZERO_POLICY_SCREEN_MAX_POLICY_ENTROPY", "1.10")
        )
        self.policy_screen_max_root_mask_rate = float(
            os.getenv("MUZERO_POLICY_SCREEN_MAX_ROOT_MASK_RATE", "0.05")
        )
        self.policy_screen_max_post_veto_rate = float(
            os.getenv("MUZERO_POLICY_SCREEN_MAX_POST_VETO_RATE", "0.01")
        )
        self.policy_screen_min_balanced_episode_rate = float(
            os.getenv("MUZERO_POLICY_SCREEN_MIN_BALANCED_EPISODE_RATE", "0.85")
        )
        self.policy_screen_min_long_entry_share = float(
            os.getenv("MUZERO_POLICY_SCREEN_MIN_LONG_ENTRY_SHARE", "0.35")
        )
        self.policy_screen_min_short_entry_share = float(
            os.getenv("MUZERO_POLICY_SCREEN_MIN_SHORT_ENTRY_SHARE", "0.35")
        )
        self.policy_precheck_window_size = int(
            os.getenv("MUZERO_POLICY_PRECHECK_WINDOW_SIZE", "500")
        )
        self.policy_precheck_max_root_mask_rate_trend = float(
            os.getenv("MUZERO_POLICY_PRECHECK_MAX_ROOT_MASK_RATE_TREND", "0.02")
        )
        self.arena_screen_recent_steps = int(
            os.getenv("MUZERO_ARENA_SCREEN_RECENT_STEPS", "2500")
        )
        self.arena_screen_candidate_count = int(
            os.getenv("MUZERO_ARENA_SCREEN_CANDIDATE_COUNT", "5")
        )
        self.arena_screen_target_steps = [
            int(value.strip())
            for value in os.getenv(
                "MUZERO_ARENA_SCREEN_TARGET_STEPS",
                "10000,12000,14000,16000",
            ).split(",")
            if value.strip().isdigit()
        ]
        self.arena_screen_window_size = int(
            os.getenv("MUZERO_ARENA_SCREEN_WINDOW_SIZE", "500")
        )
        self.arena_screen_games_per_symbol = int(
            os.getenv("MUZERO_ARENA_SCREEN_GAMES_PER_SYMBOL", "4")
        )
        self.arena_screen_min_games = int(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_GAMES", "14")
        )
        self.arena_screen_min_symbols = int(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOLS", "7")
        )
        self.arena_screen_min_profit_factor = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_PROFIT_FACTOR", "1.20")
        )
        self.arena_screen_min_return_pct = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_RETURN_PCT", "0.0")
        )
        self.arena_screen_min_expectancy_pct = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_EXPECTANCY_PCT", "0.0")
        )
        self.arena_screen_min_positive_episode_rate = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_POSITIVE_EPISODE_RATE", "55.0")
        )
        self.arena_screen_max_hold_drag_score = float(
            os.getenv("MUZERO_ARENA_SCREEN_MAX_HOLD_DRAG_SCORE", "0.80")
        )
        self.arena_screen_min_close_quality_score = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_CLOSE_QUALITY_SCORE", "0.35")
        )
        self.arena_screen_min_split_opportunities = int(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SPLIT_OPPORTUNITIES", "3")
        )
        self.arena_screen_min_split_efficiency = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SPLIT_EFFICIENCY", "0.35")
        )
        self.arena_screen_min_split_runner_capture_rate = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SPLIT_RUNNER_CAPTURE_RATE", "0.20")
        )
        self.arena_screen_min_pyramid_opportunities = int(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_PYRAMID_OPPORTUNITIES", "3")
        )
        self.arena_screen_min_pyramid_efficiency = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_PYRAMID_EFFICIENCY", "0.35")
        )
        self.arena_screen_min_pyramid_exit_capture_rate = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_PYRAMID_EXIT_CAPTURE_RATE", "0.20")
        )
        self.arena_screen_min_slbe_triggered = int(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SLBE_TRIGGERED", "3")
        )
        self.arena_screen_min_slbe_capture_rate = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SLBE_CAPTURE_RATE", "0.30")
        )
        self.arena_screen_min_profitable_symbols = int(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_PROFITABLE_SYMBOLS", "5")
        )
        self.arena_screen_min_symbol_profit_factor = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_PROFIT_FACTOR", "1.0")
        )
        self.arena_screen_min_symbol_return_pct = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_RETURN_PCT", "0.0")
        )
        self.arena_screen_min_symbol_split_efficiency = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_SPLIT_EFFICIENCY", "0.20")
        )
        self.arena_screen_min_symbol_split_runner_capture_rate = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_SPLIT_RUNNER_CAPTURE_RATE", "0.20")
        )
        self.arena_screen_min_symbol_pyramid_exit_capture_rate = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_PYRAMID_EXIT_CAPTURE_RATE", "0.20")
        )
        self.arena_screen_min_symbol_slbe_capture_rate = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_SLBE_CAPTURE_RATE", "0.25")
        )
        self.arena_screen_min_symbol_close_quality_score = float(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_CLOSE_QUALITY_SCORE", "0.20")
        )
        self.arena_screen_min_symbol_close_events = int(
            os.getenv("MUZERO_ARENA_SCREEN_MIN_SYMBOL_CLOSE_EVENTS", "6")
        )
        self.arena_inverse_min_profitable_symbols = int(
            os.getenv("MUZERO_ARENA_INVERSE_MIN_PROFITABLE_SYMBOLS", "5")
        )
        self.arena_family_probe_games_per_symbol = int(
            os.getenv("MUZERO_ARENA_FAMILY_PROBE_GAMES_PER_SYMBOL", "2")
        )
        self.arena_family_probe_min_ready_families = int(
            os.getenv("MUZERO_ARENA_FAMILY_PROBE_MIN_READY_FAMILIES", "3")
        )
        self.arena_family_probe_min_positive_families = int(
            os.getenv("MUZERO_ARENA_FAMILY_PROBE_MIN_POSITIVE_FAMILIES", "2")
        )
        self.arena_family_probe_max_inverse_pf_gap = float(
            os.getenv("MUZERO_ARENA_FAMILY_PROBE_MAX_INVERSE_PF_GAP", "0.10")
        )
        self.replay_hard_negative_ratio = float(
            os.getenv("MUZERO_REPLAY_HARD_NEGATIVE_RATIO", "0.20")
        )
        self.replay_hard_negative_type_cap = float(
            os.getenv("MUZERO_REPLAY_HARD_NEGATIVE_TYPE_CAP", "0.08")
        )
        self.arena_plateau_min_step = int(
            os.getenv("MUZERO_ARENA_PLATEAU_MIN_STEP", "10000")
        )
        self.arena_plateau_stop_enabled = str(
            os.getenv("MUZERO_ARENA_PLATEAU_STOP_ENABLED", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.arena_plateau_window_size = int(
            os.getenv("MUZERO_ARENA_PLATEAU_WINDOW_SIZE", "500")
        )
        self.arena_plateau_max_loss_pol_improvement = float(
            os.getenv("MUZERO_ARENA_PLATEAU_MAX_LOSS_POL_IMPROVEMENT", "0.10")
        )
        self.arena_plateau_min_split_runner_improvement = float(
            os.getenv("MUZERO_ARENA_PLATEAU_MIN_SPLIT_RUNNER_IMPROVEMENT", "0.02")
        )
        self.arena_plateau_min_pyramid_exit_improvement = float(
            os.getenv("MUZERO_ARENA_PLATEAU_MIN_PYRAMID_EXIT_IMPROVEMENT", "0.02")
        )
        self.arena_plateau_min_close_quality_improvement = float(
            os.getenv("MUZERO_ARENA_PLATEAU_MIN_CLOSE_QUALITY_IMPROVEMENT", "0.02")
        )
        self.seed_viability_min_step = int(
            os.getenv("MUZERO_SEED_VIABILITY_MIN_STEP", "6000")
        )
        self.seed_viability_max_step = int(
            os.getenv("MUZERO_SEED_VIABILITY_MAX_STEP", "8000")
        )
        self.seed_viability_max_root_mask_rate = float(
            os.getenv("MUZERO_SEED_VIABILITY_MAX_ROOT_MASK_RATE", "0.08")
        )
        self.seed_viability_min_split_runner_capture_rate = float(
            os.getenv("MUZERO_SEED_VIABILITY_MIN_SPLIT_RUNNER_CAPTURE_RATE", "0.0")
        )
        self.seed_viability_min_pyramid_exit_capture_rate = float(
            os.getenv("MUZERO_SEED_VIABILITY_MIN_PYRAMID_EXIT_CAPTURE_RATE", "0.0")
        )
        self.seed_viability_min_loss_pol_improvement = float(
            os.getenv("MUZERO_SEED_VIABILITY_MIN_LOSS_POL_IMPROVEMENT", "0.10")
        )
        self.seed_bootstrap_max_loss_pol = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MAX_LOSS_POL", "6.0")
        )
        self.seed_bootstrap_max_loss_pol_per_head = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MAX_LOSS_POL_PER_HEAD", "1.16")
        )
        self.seed_bootstrap_max_root_mask_rate = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MAX_ROOT_MASK_RATE", "0.06")
        )
        self.seed_bootstrap_min_split_monetization_window_count = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MIN_SPLIT_MONETIZATION_WINDOW_COUNT", "1")
        )
        self.seed_bootstrap_min_runner_profit_hold_window_count = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MIN_RUNNER_PROFIT_HOLD_WINDOW_COUNT", "1")
        )
        self.seed_bootstrap_min_pyramid_monetization_window_count = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MIN_PYRAMID_MONETIZATION_WINDOW_COUNT", "1")
        )
        self.seed_bootstrap_max_profit_peak_giveback_ratio = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MAX_PROFIT_PEAK_GIVEBACK_RATIO", "0.80")
        )
        self.seed_bootstrap_min_split_zone_capture_rate = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MIN_SPLIT_ZONE_CAPTURE_RATE", "0.08")
        )
        self.seed_bootstrap_min_runner_extension_capture_rate = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MIN_RUNNER_EXTENSION_CAPTURE_RATE", "0.05")
        )
        self.seed_bootstrap_min_pyramid_add_capture_rate = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MIN_PYRAMID_ADD_CAPTURE_RATE", "0.05")
        )
        self.seed_bootstrap_max_runner_giveback_ratio = float(
            os.getenv("MUZERO_SEED_BOOTSTRAP_MAX_RUNNER_GIVEBACK_RATIO", "0.60")
        )
        self.seed_mixed_max_loss_pol = float(
            os.getenv("MUZERO_SEED_MIXED_MAX_LOSS_POL", "5.2")
        )
        self.seed_mixed_max_loss_pol_per_head = float(
            os.getenv("MUZERO_SEED_MIXED_MAX_LOSS_POL_PER_HEAD", "1.10")
        )
        self.seed_mixed_max_root_mask_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MAX_ROOT_MASK_RATE", "0.06")
        )
        self.seed_mixed_min_split_runner_capture_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_SPLIT_RUNNER_CAPTURE_RATE", "0.05")
        )
        self.seed_mixed_min_split_zone_capture_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_SPLIT_ZONE_CAPTURE_RATE", "0.12")
        )
        self.seed_mixed_min_split_monetization_capture_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_SPLIT_MONETIZATION_CAPTURE_RATE", "0.12")
        )
        self.seed_mixed_min_pyramid_exit_capture_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_PYRAMID_EXIT_CAPTURE_RATE", "0.03")
        )
        self.seed_mixed_min_pyramid_add_capture_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_PYRAMID_ADD_CAPTURE_RATE", "0.08")
        )
        self.seed_mixed_min_pyramid_monetization_capture_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_PYRAMID_MONETIZATION_CAPTURE_RATE", "0.08")
        )
        self.seed_mixed_min_close_quality_score = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_CLOSE_QUALITY_SCORE", "0.40")
        )
        self.seed_mixed_min_slbe_capture_rate = float(
            os.getenv("MUZERO_SEED_MIXED_MIN_SLBE_CAPTURE_RATE", "0.45")
        )
        self.training_root_mask_soften_vwap = str(
            os.getenv("MUZERO_TRAINING_ROOT_MASK_SOFTEN_VWAP", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.training_root_mask_soften_adx = str(
            os.getenv("MUZERO_TRAINING_ROOT_MASK_SOFTEN_ADX", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.training_root_mask_soften_ema200 = str(
            os.getenv("MUZERO_TRAINING_ROOT_MASK_SOFTEN_EMA200", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.training_root_mask_soften_directional = str(
            os.getenv("MUZERO_TRAINING_ROOT_MASK_SOFTEN_DIRECTIONAL", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.training_root_mask_adx_extreme_ratio = float(
            os.getenv("MUZERO_TRAINING_ROOT_MASK_ADX_EXTREME_RATIO", "0.75")
        )

        self.num_simulations = int(os.getenv("MUZERO_NUM_SIMULATIONS", "100"))
        raw_collection_num_simulations = str(
            os.getenv("MUZERO_COLLECTION_NUM_SIMULATIONS", "")
        ).strip()
        if raw_collection_num_simulations:
            self.collection_num_simulations = max(1, int(raw_collection_num_simulations))
        else:
            # La collecte supporte un budget MCTS plus leger que l'optimisation.
            self.collection_num_simulations = max(1, min(self.num_simulations, 32))
        if raw_collection_max_moves:
            self.collection_max_moves = max(
                16,
                min(self.max_moves, int(raw_collection_max_moves)),
            )
        elif self.collection_num_simulations >= 256:
            # Avec un MCTS tres profond, garder 300 pas de collecte et un
            # timeout de cinq minutes produit surtout des episodes tronques.
            self.collection_max_moves = max(96, min(self.max_moves, 180))
        else:
            self.collection_max_moves = int(self.max_moves)
        self.policy_target_smoothing_alpha_root = float(
            os.getenv("MUZERO_POLICY_TARGET_SMOOTHING_ALPHA_ROOT", "0.28")
        )
        self.policy_target_smoothing_alpha_unroll = float(
            os.getenv("MUZERO_POLICY_TARGET_SMOOTHING_ALPHA_UNROLL", "0.18")
        )
        self.policy_target_smoothing_temperature = float(
            os.getenv("MUZERO_POLICY_TARGET_SMOOTHING_TEMPERATURE", "1.40")
        )
        self.policy_loss_root_weight = float(
            os.getenv("MUZERO_POLICY_LOSS_ROOT_WEIGHT", "0.90")
        )
        self.policy_loss_unroll_weight = float(
            os.getenv("MUZERO_POLICY_LOSS_UNROLL_WEIGHT", "0.55")
        )
        self.discount = 0.99
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = float(
            os.getenv("MUZERO_ROOT_EXPLORATION_FRACTION", "0.25")
        )
        self.pb_c_base = 19_652
        self.pb_c_init = float(os.getenv("MUZERO_PB_C_INIT", "1.25"))
        self.reanalyze_every_steps = int(os.getenv("MUZERO_REANALYZE_EVERY_STEPS", "500"))
        self.reanalyze_max_games = int(os.getenv("MUZERO_REANALYZE_MAX_GAMES", "16"))
        self.reanalyze_max_positions_per_game = int(
            os.getenv("MUZERO_REANALYZE_MAX_POSITIONS_PER_GAME", "24")
        )
        self.reanalyze_num_simulations = int(
            os.getenv("MUZERO_REANALYZE_NUM_SIMULATIONS", "16")
        )

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
        self.daily_stretch_target_pct = float(
            os.getenv("MUZERO_DAILY_STRETCH_TARGET_PCT", "10.0")
        )
        self.daily_stretch_max_drawdown_pct = float(
            os.getenv("MUZERO_DAILY_STRETCH_MAX_DRAWDOWN_PCT", "3.5")
        )
        self.daily_stretch_reward_bonus = float(
            os.getenv("MUZERO_DAILY_STRETCH_REWARD_BONUS", "4.0")
        )

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
