"""Tests des correctifs de signal `policy` pour MuZero."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eva_lab.muzero.environment import BUY, CLOSE, HOLD, SELL, SPLIT, TradingEnvironment
from eva_lab.muzero.replay_buffer import GameHistory, PrioritizedReplayBuffer
from eva_lab.training_utils import resolve_position_mechanics_profile


class MuZeroPolicySignalTests(unittest.TestCase):
    """Verifie les garde-fous autour du signal `policy` MuZero."""

    @staticmethod
    def _require_jax() -> None:
        """Ignore proprement un test si JAX est absent."""

        if importlib.util.find_spec("jax") is None:
            raise unittest.SkipTest("JAX indisponible sur cet environnement.")

    @staticmethod
    def _require_jax_stack() -> None:
        """Ignore proprement un test si la stack MuZero JAX est absente."""

        if importlib.util.find_spec("jax") is None or importlib.util.find_spec("haiku") is None:
            raise unittest.SkipTest("Stack JAX/Haiku indisponible sur cet environnement.")

    @staticmethod
    def _load_train_global_models_module():
        """Charge le module de training global sans dependre d'un package."""

        module_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "train_global_models.py"
        )
        spec = importlib.util.spec_from_file_location("test_train_global_models", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Impossible de charger train_global_models.py pour les tests.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _load_jax_trainer_module():
        """Charge le trainer JAX sans dependre de l'import package global."""

        module_path = (
            Path(__file__).resolve().parents[1]
            / "eva_lab"
            / "muzero"
            / "jax_trainer.py"
        )
        spec = importlib.util.spec_from_file_location("test_jax_trainer", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Impossible de charger jax_trainer.py pour les tests.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _build_scalp_mechanics_profile() -> dict[str, object]:
        """Construit un profil minimal pour tester les mecaniques scalp."""

        return {
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": False,
                "require_obv_confirmation": False,
                "allow_trend_fallback": True,
                "min_adx": 12.0,
                "trend_adx": 18.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 100,
                "stale_penalty": 0.0,
                "trend_penalty": 0.0,
                "range_penalty": 0.0,
                "drag_profit_floor": 0.0040,
                "drag_grace_steps": 0,
                "drag_penalty_cap": 1.25,
            },
            "pyramiding_policy": {
                "max_additions": 2,
                "min_profit_to_add": 0.0006,
                "reward_bonus": 0.16,
                "strong_trend_reward_bonus": 0.20,
            },
            "split_policy": {
                "max_splits": 3,
                "min_trade_return": 0.0038,
                "min_realized_pct": 0.025,
                "slbe_after_split": True,
                "failure_penalty": 0.85,
                "post_split_slbe_bonus": 0.60,
                "soft_partial_value_floor": 0.010,
            },
            "slbe_policy": {
                "activation_return": 0.0024,
                "bonus": 7.5,
                "exit_bonus": 1.75,
                "lock_profit_return": 0.0075,
                "lock_profit_buffer": 0.0010,
            },
            "close_policy": {
                "winner_threshold": 0.0055,
                "strong_winner_threshold": 0.0095,
                "tp_like_threshold": 0.0042,
                "reversal_close_bonus": 0.35,
                "early_profit_close_penalty": 0.20,
            },
            "activity_policy": {},
            "directional_policy": {},
            "reward_policy": {
                "realized_reward_multiplier": 1.0,
                "close_realized_bonus_multiplier": 1.0,
                "split_realized_bonus_multiplier": 1.0,
                "hold_drag_penalty_multiplier": 0.4,
                "pyramid_failure_penalty": 0.1,
                "pyramid_negative_exit_penalty": 0.0,
                "split_runner_profit_bonus": 0.45,
                "split_early_zone_penalty": 0.25,
                "split_decorative_penalty": 0.15,
                "pyramid_exit_capture_bonus": 0.35,
                "pyramid_bad_add_penalty": 0.35,
                "runner_protected_exit_bonus": 0.30,
                "soft_countertrend_ema_penalty": 0.0,
                "soft_countertrend_vwap_penalty": 0.0,
                "soft_low_adx_penalty": 0.0,
                "soft_obv_divergence_penalty": 0.0,
                "soft_trend_alignment_bonus": 0.0,
                "soft_strong_alignment_bonus": 0.0,
            },
        }

    def test_root_without_position_masks_split_and_close(self) -> None:
        """Masque `SPLIT` et `CLOSE` a la racine sans position ouverte."""

        env = TradingEnvironment(symbol="XAUUSD")
        observation, _ = env.reset()

        self.assertEqual(env.get_legal_root_actions(), [HOLD, BUY, SELL])
        self.assertEqual(
            TradingEnvironment.infer_legal_root_actions_from_observation(observation),
            [HOLD, BUY, SELL],
        )

    def test_root_with_position_allows_close_and_split(self) -> None:
        """Autorise `CLOSE` et `SPLIT` quand une position existe deja."""

        env = TradingEnvironment(symbol="XAUUSD")
        env.reset()
        env.position_size = env.spec.trade_size
        env.avg_entry_price = float(env.data[env.current_step, 3])
        observation = env._get_observation()

        self.assertEqual(
            TradingEnvironment.infer_legal_root_actions_from_observation(observation),
            [HOLD, BUY, SELL, SPLIT, CLOSE],
        )

    def test_root_policy_mask_removes_blocked_buy_but_keeps_hold(self) -> None:
        """Retire `BUY` du masque racine si l'entree est vetoee par le filtre."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 110.0
        data[:, 8] = 105.0
        data[:, 10] = -1.0
        data[:, 15] = 20.0

        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "strict",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            },
            "directional_policy": {},
        }
        env.reset()

        legal_actions = env.get_root_policy_actions()

        self.assertIn(HOLD, legal_actions)
        self.assertNotIn(BUY, legal_actions)
        self.assertIn(SELL, legal_actions)

    def test_training_root_mask_softens_vwap_veto_during_learning(self) -> None:
        """Conserve `BUY` en apprentissage si seul le veto VWAP est assoupli."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 95.0
        data[:, 8] = 105.0
        data[:, 10] = 1.0
        data[:, 15] = 20.0
        config = SimpleNamespace(
            quality_trade_bonus=10.0,
            final_growth_bonus=50.0,
            final_growth_threshold=0.10,
            drawdown_time_penalty_rate=0.2,
            max_drawdown_penalty=10.0,
            loss_penalty_multiplier=2.0,
            slbe_activation_bonus=6.0,
            horizon="scalp",
            primary_timeframe="M15",
            model_family=None,
            training_root_mask_soften_vwap=True,
            training_root_mask_soften_adx=False,
            training_root_mask_soften_ema200=False,
            training_root_mask_soften_directional=False,
            directional_curriculum_soft_end_step=8000,
            directional_curriculum_end_step=15000,
        )

        env = TradingEnvironment(
            data=data,
            symbol="XAUUSD",
            max_steps=4,
            config=config,
            training_mode=True,
            training_progress_step=6500,
        )
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            },
            "directional_policy": {},
        }
        env.reset()

        legal_actions = env.get_root_policy_actions()

        self.assertIn(BUY, legal_actions)
        self.assertIn(SELL, legal_actions)

    def test_observation_only_root_policy_mask_reconstructs_entry_veto(self) -> None:
        """Reconstruit le veto racine depuis l'observation seule."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 110.0
        data[:, 8] = 105.0
        data[:, 10] = -1.0
        data[:, 15] = 20.0

        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "strict",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            },
            "directional_policy": {},
        }
        observation, _ = env.reset()

        legal_actions = TradingEnvironment.infer_root_policy_actions_from_observation(
            observation,
            entry_filter=env.position_mechanics_profile["entry_filter"],
        )

        self.assertIn(HOLD, legal_actions)
        self.assertNotIn(BUY, legal_actions)
        self.assertIn(SELL, legal_actions)

    def test_runtime_family_uses_symbol_profile_when_config_is_mixed(self) -> None:
        """Utilise la famille du symbole si le run global est `mixed`."""

        config = SimpleNamespace(
            horizon="scalp",
            primary_timeframe="M5",
            model_family="mixed",
            quality_trade_bonus=10.0,
            final_growth_bonus=50.0,
            final_growth_threshold=0.10,
            drawdown_time_penalty_rate=0.2,
            max_drawdown_penalty=10.0,
            loss_penalty_multiplier=2.0,
            slbe_activation_bonus=6.0,
            daily_stretch_target_pct=10.0,
            daily_stretch_max_drawdown_pct=3.5,
            daily_stretch_reward_bonus=4.0,
            randomize_episode_start=True,
            episode_warmup_bars=100,
        )
        env = TradingEnvironment(symbol="XAUUSD", config=config, max_steps=4)

        self.assertEqual(env.family, "metals")
        self.assertEqual(env.position_mechanics_profile.get("family"), "metals")
        self.assertEqual(env.position_mechanics_profile.get("profile_name"), "scalp_metals_v2")

        runtime_filter = TradingEnvironment.build_runtime_entry_filter(
            horizon="scalp",
            symbol="XAUUSD",
            configured_family="mixed",
            training_mode=True,
            training_progress_step=9000,
            curriculum_soft_end_step=8000,
            curriculum_end_step=15000,
        )

        self.assertEqual(runtime_filter["ema_mode"], "relaxed")

    def test_root_policy_mask_moderate_ema_allows_buy_without_strong_countertrend(self) -> None:
        """N'applique plus `EMA200` en veto doux sans contre-tendance nette."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 105.0
        data[:, 8] = 101.0
        data[:, 10] = 0.25
        data[:, 15] = 20.0

        env = TradingEnvironment(data=data, symbol="EURUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": False,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            },
            "directional_policy": {},
        }
        env.reset()

        legal_actions = env.get_root_policy_actions()

        self.assertIn(BUY, legal_actions)
        self.assertEqual(env.root_mask_blocked_buy_ema200, 0)

    def test_root_policy_mask_moderate_ema_blocks_buy_on_strong_countertrend(self) -> None:
        """Conserve le veto `EMA200` si plusieurs signaux valident la contre-tendance."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 105.0
        data[:, 8] = 101.0
        data[:, 10] = -2.0
        data[:, 15] = 20.0
        data[:, 22] = 1.0

        env = TradingEnvironment(data=data, symbol="EURUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": False,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            },
            "directional_policy": {},
        }
        env.reset()

        legal_actions = env.get_root_policy_actions()

        self.assertNotIn(BUY, legal_actions)
        self.assertEqual(env.root_mask_blocked_buy_ema200, 1)

    def test_reward_policy_aliases_support_v2_keys(self) -> None:
        """Resolve correctement les cles V2 de recompense."""

        resolved = TradingEnvironment._resolve_reward_policy_terms(
            {
                "realized_reward_multiplier": 1.3,
                "close_realized_bonus_multiplier": 1.7,
                "split_realized_bonus_multiplier": 1.2,
                "hold_drag_penalty_multiplier": 0.4,
                "soft_countertrend_ema_penalty": 0.2,
                "soft_trend_alignment_bonus": 0.1,
            }
        )

        self.assertAlmostEqual(resolved["realized_reward_multiplier"], 1.3)
        self.assertAlmostEqual(resolved["close_realized_multiplier"], 1.7)
        self.assertAlmostEqual(resolved["split_realized_multiplier"], 1.2)
        self.assertAlmostEqual(resolved["hold_drag_multiplier"], 0.4)
        self.assertAlmostEqual(resolved["soft_countertrend_ema_penalty"], 0.2)
        self.assertAlmostEqual(resolved["soft_trend_alignment_bonus"], 0.1)

    def test_soft_entry_quality_penalizes_countertrend_without_veto(self) -> None:
        """Penalise une entree contre-tendance meme si elle reste autorisee."""

        env = TradingEnvironment(symbol="EURUSD", max_steps=4)
        env.horizon = "scalp"
        adjustment = env._compute_soft_entry_quality_adjustment(
            BUY,
            {
                "close": 100.0,
                "ema_200": 105.0,
                "ema_gap_pct": -0.03,
                "price_vs_vwap": -0.01,
                "obv_slope": -1.0,
                "adx": 8.0,
                "atr_pct": 0.01,
                "momentum": -0.02,
            },
            {
                "min_adx": 12.0,
                "trend_adx": 18.0,
                "allow_trend_fallback": False,
            },
            {
                "soft_countertrend_ema_penalty": 0.2,
                "soft_countertrend_vwap_penalty": 0.1,
                "soft_low_adx_penalty": 0.08,
                "soft_obv_divergence_penalty": 0.02,
                "soft_trend_alignment_bonus": 0.0,
                "soft_strong_alignment_bonus": 0.0,
            },
        )

        self.assertLess(adjustment, 0.0)
        self.assertEqual(env.soft_penalty_ema200_count, 1)
        self.assertEqual(env.soft_penalty_vwap_count, 1)
        self.assertEqual(env.soft_penalty_adx_count, 1)
        self.assertEqual(env.soft_penalty_obv_count, 0)
        self.assertEqual(env.soft_entry_penalty_count, 1)

    def test_soft_entry_quality_rewards_aligned_setup(self) -> None:
        """Verse un bonus doux quand le setup est propre et aligne."""

        env = TradingEnvironment(symbol="EURUSD", max_steps=4)
        adjustment = env._compute_soft_entry_quality_adjustment(
            BUY,
            {
                "close": 105.0,
                "ema_200": 100.0,
                "ema_gap_pct": 0.02,
                "price_vs_vwap": 0.01,
                "obv_slope": 1.0,
                "adx": 22.0,
                "atr_pct": 0.01,
                "momentum": 0.03,
            },
            {
                "min_adx": 12.0,
                "trend_adx": 18.0,
                "allow_trend_fallback": False,
            },
            {
                "soft_countertrend_ema_penalty": 0.2,
                "soft_countertrend_vwap_penalty": 0.1,
                "soft_low_adx_penalty": 0.08,
                "soft_obv_divergence_penalty": 0.02,
                "soft_trend_alignment_bonus": 0.08,
                "soft_strong_alignment_bonus": 0.14,
            },
        )

        self.assertGreater(adjustment, 0.0)
        self.assertEqual(env.soft_entry_bonus_count, 1)

    def test_soft_entry_quality_uses_weaker_early_penalty_scale(self) -> None:
        """Reduit les penalites douces en debut de training `scalp`."""

        shaping_config = SimpleNamespace(
            soft_reward_early_end_step=4000,
            soft_reward_mid_end_step=10000,
            soft_reward_penalty_scale_early=0.45,
            soft_reward_penalty_scale_mid=0.65,
            soft_reward_penalty_scale_late=0.85,
        )
        early_env = TradingEnvironment(symbol="EURUSD", max_steps=4)
        early_env.horizon = "scalp"
        early_env.training_progress_step = 1000
        early_env.config = shaping_config

        late_env = TradingEnvironment(symbol="EURUSD", max_steps=4)
        late_env.horizon = "scalp"
        late_env.training_progress_step = 12000
        late_env.config = shaping_config

        market_view = {
            "close": 100.0,
            "ema_200": 105.0,
            "ema_gap_pct": -0.03,
            "price_vs_vwap": -0.01,
            "obv_slope": -1.0,
            "adx": 8.0,
            "atr_pct": 0.01,
            "momentum": -0.02,
        }
        entry_filter = {
            "min_adx": 12.0,
            "trend_adx": 18.0,
            "allow_trend_fallback": False,
        }
        reward_terms = {
            "soft_countertrend_ema_penalty": 0.2,
            "soft_countertrend_vwap_penalty": 0.1,
            "soft_low_adx_penalty": 0.08,
            "soft_obv_divergence_penalty": 0.02,
            "soft_trend_alignment_bonus": 0.0,
            "soft_strong_alignment_bonus": 0.0,
        }

        early_adjustment = early_env._compute_soft_entry_quality_adjustment(
            BUY,
            market_view,
            entry_filter,
            reward_terms,
        )
        late_adjustment = late_env._compute_soft_entry_quality_adjustment(
            BUY,
            market_view,
            entry_filter,
            reward_terms,
        )

        self.assertLess(early_adjustment, 0.0)
        self.assertLess(late_adjustment, 0.0)
        self.assertLess(abs(early_adjustment), abs(late_adjustment))

    def test_soft_entry_quality_uses_stronger_early_bonus_scale(self) -> None:
        """Renforce les bonus doux au debut du training `scalp`."""

        shaping_config = SimpleNamespace(
            soft_reward_early_end_step=4000,
            soft_reward_mid_end_step=10000,
            soft_reward_bonus_scale_early=1.15,
            soft_reward_bonus_scale_mid=1.00,
            soft_reward_bonus_scale_late=0.95,
        )
        early_env = TradingEnvironment(symbol="EURUSD", max_steps=4)
        early_env.horizon = "scalp"
        early_env.training_progress_step = 1000
        early_env.config = shaping_config

        late_env = TradingEnvironment(symbol="EURUSD", max_steps=4)
        late_env.horizon = "scalp"
        late_env.training_progress_step = 12000
        late_env.config = shaping_config

        market_view = {
            "close": 105.0,
            "ema_200": 100.0,
            "ema_gap_pct": 0.02,
            "price_vs_vwap": 0.01,
            "obv_slope": 1.0,
            "adx": 22.0,
            "atr_pct": 0.01,
            "momentum": 0.03,
        }
        entry_filter = {
            "min_adx": 12.0,
            "trend_adx": 18.0,
            "allow_trend_fallback": False,
        }
        reward_terms = {
            "soft_countertrend_ema_penalty": 0.2,
            "soft_countertrend_vwap_penalty": 0.1,
            "soft_low_adx_penalty": 0.08,
            "soft_obv_divergence_penalty": 0.02,
            "soft_trend_alignment_bonus": 0.08,
            "soft_strong_alignment_bonus": 0.14,
        }

        early_adjustment = early_env._compute_soft_entry_quality_adjustment(
            BUY,
            market_view,
            entry_filter,
            reward_terms,
        )
        late_adjustment = late_env._compute_soft_entry_quality_adjustment(
            BUY,
            market_view,
            entry_filter,
            reward_terms,
        )

        self.assertGreater(early_adjustment, 0.0)
        self.assertGreater(late_adjustment, 0.0)
        self.assertGreater(early_adjustment, late_adjustment)

    def test_v5_exit_shaping_is_weaker_early_for_hold_drag(self) -> None:
        """Reduit la penalite `hold_drag` en debut de curriculum V5."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.2
        data[:, 2] = 100.8
        data[:, 3] = 101.0
        shaping_config = SimpleNamespace(
            exit_reward_early_end_step=4000,
            exit_reward_mid_end_step=10000,
            exit_reward_scale_early=0.25,
            exit_reward_scale_mid=0.60,
            exit_reward_scale_late=1.00,
        )

        def build_env(training_progress_step: int) -> TradingEnvironment:
            env = TradingEnvironment(data=data.copy(), symbol="XAUUSD", max_steps=4)
            env.position_mechanics_profile = self._build_scalp_mechanics_profile()
            env.config = shaping_config
            env.training_progress_step = training_progress_step
            env.reset()
            env.current_step = 0
            env.start_step = 0
            env.position_size = env.spec.trade_size
            env.avg_entry_price = 100.0
            env.slbe_active = True
            env.slbe_price = 100.0
            env._get_market_context = lambda: {
                "close": 101.0,
                "ema_200": 100.0,
                "ema_gap_pct": 0.01,
                "vwap": 102.0,
                "price_vs_vwap": -0.01,
                "obv": 1.0,
                "obv_slope": -1.0,
                "obv_divergence": 1.0,
                "adx": 10.0,
                "atr": 0.5,
                "atr_pct": 0.01,
                "bb_width_proxy": 0.0,
                "momentum": -0.03,
            }
            return env

        early_env = build_env(training_progress_step=1000)
        late_env = build_env(training_progress_step=12000)

        _early_obs, early_reward, _early_done, _early_truncated, _early_info = early_env.step(HOLD)
        _late_obs, late_reward, _late_done, _late_truncated, _late_info = late_env.step(HOLD)

        self.assertLess(early_reward, 0.0)
        self.assertLess(late_reward, 0.0)
        self.assertLess(abs(float(early_reward)), abs(float(late_reward)))

    def test_directional_policy_aliases_support_v2_keys(self) -> None:
        """Lit bien `max_directional_imbalance` depuis les profils V2."""

        resolved = TradingEnvironment._resolve_directional_policy_terms(
            {
                "min_entry_share": 0.2,
                "max_directional_imbalance": 0.6,
                "imbalance_penalty": 8.0,
            }
        )

        self.assertAlmostEqual(resolved["min_entry_share"], 0.2)
        self.assertAlmostEqual(resolved["max_directional_imbalance"], 0.6)
        self.assertAlmostEqual(resolved["imbalance_penalty"], 8.0)

    def test_hold_drag_ignores_hold_without_management_opportunity(self) -> None:
        """N'incremente pas `hold_drag` sans vrai signal de gestion."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.2
        data[:, 2] = 100.8
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.0,
            "ema_gap_pct": 0.01,
            "vwap": 100.8,
            "price_vs_vwap": 0.002,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 24.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.02,
        }

        _, reward, _, _, info = env.step(HOLD)

        self.assertEqual(env.hold_drag_opportunity_count, 0)
        self.assertEqual(env.hold_drag_penalized_count, 0)
        self.assertAlmostEqual(float(info["hold_drag_score"]), 0.0, places=6)
        self.assertGreaterEqual(reward, -0.5)

    def test_hold_drag_penalizes_passive_hold_on_reversal_signal(self) -> None:
        """Penalise `HOLD` si un trade gagnant ignore un signal de sortie."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.2
        data[:, 2] = 100.8
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.slbe_active = True
        env.slbe_price = 100.0
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.0,
            "ema_gap_pct": 0.01,
            "vwap": 102.0,
            "price_vs_vwap": -0.01,
            "obv": 1.0,
            "obv_slope": -1.0,
            "obv_divergence": 1.0,
            "adx": 10.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": -0.03,
        }

        _, reward, _, _, info = env.step(HOLD)

        self.assertEqual(env.hold_drag_opportunity_count, 1)
        self.assertEqual(env.hold_drag_penalized_count, 1)
        self.assertEqual(env.tp_like_missed_count, 1)
        self.assertAlmostEqual(float(info["hold_drag_score"]), 1.0, places=6)
        self.assertLess(reward, 0.0)

    def test_close_defensif_improves_close_quality(self) -> None:
        """Compte un `CLOSE` defensif propre comme une sortie de qualite."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 100.8
        data[:, 1] = 101.0
        data[:, 2] = 100.5
        data[:, 3] = 100.8
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env._get_market_context = lambda: {
            "close": 100.8,
            "ema_200": 100.0,
            "ema_gap_pct": 0.008,
            "vwap": 101.4,
            "price_vs_vwap": -0.006,
            "obv": 1.0,
            "obv_slope": -1.0,
            "obv_divergence": 1.0,
            "adx": 9.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": -0.02,
        }

        _, reward, _, _, info = env.step(CLOSE)

        self.assertEqual(env.defensive_close_count, 1)
        self.assertEqual(env.close_winner_count, 1)
        self.assertEqual(env.close_loser_count, 0)
        self.assertGreater(float(info["close_quality_score"]), 0.0)
        self.assertGreater(reward, 0.0)

    def test_slbe_lock_profit_phase_updates_stop(self) -> None:
        """Active la phase 2 du `SLBE` quand le trade depasse le seuil de verrouillage."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.2
        data[:, 2] = 100.8
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.slbe_active = True
        env.slbe_price = 100.0
        env.slbe_profit_locked = False
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.0,
            "ema_gap_pct": 0.01,
            "vwap": 100.6,
            "price_vs_vwap": 0.004,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 24.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.03,
        }

        env.step(HOLD)

        self.assertTrue(env.slbe_profit_locked)
        self.assertEqual(env.slbe_lock_profit_count, 1)
        self.assertGreater(env.slbe_price, 100.0)

    def test_hard_stop_forces_exit_and_counts_bad_stop(self) -> None:
        """Force une sortie immediate quand le prix touche le stop implicite."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 99.2
        data[:, 1] = 99.4
        data[:, 2] = 98.8
        data[:, 3] = 99.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.hard_stop_price = 99.5
        env.soft_tp_price = 101.0
        env.full_tp_price = 102.0
        env.position_entry_step = 0
        env._get_market_context = lambda: {
            "close": 99.0,
            "ema_200": 100.0,
            "ema_gap_pct": -0.01,
            "vwap": 99.4,
            "price_vs_vwap": -0.004,
            "obv": -1.0,
            "obv_slope": -1.0,
            "obv_divergence": 1.0,
            "adx": 18.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": -0.02,
        }

        _, reward, _, _, info = env.step(HOLD)

        self.assertEqual(env.position_size, 0.0)
        self.assertEqual(env.hard_stop_exit_count, 1)
        self.assertEqual(env.close_loser_count, 1)
        self.assertEqual(info["requested_action"], "HOLD")
        self.assertEqual(info["final_action"], "HOLD")
        self.assertLess(reward, 0.0)

    def test_split_utile_runner_profitable_increments_v6_metrics(self) -> None:
        """Compte un split utile seulement si le runner ajoute de la valeur."""

        data = np.ones((10, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.2
        data[:, 2] = 100.8
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=6)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.soft_tp_price = 100.5
        env.full_tp_price = 101.4
        env.position_entry_step = 0
        env.slbe_active = True
        env.slbe_price = 100.0
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.0,
            "ema_gap_pct": 0.01,
            "vwap": 100.7,
            "price_vs_vwap": 0.003,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 24.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.03,
        }

        _, split_reward, _, _, split_info = env.step(SPLIT)

        self.assertGreater(split_reward, 0.0)
        self.assertTrue(env.runner_active)
        self.assertEqual(env.split_executed, 1)
        self.assertEqual(env.split_profitable_count, 1)
        self.assertEqual(env.split_tp_zone_opportunity_count, 1)
        self.assertEqual(env.split_monetization_window_count, 1)
        self.assertEqual(env.split_monetization_capture_count, 1)
        self.assertEqual(env.split_runner_profitable_count, 0)
        self.assertAlmostEqual(float(split_info["split_runner_capture_rate"]), 0.0, places=6)
        self.assertAlmostEqual(float(split_info["split_zone_capture_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(split_info["split_monetization_capture_rate"]), 1.0, places=6)

        _, close_reward, _, _, close_info = env.step(CLOSE)

        self.assertGreater(close_reward, 0.0)
        self.assertEqual(env.runner_managed_exit_count, 1)
        self.assertEqual(env.runner_exit_profitable_count, 1)
        self.assertEqual(env.split_runner_profitable_count, 1)
        self.assertGreater(env.split_trade_value_delta, 0.0)
        self.assertGreater(env.runner_retained_profit_pct, 0.0)
        self.assertEqual(env.runner_giveback_pct, 0.0)
        self.assertEqual(env.split_improved_total_trade_count, 1)
        self.assertAlmostEqual(float(close_info["split_runner_capture_rate"]), 1.0, places=6)
        self.assertGreater(float(close_info["split_trade_value_delta"]), 0.0)

    def test_split_premature_is_marked_early_and_non_profitable(self) -> None:
        """Marque un split trop precoce comme decoratif ou destructeur."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 100.2
        data[:, 1] = 100.3
        data[:, 2] = 100.1
        data[:, 3] = 100.2
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        profile = self._build_scalp_mechanics_profile()
        profile["split_policy"]["min_trade_return"] = 0.0005
        profile["split_policy"]["min_realized_pct"] = 0.05
        env.position_mechanics_profile = profile
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.slbe_active = True
        env.slbe_price = 100.0
        env._get_market_context = lambda: {
            "close": 100.2,
            "ema_200": 100.0,
            "ema_gap_pct": 0.002,
            "vwap": 100.15,
            "price_vs_vwap": 0.0005,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 13.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.005,
        }

        _, reward, _, _, info = env.step(SPLIT)

        self.assertLess(reward, 0.0)
        self.assertEqual(env.split_executed, 1)
        self.assertEqual(env.split_profitable_count, 0)
        self.assertGreaterEqual(env.split_early_count, 1)
        self.assertGreaterEqual(env.split_decorative_count + env.split_runner_failed_count, 1)
        self.assertAlmostEqual(float(info["split_runner_capture_rate"]), 0.0, places=6)

    def test_runner_giveback_metric_tracks_failed_runner_exit(self) -> None:
        """Cumule un `giveback` quand le runner rend le gain au marche."""

        env = TradingEnvironment(symbol="XAUUSD")
        env.reset()
        env.runner_active = True
        env.runner_protected = False

        realized_pct = env._register_runner_exit_outcome(
            realized_trade=-15.0,
            forced_exit=True,
        )

        self.assertLess(realized_pct, 0.0)
        self.assertEqual(env.split_runner_failed_count, 1)
        self.assertEqual(env.runner_forced_stop_count, 1)
        self.assertGreater(env.runner_giveback_pct, 0.0)
        self.assertEqual(env.runner_retained_profit_pct, 0.0)

    def test_runner_giveback_metric_tracks_positive_exit_below_peak(self) -> None:
        """Compte un `giveback` meme si le runner finit encore legerement gagnant."""

        env = TradingEnvironment(symbol="XAUUSD")
        env.reset()
        env.runner_active = True
        env.runner_protected = True

        realized_pct = env._register_runner_exit_outcome(
            realized_trade=8.0,
            forced_exit=False,
            runner_peak_profit_pct=0.30,
            runner_entry_profit_pct=0.10,
        )

        self.assertGreater(realized_pct, 0.0)
        self.assertEqual(env.split_runner_profitable_count, 1)
        self.assertGreater(env.runner_retained_profit_pct, 0.0)
        self.assertGreater(env.runner_giveback_pct, 0.0)
        self.assertLess(env._last_runner_retention_ratio, 1.0)

    def test_pyramid_good_add_and_profitable_exit_feed_v6_metrics(self) -> None:
        """Compte separement la qualite d'ajout et la capture a la sortie."""

        data = np.ones((10, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.3
        data[:, 2] = 100.9
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=6)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.0,
            "ema_gap_pct": 0.01,
            "vwap": 100.5,
            "price_vs_vwap": 0.005,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 25.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.03,
        }

        _, buy_reward, _, _, buy_info = env.step(BUY)

        self.assertGreater(buy_reward, 0.0)
        self.assertEqual(env.pyramids_opened, 1)
        self.assertEqual(env.pyramid_good_add_count, 1)
        self.assertEqual(env.pyramid_bad_add_count, 0)
        self.assertEqual(env.pyramid_monetization_window_count, 1)
        self.assertEqual(env.pyramid_monetization_capture_count, 1)
        self.assertEqual(env.pyramid_add_capture_count, 1)
        self.assertAlmostEqual(float(buy_info["pyramid_add_capture_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(buy_info["pyramid_monetization_capture_rate"]), 1.0, places=6)
        next_index = min(env.current_step + 1, len(env.data) - 1)
        env.data[next_index, 0] = 102.2
        env.data[next_index, 1] = 102.4
        env.data[next_index, 2] = 101.8
        env.data[next_index, 3] = 102.2
        env._get_market_context = lambda: {
            "close": 102.2,
            "ema_200": 100.3,
            "ema_gap_pct": 0.018,
            "vwap": 101.1,
            "price_vs_vwap": 0.010,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 27.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.04,
        }

        _, reward, _, _, info = env.step(CLOSE)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.pyramid_profitable_exit_count, 1)
        self.assertGreater(env.pyramid_total_trade_improvement_pct, 0.0)
        self.assertEqual(env.pyramid_failed_to_improve_count, 0)
        self.assertAlmostEqual(float(info["pyramid_entry_quality_score"]), 1.0, places=6)
        self.assertAlmostEqual(float(info["pyramid_exit_capture_rate"]), 1.0, places=6)
        self.assertGreater(float(info["pyramid_total_trade_improvement_pct"]), 0.0)

    def test_hold_runner_extension_window_counts_offensive_capture(self) -> None:
        """Compte un HOLD offensif quand le runner peut encore etre etendu."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.2
        data[:, 1] = 101.4
        data[:, 2] = 101.0
        data[:, 3] = 101.2
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.runner_active = True
        env.runner_protected = True
        env.soft_tp_hit_active = True
        env.full_tp_hit_active = False
        env._get_market_context = lambda: {
            "close": 101.2,
            "ema_200": 100.4,
            "ema_gap_pct": 0.008,
            "vwap": 100.8,
            "price_vs_vwap": 0.004,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 26.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.03,
        }

        _, reward, _, _, info = env.step(HOLD)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.runner_extension_opportunity_count, 1)
        self.assertEqual(env.runner_profit_hold_window_count, 1)
        self.assertEqual(env.runner_profit_hold_capture_count, 1)
        self.assertEqual(env.runner_missed_extension_count, 0)
        self.assertAlmostEqual(float(info["runner_extension_capture_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(info["runner_profit_hold_capture_rate"]), 1.0, places=6)

    def test_split_tp_zone_opportunity_tolere_un_adx_en_repli_apres_soft_tp(self) -> None:
        """Active le split offensif apres `soft_tp` sans exiger un ADX encore fort."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.1
        data[:, 2] = 100.9
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.soft_tp_price = 100.5
        env.full_tp_price = 101.4
        env.slbe_active = True
        env.slbe_price = 100.0
        env.soft_tp_hit_active = True
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.2,
            "ema_gap_pct": 0.006,
            "vwap": 100.85,
            "price_vs_vwap": 0.0006,
            "obv": 1.0,
            "obv_slope": 0.6,
            "obv_divergence": 0.0,
            "adx": 9.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.0010,
        }

        _, reward, _, _, info = env.step(SPLIT)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.split_tp_zone_opportunity_count, 1)
        self.assertEqual(env.split_monetization_window_count, 1)
        self.assertAlmostEqual(float(info["split_zone_capture_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(info["split_monetization_capture_rate"]), 1.0, places=6)

    def test_split_monetization_window_tolere_un_giveback_fort_si_le_trade_reste_vert(self) -> None:
        """Ouvre encore la fenetre de split si le trade reste positif apres un fort giveback."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 100.45
        data[:, 1] = 100.53
        data[:, 2] = 100.37
        data[:, 3] = 100.45
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.position_peak_return = 0.035
        env._get_market_context = lambda: {
            "close": 100.45,
            "ema_200": 100.05,
            "ema_gap_pct": 0.004,
            "vwap": 100.32,
            "price_vs_vwap": 0.0013,
            "obv": 1.0,
            "obv_slope": 0.4,
            "obv_divergence": 0.0,
            "adx": 19.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.001,
        }

        _, reward, _, _, info = env.step(SPLIT)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.split_tp_zone_opportunity_count, 1)
        self.assertEqual(env.split_monetization_window_count, 1)
        self.assertEqual(env.split_monetization_capture_count, 1)
        self.assertAlmostEqual(float(info["split_monetization_capture_rate"]), 1.0, places=6)

    def test_pyramid_add_opportunity_aligne_le_plancher_sur_le_vrai_seuil_d_ajout(self) -> None:
        """Compte un ajout offensif avant `winner_threshold` si le vrai seuil d'ajout est atteint."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 100.17
        data[:, 1] = 100.25
        data[:, 2] = 100.10
        data[:, 3] = 100.17
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env._get_market_context = lambda: {
            "close": 100.17,
            "ema_200": 100.3,
            "ema_gap_pct": 0.007,
            "vwap": 100.12,
            "price_vs_vwap": 0.0005,
            "obv": 1.0,
            "obv_slope": 0.5,
            "obv_divergence": 0.0,
            "adx": 19.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.010,
        }

        _, reward, _, _, info = env.step(BUY)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.pyramid_good_add_count, 1)
        self.assertEqual(env.pyramid_add_opportunity_count, 1)
        self.assertEqual(env.pyramid_monetization_window_count, 1)
        self.assertEqual(env.pyramid_add_capture_count, 1)
        self.assertAlmostEqual(float(info["pyramid_add_capture_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(info["pyramid_monetization_capture_rate"]), 1.0, places=6)

    def test_runner_extension_opportunity_tolere_un_runner_protege_en_continuation(self) -> None:
        """Compte une extension de runner protege meme si l'ADX a deja commence a se detendre."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.15
        data[:, 1] = 101.25
        data[:, 2] = 101.0
        data[:, 3] = 101.15
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.runner_active = True
        env.runner_protected = True
        env.soft_tp_hit_active = True
        env._get_market_context = lambda: {
            "close": 101.15,
            "ema_200": 100.25,
            "ema_gap_pct": 0.007,
            "vwap": 101.05,
            "price_vs_vwap": 0.0004,
            "obv": 1.0,
            "obv_slope": 0.5,
            "obv_divergence": 0.0,
            "adx": 9.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.0010,
        }

        _, reward, _, _, info = env.step(HOLD)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.runner_extension_opportunity_count, 1)
        self.assertEqual(env.runner_profit_hold_window_count, 1)
        self.assertEqual(env.runner_missed_extension_count, 0)
        self.assertAlmostEqual(float(info["runner_extension_capture_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(info["runner_profit_hold_capture_rate"]), 1.0, places=6)

    def test_runner_hold_window_s_ouvre_des_l_approche_soft_tp_apres_split(self) -> None:
        """Active le runner avant le hit strict du `soft_tp` si le trade reste sain apres split."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 100.38
        data[:, 1] = 100.46
        data[:, 2] = 100.30
        data[:, 3] = 100.38
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.split_count = 1
        env.runner_active = True
        env.runner_protected = True
        env.slbe_active = True
        env.slbe_price = 100.0
        env.position_peak_return = 0.0041
        env._get_market_context = lambda: {
            "close": 100.38,
            "ema_200": 100.05,
            "ema_gap_pct": 0.006,
            "vwap": 100.34,
            "price_vs_vwap": 0.0004,
            "obv": 1.0,
            "obv_slope": 0.4,
            "obv_divergence": 0.0,
            "adx": 10.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.0010,
        }

        _, reward, _, _, info = env.step(HOLD)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.runner_viable_window_count, 1)
        self.assertEqual(env.runner_profit_hold_window_count, 1)
        self.assertEqual(env.runner_hold_after_soft_tp_count, 1)
        self.assertEqual(env.runner_profit_hold_capture_count, 1)
        self.assertGreater(float(info["runner_retained_profit_score"]), 0.0)

    def test_close_total_trop_tot_apres_soft_tp_compte_un_runner_manque(self) -> None:
        """Compte une fermeture totale trop precoce quand un runner viable existait encore."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.1
        data[:, 2] = 100.9
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.soft_tp_hit_active = True
        env.position_peak_return = 0.0102
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.2,
            "ema_gap_pct": 0.006,
            "vwap": 100.85,
            "price_vs_vwap": 0.0012,
            "obv": 1.0,
            "obv_slope": 0.5,
            "obv_divergence": 0.0,
            "adx": 18.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.002,
        }

        _, reward, _, _, info = env.step(CLOSE)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.runner_viable_window_count, 1)
        self.assertEqual(env.runner_viable_but_closed_count, 1)
        self.assertEqual(env.early_full_close_after_soft_tp_count, 1)
        self.assertEqual(float(info["runner_viable_but_closed_count"]), 1.0)
        self.assertEqual(float(info["early_full_close_after_soft_tp_count"]), 1.0)

    def test_close_total_trop_tot_compte_un_runner_manque_meme_sans_soft_tp(self) -> None:
        """Penalise encore une sortie totale si le runner reste viable hors zone `soft_tp`."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 100.34
        data[:, 1] = 100.42
        data[:, 2] = 100.26
        data[:, 3] = 100.34
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        profile = self._build_scalp_mechanics_profile()
        profile["close_policy"] = dict(profile["close_policy"])
        profile["close_policy"]["winner_threshold"] = 0.0030
        env.position_mechanics_profile = profile
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.position_peak_return = 0.0090
        env._get_market_context = lambda: {
            "close": 100.34,
            "ema_200": 100.12,
            "ema_gap_pct": 0.004,
            "vwap": 100.30,
            "price_vs_vwap": 0.0004,
            "obv": 1.0,
            "obv_slope": 0.4,
            "obv_divergence": 0.0,
            "adx": 19.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.001,
        }

        _, reward, _, _, info = env.step(CLOSE)

        self.assertGreater(reward, 0.0)
        self.assertEqual(env.runner_viable_window_count, 1)
        self.assertEqual(env.runner_viable_but_closed_count, 1)
        self.assertEqual(env.early_full_close_after_soft_tp_count, 1)
        self.assertEqual(float(info["runner_viable_but_closed_count"]), 1.0)
        self.assertEqual(float(info["early_full_close_after_soft_tp_count"]), 1.0)

    def test_split_window_manquee_est_penalisee_apres_trois_pas_inactifs(self) -> None:
        """Applique une penalite si une fenetre de split reste ouverte puis se ferme sans action."""

        data = np.ones((8, 26), dtype=np.float32)
        closes = (100.55, 100.55, 100.55, 100.55, 100.30, 100.30, 100.30, 100.30)
        for index, close in enumerate(closes):
            data[index, 0] = close
            data[index, 1] = close + 0.08
            data[index, 2] = close - 0.08
            data[index, 3] = close
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=6)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 0
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.soft_tp_price = 100.5
        env.full_tp_price = 101.4
        env.position_entry_step = 0
        env.time_stop_steps = 100
        env._get_market_context = lambda: {
            "close": float(env.data[min(env.current_step, len(env.data) - 1), 3]),
            "ema_200": 100.0,
            "ema_gap_pct": 0.01,
            "vwap": 100.20,
            "price_vs_vwap": 0.003,
            "obv": 1.0,
            "obv_slope": 1.0,
            "obv_divergence": 0.0,
            "adx": 22.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": 0.02,
        }

        env.step(HOLD)
        env.step(HOLD)
        env.step(HOLD)
        env.step(HOLD)
        env.step(HOLD)

        self.assertEqual(env.split_missed_window_count, 1)

    def test_hold_after_soft_tp_and_time_stop_is_penalized(self) -> None:
        """Penalise `HOLD` si le trade stagne apres zone TP et `time stop`."""

        data = np.ones((8, 26), dtype=np.float32)
        data[:, 0] = 101.0
        data[:, 1] = 101.2
        data[:, 2] = 100.8
        data[:, 3] = 101.0
        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = self._build_scalp_mechanics_profile()
        env.reset()
        env.current_step = 5
        env.start_step = 0
        env.position_size = env.spec.trade_size
        env.avg_entry_price = 100.0
        env.soft_tp_price = 100.5
        env.full_tp_price = 101.5
        env.time_stop_steps = 3
        env.position_entry_step = 0
        env.soft_tp_hit_active = True
        env.slbe_active = True
        env.slbe_price = 100.0
        env.slbe_profit_locked = True
        env._get_market_context = lambda: {
            "close": 101.0,
            "ema_200": 100.0,
            "ema_gap_pct": 0.01,
            "vwap": 101.8,
            "price_vs_vwap": -0.008,
            "obv": 1.0,
            "obv_slope": -1.0,
            "obv_divergence": 1.0,
            "adx": 7.0,
            "atr": 0.5,
            "atr_pct": 0.01,
            "bb_width_proxy": 0.0,
            "momentum": -0.02,
        }

        _, reward, _, _, _ = env.step(HOLD)

        self.assertEqual(env.time_stop_trigger_count, 1)
        self.assertLess(reward, 0.0)

    def test_directional_entry_feedback_penalizes_extreme_one_sided_flow(self) -> None:
        """Penalise un flux d'entrees trop unilateral avant la fin d'episode."""

        env = TradingEnvironment(symbol="XAUUSD")
        env.long_entries = 0
        env.short_entries = 4

        penalty = env._compute_directional_entry_feedback(
            {
                "min_entry_share": 0.2,
                "max_directional_imbalance": 0.6,
                "imbalance_penalty": 8.0,
                "entry_penalty_scale": 0.5,
            }
        )

        self.assertLess(penalty, 0.0)

    def test_vetoed_buy_is_converted_to_hold(self) -> None:
        """Transforme un achat vetoe en `HOLD` sans ouvrir de position."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 110.0
        data[:, 8] = 105.0
        data[:, 9] = 10.0
        data[:, 10] = -1.0
        data[:, 15] = 20.0

        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "strict",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            }
        }
        env.reset()

        _, reward, _, _, info = env.step(BUY)

        self.assertEqual(info["requested_action"], "BUY")
        self.assertEqual(info["final_action"], "HOLD")
        self.assertEqual(env.position_size, 0.0)
        self.assertEqual(env.action_counts["BUY"], 0)
        self.assertEqual(env.action_counts["HOLD"], 1)
        self.assertEqual(env.blocked_buy_entries, 1)
        self.assertEqual(env.entry_veto_to_hold, 1)
        self.assertLess(reward, 0.0)

    def test_vetoed_sell_is_converted_to_hold(self) -> None:
        """Transforme une vente vetoee en `HOLD` sans ouvrir de position."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 90.0
        data[:, 8] = 95.0
        data[:, 9] = 10.0
        data[:, 10] = 1.0
        data[:, 15] = 20.0

        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "strict",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            }
        }
        env.reset()

        _, reward, _, _, info = env.step(SELL)

        self.assertEqual(info["requested_action"], "SELL")
        self.assertEqual(info["final_action"], "HOLD")
        self.assertEqual(env.position_size, 0.0)
        self.assertEqual(env.action_counts["SELL"], 0)
        self.assertEqual(env.action_counts["HOLD"], 1)
        self.assertEqual(env.blocked_sell_entries, 1)
        self.assertEqual(env.entry_veto_to_hold, 1)
        self.assertLess(reward, 0.0)

    def test_directional_hard_veto_converts_sell_to_hold_after_four_entries(self) -> None:
        """Bloque une entree dominante quand l'episode depasse le seuil dur."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 90.0
        data[:, 8] = 95.0
        data[:, 9] = -1.0
        data[:, 10] = -1.0
        data[:, 15] = 25.0

        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "relaxed",
                "require_vwap_alignment": False,
                "require_obv_confirmation": False,
                "allow_trend_fallback": True,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            },
            "directional_policy": {
                "hard_veto_after_entries": 4,
                "hard_veto_max_share": 0.80,
                "min_entry_share": 0.20,
                "max_directional_imbalance": 0.40,
                "imbalance_penalty": 8.0,
            },
        }
        env.reset()
        env.short_entries = 4

        _, reward, _, _, info = env.step(SELL)

        self.assertEqual(info["requested_action"], "SELL")
        self.assertEqual(info["final_action"], "HOLD")
        self.assertEqual(env.blocked_sell_directional, 1)
        self.assertEqual(env.blocked_sell_entries, 1)
        self.assertEqual(env.entry_veto_to_hold, 1)
        self.assertLess(reward, 0.0)

    def test_root_policy_mask_blocks_dominant_side_after_four_entries(self) -> None:
        """Retire l'action dominante du masque racine apres quatre entrees."""

        data = np.zeros((8, 26), dtype=np.float32)
        data[:, 0] = 100.0
        data[:, 1] = 101.0
        data[:, 2] = 99.0
        data[:, 3] = 100.0
        data[:, 4] = 1000.0
        data[:, 5] = 90.0
        data[:, 8] = 95.0
        data[:, 10] = -1.0
        data[:, 15] = 25.0

        env = TradingEnvironment(data=data, symbol="XAUUSD", max_steps=4)
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "relaxed",
                "require_vwap_alignment": False,
                "require_obv_confirmation": False,
                "allow_trend_fallback": True,
                "min_adx": 5.0,
                "trend_adx": 10.0,
            },
            "directional_policy": {
                "hard_veto_after_entries": 4,
                "hard_veto_max_share": 0.80,
                "min_entry_share": 0.20,
                "max_directional_imbalance": 0.40,
                "imbalance_penalty": 8.0,
            },
        }
        env.reset()
        env.short_entries = 4

        legal_actions = env.get_root_policy_actions()

        self.assertNotIn(SELL, legal_actions)
        self.assertIn(HOLD, legal_actions)
        self.assertEqual(env.root_mask_blocked_sell_directional, 1)

    def test_rebalance_bonus_is_paid_once_for_missing_side(self) -> None:
        """N'accorde le bonus de reequilibrage qu'a la premiere entree utile."""

        env = TradingEnvironment(symbol="XAUUSD")
        env.short_entries = 3
        env.long_entries = 0

        policy = {
            "min_entry_share": 0.20,
            "max_directional_imbalance": 0.40,
            "final_max_directional_imbalance": 0.40,
            "hard_veto_after_entries": 4,
            "hard_veto_max_share": 0.80,
            "rebalance_bonus": 0.35,
        }

        first_bonus = env._compute_rebalance_bonus(BUY, policy)
        second_bonus = env._compute_rebalance_bonus(BUY, policy)

        self.assertAlmostEqual(first_bonus, 0.35)
        self.assertEqual(second_bonus, 0.0)

    def test_scalp_curriculum_relaxes_filters_longer_after_8000(self) -> None:
        """Garde un filtre `scalp` assoupli au-dela de `8000` steps."""

        env = TradingEnvironment(
            symbol="XAUUSD",
            training_mode=True,
            training_progress_step=9500,
        )
        env.config = SimpleNamespace(
            horizon="scalp",
            directional_curriculum_soft_end_step=8000,
            directional_curriculum_end_step=15000,
        )
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": True,
                "require_obv_confirmation": True,
                "allow_trend_fallback": False,
                "min_adx": 13.5,
                "trend_adx": 18.5,
            }
        }

        active_filter = env._get_active_entry_filter()

        self.assertFalse(active_filter["require_vwap_alignment"])
        self.assertFalse(active_filter["require_obv_confirmation"])
        self.assertTrue(active_filter["allow_trend_fallback"])
        self.assertAlmostEqual(float(active_filter["min_adx"]), 10.5)
        self.assertAlmostEqual(float(active_filter["trend_adx"]), 15.5)

    def test_scalp_curriculum_uses_stronger_relaxation_early(self) -> None:
        """Assouplit davantage `ADX` en debut de run `scalp`."""

        env = TradingEnvironment(
            symbol="XAUUSD",
            training_mode=True,
            training_progress_step=3000,
        )
        env.config = SimpleNamespace(
            horizon="scalp",
            directional_curriculum_soft_end_step=8000,
            directional_curriculum_end_step=15000,
        )
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": True,
                "require_obv_confirmation": True,
                "allow_trend_fallback": False,
                "min_adx": 13.5,
                "trend_adx": 18.5,
            }
        }

        active_filter = env._get_active_entry_filter()

        self.assertFalse(active_filter["require_vwap_alignment"])
        self.assertFalse(active_filter["require_obv_confirmation"])
        self.assertTrue(active_filter["allow_trend_fallback"])
        self.assertAlmostEqual(float(active_filter["min_adx"]), 8.5)
        self.assertAlmostEqual(float(active_filter["trend_adx"]), 13.5)

    def test_scalp_curriculum_restores_family_filter_after_15000(self) -> None:
        """Revient au filtre famille apres la fin du curriculum `scalp`."""

        env = TradingEnvironment(
            symbol="XAUUSD",
            training_mode=True,
            training_progress_step=16000,
        )
        env.config = SimpleNamespace(
            horizon="scalp",
            directional_curriculum_soft_end_step=8000,
            directional_curriculum_end_step=15000,
        )
        env.position_mechanics_profile = {
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 13.5,
                "trend_adx": 18.5,
            }
        }

        active_filter = env._get_active_entry_filter()

        self.assertTrue(active_filter["require_vwap_alignment"])
        self.assertFalse(active_filter["require_obv_confirmation"])
        self.assertFalse(active_filter["allow_trend_fallback"])
        self.assertAlmostEqual(float(active_filter["min_adx"]), 13.5)
        self.assertAlmostEqual(float(active_filter["trend_adx"]), 18.5)

    def test_scalp_profiles_disable_obv_as_hard_gate(self) -> None:
        """Supprime `OBV` des gates durs pour toutes les familles `scalp`."""

        for family in ("fx", "indices", "metals", "crypto"):
            profile = resolve_position_mechanics_profile("scalp", family)
            self.assertFalse(profile["entry_filter"]["require_obv_confirmation"])

    def test_priority_update_changes_tree_weight(self) -> None:
        """Met a jour la priorite stockee dans la SumTree."""

        buffer = PrioritizedReplayBuffer(max_games=4)
        game = GameHistory()
        game.store(np.zeros(4, dtype=np.float32), HOLD, 0.0, np.ones(5, dtype=np.float32) / 5.0, 0.0)
        buffer.save_game(game)

        tree_idx = buffer.tree.capacity - 1
        previous_priority = float(buffer.tree.tree[tree_idx])
        buffer.update_priorities([tree_idx], [10.0])

        self.assertGreater(float(buffer.tree.tree[tree_idx]), previous_priority)

    def test_empty_game_is_not_persisted(self) -> None:
        """Ignore un episode vide pour eviter un replay incoherent."""

        buffer = PrioritizedReplayBuffer(max_games=4)
        buffer.save_game(GameHistory())

        self.assertEqual(buffer.size, 0)

    def test_recent_games_returns_newest_first(self) -> None:
        """Retourne les episodes les plus recents dans le bon ordre."""

        buffer = PrioritizedReplayBuffer(max_games=5)
        for index in range(3):
            game = GameHistory()
            game.store(
                np.array([index], dtype=np.float32),
                HOLD,
                0.0,
                np.ones(5, dtype=np.float32) / 5.0,
                float(index),
            )
            buffer.save_game(game)

        recent = buffer.recent_games(2)

        self.assertEqual(len(recent), 2)
        self.assertEqual(float(recent[0].values[0]), 2.0)
        self.assertEqual(float(recent[1].values[0]), 1.0)

    def test_replay_sampling_limits_one_sided_buckets(self) -> None:
        """Respecte le cap de 35 % pour les episodes unilateraux."""

        random_seed = 42
        np.random.seed(random_seed)
        import random

        random.seed(random_seed)
        buffer = PrioritizedReplayBuffer(max_games=64)

        def _make_game(index: int, *, long_entries: int, short_entries: int, balanced: bool) -> GameHistory:
            game = GameHistory()
            game.store(
                np.array([index], dtype=np.float32),
                HOLD,
                0.0,
                np.ones(5, dtype=np.float32) / 5.0,
                float(index),
            )
            game.metadata.update(
                {
                    "long_entries": long_entries,
                    "short_entries": short_entries,
                    "long_present": long_entries > 0,
                    "short_present": short_entries > 0,
                    "balanced_episode": balanced,
                }
            )
            return game

        for index in range(10):
            buffer.save_game(_make_game(index, long_entries=0, short_entries=3, balanced=False))
        for index in range(10, 20):
            buffer.save_game(_make_game(index, long_entries=3, short_entries=0, balanced=False))
        for index in range(20, 32):
            buffer.save_game(_make_game(index, long_entries=2, short_entries=2, balanced=True))

        sampled = buffer.sample(batch_size=10)
        sampled_games = [game for game, _start_idx, _tree_idx in sampled]
        buy_only_count = 0
        sell_only_count = 0
        balanced_count = 0
        for game in sampled_games:
            long_present = bool(game.metadata.get("long_present"))
            short_present = bool(game.metadata.get("short_present"))
            if long_present and not short_present:
                buy_only_count += 1
            if short_present and not long_present:
                sell_only_count += 1
            if bool(game.metadata.get("balanced_episode")):
                balanced_count += 1

        self.assertLessEqual(buy_only_count, 3)
        self.assertLessEqual(sell_only_count, 3)
        self.assertGreaterEqual(balanced_count, 4)

    def test_replay_sampling_caps_symbol_concentration(self) -> None:
        """Empeche un symbole de depasser 25 % du batch si l'univers est large."""

        random_seed = 7
        np.random.seed(random_seed)
        import random

        random.seed(random_seed)
        buffer = PrioritizedReplayBuffer(max_games=64)
        symbols = ["XAUUSD", "US30.cash", "US100.cash", "BTCUSD"]

        for index in range(32):
            game = GameHistory()
            game.store(
                np.array([index], dtype=np.float32),
                HOLD,
                0.0,
                np.ones(5, dtype=np.float32) / 5.0,
                float(index),
            )
            symbol = symbols[index % len(symbols)]
            game.metadata.update(
                {
                    "symbol": symbol,
                    "long_entries": 2,
                    "short_entries": 2,
                    "long_present": True,
                    "short_present": True,
                    "balanced_episode": True,
                }
            )
            buffer.save_game(game)

        sampled = buffer.sample(batch_size=12)
        symbol_counts: dict[str, int] = {}
        for game, _start_idx, _tree_idx in sampled:
            symbol = str(game.metadata.get("symbol") or "")
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        self.assertTrue(symbol_counts)
        self.assertLessEqual(max(symbol_counts.values()), 3)

    def test_replay_diversity_stats_expose_root_mask_and_post_veto_rates(self) -> None:
        """Expose les nouvelles metriques de friction policy/environnement."""

        buffer = PrioritizedReplayBuffer(max_games=4)
        game = GameHistory()
        game.store(np.zeros(4, dtype=np.float32), HOLD, 0.0, np.ones(5, dtype=np.float32) / 5.0, 0.0)
        game.metadata.update(
            {
                "balanced_episode": True,
                "symbol": "XAUUSD",
                "long_present": True,
                "short_present": True,
                "long_entries": 3,
                "short_entries": 2,
                "requested_buy_actions": 5,
                "requested_sell_actions": 5,
                "entry_veto_to_hold": 1,
                "root_mask_directional_candidates_total": 10,
                "root_mask_blocked_buy_total": 2,
                "root_mask_blocked_sell_total": 1,
                "root_mask_blocked_buy_vwap": 1,
                "root_mask_blocked_sell_adx": 1,
                "root_mask_blocked_buy_directional": 1,
                "root_mask_ema200_share": 0.0,
                "root_mask_vwap_share": 1.0 / 3.0,
                "root_mask_adx_share": 1.0 / 3.0,
                "root_mask_directional_share": 1.0 / 3.0,
                "blocked_buy_entries": 1,
                "blocked_sell_entries": 0,
                "hold_drag_opportunity_count": 5,
                "hold_drag_penalized_count": 2,
                "split_executed": 4,
                "split_profitable_count": 2,
                "split_runner_profitable_count": 1,
                "split_runner_failed_count": 1,
                "split_early_count": 1,
                "split_decorative_count": 1,
                "split_trade_value_delta": 0.75,
                "split_improved_total_trade_count": 1,
                "split_opportunity_count": 6,
                "split_tp_zone_opportunity_count": 4,
                "pyramids_opened": 3,
                "pyramid_profitable_count": 2,
                "pyramid_good_add_count": 2,
                "pyramid_bad_add_count": 1,
                "pyramid_add_opportunity_count": 4,
                "pyramid_add_capture_count": 1,
                "pyramid_missed_add_count": 2,
                "pyramid_profitable_exit_count": 1,
                "pyramid_total_trade_improvement_pct": 1.5,
                "pyramid_failed_to_improve_count": 1,
                "pyramid_opportunity_count": 5,
                "slbe_triggered": 4,
                "slbe_profitable_exits": 2,
                "slbe_lock_profit_count": 1,
                "close_winner_count": 3,
                "close_loser_count": 1,
                "tp_like_missed_count": 2,
                "defensive_close_count": 1,
                "early_close_noise_count": 1,
                "runner_managed_exit_count": 2,
                "runner_exit_profitable_count": 1,
                "runner_forced_stop_count": 1,
                "runner_extension_opportunity_count": 3,
                "runner_extension_capture_rate": 2.0 / 3.0,
                "runner_profit_hold_window_count": 4,
                "runner_profit_hold_capture_rate": 0.5,
                "runner_missed_extension_count": 1,
                "runner_viable_window_count": 4,
                "runner_hold_after_soft_tp_count": 2,
                "runner_viable_but_closed_count": 1,
                "early_full_close_after_soft_tp_count": 1,
                "runner_retained_profit_pct": 0.75,
                "runner_retained_profit_score": 0.60,
                "runner_giveback_pct": 0.25,
            }
        )
        buffer.save_game(game)

        stats = buffer.diversity_stats()

        self.assertAlmostEqual(float(stats["root_mask_rate"]), 0.3, places=6)
        self.assertAlmostEqual(float(stats["post_veto_to_hold_rate"]), 0.1, places=6)
        self.assertAlmostEqual(float(stats["veto_to_hold_rate"]), 0.1, places=6)
        self.assertEqual(float(stats["root_mask_blocked_buy_total"]), 2.0)
        self.assertEqual(float(stats["root_mask_blocked_sell_total"]), 1.0)
        self.assertAlmostEqual(float(stats["root_mask_ema200_share"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["root_mask_vwap_share"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["root_mask_adx_share"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["root_mask_directional_share"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["hold_drag_score"]), 0.4, places=6)
        self.assertAlmostEqual(float(stats["split_efficiency"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["split_runner_capture_rate"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["split_zone_capture_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["split_trade_value_delta"]), 0.75 / 4.0, places=6)
        self.assertEqual(float(stats["split_improved_total_trade_count"]), 1.0)
        self.assertAlmostEqual(float(stats["pyramid_efficiency"]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["pyramid_entry_quality_score"]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["pyramid_exit_capture_rate"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["pyramid_add_capture_rate"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["pyramid_total_trade_improvement_pct"]), 1.5 / 3.0, places=6)
        self.assertEqual(float(stats["pyramid_failed_to_improve_count"]), 1.0)
        self.assertAlmostEqual(float(stats["slbe_capture_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["close_quality_score"]), 0.75, places=6)
        self.assertAlmostEqual(float(stats["runner_extension_capture_rate"]), 2.0 / 3.0, places=6)
        self.assertEqual(float(stats["runner_viable_window_count"]), 4.0)
        self.assertEqual(float(stats["runner_hold_after_soft_tp_count"]), 2.0)
        self.assertEqual(float(stats["runner_viable_but_closed_count"]), 1.0)
        self.assertEqual(float(stats["early_full_close_after_soft_tp_count"]), 1.0)
        self.assertAlmostEqual(float(stats["runner_retained_profit_pct"]), 0.75, places=6)
        self.assertAlmostEqual(float(stats["runner_retained_profit_score"]), 0.60, places=6)
        self.assertAlmostEqual(float(stats["runner_giveback_pct"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["runner_giveback_ratio"]), 0.25, places=6)
        self.assertEqual(int(stats["close_events_by_symbol"]["XAUUSD"]), 4)
        self.assertAlmostEqual(float(stats["close_quality_by_symbol"]["XAUUSD"]), 0.75, places=6)
        self.assertAlmostEqual(float(stats["split_efficiency_by_symbol"]["XAUUSD"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["split_runner_capture_rate_by_symbol"]["XAUUSD"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["root_mask_share_by_symbol"]["XAUUSD"]), 0.3, places=6)
        self.assertAlmostEqual(float(stats["root_mask_vwap_share_by_symbol"]["XAUUSD"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["root_mask_adx_share_by_symbol"]["XAUUSD"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["root_mask_directional_share_by_symbol"]["XAUUSD"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["pyramid_efficiency_by_symbol"]["XAUUSD"]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["pyramid_entry_quality_score_by_symbol"]["XAUUSD"]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["pyramid_exit_capture_rate_by_symbol"]["XAUUSD"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(stats["slbe_capture_rate_by_symbol"]["XAUUSD"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["hold_drag_score_by_symbol"]["XAUUSD"]), 0.4, places=6)
        self.assertAlmostEqual(float(stats["liquidity_trap_share"]), 0.0, places=6)

    def test_replay_diversity_stats_include_hard_negative_mix(self) -> None:
        """Expose le mix hard-negative derive des tags d'echec."""

        buffer = PrioritizedReplayBuffer(max_games=4)
        game = GameHistory()
        game.store(np.zeros(4, dtype=np.float32), HOLD, 0.0, np.ones(5, dtype=np.float32) / 5.0, 0.0)
        game.metadata.update(
            {
                "balanced_episode": True,
                "symbol": "US30.cash",
                "long_present": True,
                "short_present": True,
                "liquidity_trap_loss": True,
                "nemesis_type": "LIQUIDITY_TRAP",
            }
        )
        buffer.save_game(game)

        stats = buffer.diversity_stats()

        self.assertAlmostEqual(float(stats["liquidity_trap_share"]), 1.0, places=6)
        self.assertIn("LIQUIDITY_TRAP", stats["hard_negative_mix"])
        self.assertIn("LIQUIDITY_TRAP", stats["nemesis_mix"])

    def test_shadow_learning_builds_action_biased_policy(self) -> None:
        """Transforme les trades live en cible policy non uniforme."""

        from eva_lab.shadow_dataset import ACTION_MAP, build_game_from_shadow_episode

        game = build_game_from_shadow_episode(
            [
                {
                    "observation": {
                        "price": 100.0,
                        "latest_candle": {
                            "open": 99.5,
                            "high": 101.0,
                            "low": 99.0,
                            "close": 100.0,
                        },
                        "indicators": {},
                    },
                    "action": {"type": "CLOSE", "close_ratio": 0.70, "slbe": True},
                    "reward": 0.4,
                    "metadata": {"profit": 12.0, "episode_id": "shadow-positive"},
                }
            ],
            observation_size=26,
            action_space_size=5,
        )

        policy = np.asarray(game.policies[0], dtype=np.float32)
        uniform = np.ones(5, dtype=np.float32) / 5.0
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertFalse(np.allclose(policy, uniform))
        self.assertGreater(float(policy[ACTION_MAP["CLOSE"]]), float(uniform[ACTION_MAP["CLOSE"]]))
        self.assertEqual(game.metadata["shadow_policy_mode"], "action_biased")

    def test_offensive_curriculum_boosts_replay_priority(self) -> None:
        """Surpondere les episodes avec captures offensives utiles."""

        game = GameHistory()
        game.store(np.zeros(4, dtype=np.float32), HOLD, 0.0, np.ones(5, dtype=np.float32) / 5.0, 0.0)
        game.metadata.update(
            {
                "soft_tp_hit_count": 1,
                "split_monetization_capture_count": 2,
                "runner_profit_hold_capture_count": 1,
                "close_quality_score": 0.8,
                "profit_factor": 1.6,
            }
        )

        self.assertGreater(PrioritizedReplayBuffer._offensive_curriculum_multiplier(game), 1.0)

    def test_arena_score_penalizes_sell_heavy_candidate(self) -> None:
        """Degrade le score Arena d'un candidat unidirectionnel."""

        self._require_jax_stack()

        from eva_lab.arena import Arena

        balanced_metrics = {
            "return_pct": 1.0,
            "profit_factor": 1.3,
            "expectancy_pct": 0.02,
            "win_rate": 60.0,
            "positive_episode_rate": 55.0,
            "max_drawdown_pct": 2.0,
            "directional_bias": "balanced",
            "directional_imbalance": 0.10,
            "long_entries": 20,
            "short_entries": 18,
            "directional_by_symbol": {
                "XAUUSD": {"directional_bias": "balanced"},
            },
        }
        sell_heavy_metrics = {
            **balanced_metrics,
            "directional_bias": "sell_heavy",
            "directional_imbalance": 1.0,
            "long_entries": 0,
            "short_entries": 38,
            "directional_by_symbol": {
                "XAUUSD": {"directional_bias": "sell_heavy"},
                "US30.cash": {"directional_bias": "sell_heavy"},
            },
        }

        self.assertGreater(
            Arena._score_metrics(balanced_metrics),
            Arena._score_metrics(sell_heavy_metrics),
        )

    def test_arena_score_rewards_stronger_exit_mechanics(self) -> None:
        """Favorise les candidats avec meilleures mecaniques de sortie a perf brute egale."""

        self._require_jax_stack()

        from eva_lab.arena import Arena

        base_metrics = {
            "return_pct": 0.12,
            "profit_factor": 1.28,
            "expectancy_pct": 0.02,
            "win_rate": 33.0,
            "positive_episode_rate": 58.0,
            "max_drawdown_pct": 0.30,
            "directional_bias": "balanced",
            "directional_imbalance": 0.08,
            "long_entries": 20,
            "short_entries": 18,
            "directional_by_symbol": {"XAUUSD": {"directional_bias": "balanced"}},
        }
        weak_mechanics = {
            **base_metrics,
            "metrics_by_position_mechanics": {
                "close_quality_score": 0.20,
                "hold_drag_score": 0.90,
                "split_efficiency": 0.20,
                "split_executed": 4,
                "pyramid_efficiency": 0.15,
                "pyramids_opened": 4,
                "slbe_capture_rate": 0.10,
                "slbe_triggered": 4,
            },
        }
        strong_mechanics = {
            **base_metrics,
            "metrics_by_position_mechanics": {
                "close_quality_score": 0.72,
                "hold_drag_score": 0.12,
                "split_efficiency": 0.70,
                "split_executed": 4,
                "pyramid_efficiency": 0.65,
                "pyramids_opened": 4,
                "slbe_capture_rate": 0.75,
                "slbe_triggered": 4,
            },
        }

        self.assertGreater(
            Arena._score_metrics(strong_mechanics),
            Arena._score_metrics(weak_mechanics),
        )

    def test_arena_nemesis_validation_blocks_bad_close_slice(self) -> None:
        """Refuse un challenger qui perd une slice Nemesis critique."""

        self._require_jax_stack()

        from eva_lab.arena import Arena

        verdict = Arena._build_nemesis_validation(
            {
                "total_trades": 20,
                "profit_factor": 1.2,
                "metrics_by_position_mechanics": {
                    "close_quality_score": 0.10,
                    "hard_stop_exit_count": 0,
                },
            },
            {
                "evaluation_games": 12,
                "total_trades": 20,
                "profit_factor": 1.1,
                "metrics_by_position_mechanics": {
                    "close_quality_score": 0.80,
                    "hard_stop_exit_count": 0,
                },
            },
            {"profit_factor": 0.8},
        )

        self.assertFalse(verdict["allowed"])
        self.assertIn("bad_close", verdict["failed_slices"])

    def test_promotion_gate_blocks_missing_direction(self) -> None:
        """Refuse la promotion live si une direction est absente."""

        from eva_lab.champion_promoter import ChampionPromoter

        promoter = ChampionPromoter()
        battle_report = {
            "outcome": "VICTORY",
            "validation": {"sample_size_ok": True},
            "challenger": {
                "metrics": {
                    "win_rate": 70.0,
                    "return_pct": 2.0,
                    "net_realized_pct": 2.0,
                    "profit_factor": 1.5,
                    "total_trades": 40.0,
                    "evaluation_games": 20.0,
                    "evaluation_symbols": 7.0,
                    "expectancy_pct": 0.2,
                    "max_drawdown_pct": 2.0,
                    "positive_episode_rate": 70.0,
                    "long_entries": 0.0,
                    "short_entries": 40.0,
                    "long_entry_share": 0.0,
                    "short_entry_share": 1.0,
                    "directional_imbalance": 1.0,
                    "directional_bias": "sell_heavy",
                    "metrics_by_position_mechanics": {
                        "split_efficiency": 0.8,
                        "pyramid_efficiency": 0.8,
                        "slbe_capture_rate": 0.8,
                        "hold_drag_score": 0.1,
                        "close_quality_score": 0.8,
                        "split_executed": 5,
                        "pyramids_opened": 5,
                        "slbe_triggered": 5,
                        "close_winner_count": 10,
                        "close_loser_count": 2,
                    },
                    "metrics_by_symbol": {},
                },
            },
        }

        verdict = promoter.evaluate_promotion_gate(battle_report, gate_profile="standard")

        self.assertFalse(verdict["allowed"])
        self.assertFalse(verdict["checks"]["long_entries_present"])
        self.assertEqual(verdict["failure_mode"], "sell_heavy")

    def test_reset_randomizes_episode_start_when_enabled(self) -> None:
        """Varie le point de depart des episodes quand l'option est active."""

        data = np.tile(np.linspace(1.0, 2.0, 26, dtype=np.float32), (512, 1))
        config = SimpleNamespace(
            quality_trade_bonus=10.0,
            final_growth_bonus=50.0,
            final_growth_threshold=0.10,
            drawdown_time_penalty_rate=0.2,
            max_drawdown_penalty=10.0,
            loss_penalty_multiplier=2.0,
            slbe_activation_bonus=6.0,
            daily_stretch_target_pct=10.0,
            daily_stretch_max_drawdown_pct=3.5,
            daily_stretch_reward_bonus=4.0,
            horizon="scalp",
            primary_timeframe="M5",
            model_family=None,
            randomize_episode_start=True,
            episode_warmup_bars=100,
        )

        np.random.seed(42)
        env = TradingEnvironment(data=data, symbol="XAUUSD", config=config, max_steps=50)

        starts = set()
        for _ in range(12):
            env.reset()
            starts.add(int(env.start_step))

        self.assertGreater(len(starts), 1)

    def test_reset_keeps_fixed_start_when_randomization_disabled(self) -> None:
        """Conserve un depart stable si la randomisation est desactivee."""

        data = np.tile(np.linspace(1.0, 2.0, 26, dtype=np.float32), (512, 1))
        config = SimpleNamespace(
            quality_trade_bonus=10.0,
            final_growth_bonus=50.0,
            final_growth_threshold=0.10,
            drawdown_time_penalty_rate=0.2,
            max_drawdown_penalty=10.0,
            loss_penalty_multiplier=2.0,
            slbe_activation_bonus=6.0,
            daily_stretch_target_pct=10.0,
            daily_stretch_max_drawdown_pct=3.5,
            daily_stretch_reward_bonus=4.0,
            horizon="scalp",
            primary_timeframe="M5",
            model_family=None,
            randomize_episode_start=False,
            episode_warmup_bars=100,
        )

        env = TradingEnvironment(data=data, symbol="XAUUSD", config=config, max_steps=50)

        starts = set()
        for _ in range(5):
            env.reset()
            starts.add(int(env.start_step))

        self.assertEqual(starts, {100})

    def test_mcts_root_uses_initial_inference_logits(self) -> None:
        """Construit la racine depuis les logits initiaux et non la recurrente."""

        self._require_jax()
        import jax.numpy as jnp

        from eva_lab.muzero.jax_mcts import JAXMuZeroMCTS

        config = SimpleNamespace(
            action_space_size=5,
            support_size=2,
            num_simulations=0,
            root_dirichlet_alpha=0.3,
            root_exploration_fraction=0.25,
            pb_c_base=19_652,
            pb_c_init=1.25,
            discount=0.99,
        )

        def recurrent_apply(_params, hidden_state, action_onehot):
            del action_onehot
            reward_logits = jnp.zeros((1, 5), dtype=jnp.float32)
            policy_logits = jnp.array([[0.0, 12.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
            value_logits = jnp.zeros((1, 5), dtype=jnp.float32)
            return hidden_state, reward_logits, policy_logits, value_logits

        mcts = JAXMuZeroMCTS(config, params=None, apply_fns=(None, recurrent_apply))
        root = mcts.run(
            root_state=jnp.zeros((1, 3), dtype=jnp.float32),
            root_policy_logits=jnp.array([[12.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            root_value_logits=jnp.zeros((1, 5), dtype=jnp.float32),
            root_legal_actions=[HOLD, BUY, SELL],
            add_exploration_noise=False,
        )

        self.assertIn(HOLD, root.children)
        self.assertIn(BUY, root.children)
        self.assertGreater(root.children[HOLD].prior, root.children[BUY].prior)

    def test_self_play_calls_root_policy_mask(self) -> None:
        """Utilise le masque racine metier pendant le self-play."""

        self._require_jax_stack()

        from eva_lab.muzero.config import MuZeroConfigV3
        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

        class TrackingEnvironment(TradingEnvironment):
            """Expose l'appel au masque racine pour le test."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.root_policy_called = False

            def get_root_policy_actions(self) -> list[int]:
                self.root_policy_called = True
                return super().get_root_policy_actions()

        config = MuZeroConfigV3(
            horizon="scalp",
            symbols=["XAUUSD"],
            max_symbols=1,
            dataset_source="csv",
        )
        config.num_simulations = 1
        config.collection_num_simulations = 1
        config.max_moves = 1
        config.window_size = 512
        config.batch_size = 2

        synthetic_data = np.zeros((10, 26), dtype=np.float32)
        synthetic_data[:, 0] = 100.0  # open
        synthetic_data[:, 1] = 101.0  # high
        synthetic_data[:, 2] = 99.0   # low
        synthetic_data[:, 3] = 100.0  # close

        agent = JAXMuZeroAgent(config)
        env = TrackingEnvironment(data=synthetic_data, symbol="XAUUSD", config=config, max_steps=1)
        agent.play_game(env, exploration=True)

        self.assertTrue(env.root_policy_called)

    def test_reanalyze_rewrites_policy_and_value_shapes(self) -> None:
        """Reecrit politiques et valeurs d'un episode sans casser les longueurs."""

        self._require_jax_stack()

        from eva_lab.muzero.config import MuZeroConfigV3
        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

        config = MuZeroConfigV3(
            horizon="scalp",
            symbols=["XAUUSD"],
            max_symbols=1,
            dataset_source="csv",
        )
        config.num_simulations = 1
        config.max_moves = 1
        config.window_size = 512
        config.batch_size = 2

        synthetic_data = np.zeros((10, 26), dtype=np.float32)
        synthetic_data[:, 0] = 100.0  # open
        synthetic_data[:, 1] = 101.0  # high
        synthetic_data[:, 2] = 99.0   # low
        synthetic_data[:, 3] = 100.0  # close

        agent = JAXMuZeroAgent(config)
        env = TradingEnvironment(data=synthetic_data, symbol="XAUUSD", config=config, max_steps=1)
        game = agent.play_game(env, exploration=True)
        self.assertEqual(len(game.policies), 1)
        self.assertEqual(len(game.values), 1)

        game.policies[0] = np.zeros(config.action_space_size, dtype=np.float32)
        game.values[0] = 123.456

        agent.reanalyze_game(game)

        self.assertEqual(len(game.policies), 1)
        self.assertEqual(len(game.values), 1)
        self.assertAlmostEqual(float(np.sum(game.policies[0])), 1.0, places=5)
        self.assertNotAlmostEqual(float(game.values[0]), 123.456, places=3)

    def test_sanitize_metrics_keeps_text_scalars(self) -> None:
        """Conserve les metriques texte sans les convertir en flottants."""

        self._require_jax_stack()

        from eva_lab.muzero.config import MuZeroConfigV3
        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

        agent = object.__new__(JAXMuZeroAgent)
        sanitized = agent._sanitize_metrics(
            {
                "loss_total": np.array(1.25, dtype=np.float32),
                "gpu_target_mode": "cuda:0",
                "labels": np.array(["a", "b"]),
            }
        )

        self.assertEqual(sanitized["gpu_target_mode"], "cuda:0")
        self.assertEqual(sanitized["labels"], ["a", "b"])
        self.assertAlmostEqual(float(sanitized["loss_total"]), 1.25, places=6)

    def test_live_observation_preserves_position_state_without_changing_shape(self) -> None:
        """Injecte l'etat live de position dans le vecteur sans changer `[32]`."""

        self._require_jax_stack()

        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

        agent = JAXMuZeroAgent.__new__(JAXMuZeroAgent)
        agent.config = SimpleNamespace(observation_shape=(35,))
        obs_vec = agent.process_observation(
            {
                "symbol": "XAUUSD",
                "price": 101.0,
                "latest_candle": {
                    "open": 100.5,
                    "high": 101.2,
                    "low": 100.4,
                    "close": 101.0,
                    "tick_volume": 1000.0,
                    "spread": 12.0,
                },
                "indicators": {
                    "EMA_200": 100.0,
                    "RSI": 55.0,
                    "MACD_Hist": 0.2,
                    "VWAP": 100.8,
                    "OBV": 42.0,
                    "Momentum": 0.03,
                    "TRIX": 0.01,
                    "Stoch_K": 65.0,
                    "Stoch_D": 60.0,
                    "CCI": 45.0,
                    "ADX": 22.0,
                    "ADX_Plus_DI": 24.0,
                    "ADX_Minus_DI": 18.0,
                    "Ichi_Tenkan": 100.7,
                    "Ichi_Kijun": 100.6,
                    "Ichi_Senkou_A": 100.8,
                    "Ichi_Senkou_B": 100.4,
                    "ATR": 0.6,
                    "BB_Pct": 0.55,
                },
                "position_state": 1.0,
                "unrealized_return": 0.012,
                "slbe_state": 1.0,
            }
        )

        self.assertEqual(obs_vec.shape[0], 35)
        self.assertAlmostEqual(float(obs_vec[26]), 1.0, places=6)
        self.assertAlmostEqual(float(obs_vec[27]), 0.012, places=6)
        self.assertAlmostEqual(float(obs_vec[28]), 1.0, places=6)
        legal_actions = TradingEnvironment.infer_legal_root_actions_from_observation(obs_vec)
        self.assertIn(SPLIT, legal_actions)
        self.assertIn(CLOSE, legal_actions)

    def test_checkpoint_payload_preserves_training_step_count(self) -> None:
        """Serialise l'etape de training dans le payload de checkpoint."""

        from eva_lab.muzero.checkpoint_utils import build_muzero_checkpoint_payload

        config = SimpleNamespace(
            horizon="scalp",
            observation_shape=(32,),
            action_space_size=5,
            hidden_state_size=256,
            network_hidden_dims=[512, 512, 512],
            support_size=100,
            dataset_descriptor={},
            feature_profile={},
            mechanics_profile_version="v5",
            dataset_id="dataset-test",
            symbols=["XAUUSD"],
        )

        payload = build_muzero_checkpoint_payload(
            config=config,
            params={"weights": np.zeros((2,), dtype=np.float32)},
            opt_state={"step": 1},
            training_step_count=5000,
        )

        self.assertEqual(int(payload["training_step_count"]), 5000)

    def test_checkpoint_path_step_parser_reads_ckpt_suffix(self) -> None:
        """Recupere l'etape depuis le suffixe de nom de fichier legacy."""

        self._require_jax_stack()

        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

        self.assertEqual(
            JAXMuZeroAgent._extract_training_step_from_checkpoint_path(
                "C:/tmp/muzero_scalp_ckpt_5000.pkl"
            ),
            5000,
        )
        self.assertIsNone(
            JAXMuZeroAgent._extract_training_step_from_checkpoint_path(
                "C:/tmp/muzero_scalp_latest.pkl"
            )
        )

    def test_policy_precheck_window_blocks_high_friction_profile(self) -> None:
        """Refuse une fenetre si la friction masque/veto reste trop elevee."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            policy_precheck_max_loss_pol=5.8,
            policy_precheck_min_top1_share=0.75,
            policy_precheck_max_policy_entropy=1.0,
            policy_precheck_max_root_mask_rate=0.05,
            policy_precheck_max_post_veto_rate=0.01,
            policy_precheck_min_balanced_episode_rate=0.85,
            policy_precheck_min_long_entry_share=0.35,
            policy_precheck_min_short_entry_share=0.35,
            policy_screen_max_loss_pol=6.6,
            policy_screen_min_top1_share=0.88,
            policy_screen_max_policy_entropy=0.45,
            policy_screen_max_root_mask_rate=0.05,
            policy_screen_max_post_veto_rate=0.01,
            policy_screen_min_balanced_episode_rate=0.85,
            policy_screen_min_long_entry_share=0.35,
            policy_screen_min_short_entry_share=0.35,
        )
        history = [
            {
                "loss_pol": 5.4,
                "policy_top1_share": 0.90,
                "policy_entropy": 0.38,
                "root_mask_rate": 0.11,
                "post_veto_to_hold_rate": 0.004,
                "soft_penalty_to_bonus_ratio": 1.2,
                "balanced_episode_rate": 92.0,
                "long_entry_share": 0.52,
                "short_entry_share": 0.48,
            }
            for _ in range(500)
        ]

        verdict = module._evaluate_policy_precheck_window(
            history=history,
            config=config,
            step=12000,
            stage="mid_run",
        )

        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["reason"], "root_mask_rate")

    def test_seed_viability_window_marks_non_productive_seed(self) -> None:
        """Bloque un bootstrap offensif si le `root_mask` reste trop haut."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            training_steps=3500,
            seed_viability_min_step=2000,
            seed_viability_max_step=3500,
            seed_viability_max_root_mask_rate=0.08,
            seed_viability_min_split_runner_capture_rate=0.0,
            seed_viability_min_pyramid_exit_capture_rate=0.0,
            seed_viability_min_loss_pol_improvement=0.10,
        )
        history = [
            {
                "loss_pol": 5.85,
                "loss_pol_per_head": 1.12,
                "root_mask_rate": 0.11,
                "split_monetization_window_count": 3.0,
                "runner_profit_hold_window_count": 2.0,
                "pyramid_monetization_window_count": 2.0,
                "split_monetization_capture_rate": 0.15,
                "runner_profit_hold_capture_rate": 0.10,
                "pyramid_monetization_capture_rate": 0.12,
                "profit_peak_giveback_ratio": 0.20,
                "split_runner_capture_rate": 0.0,
                "pyramid_exit_capture_rate": 0.0,
            }
            for _ in range(500)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_dir = Path(tmp_dir)
            (weights_dir / "muzero_scalp_ckpt_23000.pkl").write_text("stub", encoding="utf-8")
            verdict = module._evaluate_seed_viability_window(
                history=history,
                config=config,
                step=2500,
                horizon="scalp",
                weights_dir=weights_dir,
                trial_mode="offensive_bootstrap",
            )

        self.assertFalse(bool(verdict["allowed"]))
        self.assertEqual(verdict["status"], "seed_not_viable_for_v66")
        self.assertEqual(verdict["reason"], "root_mask_rate")
        self.assertEqual(verdict["seed_stage"], "offensive_bootstrap")
        self.assertTrue(str(verdict["recommended_seed_for_v66"]).endswith("muzero_scalp_ckpt_23000.pkl"))

    def test_seed_viability_window_finalizes_short_run_at_last_step(self) -> None:
        """Valide proprement un `seed-short` a la derniere etape disponible."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            training_steps=5000,
            seed_viability_min_step=3000,
            seed_viability_max_step=5000,
            seed_viability_max_root_mask_rate=0.08,
            seed_viability_min_split_runner_capture_rate=0.0,
            seed_viability_min_pyramid_exit_capture_rate=0.0,
            seed_viability_min_loss_pol_improvement=0.10,
        )
        history = [
            {
                "loss_pol": value,
                "loss_pol_per_head": per_head,
                "root_mask_rate": 0.05,
                "split_runner_capture_rate": 0.18,
                "split_monetization_capture_rate": 0.18,
                "pyramid_exit_capture_rate": 0.22,
                "pyramid_monetization_capture_rate": 0.11,
                "close_quality_score": 0.46,
                "slbe_capture_rate": 0.52,
            }
            for value, per_head in (
                (5.40, 0.95),
                (5.34, 0.94),
                (5.20, 0.92),
                (5.05, 0.90),
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_dir = Path(tmp_dir)
            (weights_dir / "muzero_scalp_ckpt_23000.pkl").write_text("stub", encoding="utf-8")
            verdict = module._evaluate_seed_viability_window(
                history=history,
                config=config,
                step=5000,
                horizon="scalp",
                weights_dir=weights_dir,
                trial_mode="seed_short_mixed",
            )

        self.assertTrue(bool(verdict["allowed"]))
        self.assertEqual(verdict["status"], "seed_viability_passed")
        self.assertEqual(verdict["reason"], "seed_window_passed")
        self.assertEqual(int(verdict["window_max_step"]), 5000)
        self.assertEqual(verdict["seed_stage"], "seed_short_mixed")

    def test_seed_viability_window_bootstrap_tolere_zero_capture_si_fenetres_ouvertes(self) -> None:
        """N'invalide pas le bootstrap si les fenetres existent mais que la capture est encore nulle."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            training_steps=3000,
            seed_viability_min_step=2000,
            seed_viability_max_step=3000,
        )
        history = [
            {
                "loss_pol": 5.10,
                "loss_pol_per_head": 0.94,
                "root_mask_rate": 0.005,
                "split_monetization_capture_rate": 0.0,
                "runner_profit_hold_capture_rate": 0.0,
                "pyramid_monetization_capture_rate": 0.0,
                "profit_peak_giveback_ratio": 0.10,
                "split_monetization_window_count": 2.0,
                "runner_profit_hold_window_count": 3.0,
                "pyramid_monetization_window_count": 2.0,
            }
            for _ in range(120)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_dir = Path(tmp_dir)
            (weights_dir / "muzero_scalp_ckpt_23000.pkl").write_text("stub", encoding="utf-8")
            verdict = module._evaluate_seed_viability_window(
                history=history,
                config=config,
                step=2500,
                horizon="scalp",
                weights_dir=weights_dir,
                trial_mode="offensive_bootstrap",
            )

        self.assertTrue(bool(verdict["allowed"]))
        self.assertEqual(verdict["status"], "monitoring")
        self.assertEqual(verdict["reason"], "within_seed_window")

    def test_policy_target_smoothing_conserve_une_distribution_legale(self) -> None:
        """Construit une cible lissee normalisee sans rouvrir les actions illegales."""

        self._require_jax_stack()
        import jax.numpy as jnp

        module = self._load_jax_trainer_module()
        trainer = module.MuZeroTrainerJAX.__new__(module.MuZeroTrainerJAX)
        trainer.config = SimpleNamespace(
            action_space_size=5,
            policy_target_smoothing_temperature=1.15,
        )
        target_policy = jnp.asarray(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        uniform_root = trainer._build_root_legal_policy(jnp.asarray([False, True]))
        smoothed = trainer._smooth_target_policy(target_policy, uniform_root, 0.10)

        self.assertAlmostEqual(float(jnp.sum(smoothed[0])), 1.0, places=6)
        self.assertAlmostEqual(float(jnp.sum(smoothed[1])), 1.0, places=6)
        self.assertAlmostEqual(float(smoothed[0, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(smoothed[0, 4]), 0.0, places=6)
        self.assertGreater(float(smoothed[1, 4]), 0.0)

    def test_policy_target_smoothing_reduit_la_cross_entropy_synthetique(self) -> None:
        """Reduit la loss policy si les logits suivent deja une cible lissee."""

        self._require_jax_stack()
        import jax.numpy as jnp
        import optax

        module = self._load_jax_trainer_module()
        trainer = module.MuZeroTrainerJAX.__new__(module.MuZeroTrainerJAX)
        trainer.config = SimpleNamespace(
            action_space_size=5,
            policy_target_smoothing_temperature=1.15,
        )
        target_policy = jnp.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
        uniform_root = trainer._build_root_legal_policy(jnp.asarray([False]))
        smoothed = trainer._smooth_target_policy(target_policy, uniform_root, 0.10)
        logits = jnp.log(jnp.clip(smoothed, 1e-8, 1.0))
        logits_sharp = jnp.log(jnp.clip(target_policy, 1e-8, 1.0))
        loss_of_sharp_pred = optax.softmax_cross_entropy(logits_sharp, smoothed)
        smooth_loss = optax.softmax_cross_entropy(logits, smoothed)

        self.assertLess(float(smooth_loss[0]), float(loss_of_sharp_pred[0]))

    def test_loss_pol_per_head_divise_la_loss_par_nombre_de_tetes(self) -> None:
        """Normalise correctement la loss policy par tete racine plus unroll."""

        self._require_jax_stack()

        module = self._load_jax_trainer_module()
        per_head = module.MuZeroTrainerJAX._compute_loss_pol_per_head(7.2, 5)
        self.assertAlmostEqual(float(per_head), 1.2, places=6)

    def test_weighted_loss_pol_per_head_reflete_les_poids_root_et_unroll(self) -> None:
        """Calcule la moyenne par tete en respectant les poids policy."""

        self._require_jax_stack()

        module = self._load_jax_trainer_module()
        per_head = module.MuZeroTrainerJAX._compute_weighted_loss_pol_per_head(
            1.8,
            5.0,
            5,
            0.95,
            0.70,
        )

        self.assertAlmostEqual(float(per_head), float(((0.95 * 1.8) + (0.70 * 5.0)) / 4.45), places=6)

    def test_copy_episode_summary_to_game_metadata_conserve_les_metriques_offensives(self) -> None:
        """Copie les nouvelles metriques offensives vers `game.metadata`."""

        self._require_jax_stack()

        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

        game = GameHistory()
        episode_summary = {
            "symbol": "XAUUSD",
            "return_pct": 1.25,
            "metrics_by_position_mechanics": {
                "split_monetization_window_count": 4.0,
                "split_monetization_capture_rate": 0.25,
                "split_missed_window_count": 1.0,
                "runner_profit_hold_window_count": 3.0,
                "runner_viable_window_count": 5.0,
                "runner_hold_after_soft_tp_count": 2.0,
                "runner_profit_hold_capture_rate": 0.20,
                "runner_missed_extension_count": 2.0,
                "runner_viable_but_closed_count": 1.0,
                "early_full_close_after_soft_tp_count": 1.0,
                "runner_retained_profit_score": 0.42,
                "pyramid_monetization_window_count": 5.0,
                "pyramid_monetization_capture_rate": 0.40,
                "pyramid_missed_add_count": 1.0,
                "profit_peak_reached_count": 6.0,
                "profit_peak_giveback_ratio": 0.33,
            },
        }

        JAXMuZeroAgent._copy_episode_summary_to_game_metadata(game, episode_summary)

        self.assertEqual(game.metadata["symbol"], "XAUUSD")
        self.assertEqual(game.metadata["split_monetization_window_count"], 4.0)
        self.assertEqual(game.metadata["runner_profit_hold_window_count"], 3.0)
        self.assertEqual(game.metadata["runner_viable_window_count"], 5.0)
        self.assertEqual(game.metadata["runner_hold_after_soft_tp_count"], 2.0)
        self.assertEqual(game.metadata["runner_viable_but_closed_count"], 1.0)
        self.assertEqual(game.metadata["early_full_close_after_soft_tp_count"], 1.0)
        self.assertAlmostEqual(float(game.metadata["runner_retained_profit_score"]), 0.42, places=6)
        self.assertEqual(game.metadata["pyramid_monetization_window_count"], 5.0)
        self.assertEqual(game.metadata["profit_peak_reached_count"], 6.0)
        self.assertAlmostEqual(float(game.metadata["profit_peak_giveback_ratio"]), 0.33, places=6)

    def test_policy_precheck_window_accepts_balanced_five_x_profile(self) -> None:
        """Valide une fenetre `5.x` stable avec friction faible et flux equilibre."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            policy_precheck_max_loss_pol=5.8,
            policy_precheck_max_loss_pol_per_head=0.98,
            policy_precheck_min_top1_share=0.75,
            policy_precheck_max_policy_entropy=1.0,
            policy_precheck_max_root_mask_rate=0.05,
            policy_precheck_max_post_veto_rate=0.01,
            policy_precheck_min_balanced_episode_rate=0.85,
            policy_precheck_min_long_entry_share=0.35,
            policy_precheck_min_short_entry_share=0.35,
            policy_screen_max_loss_pol=6.6,
            policy_screen_max_loss_pol_per_head=1.08,
            policy_screen_min_top1_share=0.88,
            policy_screen_max_policy_entropy=0.45,
            policy_screen_max_root_mask_rate=0.05,
            policy_screen_max_post_veto_rate=0.01,
            policy_screen_min_balanced_episode_rate=0.85,
            policy_screen_min_long_entry_share=0.35,
            policy_screen_min_short_entry_share=0.35,
        )
        history = [
            {
                "loss_pol": 5.35,
                "loss_pol_per_head": 0.92,
                "policy_top1_share": 0.79,
                "policy_entropy": 0.36,
                "root_mask_rate": 0.03,
                "post_veto_to_hold_rate": 0.005,
                "soft_penalty_to_bonus_ratio": 1.1,
                "balanced_episode_rate": 91.0,
                "long_entry_share": 0.55,
                "short_entry_share": 0.45,
            }
            for _ in range(500)
        ]

        verdict = module._evaluate_policy_precheck_window(
            history=history,
            config=config,
            step=12000,
            stage="mid_run",
        )

        self.assertEqual(verdict["status"], "full_ready")
        self.assertEqual(verdict["reason"], "eligible_full")

    def test_policy_precheck_window_blocks_when_too_few_symbols_have_clean_closes(self) -> None:
        """Bloque `full_ready` si trop peu de symboles ont une sortie exploitable."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            policy_precheck_max_loss_pol=4.8,
            policy_precheck_max_loss_pol_per_head=0.98,
            policy_precheck_min_top1_share=0.75,
            policy_precheck_max_policy_entropy=1.0,
            policy_precheck_max_root_mask_rate=0.02,
            policy_precheck_max_post_veto_rate=0.01,
            policy_precheck_min_balanced_episode_rate=0.85,
            policy_precheck_min_long_entry_share=0.35,
            policy_precheck_min_short_entry_share=0.35,
            policy_precheck_min_close_quality_score=0.40,
            policy_precheck_min_split_efficiency=0.35,
            policy_precheck_min_pyramid_efficiency=0.35,
            policy_precheck_min_slbe_capture_rate=0.45,
            policy_precheck_max_hold_drag_score=0.10,
            policy_precheck_min_good_close_symbols=5,
            policy_precheck_min_symbol_close_quality_score=0.25,
            policy_precheck_min_symbol_close_events=6,
            policy_screen_max_loss_pol=6.6,
            policy_screen_max_loss_pol_per_head=1.08,
            policy_screen_min_top1_share=0.88,
            policy_screen_max_policy_entropy=0.45,
            policy_screen_max_root_mask_rate=0.05,
            policy_screen_max_post_veto_rate=0.01,
            policy_screen_min_balanced_episode_rate=0.85,
            policy_screen_min_long_entry_share=0.35,
            policy_screen_min_short_entry_share=0.35,
        )
        history = [
            {
                "loss_pol": 4.2,
                "loss_pol_per_head": 0.74,
                "policy_top1_share": 0.92,
                "policy_entropy": 0.31,
                "root_mask_rate": 0.01,
                "post_veto_to_hold_rate": 0.002,
                "soft_penalty_to_bonus_ratio": 1.0,
                "balanced_episode_rate": 0.94,
                "long_entry_share": 0.51,
                "short_entry_share": 0.49,
                "close_quality_score": 0.55,
                "split_efficiency": 0.42,
                "pyramid_efficiency": 0.44,
                "slbe_capture_rate": 0.50,
                "hold_drag_score": 0.02,
                "close_quality_by_symbol": {
                    "XAUUSD": 0.35,
                    "US30.cash": 0.31,
                    "US100.cash": 0.29,
                    "US500.cash": 0.10,
                },
                "close_events_by_symbol": {
                    "XAUUSD": 8,
                    "US30.cash": 8,
                    "US100.cash": 6,
                    "US500.cash": 7,
                },
            }
            for _ in range(500)
        ]

        verdict = module._evaluate_policy_precheck_window(
            history=history,
            config=config,
            step=12000,
            stage="pre_arena",
        )

        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["reason"], "good_close_symbols")

    def test_policy_precheck_window_allows_screen_only_profile(self) -> None:
        """Autorise le screen si la policy est proche de la cible sans etre `full_ready`."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            policy_precheck_max_loss_pol=5.8,
            policy_precheck_max_loss_pol_per_head=0.98,
            policy_precheck_min_top1_share=0.75,
            policy_precheck_max_policy_entropy=1.0,
            policy_precheck_max_root_mask_rate=0.05,
            policy_precheck_max_post_veto_rate=0.01,
            policy_precheck_min_balanced_episode_rate=0.85,
            policy_precheck_min_long_entry_share=0.35,
            policy_precheck_min_short_entry_share=0.35,
            policy_screen_max_loss_pol=6.6,
            policy_screen_max_loss_pol_per_head=1.08,
            policy_screen_min_top1_share=0.88,
            policy_screen_max_policy_entropy=0.45,
            policy_screen_max_root_mask_rate=0.05,
            policy_screen_max_post_veto_rate=0.01,
            policy_screen_min_balanced_episode_rate=0.85,
            policy_screen_min_long_entry_share=0.35,
            policy_screen_min_short_entry_share=0.35,
        )
        history = [
            {
                "loss_pol": 6.25,
                "loss_pol_per_head": 1.04,
                "policy_top1_share": 0.91,
                "policy_entropy": 0.39,
                "root_mask_rate": 0.02,
                "post_veto_to_hold_rate": 0.003,
                "soft_penalty_to_bonus_ratio": 1.25,
                "balanced_episode_rate": 90.0,
                "long_entry_share": 0.51,
                "short_entry_share": 0.49,
            }
            for _ in range(500)
        ]

        verdict = module._evaluate_policy_precheck_window(
            history=history,
            config=config,
            step=12000,
            stage="pre_arena",
        )

        self.assertEqual(verdict["status"], "screen_only")
        self.assertEqual(verdict["reason"], "eligible_screen")

    def test_policy_precheck_screen_accepts_smoothed_policy_profile(self) -> None:
        """Autorise le screen avec une policy lissee moins concentree."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            policy_precheck_max_loss_pol_per_head=1.12,
            policy_precheck_min_top1_share=0.75,
            policy_precheck_max_policy_entropy=1.0,
            policy_precheck_max_root_mask_rate=0.02,
            policy_precheck_max_post_veto_rate=0.01,
            policy_precheck_min_balanced_episode_rate=0.85,
            policy_precheck_min_long_entry_share=0.35,
            policy_precheck_min_short_entry_share=0.35,
            policy_precheck_min_close_quality_score=0.40,
            policy_precheck_min_split_efficiency=0.35,
            policy_precheck_min_pyramid_efficiency=0.35,
            policy_precheck_min_slbe_capture_rate=0.45,
            policy_precheck_max_hold_drag_score=0.10,
            policy_precheck_min_good_close_symbols=5,
            policy_precheck_min_symbol_close_quality_score=0.25,
            policy_precheck_min_symbol_close_events=6,
            policy_precheck_max_root_mask_rate_trend=0.02,
            policy_screen_max_loss_pol_per_head=1.20,
            policy_screen_min_top1_share=0.60,
            policy_screen_max_policy_entropy=1.10,
            policy_screen_max_root_mask_rate=0.05,
            policy_screen_max_post_veto_rate=0.01,
            policy_screen_min_balanced_episode_rate=0.85,
            policy_screen_min_long_entry_share=0.35,
            policy_screen_min_short_entry_share=0.35,
        )
        history = [
            {
                "loss_pol": 4.3,
                "loss_pol_per_head": 1.17,
                "loss_pol_root": 1.45,
                "loss_pol_unroll_mean": 1.08,
                "policy_top1_share": 0.69,
                "policy_entropy": 0.71,
                "root_mask_rate": 0.045,
                "post_veto_to_hold_rate": 0.0,
                "soft_penalty_to_bonus_ratio": 1.1,
                "balanced_episode_rate": 0.92,
                "long_entry_share": 0.51,
                "short_entry_share": 0.49,
                "close_quality_score": 0.47,
                "split_efficiency": 0.0,
                "pyramid_efficiency": 0.12,
                "slbe_capture_rate": 0.97,
                "hold_drag_score": 0.0,
            }
            for _ in range(500)
        ]

        verdict = module._evaluate_policy_precheck_window(
            history=history,
            config=config,
            step=10000,
            stage="pre_arena",
        )

        self.assertEqual(verdict["status"], "screen_only")
        self.assertEqual(verdict["reason"], "eligible_screen")

    def test_recent_screen_selection_prefers_best_recent_window(self) -> None:
        """Choisit les checkpoints autour de la meilleure fenetre recente, pas du dernier step."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_recent_steps=2500,
            arena_screen_candidate_count=5,
            arena_screen_window_size=5,
            checkpoint_interval=500,
        )
        history: list[dict[str, object]] = []
        for step in range(9500, 12001, 100):
            if step <= 10400:
                loss_pol = 5.2
                top1 = 0.90
                soft_ratio = 1.05
            else:
                loss_pol = 6.8
                top1 = 0.82
                soft_ratio = 1.80
            history.append(
                {
                    "training_step": step,
                    "loss_pol": loss_pol,
                    "policy_top1_share": top1,
                    "policy_entropy": 0.35,
                    "root_mask_rate": 0.02,
                    "post_veto_to_hold_rate": 0.002,
                    "soft_penalty_to_bonus_ratio": soft_ratio,
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            weights_dir = Path(temp_dir)
            for checkpoint_step in (9500, 10000, 10500, 11000, 11500, 12000):
                (weights_dir / f"muzero_scalp_ckpt_{checkpoint_step}.pkl").touch()

            selected = module._select_recent_screen_checkpoints(
                history=history,
                weights_dir=weights_dir,
                horizon="scalp",
                last_step=12000,
                config=config,
            )

        selected_steps = [int(item["checkpoint_step"]) for item in selected]
        self.assertEqual(selected_steps, [9500, 10000, 10500, 11000, 11500])
        self.assertNotIn(12000, selected_steps)
        self.assertTrue(all("selection_window" in item for item in selected))

    def test_recent_screen_selection_uses_late_targets_when_configured(self) -> None:
        """Evalue les checkpoints tardifs explicites du cycle V6.12."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_recent_steps=2500,
            arena_screen_candidate_count=4,
            arena_screen_window_size=500,
            checkpoint_interval=500,
            arena_screen_target_steps=[10000, 12000, 14000, 16000],
        )
        history = [{"training_step": step, "loss_pol": 5.0} for step in range(9000, 15001, 500)]

        with tempfile.TemporaryDirectory() as temp_dir:
            weights_dir = Path(temp_dir)
            for checkpoint_step in (9000, 10000, 12000, 14000):
                (weights_dir / f"muzero_scalp_ckpt_{checkpoint_step}.pkl").touch()

            selected = module._select_recent_screen_checkpoints(
                history=history,
                weights_dir=weights_dir,
                horizon="scalp",
                last_step=15000,
                config=config,
            )

        selected_steps = [int(item["checkpoint_step"]) for item in selected]
        self.assertEqual(selected_steps, [10000, 12000, 14000])
        self.assertTrue(all(dict(item["selection_window"]).get("mode") == "v6_12_late_target" for item in selected))

    def test_screen_winner_gate_accepts_clean_balanced_candidate(self) -> None:
        """Autorise la full Arena si le gagnant du screen est propre."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
            arena_screen_max_hold_drag_score=0.80,
            arena_screen_min_close_quality_score=0.35,
            arena_screen_min_split_opportunities=3,
            arena_screen_min_split_efficiency=0.35,
            arena_screen_min_pyramid_opportunities=3,
            arena_screen_min_pyramid_efficiency=0.35,
            arena_screen_min_slbe_triggered=3,
            arena_screen_min_slbe_capture_rate=0.30,
            arena_screen_min_profitable_symbols=0,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "outcome": "VICTORY",
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.35,
                        "return_pct": 0.08,
                        "expectancy_pct": 0.01,
                        "positive_episode_rate": 61.0,
                        "directional_bias": "balanced",
                        "metrics_by_position_mechanics": {
                            "hold_drag_score": 0.25,
                            "close_quality_score": 0.62,
                            "split_opportunity_count": 2,
                            "split_efficiency": 0.10,
                            "pyramid_opportunity_count": 1,
                            "pyramid_efficiency": 0.10,
                            "slbe_triggered": 2,
                            "slbe_capture_rate": 0.10,
                        },
                        "metrics_by_symbol": {
                            "XAUUSD": {"profit_factor": 1.2, "return_pct": 0.01},
                            "US30.cash": {"profit_factor": 1.2, "return_pct": 0.01},
                            "GER40.cash": {"profit_factor": 1.2, "return_pct": 0.01},
                            "EURUSD": {"profit_factor": 1.2, "return_pct": 0.01},
                            "BTCUSD": {"profit_factor": 1.2, "return_pct": 0.01},
                        },
                    }
                }
            },
            config,
        )

        self.assertTrue(verdict["allowed"])
        self.assertEqual(verdict["status"], "eligible")

    def test_screen_winner_gate_blocks_low_split_runner_capture_rate(self) -> None:
        """Bloque un candidat si le split ne se transforme pas en valeur finale."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
            arena_screen_max_hold_drag_score=0.80,
            arena_screen_min_close_quality_score=0.35,
            arena_screen_min_split_opportunities=3,
            arena_screen_min_split_efficiency=0.35,
            arena_screen_min_split_runner_capture_rate=0.20,
            arena_screen_min_pyramid_opportunities=3,
            arena_screen_min_pyramid_efficiency=0.35,
            arena_screen_min_pyramid_exit_capture_rate=0.20,
            arena_screen_min_slbe_triggered=3,
            arena_screen_min_slbe_capture_rate=0.30,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "outcome": "VICTORY",
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.55,
                        "return_pct": 0.12,
                        "expectancy_pct": 0.02,
                        "positive_episode_rate": 60.0,
                        "directional_bias": "balanced",
                        "metrics_by_position_mechanics": {
                            "hold_drag_score": 0.12,
                            "close_quality_score": 0.58,
                            "split_opportunity_count": 5,
                            "split_efficiency": 0.52,
                            "split_runner_capture_rate": 0.10,
                            "pyramid_opportunity_count": 4,
                            "pyramid_efficiency": 0.48,
                            "pyramid_exit_capture_rate": 0.42,
                            "slbe_triggered": 4,
                            "slbe_capture_rate": 0.42,
                        },
                    }
                }
            },
            config,
        )

        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["reason"], "split_runner_capture_rate")

    def test_screen_winner_gate_blocks_defeat_before_full_arena(self) -> None:
        """Refuse la full Arena si le screen n'a pas battu le champion."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
            arena_screen_max_hold_drag_score=0.80,
            arena_screen_min_close_quality_score=0.35,
            arena_screen_min_split_opportunities=3,
            arena_screen_min_split_efficiency=0.35,
            arena_screen_min_pyramid_opportunities=3,
            arena_screen_min_pyramid_efficiency=0.35,
            arena_screen_min_slbe_triggered=3,
            arena_screen_min_slbe_capture_rate=0.30,
        )

        verdict = module._evaluate_screen_winner_gate(
            {
                "outcome": "DEFEAT",
                "challenger": {
                    "metrics": {
                        "profit_factor": 2.0,
                        "return_pct": 0.5,
                        "expectancy_pct": 0.1,
                        "positive_episode_rate": 80.0,
                        "directional_bias": "balanced",
                        "metrics_by_position_mechanics": {
                            "hold_drag_score": 0.0,
                            "close_quality_score": 0.9,
                        },
                    }
                },
            },
            config,
        )

        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "screen_victory")

    def test_screen_winner_gate_blocks_negative_expectancy_candidate(self) -> None:
        """Refuse la full Arena si le gagnant du screen reste negatif en expectancy."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
            arena_screen_max_hold_drag_score=0.80,
            arena_screen_min_close_quality_score=0.35,
            arena_screen_min_split_opportunities=3,
            arena_screen_min_split_efficiency=0.35,
            arena_screen_min_pyramid_opportunities=3,
            arena_screen_min_pyramid_efficiency=0.35,
            arena_screen_min_slbe_triggered=3,
            arena_screen_min_slbe_capture_rate=0.30,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "outcome": "VICTORY",
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.42,
                        "return_pct": 0.04,
                        "expectancy_pct": 0.0,
                        "positive_episode_rate": 58.0,
                        "directional_bias": "balanced",
                        "metrics_by_position_mechanics": {
                            "hold_drag_score": 0.20,
                            "close_quality_score": 0.61,
                            "split_opportunity_count": 2,
                            "split_efficiency": 0.10,
                            "pyramid_opportunity_count": 2,
                            "pyramid_efficiency": 0.10,
                            "slbe_triggered": 2,
                            "slbe_capture_rate": 0.10,
                        },
                    }
                }
            },
            config,
        )

        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["reason"], "expectancy_pct")

    def test_screen_winner_gate_blocks_high_hold_drag_candidate(self) -> None:
        """Refuse un gagnant de screen qui garde trop de sorties passives."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
            arena_screen_max_hold_drag_score=0.80,
            arena_screen_min_close_quality_score=0.35,
            arena_screen_min_split_opportunities=3,
            arena_screen_min_split_efficiency=0.35,
            arena_screen_min_pyramid_opportunities=3,
            arena_screen_min_pyramid_efficiency=0.35,
            arena_screen_min_slbe_triggered=3,
            arena_screen_min_slbe_capture_rate=0.30,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "outcome": "VICTORY",
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.42,
                        "return_pct": 0.08,
                        "expectancy_pct": 0.01,
                        "positive_episode_rate": 60.0,
                        "directional_bias": "balanced",
                        "metrics_by_position_mechanics": {
                            "hold_drag_score": 0.95,
                            "close_quality_score": 0.65,
                            "split_opportunity_count": 4,
                            "split_efficiency": 0.60,
                            "pyramid_opportunity_count": 4,
                            "pyramid_efficiency": 0.60,
                            "slbe_triggered": 4,
                            "slbe_capture_rate": 0.60,
                        },
                    }
                }
            },
            config,
        )

        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "hold_drag_score")

    def test_screen_winner_gate_blocks_candidate_with_too_few_profitable_symbols(self) -> None:
        """Refuse la full Arena si le screen masque encore trop de symboles faibles."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
            arena_screen_max_hold_drag_score=0.80,
            arena_screen_min_close_quality_score=0.35,
            arena_screen_min_split_opportunities=3,
            arena_screen_min_split_efficiency=0.35,
            arena_screen_min_pyramid_opportunities=3,
            arena_screen_min_pyramid_efficiency=0.35,
            arena_screen_min_slbe_triggered=3,
            arena_screen_min_slbe_capture_rate=0.30,
            arena_screen_min_profitable_symbols=5,
            arena_screen_min_symbol_profit_factor=1.0,
            arena_screen_min_symbol_return_pct=0.0,
            arena_screen_min_symbol_split_efficiency=0.20,
            arena_screen_min_symbol_slbe_capture_rate=0.25,
            arena_screen_min_symbol_close_quality_score=0.20,
            arena_screen_min_symbol_close_events=6,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "outcome": "VICTORY",
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.45,
                        "return_pct": 0.06,
                        "expectancy_pct": 0.01,
                        "positive_episode_rate": 60.0,
                        "directional_bias": "balanced",
                        "metrics_by_position_mechanics": {
                            "hold_drag_score": 0.10,
                            "close_quality_score": 0.55,
                            "split_opportunity_count": 4,
                            "split_efficiency": 0.42,
                            "split_runner_capture_rate": 0.24,
                            "pyramid_opportunity_count": 4,
                            "pyramid_efficiency": 0.45,
                            "pyramid_exit_capture_rate": 0.30,
                            "slbe_triggered": 4,
                            "slbe_capture_rate": 0.46,
                        },
                        "metrics_by_symbol": {
                            "XAUUSD": {"profit_factor": 1.2, "return_pct": 0.01},
                            "US30.cash": {"profit_factor": 1.1, "return_pct": 0.01},
                            "US100.cash": {"profit_factor": 1.0, "return_pct": 0.01},
                            "US500.cash": {"profit_factor": 0.9, "return_pct": -0.01},
                        },
                    }
                }
            },
            config,
        )

        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "profitable_symbols")

    def test_screen_winner_gate_blocks_candidate_beaten_by_inverse(self) -> None:
        """Refuse la full Arena si le miroir directionnel fait mieux."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
            arena_screen_max_hold_drag_score=0.80,
            arena_screen_min_close_quality_score=0.35,
            arena_screen_min_split_opportunities=3,
            arena_screen_min_split_efficiency=0.35,
            arena_screen_min_split_runner_capture_rate=0.20,
            arena_screen_min_pyramid_opportunities=3,
            arena_screen_min_pyramid_efficiency=0.35,
            arena_screen_min_pyramid_exit_capture_rate=0.20,
            arena_screen_min_slbe_triggered=3,
            arena_screen_min_slbe_capture_rate=0.30,
            arena_inverse_min_profitable_symbols=5,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "outcome": "VICTORY",
                "edge_vs_inverse_pf": -0.10,
                "edge_vs_inverse_return_pct": -0.02,
                "edge_vs_inverse_profitable_symbols": 3,
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.42,
                        "return_pct": 0.08,
                        "expectancy_pct": 0.01,
                        "positive_episode_rate": 60.0,
                        "directional_bias": "balanced",
                        "metrics_by_position_mechanics": {
                            "hold_drag_score": 0.12,
                            "close_quality_score": 0.58,
                            "split_opportunity_count": 4,
                            "split_efficiency": 0.52,
                            "split_runner_capture_rate": 0.24,
                            "pyramid_opportunity_count": 4,
                            "pyramid_efficiency": 0.48,
                            "pyramid_exit_capture_rate": 0.42,
                            "slbe_triggered": 4,
                            "slbe_capture_rate": 0.42,
                        },
                    }
                },
            },
            config,
        )

        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "inverse_edge")

    def test_policy_precheck_blocks_rising_root_mask_trend(self) -> None:
        """Bloque le precheck si le root mask se degrade dans la fenetre."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            policy_precheck_max_loss_pol=4.8,
            policy_precheck_min_top1_share=0.75,
            policy_precheck_max_policy_entropy=1.0,
            policy_precheck_max_root_mask_rate=0.05,
            policy_precheck_max_root_mask_rate_trend=0.02,
            policy_precheck_max_post_veto_rate=0.01,
            policy_precheck_min_balanced_episode_rate=0.85,
            policy_precheck_min_long_entry_share=0.35,
            policy_precheck_min_short_entry_share=0.35,
            policy_precheck_min_close_quality_score=0.40,
            policy_precheck_min_split_efficiency=0.35,
            policy_precheck_min_pyramid_efficiency=0.35,
            policy_precheck_min_slbe_capture_rate=0.45,
            policy_precheck_max_hold_drag_score=0.10,
            policy_precheck_min_good_close_symbols=5,
            policy_precheck_min_symbol_close_quality_score=0.25,
            policy_precheck_min_symbol_close_events=6,
            policy_screen_max_loss_pol=6.6,
            policy_screen_min_top1_share=0.88,
            policy_screen_max_policy_entropy=0.45,
            policy_screen_max_root_mask_rate=0.05,
            policy_screen_max_post_veto_rate=0.01,
            policy_screen_min_balanced_episode_rate=0.85,
            policy_screen_min_long_entry_share=0.35,
            policy_screen_min_short_entry_share=0.35,
        )
        history = []
        for root_mask_rate in (0.01, 0.02, 0.05, 0.07):
            history.append(
                {
                    "loss_pol": 4.2,
                    "policy_top1_share": 0.90,
                    "policy_entropy": 0.30,
                    "root_mask_rate": root_mask_rate,
                    "post_veto_to_hold_rate": 0.0,
                    "soft_penalty_to_bonus_ratio": 0.4,
                    "close_quality_score": 0.50,
                    "split_efficiency": 0.50,
                    "split_runner_capture_rate": 0.24,
                    "pyramid_efficiency": 0.45,
                    "pyramid_exit_capture_rate": 0.30,
                    "slbe_capture_rate": 0.55,
                    "hold_drag_score": 0.0,
                    "balanced_episode_rate": 0.95,
                    "long_entry_share": 0.48,
                    "short_entry_share": 0.52,
                    "close_quality_by_symbol": {
                        "XAUUSD": 0.40,
                        "US30.cash": 0.40,
                        "GER40.cash": 0.40,
                        "EURUSD": 0.40,
                        "US100.cash": 0.40,
                    },
                    "close_events_by_symbol": {
                        "XAUUSD": 6,
                        "US30.cash": 6,
                        "GER40.cash": 6,
                        "EURUSD": 6,
                        "US100.cash": 6,
                    },
                }
            )

        verdict = module._evaluate_policy_precheck_window(
            history=history,
            config=config,
            step=12000,
            stage="mid_run",
        )

        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["reason"], "root_mask_rate_trend")


if __name__ == "__main__":
    unittest.main()
