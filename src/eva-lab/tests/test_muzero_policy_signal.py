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

    def test_replay_diversity_stats_expose_root_mask_and_post_veto_rates(self) -> None:
        """Expose les nouvelles metriques de friction policy/environnement."""

        buffer = PrioritizedReplayBuffer(max_games=4)
        game = GameHistory()
        game.store(np.zeros(4, dtype=np.float32), HOLD, 0.0, np.ones(5, dtype=np.float32) / 5.0, 0.0)
        game.metadata.update(
            {
                "balanced_episode": True,
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
                "blocked_buy_entries": 1,
                "blocked_sell_entries": 0,
            }
        )
        buffer.save_game(game)

        stats = buffer.diversity_stats()

        self.assertAlmostEqual(float(stats["root_mask_rate"]), 0.3, places=6)
        self.assertAlmostEqual(float(stats["post_veto_to_hold_rate"]), 0.1, places=6)
        self.assertAlmostEqual(float(stats["veto_to_hold_rate"]), 0.1, places=6)
        self.assertEqual(float(stats["root_mask_blocked_buy_total"]), 2.0)
        self.assertEqual(float(stats["root_mask_blocked_sell_total"]), 1.0)

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

        agent = JAXMuZeroAgent(config)
        env = TrackingEnvironment(symbol="XAUUSD", config=config, max_steps=1)
        agent.play_game(env, exploration=True)

        self.assertTrue(env.root_policy_called)

    def test_reanalyze_rewrites_policy_and_value_shapes(self) -> None:
        """Reecrit politiques et valeurs d'un episode sans casser les longueurs."""

        self._require_jax_stack()

        from eva_lab.muzero.config import MuZeroConfigV3
        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

    def test_sanitize_metrics_keeps_text_scalars(self) -> None:
        """Conserve les metriques texte sans les convertir en flottants."""

        self._require_jax_stack()

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

        agent = JAXMuZeroAgent(config)
        env = TradingEnvironment(symbol="XAUUSD", config=config, max_steps=1)
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

    def test_policy_precheck_window_accepts_balanced_five_x_profile(self) -> None:
        """Valide une fenetre `5.x` stable avec friction faible et flux equilibre."""

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
                "loss_pol": 5.35,
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

    def test_policy_precheck_window_allows_screen_only_profile(self) -> None:
        """Autorise le screen si la policy est proche de la cible sans etre `full_ready`."""

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
                "loss_pol": 6.25,
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

    def test_recent_screen_selection_prefers_best_recent_window(self) -> None:
        """Choisit les checkpoints autour de la meilleure fenetre recente, pas du dernier step."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_recent_steps=2500,
            arena_screen_candidate_count=5,
            arena_screen_window_size=500,
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

    def test_screen_winner_gate_accepts_clean_balanced_candidate(self) -> None:
        """Autorise la full Arena si le gagnant du screen est propre."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.35,
                        "return_pct": 0.08,
                        "expectancy_pct": 0.01,
                        "positive_episode_rate": 61.0,
                        "directional_bias": "balanced",
                    }
                }
            },
            config,
        )

        self.assertTrue(verdict["allowed"])
        self.assertEqual(verdict["status"], "eligible")

    def test_screen_winner_gate_blocks_negative_expectancy_candidate(self) -> None:
        """Refuse la full Arena si le gagnant du screen reste negatif en expectancy."""

        self._require_jax_stack()
        module = self._load_train_global_models_module()
        config = SimpleNamespace(
            arena_screen_min_profit_factor=1.20,
            arena_screen_min_return_pct=0.0,
            arena_screen_min_expectancy_pct=0.0,
            arena_screen_min_positive_episode_rate=55.0,
        )
        verdict = module._evaluate_screen_winner_gate(
            {
                "challenger": {
                    "metrics": {
                        "profit_factor": 1.42,
                        "return_pct": 0.04,
                        "expectancy_pct": 0.0,
                        "positive_episode_rate": 58.0,
                        "directional_bias": "balanced",
                    }
                }
            },
            config,
        )

        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["reason"], "expectancy_pct")


if __name__ == "__main__":
    unittest.main()
