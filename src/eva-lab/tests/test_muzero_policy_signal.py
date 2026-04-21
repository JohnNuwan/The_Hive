"""Tests des correctifs de signal `policy` pour MuZero."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from eva_lab.muzero.environment import BUY, CLOSE, HOLD, SELL, SPLIT, TradingEnvironment
from eva_lab.muzero.replay_buffer import GameHistory, PrioritizedReplayBuffer


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

    def test_reward_policy_aliases_support_v2_keys(self) -> None:
        """Resolve correctement les cles V2 de recompense."""

        resolved = TradingEnvironment._resolve_reward_policy_terms(
            {
                "realized_reward_multiplier": 1.3,
                "close_realized_bonus_multiplier": 1.7,
                "split_realized_bonus_multiplier": 1.2,
                "hold_drag_penalty_multiplier": 0.4,
            }
        )

        self.assertAlmostEqual(resolved["realized_reward_multiplier"], 1.3)
        self.assertAlmostEqual(resolved["close_realized_multiplier"], 1.7)
        self.assertAlmostEqual(resolved["split_realized_multiplier"], 1.2)
        self.assertAlmostEqual(resolved["hold_drag_multiplier"], 0.4)

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


if __name__ == "__main__":
    unittest.main()
