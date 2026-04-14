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
