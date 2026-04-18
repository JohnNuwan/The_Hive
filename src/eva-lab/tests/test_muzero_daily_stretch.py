"""Tests du bonus stretch journalier MuZero et de sa propagation Arena."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd

from eva_lab.muzero.environment import BUY, CLOSE, HOLD, TradingEnvironment
from eva_lab.training_utils import build_muzero_market_context


class MuZeroDailyStretchTests(unittest.TestCase):
    """Valide le bonus stretch journalier sans en faire une gate dure."""

    @staticmethod
    def _build_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
        """Construit un historique OHLCV minimal pour les tests.

        Args:
            index (pd.DatetimeIndex): Index temporel voulu.

        Returns:
            pd.DataFrame: Historique OHLCV compatible avec le loader MuZero.
        """
        base = np.linspace(100.0, 101.0, num=len(index), dtype=np.float32)
        return pd.DataFrame(
            {
                "open": base,
                "high": base + 0.1,
                "low": base - 0.1,
                "close": base,
                "tick_volume": np.full(len(index), 1000.0, dtype=np.float32),
                "spread": np.full(len(index), 0.01, dtype=np.float32),
            },
            index=index,
        )

    @staticmethod
    def _load_arena_class():
        """Charge `Arena` sans exiger la stack JAX complete pour ce test.

        Returns:
            type: Classe `Arena` importee avec des stubs minimums si besoin.
        """
        stub_dreamer = ModuleType("eva_lab.muzero.dreamer_agent")
        stub_dreamer.JAXDreamerAgent = object
        stub_muzero = ModuleType("eva_lab.muzero.jax_agent")
        stub_muzero.JAXMuZeroAgent = object

        with patch.dict(
            sys.modules,
            {
                "eva_lab.muzero.dreamer_agent": stub_dreamer,
                "eva_lab.muzero.jax_agent": stub_muzero,
            },
            clear=False,
        ):
            from eva_lab.arena import Arena  # pylint: disable=import-outside-toplevel

        return Arena

    @staticmethod
    def _build_env_config(
        *,
        timeframe: str = "M5",
        daily_bonus: float = 4.0,
        max_daily_drawdown_pct: float = 3.5,
    ) -> SimpleNamespace:
        """Construit une configuration MuZero minimale pour l'environnement.

        Args:
            timeframe (str): Timeframe principal du test.
            daily_bonus (float): Bonus stretch journalier voulu.
            max_daily_drawdown_pct (float): Drawdown journalier maximal autorise.

        Returns:
            SimpleNamespace: Configuration exploitable par l'environnement.
        """
        return SimpleNamespace(
            quality_trade_bonus=10.0,
            final_growth_bonus=50.0,
            final_growth_threshold=0.10,
            drawdown_time_penalty_rate=0.2,
            max_drawdown_penalty=10.0,
            loss_penalty_multiplier=2.0,
            slbe_activation_bonus=6.0,
            daily_stretch_target_pct=10.0,
            daily_stretch_max_drawdown_pct=max_daily_drawdown_pct,
            daily_stretch_reward_bonus=daily_bonus,
            horizon="scalp",
            primary_timeframe=timeframe,
            model_family=None,
        )

    @staticmethod
    def _build_market_rows(prices: list[float]) -> np.ndarray:
        """Construit une matrice de marche minimale a partir de prix.

        Args:
            prices (list[float]): Prix de reference par barre.

        Returns:
            np.ndarray: Matrice OHLCV simplifiee.
        """
        return np.asarray(
            [[price, price, price, price, 1000.0, price] for price in prices],
            dtype=np.float32,
        )

    def _build_env(
        self,
        *,
        prices: list[float],
        day_labels: list[str],
        daily_bonus: float = 4.0,
        max_daily_drawdown_pct: float = 3.5,
    ) -> TradingEnvironment:
        """Construit un environnement controle pour tester le bonus stretch.

        Args:
            prices (list[float]): Prix par barre.
            day_labels (list[str]): Jour reel de chaque barre.
            daily_bonus (float): Bonus stretch journalier voulu.
            max_daily_drawdown_pct (float): Drawdown journalier maximal autorise.

        Returns:
            TradingEnvironment: Environnement MuZero pret a jouer.
        """
        env = TradingEnvironment(
            data=self._build_market_rows(prices),
            day_labels=np.asarray(day_labels, dtype=object),
            symbol="XAUUSD",
            config=self._build_env_config(
                daily_bonus=daily_bonus,
                max_daily_drawdown_pct=max_daily_drawdown_pct,
            ),
            max_steps=max(1, len(prices) - 2),
        )
        env.reset()
        env.spec.trade_size = env.spec.initial_balance
        env.current_step = 0
        env.start_step = 0
        env.balance = env.spec.initial_balance
        env.peak_equity = env.spec.initial_balance
        env.position_size = 0.0
        env.avg_entry_price = 0.0
        env.equity_curve = [env.spec.initial_balance]
        env._active_day_label = str(env.day_labels[0])
        env._active_day_start_equity = env.spec.initial_balance
        env._active_day_peak_equity = env.spec.initial_balance
        env._active_day_trough_equity = env.spec.initial_balance
        env._active_day_max_drawdown_pct = 0.0
        env._active_day_last_equity = env.spec.initial_balance
        return env

    def test_build_muzero_day_labels_tracks_real_days_for_m5_h1_and_d1(self) -> None:
        """Conserve de vrais jours pour `M5`, `H1` et `D1`."""

        test_cases = {
            "M5": pd.date_range("2026-04-01 22:00:00", periods=300, freq="5min"),
            "H1": pd.date_range("2026-04-01 20:00:00", periods=30, freq="h"),
            "D1": pd.date_range("2026-04-01", periods=4, freq="D"),
        }

        for timeframe, index in test_cases.items():
            with self.subTest(timeframe=timeframe):
                frame = self._build_frame(index)
                market_data, day_labels = build_muzero_market_context(frame)
                expected_days = frame.index.normalize().strftime("%Y-%m-%d").to_numpy(dtype=object)

                self.assertEqual(market_data.shape[0], len(frame))
                self.assertEqual(day_labels.shape[0], len(frame))
                self.assertListEqual(day_labels.tolist(), expected_days.tolist())
                self.assertGreaterEqual(len(set(day_labels.tolist())), 1)

    def test_daily_stretch_bonus_triggers_when_day_reaches_target_without_excess_drawdown(self) -> None:
        """Declenche le bonus training sur une vraie journee a +10 % nette."""

        prices = [100.0, 100.0, 110.2, 110.2]
        day_labels = ["2026-04-01"] * len(prices)

        env_with_bonus = self._build_env(prices=prices, day_labels=day_labels, daily_bonus=4.0)
        env_without_bonus = self._build_env(prices=prices, day_labels=day_labels, daily_bonus=0.0)

        env_with_bonus.step(BUY)
        _, reward_with_bonus, done_with_bonus, _, _ = env_with_bonus.step(CLOSE)
        summary_with_bonus = env_with_bonus.get_summary()

        env_without_bonus.step(BUY)
        _, reward_without_bonus, done_without_bonus, _, _ = env_without_bonus.step(CLOSE)
        summary_without_bonus = env_without_bonus.get_summary()

        self.assertTrue(done_with_bonus)
        self.assertTrue(done_without_bonus)
        self.assertAlmostEqual(reward_with_bonus - reward_without_bonus, 4.0, places=4)
        self.assertEqual(summary_with_bonus["days_above_10pct"], 1)
        self.assertGreaterEqual(summary_with_bonus["best_day_net_return_pct"], 10.0)
        self.assertLessEqual(summary_with_bonus["daily_max_drawdown_pct"], 3.5)
        self.assertEqual(summary_without_bonus["days_above_10pct"], 1)

    def test_daily_stretch_bonus_does_not_trigger_when_drawdown_is_too_high(self) -> None:
        """Refuse le bonus stretch si la grosse journee degrade trop le drawdown."""

        env = self._build_env(
            prices=[100.0, 100.0, 80.0, 110.2, 110.2],
            day_labels=["2026-04-01"] * 5,
            daily_bonus=4.0,
            max_daily_drawdown_pct=3.5,
        )

        env.step(BUY)
        env.step(HOLD)
        _, _, done, _, _ = env.step(CLOSE)
        summary = env.get_summary()

        self.assertTrue(done)
        self.assertGreaterEqual(summary["best_day_net_return_pct"], 10.0)
        self.assertEqual(summary["days_above_10pct"], 0)
        self.assertGreater(summary["daily_max_drawdown_pct"], 3.5)

    def test_arena_scoring_keeps_robust_profile_ahead_of_casino_profile(self) -> None:
        """Le bonus stretch ne doit pas faire gagner un profil casino."""

        robust_metrics = {
            "return_pct": 0.8,
            "profit_factor": 1.4,
            "expectancy_pct": 0.08,
            "win_rate": 60.0,
            "positive_episode_rate": 80.0,
            "max_drawdown_pct": 2.0,
            "best_day_net_return_pct": 4.0,
            "days_above_10pct": 0,
        }
        casino_metrics = {
            "return_pct": 1.5,
            "profit_factor": 1.05,
            "expectancy_pct": 0.02,
            "win_rate": 55.0,
            "positive_episode_rate": 70.0,
            "max_drawdown_pct": 12.0,
            "best_day_net_return_pct": 12.5,
            "days_above_10pct": 1,
        }

        with patch.dict(
            os.environ,
            {
                "MUZERO_DAILY_STRETCH_TARGET_PCT": "10.0",
                "ARENA_DAILY_STRETCH_SCORE_BONUS": "4.0",
            },
            clear=False,
        ):
            Arena = self._load_arena_class()
            robust_score = Arena._score_metrics(robust_metrics)
            casino_score = Arena._score_metrics(casino_metrics)

        self.assertGreater(robust_score, casino_score)


if __name__ == "__main__":
    unittest.main()
