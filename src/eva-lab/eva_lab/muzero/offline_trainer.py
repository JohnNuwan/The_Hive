"""Entraine DreamerV3 hors ligne a partir des historiques de marche."""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.dreamer_networks import make_dreamer_networks
from eva_lab.muzero.dreamer_trainer import DreamerTrainerJAX
from eva_lab.muzero.replay_buffer import GameHistory, PrioritizedReplayBuffer
from eva_lab.shadow_dataset import load_shadow_games
from shared.indicators import IndicatorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfflineTrainer:
    """Charge les historiques et orchestre le pre-entrainement DreamerV3."""

    def __init__(self, data_dir: str = "data/history") -> None:
        """Initialise la configuration Dreamer et le buffer hors ligne.

        Args:
            data_dir (str): Dossier contenant les historiques CSV.
        """
        self.data_dir = data_dir
        hidden_dims = [
            int(value.strip())
            for value in os.getenv("DREAMER_NETWORK_HIDDEN_DIMS", "256,256").split(",")
            if value.strip()
        ]
        self.sequence_length = int(os.getenv("DREAMER_SEQUENCE_LENGTH", "64"))
        self.sequence_stride = int(
            os.getenv("DREAMER_SEQUENCE_STRIDE", str(max(16, self.sequence_length // 2)))
        )
        self.replay_capacity = int(os.getenv("DREAMER_REPLAY_MAX_GAMES", "2500"))
        self.shadow_data_dirs = [
            item.strip()
            for item in os.getenv("DREAMER_SHADOW_DATA_DIRS", "data/shadow_learning").split(os.pathsep)
            if item.strip()
        ]
        self.config = MuZeroConfigV3(
            batch_size=int(os.getenv("DREAMER_BATCH_SIZE", "8")),
            hidden_state_size=int(os.getenv("DREAMER_HIDDEN_STATE_SIZE", "128")),
            num_unroll_steps=int(os.getenv("DREAMER_NUM_UNROLL_STEPS", "3")),
            network_hidden_dims=hidden_dims or [256, 256],
        )
        self.config.dreamer_max_start_states = int(os.getenv("DREAMER_MAX_START_STATES", "256"))
        self.transformed = make_dreamer_networks(self.config)
        self.trainer = DreamerTrainerJAX(self.config, self.transformed)
        self.replay_buffer = PrioritizedReplayBuffer(max_games=self.replay_capacity)

        sample_obs = np.zeros((1, *self.config.observation_shape))
        self.params, _ = self.trainer.init_params(sample_obs)
        self.trainer.params["wm"] = self.params

    def load_and_process_data(self) -> None:
        """Charge les CSV, calcule les indicateurs et construit les episodes."""
        files = glob.glob(f"{self.data_dir}/*.csv")
        logger.info("Fichiers historiques detectes: %s.", len(files))

        total_steps = 0
        for file in files:
            symbol = os.path.basename(file).split("_")[0]
            logger.info("Traitement de %s depuis %s...", symbol, file)

            df = pd.read_csv(file)
            try:
                df["rsi"] = IndicatorFactory.rsi(df["close"], 14)

                macd_res = IndicatorFactory.macd(df["close"])
                df["macd"] = macd_res["macd"]
                df["macd_signal"] = macd_res["signal"]
                df["macd_hist"] = macd_res["histogram"]

                df["vwap"] = IndicatorFactory.vwap(
                    df["high"],
                    df["low"],
                    df["close"],
                    df["tick_volume"],
                )
                df["obv"] = IndicatorFactory.obv(df["close"], df["tick_volume"])
                df["momentum"] = IndicatorFactory.momentum(df["close"])
                df["trix"] = IndicatorFactory.trix(df["close"])

                stoch_res = IndicatorFactory.stochastic(df["high"], df["low"], df["close"])
                df["stoch_k"] = stoch_res["percent_k"]
                df["stoch_d"] = stoch_res["percent_d"]

                df["cci"] = IndicatorFactory.cci(df["high"], df["low"], df["close"])
                adx_res = IndicatorFactory.adx(df["high"], df["low"], df["close"])
                df["adx"] = adx_res["adx"]
                df["adx_plus_di"] = adx_res["plus_di"]
                df["adx_minus_di"] = adx_res["minus_di"]

                ichi_res = IndicatorFactory.ichimoku(df["high"], df["low"], df["close"])
                df["ichi_tenkan"] = ichi_res["tenkan_sen"]
                df["ichi_kijun"] = ichi_res["kijun_sen"]
                df["ichi_senkou_a"] = ichi_res["senkou_span_a"]
                df["ichi_senkou_b"] = ichi_res["senkou_span_b"]

                # Le remplissage inverse stabilise le debut de serie sans FutureWarning Pandas.
                df = df.bfill().fillna(0.0)
            except Exception as exc:
                logger.error("Calcul des indicateurs impossible pour %s: %s", file, exc)
                continue

            segment_length = self.sequence_length
            closes_seg = df["close"].values

            for start_idx in range(0, len(df) - segment_length, self.sequence_stride):
                end_idx = start_idx + segment_length
                if end_idx > len(df):
                    break

                seg_closes = closes_seg[start_idx:end_idx]
                game = GameHistory()

                # La politique aleatoire diversifie le pre-entrainement avant le live.
                actions = np.random.choice([0, 1, 2], size=segment_length, p=[0.4, 0.3, 0.3])

                initial_balance = 10000.0
                balance = initial_balance
                peak_balance = initial_balance
                position = 0
                entry_price = 0.0

                for i in range(segment_length):
                    idx = start_idx + i
                    price = seg_closes[i]

                    obs_vec = np.zeros(self.config.observation_shape)
                    obs_vec[0] = price / 3000.0
                    obs_vec[1] = df["rsi"].values[idx] / 100.0

                    # On conserve l'ordre des indicateurs du banker pour aligner le pre-train.
                    features_list = [
                        df["rsi"].values[idx],
                        df["macd_hist"].values[idx],
                        df["macd_signal"].values[idx],
                        df["vwap"].values[idx],
                        df["obv"].values[idx] / 10000.0,
                        df["momentum"].values[idx],
                        df["trix"].values[idx],
                        df["stoch_k"].values[idx],
                        df["stoch_d"].values[idx],
                        df["cci"].values[idx],
                        df["adx"].values[idx],
                        df["adx_plus_di"].values[idx],
                        df["adx_minus_di"].values[idx],
                        df["ichi_tenkan"].values[idx],
                        df["ichi_kijun"].values[idx],
                        df["ichi_senkou_a"].values[idx],
                        df["ichi_senkou_b"].values[idx],
                    ]

                    for f_idx, f_val in enumerate(features_list):
                        if f_idx + 2 < self.config.observation_shape[0]:
                            obs_vec[f_idx + 2] = f_val

                    action_val = actions[i]
                    reward = 0.0
                    if i < segment_length - 1:
                        next_price = seg_closes[i + 1]
                        ret = (next_price - price) / price * 100

                        if action_val == 1:
                            reward = ret - 0.02
                            if position == 0:
                                position = 1
                                entry_price = price
                        elif action_val == 2:
                            reward = -ret - 0.02
                            if position == 0:
                                position = -1
                                entry_price = price
                        elif action_val == 0 and position != 0:
                            trade_pnl = (
                                (price - entry_price) / entry_price * 100
                                if position == 1
                                else (entry_price - price) / entry_price * 100
                            )
                            balance += balance * trade_pnl / 100
                            position = 0

                        if balance > peak_balance:
                            peak_balance = balance
                        drawdown_pct = (peak_balance - balance) / peak_balance * 100

                        if drawdown_pct >= 4.0:
                            reward -= 15.0

                    action_one_hot = np.zeros(self.config.action_space_size)
                    action_one_hot[action_val] = 1.0
                    game.store(obs_vec, action_one_hot, reward, [1 / 3] * 3, 0.0)

                self.replay_buffer.save_game(game)
                total_steps += segment_length

        shadow_games = load_shadow_games(
            self.shadow_data_dirs,
            observation_size=self.config.observation_shape[0],
            action_space_size=self.config.action_space_size,
        )
        for game in shadow_games:
            self.replay_buffer.save_game(game)
            total_steps += len(game)

        logger.info(
            "Episodes shadow charges: %s depuis %s.",
            len(shadow_games),
            self.shadow_data_dirs,
        )

        logger.info(
            "Episodes hors-ligne charges: %s (%s pas de temps).",
            self.replay_buffer.size,
            total_steps,
        )

    def train_loop(self, epochs: int = 5000) -> None:
        """Execute la boucle d'optimisation Dreamer.

        Args:
            epochs (int): Nombre d'epochs a executer.
        """
        logger.info("Demarrage de l'entrainement hors-ligne DreamerV3...")

        for epoch in range(epochs):
            loss_sum = 0.0
            steps = 0

            if self.replay_buffer.size == 0:
                raise RuntimeError("Le buffer Dreamer est vide. Aucun historique n'a ete charge.")

            effective_batch_size = min(self.config.batch_size, self.replay_buffer.size)
            updates_per_epoch = max(1, self.replay_buffer.size // effective_batch_size)

            for _ in range(updates_per_epoch):
                samples = self.replay_buffer.sample(effective_batch_size)
                games = [sample[0] for sample in samples]
                batch = self.trainer.prepare_batch(games)

                metrics = self.trainer.train_step(batch)
                loss_sum += float(metrics["loss_total"])
                steps += 1

            avg_loss = loss_sum / steps if steps > 0 else 0.0
            logger.info("Epoch %s/%s - Loss: %.4f", epoch + 1, epochs, avg_loss)

    def save_checkpoint(self, path: str = "checkpoints/offline_v1") -> None:
        """Sauvegarde un checkpoint local Dreamer.

        Args:
            path (str): Prefixe du fichier de sortie.
        """
        import pickle

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path + ".pkl", "wb") as file_obj:
            pickle.dump(self.trainer.params, file_obj)
        logger.info("Checkpoint Dreamer sauvegarde dans %s.pkl", path)


if __name__ == "__main__":
    epochs = int(os.getenv("DREAMER_EPOCHS", "5000"))
    trainer = OfflineTrainer()
    trainer.load_and_process_data()
    trainer.train_loop(epochs=epochs)
    trainer.save_checkpoint("data/checkpoints/dreamer_pretrained")
