import logging
import os
import glob
import pandas as pd
import numpy as np
import jax
import time
from datetime import datetime

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.dreamer_trainer import DreamerTrainerJAX
from eva_lab.muzero.dreamer_networks import make_dreamer_networks
from eva_lab.muzero.replay_buffer import PrioritizedReplayBuffer, GameHistory
from shared.indicators import IndicatorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineTrainer:
    def __init__(self, data_dir="data/history"):
        self.data_dir = data_dir
        self.config = MuZeroConfigV3()
        self.transformed = make_dreamer_networks(self.config)
        self.trainer = DreamerTrainerJAX(self.config, self.transformed)
        self.replay_buffer = PrioritizedReplayBuffer(max_games=5000) # Larger buffer for history
        
        # Init params
        sample_obs = np.zeros((1, *self.config.observation_shape))
        self.params, _ = self.trainer.init_params(sample_obs)
        self.trainer.params["wm"] = self.params

    def load_and_process_data(self):
        """Loads CSVs, computes features, creates GameHistories."""
        files = glob.glob(f"{self.data_dir}/*.csv")
        logger.info(f"Found {len(files)} historical files.")
        
        total_steps = 0
        
        for file in files:
            symbol = os.path.basename(file).split('_')[0]
            logger.info(f"Processing {symbol} from {file}...")
            
            df = pd.read_csv(file)
            
            # Compute Indicators
            closes = df['close'].tolist()
            highs = df['high'].tolist()
            lows = df['low'].tolist()
            volumes = df['tick_volume'].tolist()
            
            # Batch computation if possible
            # New Vectorized IndicatorFactory expects Pandas Series
            
            # Pre-compute columns (Vectorized = Instant)
            try:
                # RSI
                df['rsi'] = IndicatorFactory.rsi(df['close'], 14)
                
                # MACD
                macd_res = IndicatorFactory.macd(df['close'])
                df['macd'] = macd_res['macd']
                df['macd_signal'] = macd_res['signal']
                df['macd_hist'] = macd_res['histogram']
                
                # ATR
                df['atr'] = IndicatorFactory.atr(df['high'], df['low'], df['close'], 14)
                
                # Bollinger
                bb_res = IndicatorFactory.bollinger_bands(df['close'])
                df['bb_width'] = bb_res['width']
                df['bb_pct'] = bb_res['pct_b']
                
                # Rel Vol
                df['rvol'] = IndicatorFactory.relative_volume(df['tick_volume'])
                
                # Fill NaN from rolling windows (start of file)
                df = df.fillna(method='bfill').fillna(0.0)
                
            except Exception as e:
                logger.error(f"Error computing indicators for {file}: {e}")
                continue
            
            # Simulating episodes with Synthetic Actions/Rewards
            segment_length = 200 # 200 steps per game
            
            # Vectorized operations for speed
            closes_seg = df['close'].values
            volumes_seg = df['tick_volume'].values
            
            for start_idx in range(0, len(df) - segment_length, segment_length):
                end_idx = start_idx + segment_length
                if end_idx > len(df): break
                
                seg_closes = closes_seg[start_idx:end_idx]
                seg_vols = volumes_seg[start_idx:end_idx]
                
                game = GameHistory()
                
                # Generate random actions for the whole segment to teach the model
                # about consequences of actions.
                # We want balanced classes: 40% Hold, 30% Buy, 30% Sell
                # We want balanced classes: 40% Hold, 30% Buy, 30% Sell
                actions = np.random.choice([0, 1, 2], size=segment_length, p=[0.4, 0.3, 0.3])
                
                for i in range(segment_length):
                    price = seg_closes[i]
                    vol = seg_vols[i]
                    
                    obs_vec = np.zeros(self.config.observation_shape)
                    obs_vec[0] = price
                    obs_vec[1] = np.log1p(vol)
                    # TODO: Add more indicators here efficiently
                    
                    action_val = actions[i]
                    
                    # Reward Calculation
                    reward = 0.0
                    if i < segment_length - 1:
                        next_price = seg_closes[i+1]
                        ret = np.log(next_price / price) * 100 # Percentage return approx
                        
                        if action_val == 1: # BUY
                            reward = ret - 0.02 # Spread cost
                        elif action_val == 2: # SELL
                            reward = -ret - 0.02 # Spread cost
                        elif action_val == 0: # HOLD
                            reward = 0.0 # No cost, no gain
                            
                    # One-hot action for storage
                    # But replay_buffer.store expects integer action? 
                    # Check params: buffer.store(obs, action, reward, ...)
                    # Usually action is integer or one-hot?
                    # In MuZero GameHistory, action is stored as is.
                    # DreamerTrainerJAX.prepare_batch converts it.
                    # Let's check prepare_batch... it calls jnp.array(actions).
                    # If we store int, it becomes int array.
                    # But Dreamer model expects One-Hot in standard implementations.
                    # My implementation in dreamer_trainer.py:
                    # act_t = jnp.transpose(batch.actions, (1, 0, 2))
                    # It expects [B, T, ActionDim].
                    # So we MUST store ONE-HOT actions in GameHistory!
                    
                    action_one_hot = np.zeros(self.config.action_space_size)
                    action_one_hot[action_val] = 1.0
                    
                    game.store(obs_vec, action_one_hot, reward, [1/3]*3, 0.0)
            
                self.replay_buffer.save_game(game)
                total_steps += segment_length
            
        logger.info(f"Loaded {self.replay_buffer.size} episodes ({total_steps} steps).")

    def train_loop(self, epochs=5000):
        logger.info("🎓 Starting Offline Training...")
        
        for epoch in range(epochs):
            loss_sum = 0
            steps = 0
            
            # Iterate through buffer? No, random sampling.
            # Number of updates per epoch = buffer_size / batch_size
            updates_per_epoch = int(self.replay_buffer.size / self.config.batch_size)
            
            for _ in range(updates_per_epoch):
                samples = self.replay_buffer.sample(self.config.batch_size)
                # Unpack (game, start_idx, idx) -> game
                games = [s[0] for s in samples]
                batch = self.trainer.prepare_batch(games)
                
                metrics = self.trainer.train_step(batch)
                loss_sum += metrics["loss_total"]
                steps += 1
                
            avg_loss = loss_sum / steps if steps > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
    def save_checkpoint(self, path="checkpoints/offline_v1"):
        # Save params via orbax or pickle for now
        import pickle
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.trainer.params, f)
        logger.info(f"Saved checkpoint to {path}.pkl")

if __name__ == "__main__":
    trainer = OfflineTrainer()
    trainer.load_and_process_data()
    trainer.train_loop(epochs=5000) # Hunger Mode: Massive heavy training
    trainer.save_checkpoint("data/checkpoints/dreamer_pretrained")
