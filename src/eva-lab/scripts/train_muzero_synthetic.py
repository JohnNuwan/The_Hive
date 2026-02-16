"""
Synthetic Training Script for JAX MuZero — THE HIVE

This script validates the JAX engine by training on the synthetic trading env.
"""

import os
import sys
import logging
import jax
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src", "eva-lab"))

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.muzero.environment import TradingEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🎬 Starting Synthetic MuZero JAX Training...")
    logger.info(f"Devices: {jax.devices()}")

    # 1. Setup
    config = MuZeroConfigV3()
    config.batch_size = 16 # Small for test
    config.num_simulations = 20 # Fast for test
    config.training_steps = 100
    
    agent = JAXMuZeroAgent(config)
    env = TradingEnvironment(symbol="XAUUSD", config=config, max_steps=100)
    
    # 2. Initial Self-Play to fill buffer
    logger.info("🎮 Phase 1: Warming up Replay Buffer...")
    for i in range(3):
        logger.info(f"  Episode {i+1}/3...")
        history = agent.play_game(env, exploration=True)
        logger.info(f"  Done. Last Equity: {history.info[-1]['equity']:.2f} | Buffer: {agent.replay_buffer.size}")

    # 3. Training Loop
    logger.info("🏋️ Phase 2: Training...")
    for step in range(config.training_steps):
        metrics = agent.train_step()
        if metrics:
            if step % 10 == 0:
                logger.info(f"  Step {step:03d} | Loss: {metrics['loss_total']:.4f} | Val: {metrics['loss_val']:.4f}")
        else:
            logger.warning("  No data to train on yet.")
            # Play one more game to get data
            agent.play_game(env, exploration=True)

    # 4. Final Evaluation
    logger.info("📊 Phase 3: Evaluation...")
    history = agent.play_game(env, exploration=False)
    summary = env.get_summary()
    logger.info(f"FINISH. Return: {summary['return_pct']:.2f}% | Trades: {summary['total_trades']}")
    
    # 5. Save
    agent.save("data/muzero/weights/test_synthetic.pkl")
    logger.info("✅ Synthetic Validation Complete.")

if __name__ == "__main__":
    main()
