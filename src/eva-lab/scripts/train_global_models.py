"""
Train Global Models (Sprint 16) — THE HIVE EVA-Lab

Ce script orchestre l'entraînement massif du modèle JAX MuZero / DreamerV3 
sur l'ensemble de l'univers d'actifs globaux (27 symboles).

Il effectue les actions suivantes :
1. Initialize un Replay Buffer partagé.
2. Itère sur TOUS les symboles actifs pour générer des données synthétiques 
   via du Self-Play intensif (Phase 1: Exploration & Data Collection).
3. Lance une phase d'entraînement profond (Deep Training) sur la base de 
   ce buffer multi-actifs pour apprendre les corrélations (Phase 2).
4. Évalue et sauvegarde le modèle final (Phase 3).
"""

import os
import sys
import logging
import jax
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src", "eva-lab"))
sys.path.append(os.path.join(os.getcwd(), "src", "shared"))

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.muzero.environment import TradingEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("GlobalTrainer")


def main():
    logger.info("==================================================")
    logger.info("🚀 STARTING GLOBAL TRAINING - MUZERO/DREAMERV3 🧠")
    logger.info("==================================================")
    logger.info(f"Hardware Devices: {jax.devices()}")

    # 1. Setup global configuration
    config = MuZeroConfigV3()
    # 🔥 RTX 3090 FE — Maximum Overnight Training Parameters
    # training_steps is cumulative (checkpoint resumes from latest)
    config.training_steps = 50_000    # 50k gradient steps per nightly session
    config.num_simulations = 200      # Deep MCTS planning (was 50)
    config.batch_size = 512           # Saturate GPU VRAM (was 32)
    config.checkpoint_interval = 500  # Checkpoint every 500 steps
    config.num_unroll_steps = 10      # Longer rollouts for better world model
    config.td_steps = 20              # Deeper temporal credit assignment
    
    symbols = config.symbols
    num_symbols = len(symbols)
    logger.info(f"🌍 Univers d'actifs: {num_symbols} symboles détectés.")
    logger.info(f"📝 Symboles: {symbols}")
    
    agent = JAXMuZeroAgent(config)
    
    # Try restoring weights to resume training if possible
    weights_path = os.path.join(config.weights_path, "muzero_global_latest.pkl")
    if os.path.exists(weights_path):
        logger.info(f"♻️  Reprise de l'entraînement à partir de: {weights_path}")
        try:
            agent.load(weights_path)
        except Exception as e:
            logger.warning(f"Impossible de charger les poids existants: {e}")
            logger.info("Démarrage d'un entraînement de zéro.")
    
    # 2. Phase 1: Warming Up Replay Buffer (Massive Self-Play)
    logger.info("==================================================")
    logger.info("🎮 PHASE 1: COLLECTE DE DONNEES (SELF-PLAY MULTI-ACTIFS)")
    logger.info("==================================================")
    
    games_per_symbol = 20          # 🔥 Rich replay buffer (was 3)
    total_games = num_symbols * games_per_symbol
    games_played = 0
    
    logger.info(f"  → {games_per_symbol} games × {num_symbols} symbols = {total_games} episodes")

    for symbol in symbols:
        logger.info(f"📊 Collecte de données pour: {symbol}")
        # Create a tiny localized environment for this symbol
        env = TradingEnvironment(symbol=symbol, config=config, max_steps=100)
        
        for i in range(games_per_symbol):
            history = agent.play_game(env, exploration=True)
            games_played += 1
            logger.info(
                f"  [{games_played}/{total_games}] {symbol} Ep {i+1} "
                f"| Return: {history.info[-1].get('equity', 0):.2f}$ "
                f"| Buffer: {agent.replay_buffer.size}"
            )

    logger.info(f"✅ Replay Buffer rempli avec {agent.replay_buffer.size} transitions multi-actifs.")

    # 3. Phase 2: Deep Training Loop
    logger.info("==================================================")
    logger.info(f"🏋️ PHASE 2: DEEP TRAINING ({config.training_steps} Steps)")
    logger.info("==================================================")
    
    start_time = datetime.now()
    
    for step in range(1, config.training_steps + 1):
        metrics = agent.train_step()
        
        if metrics:
            if step % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = step / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Epoch {step:05d}/{config.training_steps} "
                    f"| Loss: {metrics['loss_total']:.4f} "
                    f"| Policy: {metrics.get('loss_policy', 0):.4f} "
                    f"| Value: {metrics.get('loss_value', 0):.4f} "
                    f"| ({speed:.1f} steps/s)"
                )
            
            # Save checkpoints periodically
            if step % config.checkpoint_interval == 0:
                ckpt_path = os.path.join(config.weights_path, f"muzero_global_ckpt_{step}.pkl")
                agent.save(ckpt_path)
                logger.info(f"💾 Checkpoint sauvegardé: {ckpt_path}")
        else:
            logger.warning("Agent did not train (is the buffer empty?)")
            break

    total_time = datetime.now() - start_time
    logger.info(f"✅ Entraînement Deep Learning complété en {total_time}.")

    # 4. Phase 3: Evaluation and Arena Battle
    logger.info("==================================================")
    logger.info("🔬 PHASE 3: EVALUATION & ARENA BATTLE")
    logger.info("==================================================")
    
    # Save Challenger Models
    challenger_id = f"gen_challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    challenger_path = os.path.join(config.weights_path, f"{challenger_id}.pkl")
    agent.save(challenger_path)
    logger.info(f"💾 Modèle Challenger exporté: {challenger_path}")
    
    logger.info("🏆 Lancement de l'Arena Darwinienne...")
    
    from eva_lab.arena import Arena
    from eva_lab.genetic_updater import GeneticUpdater
    import shutil
    
    genetic = GeneticUpdater()
    arena = Arena()
    
    champion_id = genetic.get_champion()
    
    # Combat dans l'arène
    battle_report = arena.battle(challenger_id, champion_id)
    
    if battle_report["outcome"] == "VICTORY":
        logger.info(f"👑 LE CHALLENGER L'EMPORTE ! Enregistrement du nouvel ADN...")
        genetic.register_new_generation(
            gen_id=challenger_id,
            metrics=battle_report["challenger"]["metrics"],
            is_champion=True
        )
        # Faire une copie universelle pour le chargement rapide
        champion_path = os.path.join(config.weights_path, "muzero_champion.pkl")
        shutil.copy2(challenger_path, champion_path)
        logger.info(f"💾 Nouveau Champion déployé: {champion_path}")
    else:
        logger.info(f"🗑️ Le Challenger a perdu. L'ADN est rejeté.")
    logger.info("🎯 GLOBAL TRAINING PROCESS FINISHED.")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
