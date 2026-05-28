"""Entraînement auto-supervisé du modèle Market-JEPA via VICReg."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Configuration du PYTHONPATH pour EVA Lab
package_root = Path(__file__).resolve().parents[1]
shared_root = package_root.parent / 'shared'
for candidate in (package_root, shared_root):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.jepa_encoder import make_jepa_networks, compute_vicreg_loss
from scripts.train_global_models import build_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_jepa")


def collect_jepa_trajectories(
    config: MuZeroConfigV3,
    symbols: list[str],
    num_episodes_per_symbol: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Collecte des trajectoires d'observations réelles depuis le TradingEnvironment.

    Prend des paires d'observations temporelles décalées (t, t + k) pour entraîner JEPA.
    """
    logger.info("Collecte de trajectoires d'observations réelles sur %s...", symbols)
    
    obs_t_list = []
    obs_future_list = []
    k_shift = 5  # Nombre d'étapes de décalage temporel pour la prédiction cible JEPA

    for symbol in symbols:
        env = build_environment(symbol, config, for_collection=False)
        if env is None:
            logger.warning("Impossible d'initialiser l'environnement pour %s, ignoré.", symbol)
            continue

        for _ in range(num_episodes_per_symbol):
            obs, _ = env.reset()
            episode_obs = [obs]
            done = False
            steps = 0
            
            # Exécution de pas aléatoires pour générer de la diversité
            while not done and steps < config.max_moves:
                action = np.random.randint(0, config.action_space_size)
                next_obs, _, done, _, _ = env.step(action)
                episode_obs.append(next_obs)
                obs = next_obs
                steps += 1

            # Construction des paires (t, t+k)
            episode_len = len(episode_obs)
            for t in range(episode_len - k_shift):
                obs_t_list.append(episode_obs[t])
                obs_future_list.append(episode_obs[t + k_shift])

    if not obs_t_list:
        raise RuntimeError("Aucune trajectoire collectée. Vérifiez vos fichiers de données historiques.")

    return np.asarray(obs_t_list, dtype=np.float32), np.asarray(obs_future_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Pré-entraînement auto-supervisé Market-JEPA.")
    parser.add_argument("--steps", type=int, default=3000, help="Nombre de pas d'optimisation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage.")
    parser.add_argument("--batch-size", type=int, default=256, help="Taille du batch.")
    parser.add_argument("--episodes", type=int, default=8, help="Nombre d'episodes de collecte par symbole.")
    parser.add_argument("--smoke-test", action="store_true", help="Active un run rapide de test.")
    args = parser.parse_args()

    # Si smoke test, on limite grandement l'entraînement
    total_steps = 50 if args.smoke_test else args.steps
    batch_size = 64 if args.smoke_test else args.batch_size
    episodes_per_symbol = 2 if args.smoke_test else args.episodes

    # Charger la configuration et l'univers de symboles
    config = MuZeroConfigV3()
    symbols = config.symbols
    
    # 1. Collecte des données
    obs_t, obs_future = collect_jepa_trajectories(
        config,
        symbols,
        num_episodes_per_symbol=episodes_per_symbol,
    )
    logger.info("Données collectées : %d échantillons pour l'entraînement JEPA.", obs_t.shape[0])

    # 2. Définition du réseau JEPA
    latent_dim = config.jepa_latent_size
    hidden_dims = [256, 256]
    jepa_net = make_jepa_networks(latent_dim, hidden_dims)

    # Initialisation des paramètres de JEPA
    key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, obs_t.shape[1]))  # Dynamiquement adapté aux dimensions réelles
    params = jepa_net.init(key, dummy_obs, dummy_obs)

    # 3. Optimiseur et étape d'apprentissage
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch):
        def loss_fn(p):
            z_x, z_y, z_y_hat = jepa_net.apply(p, None, x_batch, y_batch)
            loss, metrics = compute_vicreg_loss(
                z_x, z_y, z_y_hat,
                sim_weight=25.0,
                var_weight=25.0,
                cov_weight=1.0,
            )
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(params)
        updates, next_opt_state = tx.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss, metrics

    # 4. Boucle d'entraînement
    logger.info("Démarrage du pré-entraînement JEPA avec la perte VICReg (%d étapes)...", total_steps)
    num_samples = obs_t.shape[0]

    for step in range(1, total_steps + 1):
        indices = np.random.choice(num_samples, batch_size, replace=True)
        x_batch = jnp.array(obs_t[indices])
        y_batch = jnp.array(obs_future[indices])

        params, opt_state, loss, metrics = train_step(params, opt_state, x_batch, y_batch)

        if step % 200 == 0 or step == 1 or step == total_steps:
            logger.info(
                "Étape %d/%d | Perte Totale: %.4f | Inv: %.4f | Var: %.4f | Cov: %.4f | Écart-Type Latent: %.3f",
                step,
                total_steps,
                float(loss),
                float(metrics["loss_invariance"]),
                float(metrics["loss_variance"]),
                float(metrics["loss_covariance"]),
                float(metrics["mean_std_z_x"]),
            )

    # 5. Sauvegarde des poids
    weights_dir = Path("data/muzero/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "jepa_encoder_latest.pkl"
    
    logger.info("Sauvegarde des poids JEPA dans %s...", weights_path)
    with open(weights_path, "wb") as f:
        pickle.dump(params, f)

    logger.info("Pré-entraînement Market-JEPA terminé avec succès !")


if __name__ == "__main__":
    main()
