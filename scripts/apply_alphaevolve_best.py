#!/usr/bin/env python3
"""
THE HIVE — AlphaEvolve Feedback Bridge (Live Bridging)
This script reads the best genome from the most recent AlphaEvolve offline campaign
and applies it to the active Master Banker local environment configuration.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apply_alphaevolve_best")

WORKDIR = Path(__file__).resolve().parents[1]
CAMPAIGNS_DIR = WORKDIR / "data" / "alphaevolve" / "campaigns"
ENV_BANKER_PATH = WORKDIR / ".env.banker.master.local"
ENV_PATH = WORKDIR / ".env"

PARAM_MAPPING = {
    "split_window_activation_bonus": "MUZERO_SPLIT_WINDOW_ACTIVATION_BONUS",
    "runner_window_hold_bonus": "MUZERO_RUNNER_HOLD_CAPTURE_BONUS",
    "pyramid_window_activation_bonus": "MUZERO_PYRAMID_WINDOW_ACTIVATION_BONUS",
    "missed_window_penalty": "MUZERO_MISSED_WINDOW_PENALTY",
    "giveback_soft_penalty": "MUZERO_RUNNER_GIVEBACK_SOFT_PENALTY",
    "giveback_hard_penalty": "MUZERO_RUNNER_GIVEBACK_HARD_PENALTY",
    "muzero_collection_num_simulations_xauusd": "MUZERO_COLLECTION_NUM_SIMULATIONS_XAUUSD",
    "muzero_collection_max_moves_xauusd": "MUZERO_COLLECTION_MAX_MOVES_XAUUSD",
    "muzero_collection_max_episode_seconds_xauusd": "MUZERO_COLLECTION_MAX_EPISODE_SECONDS_XAUUSD",
}

def load_latest_campaign() -> dict[str, Any] | None:
    """Finds and loads the latest AlphaEvolve campaign JSON file."""
    if not CAMPAIGNS_DIR.exists():
        logger.warning(f"AlphaEvolve campaigns directory does not exist: {CAMPAIGNS_DIR}")
        return None
    
    files = sorted(CAMPAIGNS_DIR.glob("alphaevolve_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        logger.warning("No AlphaEvolve campaigns found.")
        return None
    
    latest_file = files[0]
    logger.info(f"Loading latest AlphaEvolve campaign from {latest_file.name}...")
    try:
        return json.loads(latest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"Failed to read campaign file {latest_file.name}: {exc}")
        return None

def find_best_variant(campaign: dict[str, Any]) -> dict[str, Any] | None:
    """Finds the best variant based on score_arena, score_nemesis, or score_proxy."""
    variants = campaign.get("variants", [])
    if not variants:
        logger.warning("No variants found in campaign.")
        return None
    
    best_variant = None
    best_score = -999999.0
    
    for variant in variants:
        # Determine score
        score = -999999.0
        for key in ["score_arena", "score_nemesis", "score_proxy"]:
            val = variant.get(key)
            if val is not None:
                score = float(val)
                break
        
        # If all scores are null, we might use a small random fallback or fallback to the first variant
        if best_variant is None or score > best_score:
            best_variant = variant
            best_score = score if score != -999999.0 else 0.0
            
    if best_variant:
        logger.info(f"Selected best variant {best_variant.get('variant_id')} with score: {best_score}")
    return best_variant

def update_env_file(file_path: Path, params: dict[str, float]) -> bool:
    """Updates or appends the environment variables in a .env file."""
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False
        
    lines = file_path.read_text(encoding="utf-8").splitlines()
    updated_keys = set()
    new_lines = []
    
    # Map raw parameters to env variables
    env_params = {}
    for raw_name, value in params.items():
        env_name = PARAM_MAPPING.get(raw_name)
        if env_name:
            env_params[env_name] = str(value)
            
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
            
        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in env_params:
            new_lines.append(f'{key}="{env_params[key]}"')
            updated_keys.add(key)
            logger.info(f"Updated {key} -> {env_params[key]} in {file_path.name}")
        else:
            new_lines.append(line)
            
    # Append any keys that weren't found
    for key, val in env_params.items():
        if key not in updated_keys:
            new_lines.append(f'{key}="{val}"')
            logger.info(f"Appended {key} -> {val} in {file_path.name}")
            
    try:
        file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        logger.error(f"Failed to write to {file_path.name}: {exc}")
        return False

def main():
    campaign = load_latest_campaign()
    if not campaign:
        logger.error("No AlphaEvolve campaign loaded. Aborting feedback loop.")
        return
        
    best_variant = find_best_variant(campaign)
    if not best_variant:
        logger.error("No best variant resolved. Aborting feedback loop.")
        return
        
    params = best_variant.get("params", {})
    if not params:
        logger.error("Best variant has no parameters. Aborting.")
        return
        
    logger.info(f"Applying parameters from variant {best_variant.get('variant_id')}:")
    for k, v in params.items():
        logger.info(f"  {k} = {v}")
        
    # Update files
    update_env_file(ENV_BANKER_PATH, params)
    update_env_file(ENV_PATH, params)
    logger.info("AlphaEvolve feedback bridge applied successfully!")

if __name__ == "__main__":
    main()
