import json
import asyncio
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from shared.math_ops import symlog
from shared.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class ArenaSandbox:
    """
    Simulateur de marché offline pour THE HIVE.
    Permet à E.V.A. de "rêver" sur des données historiques.
    """
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.history: pd.DataFrame = pd.DataFrame()
        self.current_index = 0
        self.is_running = False

    def load_data(self, file_path: str):
        """Charge des données historiques (CSV avec colonnes: time, open, high, low, close, tick_volume)"""
        logger.info(f"Loading historical data from {file_path}")
        self.history = pd.read_csv(file_path)
        self.current_index = 0
        logger.info(f"Loaded {len(self.history)} candles.")

    async def run_scenario(self, scenario_name: str, speed: float = 10.0):
        """Charge et exécute un scénario de stress-test spécifique."""
        logger.info(f"🚨 APOCALYPSE MODE: Running scenario {scenario_name}")
        # En production, charge depuis apocalypse_scenarios.json
        # Simulation d'un gap de prix violent
        redis = get_redis_client()
        base_price = 2040.0
        
        if scenario_name == "BLACK_MONDAY_1987":
            crash_factor = 0.78 # -22%
            for i in range(10):
                base_price *= (1 - (0.022)) # Chute progressive par step
                await redis.publish("eva.market.tick", {
                    "symbol": "XAUUSD_SIM",
                    "close": base_price,
                    "type": "CATASTROPHIC_EVENT"
                })
                await asyncio.sleep(0.1)
        
        logger.info(f"Scenario {scenario_name} completed.")

    async def start_simulation(self, speed: float = 1.0):
        """Lance la diffusion des prix sur Redis comme si c'était le live MT5"""
        if self.history.empty:
            logger.error("No data loaded. Simulation aborted.")
            return

        self.is_running = True
        redis = get_redis_client()
        
        logger.info("Simulation started.")
        while self.is_running and self.current_index < len(self.history):
            row = self.history.iloc[self.current_index]
            
            # Simulation d'un événement de prix
            price_data = {
                "symbol": "XAUUSD_SIM",
                "time": str(row['time']),
                "bid": float(row['close']),
                "ask": float(row['close']) + 0.20, # Spread fixe simulé
                "last": float(row['close']),
                "volume": int(row.get('tick_volume', 0))
            }
            
            # Publication pour le Banker et le Lab
            await redis.publish("eva.market.tick", price_data)
            
            # Avancement
            self.current_index += 1
            await asyncio.sleep(1.0 / speed)
            
        logger.info("Simulation finished.")
        self.is_running = False

    def stop(self):
        self.is_running = False
