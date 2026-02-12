from pydantic import BaseModel
from typing import Optional
from enum import Enum

class TradeResult(BaseModel):
    ticket: int
    profit: float
    symbol: str
    volume: float
    duration_seconds: int
    max_drawdown_percent: float = 0.0
    efficiency_score: float = 0.0 # Placeholder for beta

class GameState(BaseModel):
    level: int = 0
    xp: float = 0.0
    sp: float = 0.0  # Sovereignty Points (Currency)
    total_profit: float = 0.0
    unlocked_techs: list[str] = []
    current_quest: str = "The Sight Awakening"
    quest_progress: float = 0.0 # 0.0 to 1.0 (4000 EUR target)

class GamificationConfig(BaseModel):
    alpha_profit: float = 10.0
    beta_efficiency: float = 5.0
    gamma_risk: float = 20.0
    quest_target_amount: float = 4000.0 # Debt + Hardware
