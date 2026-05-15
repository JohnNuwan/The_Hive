"""Agent follower client pour le copy trading distribue."""

from eva_banker.follower.agent import FollowerAgent
from eva_banker.follower.config import (
    FollowerAccountConfig,
    FollowerAgentConfig,
    FollowerFleetConfig,
    load_follower_fleet_config,
    load_follower_config,
)
from eva_banker.follower.fleet import FollowerFleetManager
from eva_banker.follower.models import FollowerCommand, FollowerCommandType

__all__ = [
    "FollowerAccountConfig",
    "FollowerAgent",
    "FollowerAgentConfig",
    "FollowerCommand",
    "FollowerCommandType",
    "FollowerFleetConfig",
    "FollowerFleetManager",
    "load_follower_fleet_config",
    "load_follower_config",
]
