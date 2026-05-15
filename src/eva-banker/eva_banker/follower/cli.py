"""CLI de l'agent follower distribue."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from eva_banker.follower.agent import FollowerAgent
from eva_banker.follower.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FLEET_CONFIG_PATH,
    FollowerAgentConfig,
    FollowerFleetConfig,
    load_follower_fleet_config,
    load_follower_config,
    save_follower_fleet_config,
    save_follower_config,
)
from eva_banker.follower.fleet import FollowerFleetManager


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI follower.

    Returns:
        argparse.ArgumentParser: Parseur pret a l'emploi.
    """

    parser = argparse.ArgumentParser(description="Lance l'agent follower THE HIVE.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Chemin du fichier JSON de configuration.")
    parser.add_argument("--init-config", action="store_true", help="Cree une configuration exemple si absente.")
    parser.add_argument("--fleet", action="store_true", help="Lance le mode multi-comptes.")
    parser.add_argument("--fleet-config", default=str(DEFAULT_FLEET_CONFIG_PATH), help="Chemin JSON de la flotte.")
    parser.add_argument("--init-fleet", action="store_true", help="Cree une configuration multi-comptes exemple.")
    parser.add_argument("--ui", action="store_true", help="Lance l'interface CustomTkinter.")
    return parser


async def _run_agent(config_path: str) -> None:
    """Lance l'agent follower en mode console.

    Args:
        config_path (str): Fichier de configuration.
    """

    config = load_follower_config(config_path)
    agent = FollowerAgent(config)
    try:
        await agent.run_forever()
    finally:
        await agent.close()


async def _run_fleet(config_path: str) -> None:
    """Lance la flotte follower en mode console.

    Args:
        config_path (str): Fichier de configuration multi-comptes.
    """

    config = load_follower_fleet_config(config_path)
    manager = FollowerFleetManager(config)
    await manager.run_forever()


def main() -> None:
    """Point d'entree console."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    fleet_config_path = Path(args.fleet_config)
    if args.init_fleet and not fleet_config_path.exists():
        save_follower_fleet_config(FollowerFleetConfig(), fleet_config_path)
        print(f"Configuration flotte follower creee: {fleet_config_path}")
        return
    if args.init_config and not config_path.exists():
        save_follower_config(FollowerAgentConfig(), config_path)
        print(f"Configuration follower creee: {config_path}")
        return
    if args.ui:
        from eva_banker.follower.ui import run_fleet_ui, run_ui

        if args.fleet:
            run_fleet_ui(fleet_config_path)
        else:
            run_ui(config_path)
        return
    if args.fleet:
        asyncio.run(_run_fleet(str(fleet_config_path)))
        return
    asyncio.run(_run_agent(str(config_path)))


if __name__ == "__main__":
    main()
