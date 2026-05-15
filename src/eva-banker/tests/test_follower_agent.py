"""Tests de l'agent follower distribue."""

import asyncio
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from eva_banker.follower.agent import FollowerAgent
from eva_banker.follower.config import (
    FollowerAccountConfig,
    FollowerAgentConfig,
    FollowerFleetConfig,
    load_follower_fleet_config,
    load_follower_config,
    save_follower_fleet_config,
    save_follower_config,
)
from eva_banker.follower.fleet import FollowerFleetManager
from eva_banker.follower.models import FollowerCommand, FollowerCommandType, FollowerExecutionResult
from eva_banker.follower.relay_server import RelayStorage, create_app
from eva_banker.follower.models import FollowerRuntimeStatus
from shared.models import Position, TradeAction, TradeOrder


class DummyRelayClient:
    """Double relay en memoire pour les tests."""

    def __init__(self, commands: list[FollowerCommand] | None = None) -> None:
        self.commands = commands or []
        self.acks: list[object] = []
        self.heartbeats: list[dict] = []

    async def fetch_commands(self, after: str | None = None) -> list[FollowerCommand]:
        """Retourne les commandes configurees."""

        return list(self.commands)

    async def acknowledge(self, result: object) -> None:
        """Memorise un acquittement."""

        self.acks.append(result)

    async def heartbeat(self, payload: dict) -> None:
        """Memorise un heartbeat."""

        self.heartbeats.append(payload)

    async def close(self) -> None:
        """Aucune ressource externe."""


class DummyMT5Service:
    """Double MT5 local pour verifier les actions follower."""

    def __init__(self) -> None:
        self.is_connected = True
        self.orders: list[TradeOrder] = []
        self.closed: list[tuple[int, Decimal | None]] = []
        self.modified: list[tuple[int, float, float]] = []
        self.positions: list[Position] = []

    async def connect(self) -> bool:
        """Simule une connexion MT5 active."""

        self.is_connected = True
        return True

    async def disconnect(self) -> None:
        """Simule une deconnexion."""

        self.is_connected = False

    async def execute_order(self, order: TradeOrder) -> dict:
        """Memorise un ordre et retourne un ticket."""

        self.orders.append(order)
        return {"success": True, "ticket": 4242, "message": "Ordre execute."}

    async def close_position(self, ticket: int, volume: Decimal | None = None) -> dict:
        """Memorise une cloture."""

        self.closed.append((ticket, volume))
        return {
            "success": True,
            "ticket": ticket,
            "message": "Cloture executee.",
            "volume_remaining": Decimal("0.06") if volume else Decimal("0"),
        }

    async def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> dict:
        """Memorise une modification."""

        self.modified.append((ticket, sl, tp))
        return {"success": True, "ticket": ticket, "message": "Position modifiee."}

    async def get_open_positions(self) -> list[Position]:
        """Retourne les positions configurees."""

        return list(self.positions)


class DummyFleetAgent:
    """Double d'agent pour verifier l'orchestration de flotte."""

    def __init__(self, config: FollowerAccountConfig, callback=None) -> None:
        self.config = config
        self.callback = callback
        self.running = False
        self.closed = False

    async def run_forever(self) -> None:
        """Simule une boucle agent courte et arretable."""

        self.running = True
        if self.callback:
            self.callback("agent demarre")
        while self.running:
            await asyncio.sleep(0.01)

    async def close(self) -> None:
        """Simule la fermeture des ressources."""

        self.running = False
        self.closed = True

    def get_status(self) -> FollowerRuntimeStatus:
        """Retourne un statut synthetique."""

        return FollowerRuntimeStatus(
            client_id=self.config.client_id,
            account_label=self.config.account_label,
            running=self.running,
            mt5_connected=self.running,
            relay_connected=self.running,
            dry_run=self.config.dry_run,
        )


class FollowerAgentTests(unittest.TestCase):
    """Verifie les regles critiques de l'agent follower."""

    def _config(self, temp_dir: Path) -> FollowerAgentConfig:
        """Construit une configuration isolee pour test."""

        return FollowerAgentConfig(
            dry_run=False,
            mock_mt5=True,
            state_path=str(temp_dir / "state.json"),
            log_path=str(temp_dir / "agent.log"),
            symbol_map={"GER40.cash": "DE40.e"},
            supported_symbols=["DE40.e", "XAUUSD.m"],
            allocation_ratio=2.0,
        )

    def test_config_roundtrip_masque_les_secrets(self) -> None:
        """Verifie la sauvegarde et le masquage des secrets."""

        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "config.json"
            config = FollowerAgentConfig(api_token="token", mt5_password="secret")
            save_follower_config(config, path)
            loaded = load_follower_config(path)

            self.assertEqual(loaded.api_token, "token")
            self.assertEqual(loaded.to_safe_dict()["api_token"], "***")
            self.assertEqual(loaded.to_safe_dict()["mt5_password"], "***")

    def test_fleet_config_roundtrip_multi_comptes(self) -> None:
        """Verifie la sauvegarde d'une flotte multi-comptes."""

        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "fleet.json"
            config = FollowerFleetConfig(
                fleet_id="client-a",
                accounts=[
                    FollowerAccountConfig(
                        client_id="client-a-1",
                        account_label="Compte 1",
                        mt5_password="secret-1",
                    ),
                    FollowerAccountConfig(
                        client_id="client-a-2",
                        account_label="Compte 2",
                        enabled=False,
                        mt5_password="secret-2",
                    ),
                ],
            )
            save_follower_fleet_config(config, path)
            loaded = load_follower_fleet_config(path)

            self.assertEqual(loaded.fleet_id, "client-a")
            self.assertEqual(len(loaded.accounts), 2)
            self.assertFalse(loaded.accounts[1].enabled)
            self.assertEqual(loaded.to_safe_dict()["accounts"][0]["mt5_password"], "***")

    def test_fleet_manager_lance_uniquement_les_comptes_actifs(self) -> None:
        """Verifie le demarrage et l'arret d'une flotte locale."""

        async def scenario() -> tuple[FollowerFleetManager, list[str]]:
            events: list[str] = []
            config = FollowerFleetConfig(
                fleet_id="client-a",
                accounts=[
                    FollowerAccountConfig(client_id="a-1", account_label="Actif", enabled=True),
                    FollowerAccountConfig(client_id="a-2", account_label="Inactif", enabled=False),
                ],
            )
            manager = FollowerFleetManager(
                config,
                agent_factory=lambda account, callback: DummyFleetAgent(account, callback),
                event_callback=events.append,
            )
            await manager.start_all()
            await asyncio.sleep(0.03)
            statuses = manager.get_statuses()
            await manager.stop_all()
            self.assertEqual(len(statuses), 2)
            self.assertTrue(statuses[0].running)
            self.assertFalse(statuses[1].running)
            return manager, events

        manager, events = asyncio.run(scenario())

        self.assertEqual(manager.agents, {})
        self.assertTrue(any("Compte demarre" in event for event in events))

    def test_open_mappe_symbole_et_reste_idempotent(self) -> None:
        """Verifie mapping, scaling et idempotence d'ouverture."""

        with tempfile.TemporaryDirectory() as raw_dir:
            mt5 = DummyMT5Service()
            config = self._config(Path(raw_dir))
            command = FollowerCommand(
                command_id="cmd-open-1",
                command_type=FollowerCommandType.OPEN,
                master_ticket=1001,
                symbol="GER40.cash",
                action=TradeAction.BUY,
                volume=Decimal("0.10"),
                stop_loss=Decimal("24000"),
            )
            agent = FollowerAgent(config, mt5_service=mt5, relay_client=DummyRelayClient())

            first = asyncio.run(agent.handle_command(command))
            second = asyncio.run(agent.handle_command(command))

            self.assertTrue(first.success)
            self.assertEqual(second.mode, "idempotent_skip")
            self.assertEqual(len(mt5.orders), 1)
            self.assertEqual(mt5.orders[0].symbol, "DE40.e")
            self.assertEqual(mt5.orders[0].volume, Decimal("0.20"))

    def test_open_applique_le_sizing_dynamique_par_capital(self) -> None:
        """Verifie le sizing dynamique selon capital maitre et follower."""

        with tempfile.TemporaryDirectory() as raw_dir:
            mt5 = DummyMT5Service()
            config = self._config(Path(raw_dir))
            config.allocation_ratio = 1.0
            config.balance_reference = 100000.0
            config.master_balance_reference = 10000.0
            command = FollowerCommand(
                command_id="cmd-open-dynamic-1",
                command_type=FollowerCommandType.OPEN,
                master_ticket=1002,
                symbol="GER40.cash",
                action=TradeAction.BUY,
                volume=Decimal("0.01"),
            )
            agent = FollowerAgent(config, mt5_service=mt5, relay_client=DummyRelayClient())

            result = asyncio.run(agent.handle_command(command))

            self.assertTrue(result.success)
            self.assertEqual(mt5.orders[0].volume, Decimal("0.10"))

    def test_open_utilise_l_equity_maitre_du_payload_si_disponible(self) -> None:
        """Verifie que le payload relay peut remplacer la reference maitre."""

        with tempfile.TemporaryDirectory() as raw_dir:
            mt5 = DummyMT5Service()
            config = self._config(Path(raw_dir))
            config.allocation_ratio = 1.0
            config.balance_reference = 50000.0
            config.master_balance_reference = 10000.0
            command = FollowerCommand(
                command_id="cmd-open-dynamic-2",
                command_type=FollowerCommandType.OPEN,
                master_ticket=1003,
                symbol="GER40.cash",
                action=TradeAction.BUY,
                volume=Decimal("0.04"),
                payload={"master_equity": "20000"},
            )
            agent = FollowerAgent(config, mt5_service=mt5, relay_client=DummyRelayClient())

            result = asyncio.run(agent.handle_command(command))

            self.assertTrue(result.success)
            self.assertEqual(mt5.orders[0].volume, Decimal("0.10"))

    def test_positive_close_fait_runner_70_slbe(self) -> None:
        """Verifie la cloture 70%% puis SL au break-even."""

        with tempfile.TemporaryDirectory() as raw_dir:
            mt5 = DummyMT5Service()
            mt5.positions = [
                Position(
                    ticket=4242,
                    symbol="DE40.e",
                    action=TradeAction.BUY,
                    volume=Decimal("0.20"),
                    open_price=Decimal("24900"),
                    current_price=Decimal("25000"),
                    profit=Decimal("100"),
                    open_time=datetime(2026, 5, 9, 10, 0, 0),
                )
            ]
            config = self._config(Path(raw_dir))
            agent = FollowerAgent(config, mt5_service=mt5, relay_client=DummyRelayClient())
            agent._ticket_links["1001"] = 4242
            command = FollowerCommand(
                command_id="cmd-close-1",
                command_type=FollowerCommandType.CLOSE,
                master_ticket=1001,
                master_profit=Decimal("10"),
                close_reason="manual_profit",
            )

            result = asyncio.run(agent.handle_command(command))

            self.assertTrue(result.success)
            self.assertEqual(result.mode, "runner_70_slbe")
            self.assertEqual(mt5.closed[0], (4242, Decimal("0.14")))
            self.assertEqual(mt5.modified[0], (4242, 24900.0, 0.0))
            self.assertEqual(agent._ticket_links["1001"], 4242)

    def test_negative_close_ferme_totalement(self) -> None:
        """Verifie la fermeture totale sur perte ou SL maitre."""

        with tempfile.TemporaryDirectory() as raw_dir:
            mt5 = DummyMT5Service()
            config = self._config(Path(raw_dir))
            agent = FollowerAgent(config, mt5_service=mt5, relay_client=DummyRelayClient())
            agent._ticket_links["1001"] = 4242
            command = FollowerCommand(
                command_id="cmd-close-2",
                command_type=FollowerCommandType.CLOSE,
                master_ticket=1001,
                master_profit=Decimal("-5"),
                close_reason="SL",
            )

            result = asyncio.run(agent.handle_command(command))

            self.assertTrue(result.success)
            self.assertEqual(result.mode, "full_close")
            self.assertEqual(mt5.closed[0], (4242, None))
            self.assertNotIn("1001", agent._ticket_links)

    def test_relay_server_distribue_ack_et_heartbeat(self) -> None:
        """Verifie le cycle publication, polling, ack et heartbeat."""

        with tempfile.TemporaryDirectory() as raw_dir:
            with patch.dict(
                "os.environ",
                {
                    "HIVE_FOLLOWER_RELAY_MASTER_TOKEN": "",
                    "HIVE_FOLLOWER_RELAY_CLIENT_TOKEN": "",
                    "HIVE_FOLLOWER_RELAY_CLIENT_TOKENS_JSON": "",
                },
            ):
                storage = RelayStorage(Path(raw_dir) / "relay.json")
                client = TestClient(create_app(storage))
                command = FollowerCommand(
                    command_id="relay-cmd-1",
                    command_type=FollowerCommandType.OPEN,
                    master_ticket=9001,
                    symbol="XAUUSD",
                    action=TradeAction.BUY,
                    volume=Decimal("0.10"),
                )
                result = FollowerExecutionResult(
                    command_id="relay-cmd-1",
                    success=True,
                    message="Commande executee.",
                    ticket=4242,
                )

                publish_response = client.post(
                    "/api/master/commands",
                    json={"client_ids": ["ftuk-100k-a"], "command": _model_to_dict(command)},
                )
                pending_response = client.get(
                    "/api/follower/commands",
                    params={"client_id": "ftuk-100k-a"},
                )
                ack_response = client.post(
                    "/api/follower/ack",
                    json=_model_to_dict(result),
                    headers={"X-Hive-Client-Id": "ftuk-100k-a"},
                )
                heartbeat_response = client.post(
                    "/api/follower/heartbeat",
                    json={"running": True},
                    headers={"X-Hive-Client-Id": "ftuk-100k-a"},
                )
                empty_response = client.get(
                    "/api/follower/commands",
                    params={"client_id": "ftuk-100k-a"},
                )

                self.assertEqual(publish_response.status_code, 200)
                self.assertEqual(publish_response.json()["published"]["ftuk-100k-a"], 1)
                self.assertEqual(pending_response.status_code, 200)
                self.assertEqual(len(pending_response.json()["commands"]), 1)
                self.assertEqual(ack_response.status_code, 200)
                self.assertEqual(heartbeat_response.status_code, 200)
                self.assertEqual(empty_response.json()["commands"], [])
                self.assertIn("ftuk-100k-a", client.get("/api/relay/status").json()["clients"])


def _model_to_dict(model: object) -> dict:
    """Convertit un modele Pydantic v1 ou v2 en payload JSON."""

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[no-any-return,union-attr]
    return model.dict()  # type: ignore[no-any-return,attr-defined]


if __name__ == "__main__":
    unittest.main()
