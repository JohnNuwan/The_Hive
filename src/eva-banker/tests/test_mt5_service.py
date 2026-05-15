"""Tests du service MT5 pour l'univers de marche et l'initialisation terminal."""

import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from eva_banker.services import mt5 as mt5_module
from eva_banker.services.mt5 import MT5Service
from shared import Position, TradeAction


def test_discover_symbols_mock_covers_all_asset_classes():
    """Verifie que le mode mock expose Forex, CFD et crypto."""
    service = MT5Service(mock_mode=True)
    symbols = asyncio.run(
        service.discover_symbols(
            include_forex=True,
            include_cfd=True,
            include_crypto=True,
            max_symbols=0,
        )
    )

    assert "EURUSD" in symbols
    assert "XAUUSD" in symbols
    assert "BTCUSD" in symbols


def test_discover_symbols_mock_respects_filters():
    """Verifie que les filtres d'univers limitent bien la decouverte."""
    service = MT5Service(mock_mode=True)
    symbols = asyncio.run(
        service.discover_symbols(
            include_forex=False,
            include_cfd=False,
            include_crypto=True,
            max_symbols=0,
        )
    )

    assert symbols
    assert all(service.classify_symbol(symbol) == "crypto" for symbol in symbols)


class _FakeAccountInfo:
    """Represente un compte MT5 minimal pour les tests."""

    def __init__(self, login: int, server: str) -> None:
        self.login = login
        self.server = server
        self.balance = 10000.0
        self.equity = 10000.0


class _FakeMT5Module:
    """Double simple de l'API MetaTrader5 pour tester l'initialisation."""

    def __init__(self, initialize_results: list[bool] | None = None) -> None:
        self.initialize_calls: list[dict] = []
        self.login_calls: list[dict] = []
        self._connected_login: int | None = None
        self._connected_server: str | None = None
        self._initialize_results = list(initialize_results or [True])
        self.shutdown_calls = 0
        self.login_result = True

    def initialize(self, **kwargs):
        """Memorise les parametres d'initialisation puis reussit."""
        self.initialize_calls.append(kwargs)
        if self._initialize_results:
            result = self._initialize_results.pop(0)
        else:
            result = True
        if result and self.login_result and kwargs.get("login") and kwargs.get("server"):
            self._connected_login = int(kwargs["login"])
            self._connected_server = str(kwargs["server"])
        return result

    def account_info(self):
        """Retourne le compte actif si un login a deja ete realise."""
        if self._connected_login is None or self._connected_server is None:
            return None
        return _FakeAccountInfo(self._connected_login, self._connected_server)

    def login(self, login: int, password: str, server: str):
        """Memorise le login cible puis reussit."""
        self.login_calls.append(
            {
                "login": login,
                "password": password,
                "server": server,
            }
        )
        if self.login_result:
            self._connected_login = login
            self._connected_server = server
            return True
        return False

    def last_error(self):
        """Retourne une erreur nulle pour l'API factice."""
        return (0, "OK")

    def shutdown(self):
        """N'effectue aucune action pour le double de test."""
        self.shutdown_calls += 1
        return True


def test_connect_uses_explicit_terminal_path_and_portable_mode():
    """Verifie que l'initialisation MT5 cible bien le terminal configure."""
    fake_mt5 = _FakeMT5Module()

    async def scenario() -> None:
        with TemporaryDirectory() as tmpdir:
            claim_dir = Path(tmpdir)
            with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
                mt5_module, "mt5", fake_mt5, create=True
            ):
                service = MT5Service(
                    mock_mode=False,
                    login=531240000,
                    password="secret",
                    server="FTMO-Server3",
                    terminal_path="C:/MT5/FTMO-Server3/terminal64.exe",
                    terminal_portable=True,
                    terminal_timeout_ms=45000,
                )
                with patch.object(service, "_get_account_claim_directory", return_value=claim_dir):
                    connected = await service.connect()
                    assert connected is True

    asyncio.run(scenario())

    assert fake_mt5.initialize_calls == [
        {
            "path": "C:/MT5/FTMO-Server3/terminal64.exe",
            "login": 531240000,
            "password": "secret",
            "server": "FTMO-Server3",
            "portable": True,
            "timeout": 45000,
        }
    ]
    assert fake_mt5.login_calls == []


def test_connect_uses_timeout_even_without_explicit_terminal_path():
    """Verifie que le timeout MT5 reste force sans chemin de terminal."""
    fake_mt5 = _FakeMT5Module()

    async def scenario() -> None:
        with TemporaryDirectory() as tmpdir:
            claim_dir = Path(tmpdir)
            with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
                mt5_module, "mt5", fake_mt5, create=True
            ):
                service = MT5Service(
                    mock_mode=False,
                    login=521044924,
                    password="secret",
                    server="FTMO-Server2",
                )
                with patch.object(service, "_get_account_claim_directory", return_value=claim_dir):
                    connected = await service.connect()
                    assert connected is True

    asyncio.run(scenario())

    assert fake_mt5.initialize_calls == [
        {
            "login": 521044924,
            "password": "secret",
            "server": "FTMO-Server2",
            "timeout": 60000,
        }
    ]


def test_connect_retry_with_alternate_portable_mode_after_ipc_timeout():
    """Verifie le repli automatique sur l'autre mode portable."""
    fake_mt5 = _FakeMT5Module(initialize_results=[False, True])

    async def scenario() -> None:
        with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
            mt5_module, "mt5", fake_mt5, create=True
        ):
            service = MT5Service(
                mock_mode=False,
                login=333382142,
                password="secret",
                server="FTUKMarkets-Trade",
                terminal_path="C:/MT5/FTUK/terminal64.exe",
                terminal_portable=False,
                terminal_timeout_ms=120000,
            )
            connected = await service.connect()
            assert connected is True

    asyncio.run(scenario())

    assert fake_mt5.initialize_calls == [
        {
            "path": "C:/MT5/FTUK/terminal64.exe",
            "login": 333382142,
            "password": "secret",
            "server": "FTUKMarkets-Trade",
            "timeout": 120000,
        },
        {
            "path": "C:/MT5/FTUK/terminal64.exe",
            "login": 333382142,
            "password": "secret",
            "server": "FTUKMarkets-Trade",
            "portable": True,
            "timeout": 120000,
        },
    ]
    assert fake_mt5.shutdown_calls >= 2


def test_connect_does_not_retry_portable_mode_when_alternance_is_disabled():
    """Verifie qu'un terminal FTUK peut interdire le repli portable."""
    fake_mt5 = _FakeMT5Module(initialize_results=[False])
    fake_settings = SimpleNamespace(
        mt5_duplicate_order_cooldown_seconds=20,
        mt5_reconnect_cooldown_seconds=15,
        mt5_warning_cooldown_seconds=30,
        mt5_try_alternate_portable_mode=False,
        mt5_initialize_retries=1,
    )

    async def scenario() -> None:
        with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
            mt5_module, "mt5", fake_mt5, create=True
        ), patch.object(mt5_module, "get_settings", return_value=fake_settings):
            service = MT5Service(
                mock_mode=False,
                login=333382142,
                password="secret",
                server="FTUKMarkets-Trade",
                terminal_path="C:/MT5/FTUK/terminal64.exe",
                terminal_portable=False,
                terminal_timeout_ms=120000,
            )
            connected = await service.connect()
            assert connected is False

    asyncio.run(scenario())

    assert fake_mt5.initialize_calls == [
        {
            "path": "C:/MT5/FTUK/terminal64.exe",
            "login": 333382142,
            "password": "secret",
            "server": "FTUKMarkets-Trade",
            "timeout": 120000,
        }
    ]


def test_close_position_mock_supports_partial_close():
    """Verifie qu'une cloture partielle mock conserve un reliquat au meme ticket."""
    service = MT5Service(mock_mode=True)
    service._mock_positions = [
        Position(
            ticket=123,
            symbol="XAUUSD",
            action=TradeAction.BUY,
            volume=Decimal("0.20"),
            open_price=Decimal("4500.5"),
            current_price=Decimal("4510.0"),
            stop_loss=Decimal("4480.0"),
            take_profit=None,
            profit=Decimal("25.0"),
            magic_number=12345,
            open_time=datetime.now(),
        )
    ]

    result = asyncio.run(service.close_position(123, volume=Decimal("0.10")))

    assert result["success"] is True
    assert result["partial_close"] is True
    assert result["volume_closed"] == 0.10
    assert result["volume_remaining"] == 0.10
    assert len(service._mock_positions) == 1
    assert service._mock_positions[0].volume == Decimal("0.10")


def test_connect_refuses_duplicate_account_claim_for_another_pid():
    """Verifie qu'un second Banker ne peut pas revendiquer le meme compte."""
    fake_mt5_a = _FakeMT5Module()
    fake_mt5_b = _FakeMT5Module()

    with TemporaryDirectory() as temp_dir:
        claim_dir = Path(temp_dir)

        async def connect_first() -> bool:
            with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
                mt5_module, "mt5", fake_mt5_a, create=True
            ):
                service = MT5Service(
                    mock_mode=False,
                    login=531240000,
                    password="secret",
                    server="FTMO-Server3",
                    terminal_path="C:/MT5/FTMO-Server3/terminal64.exe",
                )
                with patch.object(service, "_get_account_claim_directory", return_value=claim_dir), patch.object(
                    service, "_get_runtime_process_id", return_value=11111
                ):
                    return await service.connect()

        async def connect_second() -> bool:
            with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
                mt5_module, "mt5", fake_mt5_b, create=True
            ):
                service = MT5Service(
                    mock_mode=False,
                    login=531240000,
                    password="secret",
                    server="FTMO-Server3",
                    terminal_path="C:/MT5/FTMO-Server3/terminal64.exe",
                )
                with patch.object(service, "_get_account_claim_directory", return_value=claim_dir), patch.object(
                    service, "_get_runtime_process_id", return_value=22222
                ), patch.object(service, "_is_process_alive", return_value=True):
                    return await service.connect()

        assert asyncio.run(connect_first()) is True
        assert asyncio.run(connect_second()) is False


def test_disconnect_releases_account_claim():
    """Verifie que la deconnexion libere le verrou local du compte."""
    fake_mt5 = _FakeMT5Module()

    with TemporaryDirectory() as temp_dir:
        claim_dir = Path(temp_dir)

        async def scenario() -> None:
            with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
                mt5_module, "mt5", fake_mt5, create=True
            ):
                service = MT5Service(
                    mock_mode=False,
                    login=333382206,
                    password="secret",
                    server="FTUKMarkets-Trade",
                    terminal_path="C:/MT5/FTUK/terminal64.exe",
                )
                with patch.object(service, "_get_account_claim_directory", return_value=claim_dir), patch.object(
                    service, "_get_runtime_process_id", return_value=33333
                ):
                    assert await service.connect() is True
                    expected_claim = claim_dir / "FTUKMarkets-Trade__333382206.json"
                    assert expected_claim.exists()
                    await service.disconnect()
                    assert not expected_claim.exists()

        asyncio.run(scenario())


def test_get_max_spread_points_uses_symbol_specific_thresholds():
    """Vérifie que les indices critiques utilisent leurs seuils dédiés."""
    service = MT5Service(mock_mode=True)

    assert service.get_max_spread_points("US30.cash") == 250
    assert service.get_max_spread_points("GER40.cash") == 300
    assert service.get_max_spread_points("US500.cash") == 70
    assert service.get_max_spread_points("US100.cash") == 120


def test_get_max_spread_points_keeps_family_defaults():
    """Vérifie les seuils de repli pour forex, métal et crypto."""
    service = MT5Service(mock_mode=True)

    assert service.get_max_spread_points("EURUSD") == 25
    assert service.get_max_spread_points("XAUUSD") == 60
    assert service.get_max_spread_points("BTCUSD") == 1500
    assert service.get_max_spread_points("ETHUSD.e") == 1500


def test_get_symbol_risk_sizing_hint_reads_mt5_metadata():
    """Verifie que les economics MT5 du symbole remontent pour le sizing."""
    fake_mt5 = SimpleNamespace(
        symbol_info=lambda _symbol: SimpleNamespace(
            point=0.1,
            trade_tick_size=0.5,
            trade_tick_value_profit=1.25,
            trade_tick_value_loss=1.50,
            trade_tick_value=1.40,
            trade_contract_size=10.0,
            volume_min=0.10,
            volume_step=0.10,
            volume_max=50.0,
        )
    )

    async def scenario() -> None:
        with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
            mt5_module, "mt5", fake_mt5, create=True
        ):
            service = MT5Service(mock_mode=False)
            service.is_connected = True
            hint = await service.get_symbol_risk_sizing_hint("DE40.e")

        assert hint["tick_size"] == Decimal("0.5")
        assert hint["tick_value"] == Decimal("1.5")
        assert hint["contract_size"] == Decimal("10.0")
        assert hint["volume_min"] == Decimal("0.1")
        assert hint["volume_step"] == Decimal("0.1")
        assert hint["volume_max"] == Decimal("50.0")

    asyncio.run(scenario())


def test_connect_refuses_wrong_existing_account_when_target_login_fails():
    """Verifie qu'un compte deja ouvert mais incorrect n'est plus accepte."""
    fake_mt5 = _FakeMT5Module()
    fake_mt5._connected_login = 333382142
    fake_mt5._connected_server = "FTUKMarkets-Live"
    fake_mt5.login_result = False

    async def scenario() -> None:
        with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
            mt5_module, "mt5", fake_mt5, create=True
        ):
            service = MT5Service(
                mock_mode=False,
                login=333382300,
                password="secret",
                server="FTUKMarkets-Trade",
                terminal_path="C:/MT5/FTUK/terminal64.exe",
                terminal_portable=False,
                terminal_timeout_ms=120000,
            )
            connected = await service.connect()
            assert connected is False

    asyncio.run(scenario())

    assert fake_mt5.login_calls[0]["login"] == 333382300
