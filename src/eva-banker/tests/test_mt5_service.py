"""Tests du service MT5 pour l'univers de marche et l'initialisation terminal."""

import asyncio
from unittest.mock import patch

from eva_banker.services import mt5 as mt5_module
from eva_banker.services.mt5 import MT5Service


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

    def __init__(self) -> None:
        self.initialize_calls: list[dict] = []
        self.login_calls: list[dict] = []
        self._connected_login: int | None = None
        self._connected_server: str | None = None

    def initialize(self, **kwargs):
        """Memorise les parametres d'initialisation puis reussit."""
        self.initialize_calls.append(kwargs)
        return True

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
        self._connected_login = login
        self._connected_server = server
        return True

    def last_error(self):
        """Retourne une erreur nulle pour l'API factice."""
        return (0, "OK")

    def shutdown(self):
        """N'effectue aucune action pour le double de test."""
        return True


def test_connect_uses_explicit_terminal_path_and_portable_mode():
    """Verifie que l'initialisation MT5 cible bien le terminal configure."""
    fake_mt5 = _FakeMT5Module()

    async def scenario() -> None:
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
            connected = await service.connect()
            assert connected is True

    asyncio.run(scenario())

    assert fake_mt5.initialize_calls == [
        {
            "path": "C:/MT5/FTMO-Server3/terminal64.exe",
            "portable": True,
            "timeout": 45000,
        }
    ]
    assert fake_mt5.login_calls[0]["login"] == 531240000
    assert fake_mt5.login_calls[0]["server"] == "FTMO-Server3"


def test_connect_uses_timeout_even_without_explicit_terminal_path():
    """Verifie que le timeout MT5 reste force sans chemin de terminal."""
    fake_mt5 = _FakeMT5Module()

    async def scenario() -> None:
        with patch.object(mt5_module, "MT5_AVAILABLE", True), patch.object(
            mt5_module, "mt5", fake_mt5, create=True
        ):
            service = MT5Service(
                mock_mode=False,
                login=521044924,
                password="secret",
                server="FTMO-Server2",
            )
            connected = await service.connect()
            assert connected is True

    asyncio.run(scenario())

    assert fake_mt5.initialize_calls == [{"timeout": 60000}]
