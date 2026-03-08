"""Tests de decouverte d'univers pour le service MT5."""

import asyncio

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
