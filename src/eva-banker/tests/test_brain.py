"""Tests cibles du cerveau Banker pour le mode `cpu_live`."""

import asyncio
from types import MethodType, SimpleNamespace

from eva_banker.brain import AutoTradingEngine


class _FakeMT5:
    """Double minimal de MT5 pour tester la restriction d'univers."""

    async def discover_symbols(
        self,
        *,
        include_forex: bool,
        include_cfd: bool,
        include_crypto: bool,
        max_symbols: int,
    ) -> list[str]:
        """Retourne un univers mixte fixe pour le scenario de test."""
        return ["US500.cash", "XAUUSD", "US100.cash"]

    def classify_symbol(self, symbol: str) -> str:
        """Retourne une famille stable selon le symbole teste."""
        if symbol == "XAUUSD":
            return "metals"
        return "indices"

    async def ensure_symbol_selected(self, symbol: str) -> bool:
        """Retourne toujours True pour le mock."""
        return True


def _build_engine_stub() -> AutoTradingEngine:
    """Construit un moteur minimal sans dependances runtime lourdes."""
    engine = AutoTradingEngine.__new__(AutoTradingEngine)
    engine.symbols = ["EURUSD", "XAUUSD", "GBPUSD"]
    engine._dynamic_universe_enabled = True
    engine._last_universe_refresh = None
    engine._scan_forex = True
    engine._scan_cfd = True
    engine._scan_crypto = True
    engine._universe_refresh_minutes = 240
    engine._universe_max_symbols = 0
    engine._lab_universe_enabled = True
    engine._lab_universe_source = "arena_symbol_metrics"
    engine._lab_universe_symbols = ["US500.cash", "XAUUSD", "US100.cash"]
    engine._lab_universe_top_symbols = ["US500.cash", "XAUUSD", "US100.cash"]
    engine._lab_universe_gate_allowed = True
    engine._cpu_live_mode = True
    engine._cpu_live_symbols = ["EURUSD", "XAUUSD", "GBPUSD"]
    engine._symbol_cursor = 0
    engine.mt5 = _FakeMT5()
    engine.risk = SimpleNamespace(register_symbol_universe=lambda universe: universe)

    async def _fake_refresh_lab_live_universe(self, force: bool = False) -> list[str]:
        """Retourne l'univers live deja expose par EVA Lab."""
        return list(self._lab_universe_symbols)

    engine._refresh_lab_live_universe = MethodType(_fake_refresh_lab_live_universe, engine)
    return engine


def test_resolve_cpu_live_runtime_symbols_includes_live_top_symbols():
    """Verifie que `cpu_live` ajoute les top symbols du champion a l'allowlist."""
    engine = _build_engine_stub()

    effective = engine._resolve_cpu_live_runtime_symbols()

    assert effective == ["EURUSD", "XAUUSD", "GBPUSD", "US500.CASH", "US100.CASH"]


def test_refresh_symbol_universe_keeps_live_top_symbols_in_cpu_live_mode():
    """Verifie que le scan live ne se replie plus sur `XAUUSD` seul."""
    engine = _build_engine_stub()

    symbols = asyncio.run(engine.refresh_symbol_universe(force=True))

    assert symbols == ["EURUSD", "XAUUSD", "GBPUSD", "US500.cash", "US100.cash"]
    assert engine.symbols == ["EURUSD", "XAUUSD", "GBPUSD", "US500.cash", "US100.cash"]


def test_resolve_live_position_state_for_buy_position():
    """Construit un etat live coherent pour une position acheteuse."""

    position = SimpleNamespace(
        action="BUY",
        open_price=100.0,
        current_price=101.5,
        stop_loss=100.2,
    )

    state = AutoTradingEngine._resolve_live_position_state(position)

    assert state["position_state"] == 1.0
    assert state["unrealized_return"] > 0.0
    assert state["slbe_state"] == 1.0


def test_resolve_live_position_state_for_sell_position():
    """Construit un etat live coherent pour une position vendeuse."""

    position = SimpleNamespace(
        action="SELL",
        open_price=100.0,
        current_price=98.5,
        stop_loss=99.8,
    )

    state = AutoTradingEngine._resolve_live_position_state(position)

    assert state["position_state"] == -1.0
    assert state["unrealized_return"] > 0.0
    assert state["slbe_state"] == 1.0
