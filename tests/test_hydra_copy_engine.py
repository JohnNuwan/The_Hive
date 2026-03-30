"""
Tests cibles du moteur Hydra de copie master/slaves.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

LOCAL_ROOT = Path(__file__).resolve().parents[1]
for extra in ("src/shared", "src/eva-banker"):
    extra_path = LOCAL_ROOT / extra
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from eva_banker.services.hydra import HydraCopyEngine
from eva_banker.services.multi_account import MultiAccountManager
from shared import (
    CopyTradeRequest,
    HydraAccountRole,
    HydraEventType,
    HydraScalingMode,
    OrderSource,
    PropFirmAccount,
    TradeAction,
    TradeReplicationEvent,
)


class FakeRedisClient:
    """
    Faux client Redis minimal pour les tests Hydra.
    """

    def __init__(self) -> None:
        self.storage: dict[str, dict] = {}

    async def cache_set(self, key: str, value: dict, ttl_seconds: int | None = None) -> bool:
        self.storage[key] = value
        return True

    async def cache_get(self, key: str) -> dict | None:
        return self.storage.get(key)


def _make_account(
    *,
    scaling_mode: HydraScalingMode = HydraScalingMode.FIXED,
    scaling_factor: str = "1.0",
    current_balance: str = "10000",
    executor_url: str | None = "http://hydra-slave.local",
    quarantined: bool = False,
) -> PropFirmAccount:
    """
    Construit un compte esclave Hydra de test.

    Args:
        scaling_mode (HydraScalingMode): Mode de scaling du compte.
        scaling_factor (str): Facteur de copie.
        current_balance (str): Solde courant du compte.
        executor_url (str | None): URL d'executeur simulee.
        quarantined (bool): Place le compte en quarantaine si vrai.

    Returns:
        PropFirmAccount: Compte Hydra pret a l'emploi.
    """
    return PropFirmAccount(
        id=uuid4(),
        name="Slave-01",
        login=123456,
        server="FTMO-Server",
        broker="FTMO",
        role=HydraAccountRole.SLAVE,
        phase="funded",
        initial_balance=Decimal("10000"),
        current_balance=Decimal(current_balance),
        profit_target_percent=Decimal("10"),
        max_daily_loss_percent=Decimal("4"),
        max_total_loss_percent=Decimal("8"),
        scaling_mode=scaling_mode,
        scaling_factor=Decimal(scaling_factor),
        lot_min=Decimal("0.01"),
        lot_max=Decimal("5.0"),
        lot_step=Decimal("0.01"),
        symbol_map={},
        allowed_symbols=["EURUSD", "XAUUSD", "GBPUSD", "US30.cash"],
        risk_enabled=True,
        max_daily_drawdown_pct=Decimal("2.0"),
        copy_enabled=True,
        master_source_id="local-master",
        executor_url=executor_url,
        quarantined_until=datetime.now() + timedelta(hours=4) if quarantined else None,
    )


def _make_event(*, volume: str = "0.05", event_id=None, master_balance: str = "10000") -> TradeReplicationEvent:
    """
    Construit un evenement de fill maitre de test.

    Args:
        volume (str): Volume execute sur le master.
        event_id (Any): Identifiant explicite facultatif.
        master_balance (str): Balance du master pour le mode proportionnel.

    Returns:
        TradeReplicationEvent: Evenement Hydra normalise.
    """
    return TradeReplicationEvent(
        event_id=event_id or uuid4(),
        event_type=HydraEventType.FILL,
        source_account_id="local-master",
        source_login=777001,
        ticket=987654,
        symbol="EURUSD",
        action=TradeAction.SELL,
        volume=Decimal(volume),
        entry_price=Decimal("1.1000"),
        source=OrderSource.STRATEGY,
        master_balance=Decimal(master_balance),
        master_equity=Decimal(master_balance),
    )


def test_hydra_fixed_scaling_dry_run(monkeypatch) -> None:
    """
    Verifie le scaling fixe et la normalisation du volume.
    """
    fake_redis = FakeRedisClient()
    monkeypatch.setattr("eva_banker.services.hydra.get_redis_client", lambda: fake_redis)
    manager = MultiAccountManager()
    account = _make_account(scaling_factor="2.0")
    manager.accounts[account.id] = account
    engine = HydraCopyEngine(manager, master_source_id="local-master")

    result = asyncio.run(engine.replicate(CopyTradeRequest(event=_make_event(), dry_run=True)))

    assert len(result.jobs) == 1
    assert result.jobs[0].volume == Decimal("0.10")
    assert result.jobs[0].status.value == "dispatched"


def test_hydra_proportional_scaling_uses_master_balance(monkeypatch) -> None:
    """
    Verifie le scaling proportionnel en fonction du capital master/slave.
    """
    fake_redis = FakeRedisClient()
    monkeypatch.setattr("eva_banker.services.hydra.get_redis_client", lambda: fake_redis)
    manager = MultiAccountManager()
    account = _make_account(
        scaling_mode=HydraScalingMode.PROPORTIONAL,
        scaling_factor="0.5",
        current_balance="20000",
    )
    manager.accounts[account.id] = account
    engine = HydraCopyEngine(manager, master_source_id="local-master")

    result = asyncio.run(
        engine.replicate(
            CopyTradeRequest(
                event=_make_event(volume="0.10", master_balance="10000"),
                dry_run=True,
            )
        )
    )

    assert len(result.jobs) == 1
    assert result.jobs[0].volume == Decimal("0.10")


def test_hydra_quarantine_blocks_slave(monkeypatch) -> None:
    """
    Verifie qu'un compte en quarantaine ne recoit aucune replication.
    """
    fake_redis = FakeRedisClient()
    monkeypatch.setattr("eva_banker.services.hydra.get_redis_client", lambda: fake_redis)
    manager = MultiAccountManager()
    account = _make_account(quarantined=True)
    manager.accounts[account.id] = account
    engine = HydraCopyEngine(manager, master_source_id="local-master")

    result = asyncio.run(engine.replicate(CopyTradeRequest(event=_make_event(), dry_run=True)))

    assert result.jobs == []


def test_hydra_deduplicates_same_fill(monkeypatch) -> None:
    """
    Verifie qu'un meme fill ne cree pas deux jobs sur le meme slave.
    """
    fake_redis = FakeRedisClient()
    monkeypatch.setattr("eva_banker.services.hydra.get_redis_client", lambda: fake_redis)
    manager = MultiAccountManager()
    account = _make_account()
    manager.accounts[account.id] = account
    engine = HydraCopyEngine(manager, master_source_id="local-master")
    shared_event_id = uuid4()

    first = asyncio.run(
        engine.replicate(
            CopyTradeRequest(event=_make_event(event_id=shared_event_id), dry_run=True)
        )
    )
    second = asyncio.run(
        engine.replicate(
            CopyTradeRequest(event=_make_event(event_id=shared_event_id), dry_run=True)
        )
    )

    assert len(first.jobs) == 1
    assert len(second.jobs) == 1
    assert first.jobs[0].id == second.jobs[0].id
    assert any("deja replique" in item for item in second.skipped_accounts)
