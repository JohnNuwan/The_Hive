"""Tests du dashboard Rich multi-Bankers."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

from eva_banker.dashboard import (
    BankerInstanceConfig,
    BankerSnapshot,
    build_change_events,
    discover_banker_instances,
    extract_recent_decision_lines,
    parse_env_file,
    read_log_tail,
    resolve_instance_log_file,
)


def test_parse_env_file_lit_les_variables_utiles(tmp_path: Path) -> None:
    """Verifie que le parser `.env` restitue les variables attendues."""

    env_path = tmp_path / ".env.banker.demo.local"
    env_path.write_text(
        "BANKER_INSTANCE_NAME=Demo\nBANKER_API_PORT=8110\nBANKER_BIND_HOST=0.0.0.0\n",
        encoding="utf-8",
    )

    payload = parse_env_file(env_path)

    assert payload["BANKER_INSTANCE_NAME"] == "Demo"
    assert payload["BANKER_API_PORT"] == "8110"
    assert payload["BANKER_BIND_HOST"] == "0.0.0.0"


def test_discover_banker_instances_normalise_les_urls(tmp_path: Path) -> None:
    """Verifie que la decouverte convertit bien `0.0.0.0` en `127.0.0.1`."""

    (tmp_path / ".env.banker.master.local").write_text(
        "BANKER_INSTANCE_NAME=FTMO Master 10K\nBANKER_API_PORT=8100\nBANKER_BIND_HOST=0.0.0.0\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.banker.ftmo50k.local").write_text(
        "BANKER_INSTANCE_NAME=FTMO Challenge 50K\nBANKER_API_PORT=8110\nBANKER_BIND_HOST=127.0.0.1\n",
        encoding="utf-8",
    )

    instances = discover_banker_instances(tmp_path)

    assert [instance.name for instance in instances] == ["FTMO Master 10K", "FTMO Challenge 50K"]
    assert instances[0].base_url == "http://127.0.0.1:8100"
    assert instances[1].base_url == "http://127.0.0.1:8110"
    assert instances[0].log_file == tmp_path / "logs" / "ftmo_master_10k.log"
    assert instances[1].log_file == tmp_path / "logs" / "ftmo_challenge_50k.log"


def test_resolve_instance_log_file_utilise_le_chemin_explicite(tmp_path: Path) -> None:
    """Verifie que le log suit `BANKER_LOG_FILE` quand il est configure."""

    env_path = tmp_path / ".env.banker.master.local"
    log_path = resolve_instance_log_file(
        root_dir=tmp_path,
        env_path=env_path,
        env_data={"BANKER_LOG_FILE": "logs/custom_master.log"},
        name="Master",
    )

    assert log_path == tmp_path / "logs" / "custom_master.log"


def test_read_log_tail_retourne_les_derniere_lignes(tmp_path: Path) -> None:
    """Verifie que le tail des logs conserve les lignes les plus recentes."""

    log_path = tmp_path / "logs" / "banker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("ligne1\nligne2\nligne3\nligne4\n", encoding="utf-8")

    tail = read_log_tail(log_path, max_lines=2)

    assert tail == ["ligne3", "ligne4"]


def test_build_change_events_detecte_les_ecarts_significatifs() -> None:
    """Verifie que le journal remonte les changements utiles d'etat."""

    instance = BankerInstanceConfig(
        name="Master",
        env_file=Path(".env.banker.master.local"),
        base_url="http://127.0.0.1:8100",
        log_file=Path("logs/master.log"),
    )
    previous = BankerSnapshot(
        instance=instance,
        health={"status": "ok", "mt5_connected": True},
        trading_status={
            "runtime": {"runtime_mode": "demo_live", "runtime_profile": "day_live_full_stack"},
            "account": {"equity": 10000.0},
            "risk": {"open_positions": 0},
            "execution_mechanics": {"active_live_engine": None, "live_champion_id": None},
        },
        copy_status={"enabled": True, "targets": [{"name": "Follower"}]},
        accounts=[],
        recent_log_lines=[],
        fetched_at=datetime(2026, 5, 4, 1, 0, 0),
    )
    current = BankerSnapshot(
        instance=instance,
        health={"status": "ok", "mt5_connected": True},
        trading_status={
            "runtime": {"runtime_mode": "demo_live", "runtime_profile": "day_live_full_stack"},
            "account": {"equity": 10012.5},
            "risk": {"open_positions": 1},
            "execution_mechanics": {
                "active_live_engine": "muzero",
                "live_champion_id": "champion_x",
            },
        },
        copy_status={"enabled": True, "targets": [{"name": "Follower"}]},
        accounts=[],
        recent_log_lines=[],
        fetched_at=datetime(2026, 5, 4, 1, 0, 2),
    )

    events = build_change_events(previous, current)

    assert any("positions ouvertes" in event for event in events)
    assert any("moteur live" in event for event in events)
    assert any("champion live" in event for event in events)
    assert any("equity" in event for event in events)


def test_extract_recent_decision_lines_affiche_le_flux_champion() -> None:
    """Verifie que le dashboard expose les decisions recentes du champion."""

    instance = BankerInstanceConfig(
        name="Master",
        env_file=Path(".env.banker.master.local"),
        base_url="http://127.0.0.1:8100",
        log_file=Path("logs/master.log"),
    )
    snapshot = BankerSnapshot(
        instance=instance,
        health={"status": "ok", "mt5_connected": True},
        trading_status={
            "active_live_engine": "muzero",
            "decision_audit": {
                "recent": [
                    {
                        "timestamp": "2026-05-04T01:23:08.449595",
                        "symbol": "XAUUSD",
                        "post_veto_action": "BUY",
                        "engine_name": "muzero",
                        "selection": "champion",
                        "model_version": "gen_scalp_20260428_045305_ckpt10500_manual",
                    }
                ]
            },
        },
        copy_status={"enabled": True, "targets": []},
        accounts=[],
        recent_log_lines=[],
        fetched_at=datetime(2026, 5, 4, 1, 23, 9),
    )

    lines = extract_recent_decision_lines(snapshot)

    assert len(lines) == 1
    assert "XAUUSD -> BUY" in lines[0]
    assert "selection=champion" in lines[0]
    assert "gen_scalp_20260428_045305_ckpt10500_manual" in lines[0]
