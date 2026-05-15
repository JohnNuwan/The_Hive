"""
Tests du pipeline d'ingestion MT5 multi-comptes.
"""

from __future__ import annotations

from pathlib import Path

from eva_lab.mt5_history_pipeline import (
    BankerHistorySource,
    IngestionConfig,
    build_nemesis_records,
    build_shadow_transitions,
    canonicalize_symbol,
    group_deals_into_positions,
    ingest_mt5_fleet_history,
    link_copy_groups,
)


def _sample_deals() -> list[dict]:
    """Construit un historique MT5 minimal avec split gagnant puis runner perdant."""

    return [
        {
            "ticket": 101,
            "order": 101,
            "position_id": 9001,
            "symbol": "DE40.e",
            "type": "BUY",
            "entry": 0,
            "volume": 0.48,
            "price": 24933.40,
            "profit": 0.0,
            "swap": 0.0,
            "commission": 0.0,
            "time": "2026-05-06T19:53:28",
            "comment": "COPY MZ-SCP-CH-v",
            "magic": 12345,
        },
        {
            "ticket": 102,
            "order": 102,
            "position_id": 9001,
            "symbol": "DE40.e",
            "type": "SELL",
            "entry": 1,
            "volume": 0.34,
            "price": 24945.00,
            "profit": 31.0,
            "swap": 0.0,
            "commission": 0.0,
            "time": "2026-05-06T20:10:00",
            "comment": "EVA Close",
            "magic": 12345,
        },
        {
            "ticket": 103,
            "order": 103,
            "position_id": 9001,
            "symbol": "DE40.e",
            "type": "SELL",
            "entry": 1,
            "volume": 0.14,
            "price": 24910.00,
            "profit": -45.0,
            "swap": -0.5,
            "commission": -0.5,
            "time": "2026-05-06T21:00:00",
            "comment": "SL",
            "magic": 12345,
        },
    ]


def test_canonicalize_symbol_uses_ftuk_translation() -> None:
    """Verifie que les symboles FTUK reviennent vers les symboles master."""

    assert canonicalize_symbol("DE40.e", {"GER40.cash": "DE40.e"}) == "GER40.cash"
    assert canonicalize_symbol("USTEC.m", {"US100.cash": "USTEC.m"}) == "US100.cash"
    assert canonicalize_symbol("US500.e", {"US500.cash": "US500.e"}) == "US500.cash"


def test_group_deals_detects_eva_close_runner_loss() -> None:
    """Verifie la detection split/HOLD depuis l'historique MT5."""

    source = BankerHistorySource(
        name="FTUK 100K",
        base_url="http://127.0.0.1:8120",
        login=333382300,
        server="FTUKMarkets-Live",
        broker="FTUK",
        symbol_map={"GER40.cash": "DE40.e"},
    )

    positions = group_deals_into_positions(_sample_deals(), source)

    assert len(positions) == 1
    position = positions[0]
    assert position["canonical_symbol"] == "GER40.cash"
    assert position["eva_close_count"] == 1
    assert position["had_profitable_partial"] is True
    assert position["runner_final_negative"] is True
    assert position["net_pnl"] == -15.0


def test_shadow_and_nemesis_outputs_are_actionable() -> None:
    """Verifie que les sorties Shadow et Nemesis capturent le mauvais runner."""

    source = BankerHistorySource(
        name="FTUK 100K",
        base_url="http://127.0.0.1:8120",
        role="follower",
        login=333382300,
        server="FTUKMarkets-Live",
        broker="FTUK",
        symbol_map={"GER40.cash": "DE40.e"},
    )
    positions = link_copy_groups(group_deals_into_positions(_sample_deals(), source))

    transitions = build_shadow_transitions(positions)
    nemesis = build_nemesis_records(positions)

    assert [item["action"]["type"] for item in transitions] == ["BUY", "SPLIT", "CLOSE"]
    assert transitions[-1]["metadata"]["source"] == "mt5_fleet_history"
    assert nemesis
    assert "bad_runner" in nemesis[0]["tags"]
    assert nemesis[0]["symbol"] == "GER40.cash"


def test_ingest_pipeline_writes_state_and_artifacts(tmp_path: Path) -> None:
    """Verifie le batch complet sans serveur Banker reel."""

    source = BankerHistorySource(
        name="FTUK 100K",
        base_url="http://banker.local",
        login=333382300,
        server="FTUKMarkets-Live",
        broker="FTUK",
        symbol_map={"GER40.cash": "DE40.e"},
    )

    def fake_fetcher(url: str, params: dict | None = None) -> dict | list:
        """Retourne des payloads Banker factices."""

        if url.endswith("/history/deals"):
            return {
                "status": "ok",
                "account": {
                    "login": 333382300,
                    "server": "FTUKMarkets-Live",
                    "broker": "FTUK",
                    "instance_name": "FTUK 100K",
                },
                "deals": _sample_deals(),
            }
        if url.endswith("/positions"):
            return []
        raise AssertionError(f"URL inattendue: {url}")

    config = IngestionConfig(
        master_url="http://master.local",
        days=7,
        output_root=tmp_path / "ingestion",
        shadow_output_dir=tmp_path / "shadow",
        state_file=tmp_path / "ingestion" / "state.json",
        force=False,
    )

    report = ingest_mt5_fleet_history(config, fetcher=fake_fetcher, sources=[source])

    assert report["positions_imported"] == 1
    assert report["shadow_transitions"] == 3
    assert report["nemesis_records"] == 1
    assert Path(report["artefacts"]["positions"]).exists()
    assert Path(report["artefacts"]["shadow"]).exists()
    assert config.state_file.exists()
