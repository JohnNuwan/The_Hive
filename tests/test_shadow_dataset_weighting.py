import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_lab.shadow_dataset import load_shadow_games


def test_load_shadow_games_applies_weighting_and_filters_symbols(tmp_path):
    payloads = [
        {
            "timestamp": "2026-04-01T20:00:00",
            "observation": {"price": 1.10, "indicators": {}},
            "action": {"type": "BUY", "symbol": "EURUSD"},
            "reward": 6.0,
            "next_observation": {"price": 1.11, "indicators": {}},
            "metadata": {"episode_id": "winner-1", "symbol": "EURUSD"},
            "done": True,
        },
        {
            "timestamp": "2026-04-01T20:05:00",
            "observation": {"price": 1935.0, "indicators": {}},
            "action": {"type": "SELL", "symbol": "XAUUSD"},
            "reward": -9.5,
            "next_observation": {"price": 1937.0, "indicators": {}},
            "metadata": {
                "episode_id": "loser-1",
                "symbol": "XAUUSD",
                "nemesis_type_hint": "LIQUIDITY_TRAP",
            },
            "done": True,
        },
        {
            "timestamp": "2026-04-01T20:10:00",
            "observation": {"price": 95000.0, "indicators": {}},
            "action": {"type": "BUY", "symbol": "BTCUSD"},
            "reward": 12.0,
            "next_observation": {"price": 95100.0, "indicators": {}},
            "metadata": {"episode_id": "ignored-1", "symbol": "BTCUSD"},
            "done": True,
        },
    ]
    shadow_file = tmp_path / "episodes.jsonl"
    shadow_file.write_text("\n".join(json.dumps(item) for item in payloads) + "\n", encoding="utf-8")

    games, summary = load_shadow_games(
        [tmp_path],
        observation_size=32,
        action_space_size=5,
        winner_symbols=["EURUSD"],
        risk_symbols=["XAUUSD"],
        allowed_symbols=["EURUSD", "XAUUSD"],
        include_weighting_summary=True,
    )

    assert len(games) == 2
    games_by_symbol = {game.metadata["symbol"]: game for game in games}
    assert set(games_by_symbol) == {"EURUSD", "XAUUSD"}
    assert games_by_symbol["EURUSD"].metadata["episode_weight"] > 1.0
    assert (
        games_by_symbol["XAUUSD"].metadata["episode_weight"]
        > games_by_symbol["EURUSD"].metadata["episode_weight"]
    )
    assert summary["episodes_loaded"] == 2
    assert summary["weighted_episode_counts"]["winner_episode"] == 1
    assert summary["weighted_episode_counts"]["loser_episode"] == 1
    assert summary["weighted_episode_counts"]["nemesis_episode"] == 1
    assert summary["weighted_episode_counts"]["risk_symbol_episode"] == 1
