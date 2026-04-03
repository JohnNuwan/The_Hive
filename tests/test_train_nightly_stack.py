import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

MODULE_PATH = ROOT / "src" / "eva-lab" / "scripts" / "train_nightly_stack.py"
SPEC = importlib.util.spec_from_file_location("train_nightly_stack", MODULE_PATH)
assert SPEC and SPEC.loader
NIGHTLY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NIGHTLY)


def test_build_review_learning_context_translates_review_to_real_overrides():
    review = {
        "generated_at": "2026-04-01T21:00:00",
        "symbols": [
            {"symbol": "GBPUSD", "net_profit": 22.0, "win_rate": 75.0, "closed_deals": 4},
            {"symbol": "EURUSD", "net_profit": 11.5, "win_rate": 66.0, "closed_deals": 3},
            {"symbol": "XAUUSD", "net_profit": -18.0, "win_rate": 25.0, "closed_deals": 4},
        ],
        "symbol_risk_map": [
            {"symbol": "XAUUSD", "risk_level": "alerte", "net_profit": -18.0, "recent_losses_4h": 2, "recent_events_12h": 1},
            {"symbol": "GER40.cash", "risk_level": "surveillance", "net_profit": -4.0, "recent_losses_4h": 0, "recent_events_12h": 1},
        ],
        "mutation_priors": [
            {"target": "muzero_mechanics"},
            {"target": "muzero_directional_balance"},
            {"target": "gnn_consultatif", "symbols": ["XAUUSD"]},
        ],
        "live_universe": {"symbols": ["EURUSD", "XAUUSD", "GBPUSD"]},
    }

    context = NIGHTLY._build_review_learning_context(review)

    assert context["loaded"] is True
    assert context["winner_symbols"][:2] == ["GBPUSD", "EURUSD"]
    assert context["risk_symbols"][:2] == ["XAUUSD", "GER40.cash"]
    assert context["gnn_focus_symbol"] == "XAUUSD"
    assert context["priority_symbols"][:4] == ["GBPUSD", "EURUSD", "XAUUSD", "GER40.cash"]
    assert context["env_overrides"]["MUZERO_DIRECTIONAL_MAX_IMBALANCE"] == "0.58"
    assert context["env_overrides"]["MUZERO_CLOSE_WINNER_THRESHOLD"] == "0.0048"


def test_build_nightly_job_queue_uses_review_guidance_for_gnn_and_muzero():
    learning_context = {
        "priority_symbols": ["GBPUSD", "EURUSD", "XAUUSD", "USDJPY"],
        "env_overrides": {
            "TRAINING_PRIORITY_SYMBOLS": "GBPUSD,EURUSD,XAUUSD,USDJPY",
            "TRAIN_GNN_FOCUS_SYMBOL": "XAUUSD",
            "MUZERO_DIRECTIONAL_MAX_IMBALANCE": "0.58",
        },
    }

    jobs = NIGHTLY.build_nightly_job_queue(
        run_gnn=True,
        run_muzero=True,
        run_dreamer=False,
        learning_context=learning_context,
    )

    assert jobs[0]["name"] == "gnn"
    assert jobs[0]["focus_symbols"] == ["GBPUSD", "EURUSD", "XAUUSD", "USDJPY"]
    assert jobs[0]["extra_env"]["TRAIN_GNN_FOCUS_SYMBOL"] == "XAUUSD"
    assert jobs[1]["name"] == "muzero_scalp"
    assert jobs[1]["focus_symbols"] == ["GBPUSD", "EURUSD", "XAUUSD", "USDJPY"]
    assert jobs[1]["extra_env"]["MUZERO_DIRECTIONAL_MAX_IMBALANCE"] == "0.58"
    assert jobs[1]["extra_env"]["MUZERO_MODEL_FAMILY"] == ""


def test_build_training_weighting_summary_counts_weighted_shadow_episodes(tmp_path, monkeypatch):
    payloads = [
        {
            "timestamp": "2026-04-01T21:00:00",
            "observation": {"price": 1.10, "indicators": {}},
            "action": {"type": "SELL", "symbol": "GBPUSD"},
            "reward": 8.5,
            "next_observation": {"price": 1.11, "indicators": {}},
            "metadata": {"episode_id": "winner-1", "symbol": "GBPUSD"},
            "done": True,
        },
        {
            "timestamp": "2026-04-01T21:05:00",
            "observation": {"price": 1930.0, "indicators": {}},
            "action": {"type": "BUY", "symbol": "XAUUSD"},
            "reward": -12.0,
            "next_observation": {"price": 1928.0, "indicators": {}},
            "metadata": {
                "episode_id": "loser-1",
                "symbol": "XAUUSD",
                "nemesis_type_hint": "LIQUIDITY_TRAP",
            },
            "done": True,
        },
    ]
    shadow_file = tmp_path / "shadow.jsonl"
    shadow_file.write_text("\n".join(json.dumps(item) for item in payloads) + "\n", encoding="utf-8")
    monkeypatch.setattr(NIGHTLY, "SHADOW_DIR", tmp_path)

    summary = NIGHTLY._build_training_weighting_summary(
        {
            "winner_symbols": ["GBPUSD"],
            "risk_symbols": ["XAUUSD"],
            "priority_symbols": ["GBPUSD", "XAUUSD", "EURUSD"],
            "gnn_focus_symbol": "XAUUSD",
        }
    )

    assert summary["episodes_loaded"] == 2
    assert summary["winner_symbols"] == ["GBPUSD"]
    assert summary["risk_symbols"] == ["XAUUSD"]
    assert summary["weighted_episode_counts"]["winner_episode"] == 1
    assert summary["weighted_episode_counts"]["loser_episode"] == 1
    assert summary["weighted_episode_counts"]["nemesis_episode"] == 1
    assert summary["weighted_episode_counts"]["risk_symbol_episode"] == 1
    assert summary["gnn_focus_symbol"] == "XAUUSD"


def test_evaluate_gnn_refresh_policy_skips_fresh_registry(monkeypatch):
    fresh_trained_at = (datetime.now() - timedelta(hours=6)).isoformat()
    monkeypatch.setattr(
        NIGHTLY,
        "load_market_gnn_registry",
        lambda: {
            "status": "validated",
            "checkpoint_path": "/tmp/gnn_master.pth",
            "trained_at": fresh_trained_at,
            "version": "gnn_master",
        },
    )

    decision = NIGHTLY._evaluate_gnn_refresh_policy(True)

    assert decision["scheduled"] is False
    assert decision["reason"] == "already_fresh"
    assert decision["registry_status"] == "validated"
