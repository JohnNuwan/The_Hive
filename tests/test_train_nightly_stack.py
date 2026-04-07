import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_lab.champion_promoter import ChampionPromoter

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


def test_plan_continuous_scheduler_prefers_positive_scalp_candidate_seed(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "gen_scalp_20260406_053030.pkl"
    checkpoint_path.write_text("seed", encoding="utf-8")
    monkeypatch.setenv("TRAINING_CONTINUOUS_MODE", "1")

    monkeypatch.setattr(
        NIGHTLY,
        "_iter_recent_terminal_summaries",
        lambda **_: [
            {
                "run_id": "nightly_20260406_053030",
                "latest_candidate": "gen_scalp_20260406_053030",
                "arena_outcome": "VICTORY",
                "metrics": {
                    "profit_factor": 1.8044,
                    "return_pct": 0.0538,
                    "net_realized_pct": 1.5879,
                    "long_entry_share": 0.1322,
                    "short_entry_share": 0.8678,
                    "directional_imbalance": 0.7355,
                    "directional_bias": "sell_heavy",
                    "metrics_by_symbol": {
                        "EURUSD": {"net_realized_pct": -0.0244},
                        "XAUUSD": {"net_realized_pct": 0.1065},
                        "GBPUSD": {"net_realized_pct": 0.0511},
                        "USDJPY": {"net_realized_pct": -0.2180},
                        "US30.cash": {"net_realized_pct": 0.4727},
                        "GER40.cash": {"net_realized_pct": 0.9530},
                        "US500.cash": {"net_realized_pct": 0.4966},
                    },
                },
                "metrics_by_position_mechanics": {
                    "close_quality_score": 0.0,
                    "hold_drag_score": 0.32,
                    "hold_drag_score_normalized": 0.32,
                    "meaningful_exit_count": 0,
                    "close_winner_count": 0,
                    "close_loser_count": 0,
                    "slbe_profitable_exits": 0,
                    "tp_like_exit_count": 0,
                },
                "promotion_gate": {"allowed": False, "reason": "win_rate"},
                "latest_verdict": {"status": "blocked", "reason": "win_rate"},
                "artifact_state": {"battle_report_present": True},
            }
        ],
    )
    monkeypatch.setattr(
        NIGHTLY,
        "_extract_live_scalp_reference",
        lambda promoter: {
            "live_champion_id": "gen_scalp_20260308_203907",
            "live_checkpoint": str(tmp_path / "live.pkl"),
            "metrics": {
                "profit_factor": 1.36,
                "return_pct": 0.0321,
                "net_realized_pct": 2.43,
                "metrics_by_position_mechanics": {
                    "close_quality_score": 0.42,
                    "hold_drag_score": 0.31,
                },
            },
            "mechanics": {"close_quality_score": 0.42, "hold_drag_score": 0.31},
        },
    )
    promoter = ChampionPromoter(weights_dir=str(tmp_path), results_dir=str(tmp_path / "results"))

    decision = NIGHTLY._plan_continuous_scheduler(
        scheduler_state={},
        promoter=promoter,
        run_gnn_requested=True,
        run_muzero_requested=True,
        run_dreamer_requested=True,
        gnn_refresh_policy={"scheduled": False, "reason": "already_fresh", "threshold_hours": 72},
    )

    assert decision["mode"] == NIGHTLY.CONTINUOUS_SCHEDULER_MODE
    assert decision["best_for_mutation_candidate"]["candidate_id"] == "gen_scalp_20260406_053030"
    assert decision["best_for_seed_candidate"] == {}
    assert decision["seed_candidate"]["candidate_id"] == "gen_scalp_20260308_203907"
    assert decision["seed_candidate"]["checkpoint_path"] == str(tmp_path / "live.pkl")
    assert decision["seed_reuse_block_reason"] == "core_symbol_balance"
    assert decision["current_focus"] == "scalp_only"
    assert decision["next_horizons"] == ["scalp"]


def test_merge_learning_context_with_scheduler_injects_seed_and_mutations():
    merged = NIGHTLY._merge_learning_context_with_scheduler(
        {
            "loaded": False,
            "winner_symbols": [],
            "risk_symbols": [],
            "priority_symbols": ["EURUSD", "XAUUSD"],
            "env_overrides": {"TRAINING_REVIEW_AVAILABLE": "0"},
        },
        seed_candidate={
            "candidate_id": "gen_scalp_20260406_053030",
            "checkpoint_path": "/tmp/gen_scalp_20260406_053030.pkl",
            "source": "scalp_victory_candidate",
            "reason": "dernier_scalp_victorieux_positif",
        },
        mutation_targets={
            "env_overrides": {"MUZERO_DIRECTIONAL_MAX_IMBALANCE": "0.52"},
            "targets": {"directional_balance": {"reason": "sell_heavy"}},
        },
        seed_reuse_block_reason="core_symbol_balance",
    )

    assert merged["seed_candidate_id"] == "gen_scalp_20260406_053030"
    assert merged["seed_checkpoint"] == "/tmp/gen_scalp_20260406_053030.pkl"
    assert merged["env_overrides"]["TRAINING_SEED_CANDIDATE_ID"] == "gen_scalp_20260406_053030"
    assert merged["env_overrides"]["MUZERO_DIRECTIONAL_MAX_IMBALANCE"] == "0.52"
    assert merged["env_overrides"]["TRAINING_EPISODE_WEIGHT_SEED_CANDIDATE_BONUS"] == "0.20"
    assert merged["priority_symbols"][:4] == ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY"]
    assert "/tmp/gen_scalp_20260406_053030.pkl" in merged["env_overrides"]["TRAINING_SEED_CHECKPOINTS"]


def test_build_training_weighting_summary_uses_cycle_seed_bonus_override(tmp_path, monkeypatch):
    payload = {
        "timestamp": "2026-04-01T21:00:00",
        "observation": {"price": 1.10, "indicators": {}},
        "action": {"type": "SELL", "symbol": "EURUSD"},
        "reward": 3.0,
        "next_observation": {"price": 1.11, "indicators": {}},
        "metadata": {"episode_id": "seed-1", "symbol": "EURUSD", "model_version": "seed-model"},
        "done": True,
    }
    shadow_file = tmp_path / "shadow.jsonl"
    shadow_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(NIGHTLY, "SHADOW_DIR", tmp_path)

    summary = NIGHTLY._build_training_weighting_summary(
        {
            "priority_symbols": ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY"],
            "seed_model_versions": ["seed-model"],
            "env_overrides": {"TRAINING_EPISODE_WEIGHT_SEED_CANDIDATE_BONUS": "0.20"},
        }
    )

    assert summary["weighting_profile"]["seed_candidate_bonus"] == 0.2


def test_build_scalp_mutation_targets_hardens_sell_heavy_bad_exit_seed():
    targets = NIGHTLY._build_scalp_mutation_targets(
        {
            "candidate_id": "gen_scalp_20260406_172613",
            "metrics": {
                "directional_bias": "sell_heavy",
                "directional_imbalance": 0.8727,
                "long_entry_share": 0.0636,
                "short_entry_share": 0.9364,
                "close_quality_score": 0.0,
                "hold_drag_score": 100.4,
                "split_efficiency": 0.0,
                "pyramid_efficiency": 0.0,
            },
            "mechanics": {
                "close_quality_score": 0.0,
                "hold_drag_score": 100.4,
                "split_efficiency": 0.0,
                "pyramid_efficiency": 0.0,
            },
        }
    )

    env = targets["env_overrides"]
    assert env["MUZERO_ACTIVITY_MIN_ENTRIES"] == "4"
    assert env["MUZERO_DIRECTIONAL_MIN_ENTRY_SHARE"] == "0.28"
    assert env["MUZERO_DIRECTIONAL_MAX_IMBALANCE"] == "0.46"
    assert env["MUZERO_CLOSE_WINNER_THRESHOLD"] == "0.0032"
    assert env["MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER"] == "2.85"
    assert env["MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS"] == "6"
    assert env["MUZERO_HOLD_RANGE_PENALTY"] == "0.32"
    assert env["MUZERO_SPLIT_MAX_SPLITS"] == "0"
    assert env["MUZERO_PYRAMID_MAX_ADDITIONS"] == "0"
    assert targets["targets"]["directional_balance"]["reason"] == "sell_heavy"
    assert targets["targets"]["directional_balance"]["long_entry_share"] == 0.0636


def test_update_scheduler_after_cycle_failure_degrades_horizon_on_second_same_phase():
    scheduler_state = {
        "mode": NIGHTLY.CONTINUOUS_SCHEDULER_MODE,
        "cycle_index": 1,
        "horizon_failures": {
            "intraday": {
                "failed_phase": "collecte_parallel",
                "repeat_count": 1,
            }
        },
    }
    scheduler_plan = {
        "cycle_id": "cycle_20260406_000001_0002",
        "cycle_index": 2,
        "current_focus": "full_muzero",
        "seed_candidate": {},
        "best_scalp_candidate": {},
        "improvement_vs_live": {},
        "mutation_targets": {"targets": {}},
        "scheduler_decision": {},
    }

    updated = NIGHTLY._update_scheduler_after_cycle_failure(
        scheduler_state=scheduler_state,
        scheduler_plan=scheduler_plan,
        failed_job={"engine": "muzero", "horizon": "intraday", "name": "muzero_intraday"},
        failed_phase="collecte_parallel",
        exception_message="collecte intraday en erreur",
    )

    assert updated["horizon_failures"]["intraday"]["repeat_count"] == 2
    assert updated["degraded_horizons"]["intraday"]["failed_phase"] == "collecte_parallel"
    assert updated["degraded_horizons"]["intraday"]["degraded_until_cycle"] == 3
