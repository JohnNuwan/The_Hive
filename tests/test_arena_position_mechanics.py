import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_lab.arena import Arena


def test_compute_position_mechanics_metrics_normalizes_hold_drag_and_exits():
    metrics = Arena._compute_position_mechanics_metrics(
        {
            "evaluation_games": 4,
            "hold_drag_score_sum": 1.28,
            "close_winner_count": 1,
            "close_loser_count": 2,
            "slbe_profitable_exits": 1,
            "tp_like_exit_count": 1,
            "hold_under_trend_penalty_count": 37,
            "split_executed": 0,
            "pyramids_opened": 0,
            "slbe_triggered": 0,
            "hold_streak_mean_sum": 0.0,
        }
    )

    assert metrics["hold_drag_score"] == 0.32
    assert metrics["hold_drag_score_normalized"] == 0.32
    assert metrics["hold_under_trend_penalty_count"] == 37
    assert metrics["meaningful_exit_count"] == 5
