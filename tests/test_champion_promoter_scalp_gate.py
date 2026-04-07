import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_lab.champion_promoter import ChampionPromoter


def _build_promoter(tmp_path: Path) -> ChampionPromoter:
    """Construit un promoteur isole pour les tests locaux.

    Args:
        tmp_path (Path): Repertoire temporaire pytest.

    Returns:
        ChampionPromoter: Promoteur initialise sur le repertoire temporaire.
    """
    return ChampionPromoter(
        weights_dir=str(tmp_path / "weights"),
        results_dir=str(tmp_path / "results"),
    )


def test_evaluate_scalp_seed_gate_rejects_index_dominated_candidate(tmp_path):
    promoter = _build_promoter(tmp_path)

    verdict = promoter.evaluate_scalp_seed_gate(
        {
            "total_trades": 90,
            "long_entry_share": 0.24,
            "short_entry_share": 0.76,
            "directional_imbalance": 0.52,
            "metrics_by_symbol": {
                "EURUSD": {"net_realized_pct": -0.08},
                "XAUUSD": {"net_realized_pct": 0.03},
                "GBPUSD": {"net_realized_pct": 0.02},
                "USDJPY": {"net_realized_pct": -0.09},
                "US30.cash": {"net_realized_pct": 0.45},
                "GER40.cash": {"net_realized_pct": 0.62},
                "US500.cash": {"net_realized_pct": 0.41},
            },
            "metrics_by_position_mechanics": {
                "close_quality_score": 0.21,
                "hold_drag_score_normalized": 0.32,
                "close_winner_count": 1,
                "close_loser_count": 1,
                "slbe_profitable_exits": 0,
                "tp_like_exit_count": 0,
                "meaningful_exit_count": 2,
            },
        }
    )

    assert verdict["allowed"] is False
    assert verdict["primary_reason"] == "core_symbol_balance"
    assert "core_symbol_balance" in verdict["failed_checks"]
    assert verdict["metrics"]["core_net_realized_pct"] < 0.0
    assert verdict["metrics"]["index_profit_share"] > 0.75


def test_evaluate_scalp_seed_gate_flags_missing_meaningful_exits(tmp_path):
    promoter = _build_promoter(tmp_path)

    verdict = promoter.evaluate_scalp_seed_gate(
        {
            "total_trades": 30,
            "long_entry_share": 0.44,
            "short_entry_share": 0.56,
            "directional_imbalance": 0.12,
            "metrics_by_symbol": {
                "EURUSD": {"net_realized_pct": 0.02},
                "XAUUSD": {"net_realized_pct": 0.03},
                "GBPUSD": {"net_realized_pct": 0.01},
                "USDJPY": {"net_realized_pct": 0.00},
                "US30.cash": {"net_realized_pct": 0.02},
                "GER40.cash": {"net_realized_pct": 0.01},
                "US500.cash": {"net_realized_pct": 0.01},
            },
            "metrics_by_position_mechanics": {
                "close_quality_score": 0.0,
                "hold_drag_score_normalized": 0.32,
                "close_winner_count": 0,
                "close_loser_count": 0,
                "slbe_profitable_exits": 0,
                "tp_like_exit_count": 0,
                "meaningful_exit_count": 0,
            },
        }
    )

    assert verdict["allowed"] is False
    assert verdict["primary_reason"] == "meaningful_exits"
    assert verdict["checks"]["hold_drag_score"] is True
    assert verdict["metrics"]["hold_drag_score_normalized"] == 0.32
