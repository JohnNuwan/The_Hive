import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-banker"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_banker.brain import AutoTradingEngine
from shared import TradeAction


def _build_engine() -> AutoTradingEngine:
    engine = AutoTradingEngine.__new__(AutoTradingEngine)
    engine._gnn_consultative_enabled = True
    engine._gnn_consultative_symbols = ["EURUSD", "GBPUSD", "XAUUSD"]
    engine._gnn_consultative_veto_min_confidence = 0.90
    engine._gnn_consultative_require_intraday_alignment = True
    return engine


def test_gnn_consultative_filter_blocks_only_on_strong_aligned_conflict():
    engine = _build_engine()

    action, reason, metadata = engine._apply_gnn_consultative_filter(
        symbol="EURUSD",
        action=TradeAction.BUY,
        horizon="scalp",
        decision_context={
            "gnn_scalp_bias": "BEARISH",
            "gnn_scalp_confidence": 0.95,
            "gnn_intraday_bias": "BEARISH",
            "gnn_intraday_confidence": 0.72,
            "gnn_swing_bias": "RANGING",
        },
    )

    assert action is None
    assert reason == "gnn_consultatif_baissier_confirme"
    assert metadata["applied"] is True


def test_gnn_consultative_filter_keeps_action_without_intraday_confirmation():
    engine = _build_engine()

    action, reason, metadata = engine._apply_gnn_consultative_filter(
        symbol="EURUSD",
        action=TradeAction.SELL,
        horizon="scalp",
        decision_context={
            "gnn_scalp_bias": "BULLISH",
            "gnn_scalp_confidence": 0.96,
            "gnn_intraday_bias": "NEUTRAL",
            "gnn_intraday_confidence": 0.40,
        },
    )

    assert action == TradeAction.SELL
    assert reason is None
    assert metadata["applied"] is False
