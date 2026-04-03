import sys
import importlib.util
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "shared"))

RISK_PATH = ROOT / "src" / "eva-banker" / "eva_banker" / "services" / "risk.py"
RISK_SPEC = importlib.util.spec_from_file_location("risk_module_under_test", RISK_PATH)
assert RISK_SPEC is not None and RISK_SPEC.loader is not None
RISK_MODULE = importlib.util.module_from_spec(RISK_SPEC)
RISK_SPEC.loader.exec_module(RISK_MODULE)
RiskValidator = RISK_MODULE.RiskValidator


def test_anti_tilt_requires_material_losses_and_cumulative_amount():
    validator = RiskValidator(
        anti_tilt_losses=3,
        anti_tilt_hours=12,
        anti_tilt_min_loss_amount=Decimal("5"),
        anti_tilt_min_cumulative_loss_amount=Decimal("20"),
        anti_tilt_reset_streak_on_new_day=True,
    )

    validator.record_trade_result(Decimal("-4.14"))
    assert validator._consecutive_losses == 0
    assert validator._consecutive_loss_amount == Decimal("0")
    assert validator._is_anti_tilt_active() is False

    validator.record_trade_result(Decimal("-10.98"))
    validator.record_trade_result(Decimal("-8.66"))
    assert validator._consecutive_losses == 2
    assert validator._consecutive_loss_amount == Decimal("19.64")
    assert validator._is_anti_tilt_active() is False

    validator.record_trade_result(Decimal("-11.82"))
    assert validator._consecutive_losses == 3
    assert validator._consecutive_loss_amount == Decimal("31.46")
    assert validator._is_anti_tilt_active() is True


def test_new_day_resets_loss_streak_without_lifting_active_anti_tilt():
    validator = RiskValidator(
        anti_tilt_losses=3,
        anti_tilt_hours=12,
        anti_tilt_min_loss_amount=Decimal("5"),
        anti_tilt_min_cumulative_loss_amount=Decimal("20"),
        anti_tilt_reset_streak_on_new_day=True,
    )
    validator._consecutive_losses = 2
    validator._consecutive_loss_amount = Decimal("19.64")
    validator._anti_tilt_until = datetime.now() + timedelta(hours=6)
    validator._daily_pnl_date = validator._get_market_now().date() - timedelta(days=1)

    validator._roll_daily_window_if_needed()

    assert validator._consecutive_losses == 0
    assert validator._consecutive_loss_amount == Decimal("0")
    assert validator._is_anti_tilt_active() is True
