import logging
from unittest.mock import MagicMock

from eva_substrate.scheduler import Scheduler


def test_scheduler_logs_mode_switch(caplog):
    """
    Test that the Scheduler logs a message when switching modes.
    """
    # Create the scheduler
    scheduler = Scheduler()

    # Mock the circadian rhythm to return a night mode
    # Assuming the default state is "UNKNOWN" -> "STANDARD" (Day) or "DEEP_LEARNING" (Night)
    # Let's force it to switch to Night mode
    scheduler.rhythm.get_current_mode = MagicMock(
        return_value={
            "mode": "NIGHT (Full Power)",
            "is_night": True,
            "hour": 23,
            "strategy": "DEEP_LEARNING",
        }
    )

    # Also mock the allocator to avoid side effects (though it's simple enough)
    scheduler.allocator.set_profile = MagicMock()

    # Capture logs at INFO level
    with caplog.at_level(logging.INFO):
        scheduler.heartbeat()

    # Verify that the log message appears in the captured logs
    # This assertion will fail initially because the code uses print() instead of logger.info()
    assert "SWITCHING MODE: UNKNOWN -> DEEP_LEARNING" in caplog.text
