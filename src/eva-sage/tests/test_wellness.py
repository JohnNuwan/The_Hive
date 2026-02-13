import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Import target module
from eva_sage.main import WellnessService, WellnessReport, SessionStats

class TestWellnessService:
    @pytest.fixture
    def service(self):
        """Fixture to initialize WellnessService with a fixed start time"""
        with patch('eva_sage.main.datetime') as mock_datetime:
            # Start at 9:00
            start_time = datetime(2024, 1, 1, 9, 0, 0)
            mock_datetime.now.return_value = start_time
            service = WellnessService()
            return service

    @patch('eva_sage.main.datetime')
    def test_circadian_optimal_morning(self, mock_datetime, service):
        # Set time to 11:30 (Optimal productive phase)
        # service started at 9:00. 11:30 is 2.5 hours later.
        current_time = datetime(2024, 1, 1, 11, 30, 0)
        mock_datetime.now.return_value = current_time

        report = service.get_wellness_report()

        assert report.circadian_status == "optimal"
        assert "Phase productive" in report.recommendations[0]
        assert report.session_duration_minutes == 150

    @patch('eva_sage.main.datetime')
    def test_circadian_warning_afternoon(self, mock_datetime, service):
        # Set time to 15:30 (Post-lunch dip)
        current_time = datetime(2024, 1, 1, 15, 30, 0)
        mock_datetime.now.return_value = current_time

        report = service.get_wellness_report()

        assert report.circadian_status == "warning"
        assert "Creux post-prandial" in report.recommendations[0]

    @patch('eva_sage.main.datetime')
    def test_break_recommendation(self, mock_datetime, service):
        # Working for 50 minutes without break
        # Start: 9:00. Now: 9:50.
        current_time = datetime(2024, 1, 1, 9, 50, 0)
        mock_datetime.now.return_value = current_time

        report = service.get_wellness_report()

        assert report.break_recommended is True
        assert any("Pause recommandée" in r for r in report.recommendations)

    @patch('eva_sage.main.datetime')
    def test_hydration_reminder(self, mock_datetime, service):
        # Duration % 30 < 5.
        # Duration = 32 minutes. 32 % 30 = 2 < 5.
        current_time = datetime(2024, 1, 1, 9, 32, 0)
        mock_datetime.now.return_value = current_time

        report = service.get_wellness_report()

        assert report.hydration_reminder is True
        assert any("Hydratation" in r for r in report.recommendations)

    @patch('eva_sage.main.datetime')
    def test_take_break(self, mock_datetime, service):
        current_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_datetime.now.return_value = current_time

        result = service.take_break()

        assert service.breaks_taken == 1
        assert service.last_break == current_time
        assert result["breaks_today"] == 1
        assert result["break_time"] == current_time.isoformat()

    @patch('eva_sage.main.datetime')
    def test_no_break_needed_after_break(self, mock_datetime, service):
        # Take a break at 9:45
        break_time = datetime(2024, 1, 1, 9, 45, 0)
        mock_datetime.now.return_value = break_time
        service.take_break()

        # Check at 10:00 (15 mins later)
        current_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_datetime.now.return_value = current_time

        report = service.get_wellness_report()

        # Duration is 60 mins total, but last break was 15 mins ago (less than 45m/2700s)
        # The logic is: if duration > 45 and (not last_break or (now - last_break) > 2700)
        # Here last_break exists and diff is 900s < 2700s.
        assert report.break_recommended is False

    @patch('eva_sage.main.datetime')
    def test_ergonomic_tips_rotation(self, mock_datetime, service):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 0, 0)

        tips = set()
        # Collect enough tips to ensure rotation
        for _ in range(len(service.ERGONOMIC_TIPS) + 2):
            report = service.get_wellness_report()
            tips.add(report.ergonomic_tip)

        assert len(tips) > 1

    @patch('eva_sage.main.datetime')
    def test_session_stats(self, mock_datetime, service):
        # 2 hours duration (120 mins)
        current_time = datetime(2024, 1, 1, 11, 0, 0)
        mock_datetime.now.return_value = current_time

        # Expected breaks: 120 // 45 = 2.
        # Taken: 0. Score should be low.
        stats = service.get_session_stats()
        assert stats.duration_minutes == 120
        assert stats.productivity_score == 20.0 # 0 breaks / 2 * 80 + 20

        # Take 2 breaks
        service.breaks_taken = 2
        stats = service.get_session_stats()
        assert stats.productivity_score == 100.0 # 2/2 * 80 + 20
