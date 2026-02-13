import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies to avoid ImportError
sys.modules["numpy"] = MagicMock()
sys.modules["torch"] = MagicMock()
# We need to mock shared package carefully if it's imported
sys.modules["shared"] = MagicMock()
sys.modules["shared.math_ops"] = MagicMock()

# Mock eva_core.main to prevent app startup and heavy imports from there
sys.modules["eva_core.main"] = MagicMock()

# Now import the module under test
from eva_core.services.docker_monitor import SystemMonitor  # noqa: E402


@pytest.mark.asyncio
async def test_get_system_metrics_temperature_attribute_error(caplog):
    """
    Test that AttributeError in sensors_temperatures is handled gracefully
    and logged as DEBUG (since it likely means not supported).
    """
    caplog.set_level(logging.DEBUG)

    with patch("eva_core.services.docker_monitor.psutil") as mock_psutil:
        # Basic mocks for other calls
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.cpu_freq.return_value = MagicMock(current=2000)
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value = MagicMock(used=1000, total=2000, percent=50.0)
        mock_psutil.disk_usage.return_value = MagicMock(used=1000, total=2000, percent=50.0)
        mock_psutil.disk_io_counters.return_value = MagicMock(read_bytes=0, write_bytes=0)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_recv=0, bytes_sent=0)
        mock_psutil.boot_time.return_value = 0

        # Simulate AttributeError
        mock_psutil.sensors_temperatures.side_effect = AttributeError("Not implemented")

        monitor = SystemMonitor()
        monitor._docker_client = None

        # Mock GPU info to avoid subprocess
        async def mock_gpu():
            return None
        monitor._get_gpu_info = mock_gpu

        metrics = await monitor.get_system_metrics()

        assert metrics["cpu"]["temp"] == 0.0

        # Verify log message exists
        assert "CPU temperature monitoring not supported" in caplog.text


@pytest.mark.asyncio
async def test_get_system_metrics_temperature_generic_exception(caplog):
    """
    Test that generic Exception in sensors_temperatures is logged as WARNING.
    """
    caplog.set_level(logging.WARNING)

    with patch("eva_core.services.docker_monitor.psutil") as mock_psutil:
        # Basic mocks
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.cpu_freq.return_value = MagicMock(current=2000)
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value = MagicMock(used=1000, total=2000, percent=50.0)
        mock_psutil.disk_usage.return_value = MagicMock(used=1000, total=2000, percent=50.0)
        mock_psutil.disk_io_counters.return_value = MagicMock(read_bytes=0, write_bytes=0)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_recv=0, bytes_sent=0)
        mock_psutil.boot_time.return_value = 0

        # Simulate Generic Exception
        mock_psutil.sensors_temperatures.side_effect = Exception("Something went wrong")

        monitor = SystemMonitor()
        monitor._docker_client = None

        async def mock_gpu():
            return None
        monitor._get_gpu_info = mock_gpu

        metrics = await monitor.get_system_metrics()

        assert metrics["cpu"]["temp"] == 0.0

        # Verify log message exists
        assert "Failed to read CPU temperature" in caplog.text


@pytest.mark.asyncio
async def test_get_system_metrics_disk_io_exception(caplog):
    caplog.set_level(logging.WARNING)

    with patch("eva_core.services.docker_monitor.psutil") as mock_psutil:
        # Basic mocks
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.cpu_freq.return_value = MagicMock(current=2000)
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.sensors_temperatures.return_value = {}  # No temp to avoid unrelated logs
        mock_psutil.virtual_memory.return_value = MagicMock(used=1000, total=2000, percent=50.0)
        mock_psutil.disk_usage.return_value = MagicMock(used=1000, total=2000, percent=50.0)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_recv=0, bytes_sent=0)
        mock_psutil.boot_time.return_value = 0

        # Simulate Disk IO Exception
        mock_psutil.disk_io_counters.side_effect = Exception("Disk error")

        monitor = SystemMonitor()
        monitor._docker_client = None

        async def mock_gpu():
            return None
        monitor._get_gpu_info = mock_gpu

        metrics = await monitor.get_system_metrics()

        # Check default speed is 0.0
        assert metrics["disk"]["read_speed"] == 0.0
        assert "Failed to read disk IO metrics" in caplog.text


@pytest.mark.asyncio
async def test_get_gpu_info_filenotfound(caplog):
    caplog.set_level(logging.DEBUG)

    monitor = SystemMonitor()
    monitor._docker_client = None

    # Mock asyncio.create_subprocess_exec to raise FileNotFoundError
    with patch(
        "asyncio.create_subprocess_exec", side_effect=FileNotFoundError("No nvidia-smi")
    ):
        gpu_info = await monitor._get_gpu_info()
        assert gpu_info is None
        assert "nvidia-smi not found" in caplog.text


@pytest.mark.asyncio
async def test_get_gpu_info_generic_exception(caplog):
    caplog.set_level(logging.WARNING)

    monitor = SystemMonitor()
    monitor._docker_client = None

    # Mock asyncio.create_subprocess_exec to raise Exception
    with patch("asyncio.create_subprocess_exec", side_effect=Exception("GPU Error")):
        gpu_info = await monitor._get_gpu_info()
        assert gpu_info is None
        assert "Failed to read GPU metrics" in caplog.text
