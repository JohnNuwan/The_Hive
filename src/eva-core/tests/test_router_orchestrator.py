import unittest
from unittest.mock import patch, MagicMock
import logging
from eva_core.router.orchestrator import Orchestrator

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Orchestrator()

    def test_logger_initialized(self):
        """Verify that logger is initialized in __init__"""
        self.assertTrue(hasattr(self.orchestrator, 'logger'))
        self.assertIsInstance(self.orchestrator.logger, logging.Logger)
        self.assertEqual(self.orchestrator.logger.name, "eva.core.orchestrator")

    def test_delegate_to_math_logs_info(self):
        with patch.object(self.orchestrator.logger, 'info') as mock_info:
            self.orchestrator.delegate_to_math({})
            mock_info.assert_called_with("🧮 DELEGATING TO JULIA (Quant Engine)...")

    def test_delegate_to_evolution_logs_info(self):
        with patch.object(self.orchestrator.logger, 'info') as mock_info:
            self.orchestrator.delegate_to_evolution({})
            mock_info.assert_called_with("🧬 DELEGATING TO JAX (Evolver)...")

if __name__ == '__main__':
    unittest.main()
