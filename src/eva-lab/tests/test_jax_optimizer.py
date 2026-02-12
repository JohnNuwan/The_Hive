import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestJaxOptimizer(unittest.TestCase):
    def setUp(self):
        # Clean up sys.modules to ensure fresh import for each test
        # This is crucial because JAX_AVAILABLE is determined at import time
        if 'eva_lab.jax_optimizer' in sys.modules:
            del sys.modules['eva_lab.jax_optimizer']
        if 'jax' in sys.modules:
            del sys.modules['jax']
        if 'jax.numpy' in sys.modules:
            del sys.modules['jax.numpy']

    def tearDown(self):
        # Clean up sys.modules after test
        if 'eva_lab.jax_optimizer' in sys.modules:
            del sys.modules['eva_lab.jax_optimizer']

    def test_jax_missing(self):
        """Test behavior when JAX is not installed."""
        # Simulate JAX missing by ensuring it's not in sys.modules
        # and if it tries to import, it should fail.
        # However, simply removing from sys.modules might not be enough if it's installed.
        # We need to mock the import to raise ImportError.

        with patch.dict(sys.modules, {'jax': None}):
            from eva_lab import jax_optimizer
            importlib.reload(jax_optimizer)

            self.assertFalse(jax_optimizer.JAX_AVAILABLE, "JAX_AVAILABLE should be False when jax is missing")

            opt = jax_optimizer.JaxOptimizer()
            result = opt.optimize_strategy(None, None)

            self.assertEqual(result, {"status": "ERROR", "reason": "JAX_MISSING"})

    def test_jax_available(self):
        """Test behavior when JAX is available (mocked)."""
        # Create mocks for jax components
        mock_jax = MagicMock()
        mock_jnp = MagicMock()

        # Configure jax.numpy
        mock_jax.numpy = mock_jnp

        # Configure jax functions
        mock_grad = MagicMock()
        mock_jit = MagicMock()
        mock_vmap = MagicMock()

        mock_jax.grad = mock_grad
        mock_jax.jit = mock_jit
        mock_jax.vmap = mock_vmap

        # Mock jax.random
        mock_key = MagicMock()
        mock_jax.random.PRNGKey.return_value = mock_key

        # Mock params (the result of random.normal)
        # It needs to support subtraction: params - learning_rate * grads
        mock_params = MagicMock()
        mock_jax.random.normal.return_value = mock_params
        mock_params.__sub__.return_value = mock_params
        mock_params.tolist.return_value = [0.1, 0.2, 0.3] # Mock output

        # Mock grad function compilation
        # grad_fn = jit(grad(self.loss_fn))
        # jit returns the compiled function
        mock_grad_fn = MagicMock()
        mock_jit.return_value = mock_grad_fn

        # Mock grad function execution
        # grads = grad_fn(params, data_x, data_y)
        mock_grads = MagicMock()
        mock_grad_fn.return_value = mock_grads

        # Mock learning rate multiplication
        # learning_rate * grads -> float * MagicMock
        # Ensure __rmul__ (right multiplication) is supported or mock it
        mock_grads.__rmul__.return_value = mock_grads

        # Mock jax.devices
        mock_jax.devices.return_value = ["MOCK_DEVICE"]

        # Apply mocks to sys.modules
        with patch.dict(sys.modules, {'jax': mock_jax, 'jax.numpy': mock_jnp}):
            # We need to handle 'from jax import grad, jit, vmap'
            # Since mock_jax is a MagicMock, accessing attributes works.
            # But the 'from ... import ...' statement looks up the module in sys.modules.
            # Since sys.modules['jax'] is mock_jax, it should work.

            # Import (or reload) the module under test
            from eva_lab import jax_optimizer
            importlib.reload(jax_optimizer)

            self.assertTrue(jax_optimizer.JAX_AVAILABLE, "JAX_AVAILABLE should be True when jax is mocked")

            opt = jax_optimizer.JaxOptimizer()

            # Mock data input
            data_x = MagicMock()
            data_x.shape = (100, 5) # Provide shape for params initialization
            data_y = MagicMock()

            # Run optimization
            # We also need to mock opt.loss_fn execution if called inside optimize_strategy
            # But optimize_strategy calls self.loss_fn only at the end to return final_loss
            # final_loss: float(self.loss_fn(params, data_x, data_y))

            # Mock jnp used in loss_fn: jnp.mean((jnp.dot(x, params) - y)**2)
            # This is tricky because loss_fn uses jax.numpy from the module scope.
            # Since we reloaded the module, jax.numpy in jax_optimizer is our mock_jnp.

            mock_jnp.dot.return_value = MagicMock()
            mock_jnp.mean.return_value = 0.05 # Mock final loss value

            result = opt.optimize_strategy(data_x, data_y, iterations=10)

            # Assertions
            self.assertEqual(result["status"], "OPTIMIZATION_COMPLETE")
            self.assertEqual(result["optimized_params"], [0.1, 0.2, 0.3])
            self.assertEqual(result["final_loss"], 0.05)
            self.assertEqual(result["device"], "MOCK_DEVICE")

            # Verify calls
            mock_jax.random.PRNGKey.assert_called_with(0)
            mock_jax.random.normal.assert_called()
            mock_jit.assert_called()
            # Verify loop execution (called iterations times)
            self.assertEqual(mock_grad_fn.call_count, 10)
