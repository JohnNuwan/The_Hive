"""Tests de robustesse des cibles scalaires MuZero."""

from __future__ import annotations

import importlib.util
import unittest


class MuZeroSupportShapeTests(unittest.TestCase):
    """Verifie la normalisation des cibles scalaires avant projection support."""

    @staticmethod
    def _require_jax_stack() -> None:
        """Ignore proprement le test si la stack JAX/Haiku est absente."""

        if importlib.util.find_spec("jax") is None or importlib.util.find_spec("haiku") is None:
            raise unittest.SkipTest("Stack JAX/Haiku indisponible sur cet environnement.")

    def test_accepts_one_dimensional_targets(self) -> None:
        """Projette correctement des cibles deja aplaties en `(batch,)`."""

        self._require_jax_stack()
        import jax.numpy as jnp

        from eva_lab.muzero.jax_networks import scalar_to_support

        support = scalar_to_support(jnp.array([0.0, 1.5, -2.0]), support_size=3)

        self.assertEqual(support.shape, (3, 7))

    def test_accepts_column_targets(self) -> None:
        """Projette correctement les cibles legacy fournies en `(batch, 1)`."""

        self._require_jax_stack()
        import jax.numpy as jnp

        from eva_lab.muzero.jax_networks import scalar_to_support

        support = scalar_to_support(jnp.array([[0.0], [1.5], [-2.0]]), support_size=3)

        self.assertEqual(support.shape, (3, 7))

    def test_rejects_non_scalar_last_dimension(self) -> None:
        """Refuse explicitement les tenseurs ambigus non scalaires par echantillon."""

        self._require_jax_stack()
        import jax.numpy as jnp

        from eva_lab.muzero.jax_networks import scalar_to_support

        with self.assertRaisesRegex(ValueError, "formes?"):
            scalar_to_support(jnp.array([[0.0, 1.0], [1.5, -0.5]]), support_size=3)


if __name__ == "__main__":
    unittest.main()
