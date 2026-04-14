"""Tests de normalisation du panier scalp multi-univers."""

from __future__ import annotations

import unittest

from eva_lab.training_utils import (
    get_scalp_multi_universe_symbols,
    normalize_training_symbols,
)


class TrainingSymbolUniverseTests(unittest.TestCase):
    """Verifie les alias et le panier canonique a 7 symboles."""

    def test_scalp_multi_universe_matches_target_basket(self) -> None:
        """Expose exactement le panier cible retenu pour le scalp."""

        self.assertEqual(
            get_scalp_multi_universe_symbols(),
            [
                "XAUUSD",
                "US30.cash",
                "GER40.cash",
                "EURUSD",
                "US100.cash",
                "US500.cash",
                "BTCUSD",
            ],
        )

    def test_us100_aliases_are_normalized_to_canonical_symbol(self) -> None:
        """Ramene les alias broker `US100`, `USTEC` et `NAS100` au meme symbole."""

        normalized = normalize_training_symbols(
            [
                "US100.cash",
                "US100.CASH",
                "USTEC",
                "NAS100",
                "XAUUSD",
            ]
        )

        self.assertEqual(normalized, ["US100.cash", "XAUUSD"])


if __name__ == "__main__":
    unittest.main()
