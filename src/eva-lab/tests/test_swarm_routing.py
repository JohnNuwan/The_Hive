"""Tests du routage multi-champion dynamique Swarm."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.dreamer_gate import DreamerGate


class SwarmRoutingTests(unittest.TestCase):
    """Valide le routage specifique par symbole et le cache de DreamerGate/ChampionPromoter."""

    def test_cache_key_building_with_symbol(self) -> None:
        """Verifie que les cles de cache incluent le symbole s'il est present."""
        key_symbol = DreamerGate._build_agent_cache_key("muzero", "scalp", "GER40.cash")
        key_global = DreamerGate._build_agent_cache_key("muzero", "scalp", None)

        self.assertEqual(key_symbol, "GER40.CASH:muzero:scalp")
        self.assertEqual(key_global, "muzero:scalp")

    def test_swarm_routing_resolves_correct_expert_checkpoint(self) -> None:
        """Verifie que resolve_live_checkpoint charge l'expert associe et fallback proprement."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            weights_dir = root / "weights"
            results_dir = root / "results"
            weights_dir.mkdir()
            results_dir.mkdir()

            promoter = ChampionPromoter(
                weights_dir=str(weights_dir),
                results_dir=str(results_dir),
            )

            # 1. Creer les faux checkpoints physiques
            expert_dax = weights_dir / "muzero_scalp_ckpt_17500.pkl"
            expert_dax.write_text("dummy weights 17500")

            expert_eurusd = weights_dir / "muzero_scalp_ckpt_20500.pkl"
            expert_eurusd.write_text("dummy weights 20500")

            default_champion = weights_dir / "muzero_champion_scalp.pkl"
            default_champion.write_text("dummy default")

            # 2. Creer le manifeste Swarm
            swarm_manifest = results_dir / "swarm_manifest.json"
            swarm_data = {
                "scalp": {
                    "muzero": {
                        "GER40.cash": "muzero_scalp_ckpt_17500.pkl",
                        "EURUSD": "muzero_scalp_ckpt_20500.pkl",
                        "default": "muzero_champion_scalp.pkl"
                    }
                }
            }
            swarm_manifest.write_text(json.dumps(swarm_data, indent=2))

            # Mock de load_manifest et resolve_promotion_gate pour eviter de dependre de rapports complexes
            promoter.load_manifest = lambda *args, **kwargs: {"status": "promoted"}
            promoter.resolve_promotion_gate = lambda *args, **kwargs: {"allowed": True, "reason": "eligible"}
            promoter.inspect_checkpoint_compatibility = lambda *args, **kwargs: {"allowed": True}

            # 3. Executer les resolutions
            path_dax, meta_dax = promoter.resolve_live_checkpoint("scalp", engine="muzero", symbol="GER40.cash")
            path_eurusd, meta_eur = promoter.resolve_live_checkpoint("scalp", engine="muzero", symbol="EURUSD")
            path_other, meta_other = promoter.resolve_live_checkpoint("scalp", engine="muzero", symbol="BTCUSD")

            # 4. Assertions
            self.assertEqual(path_dax, expert_dax)
            self.assertEqual(path_eurusd, expert_eurusd)
            self.assertEqual(path_other, default_champion)


if __name__ == "__main__":
    unittest.main()
