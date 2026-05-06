"""Tests du verrou manuel des champions live."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eva_lab.champion_promoter import ChampionPromoter


class ChampionPromoterManualLockTests(unittest.TestCase):
    """Valide la preservation d'un champion live verrouille."""

    def test_blocked_manifest_preserves_promoted_manual_live_lock(self) -> None:
        """Preserve le champion promu quand un challenger bloque arrive ensuite."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            promoter = ChampionPromoter(
                weights_dir=str(root / "weights"),
                results_dir=str(root / "results"),
            )

            promoted = promoter.persist_challenger_manifest(
                engine="muzero",
                horizon="scalp",
                status="promoted",
                challenger_id="gen_scalp_old",
                challenger_path=str(root / "old.pkl"),
                latest_checkpoint=str(root / "old.pkl"),
                battle_report=None,
                training_metrics={"family": "mixed"},
                promotion_gate={"allowed": True, "reason": "manual_override_forced_live"},
                promotion_result={"champion_paths": [str(root / "weights" / "muzero_champion_scalp.pkl")]},
                promotion_metadata={"manual_live_lock": True},
            )

            preserved = promoter.persist_challenger_manifest(
                engine="muzero",
                horizon="scalp",
                status="blocked",
                challenger_id="gen_scalp_new",
                challenger_path=str(root / "new.pkl"),
                latest_checkpoint=str(root / "new.pkl"),
                battle_report={"challenger": {"id": "gen_scalp_new"}},
                training_metrics={"family": "mixed"},
                promotion_gate={"allowed": False, "reason": "seed_not_viable_for_v66"},
                promotion_result={"status": "blocked"},
                promotion_metadata={"note": "candidate_blocked"},
            )

            self.assertEqual(promoted["status"], "promoted")
            self.assertEqual(preserved["status"], "promoted")
            self.assertTrue(preserved["manual_live_lock"])
            self.assertEqual(preserved["challenger_id"], "gen_scalp_old")
            self.assertEqual(
                preserved["last_rejected_candidate"]["challenger_id"],
                "gen_scalp_new",
            )
            self.assertEqual(
                preserved["last_rejected_candidate"]["promotion_gate"]["reason"],
                "seed_not_viable_for_v66",
            )


if __name__ == "__main__":
    unittest.main()
