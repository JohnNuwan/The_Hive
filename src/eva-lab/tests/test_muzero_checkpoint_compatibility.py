"""Tests du contrat de checkpoint MuZero et du seed GA strict."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eva_lab.muzero.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    build_muzero_expected_context,
    inspect_muzero_checkpoint,
    save_muzero_checkpoint,
)
from scripts.deploy.run_seeded_muzero_ga_campaign import _build_seed_reference_from_status


def _build_config(*, support_size: int = 100, horizon: str = "scalp") -> SimpleNamespace:
    """Construit une configuration MuZero minimale pour les tests."""

    return SimpleNamespace(
        observation_shape=(32,),
        action_space_size=5,
        hidden_state_size=256,
        network_hidden_dims=[256, 256],
        support_size=support_size,
        horizon=horizon,
        dataset_descriptor={
            "dataset_id": "dataset_scalp",
            "feature_profile": "profile_scalp",
            "mechanics_profile_version": "v3",
        },
        dataset_id="dataset_scalp",
        feature_profile={"profile_name": "profile_scalp"},
        mechanics_profile_version="v3",
        symbols=["EURUSD", "XAUUSD"],
    )


def _build_params(*, value_head_size: int = 201) -> dict[str, object]:
    """Construit un arbre de poids simple avec une tete de valeur parametree."""

    return {
        "representation_network": {
            "dense_0": {
                "w": np.zeros((32, 256), dtype=np.float32),
                "b": np.zeros((256,), dtype=np.float32),
            }
        },
        "prediction_network": {
            "linear_4": {
                "w": np.zeros((512, value_head_size), dtype=np.float32),
                "b": np.zeros((value_head_size,), dtype=np.float32),
            }
        },
    }


def _write_legacy_checkpoint(path: Path, params: dict[str, object]) -> None:
    """Ecrit un checkpoint legacy minimal pour les tests."""

    with path.open("wb") as file_obj:
        pickle.dump(
            {"params": params, "opt_state": {"step": 0}},
            file_obj,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


class MuZeroCheckpointCompatibilityTests(unittest.TestCase):
    """Couvre le contrat de compatibilite MuZero et le seed GA strict."""

    def test_v2_checkpoint_compatible_is_accepted(self) -> None:
        """Valide qu'un checkpoint MuZero v2 compatible est chargeable."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = _build_config()
            params = _build_params(value_head_size=201)
            checkpoint_path = tmp_path / "muzero_scalp_v2.pkl"

            save_muzero_checkpoint(
                checkpoint_path,
                config=config,
                params=params,
                opt_state={"step": 1},
                artifact_kind="challenger",
                lineage={"parent_champion_id": "champion_scalp_live"},
            )
            _payload, compatibility = inspect_muzero_checkpoint(
                checkpoint_path,
                expected_context=build_muzero_expected_context(
                    config=config,
                    expected_params=params,
                ),
            )

            self.assertTrue(compatibility["allowed"])
            self.assertEqual(compatibility["status"], "compatible")
            self.assertEqual(
                compatibility["schema_version"],
                CHECKPOINT_SCHEMA_VERSION,
            )

    def test_legacy_checkpoint_with_matching_shapes_is_accepted(self) -> None:
        """Autorise un legacy uniquement si les shapes correspondent exactement."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = _build_config()
            params = _build_params(value_head_size=201)
            checkpoint_path = tmp_path / "muzero_scalp_legacy_ok.pkl"
            _write_legacy_checkpoint(checkpoint_path, params)

            _payload, compatibility = inspect_muzero_checkpoint(
                checkpoint_path,
                expected_context=build_muzero_expected_context(
                    config=config,
                    expected_params=params,
                ),
            )

            self.assertTrue(compatibility["allowed"])
            self.assertEqual(compatibility["status"], "legacy_compatible")

    def test_legacy_checkpoint_with_mismatching_shapes_is_rejected(self) -> None:
        """Refuse un legacy quand la tete de valeur ne matche plus l'architecture."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = _build_config()
            expected_params = _build_params(value_head_size=201)
            checkpoint_path = tmp_path / "muzero_scalp_legacy_bad.pkl"
            _write_legacy_checkpoint(checkpoint_path, _build_params(value_head_size=1))

            _payload, compatibility = inspect_muzero_checkpoint(
                checkpoint_path,
                expected_context=build_muzero_expected_context(
                    config=config,
                    expected_params=expected_params,
                ),
            )

            self.assertFalse(compatibility["allowed"])
            self.assertEqual(compatibility["status"], "incompatible")
            self.assertIn("forme differente", str(compatibility["reason"]))

    def test_v2_checkpoint_with_wrong_fingerprint_is_rejected(self) -> None:
        """Refuse un checkpoint v2 si le support ou l'espace d'action diverge."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expected_config = _build_config(support_size=100)
            expected_params = _build_params(value_head_size=201)
            incompatible_config = _build_config(support_size=50)
            incompatible_params = _build_params(value_head_size=101)
            checkpoint_path = tmp_path / "muzero_scalp_v2_bad_fingerprint.pkl"

            save_muzero_checkpoint(
                checkpoint_path,
                config=incompatible_config,
                params=incompatible_params,
                opt_state={"step": 2},
                artifact_kind="latest",
            )
            _payload, compatibility = inspect_muzero_checkpoint(
                checkpoint_path,
                expected_context=build_muzero_expected_context(
                    config=expected_config,
                    expected_params=expected_params,
                ),
            )

            self.assertFalse(compatibility["allowed"])
            self.assertEqual(compatibility["status"], "incompatible")
            self.assertIn(
                "empreinte de configuration differente",
                str(compatibility["reason"]),
            )

    def test_seed_reference_requires_promoted_compatible_champion(self) -> None:
        """Refuse tout seed issu d'un artefact non promu ou non champion."""

        with self.assertRaisesRegex(RuntimeError, "promu"):
            _build_seed_reference_from_status(
                {
                    "promotion_state": "candidate_only",
                    "selection": "champion",
                    "live_champion_id": "gen_scalp_candidate",
                    "champion_checkpoint": {"path": "/tmp/candidate.pkl"},
                    "artifact_compatibility": {"allowed": True},
                }
            )

        with self.assertRaisesRegex(RuntimeError, "non champions"):
            _build_seed_reference_from_status(
                {
                    "promotion_state": "promoted",
                    "selection": "latest",
                    "live_champion_id": "gen_scalp_latest",
                    "champion_checkpoint": {"path": "/tmp/latest.pkl"},
                    "artifact_compatibility": {"allowed": True},
                }
            )

    def test_seed_reference_returns_lineage_for_promoted_compatible_champion(self) -> None:
        """Retourne toutes les metadonnees de filiation pour un seed valide."""

        seed_reference = _build_seed_reference_from_status(
            {
                "promotion_state": "promoted",
                "selection": "champion",
                "live_champion_id": "gen_scalp_live",
                "champion_checkpoint": {"path": "/tmp/champion_scalp.pkl"},
                "promotion_gate": {"metrics": {"profit_factor": 1.8}},
                "mechanics_profile_version": "v3",
                "feature_profile": "profile_scalp",
                "artifact_compatibility": {
                    "allowed": True,
                    "status": "compatible",
                    "reason": "Checkpoint MuZero v2 compatible.",
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                },
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "resume_source": "champion",
                "lineage": {"parent_champion_id": "gen_scalp_parent"},
                "seed_parent_champion_id": "gen_scalp_parent",
            }
        )

        self.assertEqual(seed_reference["champion_id"], "gen_scalp_live")
        self.assertEqual(seed_reference["checkpoint_path"], "/tmp/champion_scalp.pkl")
        self.assertEqual(
            seed_reference["checkpoint_schema_version"],
            CHECKPOINT_SCHEMA_VERSION,
        )
        self.assertEqual(
            seed_reference["lineage"],
            {"parent_champion_id": "gen_scalp_parent"},
        )


if __name__ == "__main__":
    unittest.main()
