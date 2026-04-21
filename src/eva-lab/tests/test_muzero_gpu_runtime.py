"""Tests du correctif RTX 3090 pour MuZero sans regression de compatibilite."""

from __future__ import annotations

import builtins
import importlib.util
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.muzero.checkpoint_utils import (
    build_muzero_checkpoint_payload,
    build_muzero_expected_context_from_config,
)
from eva_lab.muzero.config import MuZeroConfigV3


def _load_script_module(module_name: str, relative_path: str):
    """Charge dynamiquement un script Python du depot.

    Args:
        module_name (str): Nom de module a exposer.
        relative_path (str): Chemin relatif du script dans le depot.

    Returns:
        object: Module importe dynamiquement.
    """

    module_path = Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le module {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NIGHTLY_STACK = _load_script_module(
    "train_nightly_stack",
    "src/eva-lab/scripts/train_nightly_stack.py",
)


def _build_params_tree_from_signature(signature: list[dict[str, object]]) -> dict[str, object]:
    """Reconstruit un arbre de poids minimal depuis une signature stable.

    Args:
        signature (list[dict[str, object]]): Signature de feuilles attendue.

    Returns:
        dict[str, object]: Arbre de dictionnaires compatible checkpoint.
    """

    root: dict[str, object] = {}
    for leaf in signature:
        parts = str(leaf.get("path") or "").split("/")
        cursor = root
        for token in parts[:-1]:
            cursor = cursor.setdefault(token, {})
        cursor[parts[-1]] = np.zeros(
            tuple(int(value) for value in list(leaf.get("shape") or [])),
            dtype=np.dtype(str(leaf.get("dtype") or "float32")),
        )
    return root


class MuZeroGpuRuntimeTests(unittest.TestCase):
    """Verifie le retrait de JAX du parent et la reprise GPU du trainer."""

    @staticmethod
    def _require_jax_stack() -> None:
        """Ignore proprement un test si la stack JAX/Haiku est absente."""

        if importlib.util.find_spec("jax") is None or importlib.util.find_spec("haiku") is None:
            raise unittest.SkipTest("Stack JAX/Haiku indisponible sur cet environnement.")

    def _build_config(self, horizon: str = "scalp") -> MuZeroConfigV3:
        """Construit une configuration MuZero minimale et stable pour les tests.

        Args:
            horizon (str): Horizon strategique cible.

        Returns:
            MuZeroConfigV3: Configuration compacte pour les assertions.
        """

        return MuZeroConfigV3(
            horizon=horizon,
            symbols=["XAUUSD"],
            max_symbols=1,
            dataset_source="csv",
        )

    def test_collection_num_simulations_defaults_to_lighter_budget(self) -> None:
        """Utilise un budget de collecte plus leger que l'optimisation par defaut."""

        with patch.dict(os.environ, {"MUZERO_NUM_SIMULATIONS": "100"}, clear=False):
            os.environ.pop("MUZERO_COLLECTION_NUM_SIMULATIONS", None)
            config = self._build_config("scalp")

        self.assertEqual(config.num_simulations, 100)
        self.assertEqual(config.collection_num_simulations, 32)

    def test_collection_num_simulations_can_be_overridden(self) -> None:
        """Respecte un budget de collecte explicite si l'utilisateur le force."""

        with patch.dict(
            os.environ,
            {
                "MUZERO_NUM_SIMULATIONS": "96",
                "MUZERO_COLLECTION_NUM_SIMULATIONS": "24",
            },
            clear=False,
        ):
            config = self._build_config("scalp")

        self.assertEqual(config.num_simulations, 96)
        self.assertEqual(config.collection_num_simulations, 24)

    def test_static_expected_context_matches_runtime_context(self) -> None:
        """Produit la meme empreinte et la meme signature que l'agent JAX runtime."""

        self._require_jax_stack()
        from eva_lab.muzero.jax_agent import JAXMuZeroAgent

        for horizon in ("scalp", "intraday", "swing"):
            with self.subTest(horizon=horizon):
                config = self._build_config(horizon)
                agent = JAXMuZeroAgent(config)
                runtime_context = dict(agent._expected_checkpoint_context)
                static_context = build_muzero_expected_context_from_config(config)

                self.assertEqual(
                    runtime_context.get("config_fingerprint"),
                    static_context.get("config_fingerprint"),
                )
                self.assertEqual(
                    runtime_context.get("param_signature"),
                    static_context.get("param_signature"),
                )

    def test_checkpoint_v2_is_compatible_with_static_expected_context(self) -> None:
        """Charge un checkpoint v2 artificiel sans instancier JAX."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_dir = Path(tmp_dir) / "weights"
            results_dir = Path(tmp_dir) / "results"
            weights_dir.mkdir(parents=True, exist_ok=True)
            config = self._build_config("scalp")
            expected_context = build_muzero_expected_context_from_config(config)
            params = _build_params_tree_from_signature(expected_context["param_signature"])
            checkpoint_path = weights_dir / "muzero_scalp_latest.pkl"
            payload = build_muzero_checkpoint_payload(
                config=config,
                params=params,
                opt_state={"step": 0},
                artifact_kind="latest",
                lineage={"run_id": "test_gpu_fix"},
            )
            checkpoint_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

            promoter = ChampionPromoter(
                weights_dir=str(weights_dir),
                results_dir=str(results_dir),
            )
            compatibility = promoter.inspect_checkpoint_compatibility(
                checkpoint_path,
                horizon="scalp",
            )

            self.assertTrue(compatibility.get("allowed"))
            self.assertEqual(compatibility.get("status"), "compatible")

    def test_build_horizon_status_does_not_import_jax_agent(self) -> None:
        """Construit le statut MuZero live sans importer `jax_agent` cote parent."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_dir = Path(tmp_dir) / "weights"
            results_dir = Path(tmp_dir) / "results"
            weights_dir.mkdir(parents=True, exist_ok=True)
            config = self._build_config("scalp")
            expected_context = build_muzero_expected_context_from_config(config)
            params = _build_params_tree_from_signature(expected_context["param_signature"])
            payload = build_muzero_checkpoint_payload(
                config=config,
                params=params,
                opt_state={"step": 0},
                artifact_kind="latest",
                lineage={"run_id": "test_gpu_fix"},
            )
            latest_path = weights_dir / "muzero_scalp_latest.pkl"
            latest_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            checkpoint_path = weights_dir / "muzero_scalp_ckpt_500.pkl"
            checkpoint_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

            promoter = ChampionPromoter(
                weights_dir=str(weights_dir),
                results_dir=str(results_dir),
            )
            original_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if str(name).startswith("eva_lab.muzero.jax_agent"):
                    raise AssertionError("jax_agent ne doit pas etre importe dans le parent live.")
                return original_import(name, globals, locals, fromlist, level)

            with patch.dict(os.environ, {"MUZERO_LIVE_SELECTION_POLICY": "checkpoint_preview"}, clear=False):
                with patch("builtins.__import__", side_effect=guarded_import):
                    status = promoter.build_horizon_status("scalp")

            self.assertEqual(status.get("selection_policy"), "checkpoint_preview")
            self.assertTrue((status.get("latest_checkpoint") or {}).get("exists"))
            self.assertEqual(
                ((status.get("artifact_compatibility") or {}).get("artifact_fingerprint") or {}).get("sha256"),
                expected_context["config_fingerprint"]["sha256"],
            )

    def test_run_step_routes_muzero_child_to_gpu(self) -> None:
        """Force le parent CPU-only et le child MuZero GPU-only."""

        fake_result = SimpleNamespace(returncode=0)
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.05",
                "TRAINING_CHILD_CUDA_VISIBLE_DEVICES": "0",
                "TRAINING_CHILD_JAX_PLATFORMS": "cuda",
                "TRAINING_CHILD_XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "TRAINING_CHILD_XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
            },
            clear=False,
        ):
            NIGHTLY_STACK._enforce_parent_runtime_env()
            with patch.object(NIGHTLY_STACK, "mark_step_running"), patch.object(
                NIGHTLY_STACK,
                "append_training_log",
            ), patch.object(NIGHTLY_STACK.subprocess, "run", return_value=fake_result) as mock_run:
                NIGHTLY_STACK.run_step(
                    "muzero_scalp",
                    [sys.executable, "scripts/train_global_models.py"],
                    extra_env={"MUZERO_HORIZON": "scalp"},
                )

        child_env = mock_run.call_args.kwargs["env"]
        self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(child_env["JAX_PLATFORMS"], "cuda")
        self.assertEqual(child_env["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertEqual(child_env["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.85")
        self.assertEqual(child_env["MUZERO_HORIZON"], "scalp")


if __name__ == "__main__":
    unittest.main()
