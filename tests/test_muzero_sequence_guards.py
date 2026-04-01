"""
Tests cibles des garde-fous de sequence MuZero nocturne.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parents[1]
for extra in ("src/shared", "src/eva-lab"):
    extra_path = LOCAL_ROOT / extra
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))


def _load_module(module_name: str, relative_path: str):
    """
    Charge un module projet hors package Python standard.

    Args:
        module_name (str): Nom logique du module.
        relative_path (str): Chemin relatif depuis la racine du depot.

    Returns:
        Any: Module charge dynamiquement.
    """

    module_path = LOCAL_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Chargement impossible pour {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


v4_sequence_runner = _load_module("v4_sequence_runner_test", "scripts/deploy/v4_sequence_runner.py")
night_hunt = _load_module("muzero_night_hunt_test", "scripts/deploy/run_muzero_night_ga_hunt.py")


def test_summary_guard_rejects_stale_run() -> None:
    """
    Verifie qu'un resume terminal stale n'est pas accepte pour le scoring.
    """

    summary = {
        "run_id": "nightly_old",
        "sequence_id": "seq-old",
        "window_id": "window-old",
        "trial_id": "trial-old",
    }

    assert not v4_sequence_runner._summary_matches_expected(
        summary,
        run_id="nightly_new",
        window_id="window-new",
        sequence_id="seq-new",
        trial_id="trial-new",
    )


def test_summary_guard_accepts_matching_terminal_summary() -> None:
    """
    Verifie qu'un resume terminal coherent reste scorables.
    """

    summary = {
        "run_id": "nightly_ok",
        "sequence_id": "seq-ok",
        "window_id": "window-ok",
        "trial_id": "trial-ok",
    }

    assert v4_sequence_runner._summary_matches_expected(
        summary,
        run_id="nightly_ok",
        window_id="window-ok",
        sequence_id="seq-ok",
        trial_id="trial-ok",
    )


def test_night_hunt_filters_only_current_sequence_results() -> None:
    """
    Verifie que la chasse locale ignore les scores d'une sequence precedente.
    """

    payload = {
        "results": [
            {"sequence_id": "seq-old", "trial_id": "old", "score": 999.0},
            {"sequence_id": "seq-new", "trial_id": "new-low", "score": 1.0},
            {"sequence_id": "seq-new", "trial_id": "new-high", "score": 10.0},
        ]
    }

    results = night_hunt._filter_sequence_results(payload, sequence_id="seq-new")

    assert [item["trial_id"] for item in results] == ["new-high", "new-low"]
