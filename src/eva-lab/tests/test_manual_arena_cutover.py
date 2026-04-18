"""Tests du cutover manuel `9000 -> Arena -> reprise`."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_launcher_module():
    """Charge dynamiquement le lanceur Proxmox pour tester ses helpers purs."""

    module_path = Path(__file__).resolve().parents[3] / "scripts" / "deploy" / "start_training_proxmox.py"
    spec = importlib.util.spec_from_file_location("start_training_proxmox", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le module {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_launcher_module()


class ManualArenaCutoverTests(unittest.TestCase):
    """Verifie le classement et les identifiants du cutover manuel."""

    def test_alias_id_strips_nightly_prefix(self) -> None:
        """Produit l'alias stable attendu pour un checkpoint manuel."""

        alias_id = LAUNCHER._build_manual_checkpoint_alias_id(
            "nightly_20260412_100118",
            horizon="scalp",
            checkpoint_step=9000,
        )

        self.assertEqual(alias_id, "gen_scalp_20260412_100118_ckpt9000_manual")

    def test_screen_selection_prefers_victory_before_other_metrics(self) -> None:
        """Priorise toujours une victoire Arena sur un score brut plus eleve."""

        defeat_candidate = {
            "checkpoint_step": 9000,
            "battle_report": {
                "outcome": "DEFEAT",
                "challenger": {
                    "score": 9.0,
                    "metrics": {
                        "return_pct": 2.0,
                        "profit_factor": 1.5,
                        "max_drawdown_pct": 3.0,
                    },
                },
            },
        }
        victory_candidate = {
            "checkpoint_step": 7500,
            "battle_report": {
                "outcome": "VICTORY",
                "challenger": {
                    "score": 8.0,
                    "metrics": {
                        "return_pct": 1.0,
                        "profit_factor": 1.2,
                        "max_drawdown_pct": 5.0,
                    },
                },
            },
        }

        winner = LAUNCHER._select_best_manual_screen_candidate(
            [defeat_candidate, victory_candidate]
        )

        self.assertEqual(winner["checkpoint_step"], 7500)

    def test_screen_selection_uses_newest_checkpoint_as_last_tie_breaker(self) -> None:
        """Retient le checkpoint le plus recent quand tout le reste est identique."""

        tied_7000 = {
            "checkpoint_step": 7000,
            "battle_report": {
                "outcome": "VICTORY",
                "challenger": {
                    "score": 5.0,
                    "metrics": {
                        "return_pct": 0.5,
                        "profit_factor": 1.1,
                        "max_drawdown_pct": 4.0,
                    },
                },
            },
        }
        tied_9000 = {
            "checkpoint_step": 9000,
            "battle_report": {
                "outcome": "VICTORY",
                "challenger": {
                    "score": 5.0,
                    "metrics": {
                        "return_pct": 0.5,
                        "profit_factor": 1.1,
                        "max_drawdown_pct": 4.0,
                    },
                },
            },
        }

        winner = LAUNCHER._select_best_manual_screen_candidate([tied_7000, tied_9000])

        self.assertEqual(winner["checkpoint_step"], 9000)

    def test_filter_remote_training_process_lines_ignores_probe_commands(self) -> None:
        """Ignore les commandes de verification qui se re-detectent elles-memes."""

        raw_lines = [
            "123 bash -lc pgrep -af 'scripts/train_global_models.py' || true",
            "124 python3 - <<'PY' print('probe scripts/train_global_models.py') PY",
            "125 /opt/conda/bin/python scripts/train_global_models.py",
        ]

        filtered = LAUNCHER._filter_remote_training_process_lines(raw_lines)

        self.assertEqual(filtered, ["125 /opt/conda/bin/python scripts/train_global_models.py"])


if __name__ == "__main__":
    unittest.main()
