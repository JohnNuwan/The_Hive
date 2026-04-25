"""Tests du digest Telegram des entrainements."""

from __future__ import annotations

import unittest

from eva_lab.training_notifier import build_training_digest_message


class TrainingNotifierDigestTests(unittest.TestCase):
    """Verifie le format du digest Telegram training."""

    def test_build_training_digest_message_exposes_run_metrics_and_champion_status(self) -> None:
        """Assemble un digest utile avec run actif, metriques et champion."""
        run_status = {
            "run_id": "nightly_20260424_042458",
            "status": "running",
            "strategy": "research",
            "reason": "no_deployable_champion",
            "resume_source": "explicit_resume",
            "replay_cache_entries": 128,
            "current_step": {
                "name": "muzero_scalp",
                "phase": "optimisation",
                "horizon": "scalp",
                "symbol": "US30.cash",
                "symbol_index": 2,
                "symbol_total": 7,
                "part_index": 11,
                "part_total": 28,
                "episode_step_current": 250,
                "episode_step_total": 300,
                "updated_at": "2026-04-24T06:45:00",
            },
            "latest_metrics": {
                "loss_pol": 5.32,
                "loss_total": 16.44,
                "policy_entropy": 0.44,
                "policy_top1_share": 0.86,
                "root_mask_rate": 0.11,
                "post_veto_to_hold_rate": 0.01,
                "soft_entry_penalty_rate": 0.18,
                "soft_entry_bonus_rate": 0.32,
                "soft_penalty_to_bonus_ratio": 0.56,
                "soft_penalty_net": -0.14,
                "balanced_episode_rate": 0.91,
                "long_entry_share": 0.48,
                "short_entry_share": 0.52,
            },
        }
        horizon_statuses = {
            "scalp": {
                "selection": "blocked_champion",
                "live_champion_id": None,
                "candidate_id": "gen_scalp_ckpt10000",
                "gate_reason": "profit_factor_insuffisant",
                "directional_metrics": {"directional_bias": "balanced"},
                "candidate_metrics": {
                    "profit_factor": 1.42,
                    "return_pct": 0.15,
                    "win_rate": 33.2,
                },
            }
        }

        message = build_training_digest_message(
            run_status=run_status,
            horizon_statuses=horizon_statuses,
            horizons=["scalp"],
        )

        self.assertIn("POINT ENTRAINEMENT", message)
        self.assertIn("- id: nightly-20260424-042458", message)
        self.assertIn("- phase: optimisation | horizon: scalp", message)
        self.assertIn("- policy: loss_pol=5.32", message)
        self.assertIn("root_mask=11.00%", message)
        self.assertIn("- shaping: ratio=0.56 | net=-0.14", message)
        self.assertIn("CHAMPIONS MUZERO", message)
        self.assertIn("SCALP", message)
        self.assertIn("- selection: blocked champion", message)
        self.assertIn("- Bias=balanced", message)


if __name__ == "__main__":
    unittest.main()
